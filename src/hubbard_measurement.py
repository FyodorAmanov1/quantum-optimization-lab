"""Born-sampled four-orbital Hubbard measurements, with bounded-error CIs.

Pauli strings list orbital 0 first; integer outcome bits use orbital 0 as LSB.
No energy-eigenvalue sampling or Gaussian perturbations are used.
"""
import numpy as np


SETTINGS = ("ZZZZ", "XZXI", "YZYI", "IXZX", "IYZY")
PAULI = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}


def pauli_matrix(word):
    matrix = np.ones((1, 1))
    for symbol in reversed(word):
        matrix = np.kron(matrix, PAULI[symbol])
    return matrix


def measurement_probabilities(wave, setting):
    """Rotate a full 16-amplitude state into a product measurement basis."""
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    rotations = {"I": np.eye(2), "Z": np.eye(2), "X": h,
                 "Y": h @ np.diag([1, -1j])}
    rotation = np.ones((1, 1))
    for symbol in reversed(setting):
        rotation = np.kron(rotation, rotations[symbol])
    probabilities = np.abs(rotation @ wave) ** 2
    return probabilities / probabilities.sum()


def measure_hubbard(wave, basis, interaction, shots=4096, seed=2026, confidence=0.95):
    """Measure energy, total double occupancy and kinetic energy at t=1.

    Independent multinomial counts are actual repeated Born measurements.
    One Z shot supplies all six nonconstant density Pauli terms together.
    Per-observable Hoeffding bounds use a union bound across active groups;
    these are marginal CIs for the prepared state's expectations, not for
    ansatz error. The sample SE includes all covariance within each group.
    """
    if isinstance(shots, (bool, np.bool_)) or not isinstance(shots, (int, np.integer)) or shots < 2:
        raise ValueError("shots must be an integer >= 2 per setting")
    if not np.isfinite(interaction) or not 0 < confidence < 1:
        raise ValueError("finite interaction and confidence in (0,1) required")
    wave = np.asarray(wave, dtype=complex)
    basis = np.asarray(basis)
    if (wave.shape != (4,) or basis.shape != (4,)
            or not np.issubdtype(basis.dtype, np.integer)
            or set(basis.tolist()) != {3, 6, 9, 12}
            or not np.all(np.isfinite(wave))
            or not np.isclose(np.vdot(wave, wave).real, 1, rtol=0, atol=1e-12)):
        raise ValueError("normalized wave in the one-up/one-down sector required")
    full = np.zeros(16, complex)
    full[basis] = wave
    rng = np.random.default_rng(seed)
    outcomes = np.arange(16)
    double = ((outcomes & 3) == 3).astype(float) + ((outcomes & 12) == 12)
    values = [double]
    for setting in SETTINGS[1:]:
        parity = np.ones(16)
        for orbital, symbol in enumerate(setting):
            if symbol != "I":
                parity *= 1 - 2 * ((outcomes >> orbital) & 1)
        values.append(-0.5 * parity)
    counts = [rng.multinomial(shots, measurement_probabilities(full, s)) for s in SETTINGS]
    means = np.array([c @ v / shots for c, v in zip(counts, values)])
    # Variance of the mean, calculated from grouped per-shot values.
    variances = np.array([c @ (v - m) ** 2 / (shots * (shots - 1))
                          for c, v, m in zip(counts, values, means)])
    exact_d = float(np.abs(wave) ** 2 @ double[basis])
    kinetic = sum(-0.5 * pauli_matrix(s) for s in SETTINGS[1:])
    exact_k = float(np.vdot(full, kinetic @ full).real)
    exact = {"double_occupancy": exact_d, "kinetic_energy": exact_k,
             "energy": exact_k + interaction * exact_d}
    coefficients = {"double_occupancy": np.array([1, 0, 0, 0, 0]),
                    "kinetic_energy": np.array([0, 1, 1, 1, 1]),
                    "energy": np.array([interaction, 1, 1, 1, 1])}
    observables = {}
    for name, weights in coefficients.items():
        active = np.count_nonzero(weights)
        # Each kinetic group has range 1. D has range 1 in the input sector.
        radius = float(np.abs(weights).sum() * np.sqrt(
            np.log(2 * active / (1 - confidence)) / (2 * shots)))
        estimate = float(weights @ means)
        observables[name] = {"estimate": estimate,
                             "standard_error": float(np.sqrt(weights**2 @ variances)),
                             "ci": [estimate - radius, estimate + radius],
                             "confidence": confidence, "exact_prepared_state": exact[name],
                             "error_to_prepared_state": estimate - exact[name]}
    return {"shots_per_setting": int(shots), "total_shots": int(5 * shots), "seed": int(seed),
            "settings": [{"basis": s, "counts": c.tolist()} for s, c in zip(SETTINGS, counts)],
            "observables": observables}


def finite_shot_study():
    try:
        from .studio import hubbard, vqe_state
    except ImportError:
        from studio import hubbard, vqe_state

    interactions = [0, 0.5, 1, 2, 4, 6, 8]
    budgets = [256, 4096, 65536]
    runs = []
    for index, u in enumerate(interactions):
        h, basis = hubbard(u)
        wave, metrics = vqe_state(u)
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        ground = eigenvectors[:, 0]
        double = np.array([int(s & 3 == 3) + int(s & 12 == 12) for s in basis])
        exact_d = float(ground**2 @ double)
        references = {"energy": float(eigenvalues[0]), "double_occupancy": exact_d,
                      "kinetic_energy": float(eigenvalues[0] - u * exact_d)}
        for b, shots in enumerate(budgets):
            run = measure_hubbard(wave, basis, u, shots, 202600 + 10 * index + b)
            run.update({"interaction": u, "exact_ground_state": references,
                        "vqe_residual": metrics[3]})
            runs.append(run)
    return {"interactions": interactions, "budgets_per_setting": budgets,
            "method": "Independent Born multinomial draws after local Pauli basis rotations; exact-expectation VQE training",
            "interval_method": "Two-sided Hoeffding per active group, union bound within each observable; marginal 95% CIs for prepared-state expectations",
            "cost": {"grouped_settings": 5, "ungrouped_nonidentity_terms": 10,
                     "assumptions": "Equal shots per setting; all six Z density terms share one shot; four hopping settings; identity costs no shots. Counts exclude exact training, state preparation and gate compilation. Setting reduction is not an equal-total-shot accuracy comparison."},
            "runs": runs}
