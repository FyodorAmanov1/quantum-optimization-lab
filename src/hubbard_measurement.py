"""Born-sampled four-orbital Hubbard measurements, with bounded-error CIs.

Pauli strings list orbital 0 first; integer outcome bits use orbital 0 as LSB.
No energy-eigenvalue sampling or Gaussian perturbations are used.
"""
import numpy as np


SETTINGS = ("ZZZZ", "XZXI", "YZYI", "IXZX", "IYZY")
DENSITY_TERMS = ("ZIII", "IZII", "IIZI", "IIIZ", "ZZII", "IIZZ")
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


def measure_hubbard(wave, basis, interaction, shots=4096, seed=2026, confidence=0.95,
                    grouping="grouped"):
    """Measure energy, total double occupancy and kinetic energy at t=1.

    Independent multinomial counts are actual repeated Born measurements.
    One Z shot supplies all six nonconstant density Pauli terms together.
    Per-observable Hoeffding bounds combine all independent weighted shots;
    these are marginal CIs for the prepared state's expectations, not for
    ansatz error. The sample SE includes all covariance within each group.
    grouping='ungrouped' instead measures ten Pauli terms independently,
    with shots per term. Exact sampling SE is a Born-probability diagnostic.
    """
    if isinstance(shots, (bool, np.bool_)) or not isinstance(shots, (int, np.integer)) or shots < 2:
        raise ValueError("shots must be an integer >= 2 per setting")
    if not np.isfinite(interaction) or not 0 < confidence < 1:
        raise ValueError("finite interaction and confidence in (0,1) required")
    if grouping not in ("grouped", "ungrouped"):
        raise ValueError("grouping must be grouped or ungrouped")
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
    settings = SETTINGS if grouping == "grouped" else DENSITY_TERMS + SETTINGS[1:]
    values = [double] if grouping == "grouped" else []
    for setting in settings[1:] if grouping == "grouped" else settings:
        parity = np.ones(16)
        for orbital, symbol in enumerate(setting):
            if symbol != "I":
                parity *= 1 - 2 * ((outcomes >> orbital) & 1)
        values.append(-0.5 * parity if grouping == "grouped" else parity)
    probabilities = [measurement_probabilities(full, s) for s in settings]
    counts = [rng.multinomial(shots, p) for p in probabilities]
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
    constants = dict.fromkeys(coefficients, 0.0)
    ranges = np.ones(5)
    if grouping == "ungrouped":
        d_weights = np.array([-0.25] * 4 + [0.25] * 2 + [0.] * 4)
        k_weights = np.array([0.] * 6 + [-0.5] * 4)
        coefficients = {"double_occupancy": d_weights, "kinetic_energy": k_weights,
                        "energy": interaction * d_weights + k_weights}
        constants = {"double_occupancy": 0.5, "kinetic_energy": 0.,
                     "energy": interaction * 0.5}
        ranges = np.full(10, 2.)
    # Analytic sampling variance is a diagnostic baseline, never used for CIs.
    population_variances = np.array([
        p @ (v - p @ v)**2 / shots for p, v in zip(probabilities, values)])
    observables = {}
    for name, weights in coefficients.items():
        active = np.count_nonzero(weights)
        # D has range 1 in this sector; ungrouped parity has range 2.
        union_radius = float((np.abs(weights) @ ranges) * np.sqrt(
            np.log(2 * active / (1 - confidence)) / (2 * shots)))
        # Apply Hoeffding to independent weighted individual shots, not to
        # the correlated Pauli terms within a shared density bitstring.
        squared_range_over_shots = float(np.sum((weights * ranges)**2) / shots)
        radius = float(np.sqrt(np.log(2 / (1 - confidence))
                               * squared_range_over_shots / 2))
        estimate = float(constants[name] + weights @ means)
        observables[name] = {"estimate": estimate,
                             "standard_error": float(np.sqrt(weights**2 @ variances)),
                             "exact_sampling_standard_error": float(np.sqrt(max(
                                 0., weights**2 @ population_variances))),
                             "ci": [estimate - radius, estimate + radius],
                             "union_bound_ci": [estimate - union_radius, estimate + union_radius],
                             "squared_range_over_shots": squared_range_over_shots,
                             "confidence": confidence, "exact_prepared_state": exact[name],
                             "error_to_prepared_state": estimate - exact[name]}
    return {"shots_per_setting": int(shots), "total_shots": int(len(settings) * shots),
            "seed": int(seed), "grouping": grouping,
            "settings": [{"basis": s, "counts": c.tolist()} for s, c in zip(settings, counts)],
            "observables": observables}


def finite_shot_study():
    try:
        from .studio import hubbard, vqe_state
    except ImportError:
        from studio import hubbard, vqe_state

    interactions = [0, 0.5, 1, 2, 4, 6, 8]
    budgets = [256, 4096, 65536]
    runs = []
    ungrouped_runs = []
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
            baseline = measure_hubbard(wave, basis, u, shots // 2,
                                       302600 + 10 * index + b, grouping="ungrouped")
            baseline.update({"interaction": u, "exact_ground_state": references,
                             "vqe_residual": metrics[3]})
            ungrouped_runs.append(baseline)
    # Preselected physical comparison, not a post-hoc search over the scan.
    # Different interactions use independent preparations and random streams.
    contrasts = []
    for strategy, rows in (("grouped", runs), ("ungrouped", ungrouped_runs)):
        for shots in budgets:
            pair = [next(r for r in rows if r["interaction"] == u
                         and r["total_shots"] == 5 * shots) for u in (6, 8)]
            left, right = [r["observables"]["double_occupancy"] for r in pair]
            estimate = left["estimate"] - right["estimate"]
            radius = float(np.sqrt(np.log(40) * (
                left["squared_range_over_shots"] + right["squared_range_over_shots"]) / 2))
            contrasts.append({"grouping": strategy, "observable": "double_occupancy",
                              "interaction_pair": [6, 8], "grouped_shot_budget": shots,
                              "total_shots": sum(r["total_shots"] for r in pair),
                              "estimate": estimate, "ci": [estimate - radius, estimate + radius],
                              "confidence": 0.95, "resolved_decrease": bool(estimate - radius > 0),
                              "exact_prepared_state": left["exact_prepared_state"] - right["exact_prepared_state"],
                              "exact_ground_state": pair[0]["exact_ground_state"]["double_occupancy"]
                              - pair[1]["exact_ground_state"]["double_occupancy"]})
    return {"interactions": interactions, "budgets_per_setting": budgets,
            "method": "Independent Born multinomial draws after local Pauli basis rotations; exact-expectation VQE training",
            "interval_method": "Two-sided Hoeffding across independent weighted shots, retaining within-group covariance; marginal 95% CIs for prepared-state expectations. Older group union-bound intervals are retained as a diagnostic baseline.",
            "contrast_method": "Preselected D(U/t=6) minus D(U/t=8), using independent preparations at both interactions and a direct two-sided 95% Hoeffding bound. Positive lower bound resolves a decrease; no simultaneous scan coverage is claimed.",
            "cost": {"grouped_settings": 5, "ungrouped_nonidentity_terms": 10,
                     "assumptions": "Equal shots within each strategy; five grouped settings at n shots versus ten independent Pauli terms at n/2 shots, matching total preparations. All observables share those counts. Identity costs no shots. Shot totals exclude exact training evaluations, state-preparation circuit costs, basis-switching overhead and gate compilation. Neither allocation is optimized."},
            "runs": runs, "ungrouped_runs": ungrouped_runs, "contrasts": contrasts}
