"""Feasible-subspace portfolio QAOA and a correlated-electron VQE benchmark."""

from itertools import combinations
import numpy as np
from scipy.optimize import minimize

try:
    from .common import finalize, metric
except ImportError:
    from common import finalize, metric


def constrained_portfolio(depth=2):
    n, k = 8, 3
    states = np.array(
        [[(i in subset) for i in range(n)] for subset in combinations(range(n), k)],
        float,
    )
    returns = np.array([0.06, 0.09, 0.13, 0.07, 0.11, 0.08, 0.15, 0.10])
    volatility = np.array([0.12, 0.19, 0.28, 0.14, 0.23, 0.17, 0.33, 0.21])
    covariance = np.outer(volatility, volatility) * (
        0.4 ** np.abs(np.arange(n)[:, None] - np.arange(n))
    )
    previous = np.array([1, 1, 0, 1, 0, 0, 0, 0]) / k
    weights = states / k
    costs = (
        4 * np.einsum("bi,ij,bj->b", weights, covariance, weights)
        - weights @ returns
        + 0.003 * np.abs(weights - previous).sum(axis=1)
    )
    scaled = (costs - costs.min()) / np.ptp(costs)
    # Ring XY exchange: each edge moves one selected asset into its empty neighbour.
    mixer = np.zeros((len(states), len(states)))
    lookup = {tuple(s): i for i, s in enumerate(states)}
    for row, s in enumerate(states):
        for a in range(n):
            b = (a + 1) % n
            if s[a] != s[b]:
                swap = s.copy()
                swap[a], swap[b] = swap[b], swap[a]
                mixer[row, lookup[tuple(swap)]] += 1
    eigen, vectors = np.linalg.eigh(mixer)
    initial = np.ones(len(states), complex) / np.sqrt(len(states))

    def evolve(parameters):
        wave = initial.copy()
        for gamma, beta in np.asarray(parameters).reshape(-1, 2):
            wave *= np.exp(-1j * gamma * scaled)
            wave = vectors @ (np.exp(-1j * beta * eigen) * (vectors.T @ wave))
        return np.abs(wave) ** 2

    rng = np.random.default_rng(2026)
    runs = []
    for _ in range(6):
        result = minimize(
            lambda x: float(evolve(x) @ scaled),
            rng.uniform(-2, 2, 2 * depth),
            method="Nelder-Mead",
            options={"maxiter": 350, "xatol": 1e-7, "fatol": 1e-9},
        )
        runs.append((result, evolve(result.x)))
    result, prob = min(runs, key=lambda r: r[1] @ costs)
    assert np.isclose(prob.sum(), 1) and np.all(states.sum(axis=1) == k)
    # A budget-matched small classical local-search baseline, starting from prior allocation.
    current = int(np.argmin(np.sum(abs(weights - previous), axis=1)))
    visits = 0
    while True:
        adjacent = np.flatnonzero(np.sum(abs(states - states[current]), axis=1) == 2)
        visits += len(adjacent)
        best = int(adjacent[np.argmin(costs[adjacent])])
        if costs[best] >= costs[current] - 1e-12:
            break
        current = best
    return (
        states,
        costs,
        prob,
        returns,
        covariance,
        current,
        visits,
        sum(r[0].nfev for r in runs),
    )


def fermion(state, orbital, create):
    occupied = (state >> orbital) & 1
    if occupied == int(create):
        return None
    return state ^ (1 << orbital), (-1) ** ((state & ((1 << orbital) - 1)).bit_count())


def hubbard(interaction, hopping=1.0):
    # Orbitals: site0-up, site0-down, site1-up, site1-down; N_up=N_down=1.
    basis = [
        s
        for s in range(16)
        if ((s & 1) > 0) + ((s & 4) > 0) == 1 and ((s & 2) > 0) + ((s & 8) > 0) == 1
    ]
    h = np.zeros((4, 4))
    index = {s: i for i, s in enumerate(basis)}
    for col, s in enumerate(basis):
        h[col, col] = interaction * ((s & 3 == 3) + (s & 12 == 12))
        for a, b in [(0, 2), (2, 0), (1, 3), (3, 1)]:
            ann = fermion(s, b, False)
            cre = fermion(ann[0], a, True) if ann else None
            if cre:
                h[index[cre[0]], col] += -hopping * ann[1] * cre[1]
    assert np.allclose(h, h.T)
    return h, basis


def vqe(interaction):
    h, basis = hubbard(interaction)

    def state(angles):
        # Three real two-level rotations span the real four-dimensional sector.
        v = np.array([1.0, 0, 0, 0])
        for j, a in enumerate(angles, start=1):
            old = v[0]
            v[0] = np.cos(a) * old - np.sin(a) * v[j]
            v[j] = np.sin(a) * old + np.cos(a) * v[j]
        return v

    runs = [
        minimize(
            lambda a: state(a) @ h @ state(a),
            np.random.default_rng(100 + s).uniform(-2, 2, 3),
            method="BFGS",
            options={"gtol": 1e-9},
        )
        for s in range(3)
    ]
    result = min(runs, key=lambda r: r.fun)
    wave = state(result.x)
    exact = np.linalg.eigvalsh(h)[0]
    analytic = (interaction - np.sqrt(interaction**2 + 16)) / 2
    double = np.array([int(s & 3 == 3) + int(s & 12 == 12) for s in basis])
    assert (
        abs(exact - analytic) < 1e-10
        and result.fun >= exact - 1e-10
        and abs(result.fun - exact) < 1e-6
    )
    return (
        float(result.fun),
        float(exact),
        float(wave**2 @ double),
        float(np.linalg.norm(h @ wave - result.fun * wave)),
    )


def experiment():
    states, costs, prob, returns, cov, local, visits, evaluations = (
        constrained_portfolio()
    )
    optimum = int(np.argmin(costs))
    top = np.argsort(prob)[-10:][::-1]
    noise = np.linspace(0, 0.3, 7)
    mixed = [(1 - p) * prob + p / len(prob) for p in noise]
    allocation = lambda s: ", ".join(chr(65 + i) for i in np.flatnonzero(s))
    portfolio = {
        "id": "allocation",
        "label": "Constrained allocation",
        "kicker": "QAOA / FEASIBLE-SUBSPACE MIXER",
        "title": "Optimize the allocation. Keep the constraints.",
        "description": "Eight synthetic assets, exactly three holdings, turnover costs and correlated risk. Compare a number-preserving quantum ansatz with exhaustive enumeration and classical local search.",
        "metrics": [
            metric("Exact minimum cost", costs[optimum]),
            metric("QAOA expected cost", prob @ costs),
            metric("Optimal-sample probability", prob[optimum] * 100, "%"),
            metric("Feasible configurations", len(states)),
        ],
        "charts": [
            {
                "title": "Probability on feasible portfolios",
                "x": list(range(len(states))),
                "xlabel": "Lexicographic portfolio index",
                "series": [
                    {"name": "Depth-2 QAOA", "values": prob.tolist()},
                    {
                        "name": "Uniform baseline",
                        "values": (np.ones(len(states)) / len(states)).tolist(),
                    },
                ],
            },
            {
                "title": "Sensitivity to an explicit uniform-output mixture",
                "x": noise.tolist(),
                "xlabel": "Mixture fraction (not device error rate)",
                "series": [
                    {
                        "name": "Expected cost",
                        "values": [float(p @ costs) for p in mixed],
                    },
                    {
                        "name": "Exact optimum",
                        "values": [float(costs.min())] * len(noise),
                    },
                ],
            },
        ],
        "table": {
            "columns": [
                "Holdings",
                "Cost",
                "QAOA probability",
                "Expected return",
                "Volatility",
            ],
            "rows": [
                [
                    allocation(states[i]),
                    float(costs[i]),
                    float(prob[i]),
                    float(states[i] @ returns / 3),
                    float(np.sqrt((states[i] / 3) @ cov @ (states[i] / 3))),
                ]
                for i in top
            ],
        },
        "method": [
            "Objective = 4 wᵀΣw − μᵀw + 0.003‖w−w_previous‖₁; equal weights on exactly three of eight assets. Covariance is positive definite by construction.",
            "Depth-2 cost phases alternate with an exact ring-XY mixer in the 56-state feasible sector. Six seeded Nelder–Mead restarts each have a 350-iteration cap. Uniform feasible-state preparation is assumed, not compiled.",
            f"Exact enumeration gives {allocation(states[optimum])}. Swap local search gives {allocation(states[local])}, cost {costs[local]:.6f}, after {visits} neighbour evaluations. The ansatz is not claimed to outperform either baseline.",
        ],
        "limitations": [
            "Synthetic forward-looking means/covariance: no historical backtest or investment claim. Small exact simulation is the reference, not evidence of scalability.",
            "Mixer construction and dense exponentiation operate in a constrained subspace; physical preparation, compiled gates and measurement overhead are not benchmarked.",
            "Uniform probability mixing is a sensitivity model, not calibrated gate noise. Both expected objective and success probability must be inspected.",
        ],
        "reproduce": "python src/studio.py --out docs/study.json",
    }
    interactions = [0, 0.5, 1, 2, 4, 6, 8]
    results = [vqe(u) for u in interactions]
    materials = {
        "id": "electrons",
        "label": "Correlated electrons",
        "kicker": "VQE / MATERIALS FOUNDATION",
        "title": "Follow electrons from hopping to localization.",
        "description": "A spinful two-site Hubbard dimer isolates the competition between electron hopping and on-site repulsion. A symmetry-preserving variational state is checked against exact and analytic ground-state energies.",
        "metrics": [
            metric("Maximum energy error", max(abs(r[0] - r[1]) for r in results), "t"),
            metric("Spin orbitals", 4),
            metric("Sector dimension", 4),
            metric("Double occupancy at U/t=8", results[-1][2]),
        ],
        "charts": [
            {
                "title": "Ground-state energy across interaction strength",
                "x": interactions,
                "xlabel": "U / t",
                "series": [
                    {"name": "Variational energy", "values": [r[0] for r in results]},
                    {
                        "name": "Exact diagonalization",
                        "values": [r[1] for r in results],
                    },
                ],
            },
            {
                "title": "Charge localization",
                "x": interactions,
                "xlabel": "U / t",
                "series": [
                    {
                        "name": "Total double occupancy",
                        "values": [r[2] for r in results],
                    }
                ],
            },
        ],
        "table": {
            "columns": [
                "U/t",
                "VQE energy / t",
                "Exact / t",
                "Double occupancy",
                "Eigenstate residual norm",
            ],
            "rows": [[u, *r] for u, r in zip(interactions, results)],
        },
        "method": [
            "Construct −t Σσ(c†₀σc₁σ+h.c.) + UΣᵢnᵢ↑nᵢ↓ using fermionic parity signs; restrict to one spin-up and one spin-down electron.",
            "Three real two-level Givens rotations span the four-state real sector. Three seeded BFGS starts minimize the statevector energy for each interaction.",
            "Each result obeys the variational bound and agrees with E₀=(U−√(U²+16t²))/2 within 10⁻⁶ t. Double occupancy is a separate observable, not inferred from a fitted curve.",
        ],
        "limitations": [
            "This is an effective correlated-electron benchmark, not an ab initio molecule or a battery-material prediction. It is a foundation for extending the battery-design theme.",
            "The ansatz can span this tiny real sector; success does not demonstrate a scalable chemistry algorithm.",
            "No finite-shot Pauli measurement, molecular geometry, basis-set error, device noise or battery voltage is represented.",
        ],
        "reproduce": "python src/studio.py --out docs/study.json",
    }
    return {
        "schema_version": 1,
        "slug": "quantum-optimization-lab",
        "title": "Quantum Decision Studio",
        "subtitle": "Constrained allocation and variational correlated-electron research.",
        "accent": "#c5b2ff",
        "seed": 2026,
        "cases": [portfolio, materials],
        "provenance": "Explicit synthetic financial parameters and a defined effective Hamiltonian. CPU statevector calculations; no quantum hardware calls.",
        "references": [
            {
                "title": "Quantum alternating operator ansatz",
                "url": "https://arxiv.org/abs/1709.03489",
            },
            {
                "title": "Variational quantum eigensolver",
                "url": "https://arxiv.org/abs/1304.3061",
            },
            {
                "title": "Battery-design inspiration: EnergetiQ",
                "url": "https://github.com/FyodorAmanov1/EnergetiQ",
            },
        ],
    }


if __name__ == "__main__":
    finalize(experiment())
