"""Small exact QUBO baseline and a one-layer statevector QAOA experiment."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def problem(n=6, seed=1729):
    if not 2 <= n <= 12:
        raise ValueError("This educational statevector experiment supports 2..12 assets")
    rng = np.random.default_rng(seed)
    factor = rng.normal(size=(n, n))
    covariance = factor @ factor.T / (100 * n)
    returns = rng.uniform(0.01, 0.08, size=n)
    return returns, covariance


def objective(bits, returns, covariance, risk_aversion=2.0):
    x = np.asarray(bits, dtype=float)
    return float(risk_aversion * x @ covariance @ x - returns @ x)


def exact_portfolio(returns, covariance, cardinality=3):
    if not 0 <= cardinality <= len(returns):
        raise ValueError("Invalid cardinality")
    candidates = [(objective(bits, returns, covariance), bits)
                  for bits in itertools.product((0, 1), repeat=len(returns)) if sum(bits) == cardinality]
    value, bits = min(candidates)
    return {"bits": list(bits), "objective": value, "candidates": len(candidates)}


def qaoa_probabilities(energies, gamma, beta):
    energies = np.asarray(energies, dtype=float)
    n = int(np.log2(len(energies)))
    if len(energies) != 2**n:
        raise ValueError("One energy per computational basis state required")
    state = np.exp(-1j * gamma * energies) / np.sqrt(len(energies))
    c, s = np.cos(beta), -1j * np.sin(beta)
    for bit in range(n):
        stride = 1 << bit
        for start in range(0, len(state), stride * 2):
            a = state[start:start + stride].copy()
            b = state[start + stride:start + 2 * stride].copy()
            state[start:start + stride] = c * a + s * b
            state[start + stride:start + 2 * stride] = s * a + c * b
    return np.abs(state)**2


def uniform_probabilities(size, feasible=None):
    """Exact sampling law over all states, or only the supplied feasible states."""
    if not isinstance(size, (int, np.integer)) or size <= 0:
        raise ValueError("size must be a positive integer")
    support = np.ones(size, dtype=bool) if feasible is None else np.asarray(feasible, dtype=bool)
    if support.shape != (size,) or not support.any():
        raise ValueError("feasible must select at least one state in a vector of length size")
    return support.astype(float) / support.sum()


def distribution_metrics(probabilities, raw, feasible, optimum):
    """Evaluate a sampling law for a minimization objective, before/after rejection.

    Gaps are additive in objective units, so negative optima cause no ratio ambiguity.
    Conditional metrics and rejection cost are undefined (None) at zero feasible mass.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    raw = np.asarray(raw, dtype=float)
    feasible = np.asarray(feasible, dtype=bool)
    if (probabilities.ndim != 1 or probabilities.size == 0
            or raw.shape != probabilities.shape or feasible.shape != probabilities.shape):
        raise ValueError("probabilities, raw, and feasible must be matching nonempty vectors")
    if (not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-12)):
        raise ValueError("probabilities must be finite, nonnegative, and normalized")
    if not np.all(np.isfinite(raw)) or not np.isfinite(optimum):
        raise ValueError("objectives must be finite")
    # A normalized floating-point law can sum a few ulps above one.
    mass = min(1.0, float(probabilities[feasible].sum()))
    conditional = float(probabilities[feasible] @ raw[feasible] / mass) if mass > 0 else None
    return {"feasible_probability": mass,
            "conditional_feasible_objective": conditional,
            "conditional_gap_to_optimum": conditional - float(optimum) if mass > 0 else None,
            "expected_draws_per_feasible": 1.0 / mass if mass > 0 else None}


def experiment():
    returns, covariance = problem()
    bits = np.array(list(itertools.product((0, 1), repeat=len(returns))))
    raw = np.array([objective(x, returns, covariance) for x in bits])
    feasible = bits.sum(axis=1) == 3
    penalty = 1.0
    energies = raw + penalty * (bits.sum(axis=1) - 3)**2
    best = None
    for gamma in np.linspace(0, 2 * np.pi, 17):
        for beta in np.linspace(0, np.pi, 17):
            probabilities = qaoa_probabilities(energies, gamma, beta)
            value = float(probabilities @ energies)
            if best is None or value < best[0]:
                best = value, gamma, beta, probabilities
    value, gamma, beta, probabilities = best
    optimum = exact_portfolio(returns, covariance)
    distributions = {"qaoa": probabilities,
                     "uniform_all_bitstrings": uniform_probabilities(len(bits)),
                     "uniform_feasible_portfolios": uniform_probabilities(len(bits), feasible)}
    comparison = {name: distribution_metrics(p, raw, feasible, optimum["objective"])
                  for name, p in distributions.items()}
    for name, p in distributions.items():
        comparison[name]["expected_penalized_objective"] = float(p @ energies)
    return {"project": "quantum-optimization-lab", "seed": 1729,
            "metrics": {"exact": optimum, "qaoa_expected_penalized_objective": value,
                        "feasible_probability": comparison["qaoa"]["feasible_probability"],
                        "conditional_feasible_objective": comparison["qaoa"]["conditional_feasible_objective"],
                        "conditional_gap_to_optimum": comparison["qaoa"]["conditional_gap_to_optimum"],
                        "gamma": float(gamma), "beta": float(beta), "parameter_evaluations": 289,
                        "logical_qubits": 6, "statevector_amplitudes": 64,
                        "sampling_comparison": comparison},
            "method": {"evaluation": "Exact distribution expectations; no finite-shot sampling",
                       "assets": 6, "cardinality": 3, "risk_aversion": 2.0, "penalty": penalty,
                       "objective": "Minimize 2*x.T@covariance@x - returns@x",
                       "gap": "Conditional feasible objective minus exact feasible minimum (additive)",
                       "qaoa_selection": "Minimum expected penalized objective on the fixed 17 by 17 grid",
                       "cost_assumptions": {
                           "shared_enumerated_bitstrings": len(bits),
                           "feasible_portfolios": int(feasible.sum()),
                           "qaoa_statevector_evaluations": 289,
                           "uniform_baseline_parameter_evaluations": 0,
                           "rejection_cost": "1 / feasible_probability independent draws per accepted portfolio",
                           "feasible_sampler": "Uniform over fixed-cardinality subsets; no rejected draws",
                           "excluded_costs": "No timing, hardware, monetary, or end-to-end speed comparison; enumeration shared"}},
            "provenance": {"data": "Synthetic PSD covariance and expected returns", "numpy": np.__version__},
            "limitations": ["Exponential CPU statevector and a coarse p=1 grid; no quantum speedup claim.",
                            "Feasibility is measured before postselection; conditional objective excludes rejected samples.",
                            "Binary unit holdings, not continuous normalized portfolio weights.",
                            "One synthetic instance; exact distribution metrics are not finite-shot observations.",
                            "Uniform feasible distribution uses the enumerated mask here; general constraints may be harder to sample.",
                            "Training uses exact penalized expectations; rejection cost excludes training and sampler setup."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(experiment(), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        print(result)
