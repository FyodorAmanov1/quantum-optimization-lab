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
    mass = float(probabilities[feasible].sum())
    optimum = exact_portfolio(returns, covariance)
    conditional = float(probabilities[feasible] @ raw[feasible] / mass)
    return {"project": "quantum-optimization-lab", "seed": 1729,
            "metrics": {"exact": optimum, "qaoa_expected_penalized_objective": value,
                        "feasible_probability": mass, "conditional_feasible_objective": conditional,
                        "conditional_gap_to_optimum": conditional - optimum["objective"],
                        "gamma": float(gamma), "beta": float(beta), "parameter_evaluations": 289,
                        "logical_qubits": 6, "statevector_amplitudes": 64},
            "provenance": {"data": "Synthetic PSD covariance and expected returns", "numpy": np.__version__},
            "limitations": ["Exponential CPU statevector and a coarse p=1 grid; no quantum speedup claim.",
                            "Feasibility is measured before postselection; conditional objective excludes rejected samples.",
                            "Binary unit holdings, not continuous normalized portfolio weights."]}


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
