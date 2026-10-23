# Quantum Decision Studio

**Two variational quantum problems with explicit classical reference answers: constrained portfolio selection and correlated-electron ground states.**

Quantum Decision Studio makes the objective, feasible states and approximation error visible. Its interactive application connects allocation decisions to probability distributions and electron correlation to energy and double occupancy.

[Open the studio](https://fyodoramanov1.github.io/quantum-optimization-lab/) · [Read the study](reports/applied-study.md)

## Portfolio allocation with QAOA

Choose exactly three of eight assets with equal weights. The objective combines expected return, covariance risk and turnover:

```text
cost(w) = 4 wᵀΣw − μᵀw + 0.003 ‖w − w_previous‖₁
```

There are 56 feasible portfolios. A depth-two cost circuit and a ring XY mixer evolve a uniform state within that feasible space, preserving the holding count. Six seeded Nelder–Mead restarts optimize the variational parameters.

Exhaustive enumeration supplies the exact answer; a classical swap search supplies another practical reference. Their search budgets are reported separately and should not be interpreted as a controlled runtime comparison.

### What the saved run shows

| Quantity | Recorded value |
|---|---:|
| Exact optimal holdings | A, D, H |
| Exact minimum cost | −0.038294 |
| QAOA expected cost | −0.030785 |
| Probability of the optimal portfolio | 10.53% |
| Uniform probability of that portfolio | 1.79% |

The most probable sampled allocation is **A, D, F**, which differs from the exact optimum. Expected objective value, the modal state and optimal-state probability are deliberately shown as distinct outcomes.

## Electron correlation with VQE

The second experiment studies a two-site Hubbard model with four spin orbitals in the sector containing one spin-up and one spin-down electron. Three real Givens rotations form a fully expressive ansatz for this small real sector.

Across seven interaction strengths from U/t = 0 to 8, the optimized energy is compared with exact diagonalization and the analytic ground-state expression. The recorded maximum discrepancy is approximately **5.6 × 10⁻¹⁶ t**, while double occupancy falls from **0.5000** to **0.0528** as repulsion increases.

The **Hubbard measurements** studio case now samples energy, total double occupancy and kinetic energy using five local Pauli settings. It compares 256, 4,096 and 65,536 shots per setting with exact references, reports conservative marginal 95% confidence intervals, and exports all budgets as CSV. At 4,096 shots per setting, measured double occupancy falls from **0.49976 ± 0.02122** to **0.05249 ± 0.02122** across the scan endpoints. These uncertainties are confidence-bound half-widths. The [finite-shot study](reports/QO02.md) explains the wider energy bounds, raw counts, grouping and costs.

This is a transparent demonstration of electron correlation on a tiny solvable system. It is not a battery chemistry calculation or a prediction of voltage, capacity or materials performance.

## What to examine

- Compare the full allocation distribution with the exact cost ranking.
- Inspect how turnover affects the allocation objective.
- Follow energy and double occupancy as interaction strength increases.
- Read the preparation and optimization assumptions before interpreting resource counts.

The calculations use classical simulation. Uniform feasible-state preparation is assumed rather than compiled, and the probability-mixture sensitivity is not calibrated device noise. Hubbard evaluation uses ideal finite-shot Pauli measurements after exact-expectation VQE training. No hardware advantage is claimed.

## Run locally

Use Python 3.12. From the repository root:

```bash
python -m venv .venv
```

Activate it with `.venv/Scripts/Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux. Then run:

```bash
python -m pip install -r requirements.txt
python -m http.server 8000 --directory docs
```

Open
<http://localhost:8000> to explore the bundled study without recomputing it.

To regenerate the study or compare it with the saved artifact:

```bash
python src/studio.py --out docs/study.json
python src/studio.py --check docs/study.json
python lab.py --output results/latest.json
python -m unittest discover -s tests -v
```

The browser displays recorded experiments; it does not run the Python simulation.
Use its charts, case selector and data exports to inspect the results.

## Explore the implementation

- [Experiment implementation](src/studio.py)
- [Recorded study](docs/study.json)
- [Applied study report](reports/applied-study.md)
- [References](REFERENCES.md)
- [MIT license](LICENSE)
