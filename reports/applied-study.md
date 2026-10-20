# Applied case study: Quantum Decision Studio

## Research question

Build useful constrained decision tools and correlated-electron experiments, with exact references and honest implementation costs.

## Reproduce

`python src/studio.py --out docs/study.json`

## Constrained allocation

Eight synthetic assets, exactly three holdings, turnover costs and correlated risk. Compare a number-preserving quantum ansatz with exhaustive enumeration and classical local search.

### Method

Objective = 4 wᵀΣw − μᵀw + 0.003‖w−w_previous‖₁; equal weights on exactly three of eight assets. Covariance is positive definite by construction.

Depth-2 cost phases alternate with an exact ring-XY mixer in the 56-state feasible sector. Six seeded Nelder–Mead restarts each have a 350-iteration cap. Uniform feasible-state preparation is assumed, not compiled.

Exact enumeration gives A, D, H. Swap local search gives A, D, H, cost -0.038294, after 30 neighbour evaluations. The ansatz is not claimed to outperform either baseline.

### Measured comparison

| Holdings | Cost | QAOA probability | Expected return | Volatility |
| --- | --- | --- | --- | --- |
| A, D, F | -0.0355181 | 0.28713 | 0.07 | 0.0901136 |
| B, E, H | -0.0314654 | 0.145924 | 0.1 | 0.127018 |
| A, D, H | -0.0382941 | 0.105287 | 0.0766667 | 0.095358 |
| B, D, H | -0.0357137 | 0.0950543 | 0.0866667 | 0.110627 |
| A, B, F | -0.0303504 | 0.0363758 | 0.0766667 | 0.105257 |
| A, C, F | -0.0242388 | 0.0358908 | 0.09 | 0.124259 |
| B, F, H | -0.0315535 | 0.0357117 | 0.09 | 0.116669 |
| B, D, F | -0.032497 | 0.035373 | 0.08 | 0.106657 |
| A, E, H | -0.0330764 | 0.0346274 | 0.09 | 0.115026 |
| A, B, E | -0.0274903 | 0.0299485 | 0.0866667 | 0.119558 |

### Limits

- Synthetic forward-looking means/covariance: no historical backtest or investment claim. Small exact simulation is the reference, not evidence of scalability.
- Mixer construction and dense exponentiation operate in a constrained subspace; physical preparation, compiled gates and measurement overhead are not benchmarked.
- Uniform probability mixing is a sensitivity model, not calibrated gate noise. Both expected objective and success probability must be inspected.

## Correlated electrons

A spinful two-site Hubbard dimer isolates the competition between electron hopping and on-site repulsion. A symmetry-preserving variational state is checked against exact and analytic ground-state energies.

### Method

Construct −t Σσ(c†₀σc₁σ+h.c.) + UΣᵢnᵢ↑nᵢ↓ using fermionic parity signs; restrict to one spin-up and one spin-down electron.

Three real two-level Givens rotations span the four-state real sector. Three seeded BFGS starts minimize the statevector energy for each interaction.

Each result obeys the variational bound and agrees with E₀=(U−√(U²+16t²))/2 within 10⁻⁶ t. Double occupancy is a separate observable, not inferred from a fitted curve.

### Measured comparison

| U/t | VQE energy / t | Exact / t | Double occupancy | Eigenstate residual norm |
| --- | --- | --- | --- | --- |
| 0 | -2 | -2 | 0.5 | 1.8168e-08 |
| 0.5 | -1.76556 | -1.76556 | 0.437983 | 2.29564e-08 |
| 1 | -1.56155 | -1.56155 | 0.378732 | 2.73479e-08 |
| 2 | -1.23607 | -1.23607 | 0.276393 | 2.3624e-08 |
| 4 | -0.828427 | -0.828427 | 0.146447 | 3.55486e-08 |
| 6 | -0.605551 | -0.605551 | 0.0839749 | 3.48891e-08 |
| 8 | -0.472136 | -0.472136 | 0.0527864 | 4.00022e-08 |

### Limits

- This is an effective correlated-electron benchmark, not an ab initio molecule or a battery-material prediction. It is a foundation for extending the battery-design theme.
- The ansatz can span this tiny real sector; success does not demonstrate a scalable chemistry algorithm.
- No finite-shot Pauli measurement, molecular geometry, basis-set error, device noise or battery voltage is represented.

## Interpretation

The depth-2 ansatz improves optimal-sample probability over uniform feasible sampling, but its expected objective remains above the exact minimum and local search is a strong baseline. The Hubbard energy agreement follows from a tiny fully expressive real-sector ansatz; double occupancy provides an independent physical observable.

## Concepts to explain in an interview

Explain the objective and units, where each baseline obtains its information, what was fixed before evaluation, how costs and uncertainty were measured, and one failure the model cannot capture. Reproduce the data before presenting a number; distinguish an implementation demonstration from a validated research conclusion.

## Next work

- Add finite-shot observable measurement to the Hubbard VQE, with measurement grouping and uncertainty.
- Compare constrained ansatz depths and classical solvers at matched objective-evaluation budgets.
- Extend the effective electron model and assess when the restricted ansatz fails; do not relabel toy energies as battery properties.
