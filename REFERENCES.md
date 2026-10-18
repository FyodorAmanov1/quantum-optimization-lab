# Primary references and boundaries

- Markowitz (1952), Portfolio Selection: https://doi.org/10.1111/j.1540-6261.1952.tb01525.x
- Farhi, Goldstone and Gutmann, A Quantum Approximate Optimization Algorithm: https://arxiv.org/abs/1411.4028
- Hadfield, Wang, O'Gorman, Rieffel, Venturelli and Biswas (2019), From the Quantum Approximate Optimization Algorithm to a Quantum Alternating Operator Ansatz: https://arxiv.org/abs/1709.03489 (journal DOI: https://doi.org/10.3390/a12020034). Discusses mixers restricted to feasible subspaces; such a mixer is not implemented here. The feasible uniform baseline is a classical distribution.

The experiment uses synthetic returns, unit binary holdings, and a penalty constraint. It is a small CPU simulation, not a practical portfolio recommendation or hardware advantage demonstration.

QO01 link verification: the two arXiv abstract pages and the Markowitz DOI landing page were opened successfully on 2026-10-18. These references motivate the model and algorithm family; they do not establish performance of this implementation.

The uniform baseline identities are elementary finite-sum derivations, checked in `tests/test_sampling.py`: for n bits and exactly k selected assets, feasibility under uniform bitstrings is C(n,k)/2^n. Conditioning that distribution on feasibility assigns every feasible string probability 1/C(n,k), identical to the feasible-only baseline. Independent rejection sampling has expected draws 1/P(feasible). These are sampling-law properties, not new research claims. Additive gaps avoid division by a negative or zero optimum.
