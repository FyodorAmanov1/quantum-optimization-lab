# Primary references and implementation boundaries

- [Quantum alternating operator ansatz](https://arxiv.org/abs/1709.03489)
- [Variational quantum eigensolver](https://arxiv.org/abs/1304.3061)
- [Verteletskyi, Yen and Izmaylov: Measurement Optimization in the Variational Quantum Eigensolver Using a Minimum Clique Cover (2020)](https://arxiv.org/abs/1907.03358): qubit-wise commuting measurement groups. This implementation uses an explicit five-setting partition, not a general clique-cover optimizer.
- [Hoeffding: Probability Inequalities for Sums of Bounded Random Variables (1963)](https://doi.org/10.1080/01621459.1963.10500830): bounded independent sample means. Applied per measurement group with a union bound to obtain conservative marginal confidence intervals.
- [Carrascal et al.: The Hubbard Dimer: A density functional case study of a many-body problem (2015)](https://arxiv.org/abs/1502.02194): exact solvable dimer context. The implemented symmetric, half-filled dimer uses t=1 and no external site potential.
- [Battery-design inspiration: EnergetiQ](https://github.com/FyodorAmanov1/EnergetiQ)

The implementation, data provenance and scope limits are documented in [the technical case study](reports/applied-study.md). Citation identifies an algorithmic foundation, not equivalence with the full paper or a claim of hardware performance.

New QO02 links were verified against arXiv and the publisher's indexed Hoeffding record. The DOI landing page restricts automated retrieval; its title, author, volume and pages were verified through the publisher search record. See [QO02](reports/QO02.md) for the implemented estimator and interval formula.
