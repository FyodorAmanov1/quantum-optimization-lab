# Quantum Optimization Lab

Constrained optimization with quantum circuits and strong classical baselines.

## Current implementation

Exact cardinality-constrained enumeration and a normalized one-layer QAOA statevector simulation, compared with uniform sampling over all bitstrings and uniform sampling over feasible portfolios. Each distribution reports feasibility probability, conditional feasible objective, additive gap to the exact feasible minimum, and expected draws per feasible sample.

These are exact probability-distribution expectations on one synthetic six-asset instance, not finite-shot measurements. The feasible-only baseline has no rejection; both uniform baselines have the same conditional objective. Results retain the original QAOA metric keys and add `metrics.sampling_comparison`. See [reports/QO01.md](reports/QO01.md) for measured findings, cost assumptions, and interpretation.

## Reproduce

Python 3.12 and CPU execution. From this repository:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python lab.py --output results/latest.json

The measured artifact is [results/latest.json](results/latest.json). Data provenance and scientific limitations are recorded alongside the metrics. Primary references and implementation boundaries are in [REFERENCES.md](REFERENCES.md).

## Research process



Each completed milestone adds a report under reports/ explaining the question, method, executable evidence, limitations, and concepts to understand for an interview. Negative results are retained.
