# Quantum Optimization Lab

Constrained optimization with quantum circuits and strong classical baselines.

## Current implementation

Exact cardinality-constrained enumeration and a normalized one-layer QAOA statevector simulation. It reports both feasibility probability and the conditional objective rather than hiding rejected samples.

## Reproduce

Python 3.12 and CPU execution. From this repository:

    python -m pip install -r requirements.txt
    python -m unittest discover -s tests -v
    python lab.py --output results/latest.json

The measured artifact is [results/latest.json](results/latest.json). Data provenance and scientific limitations are recorded alongside the metrics. Primary references and implementation boundaries are in [REFERENCES.md](REFERENCES.md).

## Research process

This is an AI-assisted research portfolio developed through bounded daily milestones. GPT-6.1 Sol at medium reasoning assists with implementation. Independent numerical checks and CI guard the main branch. Reports distinguish implemented behavior, exploratory results, and future work. Automated commits do not establish original research or personal mastery.

Each completed milestone adds a report under reports/ explaining the question, method, executable evidence, limitations, and concepts to understand for an interview. Negative results are retained.

The [portfolio controller](https://github.com/FyodorAmanov1/research-portfolio) contains the backlog and runtime policy. Initial baselines are runnable starting points; the full roadmap is not yet implemented.
