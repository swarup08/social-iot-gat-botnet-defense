# Environment

## Original results (Table 1 and all reported numbers)

Exact package versions and hardware specifications were not logged at the
time the reported results were produced. This project does not require a
GPU at the graph scale used here; all runs were performed on a standard
CPU-only machine. (This is the same disclosure stated in the paper's
Reproducibility Appendix.)

## Environment captured for this reproducibility release

The following was captured when preparing this release (i.e. *after* the
original runs, as part of packaging the repository for reproducibility) --
it documents an environment that reproduces the same code paths and
methodology going forward, **not** a certified match to the exact
environment the original frozen results were produced in.

- **OS:** Windows 11 (build 10.0.26200)
- **Python:** 3.13.6
- **Packages** (see `requirements.txt` for the pinned list used to install
  these): `numpy==2.2.0`, `networkx==3.5`, `matplotlib==3.10.7`,
  `torch==2.9.1+cpu`, `torch-geometric==2.7.0`, `scipy==1.16.3`,
  `adjustText==1.4.0`
- **Hardware:** CPU-only (`torch==2.9.1+cpu`, the CPU-only build); no GPU
  used or required at this project's graph scale (N up to 600 nodes).

## For future reproduction

Install with:
```bash
pip install -r requirements.txt
```

If reproducing on a different OS or Python version, note that exact
floating-point results (e.g. the 15th significant digit of a reported
mean) may differ slightly across platforms/library versions even with
identical random seeds -- the project's own conclusions are drawn from
values reported to 3 decimal places with reported standard deviations,
which is well within any such platform-level noise.
