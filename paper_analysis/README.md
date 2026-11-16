# Paper analysis scripts (tables, figures and paired tests)

These scripts regenerate the figures, tables, paired tests and stress summaries of
"Cascade Containment and Classifier Utility in Synthetic Network Pruning" from the
released per-graph CSV records. They do not rerun simulation, GAT training or DQN training.

- `build_results.py` generates `source/table_main.tex`, `source/table_containment_tests.tex`
  (paper Table 2) and `source/table_f1_tests.tex` (paper Table 3), the result figures,
  and `main_checked.csv`, `paired_checked.csv`, `stress_checked.csv`.
- `verify_release.py` checks cohort completeness, paired statistics, input hashes
  (`INPUT_SHA256.json`), achieved-budget records and correlation summaries.
- `build_protocol.py` draws the protocol schematic (no experimental outcomes).
- `data/` holds the four input CSVs (same content as the CSVs in the repository root; line endings may differ).

Run from this folder: `python build_results.py && python verify_release.py`
(requires NumPy, pandas, SciPy and Matplotlib).

## Relation to `harness_paired_tests.csv`

`../harness_paired_tests.csv` is an earlier analysis with a different comparison set
(it includes pairs not referenced to degree). The paper's 14 tests are the seven
method-minus-degree comparisons on each of containment and frozen F1, Holm-adjusted
jointly. They are produced by `build_results.py` and stored in `paired_checked.csv`.
Eigenscore and greedy search are in the eight-method main analysis; the stress suite
does not include them.
