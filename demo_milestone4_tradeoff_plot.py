"""Milestone 4: security/utility trade-off figure, the supervisor-requested
headline plot -- containment ratio vs. frozen F1, one point per method, each
annotated with its mean compute cost, with error bars showing seed-to-seed
spread in both dimensions (per reviewer request on Figure 3).

Reads harness_summary.csv (mean/std of containment_ratio, already computed
by demo_milestone4_harness.py's 40-graph run) and harness_per_graph.csv (the
raw per-graph values that CSV's std columns don't cover -- frozen_f1_std
isn't saved anywhere, so it's computed here from harness_per_graph.csv's raw
frozen_f1 column, grouped by method). No experiment is rerun here, this is
pure plotting/aggregation over already-collected results.
"""

import csv
import statistics
from collections import defaultdict

from src.milestone2_pruning import plot_containment_vs_utility_tradeoff

SUMMARY_CSV_PATH = "harness_summary.csv"
PER_GRAPH_CSV_PATH = "harness_per_graph.csv"


def load_summary_rows(path: str) -> list:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "method": row["method"],
                    "containment_ratio_mean": float(row["containment_ratio_mean"]),
                    "containment_ratio_std": float(row["containment_ratio_std"]),
                    "frozen_f1_mean": float(row["frozen_f1_mean"]),
                    "mean_time_s": float(row["mean_time_s"]),
                }
            )
        return rows


def load_frozen_f1_std_by_method(path: str) -> dict:
    """frozen_f1_std per method, computed from harness_per_graph.csv's raw
    per-graph frozen_f1 values -- not itself a saved column anywhere, unlike
    containment_ratio_std (already in harness_summary.csv)."""
    frozen_f1_by_method = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frozen_f1_by_method[row["method"]].append(float(row["frozen_f1"]))
    return {method: statistics.stdev(values) for method, values in frozen_f1_by_method.items()}


def main() -> None:
    rows = load_summary_rows(SUMMARY_CSV_PATH)
    frozen_f1_std_by_method = load_frozen_f1_std_by_method(PER_GRAPH_CSV_PATH)
    for row in rows:
        row["frozen_f1_std"] = frozen_f1_std_by_method[row["method"]]

    output_path = plot_containment_vs_utility_tradeoff(rows, output_path="containment_vs_utility_tradeoff_v2.pdf")
    print(f"saved {output_path} ({len(rows)} methods plotted from {SUMMARY_CSV_PATH} + {PER_GRAPH_CSV_PATH})")


if __name__ == "__main__":
    main()
