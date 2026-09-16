"""Milestone 4: security/utility trade-off figure, the supervisor-requested
headline plot -- containment ratio vs. frozen F1, one point per method, each
annotated with its mean compute cost.

Reads harness_summary.csv only (already produced by demo_milestone4_harness.py's
40-graph run) -- no experiment is rerun here, this is pure plotting.
"""

import csv

from src.milestone2_pruning import plot_containment_vs_utility_tradeoff

SUMMARY_CSV_PATH = "harness_summary.csv"


def load_summary_rows(path: str) -> list:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append(
                {
                    "method": row["method"],
                    "containment_ratio_mean": float(row["containment_ratio_mean"]),
                    "frozen_f1_mean": float(row["frozen_f1_mean"]),
                    "mean_time_s": float(row["mean_time_s"]),
                }
            )
        return rows


def main() -> None:
    rows = load_summary_rows(SUMMARY_CSV_PATH)
    output_path = plot_containment_vs_utility_tradeoff(rows, output_path="containment_vs_utility_tradeoff_v2.pdf")
    print(f"saved {output_path} ({len(rows)} methods plotted from {SUMMARY_CSV_PATH})")


if __name__ == "__main__":
    main()
