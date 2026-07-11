import csv
import os
import tempfile
import unittest

from src.milestone3 import build_pruning_env
from src.milestone3_xai import (
    format_explanation_report,
    most_pruned_device_type_pairs,
    most_pruned_edge_characteristics,
    per_node_pruning_summary,
    save_table_csv,
    summarize_pruning_characteristics,
)


class Milestone3XaiTests(unittest.TestCase):
    def setUp(self):
        self.env = build_pruning_env(n_nodes=40, model_seed=0, max_steps=5, chunk_fraction=0.15)
        self.env.reset()
        # Deterministically prune a couple of buckets so there's a non-trivial trajectory to summarize.
        self.env.step(0)
        self.env.step(1)
        self.trajectory_log = self.env.trajectory_log

    def test_summarize_pruning_characteristics_counts_match_trajectory(self):
        summary = summarize_pruning_characteristics(self.trajectory_log, self.env.original_graph, self.env.gat_scores, self.env.p_uv)
        total_removed = sum(len(step["edges_removed"]) for step in self.trajectory_log)
        self.assertEqual(summary["n_removed"], total_removed)
        self.assertGreater(summary["n_removed"], 0)
        self.assertIn("removed_s_uv_mean", summary)
        self.assertIn("population_s_uv_mean", summary)

    def test_most_pruned_device_type_pairs_sums_to_edge_count_when_unlimited(self):
        pairs = most_pruned_device_type_pairs(self.trajectory_log, top_n=1000)  # large enough to not truncate
        total_removed = sum(len(step["edges_removed"]) for step in self.trajectory_log)
        self.assertEqual(sum(count for _, count in pairs), total_removed)

    def test_most_pruned_device_type_pairs_respects_top_n(self):
        pairs = most_pruned_device_type_pairs(self.trajectory_log, top_n=2)
        self.assertLessEqual(len(pairs), 2)

    def test_per_node_pruning_summary_only_includes_that_nodes_edges(self):
        # Find a node that had at least one incident edge removed.
        removed_edges = [d["edge"] for step in self.trajectory_log for d in step["edges_removed_details"]]
        self.assertTrue(removed_edges)
        target_node = removed_edges[0][0]

        entries = per_node_pruning_summary(self.trajectory_log, self.env.node_features, target_node)
        self.assertTrue(entries)
        for entry in entries:
            self.assertIn(entry["neighbor"], [u if v == target_node else v for u, v in removed_edges if target_node in (u, v)])

    def test_format_explanation_report_is_nonempty_string(self):
        report = format_explanation_report(self.trajectory_log, self.env.original_graph, self.env.gat_scores, self.env.p_uv)
        self.assertIsInstance(report, str)
        self.assertIn("edges removed", report)

    def test_empty_trajectory_handled_gracefully(self):
        summary = summarize_pruning_characteristics([], self.env.original_graph, self.env.gat_scores, self.env.p_uv)
        self.assertEqual(summary["n_removed"], 0)
        report = format_explanation_report([], self.env.original_graph, self.env.gat_scores, self.env.p_uv)
        self.assertIn("No edges", report)

    def test_most_pruned_edge_characteristics_counts_match_and_are_sorted_descending(self):
        rows = most_pruned_edge_characteristics(self.trajectory_log)
        total_removed = sum(len(step["edges_removed"]) for step in self.trajectory_log)

        self.assertTrue(rows)
        self.assertEqual(sum(r["times_pruned"] for r in rows), total_removed)
        # Most-frequently-pruned pair first -- ranked descending by count.
        counts = [r["times_pruned"] for r in rows]
        self.assertEqual(counts, sorted(counts, reverse=True))
        for r in rows:
            self.assertIn("mean_s_uv", r)
            self.assertIn("mean_p_uv", r)
            self.assertIn("mean_hub_score", r)

    def test_save_table_csv_round_trips_rows(self):
        rows = most_pruned_edge_characteristics(self.trajectory_log)
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "edges.csv")
            returned_path = save_table_csv(rows, output_path)

            self.assertEqual(returned_path, output_path)
            self.assertTrue(os.path.exists(output_path))
            with open(output_path, newline="") as f:
                read_back = list(csv.DictReader(f))
            self.assertEqual(len(read_back), len(rows))
            self.assertEqual(read_back[0]["device_type_pair"], rows[0]["device_type_pair"])
            self.assertEqual(int(read_back[0]["times_pruned"]), rows[0]["times_pruned"])

    def test_save_table_csv_handles_empty_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "empty.csv")
            returned_path = save_table_csv([], output_path)

            self.assertTrue(os.path.exists(returned_path))
            with open(output_path, newline="") as f:
                content = f.read()
            self.assertEqual(content, "")


if __name__ == "__main__":
    unittest.main()
