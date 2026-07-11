import unittest

import networkx as nx

from src.milestone2_pruning import (
    calibrate_topk_for_target_fraction,
    prune_highest_score,
    prune_lowest_score,
    prune_random,
    prune_topk_per_node,
    score_edges_by_betweenness,
)


class Milestone2PruningTests(unittest.TestCase):
    def setUp(self):
        self.graph = nx.barabasi_albert_graph(30, 3, seed=1)
        # Deterministic, strictly-increasing scores make "lowest"/"highest" unambiguous to check.
        self.scores = {edge: float(i) for i, edge in enumerate(self.graph.edges())}

    def test_prune_lowest_score_removes_the_lowest_scored_edges(self):
        pruned = prune_lowest_score(self.graph, self.scores, remove_fraction=0.3)

        n_original = self.graph.number_of_edges()
        n_removed = n_original - pruned.number_of_edges()
        self.assertEqual(n_removed, round(0.3 * n_original))

        removed_edges = set(self.graph.edges()) - set(pruned.edges())
        kept_edges = set(pruned.edges())
        max_removed_score = max(self.scores[e] for e in removed_edges)
        min_kept_score = min(self.scores.get(e, self.scores.get((e[1], e[0]))) for e in kept_edges)
        self.assertLess(max_removed_score, min_kept_score)

    def test_prune_highest_score_removes_the_highest_scored_edges(self):
        pruned = prune_highest_score(self.graph, self.scores, remove_fraction=0.3)

        removed_edges = set(self.graph.edges()) - set(pruned.edges())
        kept_edges = set(pruned.edges())
        min_removed_score = min(self.scores[e] for e in removed_edges)
        max_kept_score = max(self.scores.get(e, self.scores.get((e[1], e[0]))) for e in kept_edges)
        self.assertGreater(min_removed_score, max_kept_score)

    def test_prune_random_removes_correct_count_and_preserves_nodes(self):
        pruned = prune_random(self.graph, remove_fraction=0.3, seed=1)

        n_original = self.graph.number_of_edges()
        self.assertEqual(pruned.number_of_edges(), n_original - round(0.3 * n_original))
        self.assertEqual(pruned.number_of_nodes(), self.graph.number_of_nodes())

    def test_prune_topk_per_node_preserves_all_nodes_and_reduces_edges(self):
        pruned = prune_topk_per_node(self.graph, self.scores, k=2)

        self.assertEqual(pruned.number_of_nodes(), self.graph.number_of_nodes())
        self.assertLessEqual(pruned.number_of_edges(), self.graph.number_of_edges())
        # Union rule: every node should retain AT LEAST min(k, its original degree) edges.
        original_degree = dict(self.graph.degree())
        pruned_degree = dict(pruned.degree())
        for node in self.graph.nodes():
            self.assertGreaterEqual(pruned_degree[node], min(2, original_degree[node]))

    def test_calibrate_topk_reaches_reasonably_close_to_target_fraction(self):
        k, pruned, achieved_fraction = calibrate_topk_for_target_fraction(self.graph, self.scores, target_remove_fraction=0.3)

        self.assertGreaterEqual(k, 1)
        self.assertAlmostEqual(achieved_fraction, 0.3, delta=0.15)
        # The returned pruned graph should be exactly prune_topk_per_node(graph, scores, k).
        self.assertEqual(pruned.number_of_edges(), prune_topk_per_node(self.graph, self.scores, k).number_of_edges())

    def test_score_edges_by_betweenness_covers_every_edge(self):
        scores = score_edges_by_betweenness(self.graph)

        for u, v in self.graph.edges():
            self.assertTrue((u, v) in scores or (v, u) in scores)


if __name__ == "__main__":
    unittest.main()
