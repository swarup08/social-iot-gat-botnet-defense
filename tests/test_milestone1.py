import os
import tempfile
import unittest

import networkx as nx

from src.milestone1 import (
    DEVICE_TYPES,
    build_edge_features,
    build_node_features,
    build_social_iot_graph,
    compute_edge_infection_probabilities,
    plot_degree_distribution,
    plot_infection_history,
    simulate_botnet,
)


class Milestone1Tests(unittest.TestCase):
    def test_build_social_iot_graph_is_scale_free_and_has_expected_size(self):
        graph = build_social_iot_graph(n_nodes=50, m=3, seed=7)

        self.assertEqual(graph.number_of_nodes(), 50)
        self.assertEqual(graph.number_of_edges(), 141)
        self.assertTrue(nx.is_connected(graph))

    def test_edge_infection_probabilities_are_feature_driven_and_bounded(self):
        graph = nx.Graph()
        graph.add_edge(0, 1)
        node_features = {0: {"risk": 0.9, "hub_score": 1.0, "community": 0}, 1: {"risk": 0.2, "hub_score": 0.3, "community": 0}}
        edge_features = {(0, 1): {"interaction": 0.8}}

        probabilities = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=[-2.0, 1.5, -1.0, 0.5, 0.3, 0.4, 0.8])

        self.assertIn((0, 1), probabilities)
        self.assertGreater(probabilities[(0, 1)], 0.0)
        self.assertLess(probabilities[(0, 1)], 1.0)

    def test_simulate_botnet_respects_initial_compromised_seed(self):
        graph = nx.path_graph(4)
        node_features = {node: {"risk": 0.5, "hub_score": 0.5, "community": 0} for node in graph.nodes}
        edge_features = {(u, v): {"interaction": 0.4} for u, v in graph.edges}
        probabilities = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=[-1.5, 0.8, 0.2, 0.1, 0.1, 0.0, 0.0])

        result = simulate_botnet(graph, probabilities, initial_compromised={0}, seed=11)

        self.assertEqual(result["infected_nodes"], {0})
        self.assertEqual(result["infection_history"][0], 1)

    def test_simulate_botnet_batches_simultaneous_infections_into_one_round(self):
        # A small branching graph: seed 0 has two children (1, 2), each of which
        # has one further child (3, 4). Round 1 should infect {1, 2} together,
        # and round 2 should infect {3, 4} together, if simulate_botnet is truly
        # synchronous rather than processing one node at a time.
        graph = nx.Graph()
        graph.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4)])
        node_features = {node: {"risk": 0.0, "hub_score": 0.0, "community": 0} for node in graph.nodes}
        edge_features = {(u, v): {"interaction": 0.0} for u, v in graph.edges}
        # A large bias with all other weights zeroed drives every p_uv to
        # sigmoid(50), which rounds to exactly 1.0 in float64 -- so every
        # edge trial is guaranteed to succeed and the rollout is deterministic.
        probabilities = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=[50.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

        result = simulate_botnet(graph, probabilities, initial_compromised={0}, seed=1)

        self.assertEqual(result["infected_nodes"], {0, 1, 2, 3, 4})
        # Round 0: seed only (1). Round 1: +{1,2} together (3). Round 2: +{3,4}
        # together (5). Round 3: no new neighbors left, frontier empties (5).
        # A node-by-node cascade (rather than round-synchronous) would instead
        # spread {1,2} and {3,4} across separate history entries.
        self.assertEqual(result["infection_history"], [1, 3, 5, 5])

    def test_build_node_features_uses_real_structure_and_device_types(self):
        graph = build_social_iot_graph(n_nodes=40, m=3, seed=5)

        node_features = build_node_features(graph)

        self.assertEqual(set(node_features.keys()), set(graph.nodes()))
        degree_centrality = nx.degree_centrality(graph)
        for node, features in node_features.items():
            # hub_score must be the graph's real degree centrality, not a placeholder.
            self.assertAlmostEqual(features["hub_score"], degree_centrality[node])
            self.assertIn(features["device_type"], DEVICE_TYPES)
            self.assertGreaterEqual(features["risk"], 0.0)
            self.assertLessEqual(features["risk"], 1.0)
            self.assertIsInstance(features["community"], int)
        # All five roadmap device types should appear on a graph this size.
        self.assertEqual({f["device_type"] for f in node_features.values()}, set(DEVICE_TYPES))

    def test_build_edge_features_uses_real_structure(self):
        graph = build_social_iot_graph(n_nodes=40, m=3, seed=5)

        edge_features = build_edge_features(graph)

        self.assertEqual(len(edge_features), graph.number_of_edges())
        for value in edge_features.values():
            self.assertIn("interaction", value)
            self.assertGreaterEqual(value["interaction"], 0.0)
            self.assertLessEqual(value["interaction"], 1.0)

    def test_structural_features_feed_into_infection_probabilities(self):
        graph = build_social_iot_graph(n_nodes=30, m=2, seed=3)
        node_features = build_node_features(graph)
        edge_features = build_edge_features(graph)

        probabilities = compute_edge_infection_probabilities(
            graph, node_features, edge_features, beta=[-2.0, 1.5, -1.0, 0.8, 0.4, 0.7, 0.3]
        )

        self.assertEqual(len(probabilities), graph.number_of_edges())
        for p in probabilities.values():
            self.assertGreater(p, 0.0)
            self.assertLess(p, 1.0)

    def test_plotting_helpers_save_png_files(self):
        graph = build_social_iot_graph(n_nodes=20, m=2, seed=3)
        with tempfile.TemporaryDirectory() as temp_dir:
            infection_path = os.path.join(temp_dir, "infection.png")
            degree_path = os.path.join(temp_dir, "degrees.png")

            infection_out = plot_infection_history([1, 3, 5, 5, 7], output_path=infection_path)
            degree_out = plot_degree_distribution(graph, output_path=degree_path)

            self.assertTrue(os.path.exists(infection_out))
            self.assertTrue(os.path.exists(degree_out))


if __name__ == "__main__":
    unittest.main()
