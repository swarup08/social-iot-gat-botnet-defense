import os
import tempfile
import unittest

import networkx as nx

from src.milestone1 import (
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
