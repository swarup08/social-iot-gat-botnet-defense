import os
import unittest

import torch

from src.milestone2 import (
    GATNodeClassifier,
    build_pyg_data,
    compute_class_weights,
    generate_labeled_graph,
    node_features_to_matrix,
    plot_training_curves,
    train_gat,
)


class Milestone2Tests(unittest.TestCase):
    def test_generate_labeled_graph_produces_binary_labels_for_every_node(self):
        graph, node_features, labels = generate_labeled_graph(n_nodes=40, m=2, n_rollouts=5)

        self.assertEqual(set(labels.keys()), set(graph.nodes()))
        self.assertTrue(all(label in (0, 1) for label in labels.values()))
        # Majority-vote labeling over a hub-seeded infection should give a
        # non-degenerate split, not "everyone infected" or "no one infected".
        infected_fraction = sum(labels.values()) / len(labels)
        self.assertGreater(infected_fraction, 0.0)
        self.assertLess(infected_fraction, 1.0)

    def test_node_features_to_matrix_shape_and_device_type_one_hot(self):
        graph, node_features, _ = generate_labeled_graph(n_nodes=30, m=2, n_rollouts=3)
        nodes = sorted(graph.nodes())

        matrix = node_features_to_matrix(node_features, nodes)

        # 3 continuous features (risk, hub_score, clustering) + 5 device-type one-hot dims.
        self.assertEqual(matrix.shape, (len(nodes), 8))
        one_hot_columns = matrix[:, 3:]
        self.assertTrue((one_hot_columns.sum(axis=1) == 1.0).all())

    def test_build_pyg_data_edge_index_is_bidirectional(self):
        graph, node_features, labels = generate_labeled_graph(n_nodes=30, m=2, n_rollouts=3)

        data = build_pyg_data(graph, node_features, labels)

        self.assertEqual(data.num_nodes, graph.number_of_nodes())
        self.assertEqual(data.edge_index.shape[1], 2 * graph.number_of_edges())
        self.assertTrue(data.is_undirected())

    def test_gat_forward_pass_shape_and_gradient_flow(self):
        graph, node_features, labels = generate_labeled_graph(n_nodes=30, m=2, n_rollouts=3)
        data = build_pyg_data(graph, node_features, labels)

        model = GATNodeClassifier(in_channels=data.num_node_features)
        logits = model(data.x, data.edge_index)

        self.assertEqual(logits.shape, (data.num_nodes, 2))

        loss = torch.nn.functional.cross_entropy(logits, data.y)
        loss.backward()
        # At least one parameter should have received a non-zero gradient,
        # confirming the loss is actually connected to the model's weights.
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        self.assertTrue(any(norm > 0 for norm in grad_norms))


    def test_compute_class_weights_mildness_endpoints(self):
        y = torch.tensor([0, 0, 0, 1])  # 3 benign, 1 compromised
        mask = torch.tensor([True, True, True, True])

        unweighted = compute_class_weights(y, mask, mildness=0.0)
        self.assertTrue(torch.allclose(unweighted, torch.ones(2)))

        balanced = compute_class_weights(y, mask, mildness=1.0)
        # weight_c = n_samples / (n_classes * n_samples_c) -> [4/(2*3), 4/(2*1)] = [0.667, 2.0]
        self.assertAlmostEqual(balanced[0].item(), 4 / 6, places=4)
        self.assertAlmostEqual(balanced[1].item(), 2.0, places=4)

        mild = compute_class_weights(y, mask, mildness=0.5)
        # mildness=0.5 should sit strictly between unweighted (1.0) and fully balanced.
        self.assertGreater(mild[1].item(), 1.0)
        self.assertLess(mild[1].item(), balanced[1].item())

    def test_train_gat_default_mildness_runs_and_returns_eval_model(self):
        graph, node_features, labels = generate_labeled_graph(n_nodes=30, m=2, n_rollouts=3)
        data = build_pyg_data(graph, node_features, labels)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[: data.num_nodes // 2] = True

        model = train_gat(data, train_mask, epochs=5)

        self.assertFalse(model.training)  # train_gat should leave the model in eval mode
        with torch.no_grad():
            logits = model(data.x, data.edge_index)
        self.assertEqual(logits.shape, (data.num_nodes, 2))

    def test_train_gat_log_history_is_opt_in_and_default_is_unaffected(self):
        graph, node_features, labels = generate_labeled_graph(n_nodes=30, m=2, n_rollouts=3)
        data = build_pyg_data(graph, node_features, labels)
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[:15] = True
        val_mask[15:20] = True

        # Default behavior (log_history=False) must still return just the model.
        default_result = train_gat(data, train_mask, epochs=5)
        self.assertIsInstance(default_result, GATNodeClassifier)

        model, history = train_gat(data, train_mask, val_mask=val_mask, log_history=True, epochs=5)
        self.assertIsInstance(model, GATNodeClassifier)
        for key in ("train_loss", "train_acc", "val_loss", "val_acc"):
            self.assertIn(key, history)
            self.assertEqual(len(history[key]), 5)

        path = plot_training_curves(history, output_path="test_training_curves_tmp.png")
        self.assertTrue(os.path.exists(path))
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
