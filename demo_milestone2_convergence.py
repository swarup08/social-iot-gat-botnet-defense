"""Milestone 2 demo: training/validation loss and accuracy convergence plot.

Satisfies the roadmap's "log training and validation loss versus epochs"
task, using train_gat's optional log_history flag (default off, so every
other call site's behavior is unchanged). This is a single representative
run, not a statistical sweep -- the point is to look at *how* one training
run converges (smoothly vs. noisily, over/underfitting), not to compare
methods.
"""

from src.milestone2 import build_pyg_data, generate_labeled_graph, make_node_split, plot_training_curves, train_gat


def main() -> None:
    graph, node_features, labels = generate_labeled_graph(n_nodes=300)
    data = build_pyg_data(graph, node_features, labels)
    train_mask, val_mask, test_mask = make_node_split(data.num_nodes, seed=0)

    model, history = train_gat(data, train_mask, val_mask=val_mask, log_history=True)

    print(f"epochs logged: {len(history['train_loss'])}")
    print(f"final: train_loss={history['train_loss'][-1]:.4f}  val_loss={history['val_loss'][-1]:.4f}")
    print(f"final: train_acc={history['train_acc'][-1]:.4f}  val_acc={history['val_acc'][-1]:.4f}")
    print(f"best val_loss: {min(history['val_loss']):.4f} at epoch {history['val_loss'].index(min(history['val_loss'])) + 1}")

    path = plot_training_curves(history)
    print(f"saved convergence plot to {path}")


if __name__ == "__main__":
    main()
