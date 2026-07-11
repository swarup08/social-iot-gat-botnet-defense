"""Milestone 2: GAT-based node classification (benign vs compromised).

Labels come from running Milestone 1's feature-driven botnet simulator to a
fixed point and taking the resulting infected/benign split as ground truth --
this keeps the classification task genuinely coupled to the same p_uv model
used everywhere else, rather than inventing a separate labeling rule.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from src.milestone1 import (
    DEVICE_TYPES,
    build_edge_features,
    build_node_features,
    build_social_iot_graph,
    compute_edge_infection_probabilities,
    simulate_botnet,
)

# Same beta used in Milestone 1's demo, so p_uv stays comparable across milestones.
DEFAULT_BETA: List[float] = [-2.0, 1.5, -1.0, 0.8, 0.4, 0.7, 0.3]


def generate_labeled_graph(
    n_nodes: int = 80,
    m: int = 3,
    graph_seed: int = 7,
    infection_seed: int = 7,
    beta: List[float] = DEFAULT_BETA,
    n_rollouts: int = 15,
) -> Tuple[nx.Graph, Dict[int, Dict[str, float]], Dict[int, int]]:
    """Build a graph, run repeated botnet rollouts, and majority-vote labels.

    The infection is seeded from the single highest-degree hub node: a
    compromised gateway/hub is both a realistic worst-case starting point and,
    empirically on these graphs, produces a roughly balanced infected/benign
    split (see the Milestone 1 demo), which matters for training a classifier
    -- a near-all-benign snapshot would give a trivial majority-class baseline.

    A SINGLE rollout is a noisy label: simulate_botnet's edge draws are random,
    so two rollouts on the same graph can disagree about whether a given node
    ends up infected, even though the underlying p_uv model never changed. This
    is the same noise-swamps-signal failure mode already documented for the RL
    reward in this project's notes, just showing up in labels instead -- so we
    apply the same fix: run n_rollouts independent rollouts from the same hub
    seed (different infection_seed each time) and label a node "compromised"
    only if it was infected in a MAJORITY of them. simulate_botnet and the
    p_uv model itself are unchanged; only this aggregation step is new.
    """
    graph = build_social_iot_graph(n_nodes=n_nodes, m=m, seed=graph_seed)
    node_features = build_node_features(graph)
    edge_features = build_edge_features(graph)
    probabilities = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=beta)

    degrees = dict(graph.degree())
    hub_node = max(degrees, key=degrees.get)

    infection_counts = {node: 0 for node in graph.nodes()}
    for rollout_index in range(n_rollouts):
        # A distinct, deterministic seed per rollout keeps the whole labeling
        # process reproducible given the same (graph_seed, infection_seed).
        rollout_seed = infection_seed + rollout_index
        result = simulate_botnet(graph, probabilities, initial_compromised={hub_node}, seed=rollout_seed)
        for node in result["infected_nodes"]:
            infection_counts[node] += 1

    # label 1 = compromised in a majority of rollouts, 0 = benign in a majority.
    labels = {node: int(infection_counts[node] > n_rollouts / 2) for node in graph.nodes()}
    return graph, node_features, labels


def node_features_to_matrix(node_features: Dict[int, Dict[str, float]], nodes: List[int]) -> np.ndarray:
    """Convert per-node feature dicts into a dense [n_nodes, n_features] matrix.

    Only numeric/structural fields become GAT input: risk, hub_score, and
    clustering coefficient (all continuous), plus a one-hot encoding of
    device_type (5 dims, the roadmap's device/user/service categories). Raw
    community ids are deliberately left out -- they're arbitrary integer labels
    assigned by modularity detection, not meaningfully comparable as a single
    scalar, and the GAT can recover community structure itself through message
    passing over the graph's actual edges.
    """
    rows = []
    for node in nodes:
        features = node_features[node]
        device_type_one_hot = [1.0 if features["device_type"] == dtype else 0.0 for dtype in DEVICE_TYPES]
        rows.append([features["risk"], features["hub_score"], features["clustering"], *device_type_one_hot])
    return np.array(rows, dtype=np.float32)


def build_pyg_data(graph: nx.Graph, node_features: Dict[int, Dict[str, float]], labels: Dict[int, int]) -> Data:
    """Assemble a PyTorch Geometric Data object (x, edge_index, y) for the GAT.

    Node row i corresponds to node id i, since Barabasi-Albert graphs number
    nodes 0..n-1 in insertion order and we iterate sorted(graph.nodes()).
    edge_index lists both (u, v) and (v, u) for every undirected edge, because
    GATConv treats edge_index as directed message-passing edges -- without the
    reverse direction, information would only ever flow one way along each edge.
    """
    nodes = sorted(graph.nodes())
    x = torch.tensor(node_features_to_matrix(node_features, nodes), dtype=torch.float32)
    y = torch.tensor([labels[node] for node in nodes], dtype=torch.long)

    edges = list(graph.edges())
    source = [u for u, v in edges] + [v for u, v in edges]
    target = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([source, target], dtype=torch.long)

    return Data(x=x, edge_index=edge_index, y=y)


class GATNodeClassifier(nn.Module):
    """A small 2-layer multi-head GAT for benign (0) vs compromised (1) nodes.

    Each GATConv layer computes, per edge (u, v): e_uv = LeakyReLU(a^T [W x_u
    || W x_v]), normalizes it into an attention weight alpha_uv = softmax_v
    (e_uv) over u's neighborhood, then aggregates h_u' = sum_v alpha_uv * W x_v.
    LeakyReLU (not ReLU) is used because raw attention scores are as likely to
    be negative as positive, and ReLU would zero-and-kill the gradient of any
    negative score permanently; LeakyReLU keeps a small slope so those scores
    can still be corrected during training. The softmax is taken over each
    node's own neighborhood (not globally) so attention weights sum to 1 per
    node -- this both makes them interpretable as "how much of my update comes
    from each neighbor" and normalizes away raw node degree, so a 20-neighbor
    hub and a 2-neighbor leaf produce comparably-scaled updates.
    """

    def __init__(self, in_channels: int, hidden_channels: int = 8, heads: int = 4, dropout: float = 0.3):
        super().__init__()
        self.dropout = dropout
        # Layer 1: `heads` independent attention mechanisms, each producing
        # hidden_channels features, concatenated (default concat=True) into a
        # hidden_channels * heads-wide representation. Multiple heads let the
        # model learn several different notions of "relevant neighbor"
        # simultaneously -- e.g. one head might key on risk similarity, another
        # on hub_score/degree -- the same motivation as multi-head attention
        # in Transformers.
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        # Layer 2: a single averaged head (concat=False) collapsing down to 2
        # output logits (benign vs compromised) -- we want one final score per
        # class here, not a heads-times-wider representation.
        self.gat2 = GATConv(hidden_channels * heads, 2, heads=1, concat=False, dropout=dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Dropout on the input features themselves (as in the original GAT
        # paper), not just between layers, to discourage over-reliance on any
        # single input feature given how small this feature vector is.
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)  # ELU between GAT layers, matching the original GAT paper's choice
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x  # raw logits; combine with CrossEntropyLoss / softmax outside


def accuracy(predictions: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:
    return (predictions[mask] == y[mask]).float().mean().item()


def precision_recall_f1_counts(predictions: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, target_class: int) -> Dict[str, float]:
    """Precision/recall/F1 AND the raw confusion counts they're built from.

    The raw counts (true_positive / total_positive / predicted_positive)
    matter on their own: with single-to-low-double-digit positive counts per
    split, "recall=0.5" could mean "1 of 2" or "10 of 20" -- very different
    confidence in the number -- so we keep the counts alongside the rates.
    """
    predicted = predictions[mask]
    actual = y[mask]
    true_positive = int(((predicted == target_class) & (actual == target_class)).sum())
    total_positive = int((actual == target_class).sum())
    predicted_positive = int((predicted == target_class).sum())

    precision = true_positive / predicted_positive if predicted_positive > 0 else 0.0
    recall = true_positive / total_positive if total_positive > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "total_positive": total_positive,
        "predicted_positive": predicted_positive,
    }


def compute_class_weights(y: torch.Tensor, mask: torch.Tensor, n_classes: int = 2, mildness: float = 1.0) -> torch.Tensor:
    """Per-class loss weights, interpolated between unweighted and fully balanced.

    weight_c = (n_samples / (n_classes * n_samples_c)) ^ mildness, computed
    from the TRAINING split's labels only (never test/val). mildness=0 gives
    all-ones weights (unweighted, identical to plain cross-entropy); mildness=1
    gives the standard fully-balanced weight; mildness=0.5 (the geometric mean
    of the two, i.e. sqrt of the balanced weight) is train_gat's default -- see
    its docstring for why.
    """
    train_labels = y[mask]
    n_samples = train_labels.numel()
    counts = torch.bincount(train_labels, minlength=n_classes).float().clamp(min=1.0)
    balanced_weights = n_samples / (n_classes * counts)
    return balanced_weights**mildness


def train_gat(
    data: Data,
    train_mask: torch.Tensor,
    model_seed: int = 0,
    weight_mildness: float = 0.5,
    epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    val_mask: torch.Tensor = None,
    log_history: bool = False,
):
    """Train a GATNodeClassifier on `data`, using only `train_mask` for the loss.

    weight_mildness defaults to 0.5, chosen from a three-way comparison run in
    demo_milestone2.py across 5 random train/val/test splits x 4 model-init
    seeds each, on this project's labeled Social IoT graph (n=300 nodes,
    infected_fraction ~0.20, labels from generate_labeled_graph's majority-vote
    rollouts):

      mildness=0.0 (unweighted)     : test accuracy 0.840 +/- 0.045, beats the
                                       majority-class baseline significantly
                                       (paired t-test p=0.003) -- but recall on
                                       the compromised class is only 0.178: the
                                       model is systematically, reproducibly
                                       biased toward predicting "benign".
      mildness=1.0 (fully balanced) : recall improves to 0.624, but accuracy
                                       drops below baseline (not significant,
                                       p=0.158), and one of five splits showed
                                       genuine training-convergence instability
                                       -- recall swinging from 0.43 to 1.00
                                       across model seeds trained on the exact
                                       same data split.
      mildness=0.5 (chosen default) : matches mildness=0.0's accuracy and
                                       significance (0.845 +/- 0.054, p=0.002)
                                       while more than doubling recall (0.380)
                                       and F1 (0.465) relative to unweighted,
                                       with no split flagged for convergence
                                       instability.

    This is still a real, standing limitation, not a solved problem: even at
    mildness=0.5, recall of ~38% means the model misses well over half the
    actual compromised nodes on average. See NOTES.md and demo_milestone2.py
    for the full per-split breakdown and methodology.

    If log_history=True, additionally returns a per-epoch history dict with
    "train_loss"/"train_acc" lists (and "val_loss"/"val_acc" too, if val_mask
    is given) -- one entry per epoch, satisfying the roadmap's "log training
    and validation loss versus epochs" task. This is purely additive and
    opt-in: log_history defaults to False, so every existing call site is
    unaffected and still gets just the trained model back.
    """
    torch.manual_seed(model_seed)  # seeds both weight init and dropout stochasticity
    model = GATNodeClassifier(in_channels=data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = compute_class_weights(data.y, train_mask, mildness=weight_mildness) if weight_mildness > 0 else None

    history = None
    if log_history:
        history = {"train_loss": [], "train_acc": []}
        if val_mask is not None:
            history["val_loss"] = []
            history["val_acc"] = []

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        # Cross-entropy loss computed ONLY on training nodes -- val/test labels
        # are never touched by backprop, only used to monitor generalization.
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

        if log_history:
            # A separate eval-mode forward pass (dropout off) for logging, so
            # the recorded loss/accuracy reflect the model's actual inference
            # behavior rather than the noisier dropout-affected training pass.
            model.eval()
            with torch.no_grad():
                eval_logits = model(data.x, data.edge_index)
                predictions = eval_logits.argmax(dim=1)
                train_loss = F.cross_entropy(eval_logits[train_mask], data.y[train_mask], weight=class_weights).item()
                history["train_loss"].append(train_loss)
                history["train_acc"].append(accuracy(predictions, data.y, train_mask))
                if val_mask is not None:
                    val_loss = F.cross_entropy(eval_logits[val_mask], data.y[val_mask], weight=class_weights).item()
                    history["val_loss"].append(val_loss)
                    history["val_acc"].append(accuracy(predictions, data.y, val_mask))

    model.eval()
    if log_history:
        return model, history
    return model


def _extract_directed_attention(model: GATNodeClassifier, data: Data) -> torch.Tensor:
    """Extract per-directed-edge attention, averaged over heads and the 2 layers.

    GATConv.forward() normally returns only updated node features -- the
    attention coefficients it computes internally are discarded. Passing
    return_attention_weights=True makes it ADDITIONALLY return
    (edge_index_used, alpha), where alpha has shape [num_edges_used, heads]
    and alpha[k, h] is exactly the softmax-normalized attention coefficient
    (the alpha_ij from the GAT paper, i = target, j = source) head h assigns
    to the k-th (source, target) pair in edge_index_used. edge_index_used is
    LONGER than the edge_index we pass in, because GATConv adds a self-loop
    (i, i) for every node by default (add_self_loops=True) before computing
    attention -- so we can't reuse our input edge_index to index alpha
    without accounting for that, and instead check (assert) that GATConv
    preserves our original edges, in order, at the front of edge_index_used
    before slicing them out.

    Returns a 1-D tensor of length data.edge_index.shape[1] (one score per
    INPUT directed edge, in the same order as data.edge_index, self-loops
    already excluded): the mean over heads within each layer, then the mean
    of the two layers together, per the roadmap's "average ... over heads and
    layers." Callers combine directions (u->v and v->u) themselves, since raw
    vs. degree-corrected aggregation need that split differently.
    """
    model.eval()
    n_directed = data.edge_index.shape[1]  # == 2 * n_undirected_edges, by build_pyg_data's construction

    with torch.no_grad():
        # Replicates GATNodeClassifier.forward() exactly (dropout is a no-op in
        # eval mode), but requesting attention weights at each layer.
        hidden, (edge_index_1, alpha1) = model.gat1(data.x, data.edge_index, return_attention_weights=True)
        hidden = F.elu(hidden)
        _, (edge_index_2, alpha2) = model.gat2(hidden, data.edge_index, return_attention_weights=True)

    # We rely on GATConv preserving our input edge order at the FRONT of its
    # returned edge_index and appending self-loops after; verify that instead
    # of silently trusting an internal-implementation assumption.
    assert torch.equal(edge_index_1[:, :n_directed], data.edge_index)
    assert torch.equal(edge_index_2[:, :n_directed], data.edge_index)

    layer1_score = alpha1[:n_directed].mean(dim=1)  # mean over layer 1's 4 heads
    layer2_score = alpha2[:n_directed].mean(dim=1)  # mean over layer 2's 1 head
    return (layer1_score + layer2_score) / 2.0  # mean over the 2 layers


def extract_edge_attention_scores(model: GATNodeClassifier, data: Data) -> Dict[Tuple[int, int], float]:
    """Extract one RAW attention-based importance score s_uv per undirected edge.

    Averages the two directions (u->v and v->u) of each undirected edge
    together, since GAT attention is directional (softmax is taken over each
    TARGET node's own neighborhood) but our graph and downstream pruning are
    undirected.

    CAVEAT (see NOTES.md): this raw score is confounded by degree -- softmax
    normalization mechanically hands a smaller average share to each neighbor
    of a high-degree (hub) node, regardless of how "important" that neighbor
    actually is, which shows up empirically as a strong NEGATIVE correlation
    between s_uv and hub_score/betweenness/p_uv. Use
    extract_degree_corrected_attention_scores for pruning decisions instead.
    """
    directed_score = _extract_directed_attention(model, data)
    n_edges = data.edge_index.shape[1] // 2

    # build_pyg_data lays out edge_index as [forward edges (u_i -> v_i) for all
    # i][backward edges (v_i -> u_i) for all i], so these two halves line up
    # positionally with each other and with graph.edges() order.
    forward_score = directed_score[:n_edges]  # v_i attending to u_i
    backward_score = directed_score[n_edges : 2 * n_edges]  # u_i attending to v_i

    edges = list(zip(data.edge_index[0, :n_edges].tolist(), data.edge_index[1, :n_edges].tolist()))
    return {(u, v): float((forward_score[i] + backward_score[i]) / 2.0) for i, (u, v) in enumerate(edges)}


def extract_degree_corrected_attention_scores(model: GATNodeClassifier, data: Data, graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """Extract a degree-corrected attention score s_uv per undirected edge.

    For a directed edge (source=j, target=i), alpha_ij is normalized by
    softmax over ALL of i's neighbors (i.e. sum_j alpha_ij = 1 across
    deg(i)-ish terms) -- so a high-degree target mechanically hands out a
    smaller average share to each neighbor, independent of whether that
    neighbor's raw (pre-softmax) signal was actually strong. Multiplying
    alpha_ij by deg(i) -- the size of i's neighborhood -- cancels that
    mechanical dilution: the result is ~1 if j gets exactly a uniform share
    of i's attention, >1 if j is favored above uniform, <1 if disfavored.
    That's the model's learned RELATIVE preference, with the "more competing
    neighbors -> smaller raw share" artifact removed.

    (This uses the graph's real degree, not degree+1, so it slightly
    under-corrects relative to GATConv's actual softmax denominator, which
    also includes the self-loop it adds internally -- a minor approximation,
    not expected to change the qualitative correlation picture.)

    The two directions of each undirected edge are corrected independently
    (by their own target's degree) before being averaged together, since
    u->v and v->u are diluted by different nodes' neighborhood sizes.
    """
    directed_score = _extract_directed_attention(model, data)
    n_edges = data.edge_index.shape[1] // 2
    degree = dict(graph.degree())

    targets = data.edge_index[1, : 2 * n_edges].tolist()
    corrected = torch.tensor([directed_score[i].item() * degree[target] for i, target in enumerate(targets)])

    forward_score = corrected[:n_edges]  # (v_i attending to u_i) * deg(v_i)
    backward_score = corrected[n_edges : 2 * n_edges]  # (u_i attending to v_i) * deg(u_i)

    edges = list(zip(data.edge_index[0, :n_edges].tolist(), data.edge_index[1, :n_edges].tolist()))
    return {(u, v): float((forward_score[i] + backward_score[i]) / 2.0) for i, (u, v) in enumerate(edges)}


def align_edge_metrics(scores: Dict[Tuple[int, int], float], metric: Dict[Tuple[int, int], float]) -> Tuple[np.ndarray, np.ndarray]:
    """Pair up two per-edge dicts into aligned arrays for correlation/plotting.

    `metric` may key an edge as either (u, v) or (v, u) -- Milestone 1's
    feature dicts and this module's attention scores aren't guaranteed to
    agree on direction for a given undirected edge, so both are checked.
    """
    s_values, m_values = [], []
    for (u, v), s in scores.items():
        m = metric.get((u, v), metric.get((v, u)))
        s_values.append(s)
        m_values.append(m)
    return np.array(s_values, dtype=np.float64), np.array(m_values, dtype=np.float64)


def plot_attention_score_distribution(scores: Dict[Tuple[int, int], float], output_path: str = "attention_score_distribution.png") -> str:
    """Histogram of the aggregated attention score s_uv across all edges."""
    values = list(scores.values())

    plt.figure(figsize=(6, 4))
    plt.hist(values, bins=30, color="steelblue", edgecolor="black")
    plt.xlabel("attention-based edge importance score s_uv")
    plt.ylabel("count")
    plt.title("Distribution of GAT attention edge scores")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_attention_vs_metric(s_values: np.ndarray, metric_values: np.ndarray, metric_name: str, output_path: str) -> str:
    """Scatter s_uv against another per-edge metric, with Pearson r in the title."""
    pearson_r = float(np.corrcoef(s_values, metric_values)[0, 1])

    plt.figure(figsize=(6, 4))
    plt.scatter(metric_values, s_values, alpha=0.5, color="steelblue", s=18, edgecolor="none")
    plt.xlabel(metric_name)
    plt.ylabel("attention score s_uv")
    plt.title(f"s_uv vs {metric_name} (Pearson r={pearson_r:.3f})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_training_curves(history: Dict[str, List[float]], output_path: str = "training_curves.png") -> str:
    """Plot training/validation loss and accuracy vs. epoch, side by side.

    `history` is the dict returned by train_gat(..., log_history=True):
    "train_loss"/"train_acc" always present, "val_loss"/"val_acc" present if
    a val_mask was given. Satisfies the roadmap's "log training and
    validation loss versus epochs" task -- this is the convergence diagnostic
    that the split/seed-averaged accuracy numbers elsewhere don't show: how
    smoothly (or not) a single training run actually converges.
    """
    epochs = list(range(1, len(history["train_loss"]) + 1))
    has_val = "val_loss" in history

    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4.5))

    loss_ax.plot(epochs, history["train_loss"], color="steelblue", linewidth=1.8, label="train")
    if has_val:
        loss_ax.plot(epochs, history["val_loss"], color="darkorange", linewidth=1.8, label="validation")
    loss_ax.set_xlabel("epoch")
    loss_ax.set_ylabel("cross-entropy loss")
    loss_ax.set_title("Loss vs. epoch")
    loss_ax.grid(True, alpha=0.3)
    loss_ax.legend()

    acc_ax.plot(epochs, history["train_acc"], color="steelblue", linewidth=1.8, label="train")
    if has_val:
        acc_ax.plot(epochs, history["val_acc"], color="darkorange", linewidth=1.8, label="validation")
    acc_ax.set_xlabel("epoch")
    acc_ax.set_ylabel("accuracy")
    acc_ax.set_title("Accuracy vs. epoch")
    acc_ax.set_ylim(0, 1)
    acc_ax.grid(True, alpha=0.3)
    acc_ax.legend()

    fig.suptitle("GAT training convergence (representative run)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
