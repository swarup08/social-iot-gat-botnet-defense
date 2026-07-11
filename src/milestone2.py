"""Milestone 2: GAT-based node classification (benign vs compromised).

Labels come from running Milestone 1's feature-driven botnet simulator to a
fixed point and taking the resulting infected/benign split as ground truth --
this keeps the classification task genuinely coupled to the same p_uv model
used everywhere else, rather than inventing a separate labeling rule.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

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
) -> GATNodeClassifier:
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
    """
    torch.manual_seed(model_seed)  # seeds both weight init and dropout stochasticity
    model = GATNodeClassifier(in_channels=data.num_node_features)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    class_weights = compute_class_weights(data.y, train_mask, mildness=weight_mildness) if weight_mildness > 0 else None

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        # Cross-entropy loss computed ONLY on training nodes -- val/test labels
        # are never touched by backprop, only used to monitor generalization.
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask], weight=class_weights)
        loss.backward()
        optimizer.step()

    model.eval()
    return model
