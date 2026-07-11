"""Milestone 1: baseline social-IoT graph and botnet propagation simulator.

The design follows the roadmap and supervisor feedback by using:
- a scale-free topology rather than an Erdős-Rényi graph,
- a feature-driven infection probability p_uv = sigmoid(beta^T phi(x_u, x_v, e_uv)),
- a simple simulator that is easy to inspect and extend.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def build_social_iot_graph(n_nodes: int = 80, m: int = 3, seed: int = 7) -> nx.Graph:
    """Build a scale-free social-IoT graph with a hub-heavy structure.

    We use a Barabási-Albert graph because the supervisor explicitly called for
    a topology with hubs and community structure rather than a uniform random graph.
    The parameter m controls how many new edges each new node attaches to.
    """
    graph = nx.barabasi_albert_graph(n_nodes, m, seed=seed)

    # The Barabási-Albert graph already gives us a hub-heavy, scale-free structure.
    # Keeping the edge count stable makes the baseline easier to reason about and
    # avoids injecting extra randomness beyond the network generator itself.
    return graph


# The roadmap describes each node as "a physical IoT device (sensor, actuator,
# gateway), a user, or a cloud/edge service" -- these are the five categories.
DEVICE_TYPES = ("sensor", "actuator", "gateway", "user", "service")

# Percentile cutoffs (by degree centrality, ascending) used to assign device
# types structurally: real Social IoT deployments have a few high-degree
# gateway hubs, moderately-connected cloud/edge services, mid-degree user
# devices, and a long tail of low-degree sensor/actuator leaf nodes. This
# mirrors the Barabási-Albert graph's own power-law degree distribution.
_GATEWAY_PERCENTILE = 0.90  # top 10% most-connected nodes become gateways
_SERVICE_PERCENTILE = 0.75  # next 15% become cloud/edge services
_USER_PERCENTILE = 0.50  # next 25% become user devices
# the remaining bottom 50% (leaf-like nodes) split into sensors/actuators

# Baseline infection risk per device type (roadmap's "static properties:
# device type"): cheap, rarely-patched embedded hardware (sensors/actuators)
# is more vulnerable than centrally-managed gateways or heavily-monitored
# cloud/edge services; user devices (phones/laptops) sit in between.
_DEVICE_TYPE_BASE_RISK = {
    "sensor": 0.55,
    "actuator": 0.55,
    "user": 0.45,
    "gateway": 0.35,
    "service": 0.30,
}
# Weight of the topological (clustering-coefficient) term added on top of the
# device-type baseline when computing a node's risk feature.
_CLUSTERING_RISK_WEIGHT = 0.3


def assign_device_types(graph: nx.Graph) -> Dict[int, str]:
    """Assign each node one of the roadmap's device/user/service types.

    Types are derived from degree centrality rather than an arbitrary rule
    (e.g. node id modulo): the highest-degree nodes become gateways (real
    gateways aggregate many devices), the next band becomes cloud/edge
    services, the next becomes user devices, and the low-degree tail -- which
    has no graph-theoretic signal to distinguish the two -- alternates
    deterministically between sensor and actuator by node id.
    """
    centrality = nx.degree_centrality(graph)  # real per-node degree, normalized to [0, 1]
    # Sort ascending by degree centrality so percentile cutoffs are simple index math.
    ranked_nodes = sorted(graph.nodes(), key=lambda node: centrality[node])
    n_nodes = len(ranked_nodes)

    device_types: Dict[int, str] = {}
    for rank, node in enumerate(ranked_nodes):
        # 0.0 = lowest-degree node, 1.0 = highest-degree node.
        percentile = rank / max(n_nodes - 1, 1)
        if percentile >= _GATEWAY_PERCENTILE:
            device_types[node] = "gateway"
        elif percentile >= _SERVICE_PERCENTILE:
            device_types[node] = "service"
        elif percentile >= _USER_PERCENTILE:
            device_types[node] = "user"
        else:
            device_types[node] = "sensor" if node % 2 == 0 else "actuator"
    return device_types


def build_node_features(graph: nx.Graph) -> Dict[int, Dict[str, float]]:
    """Derive per-node features from real graph structure and device type.

    Populates the keys compute_edge_infection_probabilities reads ("risk",
    "hub_score", "community"), replacing the old node-id-modulo placeholders:
    - hub_score: degree centrality -- the roadmap's "degree ... centrality
      scores" topological feature.
    - risk: a device-type baseline (roadmap's "static properties: device
      type") plus a clustering-coefficient term (roadmap's "clustering
      coefficient" topological feature). Lower clustering means a node
      bridges otherwise-separate neighborhoods (a structural hole), so it
      contributes more to risk.
    - community: a real detected community id from modularity maximization,
      not node % 3.
    Also attaches "device_type" and "clustering" for realism/reporting, even
    though the current infection model's feature vector doesn't read them.
    """
    device_types = assign_device_types(graph)
    hub_scores = nx.degree_centrality(graph)  # real degree, normalized to [0, 1]
    clustering = nx.clustering(graph)  # real local clustering coefficient per node

    # Real community detection (modularity maximization) instead of node % 3.
    communities = nx.community.greedy_modularity_communities(graph)
    community_id: Dict[int, int] = {}
    for community_index, members in enumerate(communities):
        for node in members:
            community_id[node] = community_index

    node_features: Dict[int, Dict[str, float]] = {}
    for node in graph.nodes():
        device_type = device_types[node]
        base_risk = _DEVICE_TYPE_BASE_RISK[device_type]
        structural_risk = _CLUSTERING_RISK_WEIGHT * (1.0 - clustering[node])
        risk = min(1.0, base_risk + structural_risk)  # clip to keep risk in [0, 1]

        node_features[node] = {
            "risk": risk,
            "hub_score": hub_scores[node],
            "community": community_id[node],
            "device_type": device_type,
            "clustering": clustering[node],
        }
    return node_features


def build_edge_features(graph: nx.Graph) -> Dict[Tuple[int, int], Dict[str, float]]:
    """Derive per-edge features from real graph structure.

    Populates "interaction" -- the key compute_edge_infection_probabilities
    reads -- from normalized edge betweenness centrality, replacing the old
    (u + v) % 5 placeholder: edges that sit on many shortest paths carry more
    communication traffic between otherwise-distant parts of the network, so
    higher betweenness is used as a structural proxy for communication
    frequency/importance. (Checked empirically against Jaccard neighborhood
    overlap, which is degenerate here -- zero on ~30% of edges on this graph's
    typical low clustering -- so betweenness was used instead.)
    """
    betweenness = nx.edge_betweenness_centrality(graph, normalized=True)
    edge_features: Dict[Tuple[int, int], Dict[str, float]] = {}
    for edge, value in betweenness.items():
        edge_features[edge] = {"interaction": value}
    return edge_features


def _feature_vector(node_features: Dict[int, Dict[str, float]], edge_features: Dict[Tuple[int, int], Dict[str, float]], u: int, v: int) -> List[float]:
    """Create the feature vector phi(x_u, x_v, e_uv) used by the infection model.

    The vector combines node-side risk and hubness terms with edge-level interaction.
    The exact ordering must match the beta vector supplied to compute_edge_infection_probabilities.
    """
    node_u = node_features[u]
    node_v = node_features[v]
    edge_uv = edge_features.get((u, v), edge_features.get((v, u), {}))

    return [
        1.0,  # bias term for the sigmoid linear predictor
        node_u.get("risk", 0.0),
        node_v.get("risk", 0.0),
        node_u.get("hub_score", 0.0),
        node_v.get("hub_score", 0.0),
        edge_uv.get("interaction", 0.0),
        float(node_u.get("community", 0) != node_v.get("community", 0)),
    ]


def compute_edge_infection_probabilities(
    graph: nx.Graph,
    node_features: Dict[int, Dict[str, float]],
    edge_features: Dict[Tuple[int, int], Dict[str, float]],
    beta: List[float],
) -> Dict[Tuple[int, int], float]:
    """Compute a feature-driven infection probability for every edge.

    This implements the roadmap formula p_uv = sigmoid(beta^T phi(x_u, x_v, e_uv)).
    The use of features is important because the supervisor explicitly noted that
    a constant edge probability was the root cause of the earlier disconnected model.
    """
    probabilities: Dict[Tuple[int, int], float] = {}
    for u, v in graph.edges():
        phi = _feature_vector(node_features, edge_features, u, v)
        linear_score = float(np.dot(np.array(phi, dtype=float), np.array(beta, dtype=float)))
        probability = 1.0 / (1.0 + math.exp(-linear_score))
        probabilities[(u, v)] = probability
    return probabilities


def plot_infection_history(infection_history: List[int], output_path: str = "infection_history.png", use_fraction: bool = False) -> str:
    """Plot infected-node count or fraction versus time step.

    This gives a simple visual of how quickly the outbreak grows over time.
    The function saves a PNG so the result can be inspected or included in a report.
    """
    steps = list(range(len(infection_history)))
    values = infection_history
    if use_fraction:
        if not infection_history:
            values = []
        else:
            total_nodes = max(infection_history) if infection_history else 1
            values = [count / max(total_nodes, 1) for count in infection_history]

    plt.figure(figsize=(6, 4))
    plt.plot(steps, values, marker="o", linewidth=1.8)
    plt.xlabel("time step")
    plt.ylabel("infected nodes" if not use_fraction else "infected fraction")
    plt.title("Botnet infection history")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_infection_curves_comparison(curves: Dict[str, List[float]], total_nodes: int, output_path: str = "infection_curves_comparison.png") -> str:
    """Overlay several named infection curves (infected count vs. round) on one
    plot, as infected FRACTION so different methods' curves are directly
    comparable against the same 0-1 y-axis regardless of how many edges each
    one removed. Each curve is plotted out to its own last round; the roadmap's
    "compare infection curves for no pruning vs. static vs. RL pruning" figure.
    """
    plt.figure(figsize=(7, 4.5))
    for label, counts in curves.items():
        steps = list(range(len(counts)))
        fractions = [c / total_nodes for c in counts]
        plt.plot(steps, fractions, marker="o", markersize=3, linewidth=1.8, label=label)
    plt.xlabel("time step (round)")
    plt.ylabel("infected fraction")
    plt.title("Botnet infection curves: no pruning vs. static vs. RL pruning")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_degree_distribution(graph: nx.Graph, output_path: str = "degree_distribution.png") -> str:
    """Plot the degree distribution of the graph.

    Milestone 1 includes the degree distribution as a simple structural sanity check,
    complementing the hub-heavy Barabási-Albert topology used for the simulator.
    """
    degrees = sorted((graph.degree(node) for node in graph.nodes()), reverse=True)

    plt.figure(figsize=(6, 4))
    plt.hist(degrees, bins=max(5, min(15, len(degrees) // 2)), color="steelblue", edgecolor="black")
    plt.xlabel("degree")
    plt.ylabel("count")
    plt.title("Degree distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def simulate_botnet(
    graph: nx.Graph,
    edge_probabilities: Dict[Tuple[int, int], float],
    initial_compromised: set[int],
    seed: int = 7,
) -> Dict[str, object]:
    """Simulate a synchronous, discrete-time botnet outbreak over the graph.

    Implements the roadmap's z_v(t) notion of discrete time steps: at each round
    t, every node infected as of round t-1 (the current "frontier") independently
    attempts to infect each of its still-uninfected neighbors via that edge's
    p_uv. All of a round's Bernoulli trials are checked against the infected set
    as it stood at the START of the round, so simultaneous infections from
    different frontier nodes land in the SAME history entry instead of being
    spread across several node-by-node entries. This makes infection_history
    track genuine discrete time steps (comparable across graphs/methods), and
    gives Milestone 3's RL agent a natural point to prune edges between rounds.

    Both frontier AND each node's neighbors are iterated in SORTED order, so
    the sequence of rng.random() draws -- and therefore the outcome for a
    given seed -- depends only on (graph edges, edge_probabilities, seed,
    initial_compromised), never on incidental object-construction history.
    Without the neighbor sort, nx.Graph.copy() can silently reorder a node's
    adjacency dict (same edge set, different iteration order), which pairs
    the same rng draws with different neighbors and changes the outcome for
    an identical seed -- found while building Milestone 3's environment; see
    NOTES.md for the affected-scope writeup.
    """
    rng = random.Random(seed)
    infected_nodes = set(initial_compromised)
    infection_history: List[int] = [len(infected_nodes)]

    # `frontier` holds only the nodes infected in the PREVIOUS round; only they
    # attempt new infections this round. Already-infected nodes from earlier
    # rounds never re-attempt, since each edge is tried exactly once, at the
    # round when its infected endpoint first joins the frontier.
    frontier = set(initial_compromised)
    while frontier:
        next_frontier: set[int] = set()  # nodes newly infected THIS round
        # Sort for a deterministic RNG draw order, so a fixed seed always
        # reproduces the same rollout regardless of Python's set iteration order.
        for current in sorted(frontier):
            for neighbor in sorted(graph.neighbors(current)):
                # Compare against infected_nodes (start-of-round state), not
                # next_frontier, so every incoming edge still gets its own
                # independent Bernoulli trial even if another frontier node
                # already recruited this neighbor earlier in the same round.
                if neighbor in infected_nodes:
                    continue
                edge_key = (current, neighbor) if (current, neighbor) in edge_probabilities else (neighbor, current)
                p_uv = edge_probabilities[edge_key]
                if rng.random() < p_uv:
                    next_frontier.add(neighbor)
        # Commit the whole round's newly-infected nodes at once, so they land
        # together in a single infection_history entry (one real time step).
        infected_nodes.update(next_frontier)
        infection_history.append(len(infected_nodes))
        frontier = next_frontier

    return {
        "infected_nodes": infected_nodes,
        "infection_history": infection_history,
    }
