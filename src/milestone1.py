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
            for neighbor in graph.neighbors(current):
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
