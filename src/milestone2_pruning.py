"""Milestone 2: static edge pruning strategies and their security/utility trade-off.

Roadmap: "Implement static pruning strategies: threshold-based pruning
(remove edges with s_uv below a chosen percentile), top-k pruning per node
(retain only the k most important edges in each node's neighborhood)" --
implemented here alongside the supervisor-mandated baseline suite (random,
degree-centrality, betweenness-centrality, highest-p_uv removal), not bolted
on afterward.

IMPORTANT directional note, easy to misread: the GAT-based methods (threshold,
top-k) RETAIN high-scoring edges and REMOVE low-scoring ones, per the
roadmap's literal wording -- high attention/relative-preference is treated as
"important, keep it." The baseline heuristics (degree, betweenness,
highest-p_uv) instead REMOVE the highest-scoring edges directly, per the
supervisor's containment framing ("just remove the highest-p_uv edges
directly") -- high centrality/danger is treated as "risky, cut it." These are
opposite removal directions by design, not an inconsistency -- see NOTES.md.

Also by design (see NOTES.md): node features (risk, hub_score, device_type,
clustering, community) and the p_uv values used for the security measurement
are computed ONCE on the ORIGINAL, unpruned graph and held fixed across all
methods/levels. Only the graph's edge connectivity changes between methods --
this isolates "fewer transmission paths / less structure for the GAT" as the
pruning effect, rather than conflating it with recomputed risk/structural
features on each pruned topology.
"""

from __future__ import annotations

import random
import statistics
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.milestone1 import simulate_botnet
from src.milestone2 import accuracy, build_pyg_data, precision_recall_f1_counts, train_gat


def score_edges_by_avg_hub_score(graph: nx.Graph, node_features: Dict[int, Dict[str, float]]) -> Dict[Tuple[int, int], float]:
    """Degree-centrality edge score: average of the two endpoints' hub_score."""
    return {(u, v): (node_features[u]["hub_score"] + node_features[v]["hub_score"]) / 2 for u, v in graph.edges()}


def score_edges_by_betweenness(graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """Edge betweenness centrality, recomputed fresh as a ranking criterion."""
    return dict(nx.edge_betweenness_centrality(graph, normalized=True))


def _get_score(scores: Dict[Tuple[int, int], float], u: int, v: int) -> float:
    return scores.get((u, v), scores.get((v, u)))


def _rank_edges_ascending(graph: nx.Graph, scores: Dict[Tuple[int, int], float]) -> List[Tuple[int, int]]:
    return sorted(graph.edges(), key=lambda e: _get_score(scores, *e))


def prune_lowest_score(graph: nx.Graph, scores: Dict[Tuple[int, int], float], remove_fraction: float) -> nx.Graph:
    """Remove the remove_fraction of edges with the LOWEST score.

    Threshold-style pruning that RETAINS high-scoring edges -- used for
    GAT-based scores, per the roadmap's literal "remove edges with s_uv below
    a chosen percentile."
    """
    ranked = _rank_edges_ascending(graph, scores)
    n_remove = round(remove_fraction * len(ranked))
    pruned = graph.copy()
    pruned.remove_edges_from(ranked[:n_remove])
    return pruned


def prune_highest_score(graph: nx.Graph, scores: Dict[Tuple[int, int], float], remove_fraction: float) -> nx.Graph:
    """Remove the remove_fraction of edges with the HIGHEST score.

    Containment-style pruning that cuts the most dangerous/central edges
    directly -- used for the degree-centrality, betweenness-centrality, and
    highest-p_uv baselines.
    """
    ranked = _rank_edges_ascending(graph, scores)
    n_remove = round(remove_fraction * len(ranked))
    to_remove = ranked[len(ranked) - n_remove :] if n_remove > 0 else []
    pruned = graph.copy()
    pruned.remove_edges_from(to_remove)
    return pruned


def prune_random(graph: nx.Graph, remove_fraction: float, seed: int) -> nx.Graph:
    """Remove a uniformly random remove_fraction of edges."""
    rng = random.Random(seed)
    edges = list(graph.edges())
    n_remove = round(remove_fraction * len(edges))
    to_remove = rng.sample(edges, n_remove)
    pruned = graph.copy()
    pruned.remove_edges_from(to_remove)
    return pruned


def prune_topk_per_node(graph: nx.Graph, scores: Dict[Tuple[int, int], float], k: int) -> nx.Graph:
    """Retain, for each node, only its k highest-scoring incident edges.

    Per the roadmap: "Top-k pruning per node: retain only the k most
    important edges in each node's neighborhood." An edge survives if EITHER
    endpoint keeps it in its own top-k (union rule): edges aren't
    independently owned by one node, so if either side considers it one of
    its k most important, the edge exists in the resulting graph. (The
    alternative, intersection, would risk orphaning nodes whose only strong
    edge isn't reciprocally top-ranked by the other endpoint.)
    """
    keep_edges = set()
    for node in graph.nodes():
        incident = list(graph.edges(node))
        ranked = sorted(incident, key=lambda e: _get_score(scores, *e), reverse=True)
        for u, v in ranked[:k]:
            keep_edges.add((min(u, v), max(u, v)))

    pruned = nx.Graph()
    pruned.add_nodes_from(graph.nodes())
    pruned.add_edges_from(keep_edges)
    return pruned


def calibrate_topk_for_target_fraction(
    graph: nx.Graph, scores: Dict[Tuple[int, int], float], target_remove_fraction: float, max_k: int = None
) -> Tuple[int, nx.Graph, float]:
    """Search over k for the per-node top-k pruning closest to a target removal fraction.

    Per-node top-k doesn't map linearly (or exactly) to an overall removal
    fraction because of the union rule above, so this scans all candidate k
    and returns whichever gets closest, along with the pruned graph and the
    ACTUAL achieved fraction (which callers should report alongside the
    target, not assume matches it exactly).
    """
    if max_k is None:
        max_k = max(dict(graph.degree()).values())
    original_edges = graph.number_of_edges()

    best_diff, best_k, best_pruned, best_fraction = None, None, None, None
    for k in range(1, max_k + 1):
        pruned = prune_topk_per_node(graph, scores, k)
        achieved_fraction = 1 - pruned.number_of_edges() / original_edges
        diff = abs(achieved_fraction - target_remove_fraction)
        if best_diff is None or diff < best_diff:
            best_diff, best_k, best_pruned, best_fraction = diff, k, pruned, achieved_fraction
    return best_k, best_pruned, best_fraction


def measure_security_and_utility(
    original_graph: nx.Graph,
    pruned_graph: nx.Graph,
    node_features: Dict[int, Dict[str, float]],
    labels: Dict[int, int],
    p_uv: Dict[Tuple[int, int], float],
    hub_node: int,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    model_seed: int,
    n_rollouts: int = 15,
    infection_seed_base: int = 7,
    train_epochs: int = 300,
) -> Dict[str, float]:
    """Measure both the security and utility effect of one pruned graph.

    Convenience wrapper around measure_security + measure_utility for the
    single-seed-node case. When testing security under MULTIPLE seed nodes
    (a robustness check), call measure_security directly for each seed and
    measure_utility once -- utility doesn't depend on which node infection
    was seeded from, so there's no need to retrain per seed node.
    """
    infected_fraction = measure_security(pruned_graph, p_uv, hub_node, n_rollouts=n_rollouts, infection_seed_base=infection_seed_base)
    utility = measure_utility(pruned_graph, node_features, labels, train_mask, test_mask, model_seed=model_seed, epochs=train_epochs)
    return {
        "infected_fraction": infected_fraction,
        "edges_remaining": pruned_graph.number_of_edges(),
        "fraction_removed": 1 - pruned_graph.number_of_edges() / original_graph.number_of_edges(),
        **utility,
    }


def measure_security(
    pruned_graph: nx.Graph,
    p_uv: Dict[Tuple[int, int], float],
    seed_node: int,
    n_rollouts: int = 15,
    infection_seed_base: int = 7,
) -> float:
    """Mean infected fraction over n_rollouts rollouts seeded from seed_node.

    Uses the ORIGINAL graph's p_uv values for whichever edges survive in
    pruned_graph (see module docstring for why features/p_uv aren't
    recomputed on the pruned topology). Lower is better (more contained).
    """
    fractions = []
    for rollout_index in range(n_rollouts):
        result = simulate_botnet(pruned_graph, p_uv, initial_compromised={seed_node}, seed=infection_seed_base + rollout_index)
        fractions.append(len(result["infected_nodes"]) / pruned_graph.number_of_nodes())
    return statistics.mean(fractions)


def measure_utility(
    pruned_graph: nx.Graph,
    node_features: Dict[int, Dict[str, float]],
    labels: Dict[int, int],
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    model_seed: int,
    epochs: int = 300,
    compromised_class: int = 1,
) -> Dict[str, float]:
    """Train a fresh GAT on the pruned graph's topology and score it on test_mask.

    Reports overall accuracy AND precision/recall/F1 for the compromised
    class specifically -- accuracy alone can look flat/uninformative on an
    imbalanced task while masking whether the model still finds compromised
    nodes at all post-pruning (the same reasoning as Milestone 2's classifier
    evaluation in demo_milestone2.py).
    """
    pruned_data = build_pyg_data(pruned_graph, node_features, labels)
    model = train_gat(pruned_data, train_mask, model_seed=model_seed, epochs=epochs)
    with torch.no_grad():
        logits = model(pruned_data.x, pruned_data.edge_index)
        predictions = logits.argmax(dim=1)

    result = precision_recall_f1_counts(predictions, pruned_data.y, test_mask, compromised_class)
    result["test_acc"] = accuracy(predictions, pruned_data.y, test_mask)
    return result


def edge_removal_fraction_for_node(original_graph: nx.Graph, pruned_graph: nx.Graph, node: int) -> float:
    """Fraction of `node`'s OWN incident edges that were removed by pruning.

    Diagnostic for whether a pruning method's apparent security effect comes
    from generally reducing spread network-wide, or from disconnecting/
    near-isolating a specific node (e.g. a fixed infection-seed hub) --
    a method could look artificially dominant if it happens to sever the
    seed node's edges rather than containing spread in general.
    """
    original_degree = original_graph.degree(node)
    if original_degree == 0:
        return 0.0
    pruned_degree = pruned_graph.degree(node)
    return 1 - pruned_degree / original_degree


def containment_ratio(infected_pruned: float, infected_unpruned: float, epsilon: float = 0.01) -> float:
    """Normalize a pruned graph's infected fraction against the SAME seed's
    unpruned infected fraction.

    Raw infected fractions aren't comparable across different infection-seed
    choices, since they can have very different baseline spread rates (e.g. a
    hub seed might reach ~0.36 unpruned while a low-degree seed barely
    spreads at all) -- the same absolute drop means very different things for
    each. <1 means pruning helped, 1 means no effect, >1 would mean pruning
    made things WORSE (shouldn't happen for a sound method, but is possible
    in principle and worth surfacing rather than assuming away).

    epsilon floors the denominator so a near-zero unpruned baseline doesn't
    produce a wild or undefined ratio; callers should treat ratios computed
    against a baseline near that floor as unreliable and say so.
    """
    return infected_pruned / max(infected_unpruned, epsilon)


def plot_containment_ratio_vs_pruning_level(results: List[Dict], output_path: str = "containment_ratio_vs_pruning_level.png") -> str:
    """Roadmap's required security/utility trade-off figure: containment
    ratio (y, lower = more contained) vs. pruning level (x), one line per
    method, with error bars showing the seed-to-seed standard deviation.

    `results` is a list of dicts, one per (method, level), each needing at
    least "method", "level", "containment_ratio", "containment_ratio_std"
    keys -- exactly the shape demo_milestone2_pruning.py's results list is
    already built in.
    """
    # Preserve first-seen method order and assign each a FIXED color/marker
    # once -- identity should stay stable across levels/plots, never
    # re-cycled by rank or reassigned if the method list is filtered later.
    methods = []
    for r in results:
        if r["method"] not in methods:
            methods.append(r["method"])
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    plt.figure(figsize=(7.5, 5.5))
    for i, method in enumerate(methods):
        method_results = sorted((r for r in results if r["method"] == method), key=lambda r: r["level"])
        levels = [r["level"] for r in method_results]
        ratios = [r["containment_ratio"] for r in method_results]
        stds = [r["containment_ratio_std"] for r in method_results]
        plt.errorbar(
            levels,
            ratios,
            yerr=stds,
            label=method,
            color=colors[i % len(colors)],
            marker=markers[i % len(markers)],
            capsize=4,
            linewidth=1.8,
            markersize=6,
        )

    plt.axhline(1.0, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="no effect (ratio = 1.0)")
    plt.xlabel("fraction of edges removed (pruning level)")
    plt.ylabel("containment ratio (infected_pruned / infected_unpruned)")
    plt.title("Security/utility trade-off: containment ratio vs. pruning level")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
