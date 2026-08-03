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


def pick_seed_nodes(graph: nx.Graph, hub_node: int, mid_node: int, n_random: int = 3, rng_seed: int = 42) -> Dict[str, int]:
    """hub + mid-degree + n_random random nodes, all distinct.

    Used to build the multi-seed set that every security measurement in
    Milestones 2-3 averages over, instead of a single fixed infection seed
    (see NOTES.md's fixed-hub-seed-artifact entry for why that matters).
    """
    rng = random.Random(rng_seed)
    excluded = {hub_node, mid_node}
    candidates = [n for n in graph.nodes() if n not in excluded]
    random_nodes = rng.sample(candidates, n_random)

    seed_nodes = {"hub": hub_node, "mid": mid_node}
    for i, node in enumerate(random_nodes):
        seed_nodes[f"random_{i + 1}"] = node
    return seed_nodes


def score_edges_by_avg_hub_score(graph: nx.Graph, node_features: Dict[int, Dict[str, float]]) -> Dict[Tuple[int, int], float]:
    """Degree-centrality edge score: average of the two endpoints' hub_score."""
    return {(u, v): (node_features[u]["hub_score"] + node_features[v]["hub_score"]) / 2 for u, v in graph.edges()}


def score_edges_by_betweenness(graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """Edge betweenness centrality, recomputed fresh as a ranking criterion."""
    return dict(nx.edge_betweenness_centrality(graph, normalized=True))


def score_edges_by_eigenscore(graph: nx.Graph) -> Dict[Tuple[int, int], float]:
    """Spectral/eigenvalue-based edge score: product of the two endpoints'
    eigenvector centrality -- the comparator used in the edge-removal
    epidemic-containment literature (Matamalas et al., Science Advances:
    remove the edges that most reduce the adjacency matrix's largest
    eigenvalue). Eigenvector centrality is the standard per-node proxy for
    that -- high-centrality nodes contribute most to the spectral radius, so
    ranking edges by the PRODUCT of their endpoints' centrality approximates
    targeting the largest eigenvalue directly, without recomputing an
    eigenvalue for every candidate edge at every step (which is what an
    exact greedy-on-eigenvalue search would require).

    Falls back to the exact (numpy-based) eigenvector centrality solver if
    the default power-iteration method fails to converge on this graph --
    the power-iteration method is much cheaper but occasionally doesn't
    converge within its iteration budget on some graphs.
    """
    try:
        centrality = nx.eigenvector_centrality(graph, max_iter=1000)
    except nx.PowerIterationFailedConvergence:
        centrality = nx.eigenvector_centrality_numpy(graph)
    return {(u, v): centrality[u] * centrality[v] for u, v in graph.edges()}


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


def greedy_simulation_guided_prune(
    graph: nx.Graph,
    p_uv: Dict[Tuple[int, int], float],
    seed_nodes: Dict[str, int],
    max_remove_fraction: float,
    checkpoint_fractions: List[float],
    search_rollouts: int = 1,
) -> Dict[float, nx.Graph]:
    """Greedily remove edges one at a time -- at each step, whichever
    remaining edge most reduces simulated infection (averaged across
    seed_nodes) -- as an upper-bound reference the supervisor explicitly
    requested. This is the one pruning method that looks at simulate_botnet
    OUTCOMES directly rather than a structural/attention score, so it's
    expensive: reaching a 50% removal target on an ~900-edge graph needs
    roughly (max_removals x avg_remaining_edges) candidate evaluations
    (~300K here), each running one simulate_botnet rollout per seed node.

    COMPUTE COST TRADE-OFF (report this alongside the numbers, per this
    project's own standard for RL/expensive methods): to keep that
    tractable, the SEARCH itself uses only search_rollouts (default 1)
    rollout per seed node per candidate -- a cheap, noisier signal used only
    to RANK candidates at each step. This does NOT weaken the reported
    security numbers: callers should re-measure the returned checkpoint
    graphs with the same full-rollout measure_security used for every other
    method (this function returns graphs, not final metrics). Edges are
    removed via in-place remove/restore during the search (not graph
    copying) to keep the ~300K candidate evaluations affordable at all --
    copying the whole graph that many times would dominate runtime.

    Reuses ONE incremental removal sequence for every level in
    checkpoint_fractions (each checkpoint's edge set is a strict subset of
    the previous one's), since the greedy trajectory up to 50% already
    passes through whatever it did at 25% and 10% -- so this costs one
    greedy run total, not one per pruning level.
    """
    working_graph = graph.copy()
    total_edges = graph.number_of_edges()
    max_removals = round(max_remove_fraction * total_edges)
    # Map "how many removals in" -> which checkpoint fraction that corresponds to.
    removals_at_checkpoint = {round(fraction * total_edges): fraction for fraction in checkpoint_fractions}

    checkpoints: Dict[float, nx.Graph] = {}
    for step in range(1, max_removals + 1):
        best_edge, best_score = None, None
        for u, v in list(working_graph.edges()):
            edge_attrs = working_graph.get_edge_data(u, v)
            working_graph.remove_edge(u, v)

            total_infected = 0.0
            for seed_node in seed_nodes.values():
                for rollout in range(search_rollouts):
                    result = simulate_botnet(working_graph, p_uv, initial_compromised={seed_node}, seed=rollout)
                    total_infected += len(result["infected_nodes"]) / working_graph.number_of_nodes()

            working_graph.add_edge(u, v, **edge_attrs)  # restore before testing the next candidate

            if best_score is None or total_infected < best_score:
                best_score, best_edge = total_infected, (u, v)

        working_graph.remove_edge(*best_edge)  # commit the best candidate from this step

        if step in removals_at_checkpoint:
            checkpoints[removals_at_checkpoint[step]] = working_graph.copy()

    return checkpoints


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


def measure_utility_frozen(
    model,
    pruned_graph: nx.Graph,
    node_features: Dict[int, Dict[str, float]],
    labels: Dict[int, int],
    test_mask: torch.Tensor,
    compromised_class: int = 1,
) -> Dict[str, float]:
    """Utility of a FIXED, already-trained model on a (possibly pruned) graph
    topology -- a forward pass only, no retraining.

    Milestone 3's RL reward needs a cheap, deterministic utility signal at
    every step (retraining, as measure_utility does, is far too slow to call
    hundreds/thousands of times per training run -- see NOTES.md). Since the
    model is frozen and eval-mode is deterministic, this adds zero randomness
    to the reward, on top of being ~1ms vs. measure_utility's ~3.6s.

    NOTE (see NOTES.md): this is NOT the same utility definition Milestone 2's
    baseline comparison table uses (which retrains a fresh GAT per pruned
    graph). Before comparing RL results against Milestone 2's baselines,
    re-measure the baselines with THIS function too, so the comparison is on
    a consistent utility definition.
    """
    pruned_data = build_pyg_data(pruned_graph, node_features, labels)
    model.eval()
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


def plot_utility_vs_pruning_level(results: List[Dict], output_path: str = "utility_vs_pruning_level.png") -> str:
    """Roadmap's required core-task-performance figure: recall and F1 (y, two
    side-by-side subplots) vs. pruning level (x), one line per method.

    `results` is the same list of dicts plot_containment_ratio_vs_pruning_level
    consumes -- each needing "method", "level", "recall", "f1" keys, exactly
    the shape demo_milestone2_pruning.py's results list is already built in.
    Method identity (color/marker) is assigned the same way as the
    containment-ratio plot -- first-seen order, fixed, never re-cycled by
    rank -- so a method keeps the same visual identity across both figures.
    """
    methods = []
    for r in results:
        if r["method"] not in methods:
            methods.append(r["method"])
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    fig, (recall_ax, f1_ax) = plt.subplots(1, 2, figsize=(13, 5.5))
    for i, method in enumerate(methods):
        method_results = sorted((r for r in results if r["method"] == method), key=lambda r: r["level"])
        levels = [r["level"] for r in method_results]
        recalls = [r["recall"] for r in method_results]
        f1s = [r["f1"] for r in method_results]
        style = dict(color=colors[i % len(colors)], marker=markers[i % len(markers)], linewidth=1.8, markersize=6)
        recall_ax.plot(levels, recalls, label=method, **style)
        f1_ax.plot(levels, f1s, label=method, **style)

    recall_ax.set_xlabel("fraction of edges removed (pruning level)")
    recall_ax.set_ylabel("recall (compromised class)")
    recall_ax.set_title("Recall vs. pruning level")
    recall_ax.set_ylim(0, 1)
    recall_ax.grid(True, alpha=0.3)

    f1_ax.set_xlabel("fraction of edges removed (pruning level)")
    f1_ax.set_ylabel("F1 (compromised class)")
    f1_ax.set_title("F1 vs. pruning level")
    f1_ax.set_ylim(0, 1)
    f1_ax.grid(True, alpha=0.3)
    f1_ax.legend(fontsize=8, loc="upper right")

    fig.suptitle("Core task performance vs. pruning level")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_containment_vs_utility_tradeoff(rows: List[Dict], output_path: str = "containment_vs_utility_tradeoff.png") -> str:
    """Supervisor-requested headline trade-off figure: one point per method,
    containment ratio (x, lower = more contained) vs. frozen F1 (y, higher =
    better core-task utility), each point labeled with its method name and
    mean compute cost (mean_time_s) -- so the plot shows all three axes this
    project cares about (security, utility, cost) at once, at a glance.

    `rows` is a list of dicts, one per method, needing "method",
    "containment_ratio_mean", "frozen_f1_mean", "mean_time_s" keys -- exactly
    harness_summary.csv's row shape (read via csv.DictReader).
    """
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

    xs = [row["containment_ratio_mean"] for row in rows]
    ys = [row["frozen_f1_mean"] for row in rows]
    x_range = (max(xs) - min(xs)) or 1.0
    y_range = (max(ys) - min(ys)) or 1.0

    plt.figure(figsize=(8, 6))
    for i, row in enumerate(rows):
        x, y = xs[i], ys[i]
        plt.scatter(x, y, color=colors[i % len(colors)], marker=markers[i % len(markers)], s=70, zorder=3)
        # Points that land close together in this (very different-scale) 2D
        # space -- e.g. eigenscore vs. betweenness-centrality -- would
        # otherwise render overlapping text. For each point, count how many
        # EARLIER points are within 5% of the axis range in both dimensions
        # and push the label progressively further below the marker for
        # each one found, so a cluster's labels stack readably instead of
        # overlapping.
        close_count = sum(
            1
            for j in range(i)
            if abs(x - xs[j]) / x_range < 0.05 and abs(y - ys[j]) / y_range < 0.05
        )
        y_offset = 6 - close_count * 24
        plt.annotate(
            f"{row['method']}\n({row['mean_time_s']:.2f}s)",
            (x, y),
            textcoords="offset points",
            xytext=(6, y_offset),
            fontsize=8,
        )

    plt.xlabel("containment ratio (lower = more contained)")
    plt.ylabel("frozen F1 (compromised class, higher = better utility)")
    plt.title("Security/utility trade-off across methods (label = mean compute cost per graph)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path
