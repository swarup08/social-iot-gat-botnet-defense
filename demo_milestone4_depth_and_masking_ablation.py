"""Milestone 4: GAT-depth (1/2/3 layers) and feature-masking ablations.

Two remaining roadmap-requested ablation angles not covered by
demo_milestone4_gat_ablations.py (which varied heads and added Gaussian
noise): number of GAT layers, and "incomplete" (randomly zeroed / missing)
features rather than noisy ones. Same harness pattern as that script: 15
graphs, containment only, no RL/greedy oracle (neither question involves
those).

Depth ablation uses GATNodeClassifierNLayer (src/milestone2.py) -- a
separate, additive variable-depth model added specifically for this
ablation; GATNodeClassifier (the fixed 2-layer architecture used everywhere
else in this project) is untouched, and n_layers=2 here is architecturally
identical to it (verified in tests/test_milestone2.py), so the n_layers=2
row is a fresh baseline re-measurement, not reused from the heads ablation.

Feature-masking ablation zeroes out a random subset of the same continuous
fields the noise ablation perturbed (risk, hub_score, clustering), rather
than adding Gaussian noise -- simulating a device that stops reporting a
field entirely (a missing/failed sensor reading) instead of one that reports
a corrupted value.
"""

import copy
import random
import statistics
import time

from src.milestone1 import build_edge_features, compute_edge_infection_probabilities
from src.milestone2 import (
    DEFAULT_BETA,
    GATNodeClassifierNLayer,
    build_pyg_data,
    extract_degree_corrected_attention_scores,
    extract_degree_corrected_attention_scores_nlayer,
    generate_labeled_graph,
    make_node_split,
    train_gat,
)
from src.milestone2_pruning import (
    calibrate_topk_for_target_fraction,
    containment_ratio,
    measure_security,
    pick_seed_nodes,
    prune_highest_score,
    prune_lowest_score,
    prune_random,
    score_edges_by_avg_hub_score,
    score_edges_by_betweenness,
)
from src.milestone4 import run_paired_tests_with_correction

N_GRAPHS = 15
TARGET_LEVEL = 0.50
N_RANDOM_REPEATS = 3
MODEL_SEED = 0
MASKED_KEYS = ("risk", "hub_score", "clustering")

STATIC_METHODS = ["degree-centrality", "betweenness-centrality", "highest-p_uv", "random"]


def add_feature_masking(node_features: dict, mask_prob: float, seed: int) -> dict:
    """Return a COPY of node_features with each continuous field independently
    zeroed out with probability mask_prob -- simulating a sensor that stops
    reporting a value entirely, rather than one that reports a noisy value.
    The original dict is left untouched, same discipline as add_feature_noise
    in demo_milestone4_gat_ablations.py.
    """
    rng = random.Random(seed)
    masked = copy.deepcopy(node_features)
    for node, features in masked.items():
        for key in MASKED_KEYS:
            if rng.random() < mask_prob:
                features[key] = 0.0
    return masked


def setup_graph(graph_seed: int):
    graph, node_features, labels = generate_labeled_graph(n_nodes=300, graph_seed=graph_seed)
    data = build_pyg_data(graph, node_features, labels)
    train_mask, val_mask, test_mask = make_node_split(data.num_nodes, seed=0)

    degrees = dict(graph.degree())
    hub_node = max(degrees, key=degrees.get)
    sorted_by_degree = sorted(degrees.items(), key=lambda kv: kv[1])
    median_degree = sorted_by_degree[len(sorted_by_degree) // 2][1]
    mid_node = min((n for n in degrees if n != hub_node), key=lambda n: abs(degrees[n] - median_degree))
    seed_nodes = pick_seed_nodes(graph, hub_node, mid_node)

    edge_features = build_edge_features(graph)
    p_uv = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=DEFAULT_BETA)
    baseline_infected = {name: measure_security(graph, p_uv, node) for name, node in seed_nodes.items()}

    return graph, node_features, labels, train_mask, p_uv, seed_nodes, baseline_infected


def measure_containment(pruned_graph_variants: list, p_uv, seed_nodes, baseline_infected) -> float:
    ratios = [
        containment_ratio(measure_security(g, p_uv, node), baseline_infected[name])
        for g in pruned_graph_variants
        for name, node in seed_nodes.items()
    ]
    return statistics.mean(ratios)


def run_gat_variant_depth(graph, node_features_for_gat, labels, train_mask, p_uv, seed_nodes, baseline_infected, n_layers: int) -> dict:
    data = build_pyg_data(graph, node_features_for_gat, labels)
    base_model = train_gat(
        data,
        train_mask,
        model_seed=MODEL_SEED,
        model_factory=lambda: GATNodeClassifierNLayer(in_channels=data.num_node_features, n_layers=n_layers),
    )
    gat_scores = extract_degree_corrected_attention_scores_nlayer(base_model, data, graph)

    g_threshold = prune_lowest_score(graph, gat_scores, TARGET_LEVEL)
    _, g_topk, _ = calibrate_topk_for_target_fraction(graph, gat_scores, TARGET_LEVEL)

    return {
        "GAT threshold": measure_containment([g_threshold], p_uv, seed_nodes, baseline_infected),
        "GAT top-k": measure_containment([g_topk], p_uv, seed_nodes, baseline_infected),
    }


def run_gat_variant_masked(graph, node_features_for_gat, labels, train_mask, p_uv, seed_nodes, baseline_infected) -> dict:
    data = build_pyg_data(graph, node_features_for_gat, labels)
    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED, heads=4)
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)

    g_threshold = prune_lowest_score(graph, gat_scores, TARGET_LEVEL)
    _, g_topk, _ = calibrate_topk_for_target_fraction(graph, gat_scores, TARGET_LEVEL)

    return {
        "GAT threshold": measure_containment([g_threshold], p_uv, seed_nodes, baseline_infected),
        "GAT top-k": measure_containment([g_topk], p_uv, seed_nodes, baseline_infected),
    }


def run_static_baselines(graph, node_features, p_uv, seed_nodes, baseline_infected) -> dict:
    results = {}
    degree_scores = score_edges_by_avg_hub_score(graph, node_features)
    results["degree-centrality"] = measure_containment([prune_highest_score(graph, degree_scores, TARGET_LEVEL)], p_uv, seed_nodes, baseline_infected)

    betweenness_scores = score_edges_by_betweenness(graph)
    results["betweenness-centrality"] = measure_containment([prune_highest_score(graph, betweenness_scores, TARGET_LEVEL)], p_uv, seed_nodes, baseline_infected)

    results["highest-p_uv"] = measure_containment([prune_highest_score(graph, p_uv, TARGET_LEVEL)], p_uv, seed_nodes, baseline_infected)

    variants = [prune_random(graph, TARGET_LEVEL, seed=1000 * i + 7) for i in range(N_RANDOM_REPEATS)]
    results["random"] = measure_containment(variants, p_uv, seed_nodes, baseline_infected)
    return results


def report(label: str, all_results: dict) -> None:
    print(f"\n--- {label} ---")
    methods = list(all_results.keys())
    for m in methods:
        vals = all_results[m]
        print(f"  {m:<22} containment={statistics.mean(vals):.3f}+/-{statistics.stdev(vals):.3f}")

    pairs = [(g, s) for g in ("GAT threshold", "GAT top-k") for s in ("degree-centrality", "betweenness-centrality", "highest-p_uv")]
    comparisons = run_paired_tests_with_correction(pairs, all_results)
    print(f"  {'A vs B':<40}{'mean_A':>8}{'mean_B':>8}{'diff':>8}{'holm_p':>9}{'sig?':>6}")
    for c in comparisons:
        label2 = f"{c['method_a']} vs {c['method_b']}"
        print(f"  {label2:<40}{c['mean_a']:>8.3f}{c['mean_b']:>8.3f}{c['mean_diff']:>8.3f}{c['p_value_holm']:>9.4f}{str(c['significant_holm']):>6}")


def main() -> None:
    start = time.time()

    depth_results = {n: {"GAT threshold": [], "GAT top-k": []} for n in (1, 2, 3)}
    static_results = {m: [] for m in STATIC_METHODS}
    mask_results = {p: {"GAT threshold": [], "GAT top-k": []} for p in (0.1, 0.3)}

    for graph_seed in range(N_GRAPHS):
        graph, node_features, labels, train_mask, p_uv, seed_nodes, baseline_infected = setup_graph(graph_seed)

        for m in STATIC_METHODS:
            static_results[m].append(run_static_baselines(graph, node_features, p_uv, seed_nodes, baseline_infected)[m])

        for n_layers in (1, 2, 3):
            r = run_gat_variant_depth(graph, node_features, labels, train_mask, p_uv, seed_nodes, baseline_infected, n_layers)
            depth_results[n_layers]["GAT threshold"].append(r["GAT threshold"])
            depth_results[n_layers]["GAT top-k"].append(r["GAT top-k"])

        for mask_prob in (0.1, 0.3):
            masked_features = add_feature_masking(node_features, mask_prob, seed=graph_seed)
            r = run_gat_variant_masked(graph, masked_features, labels, train_mask, p_uv, seed_nodes, baseline_infected)
            mask_results[mask_prob]["GAT threshold"].append(r["GAT threshold"])
            mask_results[mask_prob]["GAT top-k"].append(r["GAT top-k"])

        print(f"graph {graph_seed:2d}/{N_GRAPHS} done (elapsed {time.time() - start:.0f}s)", flush=True)

    print(f"\nall variants complete in {time.time() - start:.0f}s")

    for n_layers in (1, 2, 3):
        combined = {**depth_results[n_layers], **static_results}
        report(f"n_layers={n_layers}" + (" (baseline depth)" if n_layers == 2 else ""), combined)

    for mask_prob in (0.1, 0.3):
        combined = {**mask_results[mask_prob], **static_results}
        report(f"feature mask_prob={mask_prob}", combined)

    print("\nNo further interpretation forced here -- see NOTES.md for the read-through.")


if __name__ == "__main__":
    main()
