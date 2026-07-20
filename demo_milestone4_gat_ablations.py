"""Milestone 4: GAT-heads and noisy-feature ablations, using the harness.

Quick pass (15 graphs, no RL/greedy simulation-guided heuristic -- neither
ablation question is about those) answering: does varying the GAT's architecture or corrupting
its input features change the core finding (GAT-based pruning's containment
standing relative to the structural baselines), or just confirm it's robust?

Scope note: GATNodeClassifier's DEPTH is hardcoded to exactly 2 GATConv
layers (the attention-extraction code assumes this structurally throughout,
and many existing tests depend on it) -- varying depth would need a real
refactor, not a quick ablation. This pass varies HEADS only (2/4/8); depth
robustness is an open question, not tested here.

Noisy-feature ablation: noise is injected ONLY into what the GAT observes
as input (the continuous features risk/hub_score/clustering, Gaussian noise,
clipped to [0,1]) -- NOT into the features used for labels, p_uv, or the
structural baselines (which read node_features directly, e.g.
degree-centrality via hub_score). Otherwise the "ablation" would corrupt the
ground truth itself rather than testing robustness to noisy observations.
"""

import copy
import random
import statistics
import time

from src.milestone1 import build_edge_features, compute_edge_infection_probabilities
from src.milestone2 import (
    DEFAULT_BETA,
    build_pyg_data,
    extract_degree_corrected_attention_scores,
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
NOISY_KEYS = ("risk", "hub_score", "clustering")

STATIC_METHODS = ["degree-centrality", "betweenness-centrality", "highest-p_uv", "random"]


def add_feature_noise(node_features: dict, noise_std: float, seed: int) -> dict:
    """Return a COPY of node_features with Gaussian noise added to the
    continuous fields only, clipped to [0, 1] (their natural range). The
    original dict is left untouched -- callers use the noisy copy only for
    building the GAT's input tensor, never for labels/p_uv/structural scores.
    """
    rng = random.Random(seed)
    noisy = copy.deepcopy(node_features)
    for node, features in noisy.items():
        for key in NOISY_KEYS:
            noisy_value = features[key] + rng.gauss(0, noise_std)
            features[key] = max(0.0, min(1.0, noisy_value))
    return noisy


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


def run_gat_variant(graph, node_features_for_gat, labels, train_mask, p_uv, seed_nodes, baseline_infected, heads: int) -> dict:
    data = build_pyg_data(graph, node_features_for_gat, labels)
    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED, heads=heads)
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)

    g_threshold = prune_lowest_score(graph, gat_scores, TARGET_LEVEL)
    g_topk_result = calibrate_topk_for_target_fraction(graph, gat_scores, TARGET_LEVEL)
    g_topk = g_topk_result[1]

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

    heads_results = {"GAT threshold": [], "GAT top-k": []}
    heads_variant_results = {2: {"GAT threshold": [], "GAT top-k": []}, 8: {"GAT threshold": [], "GAT top-k": []}}
    static_results = {m: [] for m in STATIC_METHODS}

    noise_variant_results = {0.1: {"GAT threshold": [], "GAT top-k": []}, 0.3: {"GAT threshold": [], "GAT top-k": []}}

    for graph_seed in range(N_GRAPHS):
        graph, node_features, labels, train_mask, p_uv, seed_nodes, baseline_infected = setup_graph(graph_seed)

        for m in STATIC_METHODS:
            static_results[m].append(run_static_baselines(graph, node_features, p_uv, seed_nodes, baseline_infected)[m])

        baseline_gat = run_gat_variant(graph, node_features, labels, train_mask, p_uv, seed_nodes, baseline_infected, heads=4)
        heads_results["GAT threshold"].append(baseline_gat["GAT threshold"])
        heads_results["GAT top-k"].append(baseline_gat["GAT top-k"])

        for heads in (2, 8):
            r = run_gat_variant(graph, node_features, labels, train_mask, p_uv, seed_nodes, baseline_infected, heads=heads)
            heads_variant_results[heads]["GAT threshold"].append(r["GAT threshold"])
            heads_variant_results[heads]["GAT top-k"].append(r["GAT top-k"])

        for noise_std in (0.1, 0.3):
            noisy_features = add_feature_noise(node_features, noise_std, seed=graph_seed)
            r = run_gat_variant(graph, noisy_features, labels, train_mask, p_uv, seed_nodes, baseline_infected, heads=4)
            noise_variant_results[noise_std]["GAT threshold"].append(r["GAT threshold"])
            noise_variant_results[noise_std]["GAT top-k"].append(r["GAT top-k"])

        print(f"graph {graph_seed:2d}/{N_GRAPHS} done (elapsed {time.time() - start:.0f}s)", flush=True)

    print(f"\nall variants complete in {time.time() - start:.0f}s")

    combined_baseline = {**heads_results, **static_results}
    report("heads=4 (baseline)", combined_baseline)

    for heads in (2, 8):
        combined = {**heads_variant_results[heads], **static_results}
        report(f"heads={heads}", combined)

    for noise_std in (0.1, 0.3):
        combined = {**noise_variant_results[noise_std], **static_results}
        report(f"feature noise_std={noise_std}", combined)

    print("\nNo further interpretation forced here -- see NOTES.md for the read-through.")


if __name__ == "__main__":
    main()
