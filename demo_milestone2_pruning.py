"""Milestone 2 demo: static pruning + full baseline suite, security vs. utility.

CORRECTED METHODOLOGY (see NOTES.md's "fixed-hub-seed infection is a
methodology trap" entry for the full story): the first pass of this
evaluation always seeded infection at the fixed highest-degree hub node,
which made degree-centrality pruning look dramatically dominant purely
because it happens to fully isolate whichever node is seeded, if that node
is high-degree -- not because it generally reduces spread. Two fixes are
applied here, following the same averaging discipline already used for
botnet rollouts, model-init seeds, and train/test splits elsewhere in this
project:

  1. Security is now averaged over SEED_NODE_ROLES.keys() infection-seed
     choices per graph (hub, a mid-degree node, and several random draws),
     not a single fixed hub seed.
  2. Each seed's infected fraction is reported as a CONTAINMENT RATIO
     (infected_pruned / infected_unpruned, matched to the SAME seed's own
     unpruned baseline), since raw infected fractions aren't comparable
     across seeds with very different baseline spread rates.
  3. Utility is reported via recall/F1 for the compromised class, not raw
     accuracy -- accuracy was confirmed not to discriminate between methods
     in the first pass (flat 0.800-0.867 across all 18 configs).

Implements the roadmap's static pruning strategies (threshold-based and
top-k-per-node, both using the degree-corrected GAT attention score)
alongside the supervisor-mandated baseline suite (random, degree-centrality,
betweenness-centrality, highest-p_uv removal).
"""

import random
import statistics

from src.milestone1 import build_edge_features, compute_edge_infection_probabilities
from src.milestone2 import (
    DEFAULT_BETA,
    build_pyg_data,
    extract_degree_corrected_attention_scores,
    generate_labeled_graph,
    train_gat,
)
from src.milestone2_pruning import (
    calibrate_topk_for_target_fraction,
    containment_ratio,
    measure_security,
    measure_utility,
    plot_containment_ratio_vs_pruning_level,
    prune_highest_score,
    prune_lowest_score,
    prune_random,
    score_edges_by_avg_hub_score,
    score_edges_by_betweenness,
)
from demo_milestone2 import make_node_split

N_NODES = 300
PRUNING_LEVELS = [0.10, 0.25, 0.50]
N_RANDOM_PRUNE_REPEATS = 3
N_RANDOM_SEED_NODES = 3
MODEL_SEED = 0
SEED_NODE_RNG_SEED = 42
CONTAINMENT_RATIO_EPSILON = 0.01


def pick_seed_nodes(graph, hub_node: int, mid_node: int) -> dict:
    """hub + mid-degree + N_RANDOM_SEED_NODES random nodes, all distinct."""
    rng = random.Random(SEED_NODE_RNG_SEED)
    excluded = {hub_node, mid_node}
    candidates = [n for n in graph.nodes() if n not in excluded]
    random_nodes = rng.sample(candidates, N_RANDOM_SEED_NODES)

    seed_nodes = {"hub": hub_node, "mid": mid_node}
    for i, node in enumerate(random_nodes):
        seed_nodes[f"random_{i + 1}"] = node
    return seed_nodes


def build_pruned_graphs_for_level(graph, gat_scores, degree_scores, betweenness_scores, p_uv, level):
    """One pruned graph per method at this level (random gets N_RANDOM_PRUNE_REPEATS draws)."""
    pruned = {}
    pruned["GAT threshold (lowest s_uv removed)"] = [prune_lowest_score(graph, gat_scores, level)]

    _, topk_graph, _ = calibrate_topk_for_target_fraction(graph, gat_scores, level)
    pruned["GAT top-k-per-node"] = [topk_graph]

    pruned["degree-centrality (highest removed)"] = [prune_highest_score(graph, degree_scores, level)]
    pruned["betweenness-centrality (highest removed)"] = [prune_highest_score(graph, betweenness_scores, level)]
    pruned["highest-p_uv (highest removed)"] = [prune_highest_score(graph, p_uv, level)]
    pruned["random"] = [prune_random(graph, level, seed=1000 * repeat + 7) for repeat in range(N_RANDOM_PRUNE_REPEATS)]
    return pruned


def main() -> None:
    graph, node_features, labels = generate_labeled_graph(n_nodes=N_NODES)
    data = build_pyg_data(graph, node_features, labels)
    train_mask, val_mask, test_mask = make_node_split(data.num_nodes, seed=0)

    degrees = dict(graph.degree())
    hub_node = max(degrees, key=degrees.get)
    sorted_by_degree = sorted(degrees.items(), key=lambda kv: kv[1])
    median_degree = sorted_by_degree[len(sorted_by_degree) // 2][1]
    mid_node = min((n for n in degrees if n != hub_node), key=lambda n: abs(degrees[n] - median_degree))
    seed_nodes = pick_seed_nodes(graph, hub_node, mid_node)
    print("seed nodes: " + ", ".join(f"{name}={node} (degree {degrees[node]})" for name, node in seed_nodes.items()))

    edge_features = build_edge_features(graph)
    p_uv = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=DEFAULT_BETA)

    # Unpruned baseline security, PER SEED, computed once -- every containment
    # ratio below is this same value for its seed, never recomputed per method.
    print("computing unpruned baseline security per seed...")
    baseline_infected = {name: measure_security(graph, p_uv, node) for name, node in seed_nodes.items()}
    low_baseline_seeds = [name for name, val in baseline_infected.items() if val < CONTAINMENT_RATIO_EPSILON]
    print("  " + "  ".join(f"{name}={val:.3f}" for name, val in baseline_infected.items()))
    if low_baseline_seeds:
        print(f"  NOTE: baseline infected fraction near the containment-ratio floor for seeds {low_baseline_seeds} -- their ratios below are unreliable, flagged in the summary.")

    print("\ntraining the base GAT (for degree-corrected attention scores)...")
    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED)
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)
    degree_scores = score_edges_by_avg_hub_score(graph, node_features)
    betweenness_scores = score_edges_by_betweenness(graph)

    baseline_utility = measure_utility(graph, node_features, labels, train_mask, test_mask, model_seed=MODEL_SEED)
    print(
        f"\nUNPRUNED utility: recall={baseline_utility['recall']:.3f} f1={baseline_utility['f1']:.3f} "
        f"precision={baseline_utility['precision']:.3f} (accuracy={baseline_utility['test_acc']:.3f}, shown for reference only)"
    )

    results = []
    for level in PRUNING_LEVELS:
        print(f"\n--- pruning level: {level:.0%} ---")
        pruned_by_method = build_pruned_graphs_for_level(graph, gat_scores, degree_scores, betweenness_scores, p_uv, level)

        for method, variants in pruned_by_method.items():
            utilities, fraction_removed, all_ratios = [], [], {name: [] for name in seed_nodes}

            for pruned_graph in variants:
                fraction_removed.append(1 - pruned_graph.number_of_edges() / graph.number_of_edges())
                utilities.append(measure_utility(pruned_graph, node_features, labels, train_mask, test_mask, model_seed=MODEL_SEED))
                for name, node in seed_nodes.items():
                    infected = measure_security(pruned_graph, p_uv, node)
                    all_ratios[name].append(containment_ratio(infected, baseline_infected[name], epsilon=CONTAINMENT_RATIO_EPSILON))

            # Flatten every (variant x seed) ratio into one pool for this
            # method/level's mean+/-std -- this is the "average over multiple
            # infection-seed choices" the fix requires.
            pooled_ratios = [r for seed_list in all_ratios.values() for r in seed_list]
            mean_ratio = statistics.mean(pooled_ratios)
            std_ratio = statistics.stdev(pooled_ratios) if len(pooled_ratios) > 1 else 0.0

            mean_removed = statistics.mean(fraction_removed)
            mean_recall = statistics.mean(u["recall"] for u in utilities)
            mean_f1 = statistics.mean(u["f1"] for u in utilities)
            mean_precision = statistics.mean(u["precision"] for u in utilities)

            print(
                f"  {method:<38} removed={mean_removed:.3f}  "
                f"containment_ratio={mean_ratio:.3f}+/-{std_ratio:.3f}  "
                f"recall={mean_recall:.3f} f1={mean_f1:.3f} prec={mean_precision:.3f}"
            )

            results.append(
                {
                    "method": method,
                    "level": level,
                    "fraction_removed": mean_removed,
                    "containment_ratio": mean_ratio,
                    "containment_ratio_std": std_ratio,
                    "recall": mean_recall,
                    "f1": mean_f1,
                    "precision": mean_precision,
                    "involves_low_baseline_seed": bool(low_baseline_seeds),
                }
            )

    print("\n\n=== summary table (corrected methodology: containment ratio averaged over seeds; recall/F1 as utility) ===")
    print(f"{'method':<38}{'level':>6}{'removed':>9}{'contain_ratio':>16}{'recall':>8}{'f1':>8}{'prec':>8}")
    for r in results:
        flag = " *" if r["involves_low_baseline_seed"] else ""
        print(
            f"{r['method']:<38}{r['level']:>6.2f}{r['fraction_removed']:>9.3f}"
            f"{r['containment_ratio']:>10.3f}+/-{r['containment_ratio_std']:.3f}"
            f"{r['recall']:>8.3f}{r['f1']:>8.3f}{r['precision']:>8.3f}{flag}"
        )
    if any(r["involves_low_baseline_seed"] for r in results):
        print("* one or more pooled seeds had a near-zero unpruned baseline; those seeds' ratios are noted as unreliable above.")

    plot_path = plot_containment_ratio_vs_pruning_level(results)
    print(f"\nsaved trade-off plot to {plot_path}")


if __name__ == "__main__":
    main()
