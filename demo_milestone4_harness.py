"""Milestone 4: 40-graph statistical evaluation harness.

Foundation for every subsequent Milestone 4 ablation/stress-test. Runs all
8 methods (6 static baselines + GAT-threshold + GAT-top-k, plus greedy
oracle and a freshly-trained RL policy = 8) at a single ~50% removal level
across 40 independent graph instances (same generator, varying seed), then:
  - reports per-method mean +/- std for containment ratio and frozen utility
  - runs ~12 curated PAIRED significance tests directly tied to the two
    tentative single-graph findings from Milestones 2-3
  - applies Holm correction across that whole test family
  - reports absolute effect sizes (mean differences) alongside every p-value
  - reports aggregate compute cost per method

Three compute-driven scope decisions (frozen-only utility, 100-episode RL,
single pruning level) are recorded in NOTES.md, decided BEFORE this ran.
"""

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
    greedy_oracle_prune,
    measure_security,
    measure_utility_frozen,
    pick_seed_nodes,
    prune_highest_score,
    prune_lowest_score,
    prune_random,
    score_edges_by_avg_hub_score,
    score_edges_by_betweenness,
)
from src.milestone3 import PruningEnv
from src.milestone3_dqn import run_greedy_episode, train_dqn
from src.milestone4 import run_paired_tests_with_correction

N_GRAPHS = 40
TARGET_LEVEL = 0.50
N_RANDOM_REPEATS = 3
RL_EPISODES = 100
RL_EPSILON_DECAY_EPISODES = 70
MODEL_SEED = 0

METHODS = ["RL", "GAT threshold", "GAT top-k", "degree-centrality", "betweenness-centrality", "highest-p_uv", "random", "greedy oracle"]

# Curated pairs, directly tied to the two tentative single-graph findings --
# NOT all 28 possible pairs (that would make the Holm correction so
# conservative almost nothing could survive it, for questions we don't have).
PAIRS = [
    # "No structural method dominates" (Milestone 2, post simulate_botnet fix)
    ("GAT threshold", "GAT top-k"),
    ("GAT top-k", "degree-centrality"),
    ("degree-centrality", "betweenness-centrality"),
    ("degree-centrality", "highest-p_uv"),
    ("highest-p_uv", "betweenness-centrality"),
    # "RL beats random/oracle but not structural methods" (Milestone 3)
    ("RL", "random"),
    ("RL", "greedy oracle"),
    ("RL", "GAT threshold"),
    ("RL", "GAT top-k"),
    ("RL", "degree-centrality"),
    ("RL", "betweenness-centrality"),
    ("RL", "highest-p_uv"),
]


def run_one_graph(graph_seed: int) -> dict:
    """Build one graph instance, run every method at TARGET_LEVEL, and return
    {method: {"containment_ratio":.., "frozen_recall":.., "frozen_f1":.., "time":..}},
    plus the shared (amortized) GAT-training time for this graph.
    """
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

    t_gat = time.time()
    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED)
    gat_train_time = time.time() - t_gat
    baseline_utility = measure_utility_frozen(base_model, graph, node_features, labels, test_mask)

    t_scores = time.time()
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)
    gat_score_time = time.time() - t_scores

    results = {}

    def measure(method_name: str, pruned_graph_variants: list, elapsed: float) -> None:
        ratios = [
            containment_ratio(measure_security(g, p_uv, node), baseline_infected[name])
            for g in pruned_graph_variants
            for name, node in seed_nodes.items()
        ]
        frozen_list = [measure_utility_frozen(base_model, g, node_features, labels, test_mask) for g in pruned_graph_variants]
        results[method_name] = {
            "containment_ratio": statistics.mean(ratios),
            "frozen_recall": statistics.mean(f["recall"] for f in frozen_list),
            "frozen_f1": statistics.mean(f["f1"] for f in frozen_list),
            "time": elapsed,
        }

    t = time.time()
    g = prune_lowest_score(graph, gat_scores, TARGET_LEVEL)
    measure("GAT threshold", [g], gat_score_time + (time.time() - t))

    t = time.time()
    _, g, _ = calibrate_topk_for_target_fraction(graph, gat_scores, TARGET_LEVEL)
    measure("GAT top-k", [g], gat_score_time + (time.time() - t))

    t = time.time()
    degree_scores = score_edges_by_avg_hub_score(graph, node_features)
    g = prune_highest_score(graph, degree_scores, TARGET_LEVEL)
    measure("degree-centrality", [g], time.time() - t)

    t = time.time()
    betweenness_scores = score_edges_by_betweenness(graph)
    g = prune_highest_score(graph, betweenness_scores, TARGET_LEVEL)
    measure("betweenness-centrality", [g], time.time() - t)

    t = time.time()
    g = prune_highest_score(graph, p_uv, TARGET_LEVEL)
    measure("highest-p_uv", [g], time.time() - t)

    t = time.time()
    variants = [prune_random(graph, TARGET_LEVEL, seed=1000 * i + 7) for i in range(N_RANDOM_REPEATS)]
    measure("random", variants, time.time() - t)

    t = time.time()
    checkpoints = greedy_oracle_prune(graph, p_uv, seed_nodes, max_remove_fraction=TARGET_LEVEL, checkpoint_fractions=[TARGET_LEVEL], search_rollouts=1)
    measure("greedy oracle", [checkpoints[TARGET_LEVEL]], time.time() - t)

    t = time.time()
    env = PruningEnv(
        graph=graph,
        node_features=node_features,
        labels=labels,
        p_uv=p_uv,
        seed_nodes=seed_nodes,
        gat_scores=gat_scores,
        base_model=base_model,
        test_mask=test_mask,
        baseline_infected=baseline_infected,
        baseline_utility=baseline_utility,
        max_steps=15,
        max_removal_fraction=TARGET_LEVEL,
        chunk_fraction=0.05,
    )
    q_network, _ = train_dqn(env, n_episodes=RL_EPISODES, epsilon_decay_episodes=RL_EPSILON_DECAY_EPISODES, seed=0)
    run_greedy_episode(env, q_network)
    rl_graph = env.working_graph.copy()
    measure("RL", [rl_graph], time.time() - t)

    return results, gat_train_time


def main() -> None:
    all_results = {method: {"containment_ratio": [], "frozen_recall": [], "frozen_f1": [], "time": []} for method in METHODS}
    gat_train_times = []

    harness_start = time.time()
    for graph_seed in range(N_GRAPHS):
        graph_start = time.time()
        results, gat_train_time = run_one_graph(graph_seed)
        gat_train_times.append(gat_train_time)
        for method in METHODS:
            all_results[method]["containment_ratio"].append(results[method]["containment_ratio"])
            all_results[method]["frozen_recall"].append(results[method]["frozen_recall"])
            all_results[method]["frozen_f1"].append(results[method]["frozen_f1"])
            all_results[method]["time"].append(results[method]["time"])
        print(f"graph {graph_seed:2d}/{N_GRAPHS} done in {time.time() - graph_start:.1f}s (total elapsed {time.time() - harness_start:.0f}s)", flush=True)

    total_elapsed = time.time() - harness_start
    print(f"\nharness complete: {N_GRAPHS} graphs in {total_elapsed:.0f}s ({total_elapsed / N_GRAPHS:.1f}s/graph average)")

    print("\n=== per-method mean +/- std across 40 graphs ===")
    print(f"{'method':<24}{'containment_ratio':>20}{'frozen_recall':>16}{'frozen_f1':>12}{'mean_time_s':>13}")
    for method in METHODS:
        c = all_results[method]["containment_ratio"]
        r = all_results[method]["frozen_recall"]
        f = all_results[method]["frozen_f1"]
        t = all_results[method]["time"]
        print(
            f"{method:<24}{statistics.mean(c):>10.3f}+/-{statistics.stdev(c):.3f}"
            f"{statistics.mean(r):>16.3f}{statistics.mean(f):>12.3f}{statistics.mean(t):>13.2f}"
        )
    print(f"\nshared GAT training time: {statistics.mean(gat_train_times):.2f}s/graph average (amortized across GAT threshold, GAT top-k, and RL)")

    print(f"\n=== {len(PAIRS)} curated paired significance tests, Holm-corrected (alpha=0.05) ===")
    comparisons = run_paired_tests_with_correction(PAIRS, {m: all_results[m]["containment_ratio"] for m in METHODS})
    print(f"{'A vs B':<45}{'mean_A':>8}{'mean_B':>8}{'diff':>8}{'raw_p':>10}{'holm_p':>10}{'sig?':>6}")
    for c in comparisons:
        label = f"{c['method_a']} vs {c['method_b']}"
        print(
            f"{label:<45}{c['mean_a']:>8.3f}{c['mean_b']:>8.3f}{c['mean_diff']:>8.3f}"
            f"{c['p_value']:>10.4f}{c['p_value_holm']:>10.4f}{str(c['significant_holm']):>6}"
        )

    print(
        "\nNo further interpretation forced here -- see NOTES.md for the read-through "
        "of which tentative findings this does/doesn't confirm."
    )


if __name__ == "__main__":
    main()
