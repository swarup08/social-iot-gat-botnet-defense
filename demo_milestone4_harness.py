"""Milestone 4: 40-graph statistical evaluation harness.

Foundation for every subsequent Milestone 4 ablation/stress-test. Runs all
9 methods (7 static baselines + GAT-threshold + GAT-top-k, plus a greedy
simulation-guided heuristic and a freshly-trained RL policy = 9) at a single
~50% removal level across 40 independent graph instances (same generator,
varying seed), then:
  - reports per-method mean +/- std for containment ratio and frozen utility
  - runs 14 curated PAIRED significance tests directly tied to specific
    tentative findings from Milestones 2-3 and a supervisor-flagged
    literature gap (Milestone 4: no spectral/eigenvalue-based baseline)
  - applies Holm correction across that whole test family
  - reports absolute effect sizes (mean differences) alongside every p-value
  - reports aggregate compute cost per method

The "eigenscore" baseline (score_edges_by_eigenscore, src/milestone2_pruning.py)
was added after the original 8-method/12-pair run, per supervisor review:
the accepted comparator in the edge-removal epidemic-containment literature
(Matamalas et al., Science Advances) is removing edges that most reduce the
adjacency matrix's largest eigenvalue; eigenvector-centrality-product per
edge is the standard proxy for that. This is the one explicitly-authorized
exception to "no new experiments after week 2" -- see NOTES.md.

Three compute-driven scope decisions (frozen-only utility, 100-episode RL,
single pruning level) are recorded in NOTES.md, decided BEFORE the original
run. "eigenscore" reuses the identical harness/methodology, added as a 9th
method rather than a separate script, so it is directly comparable to the
other 8 on the same 40 graph instances.
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
    greedy_simulation_guided_prune,
    measure_security,
    measure_utility_frozen,
    pick_seed_nodes,
    prune_highest_score,
    prune_lowest_score,
    prune_random,
    score_edges_by_avg_hub_score,
    score_edges_by_betweenness,
    score_edges_by_eigenscore,
)
from src.milestone3 import PruningEnv
from src.milestone3_dqn import run_greedy_episode, train_dqn
from src.milestone3_xai import save_table_csv
from src.milestone4 import run_paired_tests_with_correction

N_GRAPHS = 40
TARGET_LEVEL = 0.50
N_RANDOM_REPEATS = 3
RL_EPISODES = 100
RL_EPSILON_DECAY_EPISODES = 70
MODEL_SEED = 0

# Supervisor-flagged fix: PruningEnv's default infection_seed_base (7) is what
# the RL agent's reward signal is computed from during every training step
# (see src/milestone3.py). If the FINAL reported containment ratio also used
# seed base 7, the RL policy would be graded on the exact same infection
# realizations it was trained/rewarded against -- not an independent test.
# This constant is therefore used for every method's FINAL measurement below
# (measure()), deliberately disjoint from PruningEnv's training seed range
# (7..7+n_rollouts-1 = 7..21), so RL's reported number reflects infection
# rollouts it never saw during training. Applied uniformly to all 9 methods
# (not just RL) so every method's Table 1 number comes from the same,
# clearly-labeled held-out evaluation seed pool.
EVAL_INFECTION_SEED_BASE = 10007

METHODS = [
    "RL",
    "GAT threshold",
    "GAT top-k",
    "degree-centrality",
    "betweenness-centrality",
    "highest-p_uv",
    "random",
    "greedy simulation-guided heuristic",
    "eigenscore",
]

# Curated pairs, directly tied to specific tentative findings/reviewer
# questions -- NOT all pairwise combinations (that would make the Holm
# correction so conservative almost nothing could survive it, for questions
# we don't have).
PAIRS = [
    # "No structural method dominates" (Milestone 2, post simulate_botnet fix)
    ("GAT threshold", "GAT top-k"),
    ("GAT top-k", "degree-centrality"),
    ("degree-centrality", "betweenness-centrality"),
    ("degree-centrality", "highest-p_uv"),
    ("highest-p_uv", "betweenness-centrality"),
    # "RL beats random/greedy simulation-guided heuristic but not structural methods" (Milestone 3)
    ("RL", "random"),
    ("RL", "greedy simulation-guided heuristic"),
    ("RL", "GAT threshold"),
    ("RL", "GAT top-k"),
    ("RL", "degree-centrality"),
    ("RL", "betweenness-centrality"),
    ("RL", "highest-p_uv"),
    # Spectral/eigenvalue-based baseline vs. the two comparisons a reviewer
    # will most want to see (supervisor-flagged literature gap, Milestone 4)
    ("eigenscore", "degree-centrality"),
    ("eigenscore", "RL"),
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

    # Two separate baselines, deliberately: `baseline_infected` (default seed
    # base 7) stays paired with PruningEnv's training-time reward signal
    # below, unchanged. `baseline_infected_eval` uses the disjoint
    # EVAL_INFECTION_SEED_BASE so that every method's FINAL containment_ratio
    # (numerator AND denominator) is computed from the SAME held-out seed
    # pool -- mixing a seed-7 baseline with a seed-10007 numerator would be
    # an internally inconsistent ratio, not a fair fix.
    baseline_infected = {name: measure_security(graph, p_uv, node) for name, node in seed_nodes.items()}
    baseline_infected_eval = {
        name: measure_security(graph, p_uv, node, infection_seed_base=EVAL_INFECTION_SEED_BASE)
        for name, node in seed_nodes.items()
    }

    t_gat = time.time()
    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED)
    gat_train_time = time.time() - t_gat
    baseline_utility = measure_utility_frozen(base_model, graph, node_features, labels, test_mask)
    # Supervisor-flagged fix: PruningEnv's reward signal must be measured on
    # a mask DISJOINT from test_mask, since test_mask is what measure()
    # below uses to report every method's FINAL frozen_recall/frozen_f1 for
    # Table 1 -- reusing test_mask for RL's reward would mean its "held-out"
    # test set was actually seen as reward signal during training. val_mask
    # (already split out by make_node_split, previously unused here) is the
    # correct, disjoint mask for this.
    baseline_utility_reward = measure_utility_frozen(base_model, graph, node_features, labels, val_mask)

    t_scores = time.time()
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)
    gat_score_time = time.time() - t_scores

    results = {}

    def measure(method_name: str, pruned_graph_variants: list, elapsed: float) -> None:
        ratios = [
            containment_ratio(
                measure_security(g, p_uv, node, infection_seed_base=EVAL_INFECTION_SEED_BASE),
                baseline_infected_eval[name],  # same held-out seed pool as the numerator above
            )
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
    eigenscore_scores = score_edges_by_eigenscore(graph)
    g = prune_highest_score(graph, eigenscore_scores, TARGET_LEVEL)
    measure("eigenscore", [g], time.time() - t)

    t = time.time()
    checkpoints = greedy_simulation_guided_prune(graph, p_uv, seed_nodes, max_remove_fraction=TARGET_LEVEL, checkpoint_fractions=[TARGET_LEVEL], search_rollouts=1)
    measure("greedy simulation-guided heuristic", [checkpoints[TARGET_LEVEL]], time.time() - t)

    t = time.time()
    env = PruningEnv(
        graph=graph,
        node_features=node_features,
        labels=labels,
        p_uv=p_uv,
        seed_nodes=seed_nodes,
        gat_scores=gat_scores,
        base_model=base_model,
        reward_mask=val_mask,
        baseline_infected=baseline_infected,
        baseline_utility=baseline_utility_reward,
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
    summary_rows = []
    for method in METHODS:
        c = all_results[method]["containment_ratio"]
        r = all_results[method]["frozen_recall"]
        f = all_results[method]["frozen_f1"]
        t = all_results[method]["time"]
        print(
            f"{method:<24}{statistics.mean(c):>10.3f}+/-{statistics.stdev(c):.3f}"
            f"{statistics.mean(r):>16.3f}{statistics.mean(f):>12.3f}{statistics.mean(t):>13.2f}"
        )
        summary_rows.append(
            {
                "method": method,
                "containment_ratio_mean": statistics.mean(c),
                "containment_ratio_std": statistics.stdev(c),
                "frozen_recall_mean": statistics.mean(r),
                "frozen_f1_mean": statistics.mean(f),
                "mean_time_s": statistics.mean(t),
                "n_graphs": len(c),
            }
        )
    print(f"\nshared GAT training time: {statistics.mean(gat_train_times):.2f}s/graph average (amortized across GAT threshold, GAT top-k, and RL)")

    summary_path = save_table_csv(summary_rows, "harness_summary.csv")
    print(f"saved per-method summary table to {summary_path}")

    # Per-reviewer request: release the raw per-graph containment_ratio
    # values behind harness_summary.csv's aggregated mean/std, so the paper
    # can publish per-graph paired outcomes for reproducibility. One row
    # per (graph_seed, method); all_results[method][metric] lists are
    # appended once per graph_seed in ascending order inside the main loop
    # above, so index i always corresponds to graph_seed=i.
    per_graph_rows = []
    for method in METHODS:
        c = all_results[method]["containment_ratio"]
        r = all_results[method]["frozen_recall"]
        f = all_results[method]["frozen_f1"]
        t = all_results[method]["time"]
        for graph_seed in range(N_GRAPHS):
            per_graph_rows.append(
                {
                    "graph_seed": graph_seed,
                    "method": method,
                    "containment_ratio": c[graph_seed],
                    "frozen_recall": r[graph_seed],
                    "frozen_f1": f[graph_seed],
                    "time_s": t[graph_seed],
                }
            )
    per_graph_path = save_table_csv(per_graph_rows, "harness_per_graph.csv")
    print(f"saved per-graph raw results table to {per_graph_path} ({len(per_graph_rows)} rows)")

    print(f"\n=== {len(PAIRS)} curated paired significance tests, Holm-corrected (alpha=0.05) ===")
    comparisons = run_paired_tests_with_correction(PAIRS, {m: all_results[m]["containment_ratio"] for m in METHODS})
    print(f"{'A vs B':<45}{'mean_A':>8}{'mean_B':>8}{'diff':>8}{'raw_p':>10}{'holm_p':>10}{'sig?':>6}")
    for c in comparisons:
        label = f"{c['method_a']} vs {c['method_b']}"
        print(
            f"{label:<45}{c['mean_a']:>8.3f}{c['mean_b']:>8.3f}{c['mean_diff']:>8.3f}"
            f"{c['p_value']:>10.4f}{c['p_value_holm']:>10.4f}{str(c['significant_holm']):>6}"
        )

    paired_rows = [
        {
            "method_a": c["method_a"],
            "method_b": c["method_b"],
            "mean_a": c["mean_a"],
            "mean_b": c["mean_b"],
            "mean_diff": c["mean_diff"],
            "t_stat": c["t_stat"],
            "p_value": c["p_value"],
            "n": c["n"],
            "holm_corrected_pvalue": c["p_value_holm"],
            "significant_holm": c["significant_holm"],
        }
        for c in comparisons
    ]
    paired_path = save_table_csv(paired_rows, "harness_paired_tests.csv")
    print(f"saved paired significance test table to {paired_path}")

    print(
        "\nNo further interpretation forced here -- see NOTES.md for the read-through "
        "of which tentative findings this does/doesn't confirm."
    )


if __name__ == "__main__":
    main()
