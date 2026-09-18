"""Milestone 4: RL reward-weighting ablation, on the 40-graph harness.

Two questions, both requiring FRESH per-graph raw values (the original
harness run printed only aggregate stats -- it never persisted per-graph
data to disk, so nothing here can be answered by re-reading old output):

  1. Does pushing w_security up (optionally removing w_utility entirely)
     close RL's containment gap with the structural methods, or does RL
     underperform regardless of reward weights? Tested via 3 new weight
     configs (high security, security-only, extreme security) x the 5
     structural methods (not random -- that comparison isn't in question).
  2. Is RL's frozen-utility edge (2nd-best in the original harness, behind
     only the greedy simulation-guided heuristic) statistically real, or
     within noise? Tested via the ORIGINAL (baseline-weight) RL policy vs.
     each of the 6 cheap-to-recompute methods, on frozen_f1.

Scope trim: the greedy simulation-guided heuristic (41.49s/graph, the single most expensive
method in the original harness) is NOT recomputed here -- neither question
needs it, and re-deriving it would add ~27 minutes for no analytical
benefit. The other 6 static/GAT methods ARE recomputed fresh (cheap, and
deterministic given the same graph_seed/model_seed, so they reproduce the
original harness's numbers -- serving as an implicit consistency check).
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
from src.milestone4 import EVAL_INFECTION_SEED_BASE, run_paired_tests_with_correction

N_GRAPHS = 40
TARGET_LEVEL = 0.50
N_RANDOM_REPEATS = 3
RL_EPISODES = 100
RL_EPSILON_DECAY_EPISODES = 70
MODEL_SEED = 0

RL_BASELINE = "RL baseline (sec=1.0 util=1.0 cost=0.1)"
RL_HIGH_SEC = "RL high-security (sec=3.0 util=1.0 cost=0.1)"
RL_VERY_HIGH_SEC = "RL very-high-security (sec=10.0 util=1.0 cost=0.1)"
RL_SEC_ONLY = "RL security-only (sec=1.0 util=0.0 cost=0.1)"

# Config choice revised after smoke-testing (see NOTES.md): moderate security
# boost (3.0) helps on the spot-check graph; AGGRESSIVE security boost (5.0,
# 10.0) collapses training to "never prune" REGARDLESS of whether w_utility
# is kept -- a scale/stability failure, not specifically about removing the
# utility incentive. These 4 configs probe both dimensions independently:
# moderate vs. aggressive security weight (both with utility kept), and
# utility removed entirely at baseline security scale.
WEIGHT_CONFIGS = {
    RL_BASELINE: dict(w_security=1.0, w_utility=1.0, w_cost=0.1),
    RL_HIGH_SEC: dict(w_security=3.0, w_utility=1.0, w_cost=0.1),
    RL_VERY_HIGH_SEC: dict(w_security=10.0, w_utility=1.0, w_cost=0.1),
    RL_SEC_ONLY: dict(w_security=1.0, w_utility=0.0, w_cost=0.1),
}

STATIC_METHODS = ["GAT threshold", "GAT top-k", "degree-centrality", "betweenness-centrality", "highest-p_uv", "random"]
ALL_METHODS = STATIC_METHODS + list(WEIGHT_CONFIGS.keys())


def run_one_graph(graph_seed: int) -> dict:
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
    # Two separate baselines, deliberately (same pattern as demo_milestone4_harness.py):
    # `baseline_infected` (default seed base 7) stays paired with PruningEnv's
    # training-time reward signal below, unchanged. `baseline_infected_eval`
    # uses the disjoint EVAL_INFECTION_SEED_BASE so that every method's FINAL
    # containment_ratio (numerator AND denominator) is computed from the SAME
    # held-out seed pool -- mixing a seed-7 baseline with a seed-10007
    # numerator would be an internally inconsistent ratio, not a fair fix.
    baseline_infected = {name: measure_security(graph, p_uv, node) for name, node in seed_nodes.items()}
    baseline_infected_eval = {
        name: measure_security(graph, p_uv, node, infection_seed_base=EVAL_INFECTION_SEED_BASE)
        for name, node in seed_nodes.items()
    }

    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED)
    baseline_utility = measure_utility_frozen(base_model, graph, node_features, labels, test_mask)
    # Supervisor-flagged fix (same as demo_milestone4_harness.py): PruningEnv's
    # reward signal must be measured on a mask disjoint from test_mask, since
    # test_mask is what measure() below uses to report every method's FINAL
    # frozen_recall/frozen_f1 -- reusing test_mask for RL's reward would mean
    # its "held-out" test set was actually seen as reward signal during
    # training. val_mask (already split out by make_node_split, previously
    # unused here) is the correct, disjoint mask for this.
    baseline_utility_reward = measure_utility_frozen(base_model, graph, node_features, labels, val_mask)
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)

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
    measure("GAT threshold", [g], time.time() - t)

    t = time.time()
    _, g, _ = calibrate_topk_for_target_fraction(graph, gat_scores, TARGET_LEVEL)
    measure("GAT top-k", [g], time.time() - t)

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

    for config_name, weights in WEIGHT_CONFIGS.items():
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
            **weights,
        )
        q_network, _ = train_dqn(env, n_episodes=RL_EPISODES, epsilon_decay_episodes=RL_EPSILON_DECAY_EPISODES, seed=0)
        run_greedy_episode(env, q_network)
        rl_graph = env.working_graph.copy()
        measure(config_name, [rl_graph], time.time() - t)

    return results


def main() -> None:
    all_results = {m: {"containment_ratio": [], "frozen_recall": [], "frozen_f1": [], "time": []} for m in ALL_METHODS}

    harness_start = time.time()
    for graph_seed in range(N_GRAPHS):
        graph_start = time.time()
        results = run_one_graph(graph_seed)
        for m in ALL_METHODS:
            all_results[m]["containment_ratio"].append(results[m]["containment_ratio"])
            all_results[m]["frozen_recall"].append(results[m]["frozen_recall"])
            all_results[m]["frozen_f1"].append(results[m]["frozen_f1"])
            all_results[m]["time"].append(results[m]["time"])
        print(f"graph {graph_seed:2d}/{N_GRAPHS} done in {time.time() - graph_start:.1f}s (total elapsed {time.time() - harness_start:.0f}s)", flush=True)

    total_elapsed = time.time() - harness_start
    print(f"\nablation complete: {N_GRAPHS} graphs in {total_elapsed:.0f}s ({total_elapsed / N_GRAPHS:.1f}s/graph average)")

    print("\n=== per-method mean +/- std across 40 graphs ===")
    print(f"{'method':<50}{'containment_ratio':>20}{'frozen_recall':>16}{'frozen_f1':>12}{'mean_time_s':>13}")
    for m in ALL_METHODS:
        c = all_results[m]["containment_ratio"]
        r = all_results[m]["frozen_recall"]
        f = all_results[m]["frozen_f1"]
        t = all_results[m]["time"]
        print(
            f"{m:<50}{statistics.mean(c):>10.3f}+/-{statistics.stdev(c):.3f}"
            f"{statistics.mean(r):>16.3f}{statistics.mean(f):>12.3f}{statistics.mean(t):>13.2f}"
        )

    print("\n=== utility paired tests: RL baseline vs. each cheap method, on frozen_f1 (Holm-corrected) ===")
    utility_pairs = [(RL_BASELINE, m) for m in STATIC_METHODS]
    utility_comparisons = run_paired_tests_with_correction(utility_pairs, {m: all_results[m]["frozen_f1"] for m in ALL_METHODS})
    print(f"{'A vs B':<45}{'mean_A':>8}{'mean_B':>8}{'diff':>8}{'raw_p':>10}{'holm_p':>10}{'sig?':>6}")
    for c in utility_comparisons:
        label = f"{c['method_a']} vs {c['method_b']}"
        print(f"{label:<45}{c['mean_a']:>8.3f}{c['mean_b']:>8.3f}{c['mean_diff']:>8.3f}{c['p_value']:>10.4f}{c['p_value_holm']:>10.4f}{str(c['significant_holm']):>6}")

    print("\n=== containment gap tests: each NEW RL weight config vs. each structural method (Holm-corrected) ===")
    gap_pairs = [(config, m) for config in [RL_HIGH_SEC, RL_VERY_HIGH_SEC, RL_SEC_ONLY] for m in STATIC_METHODS if m != "random"]
    gap_comparisons = run_paired_tests_with_correction(gap_pairs, {m: all_results[m]["containment_ratio"] for m in ALL_METHODS})
    print(f"{'A vs B':<55}{'mean_A':>8}{'mean_B':>8}{'diff':>8}{'raw_p':>10}{'holm_p':>10}{'sig?':>6}")
    for c in gap_comparisons:
        label = f"{c['method_a']} vs {c['method_b']}"
        print(f"{label:<55}{c['mean_a']:>8.3f}{c['mean_b']:>8.3f}{c['mean_diff']:>8.3f}{c['p_value']:>10.4f}{c['p_value_holm']:>10.4f}{str(c['significant_holm']):>6}")

    print("\nNo further interpretation forced here -- see NOTES.md for the read-through.")


if __name__ == "__main__":
    main()
