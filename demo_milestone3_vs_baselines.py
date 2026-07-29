"""Milestone 3 acceptance-criteria comparison: RL-based adaptive pruning vs.
Milestone 2's static baselines, at the same ~50% removal level.

This is the roadmap's "Compare RL-based pruning versus static thresholds in
terms of infection metrics and task performance" task.

Reports BOTH utility definitions for every method, not just one, per the
NOTES.md plan recorded when the RL environment was designed: Milestone 2's
baselines were originally scored with RETRAINED utility (measure_utility,
~3.6s/call, a fresh GAT per pruned graph); Milestone 3's RL reward uses
FROZEN utility (measure_utility_frozen, ~1ms/call, the already-trained base
GAT's forward pass, no retraining) for tractability. Comparing RL's numbers
against Milestone 2's old retrained-utility table directly would not be
apples-to-apples -- so every method here, including the RL policy, is scored
on BOTH definitions using the exact SAME frozen base_model and the exact
same retraining procedure, so the comparison is consistent either way you
want to read it.

Containment ratio uses the identical 5-seed x 15-rollout methodology as
every other security measurement in this project (measure_security +
containment_ratio) -- not a separate metric invented for RL.
"""

import statistics

from src.milestone2 import make_node_split
from src.milestone2_pruning import (
    calibrate_topk_for_target_fraction,
    containment_ratio,
    greedy_simulation_guided_prune,
    measure_security,
    measure_utility,
    measure_utility_frozen,
    prune_highest_score,
    prune_lowest_score,
    prune_random,
    score_edges_by_avg_hub_score,
    score_edges_by_betweenness,
)
from src.milestone3 import build_pruning_env
from src.milestone3_dqn import run_greedy_episode, train_dqn

TARGET_LEVEL = 0.50
N_RANDOM_REPEATS = 3
MODEL_SEED = 0


def report_row(method: str, fraction_removed: float, pruned_graph, env, node_features, labels, test_mask) -> dict:
    # Per-seed ratios kept RAW (not pre-averaged) so callers can pool them
    # across variants/seeds for a proper std -- matching
    # demo_milestone2_pruning.py's "pool every (variant x seed) ratio" approach,
    # not a std computed across variant-count alone (which is meaningless for
    # any method with only 1 variant, and was a bug in an earlier version of
    # this script).
    ratios = [
        containment_ratio(measure_security(pruned_graph, env.p_uv, node), env.baseline_infected[name])
        for name, node in env.seed_nodes.items()
    ]

    frozen = measure_utility_frozen(env.base_model, pruned_graph, node_features, labels, test_mask)
    retrained = measure_utility(pruned_graph, node_features, labels, env.train_mask, test_mask, model_seed=MODEL_SEED)

    return {
        "method": method,
        "fraction_removed": fraction_removed,
        "containment_ratios": ratios,
        "frozen_recall": frozen["recall"],
        "frozen_f1": frozen["f1"],
        "retrained_recall": retrained["recall"],
        "retrained_f1": retrained["f1"],
    }


def main() -> None:
    print("building environment and training RL policy (this also fixes the environment's graph/labels/model)...")
    env = build_pruning_env(n_nodes=300, model_seed=MODEL_SEED, max_steps=15, max_removal_fraction=0.5, chunk_fraction=0.05)
    # measure_utility (retrained) needs a train_mask; build_pruning_env doesn't
    # keep one on the env, so rebuild the identical split here for that purpose only.
    train_mask, val_mask, test_mask = make_node_split(env.original_graph.number_of_nodes(), seed=0)
    env.train_mask = train_mask

    q_network, log = train_dqn(env, n_episodes=300, epsilon_decay_episodes=200, seed=0)
    rl_result = run_greedy_episode(env, q_network)
    rl_graph = env.working_graph.copy()
    rl_fraction = rl_result["final_removed"]
    print(f"RL policy final pruned graph: {rl_fraction:.3f} fraction removed ({rl_graph.number_of_edges()} / {env.original_graph.number_of_edges()} edges remain)")

    print(f"\nbuilding Milestone 2 baseline pruned graphs at the matching ~{TARGET_LEVEL:.0%} level...")
    degree_scores = score_edges_by_avg_hub_score(env.original_graph, env.node_features)
    betweenness_scores = score_edges_by_betweenness(env.original_graph)

    pruned_by_method = {
        "RL adaptive pruning (DQN)": [rl_graph],
        "GAT threshold": [prune_lowest_score(env.original_graph, env.gat_scores, TARGET_LEVEL)],
        "GAT top-k-per-node": [calibrate_topk_for_target_fraction(env.original_graph, env.gat_scores, TARGET_LEVEL)[1]],
        "degree-centrality": [prune_highest_score(env.original_graph, degree_scores, TARGET_LEVEL)],
        "betweenness-centrality": [prune_highest_score(env.original_graph, betweenness_scores, TARGET_LEVEL)],
        "highest-p_uv": [prune_highest_score(env.original_graph, env.p_uv, TARGET_LEVEL)],
        "random": [prune_random(env.original_graph, TARGET_LEVEL, seed=1000 * i + 7) for i in range(N_RANDOM_REPEATS)],
    }

    print("running greedy simulation-guided heuristic search to the 50% checkpoint (~25-30s)...")
    greedy_checkpoints = greedy_simulation_guided_prune(
        env.original_graph, env.p_uv, env.seed_nodes, max_remove_fraction=TARGET_LEVEL, checkpoint_fractions=[TARGET_LEVEL], search_rollouts=1
    )
    pruned_by_method["greedy simulation-guided heuristic"] = [greedy_checkpoints[TARGET_LEVEL]]

    print("\nmeasuring containment ratio + BOTH utility definitions for every method...")
    results = []
    for method, variants in pruned_by_method.items():
        rows = [report_row(method, 1 - g.number_of_edges() / env.original_graph.number_of_edges(), g, env, env.node_features, env.labels, test_mask) for g in variants]
        # Pool every (variant x seed) containment ratio into one list -- the
        # "average over multiple infection-seed choices" discipline applies
        # here exactly as it does everywhere else in this project.
        pooled_ratios = [ratio for r in rows for ratio in r["containment_ratios"]]
        merged = {
            "method": method,
            "fraction_removed": statistics.mean(r["fraction_removed"] for r in rows),
            "containment_ratio": statistics.mean(pooled_ratios),
            "containment_ratio_std": statistics.stdev(pooled_ratios) if len(pooled_ratios) > 1 else 0.0,
            "frozen_recall": statistics.mean(r["frozen_recall"] for r in rows),
            "frozen_f1": statistics.mean(r["frozen_f1"] for r in rows),
            "retrained_recall": statistics.mean(r["retrained_recall"] for r in rows),
            "retrained_f1": statistics.mean(r["retrained_f1"] for r in rows),
        }
        results.append(merged)
        print(f"  {method:<28} done")

    print("\n\n=== Milestone 3 acceptance comparison: RL vs. static baselines (~50% removal) ===")
    header = f"{'method':<28}{'removed':>9}{'contain_ratio':>15}{'frozen_recall':>14}{'frozen_f1':>11}{'retrain_recall':>15}{'retrain_f1':>11}"
    print(header)
    for r in results:
        print(
            f"{r['method']:<28}{r['fraction_removed']:>9.3f}{r['containment_ratio']:>10.3f}+/-{r['containment_ratio_std']:.2f}"
            f"{r['frozen_recall']:>14.3f}{r['frozen_f1']:>11.3f}{r['retrained_recall']:>15.3f}{r['retrained_f1']:>11.3f}"
        )
    print(
        "\nNo conclusion drawn here about which method 'wins' -- single graph, single RL training run, "
        "same caveats as every other single-graph comparison in this project. See NOTES.md."
    )


if __name__ == "__main__":
    main()
