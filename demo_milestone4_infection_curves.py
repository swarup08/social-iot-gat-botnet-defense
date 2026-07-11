"""Final write-up figure: infection curves over time, no pruning vs. the best
static method vs. RL, at a fixed (~50%) pruning level.

This is the dynamics-over-time companion to the containment_ratio tables used
everywhere else in this project: those compare only the FINAL infected
fraction once an outbreak burns out, this figure shows HOW infection spreads
round-by-round under each method.

Reuses the exact same n=300 graph/model setup as demo_milestone3_vs_baselines.py
(build_pruning_env), so the RL policy and pruned graphs here are the same
objects/definitions used in that comparison and throughout Milestone 3/4.
"degree-centrality" is used as "the best static method" because it has the
lowest mean containment_ratio in the 40-graph harness (0.033, see NOTES.md's
Milestone 4 harness RESULTS entry) -- not re-derived here, just cited.

Infection is seeded from the hub node only (not averaged across all 5 seed
nodes used for containment_ratio elsewhere) -- a single, illustrative seed
choice makes for one clean curve per method; N_ROLLOUTS rollouts from that
seed are averaged to smooth out simulate_botnet's per-rollout randomness.
"""

import statistics
import time

from src.milestone1 import plot_infection_curves_comparison, simulate_botnet
from src.milestone2_pruning import prune_highest_score, score_edges_by_avg_hub_score
from src.milestone3 import build_pruning_env
from src.milestone3_dqn import run_greedy_episode, train_dqn

TARGET_LEVEL = 0.50
N_ROLLOUTS = 30
MODEL_SEED = 0
RL_EPISODES = 300  # matches demo_milestone3_vs_baselines.py's rigor for this final comparison


def average_infection_curve(graph, p_uv, seed_node: int, n_rollouts: int = N_ROLLOUTS, infection_seed_base: int = 7) -> list:
    """Run n_rollouts simulate_botnet rollouts from seed_node and return the
    round-by-round MEAN infected count. Rollouts that settle (frontier empties)
    in fewer rounds than the longest one are padded by holding their final,
    already-settled count constant -- so every rollout contributes to every
    round up to the longest observed, without pretending infection kept
    spreading after it actually stopped.
    """
    histories = [
        simulate_botnet(graph, p_uv, initial_compromised={seed_node}, seed=infection_seed_base + i)["infection_history"]
        for i in range(n_rollouts)
    ]
    max_len = max(len(h) for h in histories)
    padded = [h + [h[-1]] * (max_len - len(h)) for h in histories]
    return [statistics.mean(round_values) for round_values in zip(*padded)]


def main() -> None:
    start = time.time()
    print("building environment and training RL policy (this fixes the graph/labels/model)...")
    env = build_pruning_env(n_nodes=300, model_seed=MODEL_SEED, max_steps=15, max_removal_fraction=TARGET_LEVEL, chunk_fraction=0.05)
    q_network, _ = train_dqn(env, n_episodes=RL_EPISODES, epsilon_decay_episodes=200, seed=0)
    rl_result = run_greedy_episode(env, q_network)
    rl_graph = env.working_graph.copy()
    print(f"RL final removed fraction: {rl_result['final_removed']:.3f} (elapsed {time.time() - start:.0f}s)")

    degree_scores = score_edges_by_avg_hub_score(env.original_graph, env.node_features)
    static_graph = prune_highest_score(env.original_graph, degree_scores, TARGET_LEVEL)
    static_removed = 1 - static_graph.number_of_edges() / env.original_graph.number_of_edges()
    print(f"degree-centrality (best static method) removed fraction: {static_removed:.3f}")

    hub_node = env.seed_nodes["hub"]
    total_nodes = env.original_graph.number_of_nodes()
    print(f"\nrunning {N_ROLLOUTS} rollouts per graph from the hub seed node (node {hub_node})...")

    curves = {
        "no pruning": average_infection_curve(env.original_graph, env.p_uv, hub_node),
        f"degree-centrality, best static ({static_removed:.0%} removed)": average_infection_curve(static_graph, env.p_uv, hub_node),
        f"RL adaptive pruning ({rl_result['final_removed']:.0%} removed)": average_infection_curve(rl_graph, env.p_uv, hub_node),
    }

    print(f"\n{'method':<45}{'final mean infected':>22}{'final infected fraction':>26}")
    for label, curve in curves.items():
        print(f"{label:<45}{curve[-1]:>22.1f}{curve[-1] / total_nodes:>26.3f}")

    path = plot_infection_curves_comparison(curves, total_nodes=total_nodes, output_path="infection_curves_comparison.png")
    print(f"\nsaved {path} (total elapsed {time.time() - start:.0f}s)")
    print(
        "\nSingle graph, single RL training run, hub-seeded only -- same caveats as "
        "every other single-graph illustration in this project; the multi-graph, "
        "multi-seed containment_ratio harness results are the statistically-backed "
        "numbers, this figure is the roadmap's requested dynamics-over-time view."
    )


if __name__ == "__main__":
    main()
