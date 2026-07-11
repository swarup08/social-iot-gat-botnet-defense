"""Milestone 3 demo: one full PruningEnv episode, mechanics only (no learning).

Runs a random policy (fixed seed, for reproducibility) through the
environment built on the standard n=300 graph, printing each step's action,
containment ratio, utility ratio, and reward, plus the final outcome. This
is a sanity check that state/action/reward mechanics behave sensibly before
building the DQN training loop on top -- not a claim about policy quality.
"""

import random

from src.milestone3 import ACTION_SIZE, STOP_ACTION, build_pruning_env

POLICY_SEED = 0
ACTION_NAMES = [f"prune bucket {b} (lowest s_uv first)" for b in range(ACTION_SIZE - 1)] + ["STOP"]


def main() -> None:
    print("building environment (graph, labels, frozen base GAT, baselines)...")
    env = build_pruning_env(n_nodes=300, model_seed=0, max_steps=15, max_removal_fraction=0.5, chunk_fraction=0.05)

    print(f"original graph: {env.original_graph.number_of_nodes()} nodes, {env.original_graph.number_of_edges()} edges")
    print(f"action_size={env.action_size}  state_size={env.state_size}  chunk_size={env.chunk_size} edges/action")
    print(f"bucket_original_counts (quintiles, lowest to highest s_uv): {env.bucket_original_counts}")
    print(f"seed_nodes: {env.seed_nodes}")
    print(f"baseline_infected: { {k: round(v, 3) for k, v in env.baseline_infected.items()} }")
    print(f"baseline_utility recall: {env.baseline_utility['recall']:.3f}")

    state = env.reset()
    print(f"\nepisode start: containment_ratio={env._current_containment_ratio:.3f}  utility_ratio={env._current_utility_ratio:.3f}")
    print("(both should read exactly 1.0 -- nothing pruned yet, matches the unpruned baseline exactly post-fix)\n")

    rng = random.Random(POLICY_SEED)
    print(f"{'step':>4} {'action':<32} {'edges_removed':>13} {'cum_removed':>11} {'containment':>11} {'utility':>7} {'reward':>7}")
    done = False
    step = 0
    while not done:
        valid_actions = [a for a in range(env.action_size) if env.action_mask()[a]]
        action = rng.choice(valid_actions)
        next_state, reward, done, info = env.step(action)
        n_removed_this_step = len(env.trajectory_log[-1]["edges_removed"])
        print(
            f"{step:>4} {ACTION_NAMES[action]:<32} {n_removed_this_step:>13} "
            f"{info['cumulative_removed_fraction']:>11.3f} {env._current_containment_ratio:>11.3f} "
            f"{env._current_utility_ratio:>7.3f} {reward:>7.3f}"
        )
        step += 1

    print(f"\nepisode ended after {step} steps (action={'STOP' if env.trajectory_log[-1]['action'] == STOP_ACTION else 'max_steps/budget reached'})")
    print(f"final edges remaining: {env.working_graph.number_of_edges()} / {env.original_graph.number_of_edges()}")
    print(f"final cumulative_removed_fraction: {env.cumulative_removed_fraction:.3f}")
    print(f"final containment_ratio: {env._current_containment_ratio:.3f}  (lower = more contained)")
    print(f"final utility_ratio: {env._current_utility_ratio:.3f}  (1.0 = matches unpruned baseline)")
    print(f"trajectory_log has {len(env.trajectory_log)} entries, {sum(len(e['edges_removed']) for e in env.trajectory_log)} edges removed total")


if __name__ == "__main__":
    main()
