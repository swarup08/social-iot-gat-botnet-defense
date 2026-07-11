"""Milestone 3 pre-training diagnostic: does the reward ever favor STOPping?

Runs N_EPISODES full episodes with DIFFERENT random action policies (same
fixed graph, same environment -- only the action sequence varies) and checks
whether utility_ratio/reward consistently increase with more pruning across
episodes, or whether that was a one-off property of a single trajectory.

Why this matters before building the DQN: if reward NEVER favors stopping
under any of these random trajectories, a DQN trained on the current reward
weights will trivially learn "prune to the budget cap every time" -- not an
interesting adaptive policy, and evidence that w_cost needs to go up (or
w_utility down) before spending time on the training loop. If the pattern
varies by trajectory (sometimes utility drops enough that reward would have
preferred stopping earlier), the current weights may already be fine and
this is just something to monitor during training instead.
"""

import random

from src.milestone3 import build_pruning_env

N_EPISODES = 8
BASE_SEED = 100  # offset from the single-episode demo's seed=0, so this is a genuinely different sweep


def run_episode(env, policy_seed: int) -> dict:
    rng = random.Random(policy_seed)
    env.reset()

    reward_trajectory = []
    utility_trajectory = []
    containment_trajectory = []
    removed_trajectory = []

    done = False
    while not done:
        valid_actions = [a for a in range(env.action_size) if env.action_mask()[a]]
        action = rng.choice(valid_actions)
        _, reward, done, info = env.step(action)
        reward_trajectory.append(reward)
        utility_trajectory.append(env._current_utility_ratio)
        containment_trajectory.append(env._current_containment_ratio)
        removed_trajectory.append(info["cumulative_removed_fraction"])

    # Did reward ever DECREASE step-to-step? If so, stopping at the earlier
    # (higher-reward) step would have scored better than continuing -- direct
    # evidence STOP can be favorable somewhere in this trajectory.
    reward_ever_decreased = any(reward_trajectory[i] < reward_trajectory[i - 1] for i in range(1, len(reward_trajectory)))
    utility_ever_below_baseline = any(u < 1.0 for u in utility_trajectory)
    best_step = max(range(len(reward_trajectory)), key=lambda i: reward_trajectory[i])

    return {
        "policy_seed": policy_seed,
        "n_steps": len(reward_trajectory),
        "final_removed": removed_trajectory[-1],
        "final_reward": reward_trajectory[-1],
        "final_utility_ratio": utility_trajectory[-1],
        "final_containment_ratio": containment_trajectory[-1],
        "reward_ever_decreased": reward_ever_decreased,
        "utility_ever_below_baseline": utility_ever_below_baseline,
        "best_step_is_last_step": best_step == len(reward_trajectory) - 1,
        "max_reward": reward_trajectory[best_step],
        "reward_trajectory": reward_trajectory,
        "utility_trajectory": utility_trajectory,
    }


def main() -> None:
    print("building environment (graph, labels, frozen base GAT, baselines)...")
    env = build_pruning_env(n_nodes=300, model_seed=0, max_steps=15, max_removal_fraction=0.5, chunk_fraction=0.05)
    print(f"w_security={env.w_security} w_utility={env.w_utility} w_cost={env.w_cost}\n")

    results = []
    print(f"{'seed':>5}{'steps':>6}{'removed':>9}{'final_util':>11}{'final_rwd':>10}{'max_rwd':>9}{'best=last?':>11}{'rwd_dipped?':>12}{'util<1?':>9}")
    for i in range(N_EPISODES):
        seed = BASE_SEED + i
        result = run_episode(env, seed)
        results.append(result)
        print(
            f"{seed:>5}{result['n_steps']:>6}{result['final_removed']:>9.3f}{result['final_utility_ratio']:>11.3f}"
            f"{result['final_reward']:>10.3f}{result['max_reward']:>9.3f}"
            f"{str(result['best_step_is_last_step']):>11}{str(result['reward_ever_decreased']):>12}"
            f"{str(result['utility_ever_below_baseline']):>9}"
        )

    n_reward_dipped = sum(r["reward_ever_decreased"] for r in results)
    n_best_is_last = sum(r["best_step_is_last_step"] for r in results)
    n_util_below_baseline = sum(r["utility_ever_below_baseline"] for r in results)

    print(f"\nacross {N_EPISODES} episodes (different random action sequences, same graph):")
    print(f"  reward decreased at some step (stopping earlier would have scored higher): {n_reward_dipped}/{N_EPISODES}")
    print(f"  the LAST step had the highest reward in the episode (never beneficial to stop early): {n_best_is_last}/{N_EPISODES}")
    print(f"  utility_ratio dropped below 1.0 (below baseline) at some point: {n_util_below_baseline}/{N_EPISODES}")

    if n_best_is_last == N_EPISODES:
        print(
            "\n  -> VERDICT: reward NEVER favored stopping early, in ANY episode. This is not a "
            "one-off coincidence of the earlier single-episode demo. Under the current weights "
            "(w_security=1.0, w_utility=1.0, w_cost=0.1), a DQN would trivially learn 'always prune "
            "to the budget cap' -- w_cost should be increased (or w_utility reduced) before training."
        )
    elif n_best_is_last == 0:
        print("\n  -> VERDICT: stopping early was favorable in EVERY episode -- current weights look fine, monitor during training.")
    else:
        print(
            f"\n  -> VERDICT: MIXED -- stopping early was favorable in {N_EPISODES - n_best_is_last}/{N_EPISODES} episodes "
            "but not the others. The pattern is trajectory-dependent, not universal; current weights may be usable "
            "as a starting point, with this monitored during training rather than retuned blind."
        )


if __name__ == "__main__":
    main()
