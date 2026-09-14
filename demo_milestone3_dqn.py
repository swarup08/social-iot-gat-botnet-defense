"""Milestone 3 demo: train the DQN, with monitoring built in from the start.

Reports, per this project's own instrumentation-first discipline:
  1. What fraction of replay-buffer transitions come from deep trajectories
     (>40% pruned), tracked throughout training, not just at the end.
  2. Whether the epsilon-greedy schedule reaches deep pruning BEFORE epsilon
     decays too low to explore much further (first_deep_episode vs.
     low_epsilon_episode).
  3. Whether the learned policy ever chooses STOP before the removal budget,
     both during training (behavior policy) and in the final greedy rollout.

If the final greedy policy never stops early, this does NOT guess why --
it checks the Q-network's own STOP-vs-continue estimate at ACTUAL deep
states from the replay buffer (proof those states were trained on) to tell
exploration failure apart from a genuine learned preference.
"""

import statistics

from src.milestone3 import ACTION_SIZE, STOP_ACTION, build_pruning_env
from src.milestone3_dqn import inspect_deep_state_q_values, run_greedy_episode, train_dqn

N_EPISODES = 300
DEEP_THRESHOLD = 0.4
LOW_EPSILON_THRESHOLD = 0.2


def main() -> None:
    print("building environment (graph, labels, frozen base GAT, baselines)...")
    env = build_pruning_env(n_nodes=300, model_seed=0, max_steps=15, max_removal_fraction=0.5, chunk_fraction=0.05)
    print(f"w_security={env.w_security} w_utility={env.w_utility} w_cost={env.w_cost}")

    print(f"\ntraining DQN for {N_EPISODES} episodes...")
    q_network, log = train_dqn(
        env,
        n_episodes=N_EPISODES,
        epsilon_decay_episodes=200,
        deep_threshold=DEEP_THRESHOLD,
        low_epsilon_threshold=LOW_EPSILON_THRESHOLD,
        seed=0,
    )

    # ---- Monitoring report ----
    print("\n=== monitoring: replay buffer coverage of deep trajectories ===")
    for episode_idx, fraction in log["buffer_deep_fraction_checkpoints"]:
        print(f"  episode {episode_idx:4d}: buffer fraction from deep (>({DEEP_THRESHOLD:.0%}) trajectories = {fraction:.3f}")
    print(f"  final buffer deep-fraction: {log['final_buffer_deep_fraction']:.3f}")

    print("\n=== monitoring: exploration schedule vs. deep-trajectory discovery ===")
    print(f"  first episode whose trajectory reached >{DEEP_THRESHOLD:.0%} removed: {log['first_deep_episode']}")
    print(f"  first episode where epsilon dropped to <= {LOW_EPSILON_THRESHOLD}: {log['low_epsilon_episode']}")
    if log["first_deep_episode"] is not None and log["low_epsilon_episode"] is not None:
        margin = log["low_epsilon_episode"] - log["first_deep_episode"]
        print(f"  margin: deep trajectories were first seen {margin} episodes BEFORE epsilon got low" if margin > 0 else f"  WARNING: epsilon got low {-margin} episodes BEFORE any deep trajectory was seen")

    n_stopped_early_during_training = sum(log["episode_stopped_early"])
    print(f"\n=== monitoring: STOP-before-budget, during training (behavior policy) ===")
    print(f"  episodes where the agent chose STOP before the removal budget: {n_stopped_early_during_training} / {N_EPISODES}")

    # Supervisor-flagged fix: diagnose whether stopping early actually helped
    # or hurt using the COMPLETE DISCOUNTED return (the agent's real
    # objective, gamma-weighted like its Bellman target) -- not a plain
    # undiscounted sum, which treats a step-1 reward and a step-15 reward as
    # equally important when the agent itself does not.
    stopped_returns = [g for g, stopped in zip(log["episode_discounted_return"], log["episode_stopped_early"]) if stopped]
    continued_returns = [g for g, stopped in zip(log["episode_discounted_return"], log["episode_stopped_early"]) if not stopped]
    print(f"\n=== monitoring: STOP vs. continue, complete discounted return ===")
    if stopped_returns and continued_returns:
        print(f"  mean discounted return, STOPPED-early episodes ({len(stopped_returns)}): {statistics.mean(stopped_returns):.3f}")
        print(f"  mean discounted return, ran-to-budget episodes ({len(continued_returns)}): {statistics.mean(continued_returns):.3f}")
    elif stopped_returns:
        print(f"  all {len(stopped_returns)} episodes stopped early -- no ran-to-budget episodes to compare against")
    else:
        print("  no episodes stopped early -- no STOP-vs-continue discounted-return comparison possible")

    mean_return_last_20 = statistics.mean(log["episode_return"][-20:])
    mean_return_first_20 = statistics.mean(log["episode_return"][:20])
    print(f"\n=== training summary ===")
    print(f"  mean episode return, first 20 episodes: {mean_return_first_20:.3f}")
    print(f"  mean episode return, last 20 episodes:  {mean_return_last_20:.3f}")
    print(f"  mean TD loss, last 50 updates: {statistics.mean(log['loss'][-50:]):.4f}")

    # ---- Final greedy policy rollout ----
    print("\n=== final greedy (epsilon=0) policy rollout ===")
    result = run_greedy_episode(env, q_network)
    for i, step_info in enumerate(result["trajectory"]):
        action_name = "STOP" if step_info["action"] == STOP_ACTION else f"prune bucket {step_info['action']}"
        print(
            f"  step {i:2d}: action={action_name:<16} removed={step_info['cumulative_removed_fraction']:.3f} "
            f"containment={step_info['containment_ratio']:.3f} utility={step_info['utility_ratio']:.3f} reward={step_info['reward']:.3f}"
        )
    print(f"  stopped before budget: {result['stopped_early']}  final_removed: {result['final_removed']:.3f}")

    # ---- Diagnose, don't guess, if the policy never stops early ----
    if not result["stopped_early"]:
        print("\n=== DIAGNOSIS: trained policy never stopped early -- checking WHY ===")
        deep_samples = inspect_deep_state_q_values(q_network, log["replay_buffer"], deep_threshold=DEEP_THRESHOLD, n_samples=8)
        if not deep_samples:
            print(
                f"  -> Replay buffer contains ZERO transitions with >{DEEP_THRESHOLD:.0%} already removed. "
                "This IS an exploration gap: the network was never trained on this region at all, "
                "so it has no basis to prefer STOP there. Fix: extend epsilon decay, increase n_episodes, "
                "or increase max_removal_fraction relative to chunk_fraction so deep states are reached sooner."
            )
        else:
            n_stop_favored = sum(1 for s in deep_samples if s["q_stop"] > s["q_best_non_stop"])
            print(f"  Replay buffer HAS {len(log['replay_buffer'])} total transitions, {log['final_buffer_deep_fraction']:.1%} of them deep.")
            print(f"  Sampled {len(deep_samples)} actual deep transitions the network trained on:")
            for s in deep_samples:
                favors = "STOP" if s["q_stop"] > s["q_best_non_stop"] else "continue"
                print(
                    f"    removed={s['cumulative_removed_fraction']:.3f}  Q(STOP)={s['q_stop']:.3f}  "
                    f"Q(best continue)={s['q_best_non_stop']:.3f}  network favors: {favors}  "
                    f"(observed reward at collection time: {s['reward_observed']:.3f})"
                )
            if n_stop_favored == 0:
                print(
                    f"\n  -> VERDICT: GENUINE LEARNED PREFERENCE, not an exploration gap. The network WAS "
                    f"trained on deep states ({log['final_buffer_deep_fraction']:.1%} of the buffer) and its own "
                    "Q-values consistently favor continuing to prune over STOP at every deep state checked. "
                    "Given the current reward weights, the network has concluded further pruning is better in "
                    "expectation even this deep -- this points at the reward weights (w_cost too low relative "
                    "to w_security/w_utility), not at insufficient training or exploration."
                )
            else:
                print(
                    f"\n  -> VERDICT: MIXED -- the network favors STOP at {n_stop_favored}/{len(deep_samples)} sampled "
                    "deep states, but the greedy rollout never actually reached/chose it this episode. "
                    "Worth checking the specific rollout trajectory above against these sampled states."
                )


if __name__ == "__main__":
    main()
