"""Reward diagnostic control experiments -- reviewer comment "REWARD
DIAGNOSTIC: early stopping does not establish training collapse" on the
Applied Network Science submission.

THE REVIEWER'S CONCERN: the environment (src/milestone3.py's PruningEnv)
accumulates R_t = -w_security*containment_ratio_t + w_utility*utility_ratio_t
- w_cost*cumulative_removed_fraction_t at EVERY step, with gamma=0.95 and a
terminal STOP action. The paper's current framing ("security-heavy configs
destabilize/collapse training") compared STOP's IMMEDIATE reward against a
continuing trajectory's immediate reward (see NOTES.md's 2026-07-11 entry) --
insufficient, since the policy optimizes the DISCOUNTED SUM over the whole
trajectory, not any single step's reward. Two things are needed before that
framing is defensible:
  (a) compare STOP's return against the COMPLETE discounted return of a
      fixed (non-learned) continuing trajectory -- Part A below.
  (b) rule out that incremental reward delivery itself (as opposed to the
      reward WEIGHTS) is what's producing the collapse, via a terminal-
      objective control -- Part B below.
Log achieved removal fractions and STOP frequencies throughout -- Part C.

PART A -- G_STOP vs. G_FIXED, CHECKPOINTED ALONG THE WHOLE TRAJECTORY.
For each of the 4 existing weight configs (demo_milestone4_reward_ablation.py)
on N_GRAPHS=10 graphs:
  G_STOP: the discounted return of choosing STOP at step 0. A single-step
    episode, so G_STOP = gamma**0 * r_0 = r_0 -- computed by actually calling
    env.reset()/env.step(STOP_ACTION) (reusing the real reward path), not by
    hand-deriving the formula's value.
  G_FIXED(t): the discounted return of a FIXED, non-learned, deterministic
    schedule -- at each step, prune the highest-index non-empty attention
    bucket (largest remaining s_uv) -- RECORDED AT EVERY STEP t along the
    way (the partial sum up to and including step t), not only at the
    trajectory's end. The environment's own `done` condition
    (cumulative_removed_fraction >= max_removal_fraction) fires automatically
    on the step that crosses the budget, so this schedule never needs to
    issue an explicit terminal STOP action -- and never touches the trained/
    exploring DQN policy, so the comparison isn't contaminated by whether
    training succeeded.
Checkpointing every step (not just comparing G_STOP to the single full-
budget G_FIXED) is required because the reviewer's suggested replacement
wording specifically names EPISODE LENGTH as a possible confound: STOP's
apparent advantage could come from being a short episode (fewer discounted
negative terms), not from continuing genuinely scoring worse once it
reaches a comparable point. Comparing only the endpoint can't distinguish
"continuing never catches up" from "continuing crosses over early, but the
full-budget comparison happened to land on a bad final step" -- reporting
whether G_FIXED(t) EVER exceeds G_STOP, and at which removal fraction it
first does, answers that directly. If it never crosses over, continuing is
reward-dispreferred at every length tested, not just at the endpoint --
a stronger, more specific claim than the single-length comparison could
support either way.

If G_FIXED(t) > G_STOP at some t (higher = more reward, since the agent
maximizes the sum), continuing to that length IS reward-preferable to
stopping immediately -- so a trained policy that nonetheless collapses to
STOP is failing to reach an achievable better outcome (a real optimization
gap), not making a genuine reward-optimal choice.

PART B -- terminal-objective control.
TerminalRewardPruningEnv (below) is a thin PruningEnv subclass that pays
ZERO reward at every non-terminal step and the SAME formula (inherited
_compute_reward(), UNCHANGED) once, as a lump sum, on whichever step
actually ends the episode. Isolates whether incremental delivery -- which
the current narrative blames for making STOP's immediately-known value look
prematurely competitive early in training -- is causally responsible for the
security-heavy collapse, or whether an identical total reward produces the
same collapse regardless of how it's paid out over the episode. Retrains
DQN from scratch (same train_dqn/run_greedy_episode, same
RL_EPISODES=100/RL_EPSILON_DECAY_EPISODES=70/gamma=0.95 as
demo_milestone4_reward_ablation.py) under this variant, same 4 configs, same
10 graphs.

If a config's collapse (STOP frequency, achieved removal near 0) PERSISTS
under the terminal variant, that argues the collapse is a property of the
reward WEIGHTS themselves (what's being optimized), not an artifact of
incremental shaping. If it drops substantially, incremental delivery is
causally implicated.

PART C -- logging.
G_STOP and G_FIXED are computed under BOTH reward variants (not just
incremental): they're necessarily IDENTICAL for G_STOP (a 1-step episode
can't distinguish "every step" from "terminal only" -- there is only one
step), but G_FIXED (a ~10-step trajectory) genuinely differs between
variants, extending rather than narrowing the reviewer's requested
comparison. reward_diagnostic_results.csv: one row per (graph_seed, config,
reward_variant), columns graph_seed/config/reward_variant/g_stop/
g_fixed_trajectory (G_FIXED at the full removal budget, for continuity with
the original single-length ask)/fixed_beats_stop (same, at full budget)/
fixed_ever_beats_stop (does G_FIXED(t) cross above G_STOP at ANY checkpoint
along the trajectory)/first_crossover_removed_fraction (the removal
fraction at first crossover, blank if it never crosses)/
trained_final_removed/trained_stopped_early. STOP frequency (fraction of
the 10 graphs where the trained policy's greedy rollout stopped before
budget) is an aggregate over graph_seed, reported in the printed summary
rather than as a CSV column.

A second file, reward_diagnostic_checkpoints.csv, saves the FULL per-step
G_fixed(t) trajectory behind the summary's endpoint/crossover columns --
one row per (graph_seed, config, reward_variant, step_index), columns
graph_seed/config/reward_variant/step_index/removed_fraction/
g_fixed_at_step/g_stop (repeated per row, a convenient reference line for
plotting without a join)/beats_stop_at_step. Every chunk-removal step is
recorded, not just nominal 10/25/50% samples -- the smoke-tested crossover
for RL_SEC_ONLY happened at ~5% removed, below the first nominal 10%
checkpoint, so coarser sampling would have missed it. This lets the "peaks
early then declines" pattern be analyzed at full resolution across every
graph/config/variant after the sweep, without rerunning anything.

Does NOT touch src/milestone3.py, src/milestone3_dqn.py,
demo_milestone4_reward_ablation.py, or any LaTeX file -- diagnostic-only,
for review before any paper text changes.
"""

import csv
import statistics
import time
from typing import Tuple

import numpy as np

from src.milestone1 import build_edge_features, compute_edge_infection_probabilities
from src.milestone2 import (
    DEFAULT_BETA,
    build_pyg_data,
    extract_degree_corrected_attention_scores,
    generate_labeled_graph,
    make_node_split,
    train_gat,
)
from src.milestone2_pruning import measure_security, measure_utility_frozen, pick_seed_nodes
from src.milestone3 import PruningEnv, STOP_ACTION, _get
from src.milestone3_dqn import run_greedy_episode, train_dqn

N_GRAPHS = 10
GAMMA = 0.95
RL_EPISODES = 100
RL_EPSILON_DECAY_EPISODES = 70
MODEL_SEED = 0
TARGET_LEVEL = 0.50
MAX_STEPS = 15
CHUNK_FRACTION = 0.05

RL_BASELINE = "RL baseline (sec=1.0 util=1.0 cost=0.1)"
RL_HIGH_SEC = "RL high-security (sec=3.0 util=1.0 cost=0.1)"
RL_VERY_HIGH_SEC = "RL very-high-security (sec=10.0 util=1.0 cost=0.1)"
RL_SEC_ONLY = "RL security-only (sec=1.0 util=0.0 cost=0.1)"

# Identical to demo_milestone4_reward_ablation.py's WEIGHT_CONFIGS.
WEIGHT_CONFIGS = {
    RL_BASELINE: dict(w_security=1.0, w_utility=1.0, w_cost=0.1),
    RL_HIGH_SEC: dict(w_security=3.0, w_utility=1.0, w_cost=0.1),
    RL_VERY_HIGH_SEC: dict(w_security=10.0, w_utility=1.0, w_cost=0.1),
    RL_SEC_ONLY: dict(w_security=1.0, w_utility=0.0, w_cost=0.1),
}

REWARD_VARIANTS = ["incremental", "terminal"]

OUTPUT_CSV = "reward_diagnostic_results.csv"
CHECKPOINT_CSV = "reward_diagnostic_checkpoints.csv"


class TerminalRewardPruningEnv(PruningEnv):
    """Part B control: pays ZERO reward at every non-terminal step, and the
    SAME formula _compute_reward() already implements (inherited, UNCHANGED)
    as a single lump sum on whichever step actually ends the episode.

    Overrides ONLY step()'s reward-vs-done ORDERING: the base class computes
    reward before done (fine when reward is paid every step, but can't tell
    "terminal" from "non-terminal" that way) -- this override computes done
    FIRST, then gates the reward on it. State, action mask, chunk mechanics,
    and the reward formula itself are all inherited unchanged from
    PruningEnv; only this class exists to implement the control, nothing in
    src/milestone3.py is modified.
    """

    def step(self, action: int):
        assert 0 <= action < self.action_size, f"action {action} out of range [0, {self.action_size})"
        self.step_count += 1
        edges_removed = []

        if action != STOP_ACTION:
            bucket = action
            candidates = [(u, v) for u, v in self.working_graph.edges() if _get(self.bucket_of, u, v) == bucket]
            candidates.sort(key=lambda e: _get(self.gat_scores, *e))
            edges_removed = candidates[: self.chunk_size]
            self.working_graph.remove_edges_from(edges_removed)
            self.cumulative_removed_fraction = 1 - self.working_graph.number_of_edges() / self.original_graph.number_of_edges()

        self._current_containment_ratio, self._current_utility_ratio = self._measure_current()

        done = (
            action == STOP_ACTION
            or self.step_count >= self.max_steps
            or self.cumulative_removed_fraction >= self.max_removal_fraction
            or self.working_graph.number_of_edges() == 0
        )

        # The only behavioral difference from PruningEnv.step(): zero reward
        # on every non-terminal step, full (inherited, unmodified) formula
        # exactly once, on the step that ends the episode.
        reward = self._compute_reward() if done else 0.0

        self.last_action_one_hot = np.zeros(self.action_size, dtype=np.float32)
        self.last_action_one_hot[action] = 1.0
        self.last_reward = reward

        info = {"cumulative_removed_fraction": self.cumulative_removed_fraction}
        return self._compute_state(), reward, done, info


def build_env(env_cls, graph, node_features, labels, p_uv, seed_nodes, gat_scores, base_model, val_mask, baseline_infected, baseline_utility_reward, weights):
    return env_cls(
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
        max_steps=MAX_STEPS,
        max_removal_fraction=TARGET_LEVEL,
        chunk_fraction=CHUNK_FRACTION,
        **weights,
    )


def stop_immediately_return(env) -> float:
    """Part A: G_STOP. A single-step episode, so G_STOP = gamma**0 * r_0 =
    r_0 -- computed via the real env.reset()/env.step(STOP_ACTION) path."""
    env.reset()
    _, reward, done, _ = env.step(STOP_ACTION)
    assert done, "STOP must end the episode in exactly one step"
    return reward


def fixed_continuing_trajectory_checkpoints(env, gamma: float) -> list:
    """Part A (revised per reviewer's episode-length-confound point):
    Deterministic, non-learned schedule -- always prune the highest-index
    remaining attention bucket (largest remaining s_uv) until the
    environment's own `done` fires from crossing the removal budget -- no
    explicit terminal STOP needed.

    Records the discounted return AT EVERY STEP along the way (the partial
    sum sum_{i=0}^{t} gamma**i * r_i, paired with the removed_fraction at
    that point), not just the trajectory's final value. This is free -- same
    trajectory, same simulate_botnet calls, just also keeping the running
    sum -- and it directly answers whether continuing EVER becomes reward-
    preferable over STOP, and at what removal fraction, rather than only
    checking the endpoint (which conflates "continuing scores worse" with
    "STOP is a short episode with fewer discounted penalty terms" -- exactly
    the confound the reviewer's suggested replacement wording names).

    Returns a list of {"removed_fraction": ..., "discounted_return": ...}
    dicts in step order (index 0 = after the first action); the last
    entry's discounted_return is the same "G_FIXED at full budget" value
    the original single-length version reported.
    """
    env.reset()
    checkpoints = []
    running_return = 0.0
    done = False
    step_index = 0
    while not done:
        bucket_mask = env.action_mask()[:-1]  # exclude STOP; True iff that bucket still has edges
        nonzero = np.flatnonzero(bucket_mask)
        # Defensive fallback (shouldn't trigger at TARGET_LEVEL=0.5, since
        # 5 roughly-equal buckets can't all empty before a 50% budget) --
        # if every bucket were somehow exhausted, STOP is the only valid
        # action left.
        action = int(nonzero.max()) if len(nonzero) > 0 else STOP_ACTION
        _, reward, done, info = env.step(action)
        running_return += (gamma ** step_index) * reward
        checkpoints.append({"removed_fraction": info["cumulative_removed_fraction"], "discounted_return": running_return})
        step_index += 1
    return checkpoints


def first_crossover(checkpoints: list, g_stop: float):
    """First checkpoint (if any) where the fixed trajectory's running
    discounted return exceeds G_STOP -- i.e. the removal fraction at which
    continuing FIRST becomes reward-preferable over stopping immediately.
    Returns (ever_crosses: bool, removed_fraction_at_first_crossover: float|None).
    """
    for cp in checkpoints:
        if cp["discounted_return"] > g_stop:
            return True, cp["removed_fraction"]
    return False, None


def run_one_graph(graph_seed: int) -> Tuple[list, list]:
    """Returns (summary_rows, checkpoint_rows).

    summary_rows: one row per (config, reward_variant) -- the per-config
    headline numbers (G_STOP, G_FIXED at full budget, crossover summary,
    trained-policy outcome).

    checkpoint_rows: one row per (config, reward_variant, step) -- the FULL
    per-step G_fixed(t) trajectory (every chunk removed, not just nominal
    10/25/50% samples), so the "peaks early then declines" pattern found
    during smoke-testing can be analyzed at full resolution across every
    graph/config/variant without rerunning. Sampling only at coarse nominal
    checkpoints would risk missing exactly this kind of early crossover --
    the smoke test's crossover for RL_SEC_ONLY happened at ~5% removed,
    BELOW the first nominal 10% checkpoint.
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

    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED)
    # Same val_mask-as-reward-mask fix as demo_milestone4_reward_ablation.py
    # (do not regress it): the reward-time utility baseline must be measured
    # on a mask disjoint from test_mask.
    baseline_utility_reward = measure_utility_frozen(base_model, graph, node_features, labels, val_mask)
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)

    env_classes = {"incremental": PruningEnv, "terminal": TerminalRewardPruningEnv}

    summary_rows = []
    checkpoint_rows = []
    for config_name, weights in WEIGHT_CONFIGS.items():
        for variant_name in REWARD_VARIANTS:
            env = build_env(
                env_classes[variant_name], graph, node_features, labels, p_uv, seed_nodes,
                gat_scores, base_model, val_mask, baseline_infected, baseline_utility_reward, weights,
            )

            # Part A: fixed, non-learned comparisons (this env instance's
            # reward variant applies to both). Checkpointed at every step,
            # not just the full-budget endpoint (reviewer's episode-length-
            # confound point) -- free, since it's the same trajectory/same
            # simulate_botnet calls, just also recording the running sum.
            g_stop = stop_immediately_return(env)
            checkpoints = fixed_continuing_trajectory_checkpoints(env, GAMMA)
            g_fixed_final = checkpoints[-1]["discounted_return"]
            ever_crosses, crossover_fraction = first_crossover(checkpoints, g_stop)

            for step_index, cp in enumerate(checkpoints):
                checkpoint_rows.append(
                    {
                        "graph_seed": graph_seed,
                        "config": config_name,
                        "reward_variant": variant_name,
                        "step_index": step_index,
                        "removed_fraction": cp["removed_fraction"],
                        "g_fixed_at_step": cp["discounted_return"],
                        "g_stop": g_stop,  # repeated per row -- convenient reference line for plotting, no join needed
                        "beats_stop_at_step": cp["discounted_return"] > g_stop,
                    }
                )

            # Part B: train a FRESH policy under this reward variant, then
            # roll out the trained greedy policy once.
            q_network, _ = train_dqn(env, n_episodes=RL_EPISODES, epsilon_decay_episodes=RL_EPSILON_DECAY_EPISODES, gamma=GAMMA, seed=0)
            greedy_result = run_greedy_episode(env, q_network)

            summary_rows.append(
                {
                    "graph_seed": graph_seed,
                    "config": config_name,
                    "reward_variant": variant_name,
                    "g_stop": g_stop,
                    "g_fixed_trajectory": g_fixed_final,
                    "fixed_beats_stop": g_fixed_final > g_stop,
                    "fixed_ever_beats_stop": ever_crosses,
                    "first_crossover_removed_fraction": crossover_fraction,
                    "trained_final_removed": greedy_result["final_removed"],
                    "trained_stopped_early": greedy_result["stopped_early"],
                }
            )
    return summary_rows, checkpoint_rows


def print_summary(all_rows: list) -> None:
    print(f"\n=== PART A: G_STOP vs. G_FIXED at full budget, mean +/- std across {N_GRAPHS} graphs ===")
    print(f"{'config':<42}{'variant':<13}{'G_STOP':>16}{'G_FIXED(full)':>18}{'fixed>stop?':>13}")
    for config_name in WEIGHT_CONFIGS:
        for variant_name in REWARD_VARIANTS:
            subset = [r for r in all_rows if r["config"] == config_name and r["reward_variant"] == variant_name]
            g_stop_vals = [r["g_stop"] for r in subset]
            g_fixed_vals = [r["g_fixed_trajectory"] for r in subset]
            frac_fixed_wins = sum(r["fixed_beats_stop"] for r in subset) / len(subset)
            stop_std = statistics.stdev(g_stop_vals) if len(g_stop_vals) > 1 else 0.0
            fixed_std = statistics.stdev(g_fixed_vals) if len(g_fixed_vals) > 1 else 0.0
            print(
                f"{config_name:<42}{variant_name:<13}"
                f"{statistics.mean(g_stop_vals):>9.3f}+/-{stop_std:<5.3f}"
                f"{statistics.mean(g_fixed_vals):>10.3f}+/-{fixed_std:<5.3f}"
                f"{frac_fixed_wins:>12.0%}"
            )

    print(f"\n=== PART A (revised): does continuing EVER cross over G_STOP before full budget, and where? (n={N_GRAPHS} graphs) ===")
    print(f"{'config':<42}{'variant':<13}{'ever crosses':>13}{'mean crossover @removed':>26}{'(n crossing)':>13}")
    for config_name in WEIGHT_CONFIGS:
        for variant_name in REWARD_VARIANTS:
            subset = [r for r in all_rows if r["config"] == config_name and r["reward_variant"] == variant_name]
            n_crossing = sum(r["fixed_ever_beats_stop"] for r in subset)
            frac_crossing = n_crossing / len(subset)
            crossing_fractions = [r["first_crossover_removed_fraction"] for r in subset if r["fixed_ever_beats_stop"]]
            mean_crossover = f"{statistics.mean(crossing_fractions):.3f}" if crossing_fractions else "n/a"
            print(f"{config_name:<42}{variant_name:<13}{frac_crossing:>13.0%}{mean_crossover:>26}{n_crossing:>13d}")

    print(f"\n=== PART B: trained-policy achieved removal fraction / STOP frequency, mean +/- std across {N_GRAPHS} graphs ===")
    print(f"{'config':<42}{'variant':<13}{'mean_removed':>16}{'STOP freq':>12}")
    for config_name in WEIGHT_CONFIGS:
        for variant_name in REWARD_VARIANTS:
            subset = [r for r in all_rows if r["config"] == config_name and r["reward_variant"] == variant_name]
            removed_vals = [r["trained_final_removed"] for r in subset]
            removed_std = statistics.stdev(removed_vals) if len(removed_vals) > 1 else 0.0
            stop_freq = sum(r["trained_stopped_early"] for r in subset) / len(subset)
            print(
                f"{config_name:<42}{variant_name:<13}"
                f"{statistics.mean(removed_vals):>10.3f}+/-{removed_std:<5.3f}"
                f"{stop_freq:>12.0%}"
            )

    print(
        "\nInterpretation guide (not forced here): if a config's TERMINAL-variant STOP "
        "frequency / near-zero achieved removal PERSISTS from the incremental variant, "
        "that argues the collapse is a property of the reward WEIGHTS themselves (what's "
        "being optimized), not an artifact of incremental shaping. If the terminal "
        "variant's STOP frequency drops substantially for the same config, incremental "
        "delivery is causally implicated. Cross-check against Part A: if G_FIXED > G_STOP "
        "for a config/variant (continuing IS reward-preferable) but the trained policy "
        "still collapses to STOP, that combination points at a real optimization gap for "
        "that setting, not a genuine reward-optimal preference."
    )


def main() -> None:
    start = time.time()
    all_rows = []
    all_checkpoint_rows = []
    for graph_seed in range(N_GRAPHS):
        graph_start = time.time()
        rows, checkpoint_rows = run_one_graph(graph_seed)
        all_rows.extend(rows)
        all_checkpoint_rows.extend(checkpoint_rows)
        print(f"graph {graph_seed:2d}/{N_GRAPHS} done in {time.time() - graph_start:.1f}s (total elapsed {time.time() - start:.0f}s)", flush=True)

    total_elapsed = time.time() - start
    print(f"\ndiagnostic complete: {N_GRAPHS} graphs in {total_elapsed:.0f}s")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_seed", "config", "reward_variant", "g_stop", "g_fixed_trajectory",
                "fixed_beats_stop", "fixed_ever_beats_stop", "first_crossover_removed_fraction",
                "trained_final_removed", "trained_stopped_early",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved {len(all_rows)} rows to {OUTPUT_CSV}")

    with open(CHECKPOINT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "graph_seed", "config", "reward_variant", "step_index",
                "removed_fraction", "g_fixed_at_step", "g_stop", "beats_stop_at_step",
            ],
        )
        writer.writeheader()
        writer.writerows(all_checkpoint_rows)
    print(f"saved {len(all_checkpoint_rows)} rows to {CHECKPOINT_CSV} (full per-step G_fixed(t) trajectory, no rerun needed to analyze the 'peaks early then declines' pattern)")

    print_summary(all_rows)


if __name__ == "__main__":
    main()
