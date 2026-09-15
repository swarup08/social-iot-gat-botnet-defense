"""Standalone diagnostic: reuses demo_milestone4_harness.py's exact RL
training setup (same 40 graphs, same PruningEnv hyperparameters, same
train_dqn call, same run_greedy_episode rollout) to log, per graph, the
ACHIEVED edge-removal fraction and whether the trained policy invoked STOP
before the 50% budget.

Not a new experiment -- this is the identical RL policy training already
behind the "RL" row in Table 1 / harness_summary.csv, just with
final_removed/stopped_early logged per graph instead of only feeding into
the aggregate containment_ratio. Answers: does chunk-boundary rounding
(chunk_size = round(0.05*E) edges per step, see src/milestone3.py) push the
achieved fraction meaningfully above 0.50 when the policy runs to budget,
and how often does the policy stop before the budget at all (already
documented as graph-independent -- always False -- for the collapsed
high-security reward configs, but this is the BASELINE reward weights used
for Table 1's RL row).

Does NOT modify demo_milestone4_harness.py or any other existing file.
"""

import csv
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
from src.milestone2_pruning import measure_security, measure_utility_frozen, pick_seed_nodes
from src.milestone3 import PruningEnv
from src.milestone3_dqn import run_greedy_episode, train_dqn

N_GRAPHS = 40
RL_EPISODES = 100
RL_EPSILON_DECAY_EPISODES = 70
MODEL_SEED = 0
MAX_STEPS = 15
MAX_REMOVAL_FRACTION = 0.5
CHUNK_FRACTION = 0.05

OUTPUT_CSV = "rl_achieved_budget_diagnostic.csv"


def run_one_graph(graph_seed: int) -> dict:
    """Same per-graph RL setup as demo_milestone4_harness.py's run_one_graph()
    RL branch -- graph/GAT/PruningEnv construction, train_dqn, then a single
    greedy rollout via run_greedy_episode."""
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

    # infection_seed_base default (7) -- this is the RL TRAINING-time reward
    # baseline (see src/milestone3.py's PruningEnv), unrelated to the
    # disjoint held-out eval seed base used to report Table 1's containment
    # ratios; this script only cares about achieved removal fraction and
    # STOP behavior, not containment_ratio, so the training-time baseline is
    # the correct (and only) one needed here.
    baseline_infected = {name: measure_security(graph, p_uv, node) for name, node in seed_nodes.items()}

    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED)
    # Supervisor-flagged fix (same as demo_milestone4_harness.py): PruningEnv's
    # reward signal must be measured on a mask disjoint from whatever mask
    # would report a genuinely held-out final evaluation -- this script only
    # logs final_removed/stopped_early (no frozen utility reporting at all),
    # so baseline_utility here has only ever fed PruningEnv's reward
    # pathway; using val_mask (not test_mask) for it is correct.
    baseline_utility = measure_utility_frozen(base_model, graph, node_features, labels, val_mask)
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)

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
        baseline_utility=baseline_utility,
        max_steps=MAX_STEPS,
        max_removal_fraction=MAX_REMOVAL_FRACTION,
        chunk_fraction=CHUNK_FRACTION,
    )
    q_network, _ = train_dqn(env, n_episodes=RL_EPISODES, epsilon_decay_episodes=RL_EPSILON_DECAY_EPISODES, seed=0)
    result = run_greedy_episode(env, q_network)

    return {
        "graph_seed": graph_seed,
        "final_removed": result["final_removed"],
        "stopped_early": result["stopped_early"],
    }


def main() -> None:
    rows = []
    harness_start = time.time()

    for graph_seed in range(N_GRAPHS):
        graph_start = time.time()
        row = run_one_graph(graph_seed)
        rows.append(row)
        print(
            f"graph {graph_seed:2d}/{N_GRAPHS} done in {time.time() - graph_start:.1f}s "
            f"(total elapsed {time.time() - harness_start:.0f}s)  "
            f"final_removed={row['final_removed']:.4f}  stopped_early={row['stopped_early']}",
            flush=True,
        )

    total_elapsed = time.time() - harness_start
    print(f"\ndiagnostic complete: {N_GRAPHS} graphs in {total_elapsed:.0f}s ({total_elapsed / N_GRAPHS:.1f}s/graph average)")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["graph_seed", "final_removed", "stopped_early"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {len(rows)} rows to {OUTPUT_CSV}")

    n_stopped = sum(1 for r in rows if r["stopped_early"])
    n_ran_to_budget = N_GRAPHS - n_stopped
    all_removed = [r["final_removed"] for r in rows]
    budget_removed = [r["final_removed"] for r in rows if not r["stopped_early"]]

    print(f"\n=== STOP behavior across {N_GRAPHS} graphs ===")
    print(f"stopped early: {n_stopped}/{N_GRAPHS}   ran to budget: {n_ran_to_budget}/{N_GRAPHS}")

    print(f"\n=== final_removed, ALL {N_GRAPHS} graphs ===")
    print(f"mean={statistics.mean(all_removed):.4f}  min={min(all_removed):.4f}  max={max(all_removed):.4f}")

    print(f"\n=== final_removed, ran-to-budget graphs only (n={n_ran_to_budget}) ===")
    if budget_removed:
        print(f"mean={statistics.mean(budget_removed):.4f}  min={min(budget_removed):.4f}  max={max(budget_removed):.4f}")
        print(f"(target was MAX_REMOVAL_FRACTION={MAX_REMOVAL_FRACTION}; chunk_size overshoot above target = mean - target = {statistics.mean(budget_removed) - MAX_REMOVAL_FRACTION:+.4f})")
    else:
        print("no graphs ran to budget -- every one stopped early")


if __name__ == "__main__":
    main()
