"""Standalone diagnostic: replicates demo_milestone4_harness.py's per-graph
pipeline EXACTLY (same 40 graph seeds, same 9 methods, same pruning
functions, same RL training, same TARGET_LEVEL=0.50, same
EVAL_INFECTION_SEED_BASE=10007), but additionally logs the ABSOLUTE
infected_pruned/infected_unpruned values behind every containment_ratio,
per (graph_seed, method, seed_condition) -- not just the aggregated mean
containment_ratio the harness itself persists to harness_summary.csv.

Purpose: containment_ratio alone doesn't tell a reader what fraction of the
network is actually still getting infected in absolute terms -- two methods
with the same ratio could correspond to very different absolute outcomes if
their baselines differ. This produces the paired numbers for the paper.

Does NOT modify demo_milestone4_harness.py or any other existing file --
this is an independent script that happens to reuse the same building
blocks and mirrors the same per-graph procedure, including training a
fresh GAT and a fresh DQN policy per graph (the expensive part -- same
cost profile as a full harness run, RL training is the bottleneck).
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

N_GRAPHS = 40
TARGET_LEVEL = 0.50
N_RANDOM_REPEATS = 3
RL_EPISODES = 100
RL_EPSILON_DECAY_EPISODES = 70
MODEL_SEED = 0
EVAL_INFECTION_SEED_BASE = 10007  # must match demo_milestone4_harness.py exactly

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

OUTPUT_CSV = "absolute_infection_diagnostic.csv"


def run_one_graph(graph_seed: int) -> list:
    """Mirrors demo_milestone4_harness.py's run_one_graph() exactly, but
    returns one row per (method, seed_condition) with absolute infected
    fractions logged alongside containment_ratio, instead of collapsing
    straight to a per-method mean."""
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

    # Same split as the harness: seed base 7 stays paired with PruningEnv's
    # training-time reward signal; seed base EVAL_INFECTION_SEED_BASE is the
    # disjoint, held-out pool every method's FINAL measurement (numerator
    # AND denominator) uses below.
    baseline_infected = {name: measure_security(graph, p_uv, node) for name, node in seed_nodes.items()}
    baseline_infected_eval = {
        name: measure_security(graph, p_uv, node, infection_seed_base=EVAL_INFECTION_SEED_BASE)
        for name, node in seed_nodes.items()
    }

    base_model = train_gat(data, train_mask, model_seed=MODEL_SEED)
    # Supervisor-flagged fix (same as demo_milestone4_harness.py): PruningEnv's
    # reward signal must be measured on a mask disjoint from whatever mask
    # would report a genuinely held-out final evaluation -- this script
    # doesn't report frozen utility at all (only infected_pruned/
    # containment_ratio, unrelated to node-classification masks), so
    # baseline_utility here has only ever fed PruningEnv's reward pathway;
    # using val_mask (not test_mask) for it is correct and there is no
    # second, test_mask-based consumer to preserve, unlike the harness.
    baseline_utility = measure_utility_frozen(base_model, graph, node_features, labels, val_mask)
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)

    rows = []

    def measure_and_log(method_name: str, pruned_graph_variants: list) -> None:
        """For each seed_condition, average infected_pruned/containment_ratio
        across pruned_graph_variants (>1 only for random's N_RANDOM_REPEATS
        repeats), then log ONE row per (graph_seed, method, seed_condition)."""
        for name, node in seed_nodes.items():
            infected_unpruned = baseline_infected_eval[name]
            variant_infected = [
                measure_security(g, p_uv, node, infection_seed_base=EVAL_INFECTION_SEED_BASE)
                for g in pruned_graph_variants
            ]
            infected_pruned = statistics.mean(variant_infected)
            ratio = containment_ratio(infected_pruned, infected_unpruned)
            rows.append(
                {
                    "graph_seed": graph_seed,
                    "method": method_name,
                    "seed_condition": name,
                    "infected_pruned": infected_pruned,
                    "infected_unpruned": infected_unpruned,
                    "containment_ratio": ratio,
                }
            )

    g = prune_lowest_score(graph, gat_scores, TARGET_LEVEL)
    measure_and_log("GAT threshold", [g])

    _, g, _ = calibrate_topk_for_target_fraction(graph, gat_scores, TARGET_LEVEL)
    measure_and_log("GAT top-k", [g])

    degree_scores = score_edges_by_avg_hub_score(graph, node_features)
    g = prune_highest_score(graph, degree_scores, TARGET_LEVEL)
    measure_and_log("degree-centrality", [g])

    betweenness_scores = score_edges_by_betweenness(graph)
    g = prune_highest_score(graph, betweenness_scores, TARGET_LEVEL)
    measure_and_log("betweenness-centrality", [g])

    g = prune_highest_score(graph, p_uv, TARGET_LEVEL)
    measure_and_log("highest-p_uv", [g])

    variants = [prune_random(graph, TARGET_LEVEL, seed=1000 * i + 7) for i in range(N_RANDOM_REPEATS)]
    measure_and_log("random", variants)

    eigenscore_scores = score_edges_by_eigenscore(graph)
    g = prune_highest_score(graph, eigenscore_scores, TARGET_LEVEL)
    measure_and_log("eigenscore", [g])

    checkpoints = greedy_simulation_guided_prune(graph, p_uv, seed_nodes, max_remove_fraction=TARGET_LEVEL, checkpoint_fractions=[TARGET_LEVEL], search_rollouts=1)
    measure_and_log("greedy simulation-guided heuristic", [checkpoints[TARGET_LEVEL]])

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
        max_steps=15,
        max_removal_fraction=TARGET_LEVEL,
        chunk_fraction=0.05,
    )
    q_network, _ = train_dqn(env, n_episodes=RL_EPISODES, epsilon_decay_episodes=RL_EPSILON_DECAY_EPISODES, seed=0)
    run_greedy_episode(env, q_network)
    rl_graph = env.working_graph.copy()
    measure_and_log("RL", [rl_graph])

    return rows


def main() -> None:
    all_rows = []
    harness_start = time.time()

    for graph_seed in range(N_GRAPHS):
        graph_start = time.time()
        rows = run_one_graph(graph_seed)
        all_rows.extend(rows)
        print(f"graph {graph_seed:2d}/{N_GRAPHS} done in {time.time() - graph_start:.1f}s (total elapsed {time.time() - harness_start:.0f}s)", flush=True)

    total_elapsed = time.time() - harness_start
    print(f"\ndiagnostic complete: {N_GRAPHS} graphs in {total_elapsed:.0f}s ({total_elapsed / N_GRAPHS:.1f}s/graph average)")

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["graph_seed", "method", "seed_condition", "infected_pruned", "infected_unpruned", "containment_ratio"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"saved {len(all_rows)} rows to {OUTPUT_CSV}")

    print(f"\n=== per-method mean absolute infected_pruned vs. mean containment_ratio (n=200 = 40 graphs x 5 seed conditions) ===")
    print(f"{'method':<36}{'mean_infected_pruned':>22}{'mean_containment_ratio':>24}")
    for method in METHODS:
        method_rows = [r for r in all_rows if r["method"] == method]
        mean_infected = statistics.mean(r["infected_pruned"] for r in method_rows)
        mean_ratio = statistics.mean(r["containment_ratio"] for r in method_rows)
        print(f"{method:<36}{mean_infected:>22.4f}{mean_ratio:>24.4f}")


if __name__ == "__main__":
    main()
