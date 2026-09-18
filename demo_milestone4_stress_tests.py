"""Milestone 4: stress tests using the multi-graph harness.

Tests whether the core finding from the 40-graph harness (structural
heuristics beat GAT/RL on containment) holds across graph size, density,
and botnet aggressiveness, or whether there's a regime where it flips.

6 new conditions, reusing the existing n=300/m=3/default-beta 40-graph data
as the baseline reference (not rerun):
  - size:            n=150, m=3          |  n=600, m=3
  - density:         n=300, m=2 (sparse) |  n=300, m=5 (dense)
  - aggressiveness:  n=300, bias=-1.0 (~2x p_uv) | n=300, bias=0.0 (~3.5x p_uv)

Scope trims for tractability (smoke-tested first, see NOTES.md): the 3
alternate RL reward-weight configs are dropped (already answered -- pushing
weights doesn't close the gap, see the reward-ablation entry) and the
greedy simulation-guided heuristic is dropped (too expensive to scale across 6 conditions and
not central to the core "structural vs. GAT/RL" question). 15 graphs/
condition (not 40) -- lower statistical power than the main harness,
reported honestly as such.
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

N_GRAPHS = 15
TARGET_LEVEL = 0.50
N_RANDOM_REPEATS = 3
RL_EPISODES = 100
RL_EPSILON_DECAY_EPISODES = 70
MODEL_SEED = 0

STATIC_METHODS = ["GAT threshold", "GAT top-k", "degree-centrality", "betweenness-centrality", "highest-p_uv", "random"]
RL_NAME = "RL"
ALL_METHODS = STATIC_METHODS + [RL_NAME]

CONDITIONS = {
    "size_small (n=150, m=3)": dict(n_nodes=150, m=3, beta=DEFAULT_BETA),
    "size_large (n=600, m=3)": dict(n_nodes=600, m=3, beta=DEFAULT_BETA),
    "density_sparse (n=300, m=2)": dict(n_nodes=300, m=2, beta=DEFAULT_BETA),
    "density_dense (n=300, m=5)": dict(n_nodes=300, m=5, beta=DEFAULT_BETA),
    "aggressive_moderate (bias=-1.0, ~2x p_uv)": dict(n_nodes=300, m=3, beta=[-1.0] + DEFAULT_BETA[1:]),
    "aggressive_high (bias=0.0, ~3.5x p_uv)": dict(n_nodes=300, m=3, beta=[0.0] + DEFAULT_BETA[1:]),
}


def run_one_graph(graph_seed: int, n_nodes: int, m: int, beta: list) -> dict:
    graph, node_features, labels = generate_labeled_graph(n_nodes=n_nodes, m=m, graph_seed=graph_seed, beta=beta)
    data = build_pyg_data(graph, node_features, labels)
    train_mask, val_mask, test_mask = make_node_split(data.num_nodes, seed=0)

    degrees = dict(graph.degree())
    hub_node = max(degrees, key=degrees.get)
    sorted_by_degree = sorted(degrees.items(), key=lambda kv: kv[1])
    median_degree = sorted_by_degree[len(sorted_by_degree) // 2][1]
    mid_node = min((n for n in degrees if n != hub_node), key=lambda n: abs(degrees[n] - median_degree))
    seed_nodes = pick_seed_nodes(graph, hub_node, mid_node)

    edge_features = build_edge_features(graph)
    p_uv = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=beta)
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
    # frozen_f1 -- reusing test_mask for RL's reward would mean its
    # "held-out" test set was actually seen as reward signal during
    # training. val_mask (already split out by make_node_split, previously
    # unused here) is the correct, disjoint mask for this.
    baseline_utility_reward = measure_utility_frozen(base_model, graph, node_features, labels, val_mask)
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)

    results = {}

    def measure(method_name: str, pruned_graph_variants: list) -> None:
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
            "frozen_f1": statistics.mean(f["f1"] for f in frozen_list),
        }

    g = prune_lowest_score(graph, gat_scores, TARGET_LEVEL)
    measure("GAT threshold", [g])

    _, g, _ = calibrate_topk_for_target_fraction(graph, gat_scores, TARGET_LEVEL)
    measure("GAT top-k", [g])

    degree_scores = score_edges_by_avg_hub_score(graph, node_features)
    g = prune_highest_score(graph, degree_scores, TARGET_LEVEL)
    measure("degree-centrality", [g])

    betweenness_scores = score_edges_by_betweenness(graph)
    g = prune_highest_score(graph, betweenness_scores, TARGET_LEVEL)
    measure("betweenness-centrality", [g])

    g = prune_highest_score(graph, p_uv, TARGET_LEVEL)
    measure("highest-p_uv", [g])

    variants = [prune_random(graph, TARGET_LEVEL, seed=1000 * i + 7) for i in range(N_RANDOM_REPEATS)]
    measure("random", variants)

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
        w_security=1.0,
        w_utility=1.0,
        w_cost=0.1,
    )
    q_network, _ = train_dqn(env, n_episodes=RL_EPISODES, epsilon_decay_episodes=RL_EPSILON_DECAY_EPISODES, seed=0)
    run_greedy_episode(env, q_network)
    rl_graph = env.working_graph.copy()
    measure(RL_NAME, [rl_graph])

    return results


def run_condition(condition_name: str, n_nodes: int, m: int, beta: list) -> dict:
    all_results = {m_name: {"containment_ratio": [], "frozen_f1": []} for m_name in ALL_METHODS}
    cond_start = time.time()
    for graph_seed in range(N_GRAPHS):
        results = run_one_graph(graph_seed, n_nodes, m, beta)
        for m_name in ALL_METHODS:
            all_results[m_name]["containment_ratio"].append(results[m_name]["containment_ratio"])
            all_results[m_name]["frozen_f1"].append(results[m_name]["frozen_f1"])
    elapsed = time.time() - cond_start
    print(f"\n--- {condition_name} done in {elapsed:.0f}s ({elapsed / N_GRAPHS:.1f}s/graph) ---", flush=True)

    print(f"{'method':<24}{'containment_ratio':>20}{'frozen_f1':>12}")
    for m_name in ALL_METHODS:
        c = all_results[m_name]["containment_ratio"]
        f = all_results[m_name]["frozen_f1"]
        print(f"{m_name:<24}{statistics.mean(c):>10.3f}+/-{statistics.stdev(c):.3f}{statistics.mean(f):>12.3f}")

    pairs = [(RL_NAME, m_name) for m_name in STATIC_METHODS if m_name != "random"]
    comparisons = run_paired_tests_with_correction(pairs, {m_name: all_results[m_name]["containment_ratio"] for m_name in ALL_METHODS})
    print(f"{'A vs B':<30}{'mean_A':>8}{'mean_B':>8}{'diff':>8}{'raw_p':>10}{'holm_p':>10}{'sig?':>6}")
    for c in comparisons:
        label = f"{c['method_a']} vs {c['method_b']}"
        print(f"{label:<30}{c['mean_a']:>8.3f}{c['mean_b']:>8.3f}{c['mean_diff']:>8.3f}{c['p_value']:>10.4f}{c['p_value_holm']:>10.4f}{str(c['significant_holm']):>6}")

    return {"all_results": all_results, "comparisons": comparisons}


def main() -> None:
    overall_start = time.time()
    for condition_name, params in CONDITIONS.items():
        print(f"\n\n=== running condition: {condition_name} ===", flush=True)
        run_condition(condition_name, **params)
    print(f"\n\nALL CONDITIONS COMPLETE in {time.time() - overall_start:.0f}s")
    print("No further interpretation forced here -- see NOTES.md for the read-through.")


if __name__ == "__main__":
    main()
