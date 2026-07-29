"""Milestone 2 diagnostic, promoted from a single-graph footnote to a proper
small-n result (supervisor-requested follow-up).

The original search_rollouts=1-vs-5 diagnostic (see NOTES.md's "Milestone 2
greedy oracle" entry) ran on ONE graph instance only, showing that giving
greedy_simulation_guided_prune a MORE accurate per-step search signal (5
rollouts instead of 1) made containment WORSE at every pruning level, not
better -- evidence the method's underperformance vs. structural heuristics is
inherent one-step-at-a-time MYOPIA, not search noise (if it were noise, more
rollouts should have closed the gap). Since every other result in this
project is n=40 (or n=15 for ablations), a single graph is not enough to
stand behind that conclusion -- this script reruns the identical comparison
across N_GRAPHS independent graph instances (different seeds, same
n=300/m=3/DEFAULT_BETA generator, same 3 pruning levels, same 5-seed-node
containment methodology used everywhere else) and reports mean +/- std at
each level for both rollout settings, plus a paired t-test per level (same
graphs used for both settings -- a genuine within-subject comparison).

COST WARNING: search_rollouts=5 is expensive (~24-29x search_rollouts=1's
cost, not just 5x, per the original diagnostic's finding) -- budget
accordingly before running with a large N_GRAPHS. See NOTES.md for the
per-graph timing this script reports.
"""

import statistics
import time

from src.milestone1 import build_edge_features, compute_edge_infection_probabilities
from src.milestone2 import DEFAULT_BETA, generate_labeled_graph
from src.milestone2_pruning import (
    containment_ratio,
    greedy_simulation_guided_prune,
    measure_security,
    pick_seed_nodes,
)
from src.milestone4 import paired_comparison

N_GRAPHS = 10
PRUNING_LEVELS = [0.10, 0.25, 0.50]
ROLLOUT_SETTINGS = [1, 5]


def run_one_graph(graph_seed: int) -> dict:
    """Returns {rollouts: {"levels": {level: containment_ratio}, "time": seconds}}."""
    graph, node_features, labels = generate_labeled_graph(n_nodes=300, graph_seed=graph_seed)

    degrees = dict(graph.degree())
    hub_node = max(degrees, key=degrees.get)
    sorted_by_degree = sorted(degrees.items(), key=lambda kv: kv[1])
    median_degree = sorted_by_degree[len(sorted_by_degree) // 2][1]
    mid_node = min((n for n in degrees if n != hub_node), key=lambda n: abs(degrees[n] - median_degree))
    seed_nodes = pick_seed_nodes(graph, hub_node, mid_node)

    edge_features = build_edge_features(graph)
    p_uv = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=DEFAULT_BETA)
    baseline_infected = {name: measure_security(graph, p_uv, node) for name, node in seed_nodes.items()}

    results = {}
    for rollouts in ROLLOUT_SETTINGS:
        t = time.time()
        # ONE incremental greedy run covers all 3 checkpoint levels at once
        # (see greedy_simulation_guided_prune's docstring) -- the two
        # rollout settings need SEPARATE calls since they're different
        # search signals producing different greedy trajectories.
        checkpoints = greedy_simulation_guided_prune(
            graph, p_uv, seed_nodes, max_remove_fraction=max(PRUNING_LEVELS), checkpoint_fractions=PRUNING_LEVELS, search_rollouts=rollouts
        )
        elapsed = time.time() - t

        level_ratios = {}
        for level in PRUNING_LEVELS:
            pruned = checkpoints[level]
            # Full-rigor re-measurement (same 5-seed-node methodology used
            # everywhere else) -- only the SEARCH itself used the cheap signal.
            ratios = [containment_ratio(measure_security(pruned, p_uv, node), baseline_infected[name]) for name, node in seed_nodes.items()]
            level_ratios[level] = statistics.mean(ratios)

        results[rollouts] = {"levels": level_ratios, "time": elapsed}
    return results


def main() -> None:
    start = time.time()
    # per_rollout_level[rollouts][level] -> list of per-graph containment ratios
    per_rollout_level = {r: {level: [] for level in PRUNING_LEVELS} for r in ROLLOUT_SETTINGS}
    time_totals = {r: [] for r in ROLLOUT_SETTINGS}

    for graph_seed in range(N_GRAPHS):
        graph_start = time.time()
        results = run_one_graph(graph_seed)
        for rollouts in ROLLOUT_SETTINGS:
            for level in PRUNING_LEVELS:
                per_rollout_level[rollouts][level].append(results[rollouts]["levels"][level])
            time_totals[rollouts].append(results[rollouts]["time"])
        print(
            f"graph {graph_seed:2d}/{N_GRAPHS} done in {time.time() - graph_start:.1f}s "
            f"(rollouts=1: {results[1]['time']:.1f}s, rollouts=5: {results[5]['time']:.1f}s) "
            f"(total elapsed {time.time() - start:.0f}s)",
            flush=True,
        )

    total_elapsed = time.time() - start
    print(f"\ndiagnostic complete: {N_GRAPHS} graphs in {total_elapsed:.0f}s")
    print(f"mean search time/graph -- rollouts=1: {statistics.mean(time_totals[1]):.1f}s, rollouts=5: {statistics.mean(time_totals[5]):.1f}s")

    print(f"\n=== containment ratio, mean +/- std across {N_GRAPHS} graphs ===")
    print(f"{'level':>8}{'rollouts=1':>18}{'rollouts=5':>18}{'paired diff':>14}{'paired p':>12}{'holm-relevant?':>16}")
    for level in PRUNING_LEVELS:
        v1 = per_rollout_level[1][level]
        v5 = per_rollout_level[5][level]
        comparison = paired_comparison(f"rollouts=1 @{level}", f"rollouts=5 @{level}", v1, v5)
        print(
            f"{level:>8.2f}"
            f"{statistics.mean(v1):>12.3f}+/-{statistics.stdev(v1):.3f}"
            f"{statistics.mean(v5):>12.3f}+/-{statistics.stdev(v5):.3f}"
            f"{comparison['mean_diff']:>14.3f}{comparison['p_value']:>12.4f}"
            f"{'n=3 tests, not corrected here' if level == PRUNING_LEVELS[0] else '':>16}"
        )

    print(
        "\nNo Holm correction applied here (only 3 tests, all reported plainly with raw p-values) -- "
        "this is a targeted diagnostic re-confirming ONE prior finding at proper n, not a new family "
        "of exploratory comparisons. See NOTES.md for the read-through of whether the original "
        "single-graph conclusion holds."
    )


if __name__ == "__main__":
    main()
