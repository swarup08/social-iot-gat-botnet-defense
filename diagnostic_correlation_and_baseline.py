"""Lightweight diagnostic: score-correlation + unpruned-infection sanity check.

Two questions this answers, cheaply (no GAT/RL training, so this runs in a few
minutes instead of the ~60 min demo_milestone4_harness.py takes):

1. How correlated is degree-centrality (score_edges_by_avg_hub_score) with the
   other structural/feature-driven edge scores (eigenscore, betweenness,
   highest-p_uv)? If they're nearly rank-identical, that's a candidate
   explanation for why those methods tie each other in the main harness --
   they may just be re-discovering the same edges by different routes.
2. What does the UNPRUNED infected fraction look like across the same 40
   graphs x seed types the main harness uses as its containment_ratio
   denominator? containment_ratio = pruned_infected / baseline_infected, so if
   baseline_infected is tiny or highly variable, that denominator alone could
   be distorting the ratio independent of which pruning method is used.

Uses the identical 40 graphs (graph_seed 0..39, n_nodes=300) and the identical
hub/mid seed-node selection as demo_milestone4_harness.py's run_one_graph, so
results here are directly comparable to that harness's containment_ratio
population.

Reviewer-requested (Discussion comment on the correlation numbers): besides
the aggregate mean+/-std saved to diagnostic_results.csv, also saves
diagnostic_correlation_per_graph.csv -- one row per (graph_seed, metric,
correlation_value) for all three Spearman correlations across all 40
graphs, so the exact per-graph values behind the paper's reported numbers
are inspectable without rerunning.
"""

import statistics

from scipy.stats import spearmanr

from src.milestone1 import build_edge_features, compute_edge_infection_probabilities
from src.milestone2 import DEFAULT_BETA, generate_labeled_graph
from src.milestone2_pruning import (
    measure_security,
    pick_seed_nodes,
    score_edges_by_avg_hub_score,
    score_edges_by_betweenness,
    score_edges_by_eigenscore,
)
from src.milestone3_xai import save_table_csv

N_GRAPHS = 40  # matches demo_milestone4_harness.py's N_GRAPHS -- same 40 graph instances


def spearman_against_degree(degree_scores: dict, other_scores: dict) -> float:
    """Spearman rank correlation between degree_scores and other_scores,
    aligned on their common (u, v) edge keys (both are dicts keyed by
    graph.edges() tuples, but built by different functions, so we don't
    assume identical key order -- align explicitly via two parallel lists
    built from the shared key set).
    """
    common_edges = [e for e in degree_scores if e in other_scores]
    degree_vals = [degree_scores[e] for e in common_edges]
    other_vals = [other_scores[e] for e in common_edges]
    correlation, _p_value = spearmanr(degree_vals, other_vals)
    return correlation


def main() -> None:
    corr_degree_eigenscore = []
    corr_degree_betweenness = []
    corr_degree_puv = []
    unpruned_infected_fractions = []
    per_graph_rows = []  # reviewer-requested: the exact values behind the aggregate mean+/-std

    for graph_seed in range(N_GRAPHS):
        # Step 1: build the same graph instance the main harness uses at this seed.
        graph, node_features, labels = generate_labeled_graph(n_nodes=300, graph_seed=graph_seed)

        # Step 2: feature-driven infection probability per edge (same p_uv the
        # harness's baseline_infected and highest-p_uv baseline both use).
        edge_features = build_edge_features(graph)
        p_uv = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=DEFAULT_BETA)

        # Step 3: the four edge-scoring functions being compared for rank agreement.
        degree_scores = score_edges_by_avg_hub_score(graph, node_features)
        betweenness_scores = score_edges_by_betweenness(graph)
        eigenscore_scores = score_edges_by_eigenscore(graph)

        # Step 4: Spearman rank correlation of each score against degree_scores.
        eigenscore_corr = spearman_against_degree(degree_scores, eigenscore_scores)
        betweenness_corr = spearman_against_degree(degree_scores, betweenness_scores)
        puv_corr = spearman_against_degree(degree_scores, p_uv)
        corr_degree_eigenscore.append(eigenscore_corr)
        corr_degree_betweenness.append(betweenness_corr)
        corr_degree_puv.append(puv_corr)

        per_graph_rows.append({"graph_seed": graph_seed, "metric": "degree vs eigenscore", "correlation_value": eigenscore_corr})
        per_graph_rows.append({"graph_seed": graph_seed, "metric": "degree vs betweenness", "correlation_value": betweenness_corr})
        per_graph_rows.append({"graph_seed": graph_seed, "metric": "degree vs highest-p_uv", "correlation_value": puv_corr})

        # Step 5: unpruned infected fraction per seed node, matching
        # run_one_graph's hub/mid selection and measure()'s "iterate all of
        # seed_nodes.items()" behavior exactly.
        degrees = dict(graph.degree())
        hub_node = max(degrees, key=degrees.get)
        sorted_by_degree = sorted(degrees.items(), key=lambda kv: kv[1])
        median_degree = sorted_by_degree[len(sorted_by_degree) // 2][1]
        mid_node = min((n for n in degrees if n != hub_node), key=lambda n: abs(degrees[n] - median_degree))
        seed_nodes = pick_seed_nodes(graph, hub_node, mid_node)

        for _name, node in seed_nodes.items():
            unpruned_infected_fractions.append(measure_security(graph, p_uv, node))

        print(f"graph {graph_seed:2d}/{N_GRAPHS} done", flush=True)

    # --- summary (a) : Spearman correlations, mean +/- std across 40 graphs ---
    print("\n=== (a) Spearman rank correlation vs. degree-centrality, mean +/- std across 40 graphs ===")
    corr_summary = [
        ("degree vs eigenscore", corr_degree_eigenscore),
        ("degree vs betweenness", corr_degree_betweenness),
        ("degree vs highest-p_uv", corr_degree_puv),
    ]
    for label, values in corr_summary:
        print(f"{label:<26}{statistics.mean(values):>8.3f} +/- {statistics.stdev(values):.3f}  (n={len(values)})")

    # --- summary (b) : unpruned infected fraction, mean +/- std over all (graph, seed_type) pairs ---
    print("\n=== (b) unpruned infected fraction (containment_ratio denominator population) ===")
    print(
        f"{'unpruned infected fraction':<26}"
        f"{statistics.mean(unpruned_infected_fractions):>8.3f} +/- {statistics.stdev(unpruned_infected_fractions):.3f}"
        f"  (n={len(unpruned_infected_fractions)})"
    )

    # --- save CSV: one row per metric with mean/std/n ---
    rows = []
    for label, values in corr_summary:
        rows.append({"metric": label, "mean": statistics.mean(values), "std": statistics.stdev(values), "n": len(values)})
    rows.append(
        {
            "metric": "unpruned_infected_fraction",
            "mean": statistics.mean(unpruned_infected_fractions),
            "std": statistics.stdev(unpruned_infected_fractions),
            "n": len(unpruned_infected_fractions),
        }
    )
    csv_path = save_table_csv(rows, "diagnostic_results.csv")
    print(f"\nsaved summary table to {csv_path}")

    # --- save CSV: one row per (graph_seed, metric) -- reviewer-requested
    # per-graph values behind the aggregate mean/std above (correlation-only;
    # the unpruned-infected-fraction check isn't part of the reviewer's ask). ---
    per_graph_csv_path = save_table_csv(per_graph_rows, "diagnostic_correlation_per_graph.csv")
    print(f"saved per-graph correlation table to {per_graph_csv_path} ({len(per_graph_rows)} rows)")


if __name__ == "__main__":
    main()
