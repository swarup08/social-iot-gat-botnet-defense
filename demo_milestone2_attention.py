"""Milestone 2 demo: extract GAT attention-based edge importance scores.

Trains a GAT (via train_gat's default weight_mildness=0.5) and extracts a
single aggregated attention score per edge from its trained GATConv layers,
checking whether it correlates with the actual infection probability p_uv
(the quantity the whole botnet model is built around) or with structural
properties (edge betweenness centrality, average endpoint hub_score).

Two versions are compared:
  - RAW s_uv (extract_edge_attention_scores): the straightforward average of
    softmax-normalized attention over heads/layers/directions. This turned
    out to be strongly NEGATIVELY correlated with hub_score/betweenness/p_uv
    (Pearson r as low as -0.62, Spearman -0.79) -- but a 1/x-shaped scatter
    against hub_score gave it away as a softmax-normalization artifact:
    high-degree target nodes mechanically hand out a smaller average share to
    each neighbor, independent of the model's actual learned preference.
  - DEGREE-CORRECTED (extract_degree_corrected_attention_scores): multiplies
    each directed alpha_ij by the target's degree before combining
    directions, canceling that mechanical dilution while keeping the
    learned relative-preference signal. This is the version that should
    actually be used for pruning decisions -- see its docstring in
    src/milestone2.py for why.
"""

import numpy as np
from scipy import stats

from src.milestone1 import build_edge_features, compute_edge_infection_probabilities
from src.milestone2 import (
    DEFAULT_BETA,
    align_edge_metrics,
    build_pyg_data,
    extract_degree_corrected_attention_scores,
    extract_edge_attention_scores,
    generate_labeled_graph,
    make_node_split,
    plot_attention_score_distribution,
    plot_attention_vs_metric,
    train_gat,
)

N_NODES = 300


def report_correlation(name: str, s_values: np.ndarray, m_values: np.ndarray) -> None:
    pearson_r, pearson_p = stats.pearsonr(s_values, m_values)
    spearman_r, spearman_p = stats.spearmanr(s_values, m_values)
    print(f"  {name}: Pearson r={pearson_r:+.3f} (p={pearson_p:.3g})  Spearman r={spearman_r:+.3f} (p={spearman_p:.3g})")


def analyze(label: str, prefix: str, scores: dict, p_uv: dict, betweenness: dict, avg_hub_score: dict) -> None:
    values = np.array(list(scores.values()))
    print(f"\n=== {label} ===")
    print(f"distribution over {len(scores)} edges: min={values.min():.4f}  max={values.max():.4f}  mean={values.mean():.4f}  std={values.std():.4f}")
    print(
        f"quartiles: 25%={np.percentile(values, 25):.4f}  50%={np.percentile(values, 50):.4f}  "
        f"75%={np.percentile(values, 75):.4f}"
    )

    print("correlation with:")
    s_vs_puv, puv_vals = align_edge_metrics(scores, p_uv)
    report_correlation("p_uv (infection probability)", s_vs_puv, puv_vals)

    s_vs_bet, bet_vals = align_edge_metrics(scores, betweenness)
    report_correlation("edge betweenness centrality", s_vs_bet, bet_vals)

    s_vs_hub, hub_vals = align_edge_metrics(scores, avg_hub_score)
    report_correlation("avg endpoint hub_score", s_vs_hub, hub_vals)

    dist_path = plot_attention_score_distribution(scores, output_path=f"{prefix}_distribution.png")
    hub_path = plot_attention_vs_metric(s_vs_hub, hub_vals, "avg endpoint hub_score", f"{prefix}_vs_hub_score.png")
    print(f"saved: {dist_path}, {hub_path}")


def main() -> None:
    graph, node_features, labels = generate_labeled_graph(n_nodes=N_NODES)
    data = build_pyg_data(graph, node_features, labels)
    train_mask, val_mask, test_mask = make_node_split(data.num_nodes, seed=0)

    model = train_gat(data, train_mask)  # default weight_mildness=0.5

    # Recompute the SAME p_uv (same beta) that generated this graph's labels,
    # and the SAME structural edge feature (betweenness) Milestone 1 used, so
    # the comparison is against the actual quantities driving this graph.
    edge_features = build_edge_features(graph)
    p_uv = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=DEFAULT_BETA)
    betweenness = {edge: features["interaction"] for edge, features in edge_features.items()}
    avg_hub_score = {(u, v): (node_features[u]["hub_score"] + node_features[v]["hub_score"]) / 2 for u, v in graph.edges()}

    raw_scores = extract_edge_attention_scores(model, data)
    analyze("RAW attention score s_uv", "attention_raw", raw_scores, p_uv, betweenness, avg_hub_score)

    corrected_scores = extract_degree_corrected_attention_scores(model, data, graph)
    analyze("DEGREE-CORRECTED attention score", "attention_corrected", corrected_scores, p_uv, betweenness, avg_hub_score)


if __name__ == "__main__":
    main()
