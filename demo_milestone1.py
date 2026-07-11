"""Small demo script for Milestone 1.

Run this to inspect the generated graph and a sample infection rollout.
"""

from src.milestone1 import (
    build_edge_features,
    build_node_features,
    build_social_iot_graph,
    compute_edge_infection_probabilities,
    plot_degree_distribution,
    plot_infection_history,
    simulate_botnet,
)

BETA = [-2.0, 1.5, -1.0, 0.8, 0.4, 0.7, 0.3]


def main() -> None:
    graph = build_social_iot_graph(n_nodes=60, m=3, seed=11)
    # Features are now derived from real graph structure (degree centrality,
    # clustering coefficient, detected communities, edge betweenness) and
    # roadmap device types, instead of node-id-modulo placeholders.
    node_features = build_node_features(graph)
    edge_features = build_edge_features(graph)
    probabilities = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=BETA)

    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

    # Different initial conditions: node 0 (arbitrary), the highest-degree hub
    # excluding node 0 (worst case -- a compromised gateway/hub), and the
    # lowest-degree leaf excluding both (best case -- a compromised sensor at
    # the edge of the network). Excluding already-picked nodes from each next
    # pick keeps the three seeds distinct even when node 0 itself happens to
    # be the graph's hub (common in Barabási-Albert graphs, since the earliest
    # nodes accumulate the most attachments). This satisfies the "different
    # initial conditions" acceptance criterion.
    degrees = dict(graph.degree())
    hub_node = max((n for n in degrees if n != 0), key=degrees.get)
    leaf_node = min((n for n in degrees if n not in (0, hub_node)), key=degrees.get)
    seed_nodes = {"node 0": 0, f"hub node {hub_node} (degree {degrees[hub_node]})": hub_node, f"leaf node {leaf_node} (degree {degrees[leaf_node]})": leaf_node}

    first_result = None
    for label, start_node in seed_nodes.items():
        result = simulate_botnet(graph, probabilities, initial_compromised={start_node}, seed=11)
        infected_fraction = len(result["infected_nodes"]) / graph.number_of_nodes()
        print(f"[{label}] infected_fraction={infected_fraction:.3f} rounds={len(result['infection_history']) - 1}")
        print(f"[{label}] infection_history={result['infection_history']}")
        if first_result is None:
            first_result = result

    # Save the two Milestone 1 acceptance-criteria plots: infection growth over
    # time (for the node-0 rollout), and the degree distribution showing the
    # scale-free hub structure.
    infection_plot_path = plot_infection_history(first_result["infection_history"], output_path="infection_history.png")
    degree_plot_path = plot_degree_distribution(graph, output_path="degree_distribution.png")
    print(f"saved infection plot to {infection_plot_path}")
    print(f"saved degree plot to {degree_plot_path}")


if __name__ == "__main__":
    main()
