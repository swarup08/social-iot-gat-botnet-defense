"""Small demo script for Milestone 1.

Run this to inspect the generated graph and a sample infection rollout.
"""

from src.milestone1 import (
    build_social_iot_graph,
    compute_edge_infection_probabilities,
    plot_degree_distribution,
    plot_infection_history,
    simulate_botnet,
)


def main() -> None:
    graph = build_social_iot_graph(n_nodes=60, m=3, seed=11)
    node_features = {
        node: {"risk": 0.6 + (node % 5) * 0.05, "hub_score": 0.4 + (node % 3) * 0.1, "community": node % 3}
        for node in graph.nodes
    }
    edge_features = {
        (u, v): {"interaction": 0.2 + (u + v) % 5 * 0.08}
        for u, v in graph.edges
    }
    probabilities = compute_edge_infection_probabilities(
        graph,
        node_features,
        edge_features,
        beta=[-2.0, 1.5, -1.0, 0.8, 0.4, 0.7, 0.3],
    )
    result = simulate_botnet(graph, probabilities, initial_compromised={0}, seed=11)

    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")
    print(f"infected_nodes={sorted(result['infected_nodes'])}")
    print(f"infection_history={result['infection_history']}")

    # Save the two Milestone 1 acceptance-criteria plots: infection growth over
    # time, and the degree distribution showing the scale-free hub structure.
    infection_plot_path = plot_infection_history(result["infection_history"], output_path="infection_history.png")
    degree_plot_path = plot_degree_distribution(graph, output_path="degree_distribution.png")
    print(f"saved infection plot to {infection_plot_path}")
    print(f"saved degree plot to {degree_plot_path}")


if __name__ == "__main__":
    main()
