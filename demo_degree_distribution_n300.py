"""Regenerates degree_distribution_n300.png/.pdf -- the paper's N=300 degree-
distribution figure (Reproducibility Appendix), which previously had no
committed script producing it. It was originally made via an ad-hoc,
un-committed interactive call (see commit dfddf2b's message): generate_labeled_graph
(n_nodes=300, graph_seed=0) + the existing plot_degree_distribution(), the same
generator and graph_seed=0 the 40-graph harness itself uses for its first
instance -- this script just makes that reproducible from a committed file.

Per the paper's own caption/Reproducibility Appendix, the exact library
versions used for the ORIGINAL frozen run that produced the committed figure
were not logged at the time (see ENVIRONMENT.md) -- so re-running this script
today is offered as an illustrative reproduction of the same generator and
parameters, not a byte-for-byte guaranteed match to the original image.
"""

from src.milestone1 import plot_degree_distribution
from src.milestone2 import generate_labeled_graph

N_NODES = 300
GRAPH_SEED = 0


def main() -> None:
    graph, node_features, labels = generate_labeled_graph(n_nodes=N_NODES, graph_seed=GRAPH_SEED)
    print(f"nodes={graph.number_of_nodes()} edges={graph.number_of_edges()}")

    png_path = plot_degree_distribution(graph, output_path="degree_distribution_n300.png")
    print(f"saved {png_path}")

    pdf_path = plot_degree_distribution(graph, output_path="degree_distribution_n300.pdf")
    print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
