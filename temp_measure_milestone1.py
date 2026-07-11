from src.milestone1 import build_social_iot_graph, compute_edge_infection_probabilities, simulate_botnet

beta = [-2.0, 1.5, -1.0, 0.8, 0.4, 0.7, 0.3]
n_nodes = 60
m = 3
seeds = [7, 11, 13, 17, 23]

for seed in seeds:
    graph = build_social_iot_graph(n_nodes=n_nodes, m=m, seed=seed)
    node_features = {
        node: {"risk": 0.6 + (node % 5) * 0.05, "hub_score": 0.4 + (node % 3) * 0.1, "community": node % 3}
        for node in graph.nodes
    }
    edge_features = {
        (u, v): {"interaction": 0.2 + ((u + v) % 5) * 0.08}
        for u, v in graph.edges
    }
    probabilities = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=beta)
    result = simulate_botnet(graph, probabilities, initial_compromised={0}, seed=seed)
    infected_fraction = len(result["infected_nodes"]) / graph.number_of_nodes()
    print(f"seed={seed} infected_fraction={infected_fraction:.3f} infected_nodes={len(result['infected_nodes'])}")

seed = 11
graph = build_social_iot_graph(n_nodes=n_nodes, m=m, seed=seed)
node_features = {
    node: {"risk": 0.6 + (node % 5) * 0.05, "hub_score": 0.4 + (node % 3) * 0.1, "community": node % 3}
    for node in graph.nodes
}
edge_features = {
    (u, v): {"interaction": 0.2 + ((u + v) % 5) * 0.08}
    for u, v in graph.edges
}
probabilities = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=beta)
values = list(probabilities.values())
print(f"p_uv_summary min={min(values):.4f} max={max(values):.4f} mean={sum(values)/len(values):.4f}")
items = sorted(probabilities.items(), key=lambda kv: kv[1])
print("sample_edges=" + str([(edge, round(p, 4)) for edge, p in items[:10]]))
