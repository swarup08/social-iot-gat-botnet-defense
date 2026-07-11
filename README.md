# Milestone 1: Baseline social-IoT graph and botnet simulator

This workspace now contains an initial implementation of Milestone 1.

## What is included

- A scale-free social-IoT graph built with a Barabási-Albert model.
- A feature-driven infection model based on the roadmap formula
  $p_{uv} = \sigma(\beta^T \phi(x_u, x_v, e_{uv}))$.
- A simple botnet simulator that spreads infection over the graph.

## Run the demo

```bash
python -m unittest discover -s tests -v
python -c "from src.milestone1 import build_social_iot_graph, compute_edge_infection_probabilities, simulate_botnet; import networkx as nx; g=build_social_iot_graph(); nodes={n: {'risk':0.6,'hub_score':0.7,'community':0} for n in g.nodes}; edges={(u,v):{'interaction':0.3} for u,v in g.edges}; probs=compute_edge_infection_probabilities(g,nodes,edges,[-2.0,1.5,-1.0,0.6,0.5,0.8,0.4]); print(simulate_botnet(g,probs,{0},seed=7))"
```
