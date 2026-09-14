import math
from src.milestone1 import _feature_vector, compute_edge_infection_probabilities
import networkx as nx

node_features = {
    0: {"risk": 0.9, "hub_score": 0.1, "community": 0},
    1: {"risk": 0.2, "hub_score": 0.8, "community": 1},
}
edge_features = {(0, 1): {"interaction": 0.5}}
beta = [-2.0, 0.25, 0.6, 0.7, 0.3]

def prob(u, v):
    phi = _feature_vector(node_features, edge_features, u, v)
    score = sum(b * x for b, x in zip(beta, phi))
    return 1.0 / (1.0 + math.exp(-score))

p_uv = prob(0, 1)
p_vu = prob(1, 0)
print(f"p(u=0,v=1) = {p_uv:.5f}")
print(f"p(u=1,v=0) = {p_vu:.5f}")
print("SYMMETRIC OK" if abs(p_uv - p_vu) < 1e-12 else "STILL ASYMMETRIC")