# GAT-RL Social IoT Botnet Containment

![Status](https://img.shields.io/badge/Status-All%204%20Milestones%20Complete-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch-Geometric-orange)
![NetworkX](https://img.shields.io/badge/NetworkX-Latest-success)
![Tests](https://img.shields.io/badge/Tests-49%20passing-brightgreen)

Graph Attention Networks (GAT) and Reinforcement Learning (RL) for
containing botnet propagation in Social Internet of Things (SIoT)
networks, with an attention/XAI explainability layer and a rigorous
multi-graph statistical evaluation harness. Built over four milestones
following the project roadmap (`docs/Swarup_Research.pdf`), explicitly
incorporating supervisor feedback from an earlier attempt (see
`CLAUDE.md` for the full feedback record and how each point was fixed).

## Overview

Social IoT devices form trust/communication graphs that are a natural
substrate for botnet propagation. This project asks: can a Graph
Attention Network's learned attention weights identify which edges to
prune to contain a botnet, and can Reinforcement Learning improve on
that with an adaptive policy -- and do either of these *actually* beat
simple structural heuristics (degree centrality, betweenness, direct
infection-risk ranking), once tested rigorously?

## Key Contributions

- A feature-driven botnet propagation simulator: infection probability
  `p_uv = sigmoid(beta^T phi(x_u, x_v, e_uv))` is a function of node/edge
  features, not a flat constant, so pruning a risky edge measurably
  changes the epidemic outcome.
- A scale-free (Barabasi-Albert) Social IoT graph generator with
  structural node features (hub score, clustering, community, device
  type) and edge features (normalized betweenness).
- A 2-layer GAT trained for benign/compromised node classification,
  with degree-corrected attention-based edge importance scoring.
- Two static pruning strategies (threshold, top-k-per-node) plus a full
  baseline suite: random, degree-centrality, betweenness-centrality,
  highest-p_uv (direct risk ranking), eigenscore (spectral/eigenvector-
  centrality, matching the edge-removal epidemic-containment literature's
  standard baseline), and a greedy simulation-guided heuristic.
- A custom DQN agent that learns an adaptive edge-pruning policy on top
  of GAT attention scores, with per-decision trajectory logging for
  explainability (XAI).
- A 40-graph statistical evaluation harness: paired significance tests,
  Holm-Bonferroni correction, absolute effect sizes, and compute cost
  reported per method -- not just point estimates from one graph.
- Ablation studies (GAT heads, GAT depth, RL reward weighting, noisy
  and incomplete features) and stress tests (graph size, density,
  attack aggressiveness), each run across multiple graph instances.

## Pipeline

```
Social IoT Graph Generator (scale-free, featured nodes/edges)
            |
Feature-driven Botnet Propagation Simulator (p_uv)
            |
GAT Node Classifier (benign / compromised)
            |
Degree-corrected Attention -> Edge Importance Scores
            |
     -----------------------------------
     |              |                  |
Static Pruning   Baseline Suite     RL Adaptive Pruning (DQN)
(threshold/top-k) (random/degree/    (attention-informed state,
                   betweenness/       XAI trajectory logging)
                   highest-p_uv/
                   greedy simulation-
                   guided heuristic)
     -----------------------------------
            |
Botnet Containment + Core-task Utility Evaluation
            |
40-graph Statistical Harness (paired tests, Holm correction,
effect sizes, compute cost) + Ablations + Stress Tests
            |
XAI Summaries (most-pruned-edge characteristics, per-node reports)
```

## Milestone Status

| Milestone | Description | Status |
|---|---|---|
| 1 | Social IoT graph + feature-driven botnet simulator | Complete |
| 2 | GAT edge importance + static pruning | Complete |
| 3 | RL adaptive pruning + XAI hooks | Complete |
| 4 | Baselines, ablations, stress tests, statistical harness | Complete |

## Repository Structure

```
.
├── src/
│   ├── milestone1.py             # graph generator, feature-driven simulator
│   ├── milestone2.py              # GAT model, training, attention extraction
│   ├── milestone2_pruning.py      # static pruning + full baseline suite
│   ├── milestone3.py              # RL environment (PruningEnv)
│   ├── milestone3_dqn.py          # custom DQN agent
│   ├── milestone3_xai.py          # XAI trajectory summaries
│   └── milestone4.py              # paired tests + Holm correction utilities
├── demo_milestone1.py ... demo_milestone4_*.py   # runnable experiment scripts
├── tests/                         # 49 unit tests across all milestones
├── *.png                          # generated figures (see below)
├── harness_summary.csv            # 40-graph per-method mean/std containment & utility
├── harness_paired_tests.csv       # paired significance tests, Holm-corrected
├── pruned_edge_characteristics.csv, hub_node_pruning_summary.csv   # XAI tables
├── NOTES.md                        # dated, running log of every finding, fix, and caveat
├── docs/Swarup_Research.pdf        # original project roadmap
└── CLAUDE.md                       # supervisor feedback record and how it was addressed
```

## Setup

```bash
pip install -r requirements.txt
```

## Running the experiments

**Milestone 1 -- graph and simulator**
```bash
python demo_milestone1.py
```

**Milestone 2 -- GAT, attention, static pruning**
```bash
python demo_milestone2.py
python demo_milestone2_attention.py
python demo_milestone2_convergence.py
python demo_milestone2_pruning.py
```

**Milestone 3 -- RL adaptive pruning + XAI**
```bash
python demo_milestone3_env.py
python demo_milestone3_dqn.py
python demo_milestone3_vs_baselines.py
python demo_milestone3_reward_diagnostic.py
python demo_milestone3_xai.py
```

**Milestone 4 -- baselines, ablations, stress tests, statistical harness**
```bash
python demo_milestone4_harness.py                    # ~60 min, the core 40-graph evaluation
python demo_milestone4_reward_ablation.py
python demo_milestone4_stress_tests.py
python demo_milestone4_gat_ablations.py
python demo_milestone4_depth_and_masking_ablation.py
python demo_milestone4_infection_curves.py
```

**Tests**
```bash
python -m unittest discover -s tests -v
```

## Headline Result (stated plainly, not oversold)

Edge pruning generally slows botnet spread while preserving core-task
performance -- that part works. But across 40 independent graph
instances, with paired significance tests and Holm-Bonferroni
correction, the honest headline is stronger and simpler: **on this
problem, simple structural pruning beats learned GAT+RL containment,
at a fraction of the cost.** RL loses significantly to every one of
GAT-threshold, GAT-top-k, degree-centrality, betweenness-centrality,
highest-p_uv, and eigenscore (all Holm-corrected p<0.05) -- it only
beats random and ties the greedy simulation-guided heuristic. Eigenscore
(edges ranked by the product of their endpoints' eigenvector
centrality) is the field's accepted spectral/eigenvalue standard for
edge-removal epidemic containment -- and it ties degree-centrality
exactly (Holm-corrected p=1.0), it does not beat it -- even the
spectral method the epidemic-containment literature treats as the
principled optimum does not beat plain degree centrality here, and it
still buries the learned pipeline.

The result is a genuine Pareto frontier, not a single winner: **degree-
centrality gives the best containment at trivial cost** (0.0108s/graph,
~3,595x cheaper than RL's 38.72s/graph), while **the greedy
simulation-guided heuristic buys the best security-utility balance**
(frozen F1 0.634 vs. degree's 0.562) at ~4,216x degree's compute cost
(45.41s/graph). Ratios are recomputed from the full-precision timing
column in `harness_summary.csv` and used consistently wherever quoted.
Neither dominates the other, and the learned methods (GAT, RL) sit
inside this frontier rather than on it. Degree-centrality is, in fact,
the *only* structural method that holds both axes at once: eigenscore
(frozen F1 0.458), betweenness-centrality (0.455), and highest-p_uv
(0.423) all buy their containment by cutting edges the core task needs,
landing in the same low-utility corner rather than matching degree's
combination of cheap, strong containment, and preserved utility. Even the greedy heuristic --
which looks directly at simulated infection outcomes step by step --
does not beat degree-centrality on containment, and a follow-up
diagnostic (more search accuracy per step making outcomes *worse*, not
better) confirms this is one-step-at-a-time myopia, not search noise --
containment on these graphs is structure-dominated. Ablations further
show the GAT+RL pipeline's containment finding is conditional on the
specific architecture used (heads=4, depth=2) -- shallower, deeper, or
wider architectures break it. The GAT classifier feeding two of the
pruning methods also recovers only ~46% of compromised nodes, a
limitation worth stating plainly since GAT attention is not a mature
signal on its own.

Full numbers, every caveat, and the dated record of how each finding
was reached (including two methodological bugs found and fixed
mid-project) are in [NOTES.md](NOTES.md) -- read this before quoting
any single number from this project.

## Known Limitations

- All experiments use one graph family (Barabasi-Albert scale-free);
  results are not yet validated on other topologies or real-world SIoT
  traffic datasets.
- RL is a single custom DQN implementation; whether a different RL
  algorithm would close the gap to structural heuristics is untested.
- The core containment finding is architecture-conditional (see above),
  not shown to be a general property of GAT-based attention pruning.
- The GAT node classifier's recall on the compromised class is
  moderate (~46% at the chosen class-weighting), not production-grade.

## Future Work

- Validation on real IoT/botnet traffic datasets.
- Alternative RL algorithms (policy gradient, PPO) as a further check
  on the containment gap.
- Broader topology families (community-structured / stochastic block
  model graphs) to test generality of the architecture-conditional
  finding.

## Acknowledgements

Built against feedback from a prior attempt at this project (see
`CLAUDE.md`), addressing: feature-driven infection modeling, realistic
(scale-free) topology, RL training-noise control, a complete baseline
suite, and full statistical rigor (multi-graph evaluation, paired
tests, multiple-comparison correction, effect sizes, compute cost).

## Author

**Swarup Sarkar**
