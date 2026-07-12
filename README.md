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
  highest-p_uv (direct risk ranking), and a greedy simulation oracle.
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
                   greedy oracle)
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
correction: **GAT+RL does not beat simple structural heuristics
(degree-centrality, highest-p_uv) on containment.** It ties or loses,
at far higher compute cost (RL: ~33s/graph vs. ~0.007-0.55s/graph for the
structural heuristics). RL does reliably beat random pruning and tie a
greedy simulation oracle. Ablations further show this containment
finding is conditional on the specific GAT architecture used (heads=4,
depth=2) -- shallower, deeper, or wider architectures break it.

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
