# Notes / known limitations

Running log of findings that need to survive into the final write-up but
don't belong in code comments or README. Newest entries at the top.

## 2026-07-11 -- Milestone 2 GAT: minority-class recall is weak (~38%)

The Milestone 2 GAT (benign vs compromised node classification,
`src/milestone2.py`) uses `train_gat`'s default `weight_mildness=0.5`, chosen
from a three-way comparison across 5 random train/val/test splits x 4
model-init seeds each, on one labeled Social IoT graph (n=300 nodes,
infected_fraction ~0.20, labels from `generate_labeled_graph`'s majority-vote
rollouts). See `demo_milestone2.py` for the full methodology and per-split
breakdown, and `train_gat`'s docstring for the summary numbers.

| weight_mildness | test accuracy | vs. baseline (paired t-test) | recall (compromised) | F1 |
|---|---|---|---|---|
| 0.0 (unweighted) | 0.840 +/- 0.045 | p=0.003, significant | 0.178 | 0.294 |
| **0.5 (chosen)** | **0.845 +/- 0.054** | **p=0.002, significant** | **0.380** | **0.465** |
| 1.0 (fully balanced) | 0.746 +/- 0.054 | p=0.158, not significant | 0.624 | 0.483 (1/5 splits flagged unstable) |

**Known limitation -- do not let this get lost before the final write-up:**
even at the chosen mildness=0.5, recall on the compromised class is only
**~38%**, meaning the model misses well over half of the actual compromised
nodes on average across splits. Per-split recall is also still fairly
variable (0.229-0.643 across the 5 splits), and two of five splits show
seed-to-seed recall spread just under the convergence-instability flag
threshold (std 0.158 and 0.202 vs. the 0.25 flag line used in
`demo_milestone2.py`) -- not flagged as unstable, but not fully settled
either.

This is real, reproducible progress over the unweighted baseline (which
systematically, consistently misses ~82% of compromised nodes -- confirmed
via multi-seed averaging to be a genuine bias, not noise), but it is **not**
a mature or production-ready detector. The final write-up should describe
this plainly rather than characterizing Milestone 2's classifier as solved.

Full accuracy is also not the metric that matters here -- the unweighted
model's higher raw accuracy (0.840 vs 0.845 is actually a wash) comes almost
entirely from correctly classifying the easy majority (benign) class while
still missing most compromised nodes, which is the opposite of what a
security-relevant classifier should optimize for.
