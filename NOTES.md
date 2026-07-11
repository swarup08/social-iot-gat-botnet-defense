# Notes / known limitations

Running log of findings that need to survive into the final write-up but
don't belong in code comments or README. Newest entries at the top.

## 2026-07-11 -- Milestone 2 training convergence: plateaus well before 300 epochs, mild overfitting in the tail

Added optional per-epoch train/val loss+accuracy logging to `train_gat`
(`log_history=True`, off by default -- see `demo_milestone2_convergence.py`
and `training_curves.png`). On a representative run (n=300 nodes, split
seed 0): both losses drop sharply for ~75-100 epochs, then train loss keeps
slowly decreasing while validation loss flattens and drifts up slightly
(best val_loss at epoch 281 of 300, but the whole 150-300 range is a noisy
plateau, not a clear minimum) -- a mild, unsurprising overfitting signature
given dropout is already regularizing against it. Practical takeaway: 300
epochs is more than strictly needed for convergence on this graph size;
kept as the default anyway since the cost is small and this hasn't been
tuned/validated across multiple graphs yet (that's Milestone 3 territory).

## 2026-07-11 -- Milestone 2 greedy oracle: does NOT behave as an upper bound at low pruning levels (search-signal noise), only catches up by 50%

Added the supervisor-requested greedy oracle baseline
(`greedy_oracle_prune`, `src/milestone2_pruning.py`): at each step, remove
whichever remaining edge most reduces simulated infection (averaged across
the same 5 seed nodes used everywhere else), one at a time. A full O(steps x
remaining_edges) search to 50% removal needs ~300K candidate evaluations on
this ~900-edge graph -- empirically ~25s total (all 3 checkpoints, one
incremental run) using in-place edge remove/restore (not graph copying) and
a cheap 1-rollout-per-candidate search signal. The checkpoint graphs it
returns ARE re-measured with the same full-rollout rigor as every other
method for the reported numbers -- only the SEARCH itself uses the cheap
signal.

**Result, stated plainly (see the updated containment_ratio_vs_pruning_level.png):**
the oracle is NOT the best-performing method at low pruning levels --
at 10% removal its containment ratio (0.599+/-0.123) is WORSE than 4 of the
5 other methods (degree-centrality 0.395, highest-p_uv 0.379, GAT top-k
0.402, betweenness 0.451), beating only random (0.703). At 25% it's still
mid-pack (0.223 vs. 0.104-0.129 for the structural methods). It only
converges with the pack by 50% (0.035, tied with GAT top-k/degree-
centrality/highest-p_uv). It DOES show the best utility (recall/F1) at
every level, notably 0.500 recall / 0.571 F1 at 50% -- the best F1 in the
entire table.

**Follow-up diagnostic (same day): confirmed as myopia, not search noise.**
Reran the search with search_rollouts=5 instead of 1 (everything else
identical -- same graph, seeds, checkpoints; final numbers still measured at
full rigor). If noise were the main cause, more rollouts should have
consistently closed the gap at every level. It didn't:

| level | search_rollouts=1 | search_rollouts=5 | other methods' range |
|---|---|---|---|
| 10% | 0.599 | **0.711 (worse)** | 0.10-0.45 |
| 25% | 0.223 | 0.191 (barely better) | 0.10-0.13 |
| 50% | 0.035 | 0.019 (now best) | ~0.03 |

At 10% -- where the gap was largest -- 5x the search signal made
containment measurably WORSE, not better, and 25% barely moved and
remained clearly mid-pack. Only at 50% did more rollouts help meaningfully.
This is the opposite of what the noise hypothesis predicts (consistent
improvement at every level) and matches the myopia hypothesis instead:
early in the trajectory, picking the single locally-best edge one at a time
just doesn't find as good a combination as a heuristic that ranks ALL edges
by a global criterion at once, no matter how accurately each candidate is
scored. The 5-rollout search also cost ~415s vs. ~25s (~16x, more than the
naive 5x) for no reliable improvement at the levels that mattered -- not
pursuing higher rollout counts further, per the "confirm and stop" scope of
this diagnostic.

**Consequence for the write-up:** do NOT call this method an "upper bound"
without this caveat attached -- the plot/table label it plainly as "greedy
oracle" (no parenthetical claim) for exactly this reason, and this stands
confirmed, not just suspected. Report it as "a greedy, outcome-driven
baseline that only pays off at higher pruning levels because of its
inherent one-step-at-a-time myopia," not as a validated ceiling on
achievable containment.

## 2026-07-11 -- Milestone 2 pruning: corrected re-run (multi-seed containment ratio + recall/F1) -- no method dominates; GAT top-k looks most balanced but NOT statistically confirmed

Follow-up to the fixed-hub-seed artifact entry directly below: re-ran the full
6-method x 3-level comparison with the corrected harness (5 seed nodes --
hub, mid, 3 random draws -- containment ratio per seed averaged, recall/F1 as
the utility metric). Full numbers in `demo_milestone2_pruning.py`'s output;
summary (containment ratio: lower = more contained, 1.0 = no effect):

| method | 10% | 25% | 50% |
|---|---|---|---|
| GAT threshold | 0.537+/-0.122 | 0.129+/-0.082 | 0.047+/-0.024 |
| GAT top-k | 0.402+/-0.164 | 0.108+/-0.012 | 0.031+/-0.007 |
| degree-centrality | 0.395+/-0.242 | 0.104+/-0.060 | 0.031+/-0.020 |
| betweenness | 0.451+/-0.062 | 0.110+/-0.069 | 0.037+/-0.019 |
| highest-p_uv | 0.379+/-0.205 | 0.110+/-0.064 | 0.031+/-0.009 |
| random | 0.703+/-0.125 | 0.424+/-0.151 | 0.106+/-0.073 |

**What changed:** degree-centrality's dramatic single-hub-seed "dominance"
(0.003 containment) is gone -- averaged across seeds it's statistically
indistinguishable from several other methods at every level (heavily
overlapping std bands, e.g. 10%: degree 0.395+/-0.242 vs. p_uv 0.379+/-0.205).
**Sanity check passed:** random pruning is the clear worst at every level,
confirming targeted pruning has genuine value over blind removal even after
the seed-artifact correction -- the fix didn't erase the whole result, just
the specific "degree-centrality wins" claim.

**Tentative read (single graph, 5 seeds, single model-seed per utility run --
NOT yet the statistically rigorous comparison Milestone 3 will do; treat as
directional, not confirmed):**
- GAT top-k-per-node looks like the most balanced performer: competitive-to-
  best containment at every level AND the best-or-tied recall/F1 at higher
  pruning (0.417 recall, 0.556 F1 at 50% -- the highest F1 anywhere in the
  table).
- GAT threshold (the OTHER GAT-based mechanism, same underlying score) is
  consistently the WEAKEST of the targeted methods on containment at every
  level, despite preserving utility perfectly (identical to baseline) up to
  25% removal -- the two GAT pruning mechanisms behave quite differently
  from the same score, and top-k is the stronger of the two here.
- highest-p_uv is strong on security at low pruning (best of all methods at
  10%) but its recall drops sharply at 25%+ (0.250, tied-worst) -- plausibly
  because cutting high-p_uv edges also removes some of the structural signal
  the classifier relies on for the compromised class.
- Do not treat any of these as confirmed: most pairwise differences sit
  within a standard deviation of each other (e.g. degree-centrality and
  highest-p_uv are within noise of each other at every level tested).
  Confirming any of this needs Milestone 3's harness -- many graphs, paired
  significance tests, multiple-comparison correction.

## 2026-07-11 -- Milestone 2 pruning: fixed-hub-seed infection is a methodology trap; fixed going forward (KEY METHODOLOGICAL RESULT)

The first pass of static-pruning evaluation (`demo_milestone2_pruning.py`)
always seeded infection at the fixed highest-degree hub node. Under that
setup, degree-centrality pruning looked dramatically dominant (infected
fraction ~0.003, near-total containment, at every pruning level from 10% up)
while every other method landed in the 0.02-0.29 range. This turned out to be
a methodology artifact, not a real result, confirmed two ways:

1. **Hub-edge-removal diagnostic** (`edge_removal_fraction_for_node`): at
   EVERY pruning level tested (10%, 25%, 50%), degree-centrality pruning
   removed **100%** of the fixed hub node's own edges -- even at 10% overall
   graph-wide removal. Betweenness-centrality did nearly the same (63% at
   10%, 100% by 25%). Mechanically this is inevitable: avg-endpoint-degree
   ranking puts every one of the hub's ~60 edges near the top of the global
   score, so even a small removal budget gets entirely consumed by the hub's
   own edges before touching anyone else's. Random pruning, as a clean
   reference point, tracked the target removal rate almost exactly at the
   hub (11.7% removed at a 10% target) -- confirming it isn't favoring the
   hub at all, which is what "no artifact" looks like.

2. **Multi-seed robustness check**: re-seeding infection from a random node
   and a mid-degree node (instead of only the hub) made degree-centrality's
   advantage shrink and, at low pruning levels, REVERSE relative to other
   methods. At 10% pruning, mid-seeded infected fraction: degree-centrality
   0.049 vs. highest-p_uv 0.008 -- p_uv contained mid-seeded spread over 6x
   better than degree-centrality, the opposite of the hub-seeded picture.

**Conclusion:** degree-centrality's apparent dominance in the original
single-hub-seed results was substantially a fixed-seed artifact (it isolates
whichever node happens to be seeded, if that node is high-degree), not
evidence of generally superior containment. This does not mean
degree-centrality pruning is worthless -- isolating a hub IS a real
containment strategy in scenarios where the attacker is known to target
hubs -- but the single-seed comparison overstated and mischaracterized why it
worked, and this needs to be corrected before any method is judged to "win."

**Fix applied going forward** (see `src/milestone2_pruning.py` /
`demo_milestone2_pruning.py` after this entry's date): every future security
comparison averages over multiple infection-seed choices per graph (hub, a
mid-degree node, and several random draws) -- the same averaging discipline
already applied to botnet rollouts (`generate_labeled_graph`'s
n_rollouts), model-init seeds, and train/test splits elsewhere in this
project. Raw infected fractions are also not directly comparable across seed
choices with very different baseline spread rates (e.g. hub-seeded ~0.36
unpruned vs. mid-seeded ~0.12 unpruned), so results are reported as a
containment ratio (infected_pruned / infected_unpruned, matched per seed)
rather than raw fractions going forward.

Additionally, plain accuracy was confirmed NOT to discriminate between
pruning methods in the first pass -- it stayed in a narrow 0.800-0.867 band
across all 18 pruned configurations, while recall (0.250-0.417) and F1
(0.333-0.556) varied far more in relative terms across the same cells. Recall
and F1, not accuracy, are the utility metrics used for pruning comparisons
going forward -- the same lesson already learned for the base classifier
(see the entry below).

## 2026-07-11 -- Milestone 2 attention scores: raw s_uv is degree-confounded; correction only partially explains the negative correlation

Raw attention-based edge scores (`extract_edge_attention_scores`,
`src/milestone2.py`) turned out strongly, negatively correlated with
p_uv/betweenness/hub_score. A scatter of s_uv against hub_score showed a
clean 1/x decay -- the signature of softmax normalization mechanically
diluting attention on high-degree target nodes (more neighbors = smaller
average share each, independent of the model's actual learned preference),
not a considered judgment by the GAT. Multiplying each directed alpha_ij by
the target's degree (`extract_degree_corrected_attention_scores`) cancels
that mechanical effect and gives a "relative preference vs. uniform
attention" score (~1.0 = uniform share, >1 = favored, <1 = disfavored).

| metric | raw Pearson r | corrected Pearson r | raw Spearman r | corrected Spearman r |
|---|---|---|---|---|
| p_uv | -0.490 | -0.276 | -0.521 | -0.417 |
| edge betweenness centrality | -0.579 | -0.311 | -0.667 | -0.481 |
| avg endpoint hub_score | -0.616 | -0.220 | -0.791 | -0.504 |

**Honest conclusion: the correction is not a full explanation, and the
correlation did not disappear or flip.** Pearson (linear) correlations
roughly halved -- the correction clearly removed the dominant mechanical
artifact (the 1/x shape is gone from the scatter plot) -- but every
correlation remains negative and highly significant (worst case p=2.9e-11),
and Spearman (rank/monotonic) correlations dropped much less than Pearson
did. Visually, the residual effect is concentrated at the extreme high-degree
end: edges touching the very highest-hub_score nodes cluster tightly BELOW
the uniform baseline (~0.2-0.3x), while lower/mid-degree edges scatter widely
in both directions (0.2x to 8x). This reads as a real, if attenuated, learned
de-prioritization of the highest-degree hub edges specifically -- not pure
noise, and not solely a normalization artifact either. See
`demo_milestone2_attention.py` for the full methodology and both sets of
plots (`attention_raw_*` vs `attention_corrected_*`).

**Implication for pruning:** raw s_uv should not be used for pruning
decisions (it would disproportionately target hub-adjacent edges purely from
the dilution artifact, which are also the highest-p_uv edges -- the opposite
of sound security pruning). `extract_degree_corrected_attention_scores` is
the version used for static pruning going forward, with this residual
high-degree de-prioritization noted as a real, not fully understood,
property of what the trained GAT learned -- worth revisiting once more
graphs are available (Milestone 3) rather than over-interpreting from one
graph instance.

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
