# Notes / known limitations

Running log of findings that need to survive into the final write-up but
don't belong in code comments or README. Newest entries at the top.

## 2026-07-11 -- Milestone 4 stress tests RESULTS: core finding holds robustly across size/density, BREAKS DOWN under aggressive infection (KEY CONDITIONAL FINDING FOR THE FINAL WRITE-UP)

6-condition stress test (`demo_milestone4_stress_tests.py`, 15 graphs/
condition -- reduced from the main harness's 40, lower statistical power,
reported honestly as such throughout). Reused the existing n=300/m=3/
default-beta 40-graph data as the baseline reference (RL 0.072+/-0.043 vs.
structural 0.033-0.055, RL significantly worse than all 5). Scope trims:
alternate RL reward weights and the greedy oracle dropped from all 6 new
conditions (too expensive to scale x6; not central to this question).

**Size and density: core finding HOLDS, and strengthens.**

| condition | RL containment | structural range | RL significantly worse vs. |
|---|---|---|---|
| size=150 | 0.126+/-0.068 | 0.057-0.092 | 4/5 (all but GAT threshold, p=0.11) |
| size=600 | 0.051+/-0.037 | 0.019-0.032 | 4/5 (all but GAT threshold, p=0.07) |
| density sparse (m=2) | **0.570+/-0.396** | 0.142-0.181 | **5/5, dramatically** (diff 0.39-0.43) |
| density dense (m=5) | 0.243+/-0.087 | 0.042-0.132 | 5/5 |

Sparse graphs are the most dramatic gap seen anywhere in this project --
RL's huge variance there (+/-0.396) echoes the SAME training-instability
signature seen in the reward-weighting ablation (aggressive w_security
collapsing to near-zero pruning): this looks like RL/DQN training itself
becomes fragile on sparse graphs (fewer edges = less room for the
bucket-based action space to work with?), not just a modest underperformance.
Worth investigating further if pursuing RL as a serious approach -- this is
a second, independent condition (sparsity, not just reward scale) that
triggers the same fragility pattern.

**Aggressiveness: core finding WEAKENS and PARTIALLY REVERSES.**

| condition | RL vs GAT threshold | RL vs GAT top-k | RL vs degree | RL vs betweenness | RL vs highest-p_uv |
|---|---|---|---|---|---|
| moderate (bias=-1.0) | tied (p=0.47) | tied (p=0.23) | RL worse (p<0.0001) | RL worse (p<0.0001) | RL worse (p<0.0001) |
| high (bias=0.0) | **RL BETTER (p=0.010)** | RL worse (p=0.010) | RL worse (p=0.008) | tied (p=0.56) | tied (p=0.56) |

At moderate aggressiveness, RL catches up to (ties) both GAT-based methods
specifically, while still losing clearly to the three non-GAT heuristics.
At high aggressiveness, RL is significantly BETTER than GAT threshold and
statistically tied with betweenness and highest-p_uv -- only GAT top-k and
degree-centrality still clearly beat it. **This is a genuine, real reversal
for at least one comparison (RL vs. GAT threshold), not just "ties appear."**

**Caveat, stated honestly rather than oversold:** under high aggressiveness,
EVERY method's containment degrades substantially (0.36-0.63, vs. 0.02-0.13
in every other condition) and frozen-F1 saturates near 1.0 for ALL methods
(0.995-0.998) -- the classification task itself becomes near-trivial when
almost every node ends up infected in the majority-vote labeling. Some of
the apparent "reversal" may be a ceiling/floor effect (everyone struggles
comparably against near-unstoppable infection, so the structural methods'
edge shrinks) rather than RL becoming genuinely more capable at high
aggressiveness. Both readings are plausible from this data; distinguishing
them would need finer-grained aggressiveness levels between the default and
bias=0.0 extreme, which this pass didn't run.

**Overall, honest answer to "is the pattern consistent or condition-
dependent": CONDITION-DEPENDENT, cleanly split by dimension.** Structural
heuristics beating RL on containment is a robust, even strengthening
finding across graph size and density -- NOT an artifact of the one graph
configuration used throughout Milestones 2-3. But it is NOT a universal law:
it measurably weakens and partially reverses as the botnet becomes more
aggressive, specifically for the GAT-based methods. The final write-up
should state the size/density robustness plainly and present the
aggressiveness reversal as a real, if partially confounded, boundary
condition -- not bury it, and not claim more generality for "structural
beats learned methods" than the size/density data actually supports.

## 2026-07-11 -- Milestone 4 reward ablation RESULTS: no weighting closes RL's containment gap; pushing security actively backfires; utility edge is real but only against weaker baselines (KEY FRAMING RESULT FOR THE FINAL WRITE-UP)

## 2026-07-11 -- Milestone 4 reward ablation RESULTS: no weighting closes RL's containment gap; pushing security actively backfires; utility edge is real but only against weaker baselines (KEY FRAMING RESULT FOR THE FINAL WRITE-UP)

40-graph ablation (`demo_milestone4_reward_ablation.py`, design/scope in the
smoke-test entry directly below) completed in 2820s (47 min). RL baseline's
containment ratio (0.072+/-0.043) exactly reproduces the original harness
run -- good cross-run consistency check, same seeds/procedure.

**Per-method mean +/- std, n=40:**

| method | containment_ratio | frozen_f1 |
|---|---|---|
| RL baseline (sec=1, util=1) | 0.072+/-0.043 | 0.543 |
| RL high-security (sec=3, util=1) | 0.153+/-0.287 | 0.566 |
| RL very-high-security (sec=10, util=1) | 0.930+/-0.250 | 0.513 |
| RL security-only (sec=1, util=0) | 0.978+/-0.139 | 0.501 |
| GAT threshold / top-k / degree / betweenness / highest-p_uv | 0.033-0.055 | 0.423-0.562 |
| random | 0.122+/-0.018 | 0.466 |

**The single-graph smoke test result was misleading -- exactly the failure
mode multi-graph testing exists to catch.** On the spot-check graph,
sec=3.0 looked like a clean win (containment 0.051->0.035, better on both
axes). At 40-graph scale it is WORSE on average (0.153) with enormous
variance (+/-0.287) -- helps on some graphs, doesn't on many, sometimes
collapses. sec=10.0 and security-only (util=0) both confirm the training
collapse found during smoke-testing generalizes broadly across graphs
(0.93-0.98, i.e. essentially unpruned).

**Containment gap: does NOT close, and pushing harder backfires.**
High-security's differences vs. every structural method (diff 0.098-0.120,
RL worse) do not survive Holm correction (holm_p~0.059, just above 0.05)
-- the added instability/variance prevents confirming the direction at
this n, even though it points the same way as baseline's already-confirmed
underperformance. Very-high-security and security-only are catastrophically
and SIGNIFICANTLY worse than every structural method (diffs 0.87-0.95,
holm_p<0.0001 in all 10 comparisons).

**RL's frozen-utility edge: real, but only against the weaker baselines,
not the strongest ones.** Newly paired-tested (previously only reported
descriptively): RL baseline significantly beats betweenness-centrality
(holm_p=0.0060), highest-p_uv (holm_p=0.0001), and random (holm_p=0.0010)
on frozen F1 -- but is statistically indistinguishable from GAT threshold
(holm_p=1.0), GAT top-k (holm_p=1.0), and degree-centrality (holm_p=1.0),
the three strongest utility performers.

**Recommended framing for the final write-up:** RL's containment
underperformance is NOT a simple reward-tuning problem -- naive attempts to
push security harder within this training setup actively destabilize
learning rather than closing the gap, and this destabilization (STOP being
a low-variance, easy-to-learn terminal-action target vs. noisy bootstrapped
"continue" targets) generalizes robustly across graphs, not a single-graph
fluke. RL's genuine, defensible strengths are narrower than "RL wins":
statistically tied with the (much more expensive) greedy oracle on
containment, reliably beats random, and has a real but partial utility
edge over the weaker structural baselines specifically. A write-up that
says "RL needs a different fix (reward shaping/normalization, a different
algorithm, or a redesigned action space), not just different weights" is
the defensible claim here -- not "reward tuning didn't help" understated,
and not "RL is worse across the board" overstated either.

## 2026-07-11 -- Milestone 4 reward ablation: aggressive w_security COLLAPSES DQN training (found during smoke-test, before committing 40-graph compute)

## 2026-07-11 -- Milestone 4 reward ablation: aggressive w_security COLLAPSES DQN training (found during smoke-test, before committing 40-graph compute)

While spot-checking planned weight configs before the full 40-graph
ablation run, found that pushing w_security too high breaks training
entirely -- converges to choosing STOP at step 0 every time (containment
ratio = 1.000, i.e. literally zero pruning), REGARDLESS of whether
w_utility is kept:

| config | mean return, first 20 episodes | mean return, last 20 | outcome |
|---|---|---|---|
| sec=1.0, util=1.0 (baseline) | ~3.9 | ~12.7 | prunes to budget cap, as established |
| sec=3.0, util=1.0 | -- | -- | works: containment 0.051->0.035 (better), frozen_f1 0.500->0.737 (also better) |
| sec=5.0, util=1.0 | -9.1 | -3.9 | COLLAPSES to never pruning |
| sec=10.0, util=1.0 | -24.8 | -9.0 | COLLAPSES to never pruning |
| sec=1.0, util=0.0 | -3.25 | -1.0 (exactly, every episode) | COLLAPSES to never pruning |

**Why, mechanistically:** STOP is a terminal action, so its Q-value target
is just the immediate reward -- no bootstrapping, a fast, low-variance,
easy-to-learn estimate. "Continue pruning" targets require correctly
propagating value through a noisier multi-step trajectory, which takes
longer to converge. Early in training, before those downstream estimates
are any good, an inflated or unshaped reward magnitude (very high
w_security, or a reward with the utility term's shaping signal removed)
makes the immediately-known "safe" STOP value look competitive well before
the network has learned that continuing is actually better -- so the
policy collapses onto the easy, low-information option before ever
discovering the eventually-larger payoff of pruning. This is a DQN
training-stability failure mode, not evidence that "doing nothing" is
reward-optimal (mathematically, ~50%-pruned structural-level containment
SHOULD beat the -1.0 no-pruning reward under this formula).

**Consequence for the ablation design:** dropped the originally-planned
"extreme security" (sec=10, util=0, a compound/confounded condition) in
favor of testing the two dimensions separately: sec=3.0 (moderate,
validated working) and sec=10.0 (aggressive, expected to collapse) both
WITH utility kept, plus sec=1.0/util=0.0 (utility removed at baseline
security scale) as its own condition. This directly tests whether the
40-graph run confirms the collapse generalizes (or is graph-dependent) at
both ends, rather than running a single confounded extreme condition.

## 2026-07-11 -- Milestone 3 RL environment: utility-metric divergence from Milestone 2, and single-graph scope (both flagged BEFORE training, not discovered after)

## 2026-07-11 -- Milestone 4 harness RESULTS: 40-graph statistical confirmation -- RL loses to every structural method on containment, and the expensive methods don't earn their compute cost

The 40-graph harness (`demo_milestone4_harness.py`, scope decisions in the
entry directly below) completed in 2950s (49.2 min -- longer than the
~35-40 min estimate; graphs 6-31 took ~85-100s each vs. ~30-50s for the
rest, most plausibly structural variance across BA graph instances
affecting greedy-oracle search / top-k calibration cost, not a bug).

**Per-method mean +/- std, containment ratio (lower = more contained), n=40:**

| method | containment_ratio | frozen_recall | frozen_f1 | mean time/graph |
|---|---|---|---|---|
| RL | 0.072+/-0.043 | 0.511 | 0.543 | 26.44s |
| GAT threshold | 0.055+/-0.013 | 0.490 | 0.544 | 0.10s |
| GAT top-k | 0.037+/-0.006 | 0.452 | 0.532 | 0.52s |
| degree-centrality | 0.033+/-0.006 | 0.462 | 0.562 | 0.01s |
| betweenness-centrality | 0.036+/-0.006 | 0.363 | 0.455 | 0.44s |
| highest-p_uv | 0.037+/-0.005 | 0.341 | 0.423 | 0.01s |
| random | 0.122+/-0.018 | 0.406 | 0.466 | 0.02s |
| greedy oracle | 0.066+/-0.034 | 0.550 | 0.634 | **41.49s** |

Sanity check passes: random is clearly worst. Shared GAT training:
3.72s/graph (amortized across GAT threshold, GAT top-k, RL).

**"No structural method dominates" (Milestone 2) -- PARTIALLY OVERTURNED.**
GAT threshold is now significantly AND substantially worse than GAT top-k
(diff=0.018, holm_p<0.0001) -- a real finding multi-graph data newly
supports; single-graph runs had this ranking flip depending on which run.
Among the remaining cluster (degree-centrality, betweenness, highest-p_uv,
GAT top-k, all 0.033-0.037), pairwise differences ARE Holm-significant
(n=40 has power to detect them) but the absolute effect sizes are tiny
(<=0.004) -- statistically real, practically close to negligible. One pair
(highest-p_uv vs. betweenness) is genuinely indistinguishable (holm_p=0.91).

**"RL beats random/oracle but not structural methods" (Milestone 3) --
CONFIRMED AND STRENGTHENED.** RL significantly beats random (diff=-0.050,
large effect, holm_p<0.0001) and is statistically tied with greedy oracle
(holm_p=0.91) -- matching the tentative single-graph read exactly. But RL
is significantly WORSE than all 5 structural methods on containment
(diffs 0.017-0.039, all holm-significant, moderate-to-large effect sizes,
not tiny ones) -- this is stronger than "RL doesn't clearly win," it's "RL
reliably loses" on containment specifically, now with real statistical
weight behind it (previously this was one graph's worth of evidence).

**The compute-cost finding that matters most: greedy oracle (41.49s/graph)
is the single most expensive method -- MORE expensive than RL
(26.44s/graph) -- and does not significantly beat RL on containment.**
Every structural heuristic (0.01-0.52s/graph, up to ~4000x cheaper) beats
BOTH expensive methods on containment, significantly. Per this project's
own standard ("is the expensive method earning its cost"), the honest
answer for containment specifically is no: neither RL nor the greedy
oracle currently earns its compute cost over the cheap structural
heuristics, at this pruning level, on this graph family.

**What this does NOT cover** (see the scope-decisions entry below): frozen
utility only (RL's actual advantage there -- 2nd-highest recall/F1 in the
table, behind only greedy oracle -- has NOT been paired-tested here, only
reported descriptively); Milestone 2's retrained-utility comparison remains
single-graph and unconfirmed at this scale; single pruning level (50%)
only, not swept.

## 2026-07-11 -- Milestone 4 multi-graph harness: scope decisions made BEFORE running, not discovered after

Building the 40-graph statistical harness (the foundation for all of
Milestone 4's ablations/stress-tests). Three compute-driven trade-offs,
decided and recorded before the harness ran, not rationalized after seeing
results:

1. **Frozen utility only, not retrained.** Retraining a GAT per method per
   graph (measure_utility, ~3.6s/call) x 8 methods x 40 graphs is ~19
   minutes on its own. Frozen utility (measure_utility_frozen, ~1ms/call)
   is functionally free. **Consequence: Milestone 2's original
   retrained-utility comparison (e.g. "GAT threshold has the best F1 at
   50%") remains a single-graph, tentative finding and is NOT re-validated
   by this harness.** If retrained utility needs multi-graph validation
   later, that is a separate, additional compute cost, not something this
   harness's results can be read as covering.
2. **RL trained for 100 episodes/graph, not 300.** Spot-checked on the
   standard graph before committing across all 40: both settings converge
   to the same qualitative "prune to the removal budget cap" behavior
   (stopped_early=False either way), mean episode return improves
   substantially under both (100ep: 4.5->10.9; 300ep: 3.9->12.7), and
   replay-buffer deep-trajectory coverage is comparable (0.160 vs 0.169).
   300 episodes does reach a somewhat higher final return, as expected --
   this is a real, acknowledged fidelity/cost trade-off, not a free lunch.
3. **Single pruning level (50%)**, matching Milestone 3's existing
   RL-vs-baselines comparison, rather than sweeping multiple levels per
   graph across all 40 instances.

See the harness results entry (same date, above this one once posted) for
the actual multi-graph numbers.

## 2026-07-11 -- Milestone 3 DQN: confirmed reward-driven "always max-prune" policy (sets up Milestone 4's reward-weighting ablation), plus RL vs. static baselines comparison

**Exhausted-bucket action-mask fix.** The trained greedy policy was getting
stuck repeatedly "choosing" an already-fully-pruned attention bucket (a
no-op) for several steps until `max_steps` was hit, instead of reaching the
removal budget cleanly. Fixed by adding `PruningEnv.action_mask()` (True for
buckets with >=1 remaining edge, STOP always True) and threading it through
`select_action` (both epsilon-greedy exploration and the final greedy
policy now only ever pick a currently-valid action). Also applied to the
earlier mechanics-only random-policy demos for consistency. After the fix,
the greedy rollout reaches the removal budget in clean, distinct steps
(0.503 final, matching the 0.5 cap) instead of repeating a dead action.

**Confirmed finding: under w_security=1.0/w_utility=1.0/w_cost=0.1, the
learned policy converges to always pruning to the removal budget cap.**
This was checked, not assumed -- see the DQN monitoring writeup: replay
buffer had 14-17% of transitions from deep (>40% removed) trajectories
throughout training, deep trajectories were first seen at episode 0 (169
episodes before epsilon decayed below 0.2), and 44% of TRAINING episodes
(behavior policy) chose STOP before the budget. None of that points to an
exploration gap. Directly inspecting the trained Q-network's STOP-vs-continue
estimate at 8 actual deep-state transitions it trained on: continuing to
prune wins in all 8 cases, several by 3-6x margins (e.g. Q(STOP)=1.953 vs.
Q(continue)=6.345). **This is a genuine learned preference given the current
reward weights, not insufficient training or exploration** -- the network
correctly learned that on this graph, containment and utility keep
improving enough with more pruning that the small (0.1-weighted) cost term
never catches up. Per explicit instruction, reward weights were NOT
retuned to chase a "more adaptive-looking" policy -- reward-weighting
comparison is Milestone 4's ablation task, not Milestone 3's. This finding
is exactly the starting point that ablation needs.

**RL vs. static baselines, at the matching ~50% removal level (the roadmap's
"compare RL-based pruning versus static thresholds" task).** Every method
scored on containment ratio (same 5-seed x 15-rollout methodology used
throughout) AND on BOTH utility definitions (frozen-base-GAT forward pass,
matching what the RL reward actually optimizes; and Milestone 2's original
retrained-GAT utility), per the plan recorded when this divergence was
first flagged -- so this is the first comparison in the project where that
plan is actually carried out, not just noted for later.

| method | removed | containment_ratio | frozen_recall | frozen_f1 | retrained_recall | retrained_f1 |
|---|---|---|---|---|---|---|
| RL adaptive pruning (DQN) | 0.503 | 0.060+/-0.04 | **0.500** | **0.609** | 0.429 | 0.522 |
| GAT threshold | 0.501 | 0.041+/-0.01 | 0.214 | 0.353 | **0.571** | **0.696** |
| GAT top-k-per-node | 0.558 | 0.041+/-0.02 | 0.357 | 0.526 | 0.500 | 0.636 |
| degree-centrality | 0.501 | 0.039+/-0.03 | 0.286 | 0.444 | 0.286 | 0.444 |
| betweenness-centrality | 0.501 | 0.044+/-0.03 | 0.214 | 0.353 | 0.286 | 0.444 |
| highest-p_uv | 0.501 | **0.036+/-0.01** | 0.000 | 0.000 | 0.143 | 0.222 |
| random | 0.501 | 0.117+/-0.07 | 0.476 | 0.602 | 0.310 | 0.432 |
| greedy oracle | 0.501 | 0.088+/-0.07 | 0.286 | 0.444 | 0.357 | 0.500 |

**Reading this honestly, on each axis separately:**
- **Containment**: RL is NOT the best -- it's worse than all 5 static
  structural methods (0.060 vs. 0.036-0.044), though clearly better than
  random (0.117) and the greedy oracle (0.088). Expected: RL is balancing
  three reward terms at once, not purely optimizing containment the way
  "remove the highest-X edges" methods do.
- **Frozen utility (what the RL reward actually optimizes)**: RL has the
  BEST recall/F1 in the whole table (0.500/0.609) -- the training pipeline
  is doing what it was told to do. random is a surprisingly close second
  (0.476/0.602).
- **Retrained utility (Milestone 2's original metric)**: RL is mid-pack,
  3rd of 8 (0.429/0.522), behind both GAT threshold (0.696 F1) and GAT
  top-k (0.636 F1).
- No method wins on every axis. This is a single graph, single RL training
  run -- same caveats as every other single-graph comparison in this
  project apply here too; Milestone 4's proper multi-graph harness is
  still where a real "RL beats/loses to static pruning" claim would need
  to be established.

**Notable side-finding: highest-p_uv's frozen utility collapses to
0.000/0.000 recall/F1 at 50% removal**, despite having the BEST
containment ratio (0.036) and non-trivial RETRAINED utility (0.222 F1).
This is a concrete, striking illustration of exactly why the
frozen-vs-retrained utility divergence needed to be resolved before any
final comparison: removing the highest-p_uv edges apparently strips out
structural signal the FROZEN model specifically depended on, while a
FRESH retrained model can partially relearn around their absence. A method
that looks reasonable under one utility definition can look catastrophic
under the other, for the identical pruned graph -- worth a dedicated
mention in the final write-up, not just a footnote.

**Sanity-checked, not assumed: the 0.000 frozen recall is genuine, not a
divide-by-zero bug.** Raw counts on the highest-p_uv 50%-pruned graph:
`total_positive=14` (real positives exist in the test set), `predicted_positive=1`,
`true_positive=0` -- the frozen model collapsed to predicting almost every
test node as benign (only 1 positive prediction out of 60) and that one
prediction was wrong. `test_acc=0.75` looks fine only because accuracy
rewards the collapse-to-majority-class behavior -- the same accuracy-vs-recall
trap flagged elsewhere in this project, now visible in the frozen-utility
metric specifically.

## 2026-07-11 -- simulate_botnet neighbor-order bug: found, fixed, and every affected number rerun (MOST SIGNIFICANT METHODOLOGICAL FIX THIS SESSION)

**Found while building Milestone 3's environment.** `PruningEnv.reset()`
expected its episode-start containment ratio to read ~1.0 (nothing pruned
yet, should match the unpruned baseline exactly). It read 0.986 instead.
Investigating rather than shrugging it off: `nx.Graph.copy()` does NOT
preserve neighbor-iteration order (verified: 35/60 nodes had reordered
adjacency after `.copy()`, despite an identical edge set), and
`simulate_botnet` iterated `graph.neighbors(current)` in whatever order the
adjacency dict yielded, consuming one `rng.random()` draw per neighbor in
that order -- so a reordered adjacency pairs the same random draws with
different neighbors, changing the outcome for an identical seed. (Copies of
the same graph are internally CONSISTENT with each other, just different
from the original -- this is not run-to-run randomness, it's a deterministic
dependency on incidental object-construction history that shouldn't exist.)

**Fix:** `simulate_botnet` now sorts neighbor iteration too (previously only
`frontier` was sorted). Verified fixed: original graph and `.copy()` of it
now give byte-identical results for the same seed.

**Scope, precisely checked rather than assumed:** grepped every
`simulate_botnet` call site.
- UNAFFECTED by the copy-order bug specifically: `demo_milestone1.py`,
  both `tests/test_milestone1.py` calls, `generate_labeled_graph`
  (src/milestone2.py) -- none of these call simulate_botnet on a `.copy()`'d
  graph.
- HOWEVER: the FIX ITSELF (sorting neighbors) changes simulate_botnet's
  output relative to the OLD behavior for basically any graph, copied or
  not, since old behavior used native/insertion adjacency order and the new
  behavior uses sorted order -- these generally differ. The one exception:
  node 0 in a Barabasi-Albert graph can only ever be a "later-attached-to"
  node (it has zero neighbors at creation), so its adjacency list is built
  up in strictly increasing node-index order over time -- already sorted by
  construction. This is why `demo_milestone1.py`'s node-0-seeded rollout
  happened to be invariant; no other node has that property in general.
- CONFIRMED via direct comparison (`git stash` to diff old vs. new
  `generate_labeled_graph` output on the identical graph): labels changed
  (61/300 infected pre-fix vs. 60/300 post-fix, different per-node
  assignment, not just count) -- because label generation seeds infection
  from the HUB node, not node 0. This cascades into every downstream
  Milestone 2 number: GAT training, attention scores, recall/F1/precision,
  the pruning comparison table, the greedy oracle, the convergence plot --
  essentially everything, not just `demo_milestone2_pruning.py` as
  originally (incorrectly) scoped.

**Full test suite (22 tests) still passes** -- these check structural
properties (counts, label validity, shape, monotonicity), not exact
numeric equality, so none needed changes.

**Every affected demo was rerun and every NOTES.md entry below with
specific numbers has been updated in place** (each marked "[numbers updated
post-fix]" in its heading) rather than left stale: the Milestone 2 mildness
comparison, the attention-score correlation, the training convergence plot,
and the full pruning comparison (including a full redo of the 1-vs-5-rollout
greedy-oracle diagnostic, ~12 more minutes of compute). In every case the
QUALITATIVE conclusion held or got cleaner -- except the pruning
comparison's "GAT top-k looks most balanced" tentative lean, which did NOT
survive and has been corrected to a more conservative reading (see that
entry). This is itself a useful data point for the final write-up: a single
upstream implementation bug was able to flip which heuristic looked best in
a single-graph comparison, which is exactly the kind of fragility Milestone
3's proper multi-graph statistical harness needs to guard against.

## 2026-07-11 -- Milestone 3 RL environment: utility-metric divergence from Milestone 2, and single-graph scope (both flagged BEFORE training, not discovered after)

Two design decisions made while formalizing the RL environment, recorded now
so they don't get lost or misread as oversights later:

**Utility metric divergence.** Milestone 2's `measure_utility` retrains a
fresh GAT from scratch on each pruned graph (~3.6s/call) -- fine for a
one-off comparison table, but far too slow to call at every RL step across
hundreds of episodes. The RL reward's utility term instead uses the
ALREADY-TRAINED base GAT's forward-pass performance on the current pruned
topology -- no retraining, ~1ms/call, and (since the model is frozen and
eval-mode is deterministic) zero added randomness. This is a deliberate,
tractability-driven choice, not an oversight, and arguably more realistic
for an adaptive agent (a live system doesn't retrain its classifier after
every edge cut) -- but it means Milestone 2's baseline numbers
(retrain-based utility) and Milestone 3's RL numbers (frozen-forward-pass
utility) are NOT currently on the same utility definition. **Plan:** before
any final "RL vs. static baselines" comparison, also compute the
frozen-base-GAT forward-pass utility for Milestone 2's baseline methods
(random/degree/betweenness/highest-p_uv/GAT-threshold/GAT-top-k/greedy
oracle), so the comparison is apples-to-apples on whichever utility
definition is used. Do not compare Milestone 2's retrain-based numbers
directly against Milestone 3's frozen-forward-pass numbers without doing
this first.

**Single-graph scope.** Like Milestone 2's pruning evaluation, the RL
environment trains and evaluates on ONE fixed graph instance (the same
n=300 graph used throughout). Generalization across different graph
instances/topologies is explicitly Milestone 4's stress-test concern, not
Milestone 3's. Any "the agent learned X" or "RL beats static pruning"
claim from this environment needs the same caveat Milestone 2's pruning
results already carry: directional evidence from one graph, not a
validated general result.

**Reward-evaluation compute cost (checked empirically before committing to
the training loop, per this project's own standard for expensive methods):**
one full 5-seed x 15-rollout security measurement takes ~7.2ms; the frozen
base-GAT forward pass for utility takes ~1.1ms. Combined ~8.3ms per reward
evaluation. A realistic training run (200-500 episodes x 10-20 steps) is
therefore ~17-83s of reward-evaluation cost alone -- comfortably tractable
at FULL Milestone-2-grade rigor. Unlike the greedy oracle, this does NOT
need a cheap-search-signal-vs-full-report split; the full 5-seed/15-rollout
measurement is used throughout training, not just at final evaluation.

## 2026-07-11 -- Milestone 2 training convergence: plateaus well before 300 epochs [numbers/description updated post-fix]

Added optional per-epoch train/val loss+accuracy logging to `train_gat`
(`log_history=True`, off by default -- see `demo_milestone2_convergence.py`
and `training_curves.png`). **Rerun post the simulate_botnet neighbor-order
fix** (different labels -> different representative run). On the current
run (n=300 nodes, split seed 0): both losses drop from ~0.66 to ~0.49 over
the first ~150 epochs (validation drops faster and plateaus earlier, around
epoch 40-50; train catches up by ~150-200), then both hover in a noisy
plateau (~0.48-0.52) for the remaining epochs with no clear separating gap
-- train and validation stay close together throughout, sometimes train
even slightly above validation, which is actually LESS evidence of
overfitting than the pre-fix run showed (which had validation loss
drifting up above a still-decreasing train loss). Best val_loss now at
epoch 118 (vs. epoch 281 pre-fix), but as before the whole tail is a noisy
plateau, not a single clear minimum. Practical takeaway unchanged: 300
epochs is more than strictly needed for convergence on this graph size;
kept as the default anyway since the cost is small and this hasn't been
tuned/validated across multiple graphs yet (that's Milestone 3/4 territory).

## 2026-07-11 -- Milestone 2 greedy oracle: does NOT behave as an upper bound at ANY pruning level tested (search-signal noise ruled out) [numbers updated post-fix]

Added the supervisor-requested greedy oracle baseline
(`greedy_oracle_prune`, `src/milestone2_pruning.py`): at each step, remove
whichever remaining edge most reduces simulated infection (averaged across
the same 5 seed nodes used everywhere else), one at a time. A full O(steps x
remaining_edges) search to 50% removal needs ~300K candidate evaluations on
this ~900-edge graph -- empirically ~25-31s total (all 3 checkpoints, one
incremental run) using in-place edge remove/restore (not graph copying) and
a cheap 1-rollout-per-candidate search signal. The checkpoint graphs it
returns ARE re-measured with the same full-rollout rigor as every other
method for the reported numbers -- only the SEARCH itself uses the cheap
signal.

**Numbers below are POST the simulate_botnet neighbor-order fix** (see that
dated entry). The finding is the SAME shape, if anything stronger:

**Result, stated plainly (see the updated containment_ratio_vs_pruning_level.png):**
the oracle is NOT the best-performing method at ANY of the three pruning
levels tested. At 10% removal its containment ratio (0.634+/-0.101) is worse
than all 5 structural methods (0.433-0.508), beating only random (0.787). At
25% it's still clearly worse than every structural method (0.270 vs.
0.119-0.188). At 50% -- where the PRE-FIX numbers showed it converging with
the pack -- it is now still clearly worse (0.088 vs. 0.036-0.044 for the
structural methods, roughly 2x), not converging at all. It also no longer
has the best F1 in the table -- GAT threshold does now (0.696 at 50%,
greedy oracle's is 0.500). The "only pays off at higher pruning levels"
softening from the pre-fix version of this entry did not survive: in the
corrected numbers, the oracle simply underperforms the structural
heuristics at every level checked, on both axes.

**Follow-up diagnostic: confirmed as myopia, not search noise -- and the
post-fix rerun is an even cleaner confirmation than the original.**
Reran the search with search_rollouts=5 instead of 1 (everything else
identical -- same graph, seeds, checkpoints; final numbers still measured at
full rigor). If noise were the main cause, more rollouts should have
consistently closed the gap at every level. It didn't -- and post-fix, more
rollouts made things WORSE at every single level, not just at 10%:

| level | search_rollouts=1 | search_rollouts=5 | other methods' range |
|---|---|---|---|
| 10% | 0.634 | **0.889 (much worse)** | 0.43-0.79 |
| 25% | 0.270 | **0.715 (much worse)** | 0.12-0.19 |
| 50% | 0.088 | **0.320 (much worse)** | ~0.04 |

(Pre-fix, this same diagnostic showed a MIXED picture -- worse at 10%,
barely better at 25%, better at 50% -- which was already enough to rule out
noise as the main driver. Post-fix, the direction is unanimous: 5x the
search accuracy makes the outcome worse at every level checked.) This
actually makes sense under the myopia explanation and not under the noise
explanation: a noisier 1-rollout signal sometimes fails to fully commit to
the locally-optimal-but-globally-poor edge at a given step (accidentally
hedging by chance), while a more accurate 5-rollout signal reliably finds
and commits to the TRUE locally-best edge every time -- which, if the
greedy strategy's failure mode is structural (not estimation error), makes
the outcome MORE consistently myopic, not less. A heuristic that ranks ALL
remaining edges by a global criterion at once (degree, betweenness, p_uv)
simply doesn't have this failure mode, regardless of how precisely any
single greedy step is scored. The 5-rollout search also cost ~719s vs.
~25-31s (~23-29x, far more than the naive 5x) for a WORSE outcome at every
level -- not pursuing higher rollout counts further, per the "confirm and
stop" scope of this diagnostic.

**Consequence for the write-up:** do NOT call this method an "upper bound"
without this caveat attached -- the plot/table label it plainly as "greedy
oracle" (no parenthetical claim) for exactly this reason. Post-fix, the
finding is if anything MORE clear-cut than before: report it as "a greedy,
outcome-driven baseline that underperforms simple structural heuristics at
every pruning level tested, because of its inherent one-step-at-a-time
myopia," not as a validated ceiling on achievable containment.

## 2026-07-11 -- Milestone 2 pruning: corrected re-run (multi-seed containment ratio + recall/F1) -- no method dominates, and the "GAT top-k most balanced" lean did NOT survive the simulate_botnet fix [numbers updated post-fix]

Follow-up to the fixed-hub-seed artifact entry directly below: re-ran the full
7-method x 3-level comparison (6 baselines + greedy oracle) with the
corrected harness (5 seed nodes -- hub, mid, 3 random draws -- containment
ratio per seed averaged, recall/F1 as the utility metric). Full numbers in
`demo_milestone2_pruning.py`'s output.

**These numbers are POST the simulate_botnet neighbor-order fix** (see that
dated entry) and supersede the table originally here. The ORIGINAL 6-method
table (pre-fix) was:

| method | 10% | 25% | 50% |
|---|---|---|---|
| GAT threshold | 0.537+/-0.122 | 0.129+/-0.082 | 0.047+/-0.024 |
| GAT top-k | 0.402+/-0.164 | 0.108+/-0.012 | 0.031+/-0.007 |
| degree-centrality | 0.395+/-0.242 | 0.104+/-0.060 | 0.031+/-0.020 |
| betweenness | 0.451+/-0.062 | 0.110+/-0.069 | 0.037+/-0.019 |
| highest-p_uv | 0.379+/-0.205 | 0.110+/-0.064 | 0.031+/-0.009 |
| random | 0.703+/-0.125 | 0.424+/-0.151 | 0.106+/-0.073 |

The CURRENT (post-fix), 7-method table:

| method | 10% | 25% | 50% |
|---|---|---|---|
| GAT threshold | 0.489+/-0.095 | 0.188+/-0.064 | 0.041+/-0.015 |
| GAT top-k | 0.461+/-0.123 | 0.142+/-0.064 | 0.041+/-0.016 |
| degree-centrality | 0.458+/-0.276 | 0.119+/-0.063 | 0.039+/-0.030 |
| betweenness | 0.508+/-0.075 | 0.125+/-0.071 | 0.044+/-0.026 |
| highest-p_uv | 0.433+/-0.232 | 0.135+/-0.104 | 0.036+/-0.010 |
| random | 0.787+/-0.111 | 0.486+/-0.193 | 0.117+/-0.073 |
| greedy oracle | 0.634+/-0.101 | 0.270+/-0.154 | 0.088+/-0.071 |

**Sanity check still passes:** random is still clearly worst at every level;
degree-centrality's dramatic single-hub-seed "dominance" is still gone
(statistically indistinguishable from the other structural methods). Both
core conclusions from the original corrected re-run survive the fix.

**What did NOT survive: the "GAT top-k looks most balanced" tentative lean.**
With the new numbers:
- Containment ranking shuffled: at 10%, highest-p_uv (0.433) is now best,
  not GAT top-k; at 25%, degree-centrality (0.119) is best; at 50%,
  highest-p_uv (0.036) is best again. GAT top-k is never clearly best on
  containment in the new numbers, just mid-pack.
- Utility flipped too: GAT THRESHOLD now has the highest F1 in the entire
  table (0.696 at 50%), not GAT top-k (0.636) -- the opposite of what the
  pre-fix run showed.
- GAT threshold is no longer consistently the weakest on containment either
  -- it's worst of the 5 structural methods at 25% (0.188), but NOT at 10%
  (betweenness is worse, 0.508) or 50% (betweenness is worst, 0.044, GAT
  threshold ties for 2nd).

**Updated, more conservative read:** no method -- including either
GAT-based mechanism -- shows a stable edge across both containment and
utility once the exact numbers are corrected. Which method looks "best"
depends on which level and which axis (containment vs. utility) you look
at, and that dependence itself flipped when a single upstream bug was
fixed. This is the clearest evidence yet in this project that single-graph,
few-seed comparisons like this one are NOT resilient enough to support a
"method X wins" claim -- treat every number here as directional pending
Milestone 3's proper multi-graph, paired-significance harness, more
emphatically than the pre-fix version of this entry did.

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

   (Note: point 1's 100%/63% hub-edge-removal numbers are a pure graph-
   structure/ranking fact -- no simulate_botnet call involved -- so they are
   UNAFFECTED by the later simulate_botnet neighbor-order fix described
   further down this file. Point 2's specific infected-fraction numbers
   (0.049/0.008) predate that fix and came from a since-superseded script
   version [the hub-edge-removal print was later dropped from
   demo_milestone2_pruning.py]; they were not re-verified bit-exact, but the
   qualitative finding they illustrate -- p_uv targets risk directly, so it
   isn't vulnerable to the same hub-isolation artifact degree-centrality is
   -- is a structural argument, not a numerical coincidence, and the LIVE,
   current numbers for this comparison are in the "corrected re-run" entry
   above, which IS fully updated post-fix.)

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

## 2026-07-11 -- Milestone 2 attention scores: raw s_uv is degree-confounded; correction only partially explains the negative correlation [numbers updated post-fix]

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

**Numbers below are POST the simulate_botnet neighbor-order fix** (see that
dated entry) -- rerun in full since the fix changes `generate_labeled_graph`'s
labels and therefore the trained GAT being analyzed. Same pattern, very
similar magnitudes; the plots (re-viewed) show the identical 1/x-gone,
high-degree-compression shape as before.

| metric | raw Pearson r | corrected Pearson r | raw Spearman r | corrected Spearman r |
|---|---|---|---|---|
| p_uv | -0.494 | -0.348 | -0.540 | -0.476 |
| edge betweenness centrality | -0.539 | -0.329 | -0.624 | -0.422 |
| avg endpoint hub_score | -0.591 | -0.272 | -0.772 | -0.481 |

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

## 2026-07-11 -- Milestone 2 GAT: minority-class recall is weak (~46%) [numbers updated post simulate_botnet fix]

The Milestone 2 GAT (benign vs compromised node classification,
`src/milestone2.py`) uses `train_gat`'s default `weight_mildness=0.5`, chosen
from a three-way comparison across 5 random train/val/test splits x 4
model-init seeds each, on one labeled Social IoT graph (n=300 nodes,
infected_fraction ~0.20, labels from `generate_labeled_graph`'s majority-vote
rollouts). See `demo_milestone2.py` for the full methodology and per-split
breakdown, and `train_gat`'s docstring for the summary numbers.

**Numbers below are POST the simulate_botnet neighbor-order fix** (see that
dated entry) -- the fix changes `generate_labeled_graph`'s labels (since it
seeds infection from the hub node, not node 0), so this comparison was
rerun in full. The table originally reported 0.840/0.845/0.746 accuracy and
0.178/0.380/0.624 recall for mildness 0.0/0.5/1.0 -- the pattern below is the
same shape, if anything a cleaner version of it.

| weight_mildness | test accuracy | vs. baseline (paired t-test) | recall (compromised) | F1 |
|---|---|---|---|---|
| 0.0 (unweighted) | 0.823 +/- 0.041 | p=0.029, significant | 0.178 | 0.267 |
| **0.5 (chosen)** | **0.849 +/- 0.025** | **p=0.036, significant** | **0.457** | **0.537** |
| 1.0 (fully balanced) | 0.760 +/- 0.047 | p=0.336, not significant | 0.581 | 0.498 (no split flagged unstable this run) |

**Known limitation -- do not let this get lost before the final write-up:**
even at the chosen mildness=0.5, recall on the compromised class is only
**~46%**, meaning the model still misses over half of the actual compromised
nodes on average across splits.

This is real, reproducible progress over the unweighted baseline (which
systematically, consistently misses ~82% of compromised nodes -- confirmed
via multi-seed averaging to be a genuine bias, not noise), but it is **not**
a mature or production-ready detector. The final write-up should describe
this plainly rather than characterizing Milestone 2's classifier as solved.

Full accuracy is also not the metric that matters here -- the unweighted
model's higher raw accuracy is close to a wash against mildness=0.5's, and
comes almost entirely from correctly classifying the easy majority (benign)
class while still missing most compromised nodes, which is the opposite of
what a security-relevant classifier should optimize for.
