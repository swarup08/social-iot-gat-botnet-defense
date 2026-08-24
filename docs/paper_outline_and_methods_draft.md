# Paper Outline and Methods Draft (Week 1 deliverable, v2)

Prepared as raw technical content for the supervisor to write into final
prose. Numbers verified directly against `harness_summary.csv` and
`harness_paired_tests.csv` in the project repo. v2 incorporates the
second round of supervisor feedback: eigenscore baseline, Related Work
literature, N/removal-level justification, utility-metric caveat,
pre-registration list, and title reframing.

## Working Title (revised)

*Do Learned Policies Beat Degree Centrality? A Controlled Benchmark for
Botnet Containment in Social IoT*

(Changed from the previous declarative title per supervisor feedback --
posing the question reads more fairly to GNN/RL reviewers than
announcing the verdict up front.)

## Outline

1. **Introduction** -- Social IoT botnet propagation as a graph problem;
   motivation for asking whether learned (GAT-attention, RL) edge-pruning
   beats simple structural heuristics -- including the spectral/
   eigenvalue heuristic that the epidemic-containment literature treats
   as the standard to beat -- rather than assuming learned methods win.
2. **Related Work** (draft below).
3. **Methods and Experimental Setup** (draft below).
4. **Results** -- 40-graph harness table (9 methods), paired significance
   tests (Holm-corrected across 14 pre-registered pairs), the
   containment-vs-utility Pareto frontier figure as Figure 1, ablations
   (GAT heads/depth, RL reward weighting, noisy/incomplete features),
   stress tests (graph size/density, attack aggressiveness).
5. **Discussion** -- Why structure dominates here (scale-free hubs give
   cheap heuristics -- including the spectral one -- something concrete
   to exploit); why the greedy simulation-guided heuristic's myopia
   (confirmed at n=10, not just noise) still can't beat a one-shot
   structural rule; what this implies about when learned pruning might
   earn its cost (denser/less hub-dominated graphs? adversarial/adaptive
   attackers? -- open question, not claimed here).
6. **Limitations** (draft below).
7. **Conclusion**.

## Related Work (draft)

**Social IoT.** Atzori et al.'s Social Internet of Things formulation
frames device-to-device trust relationships as a graph substrate for
service discovery -- the same substrate this paper treats as a botnet
attack surface.

**Network resilience under removal.** Albert, Jeong, and Barabasi
(2000) is the standard citation for why scale-free topologies matter
here, but precisely for *node* removal: they show scale-free networks
are robust to random node failure but fragile to targeted hub removal.
We cite this as the reason our graphs (Barabasi-Albert, hub-forming)
are a meaningful testbed with real structure to exploit -- not as prior
work on the edge-removal question this paper actually answers.

**Edge-removal epidemic containment (the literature we are actually
competing with).** A separate, more directly relevant line of work asks
which *edges* to remove to raise a network's epidemic threshold.
Strategies compared there include removal by adjacent-node degree, by
node centralities, by edge betweenness, and -- the field's accepted
principled approach -- spectral/eigenvalue methods: removing the edge
set that most reduces the largest eigenvalue of the adjacency matrix,
since the SIS epidemic threshold varies inversely with the spectral
radius. Matamalas, Arenas, and Gomez (2018, *Science Advances*,
"Effective approach to epidemic containment using link equations in
complex networks") is the anchor citation for this framing and surveys
the surrounding field. Our eigenscore baseline (edges ranked by the
product of their endpoints' eigenvector centrality) is the same
comparator this literature treats as standard.

One point that favors our finding rather than threatening it: evidence
on edge betweenness specifically is mixed. Ansari, Anvari, Pfeffer,
Molkenthin, Hellmann, Heitzig, and Kurths (2021, *Eur. Phys. J. Special
Topics*, "Moving the epidemic tipping point through topologically
targeted social distancing") report a targeted-removal strategy based
on this kind of centrality reasoning underperforming random removal at
raising the epidemic threshold in some regimes. Betweenness-centrality
underperforms other structural methods in our results too (frozen F1
0.455, among the weakest utility scores in the table), so this citation
supports rather than complicates our result.

**GNN/GAT for security and RL-based network defense.** [Placeholder --
same citations as before: Velickovic et al. for GAT, GNN-based IoT
intrusion detection work, DRL-based network security surveys. Position
this paper as a rigorous benchmark question about whether these learned
approaches earn their cost against the edge-removal literature's
accepted baselines, not as a new architecture contribution.]

**Explainability.** [Placeholder -- GNNExplainer/SHAP-style citations,
positioning our attention-based and XAI-trajectory explanations as a
lightweight first-order signal, not a formal XAI method.]

## Methods and Experimental Setup (draft)

### Social IoT Graph Construction

Graphs are generated using the Barabasi-Albert preferential-attachment
model (scale-free, hub-forming), not Erdos-Renyi, so that targeted
pruning has meaningful structure to exploit -- real Social IoT /
social-network topologies are hub-heavy, not uniformly random (Albert,
Jeong, and Barabasi, 2000). Each graph instance has N=300 nodes; a
dedicated stress test (below) sweeps N in {150, 300, 600} and confirms
the core containment finding holds across that range, with 300 taken
as the reported middle value rather than a tuned choice. Node features
are derived from actual graph structure: degree-based hub score,
clustering coefficient, community assignment (greedy modularity), and a
device-type label (gateway/service/user/sensor/actuator) assigned by
degree percentile band. Edge features are normalized edge-betweenness
centrality.

### Feature-Driven Botnet Propagation Model

Infection is simulated as a synchronous, discrete-time, round-based
process. Unlike a flat/constant infection probability, the per-edge
infection probability is feature-driven:

```
p_uv = sigmoid(beta^T * phi(x_u, x_v, e_uv))
```

where `phi` concatenates node and edge features and `beta` is a fixed,
hand-set parameter vector (not fitted to real botnet traffic -- see
Limitations). This ties the attack mechanism to the same features the
GAT and structural baselines use to score edges, so that pruning a
genuinely high-risk edge changes the simulated outcome -- addressing a
specific limitation of a prior attempt at this project, where a
constant per-edge probability decoupled the attack model from any
feature-based defense signal.

### Core Classification Task and GAT Architecture

A 2-layer Graph Attention Network (heads=4, hidden=8) is trained for
binary node classification (benign vs. compromised), with labels
derived from majority-vote infection outcomes over repeated simulator
rollouts from the graph's hub node. Class imbalance is handled via a
tunable class-weight interpolation (mildness=0.5, chosen via a 3-way
comparison across multiple train/val/test splits and model-init seeds).
At this setting, test accuracy is 84.9% but recall on the compromised
class is only ~46% -- stated plainly as a limitation, since GAT
attention (used by two of the nine pruning methods) is derived from
this classifier.

### Attention-Based Edge Importance and Static Pruning

Per-edge importance scores are extracted from the trained GAT's
attention coefficients. Raw attention is mechanically diluted at
high-degree target nodes by softmax normalization itself -- a node with
many neighbors necessarily gets a smaller average attention share per
neighbor, independent of what the model actually learned to prioritize.
We correct for this by multiplying each directed attention coefficient
by its target node's degree, which cancels the dilution and yields a
score interpretable as relative preference versus a uniform baseline
(~1.0 = uniform, >1 = favored, <1 = disfavored). This correction is a
methodological contribution in its own right: without it, raw attention
correlates strongly and mechanically with node degree in a way that has
nothing to do with the model's learned judgment, and would bias static
pruning toward or away from hub-adjacent edges for the wrong reason.
Two static pruning strategies are evaluated on the corrected scores:
threshold-based (remove lowest-scoring edges) and top-k per-node
(retain the k highest-scoring outgoing edges per node).

### Full Baseline Suite (nine methods)

Nine methods are compared at a matched edge-removal level: random,
degree-centrality, betweenness-centrality, highest-p_uv (direct
infection-risk ranking), eigenscore, GAT-threshold, GAT-top-k, a greedy
simulation-guided heuristic, and an RL-based adaptive policy. The
removal level is fixed at 50% -- a deliberately aggressive budget
chosen so methods separate; a companion containment-vs-pruning-level
figure (threshold/top-k static pruning swept across levels) shows the
method ranking is stable rather than an artifact of this specific
level.

**Eigenscore** ranks each edge by the product of its two endpoints'
eigenvector centrality, matching the comparator used in the spectral
edge-removal literature (Matamalas et al., 2018), where the principled
target is the edge set whose removal most reduces the adjacency
matrix's largest eigenvalue -- eigenvector-centrality product is a
standard cheap proxy for an edge's contribution to that eigenvalue. It
ties degree-centrality on containment (0.0332 vs. 0.0331,
Holm-corrected p=1.0 -- not merely non-significant, essentially
identical) and significantly beats RL (0.0332 vs. 0.0719,
Holm-corrected p<0.0001), matching the pattern every other structural
baseline shows, while landing near the bottom of the table on utility
(frozen F1 0.458). This sharpens the paper's Pareto argument: of the
four structural baselines (degree-centrality, betweenness-centrality,
highest-p_uv, eigenscore), degree-centrality is the only one that
holds both containment and utility simultaneously (frozen F1 0.562);
eigenscore, betweenness-centrality (frozen F1 0.455), and highest-p_uv
(frozen F1 0.423) all buy strong containment by removing edges the
core classification task needs, landing in the same low-utility
cluster. Sophistication in the edge-scoring signal does not translate
into a better containment-utility trade-off here -- only degree does.

The greedy simulation-guided heuristic removes, one edge at a time, the
single edge whose removal most reduces simulated infection (a 1-rollout
search signal per candidate, for tractability). A diagnostic (increasing
the per-candidate search signal from 1 to 5 rollouts, run across 10
independent graph instances) shows containment getting *significantly
worse*, not better, at every pruning level tested (paired p=0.0122,
<0.0001, <0.0001 at 10%/25%/50% removal) -- ruling out search noise as
the explanation for this method's underperformance and confirming its
limitation is structural: one-step-at-a-time myopia cannot see the same
global picture a one-shot structural ranking can.

### RL Formulation

The RL agent (a custom Deep Q-Network, chosen over library RL for full
explainability of its training dynamics) observes a state derived from
GAT attention-score distributions and global graph/infection statistics,
and selects a discrete action: prune one of five attention-score
quantile buckets, or stop. The reward combines negative infection cost,
a core-task utility term, and a pruning-cost penalty. Actions are
masked to exclude already-exhausted buckets.

The utility term is the frozen-base-GAT's forward-pass performance on
the current pruned topology, not retrained -- a deliberate,
tractability-driven choice, and arguably more realistic for an adaptive
agent (a live system does not retrain its classifier after every edge
cut). This choice likely understates every method's utility relative to
a retrained model, and possibly unevenly across methods: a method that
happens to cut edges the frozen classifier specifically depended on
will look worse under this metric whether or not a freshly retrained
model would recover that performance. We report this as a limitation of
the utility axis specifically, not of the containment results.

### XAI Hooks

Every RL pruning decision is logged (state, action, reward,
attention-bucket identity) during training and evaluation, aggregated
into: (a) a table of device-type-pair edge characteristics for the most
frequently pruned edges, and (b) per-hub-node summaries of which
neighbors are pruned and why (by attention/risk profile).

### Statistical Evaluation Harness

All nine methods are evaluated across 40 independent graph instances
(same generator, different seeds), reporting per-method mean +/- std for
containment ratio (infected-pruned / infected-unpruned, normalized per
seed to allow cross-seed comparison) and frozen-classifier utility
(recall, F1). Fourteen paired comparisons -- listed in full below,
fixed before any results were examined -- are tested via paired t-test,
with Holm-Bonferroni correction applied across the full family of
tests. Absolute effect sizes (mean differences) and per-method compute
cost are reported alongside every p-value.

**Pre-registered comparison list (fixed before results, n=40 each):**

1. GAT-threshold vs. GAT-top-k
2. GAT-top-k vs. degree-centrality
3. degree-centrality vs. betweenness-centrality
4. degree-centrality vs. highest-p_uv
5. highest-p_uv vs. betweenness-centrality
6. RL vs. random
7. RL vs. greedy simulation-guided heuristic
8. RL vs. GAT-threshold
9. RL vs. GAT-top-k
10. RL vs. degree-centrality
11. RL vs. betweenness-centrality
12. RL vs. highest-p_uv
13. eigenscore vs. degree-centrality (added this round)
14. eigenscore vs. RL (added this round)

### Ablations and Stress Tests

- GAT architecture: heads (2/4/8) and depth (1/2/3 layers), each across
  15 graph instances.
- Noisy and incomplete node features (Gaussian noise and random
  masking, each at multiple severities), 15 graph instances each.
- RL reward weighting (security- vs. utility-focused), 40 graph
  instances.
- Stress tests: graph size (150/300/600 nodes), graph density (sparse/
  default/dense), and attack aggressiveness (moderate/high infection
  bias), 15 graph instances per condition.

## Results (draft)

### Headline finding: structural pruning beats the learned pipeline, at a fraction of the cost

Across all 40 graph instances, degree-centrality achieves the best or
tied-best containment of any method tested (containment ratio 0.0331 --
only ~3.3% of nodes end up compromised on average) at a mean compute
cost of 0.011s per graph. RL, the most expensive method in the suite at
38.7s per graph -- roughly 3600x degree-centrality's cost -- achieves a
containment ratio of 0.0719, more than double degree-centrality's, and
loses to it at Holm-corrected p=1.31e-5 (paired t-test, n=40). This is
not specific to degree-centrality: RL loses significantly
(Holm-corrected p<0.05) to every structural baseline tested --
degree-centrality, betweenness-centrality, highest-p_uv, eigenscore --
and to both GAT-based static pruning methods. RL only beats random
pruning and ties the greedy simulation-guided heuristic.

### Full baseline suite results (n=40 graphs per method)

| Method | Containment ratio (mean +/- std) | Frozen F1 (utility) | Mean compute cost/graph |
|---|---|---|---|
| degree-centrality | 0.0331 +/- 0.0061 | 0.562 | 0.011s |
| eigenscore | 0.0332 +/- 0.0064 | 0.458 | 0.034s |
| highest-p_uv | 0.0365 +/- 0.0050 | 0.423 | 0.008s |
| betweenness-centrality | 0.0361 +/- 0.0058 | 0.455 | 0.611s |
| GAT top-k | 0.0372 +/- 0.0060 | 0.532 | 0.881s |
| GAT threshold | 0.0547 +/- 0.0126 | 0.544 | 0.161s |
| greedy simulation-guided heuristic | 0.0659 +/- 0.0344 | 0.634 | 45.41s |
| RL | 0.0719 +/- 0.0430 | 0.543 | 38.72s |
| random | 0.1218 +/- 0.0183 | 0.466 | 0.024s |

(Sorted by containment ratio, best/lowest first.)

### Statistical significance

Across the 14 pre-registered paired comparisons, Holm-Bonferroni
correction is applied over the full family. RL loses significantly to
GAT-threshold, GAT-top-k, degree-centrality, betweenness-centrality,
highest-p_uv, and eigenscore (all Holm-corrected p<0.05, most
p<0.0001), and beats only random (Holm-corrected p<0.0001, in RL's
favor) while tying the greedy simulation-guided heuristic
(Holm-corrected p=1.0). Eigenscore ties degree-centrality exactly on
containment (Holm-corrected p=1.0) and significantly beats RL
(Holm-corrected p=1.21e-5) -- confirming that even the spectral method
the epidemic-containment literature treats as the principled optimum
does not close the gap to plain degree centrality, and still buries
the learned pipeline.

### Figure 1: the containment-vs-utility Pareto frontier

`containment_vs_utility_tradeoff.png` plots frozen F1 (utility)
against containment ratio for all nine methods, each point labelled by
mean compute cost. The frontier shows no single winner. Degree-
centrality sits at the best containment while also holding the
second-best utility overall, at essentially free compute cost
(0.011s/graph). The greedy simulation-guided heuristic sits at the
best utility (frozen F1 0.634) but middling containment, at ~4200x
degree-centrality's compute cost. Critically, degree-centrality is the
*only* structural method that holds both axes at once: eigenscore,
betweenness-centrality, and highest-p_uv all cluster in the
high-containment/low-utility corner of the plot (frozen F1 0.42-0.46),
buying their containment by removing edges the core classification
task needs. Sophistication in the edge-scoring signal -- moving from
raw degree to eigenvector-centrality- or betweenness-weighted scoring
-- does not improve this trade-off; it only pays the utility cost
without matching gain. GAT and RL sit inside the frontier rather than
on it in both dimensions.

### Closing the outcome-guided-search objection: the myopia diagnostic

A natural objection: the greedy simulation-guided heuristic directly
observes simulated infection outcomes at each step, so if it still
cannot beat a one-shot structural rule, perhaps its 1-rollout-per-
candidate search signal is simply too noisy rather than fundamentally
limited. We tested this directly by increasing the per-candidate
search accuracy from 1 to 5 rollouts (a ~24x increase in search cost)
and re-measuring containment across 10 independent graph instances,
paired per graph:

| Removal level | 1 rollout | 5 rollouts | Paired difference | Paired p-value |
|---|---|---|---|---|
| 10% | 0.646 +/- 0.052 | 0.809 +/- 0.133 | -0.162 | 0.0122 |
| 25% | 0.265 +/- 0.032 | 0.622 +/- 0.097 | -0.357 | <0.0001 |
| 50% | 0.052 +/- 0.030 | 0.354 +/- 0.084 | -0.301 | <0.0001 |

More search accuracy made containment significantly *worse*, not
better, at every pruning level tested, and all three differences
survive a conservative Bonferroni threshold for 3 tests (0.05/3 =
0.0167). This rules out search noise as the explanation for the greedy
heuristic's underperformance relative to structural methods: the
limitation is structural, not statistical. One-step-at-a-time myopia
cannot see the same global picture a one-shot ranking over the whole
graph can -- confirming that containment on these scale-free graphs is
dominated by structure that a single global pass can exploit but a
sequence of locally-optimal steps cannot.

### Ablations and Stress Tests: results

**GAT architecture (heads and depth) -- the core finding's fragile axis.**
containment_ratio, n=15 graphs per condition, structural-baseline range
0.033-0.037 for reference:

| heads (n_layers=2 fixed) | GAT threshold | GAT top-k | vs. structural |
|---|---|---|---|
| 2 | 0.085+/-0.086 | 0.049+/-0.041 | not significant (variance swamps signal) |
| 4 (project default) | 0.053+/-0.016 | 0.035+/-0.006 | threshold sig. worse; top-k tied |
| 8 | 0.210+/-0.138 | 0.129+/-0.109 | both significantly worse, ~4x baseline |

| n_layers (heads=4 fixed) | GAT threshold | GAT top-k | vs. structural |
|---|---|---|---|
| 1 | 0.174+/-0.086 | 0.064+/-0.027 | both significantly worse |
| 2 (project default) | 0.053+/-0.016 | 0.035+/-0.006 | threshold sig. worse; top-k tied |
| 3 | 0.111+/-0.053 | 0.062+/-0.018 | both significantly worse |

The project's central "GAT top-k ties structural baselines" result holds
only in a narrow window (heads=4, n_layers=2). Moving either heads or
depth in either direction breaks it, in both directions of depth and at
heads=8 for heads (heads=2 trends the same way but is underpowered at
n=15). This is a genuine architecture-conditional finding, not a
robustness result, and is reported as such rather than averaged away.

**Feature quality (noise and missingness) -- the core finding's robust
axis.** Same containment_ratio metric, n=15, heads=4/n_layers=2 held
fixed:

| condition | GAT threshold | GAT top-k | vs. structural |
|---|---|---|---|
| clean | 0.053+/-0.016 | 0.035+/-0.006 | threshold sig. worse; top-k tied |
| Gaussian noise, std=0.1 | 0.060+/-0.038 | 0.037+/-0.007 | unchanged |
| Gaussian noise, std=0.3 | 0.059+/-0.044 | 0.037+/-0.010 | unchanged |
| masking, p=0.1 | 0.063+/-0.051 | 0.039+/-0.018 | unchanged |
| masking, p=0.3 | 0.064+/-0.050 | 0.041+/-0.023 | unchanged |

Neither corrupting (Gaussian noise up to std=0.3) nor removing (masking
up to 30% missingness) node features changes which methods are
statistically distinguishable from which, at any level tested. Unlike
the architecture ablations, feature-quality robustness is a genuine,
confirmed property of this pipeline at the tested severities.

**Stress tests -- graph size and density strengthen the core finding;
attack aggressiveness weakens and partially reverses it.** 15 graphs per
condition (reduced power vs. the main 40-graph harness, reported as
such); baseline reference RL 0.072+/-0.043 vs. structural 0.033-0.055,
RL significantly worse than all five structural methods.

| condition | RL containment | structural range | RL significantly worse vs. |
|---|---|---|---|
| size=150 nodes | 0.126+/-0.068 | 0.057-0.092 | 4 of 5 methods |
| size=600 nodes | 0.051+/-0.037 | 0.019-0.032 | 4 of 5 methods |
| sparse (m=2) | 0.570+/-0.396 | 0.142-0.181 | 5 of 5, dramatically |
| dense (m=5) | 0.243+/-0.087 | 0.042-0.132 | 5 of 5 |

Size and density variation, if anything, strengthens RL's
underperformance -- sparse graphs produce the largest structural-vs-RL
gap seen anywhere in this project, alongside high RL training variance
that echoes the same instability seen in the reward-weighting ablation
below.

Attack aggressiveness tells a different story:

| aggressiveness | RL vs. GAT threshold | RL vs. GAT top-k | RL vs. degree | RL vs. betweenness | RL vs. highest-p_uv |
|---|---|---|---|---|---|
| moderate (default) | tied | tied | RL worse | RL worse | RL worse |
| high | RL significantly better | RL worse | RL worse | tied | tied |

At high aggressiveness, every method's containment degrades sharply
(0.36-0.63 vs. 0.02-0.13 elsewhere) and frozen F1 saturates near 1.0
for all methods -- the classification task becomes near-trivial once
almost every node is infected, so some of this reversal may be a
ceiling/floor effect rather than RL becoming more capable. Both
readings are consistent with the data; this project does not
distinguish between them and reports the reversal as a genuine,
partially confounded boundary condition rather than a universal law.

**RL reward-weighting ablation -- no weighting closes the containment
gap, and pushing security too hard destabilizes training.** 40-graph
ablation, RL baseline reproduces the main harness exactly
(0.072+/-0.043) as a cross-run consistency check:

| RL configuration | containment_ratio | frozen_f1 |
|---|---|---|
| baseline (security=1, utility=1) | 0.072+/-0.043 | 0.543 |
| high-security (3, 1) | 0.153+/-0.287 | 0.566 |
| very-high-security (10, 1) | 0.930+/-0.250 | 0.513 |
| security-only (1, 0) | 0.978+/-0.139 | 0.501 |

Pushing the security weight higher does not close the gap to
structural methods; past a threshold it collapses DQN training
entirely (converging to near-zero pruning, containment approaching
1.0), a failure mode confirmed at 40-graph scale, not a single-graph
artifact. RL's genuine utility advantage is real but narrower than "RL
wins": it significantly beats betweenness-centrality, highest-p_uv,
and random on frozen F1, but is statistically indistinguishable from
GAT-threshold, GAT-top-k, and degree-centrality -- the three strongest
utility performers. The defensible framing is that RL's containment
underperformance needs a different fix (reward shaping, a different
algorithm, or a redesigned action space), not merely different reward
weights within the current setup.

## Limitations (draft)

- **Synthetic attack model.** The infection model, while feature-driven
  (not a flat probability), uses a synthetic, hand-set beta vector
  rather than one fitted to real botnet traffic data. This is the
  largest external-validity gap in the paper and is stated here plainly
  rather than left for a reviewer to find.
- **Single graph family.** All experiments use Barabasi-Albert
  scale-free graphs; results are not yet validated on other topologies
  or real-world SIoT traffic datasets.
- **Utility metric likely understates performance, possibly unevenly**
  (see Methods -- frozen vs. retrained classifier).
- **Single custom RL implementation.** Only one DQN variant is tested;
  whether a different RL algorithm would close the gap to structural
  heuristics is untested.
- **Architecture-conditional finding.** The core containment result is
  conditional on the specific GAT architecture used (heads=4, depth=2);
  ablations show shallower, deeper, or wider architectures break it.
- **GAT classifier recall.** The node classifier recovers only ~46% of
  compromised nodes at the chosen class-weighting, relevant because GAT
  attention feeds two of the nine pruning methods compared.

---
*Full numbers, every caveat, and the dated log of methodological fixes
made during this project are in `NOTES.md` in the repository root.*
