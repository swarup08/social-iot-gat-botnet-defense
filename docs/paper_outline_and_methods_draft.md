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
