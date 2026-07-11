# Project context for Claude Code

## What this project is

Graph Attention Networks (GAT) for securing Social IoT networks from botnet
propagation. Full 8-week roadmap has 4 milestones:

1. **Weeks 1-2** — Baseline Social IoT graph + botnet propagation simulator.
2. **Weeks 3-4** — Train a GAT on a core task (node classification benign/compromised),
   extract attention-based edge importance scores, implement static pruning
   (threshold + top-k), measure security/utility trade-off.
3. **Weeks 5-6** — RL agent (DQN or policy gradient) that learns an adaptive edge-pruning
   policy on top of GAT scores. Add XAI hooks: log which edges get pruned and why.
4. **Weeks 7-8** — Ablations (GAT depth/heads, RL reward weights, noisy features),
   stress tests (graph size/density, attack aggressiveness), final report + slide deck.

## Current status

**Nothing has been built yet in this folder — this is a from-scratch start.**
No milestone is done. There is a note below ("Supervisor feedback from a prior
attempt") describing lessons from an *earlier, separate* attempt at this same
project — that earlier code does not exist in this repo and must not be
assumed to exist. Treat every file as something to be created from zero,
starting with Milestone 1.

## Coding conventions — IMPORTANT

The student wants to understand every line of code, not just get working output.
When writing or editing any file in this repo:
- Comment every non-trivial line explaining *what* it does and, where it maps to
  a roadmap formula (e.g. attention coefficients, p_uv, reward function), *why*.
- Before writing new code, briefly explain the plan in plain language first.
- After writing code, walk through it function by function if asked — don't just
  say "done," confirm the student understands the mechanism (e.g. why softmax
  normalizes attention over a neighborhood, why GAT uses LeakyReLU, etc.).
- Prefer small, runnable increments over big untested blocks. Run the script after
  each meaningful change and show the output.

## Environment

- Python 3.10+, see `requirements.txt`.
- Milestone 1 deps: numpy, networkx, matplotlib, pandas.
- Milestone 2 needs PyTorch + PyTorch Geometric (add when starting that milestone;
  GPU is not required for graphs this size, CPU training is fine).
- Milestone 3 needs an RL library (stable-baselines3, or a small custom DQN loop
  is fine for a graph this size and easier to explain line by line).

## Workflow suggestion

Work one milestone at a time. After each milestone's acceptance criteria (listed
in the original roadmap PDF) are met, commit to git with a message like
`"Milestone 2: GAT training + static pruning done"` before moving on.

## Supervisor feedback from a prior attempt — READ BEFORE BUILDING

This project was attempted once before. The supervisor's review found the
results were a null result, for identifiable, fixable reasons. This time,
build these fixes in from the start instead of discovering them after months
of work. Do not treat any of the below as optional polish — they are
correctness requirements.

### What went wrong last time

1. **Infection model ignored features.** The botnet simulator used a single
   constant infection probability per edge, completely ignoring node/edge
   features. The roadmap always intended `p_uv = sigmoid(beta^T phi(x_u, x_v,
   e_uv))` to be feature-driven. Because it wasn't, infection spread was
   driven purely by raw connectivity, saturated the graph (97.5% infected
   with no pruning), and had *no relationship* to the GAT attention being
   used to decide what to prune — the attack mechanism and the defense
   mechanism were disconnected from each other.
2. **Unrealistic topology.** The graph was small Erdos-Renyi (uniform random).
   ER graphs have no hubs and no community structure, so there are no
   critical bottleneck edges for smart pruning to exploit. Real IoT/social
   networks are hub-heavy (scale-free / community-structured) — that's
   exactly where targeted pruning has something to win. **Use a scale-free
   (Barabasi-Albert) or community-structured (stochastic block model) graph,
   or a real IoT dataset, not plain Erdos-Renyi.**
3. **RL wasn't actually learning.** Different reward weightings (security-
   focused, balanced, utility-focused) all produced identical pruning
   decisions and identical infection outcomes. Root cause: infection was
   re-randomized every episode/step, so noise swamped the effect of any
   single edge removal. **Fix: control the randomness — e.g. fix the RNG
   seed per evaluation episode, or average over enough repeated rollouts
   per graph that the signal isn't drowned by noise, before concluding the
   policy learned anything.**
4. **Missing baselines, first pass.** No comparison against random pruning,
   degree-centrality pruning, or betweenness-centrality pruning. Without
   these, there's no way to know if GAT+RL contributes anything.
5. **Missing baseline, second pass — the important one.** Once infection
   became feature-driven (`p_uv` known per edge), the natural strong
   baseline is: **just remove the highest-`p_uv` edges directly.** It's
   cheap, feature-aware, and targets the actual infection mechanism. RL is
   effectively spending hundreds of episodes per graph trying to discover
   those same edges by trial and error — if a one-shot cut-highest-p_uv
   rule matches RL, RL isn't earning its cost. Also add a **greedy oracle**
   (removes whichever edges most reduce simulated infection) as an upper
   bound.
6. **Statistical rigor was insufficient.**
   - Single-graph results aren't evidence. Evaluate over many (~40+) random
     graph instances and report mean ± standard deviation.
   - "Method A beats random (p<0.05) and Method B doesn't" does **not**
     imply "A beats B." To claim A beats B, run a *direct paired test*
     between A and B across the same graph instances.
   - Running many comparisons against a random baseline without multiple-
     comparison correction inflates apparent significance. Apply Holm or
     Bonferroni correction; a p-value that only survives the gentler Holm
     correction is a fragile result, and the write-up should say so plainly.
   - Report absolute effect sizes (e.g. "48% infected vs 62% with no
     pruning"), not just p-values — a statistically significant but small
     absolute improvement should be described as such, not oversold.
   - Report compute cost per method alongside accuracy/containment — RL's
     training cost is part of the honest comparison against a one-shot
     heuristic.

### What this means for how we (re)build this time

- Build the **feature-driven infection model and a scale-free/community
  topology into Milestone 1**, not as an afterthought later.
- Build the **full baseline suite** (random, degree centrality, betweenness
  centrality, highest-p_uv removal, greedy oracle) alongside GAT+RL from
  Milestone 2/3 onward, not bolted on at the end.
- Design the evaluation harness (many graphs, paired statistical tests,
  multiple-comparison correction, absolute effect sizes, compute cost) as
  part of Milestone 3, before drawing any conclusions about what "wins."
- **Be honest about the likely paper framing.** The prior attempt's honest
  result was: learned methods (GAT+RL) roughly tie degree-centrality pruning
  and only modestly beat random, at much higher computational cost. That is
  itself a legitimate, publishable finding — "do learned pruning methods
  actually beat simple heuristics for botnet containment on realistic
  graphs?" is a real question with a real (if unglamorous) answer. Don't
  force the narrative toward "our method wins" if the evidence says
  otherwise; a rigorous negative/mixed result is more defensible than an
  overclaimed positive one.
