"""Milestone 3: RL environment for adaptive edge pruning (mechanics only in
this increment -- state/action/step wiring; the learning algorithm is a
later increment).

DESIGN SUMMARY (see NOTES.md for the two flagged divergences from
Milestone 2 -- frozen-model utility instead of retraining, and single fixed
graph instance -- and the empirical compute-cost check that justified using
FULL Milestone-2-grade rigor throughout, not just at final evaluation):

State (fixed-size vector, not the raw graph -- a small MLP policy is enough):
  - cumulative fraction of edges removed so far
  - step / max_steps
  - mean and std of degree-corrected s_uv over edges STILL PRESENT
  - remaining-edge fraction per attention bucket (5 values)
  - current containment ratio (security) and utility ratio (utility)
  - one-hot of the last action taken, and the last reward received

Action: discrete choice of "remove a fixed-size chunk from attention bucket
X" (X = 0..4, quintiles of degree-corrected s_uv, lowest to highest) or
STOP (end the episode now). 6 actions total.

Reward, combining the roadmap's three terms on a consistent "relative to
unpruned baseline" scale:
  R_t = -w_security * containment_ratio_t
        + w_utility  * utility_ratio_t
        - w_cost     * cumulative_removed_fraction_t
  containment_ratio_t reuses Milestone 2's measure_security + containment_ratio
  (5 seed nodes x 15 rollouts each -- the SAME noise-controlled harness, not
  a single re-randomized seed, which is exactly what sank the prior attempt's
  RL run per this project's own history). utility_ratio_t uses the frozen
  base GAT's forward pass (see measure_utility_frozen in
  src/milestone2_pruning.py), normalized against its own unpruned baseline.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import torch

from src.milestone1 import build_edge_features, compute_edge_infection_probabilities
from src.milestone2 import (
    DEFAULT_BETA,
    GATNodeClassifier,
    build_pyg_data,
    extract_degree_corrected_attention_scores,
    generate_labeled_graph,
    make_node_split,
    train_gat,
)
from src.milestone2_pruning import containment_ratio, measure_security, measure_utility_frozen, pick_seed_nodes

N_BUCKETS = 5
STOP_ACTION = N_BUCKETS  # action index N_BUCKETS means "stop pruning, end episode"
ACTION_SIZE = N_BUCKETS + 1
# 1 (cum_removed) + 1 (step_frac) + 2 (mean/std s_uv) + N_BUCKETS (bucket remaining fractions)
# + 1 (containment_ratio) + 1 (utility_ratio) + ACTION_SIZE (last action one-hot) + 1 (last_reward)
STATE_SIZE = 1 + 1 + 2 + N_BUCKETS + 1 + 1 + ACTION_SIZE + 1


def assign_attention_buckets(gat_scores: Dict[Tuple[int, int], float], n_buckets: int = N_BUCKETS) -> Dict[Tuple[int, int], int]:
    """Assign each edge a FIXED bucket index (0 = lowest attention, n_buckets-1
    = highest), based on quantiles of the ORIGINAL graph's degree-corrected
    attention score distribution.

    Bucket assignment is computed ONCE, from the full unpruned graph, and
    never changes as edges get removed during an episode -- otherwise
    "bucket X" would mean a different, shifting set of edges over time,
    which would make the action's meaning (and the state's bucket-remaining-
    fraction features) inconsistent step to step.
    """
    edges = list(gat_scores.keys())
    scores = np.array([gat_scores[e] for e in edges])
    quantile_boundaries = np.quantile(scores, np.linspace(0, 1, n_buckets + 1))

    bucket_of: Dict[Tuple[int, int], int] = {}
    for edge, score in zip(edges, scores):
        # interior boundaries only (drop the 0th/last, which are just min/max)
        bucket = int(np.searchsorted(quantile_boundaries[1:-1], score, side="right"))
        bucket_of[edge] = min(bucket, n_buckets - 1)
    return bucket_of


def _get(d: Dict[Tuple[int, int], float], u: int, v: int):
    return d.get((u, v), d.get((v, u)))


class PruningEnv:
    """One episode = one adaptive pruning trajectory over a FIXED graph.

    All the "expensive" inputs (graph, features, labels, p_uv, seed nodes,
    degree-corrected GAT scores, the frozen base model, and the unpruned
    baselines) are computed ONCE outside this class and passed in -- the
    environment itself only does cheap per-step bookkeeping plus the
    per-step security/utility measurement.
    """

    def __init__(
        self,
        graph: nx.Graph,
        node_features: Dict[int, Dict[str, float]],
        labels: Dict[int, int],
        p_uv: Dict[Tuple[int, int], float],
        seed_nodes: Dict[str, int],
        gat_scores: Dict[Tuple[int, int], float],
        base_model: GATNodeClassifier,
        test_mask: torch.Tensor,
        baseline_infected: Dict[str, float],
        baseline_utility: Dict[str, float],
        max_steps: int = 15,
        max_removal_fraction: float = 0.5,
        chunk_fraction: float = 0.05,
        w_security: float = 1.0,
        w_utility: float = 1.0,
        w_cost: float = 0.1,
        n_rollouts: int = 15,
        infection_seed_base: int = 7,
    ):
        self.original_graph = graph
        self.node_features = node_features
        self.labels = labels
        self.p_uv = p_uv
        self.seed_nodes = seed_nodes
        self.gat_scores = gat_scores
        self.base_model = base_model
        self.test_mask = test_mask
        self.baseline_infected = baseline_infected
        self.baseline_utility = baseline_utility

        self.max_steps = max_steps
        self.max_removal_fraction = max_removal_fraction
        self.chunk_size = max(1, round(chunk_fraction * graph.number_of_edges()))
        self.w_security = w_security
        self.w_utility = w_utility
        self.w_cost = w_cost
        self.n_rollouts = n_rollouts
        # Supervisor-flagged fix: this seed base is used ONLY for the reward
        # signal the RL agent trains on (every env.step() during training
        # calls _measure_current, below). The harness's FINAL reported
        # containment ratio for every method -- including RL -- must be
        # measured with a DIFFERENT, disjoint seed base (see
        # demo_milestone4_harness.py's EVAL_INFECTION_SEED_BASE), so the
        # policy is graded on infection realizations it never saw or was
        # rewarded against during training. Keeping this as a named,
        # overridable field (rather than always relying on measure_security's
        # own default) makes that separation explicit and auditable.
        self.infection_seed_base = infection_seed_base

        self.bucket_of = assign_attention_buckets(gat_scores, N_BUCKETS)
        self.bucket_original_counts = [0] * N_BUCKETS
        for bucket in self.bucket_of.values():
            self.bucket_original_counts[bucket] += 1

        self.action_size = ACTION_SIZE
        self.state_size = STATE_SIZE

        self.working_graph: nx.Graph = None
        self.step_count = 0
        self.cumulative_removed_fraction = 0.0
        self.last_action_one_hot = np.zeros(self.action_size, dtype=np.float32)
        self.last_reward = 0.0
        self.trajectory_log: List[dict] = []
        self._current_containment_ratio = 0.0
        self._current_utility_ratio = 1.0

        self.reset()

    def reset(self) -> np.ndarray:
        """Start a fresh episode from the full, unpruned graph."""
        self.working_graph = self.original_graph.copy()
        self.step_count = 0
        self.cumulative_removed_fraction = 0.0
        self.last_action_one_hot = np.zeros(self.action_size, dtype=np.float32)
        self.last_reward = 0.0
        self.trajectory_log = []
        self._current_containment_ratio, self._current_utility_ratio = self._measure_current()
        return self._compute_state()

    def _measure_current(self) -> Tuple[float, float]:
        """Security (mean containment ratio across seed_nodes) and utility
        (frozen-model recall ratio vs. its own unpruned baseline) on the
        CURRENT working_graph."""
        ratios = []
        for name, node in self.seed_nodes.items():
            infected = measure_security(
                self.working_graph, self.p_uv, node,
                n_rollouts=self.n_rollouts,
                infection_seed_base=self.infection_seed_base,  # training-time reward seeds, kept separate from final-report seeds
            )
            ratios.append(containment_ratio(infected, self.baseline_infected[name]))
        mean_containment = statistics.mean(ratios)

        utility = measure_utility_frozen(self.base_model, self.working_graph, self.node_features, self.labels, self.test_mask)
        utility_ratio = utility["recall"] / max(self.baseline_utility["recall"], 1e-6)
        return mean_containment, utility_ratio

    def _bucket_remaining_counts(self) -> List[int]:
        counts = [0] * N_BUCKETS
        for u, v in self.working_graph.edges():
            counts[_get(self.bucket_of, u, v)] += 1
        return counts

    def action_mask(self) -> np.ndarray:
        """Boolean mask over actions: True if the action is currently valid.

        A "prune bucket X" action is valid only if bucket X still has at
        least one remaining edge -- otherwise it would be a no-op (and,
        without this mask, a policy can get stuck repeatedly "choosing" an
        already-exhausted bucket instead of pruning elsewhere or stopping).
        STOP is always valid.
        """
        bucket_remaining_counts = self._bucket_remaining_counts()
        return np.array([count > 0 for count in bucket_remaining_counts] + [True], dtype=bool)

    def _compute_state(self) -> np.ndarray:
        bucket_remaining_counts = self._bucket_remaining_counts()
        remaining_scores = [_get(self.gat_scores, u, v) for u, v in self.working_graph.edges()]

        mean_score = float(np.mean(remaining_scores)) if remaining_scores else 0.0
        std_score = float(np.std(remaining_scores)) if remaining_scores else 0.0
        bucket_remaining_fractions = [
            bucket_remaining_counts[b] / max(self.bucket_original_counts[b], 1) for b in range(N_BUCKETS)
        ]

        state = np.concatenate(
            [
                [self.cumulative_removed_fraction],
                [self.step_count / self.max_steps],
                [mean_score, std_score],
                bucket_remaining_fractions,
                [self._current_containment_ratio],
                [self._current_utility_ratio],
                self.last_action_one_hot,
                [self.last_reward],
            ]
        ).astype(np.float32)
        return state

    def _compute_reward(self) -> float:
        return (
            -self.w_security * self._current_containment_ratio
            + self.w_utility * self._current_utility_ratio
            - self.w_cost * self.cumulative_removed_fraction
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Apply one action, re-measure security/utility, and return
        (next_state, reward, done, info) -- the standard Gym-style interface.
        """
        assert 0 <= action < self.action_size, f"action {action} out of range [0, {self.action_size})"
        self.step_count += 1
        edges_removed: List[Tuple[int, int]] = []

        if action != STOP_ACTION:
            bucket = action
            candidates = [(u, v) for u, v in self.working_graph.edges() if _get(self.bucket_of, u, v) == bucket]
            # Remove the LOWEST-scored remaining edges within the chosen
            # bucket first (a modest, defensible tie-break within a bucket;
            # the agent's real lever is WHICH bucket, not within-bucket order).
            candidates.sort(key=lambda e: _get(self.gat_scores, *e))
            edges_removed = candidates[: self.chunk_size]
            self.working_graph.remove_edges_from(edges_removed)
            self.cumulative_removed_fraction = 1 - self.working_graph.number_of_edges() / self.original_graph.number_of_edges()

        self._current_containment_ratio, self._current_utility_ratio = self._measure_current()
        reward = self._compute_reward()

        self.last_action_one_hot = np.zeros(self.action_size, dtype=np.float32)
        self.last_action_one_hot[action] = 1.0
        self.last_reward = reward

        done = (
            action == STOP_ACTION
            or self.step_count >= self.max_steps
            or self.cumulative_removed_fraction >= self.max_removal_fraction
            or self.working_graph.number_of_edges() == 0
        )

        # XAI hook: log enough detail per step for later "which edges get
        # pruned and why" summaries -- built in from this first increment,
        # not bolted on after training already works.
        self.trajectory_log.append(
            {
                "step": self.step_count,
                "action": int(action),
                "bucket": None if action == STOP_ACTION else action,
                "edges_removed": edges_removed,
                "edges_removed_details": [
                    {
                        "edge": (u, v),
                        "s_uv": _get(self.gat_scores, u, v),
                        "p_uv": _get(self.p_uv, u, v),
                        "device_type_u": self.node_features[u]["device_type"],
                        "device_type_v": self.node_features[v]["device_type"],
                        "hub_score_u": self.node_features[u]["hub_score"],
                        "hub_score_v": self.node_features[v]["hub_score"],
                    }
                    for u, v in edges_removed
                ],
                "reward": reward,
                "containment_ratio": self._current_containment_ratio,
                "utility_ratio": self._current_utility_ratio,
                "cumulative_removed_fraction": self.cumulative_removed_fraction,
            }
        )

        info = {"cumulative_removed_fraction": self.cumulative_removed_fraction}
        return self._compute_state(), reward, done, info


def build_pruning_env(n_nodes: int = 300, model_seed: int = 0, **env_kwargs) -> PruningEnv:
    """One-time setup: build the graph, labels, p_uv, seed nodes, the frozen
    base GAT (and its degree-corrected attention scores), and the unpruned
    security/utility baselines -- everything the environment needs but that
    must only be computed ONCE, not per episode or per step. Mirrors
    demo_milestone2_pruning.py's setup exactly, so this environment's
    "unpruned" reference point matches Milestone 2's.
    """
    graph, node_features, labels = generate_labeled_graph(n_nodes=n_nodes)
    data = build_pyg_data(graph, node_features, labels)
    train_mask, val_mask, test_mask = make_node_split(data.num_nodes, seed=0)

    degrees = dict(graph.degree())
    hub_node = max(degrees, key=degrees.get)
    sorted_by_degree = sorted(degrees.items(), key=lambda kv: kv[1])
    median_degree = sorted_by_degree[len(sorted_by_degree) // 2][1]
    mid_node = min((n for n in degrees if n != hub_node), key=lambda n: abs(degrees[n] - median_degree))
    seed_nodes = pick_seed_nodes(graph, hub_node, mid_node)

    edge_features = build_edge_features(graph)
    p_uv = compute_edge_infection_probabilities(graph, node_features, edge_features, beta=DEFAULT_BETA)

    base_model = train_gat(data, train_mask, model_seed=model_seed)  # frozen from here on
    gat_scores = extract_degree_corrected_attention_scores(base_model, data, graph)

    baseline_infected = {name: measure_security(graph, p_uv, node) for name, node in seed_nodes.items()}
    baseline_utility = measure_utility_frozen(base_model, graph, node_features, labels, test_mask)

    return PruningEnv(
        graph=graph,
        node_features=node_features,
        labels=labels,
        p_uv=p_uv,
        seed_nodes=seed_nodes,
        gat_scores=gat_scores,
        base_model=base_model,
        test_mask=test_mask,
        baseline_infected=baseline_infected,
        baseline_utility=baseline_utility,
        **env_kwargs,
    )
