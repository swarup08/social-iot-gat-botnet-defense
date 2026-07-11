"""Milestone 3: XAI summary functions for RL pruning decisions.

Roadmap: "Log attention distributions and node features for edges that are
pruned or retained by the RL policy... Generate simple explanation
artifacts, such as: Lists of edges most frequently pruned and their typical
characteristics; Per-node summaries explaining why certain neighbors are
removed." Also: "RL policy explanations: provide summaries of policy
behavior, such as which attention ranges or node types are most likely to
be pruned."

The LOGGING half of this was already built into PruningEnv.step() from the
first environment increment (trajectory_log, with per-edge s_uv/p_uv/
device_type/hub_score recorded at the moment of removal) -- these are the
SUMMARY/aggregation functions that consume that log and turn it into the
"simple explanation artifacts" the roadmap asks for. Counterfactual analysis
("what would have happened if an edge had not been pruned") is a distinct,
more expensive piece (needs a fresh simulate_botnet measurement per
candidate edge) and is intentionally NOT included in this first increment.
"""

from __future__ import annotations

import statistics
from collections import Counter
from typing import Dict, List, Tuple

import networkx as nx


def _all_removed_edges(trajectory_log: List[dict]) -> List[dict]:
    """Flatten every edge-removal detail dict across all steps of one trajectory."""
    return [detail for step in trajectory_log for detail in step["edges_removed_details"]]


def summarize_pruning_characteristics(trajectory_log: List[dict], original_graph: nx.Graph, gat_scores: Dict, p_uv: Dict) -> Dict:
    """Aggregate characteristics of every edge the policy removed, and compare
    against the ORIGINAL graph's population statistics -- so a claim like
    "pruned edges have lower attention on average" is checked against a
    baseline, not asserted from the pruned set alone.
    """
    removed = _all_removed_edges(trajectory_log)
    if not removed:
        return {"n_removed": 0}

    removed_s_uv = [d["s_uv"] for d in removed]
    removed_p_uv = [d["p_uv"] for d in removed]
    removed_hub = [(d["hub_score_u"] + d["hub_score_v"]) / 2 for d in removed]

    population_s_uv = list(gat_scores.values())
    population_p_uv = list(p_uv.values())

    device_type_counts = Counter()
    for d in removed:
        device_type_counts[d["device_type_u"]] += 1
        device_type_counts[d["device_type_v"]] += 1

    return {
        "n_removed": len(removed),
        "n_steps_with_removals": sum(1 for step in trajectory_log if step["edges_removed"]),
        "removed_s_uv_mean": statistics.mean(removed_s_uv),
        "population_s_uv_mean": statistics.mean(population_s_uv),
        "removed_p_uv_mean": statistics.mean(removed_p_uv),
        "population_p_uv_mean": statistics.mean(population_p_uv),
        "removed_hub_score_mean": statistics.mean(removed_hub),
        "device_type_endpoint_counts": dict(device_type_counts.most_common()),
    }


def most_pruned_device_type_pairs(trajectory_log: List[dict], top_n: int = 5) -> List[Tuple[Tuple[str, str], int]]:
    """Rank (device_type_u, device_type_v) pairs by how often an edge between
    that pair of types was pruned -- endpoint order-independent (a
    gateway-sensor edge and a sensor-gateway edge are the same pair), which
    is what the roadmap's "which node types are most likely to be pruned"
    summary is asking for.
    """
    pair_counts = Counter()
    for d in _all_removed_edges(trajectory_log):
        pair = tuple(sorted((d["device_type_u"], d["device_type_v"])))
        pair_counts[pair] += 1
    return pair_counts.most_common(top_n)


def per_node_pruning_summary(trajectory_log: List[dict], node_features: Dict, node: int) -> List[Dict]:
    """For ONE node, list every incident edge the policy removed and why it
    plausibly stood out (its s_uv relative to the node's own risk/hub_score),
    directly matching the roadmap's "per-node summaries explaining why
    certain neighbors are removed."
    """
    entries = []
    for d in _all_removed_edges(trajectory_log):
        u, v = d["edge"]
        if node not in (u, v):
            continue
        neighbor = v if u == node else u
        neighbor_device_type = d["device_type_v"] if u == node else d["device_type_u"]
        neighbor_hub_score = d["hub_score_v"] if u == node else d["hub_score_u"]
        entries.append(
            {
                "neighbor": neighbor,
                "neighbor_device_type": neighbor_device_type,
                "neighbor_hub_score": neighbor_hub_score,
                "s_uv": d["s_uv"],
                "p_uv": d["p_uv"],
            }
        )
    entries.sort(key=lambda e: e["s_uv"])
    return entries


def format_explanation_report(trajectory_log: List[dict], original_graph: nx.Graph, gat_scores: Dict, p_uv: Dict) -> str:
    """Human-readable text report tying the above together -- the roadmap's
    "transparent justifications... suitable for security analysts or system
    operators," not just raw numbers.
    """
    summary = summarize_pruning_characteristics(trajectory_log, original_graph, gat_scores, p_uv)
    if summary["n_removed"] == 0:
        return "No edges were pruned in this trajectory."

    pairs = most_pruned_device_type_pairs(trajectory_log)

    lines = [
        f"Pruning explanation report: {summary['n_removed']} edges removed over {summary['n_steps_with_removals']} steps.",
        "",
        f"Attention score (s_uv): pruned-edge mean = {summary['removed_s_uv_mean']:.3f} vs. "
        f"graph-wide mean = {summary['population_s_uv_mean']:.3f} "
        f"({'lower' if summary['removed_s_uv_mean'] < summary['population_s_uv_mean'] else 'higher'} than typical).",
        f"Infection probability (p_uv): pruned-edge mean = {summary['removed_p_uv_mean']:.3f} vs. "
        f"graph-wide mean = {summary['population_p_uv_mean']:.3f} "
        f"({'higher' if summary['removed_p_uv_mean'] > summary['population_p_uv_mean'] else 'lower'} than typical -- "
        f"{'consistent with' if summary['removed_p_uv_mean'] > summary['population_p_uv_mean'] else 'NOT obviously consistent with'} "
        "targeting risky edges).",
        "",
        "Most frequently pruned device-type pairs:",
    ]
    for (type_a, type_b), count in pairs:
        lines.append(f"  {type_a} <-> {type_b}: {count} edges")

    lines.append("")
    lines.append("Device types appearing on a pruned edge (endpoint counts):")
    for device_type, count in summary["device_type_endpoint_counts"].items():
        lines.append(f"  {device_type}: {count}")

    return "\n".join(lines)
