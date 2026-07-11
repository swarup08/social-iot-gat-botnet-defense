"""Milestone 4: multi-graph statistical evaluation harness.

The foundation for every Milestone 4 ablation/stress-test: everything else
in this milestone (GAT-depth/heads ablation, RL reward-weighting ablation,
noisy-feature ablation, graph-size/density and attack-aggressiveness stress
tests) reuses this same "many graph instances, paired tests, Holm
correction, absolute effect sizes, compute cost" methodology rather than
each inventing its own.

Per this project's own supervisor-feedback history (see CLAUDE.md): "Method
A beats random and Method B doesn't" does NOT imply "A beats B" -- a direct
PAIRED test between A and B, on the SAME graph instances, is required to
claim that. And running many comparisons without correcting for multiple
comparisons inflates apparent significance -- Holm (a step-down, less
conservative than plain Bonferroni but still valid) is applied here.
"""

from __future__ import annotations

import statistics
from typing import Dict, List

from scipy import stats


def holm_correction(p_values: List[float]) -> List[float]:
    """Holm-Bonferroni step-down correction.

    Returns adjusted p-values in the SAME order as the input list (not
    sorted) -- each can be compared directly against alpha to decide
    significance (reject H0 if adjusted_p < alpha).

    Algorithm: sort ascending, multiply the i-th smallest (1-indexed) raw
    p-value by (m - i + 1), then take a running maximum over the sorted
    order so adjusted p-values are monotonically non-decreasing (a
    correction step required by the procedure -- without it, a smaller raw
    p-value could end up with a LARGER adjusted p-value than one ranked
    after it, which would be incoherent), then clip to 1.0.
    """
    m = len(p_values)
    if m == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (original_index, p) in enumerate(indexed):
        multiplier = m - rank  # 1-indexed: rank 0 -> m, rank 1 -> m-1, ...
        running_max = max(running_max, p * multiplier)
        adjusted[original_index] = min(running_max, 1.0)
    return adjusted


def paired_comparison(method_a: str, method_b: str, values_a: List[float], values_b: List[float]) -> Dict:
    """Paired t-test + absolute effect size between two methods' per-graph values.

    values_a/values_b must be the SAME length, index-aligned by graph
    instance (values_a[i] and values_b[i] measured on the same graph) --
    that pairing is what makes this a within-subject comparison, more
    powerful than an unpaired test on the same sample size.
    """
    assert len(values_a) == len(values_b), "paired comparison requires equal-length, index-aligned samples"
    t_stat, p_value = stats.ttest_rel(values_a, values_b)
    return {
        "method_a": method_a,
        "method_b": method_b,
        "mean_a": statistics.mean(values_a),
        "mean_b": statistics.mean(values_b),
        "mean_diff": statistics.mean(values_a) - statistics.mean(values_b),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "n": len(values_a),
    }


def run_paired_tests_with_correction(pairs: List[tuple], results_by_method: Dict[str, List[float]], alpha: float = 0.05) -> List[Dict]:
    """Run paired_comparison for every (method_a, method_b) pair, then apply
    Holm correction across the WHOLE family of tests at once (not per-pair) --
    the family is exactly the set of tests passed in, so callers should pass
    the full curated set they intend to interpret together, not call this
    repeatedly with growing subsets (that would silently under-correct).
    """
    comparisons = [paired_comparison(a, b, results_by_method[a], results_by_method[b]) for a, b in pairs]
    raw_p_values = [c["p_value"] for c in comparisons]
    adjusted_p_values = holm_correction(raw_p_values)
    for comparison, adjusted_p in zip(comparisons, adjusted_p_values):
        comparison["p_value_holm"] = adjusted_p
        comparison["significant_holm"] = adjusted_p < alpha
    return comparisons
