"""Small demo script for Milestone 2.

Trains a small GAT on one labeled Social IoT graph (benign vs compromised node
classification) and reports test accuracy against a majority-class baseline.

Two sources of noise are controlled for here, following the same "don't trust
a single run" lesson from Milestone 1's supervisor feedback:
  1. Split-hardness: which nodes land in train/val/test varies by random split.
     Repeated over N_SPLITS different splits.
  2. Training-convergence noise: for a FIXED split, model weight init and
     dropout stochasticity can still make one training run converge better or
     worse than another. Repeated over N_MODEL_SEEDS different seeds PER split.

Separating these two lets us tell "this split's test set was just unlucky"
apart from "this training run didn't converge well" -- see the per-split
diagnostics printed below. The graph is sized (n_nodes=300) so each split's
test set has double-digit-plus compromised nodes, since single-digit counts
made per-split recall swing wildly on pure sampling noise alone (a 1-2 node
difference moved recall by 10+ points at n_nodes=150).

This is still not Milestone 3's full statistical harness (many DIFFERENT
graphs, multiple-comparison correction) -- just the same "one result isn't
evidence" discipline applied within a single graph.
"""

import statistics

import torch
from scipy import stats

from src.milestone2 import accuracy, build_pyg_data, generate_labeled_graph, make_node_split, precision_recall_f1_counts, train_gat

N_NODES = 300  # sized so ~20% test masks have comfortably double-digit+ compromised nodes
EPOCHS = 300
N_SPLITS = 5
N_MODEL_SEEDS = 4
COMPROMISED_CLASS = 1
# A split is flagged as an apparent convergence issue (rather than plain split
# hardness) if recall varies this much across model seeds trained on the SAME
# split -- i.e. some inits find the compromised nodes and others don't.
WITHIN_SPLIT_RECALL_STD_FLAG = 0.25


def majority_baseline_accuracy(y: torch.Tensor, train_mask: torch.Tensor, test_mask: torch.Tensor) -> float:
    """Accuracy of always predicting the majority class -- fit on TRAIN, scored on TEST.

    Using only the training split's majority class (rather than the global
    label distribution) keeps this a fair, apples-to-apples baseline: like the
    GAT, it never gets to look at test labels before making its prediction.
    """
    majority_class = int(y[train_mask].float().mean().item() >= 0.5)
    predictions = torch.full_like(y, majority_class)
    return accuracy(predictions, y, test_mask)


def train_one_run(data, train_mask, val_mask, test_mask, model_seed: int, weight_mildness: float) -> dict:
    """Train one GAT (one model-init seed) on a FIXED split and return test metrics.

    Delegates the actual training loop to src.milestone2.train_gat, the
    canonical trainer used by both this evaluation sweep and (going forward)
    attention-score extraction, so there's one place the training logic lives.
    """
    model = train_gat(data, train_mask, model_seed=model_seed, weight_mildness=weight_mildness, epochs=EPOCHS)
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = logits.argmax(dim=1)

    result = precision_recall_f1_counts(predictions, data.y, test_mask, COMPROMISED_CLASS)
    result["test_acc"] = accuracy(predictions, data.y, test_mask)
    return result


def run_experiment(data, weight_mildness: float) -> list:
    """Train N_MODEL_SEEDS runs on each of N_SPLITS splits; return per-split summaries.

    Each entry separates the two noise sources: `seed_runs` holds the raw
    per-model-seed results (training-convergence noise, split held fixed),
    while the split-level fields (baseline_acc, test_positive_count) are fixed
    properties of that split (split-hardness), unaffected by model seed.
    """
    split_summaries = []
    for split_seed in range(N_SPLITS):
        train_mask, val_mask, test_mask = make_node_split(data.num_nodes, seed=split_seed)
        baseline_acc = majority_baseline_accuracy(data.y, train_mask, test_mask)
        test_positive_count = int(data.y[test_mask].sum())

        seed_runs = []
        for model_seed in range(N_MODEL_SEEDS):
            # Offset by split_seed so no (split, model_seed) pair anywhere in
            # the sweep reuses the exact same torch RNG state as another.
            run = train_one_run(data, train_mask, val_mask, test_mask, model_seed=split_seed * 1000 + model_seed, weight_mildness=weight_mildness)
            seed_runs.append(run)

        split_summaries.append(
            {
                "split_seed": split_seed,
                "baseline_acc": baseline_acc,
                "test_positive_count": test_positive_count,
                "seed_runs": seed_runs,
            }
        )
    return split_summaries


def report(label: str, split_summaries: list) -> None:
    print(f"\n=== {label} ===")
    all_split_mean_acc = []
    all_split_mean_recall = []
    all_split_mean_precision = []
    all_split_mean_f1 = []
    baseline_accs = []

    for summary in split_summaries:
        seed_runs = summary["seed_runs"]
        accs = [r["test_acc"] for r in seed_runs]
        recalls = [r["recall"] for r in seed_runs]
        precisions = [r["precision"] for r in seed_runs]
        f1s = [r["f1"] for r in seed_runs]
        found_counts = [r["true_positive"] for r in seed_runs]
        predicted_counts = [r["predicted_positive"] for r in seed_runs]

        recall_std = statistics.stdev(recalls) if len(recalls) > 1 else 0.0
        mean_recall = statistics.mean(recalls)
        mean_predicted = statistics.mean(predicted_counts)

        # Convergence-issue flag: recall swings widely ACROSS model seeds on
        # the very same split (some inits find the compromised nodes, others
        # don't) -- that's optimization instability, not "this split was hard".
        # Split-hardness (as opposed to a convergence bug) instead shows up as
        # LOW recall that's fairly CONSISTENT across all seeds.
        is_unstable = recall_std >= WITHIN_SPLIT_RECALL_STD_FLAG
        is_collapsed = mean_recall < 0.2 and mean_predicted < 0.5 * summary["test_positive_count"] and recall_std < WITHIN_SPLIT_RECALL_STD_FLAG

        flag = ""
        if is_unstable:
            flag = "  <-- FLAG: high variance across model seeds (apparent convergence instability, not split hardness)"
        elif is_collapsed:
            flag = "  <-- FLAG: consistently low recall across ALL seeds (looks like split hardness / systematic under-prediction, not noise)"

        print(
            f"  split {summary['split_seed']}: test_positives={summary['test_positive_count']:2d}  "
            f"baseline_acc={summary['baseline_acc']:.3f}  "
            f"found/seed={found_counts}  "
            f"recall={[round(r, 3) for r in recalls]} (mean={mean_recall:.3f} std={recall_std:.3f}){flag}"
        )

        all_split_mean_acc.append(statistics.mean(accs))
        all_split_mean_recall.append(mean_recall)
        all_split_mean_precision.append(statistics.mean(precisions))
        all_split_mean_f1.append(statistics.mean(f1s))
        baseline_accs.append(summary["baseline_acc"])

    print(f"\n  per-split GAT test accuracy (avg over {N_MODEL_SEEDS} model seeds): {[round(a, 3) for a in all_split_mean_acc]}")
    print(f"  per-split baseline accuracy:                                    {[round(a, 3) for a in baseline_accs]}")
    print(
        f"  overall: acc={statistics.mean(all_split_mean_acc):.3f}+/-{statistics.stdev(all_split_mean_acc):.3f}  "
        f"precision={statistics.mean(all_split_mean_precision):.3f}+/-{statistics.stdev(all_split_mean_precision):.3f}  "
        f"recall={statistics.mean(all_split_mean_recall):.3f}+/-{statistics.stdev(all_split_mean_recall):.3f}  "
        f"f1={statistics.mean(all_split_mean_f1):.3f}+/-{statistics.stdev(all_split_mean_f1):.3f}"
    )

    # Paired t-test on split-level means: each pair is (this split's GAT
    # accuracy averaged over model seeds) vs (this split's baseline accuracy),
    # so training-convergence noise is already averaged out before the test.
    t_stat, p_value = stats.ttest_rel(all_split_mean_acc, baseline_accs)
    print(f"  paired t-test (split-level GAT mean vs. baseline): t={t_stat:.3f} p={p_value:.3f} (n={N_SPLITS} splits, still a small n)")
    if p_value < 0.05:
        print("  -> statistically distinguishable from noise at alpha=0.05")
    else:
        print("  -> NOT statistically distinguishable from noise at alpha=0.05 -- treat as an unproven, weak signal")


def main() -> None:
    graph, node_features, labels = generate_labeled_graph(n_nodes=N_NODES)
    data = build_pyg_data(graph, node_features, labels)
    infected_fraction = data.y.float().mean().item()
    print(f"nodes={data.num_nodes} edges={data.num_edges // 2} features={data.num_node_features}")
    print(f"infected_fraction={infected_fraction:.3f}")
    print(f"running {N_SPLITS} splits x {N_MODEL_SEEDS} model seeds = {N_SPLITS * N_MODEL_SEEDS} training runs per config...")

    for mildness, label in [(0.0, "UNWEIGHTED (mildness=0.0)"), (0.5, "MILDLY-WEIGHTED (mildness=0.5, sqrt of balanced)"), (1.0, "FULLY-BALANCED (mildness=1.0)")]:
        summaries = run_experiment(data, weight_mildness=mildness)
        report(label, summaries)


if __name__ == "__main__":
    main()
