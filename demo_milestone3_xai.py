"""Milestone 3 demo: XAI explanation report for the trained RL policy.

Trains the DQN (same setup as demo_milestone3_dqn.py), runs the final
greedy policy, and generates the roadmap's "simple explanation artifacts"
from its trajectory_log: aggregate pruned-edge characteristics, most
frequently pruned device-type pairs, and a per-node summary for a couple of
representative nodes. The device-type-pair characteristics table and the
hub node's per-node summary are also saved as persistent CSV files (not just
printed to console), since these are meant to be reusable report artifacts.
"""

from src.milestone3 import build_pruning_env
from src.milestone3_dqn import run_greedy_episode, train_dqn
from src.milestone3_xai import (
    format_explanation_report,
    most_pruned_edge_characteristics,
    per_node_pruning_summary,
    save_table_csv,
)


def main() -> None:
    print("building environment and training RL policy...")
    env = build_pruning_env(n_nodes=300, model_seed=0, max_steps=15, max_removal_fraction=0.5, chunk_fraction=0.05)
    q_network, _ = train_dqn(env, n_episodes=300, epsilon_decay_episodes=200, seed=0)
    run_greedy_episode(env, q_network)
    trajectory_log = env.trajectory_log

    print("\n" + format_explanation_report(trajectory_log, env.original_graph, env.gat_scores, env.p_uv))

    edge_table = most_pruned_edge_characteristics(trajectory_log)
    edge_table_path = save_table_csv(edge_table, "pruned_edge_characteristics.csv")
    print(f"\nsaved most-frequently-pruned-edges characteristics table to {edge_table_path} ({len(edge_table)} device-type-pair rows)")

    print("\n\nper-node summaries for the hub and mid-degree seed nodes:")
    for role in ("hub", "mid"):
        node = env.seed_nodes[role]
        entries = per_node_pruning_summary(trajectory_log, env.node_features, node)
        print(f"\n  node {node} ({role}, device_type={env.node_features[node]['device_type']}, degree={env.original_graph.degree(node)}):")
        if not entries:
            print("    no incident edges were pruned")
        for e in entries:
            print(
                f"    removed edge to neighbor {e['neighbor']} ({e['neighbor_device_type']}, "
                f"hub_score={e['neighbor_hub_score']:.3f}): s_uv={e['s_uv']:.3f}  p_uv={e['p_uv']:.3f}"
            )
        if role == "hub":
            hub_summary_path = save_table_csv(entries, "hub_node_pruning_summary.csv")
            print(f"    saved hub per-node pruning summary to {hub_summary_path} ({len(entries)} rows)")


if __name__ == "__main__":
    main()
