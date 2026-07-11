import unittest

import numpy as np
import torch

from src.milestone3 import ACTION_SIZE, build_pruning_env
from src.milestone3_dqn import QNetwork, ReplayBuffer, Transition, select_action, train_dqn


class Milestone3DqnTests(unittest.TestCase):
    def test_select_action_respects_action_mask(self):
        state = np.zeros(18, dtype=np.float32)
        q_network = QNetwork(state_size=18, action_size=ACTION_SIZE)
        mask = np.array([False, False, True, False, False, True])  # only bucket 2 and STOP valid

        for _ in range(20):
            action_explore = select_action(q_network, state, epsilon=1.0, action_size=ACTION_SIZE, action_mask=mask)
            self.assertIn(action_explore, (2, 5))
        action_greedy = select_action(q_network, state, epsilon=0.0, action_size=ACTION_SIZE, action_mask=mask)
        self.assertIn(action_greedy, (2, 5))

    def test_replay_buffer_fraction_deep(self):
        buffer = ReplayBuffer(capacity=10)
        shallow_state = np.array([0.1] + [0.0] * 17, dtype=np.float32)
        deep_state = np.array([0.5] + [0.0] * 17, dtype=np.float32)
        for _ in range(3):
            buffer.push(Transition(shallow_state, 0, 0.0, shallow_state, False))
        for _ in range(2):
            buffer.push(Transition(deep_state, 0, 0.0, deep_state, False))
        self.assertAlmostEqual(buffer.fraction_deep(threshold=0.4), 2 / 5)

    def test_train_dqn_runs_end_to_end_and_returns_log(self):
        env = build_pruning_env(n_nodes=40, model_seed=0, max_steps=5, chunk_fraction=0.15)
        q_network, log = train_dqn(env, n_episodes=10, epsilon_decay_episodes=5, min_buffer_before_training=16, batch_size=16, buffer_capacity=200)

        self.assertIsInstance(q_network, QNetwork)
        self.assertEqual(len(log["episode_return"]), 10)
        self.assertEqual(len(log["episode_epsilon"]), 10)
        with torch.no_grad():
            q_values = q_network(torch.zeros(1, env.state_size))
        self.assertEqual(q_values.shape, (1, env.action_size))


if __name__ == "__main__":
    unittest.main()
