import unittest

import numpy as np

from src.milestone3 import ACTION_SIZE, STOP_ACTION, build_pruning_env


class Milestone3EnvTests(unittest.TestCase):
    def test_reset_matches_unpruned_baseline_exactly(self):
        env = build_pruning_env(n_nodes=40, model_seed=0, max_steps=5, chunk_fraction=0.1)
        env.reset()
        # Post simulate_botnet neighbor-order fix, a freshly reset (unpruned)
        # episode must read EXACTLY the unpruned baseline -- not approximately.
        self.assertEqual(env._current_containment_ratio, 1.0)
        self.assertEqual(env._current_utility_ratio, 1.0)

    def test_action_mask_all_valid_at_reset(self):
        env = build_pruning_env(n_nodes=40, model_seed=0, max_steps=5, chunk_fraction=0.1)
        env.reset()
        mask = env.action_mask()
        self.assertEqual(mask.shape, (ACTION_SIZE,))
        self.assertTrue(mask.all())  # every bucket has edges, and STOP is always valid

    def test_action_mask_excludes_exhausted_bucket(self):
        env = build_pruning_env(n_nodes=40, model_seed=0, max_steps=20, chunk_fraction=1.0, max_removal_fraction=1.0)
        env.reset()
        # chunk_fraction=1.0 means a single "prune bucket 0" action removes the
        # WHOLE bucket at once, so it should immediately become invalid after.
        env.step(0)
        mask = env.action_mask()
        self.assertFalse(mask[0])
        self.assertTrue(mask[STOP_ACTION])  # STOP always stays valid

    def test_step_rejects_action_outside_range(self):
        env = build_pruning_env(n_nodes=40, model_seed=0, max_steps=5, chunk_fraction=0.1)
        env.reset()
        with self.assertRaises(AssertionError):
            env.step(ACTION_SIZE)


if __name__ == "__main__":
    unittest.main()
