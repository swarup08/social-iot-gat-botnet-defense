import unittest

from src.milestone4 import holm_correction, paired_comparison, run_paired_tests_with_correction


class Milestone4HarnessTests(unittest.TestCase):
    def test_holm_correction_known_example(self):
        # Worked example: 4 raw p-values, sorted ascending -> multipliers 4,3,2,1.
        # p=(0.01, 0.02, 0.03, 0.04) -> candidates (0.04, 0.06, 0.06, 0.04) ->
        # running max enforces monotonic non-decreasing: (0.04, 0.06, 0.06, 0.06).
        raw = [0.01, 0.02, 0.03, 0.04]
        adjusted = holm_correction(raw)
        self.assertAlmostEqual(adjusted[0], 0.04, places=6)
        self.assertAlmostEqual(adjusted[1], 0.06, places=6)
        self.assertAlmostEqual(adjusted[2], 0.06, places=6)
        self.assertAlmostEqual(adjusted[3], 0.06, places=6)

    def test_holm_correction_preserves_input_order(self):
        # Same values as above but scrambled input order -- output must track input order.
        raw = [0.04, 0.01, 0.03, 0.02]
        adjusted = holm_correction(raw)
        # index 1 (raw=0.01, smallest) should map to the smallest adjusted value.
        self.assertEqual(adjusted.index(min(adjusted)), 1)

    def test_holm_correction_clips_at_one(self):
        adjusted = holm_correction([0.9, 0.9, 0.9])
        self.assertTrue(all(a <= 1.0 for a in adjusted))

    def test_holm_correction_empty_list(self):
        self.assertEqual(holm_correction([]), [])

    def test_paired_comparison_detects_clear_difference(self):
        values_a = [1.0] * 20
        values_b = [0.0] * 20
        result = paired_comparison("A", "B", values_a, values_b)
        self.assertAlmostEqual(result["mean_diff"], 1.0)
        self.assertLess(result["p_value"], 0.001)

    def test_paired_comparison_requires_equal_length(self):
        with self.assertRaises(AssertionError):
            paired_comparison("A", "B", [1.0, 2.0], [1.0])

    def test_run_paired_tests_with_correction_applies_holm_across_family(self):
        results_by_method = {
            "A": [1.0] * 10,
            "B": [0.0] * 10,
            "C": [0.5] * 10,
        }
        pairs = [("A", "B"), ("A", "C"), ("B", "C")]
        comparisons = run_paired_tests_with_correction(pairs, results_by_method)
        self.assertEqual(len(comparisons), 3)
        for c in comparisons:
            self.assertIn("p_value_holm", c)
            self.assertGreaterEqual(c["p_value_holm"], c["p_value"])  # Holm-adjusted p >= raw p, always


if __name__ == "__main__":
    unittest.main()
