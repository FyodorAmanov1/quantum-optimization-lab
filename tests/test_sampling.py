import itertools
import json
import math
import unittest

import numpy as np

import lab


class SamplingBaselineTests(unittest.TestCase):
    def test_analytic_uniform_laws_for_cardinalities(self):
        # Includes unique empty/full portfolios and interior binomial supports.
        for n in range(2, 7):
            bits = np.array(list(itertools.product((0, 1), repeat=n)))
            raw = -bits @ np.arange(1, n + 1, dtype=float)
            for k in range(n + 1):
                with self.subTest(n=n, k=k):
                    feasible = bits.sum(axis=1) == k
                    optimum = -sum(range(n - k + 1, n + 1))
                    all_p = lab.uniform_probabilities(2**n)
                    feasible_p = lab.uniform_probabilities(2**n, feasible)
                    np.testing.assert_array_equal(feasible_p[~feasible], 0)
                    np.testing.assert_allclose(feasible_p[feasible], 1 / math.comb(n, k))
                    all_m = lab.distribution_metrics(all_p, raw, feasible, optimum)
                    feasible_m = lab.distribution_metrics(feasible_p, raw, feasible, optimum)
                    self.assertAlmostEqual(all_m["feasible_probability"], math.comb(n, k) / 2**n)
                    self.assertAlmostEqual(feasible_m["feasible_probability"], 1)
                    # Each asset belongs to k/n of the feasible subsets.
                    expected_objective = -k * (n + 1) / 2
                    for metrics in (all_m, feasible_m):
                        self.assertAlmostEqual(metrics["conditional_feasible_objective"], expected_objective)
                        self.assertAlmostEqual(metrics["conditional_gap_to_optimum"], expected_objective - optimum)
                    self.assertAlmostEqual(all_m["expected_draws_per_feasible"], 2**n / math.comb(n, k))

    def test_weighted_conditioning_excludes_infeasible_objectives(self):
        metrics = lab.distribution_metrics([0.1, 0.2, 0.3, 0.4],
                                           [-1000, -5, -1, 1000],
                                           [False, True, True, False], -5)
        self.assertAlmostEqual(metrics["feasible_probability"], 0.5)
        self.assertAlmostEqual(metrics["conditional_feasible_objective"], -2.6)
        self.assertAlmostEqual(metrics["conditional_gap_to_optimum"], 2.4)
        self.assertAlmostEqual(metrics["expected_draws_per_feasible"], 2)

    def test_zero_feasible_mass_is_explicit_and_json_safe(self):
        for feasible in ([False, True], [False, False]):
            metrics = lab.distribution_metrics([1, 0], [-4, -2], feasible, -2)
            self.assertEqual(metrics["feasible_probability"], 0)
            self.assertIsNone(metrics["conditional_feasible_objective"])
            self.assertIsNone(metrics["conditional_gap_to_optimum"])
            self.assertIsNone(metrics["expected_draws_per_feasible"])
            json.dumps(metrics, allow_nan=False)

    def test_invalid_distributions_rejected(self):
        for probabilities in ([0.2, 0.2], [-0.1, 1.1], [np.nan, 0],
                              [np.inf, 0], [[0.5, 0.5]], []):
            with self.subTest(probabilities=probabilities), self.assertRaises(ValueError):
                lab.distribution_metrics(probabilities, [-2, -1], [True, False], -2)
        with self.assertRaises(ValueError):
            lab.distribution_metrics([0.5, 0.5], [-2], [True, False], -2)
        for size, support in ((0, None), (2, [False, False]), (2, [True]), (2.5, None)):
            with self.subTest(size=size, support=support), self.assertRaises(ValueError):
                lab.uniform_probabilities(size, support)

    def test_zero_angle_qaoa_has_uniform_baseline_metrics(self):
        raw = np.array([0, -3, -1, -4], dtype=float)
        feasible = np.array([False, True, True, False])
        qaoa = lab.qaoa_probabilities(raw, 0, 0.37)
        actual = lab.distribution_metrics(qaoa, raw, feasible, -3)
        expected = lab.distribution_metrics(lab.uniform_probabilities(4), raw, feasible, -3)
        for key in expected:
            self.assertAlmostEqual(actual[key], expected[key])

    def test_experiment_comparison_and_legacy_metrics(self):
        result = lab.experiment()
        json.dumps(result, allow_nan=False)
        metrics = result["metrics"]
        comparison = metrics["sampling_comparison"]
        self.assertEqual(set(comparison), {"qaoa", "uniform_all_bitstrings", "uniform_feasible_portfolios"})
        for key in ("feasible_probability", "conditional_feasible_objective", "conditional_gap_to_optimum"):
            self.assertEqual(metrics[key], comparison["qaoa"][key])
        all_m = comparison["uniform_all_bitstrings"]
        feasible_m = comparison["uniform_feasible_portfolios"]
        self.assertAlmostEqual(all_m["feasible_probability"], 20 / 64)
        self.assertAlmostEqual(feasible_m["feasible_probability"], 1)
        self.assertAlmostEqual(all_m["conditional_feasible_objective"], feasible_m["conditional_feasible_objective"])
        for method in comparison.values():
            self.assertTrue(0 <= method["feasible_probability"] <= 1)
            self.assertGreaterEqual(method["expected_draws_per_feasible"], 1)
            self.assertGreaterEqual(method["conditional_gap_to_optimum"], -1e-12)
            self.assertAlmostEqual(method["conditional_gap_to_optimum"],
                                   method["conditional_feasible_objective"] - metrics["exact"]["objective"])
        # The existing grid includes gamma=0, hence the all-bitstring law.
        self.assertLessEqual(comparison["qaoa"]["expected_penalized_objective"],
                             all_m["expected_penalized_objective"] + 1e-12)


if __name__ == "__main__":
    unittest.main()
