import unittest
import numpy as np
import lab


class OptimizationTests(unittest.TestCase):
    def test_exact_known_selection(self):
        result = lab.exact_portfolio(np.array([1, 2, 3]), np.zeros((3, 3)), cardinality=2)
        self.assertEqual(result["bits"], [0, 1, 1])
        self.assertEqual(result["objective"], -5)

    def test_unitarity_and_zero_angles(self):
        energy = np.arange(8)
        np.testing.assert_allclose(lab.qaoa_probabilities(energy, 0, 0), np.ones(8) / 8)
        for gamma, beta in [(0.7, 0.2), (2.1, 1.3)]:
            self.assertAlmostEqual(float(lab.qaoa_probabilities(energy, gamma, beta).sum()), 1)

    def test_covariance_psd(self):
        _, covariance = lab.problem()
        self.assertGreaterEqual(float(np.linalg.eigvalsh(covariance).min()), -1e-12)

    def test_gap_and_feasibility(self):
        result = lab.experiment()["metrics"]
        self.assertGreaterEqual(result["conditional_gap_to_optimum"], -1e-12)
        self.assertTrue(0 <= result["feasible_probability"] <= 1)


if __name__ == "__main__":
    unittest.main()
