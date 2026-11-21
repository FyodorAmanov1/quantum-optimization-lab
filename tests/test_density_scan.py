"""Check scan coverage without treating shared adjacent endpoints as independent."""
import copy
import itertools
import math
import unittest

import numpy as np

from src.hubbard_measurement import (
    density_binomial_interval, density_scan_intervals, finite_shot_study, measure_hubbard,
)


class DensityScanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.study = finite_shot_study()

    def test_family_reconstructs_adjusted_intervals_references_and_unique_cost(self):
        self.assertEqual(len(self.study["scan_families"]), 9)
        sources = {"grouped": "runs", "ungrouped": "ungrouped_runs",
                   "energy_range_weighted": "allocated_runs"}
        for family in self.study["scan_families"]:
            rows = [r for r in self.study[sources[family["grouping"]]]
                    if r["total_shots"] == 5 * family["grouped_shot_budget"]]
            self.assertEqual(family["total_shots"], sum(r["total_shots"] for r in rows))
            self.assertEqual(family["component_confidence"], 1 - .05/7)
            self.assertEqual(family, dict(density_scan_intervals(list(reversed(rows))),
                                         grouped_shot_budget=family["grouped_shot_budget"]))
            bounds = [density_binomial_interval(r["settings"], r["grouping"], 1-.05/7)
                      for r in rows]
            np.testing.assert_allclose(family["density_component_ci"], bounds)
            self.assertEqual(len(family["comparisons"]), 6)
            for i, c in enumerate(family["comparisons"]):
                a, b = [r["observables"]["double_occupancy"] for r in rows[i:i+2]]
                np.testing.assert_allclose(c["binomial_ci"],
                    [bounds[i][0]-bounds[i+1][1], bounds[i][1]-bounds[i+1][0]])
                proxy = a["squared_range_over_shots"] + b["squared_range_over_shots"]
                radius = np.sqrt(np.log(240)*proxy/2)
                np.testing.assert_allclose(c["ci"], [c["estimate"]-radius, c["estimate"]+radius])
                self.assertAlmostEqual(c["estimate"], a["estimate"]-b["estimate"])
                u, v = c["interaction_pair"]
                self.assertAlmostEqual(c["exact_ground_state"],
                    (v/np.sqrt(v*v+16)-u/np.sqrt(u*u+16))/2)
                self.assertEqual(c["resolved_decrease"], c["ci"][0] > 0)
                self.assertEqual(c["binomial_resolved_decrease"], c["binomial_ci"][0] > 0)
            self.assertEqual(family["binomial_resolved_count"],
                             sum(c["binomial_ci"][0] > 0 for c in family["comparisons"]))

    def test_exact_joint_coverage_with_shared_middle_interaction(self):
        # Enumerate every possible density count triple, not just sampled seeds.
        n, probabilities = 4, (.8, .4, .1)
        templates = []
        for u, p in enumerate(probabilities):
            r = measure_hubbard([np.sqrt(p), np.sqrt(1-p), 0, 0], [3, 6, 9, 12], u, n)
            r.update(interaction=u, exact_ground_state={"double_occupancy": p})
            templates.append(r)
        coverage = np.zeros(2)
        for successes in itertools.product(range(n+1), repeat=3):
            rows = copy.deepcopy(templates)
            probability = 1.
            for r, k, p in zip(rows, successes, probabilities):
                counts = [0]*16
                counts[3], counts[6] = k, n-k
                r["settings"][0]["counts"] = counts
                r["observables"]["double_occupancy"]["estimate"] = k/n
                probability *= math.comb(n, k)*p**k*(1-p)**(n-k)
            comparisons = density_scan_intervals(rows)["comparisons"]
            for j, key in enumerate(("ci", "binomial_ci")):
                if all(c[key][0] <= c["exact_prepared_state"] <= c[key][1] for c in comparisons):
                    coverage[j] += probability
        self.assertTrue(np.all(coverage >= .95-1e-12), coverage)

    def test_invalid_families(self):
        rows = self.study["runs"][:2]
        for runs, confidence in (([], .95), (rows[:1], .95), (rows, 0),
                                 (rows, 1), (rows, np.nan), ([rows[0]]*2, .95)):
            with self.assertRaises(ValueError):
                density_scan_intervals(runs, confidence)
        mixed = [self.study["runs"][0], self.study["ungrouped_runs"][3]]
        with self.assertRaises(ValueError):
            density_scan_intervals(mixed)


if __name__ == "__main__":
    unittest.main()
