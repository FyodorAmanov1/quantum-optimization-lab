import json
import unittest

import numpy as np

from src.hubbard_measurement import (
    SETTINGS, binomial_interval, density_binomial_interval, finite_shot_study,
    measure_hubbard, measurement_probabilities, pauli_matrix,
)
from src.studio import experiment as studio_experiment, hubbard, vqe, vqe_state


class HubbardMeasurementTests(unittest.TestCase):
    def test_pauli_mapping_matches_independent_fermion_hamiltonian(self):
        kinetic = sum(-0.5 * pauli_matrix(s) for s in SETTINGS[1:])
        z = [pauli_matrix("I" * q + "Z" + "I" * (3 - q)) for q in range(4)]
        density = (2 * np.eye(16) - sum(z) + z[0] @ z[1] + z[2] @ z[3]) / 4
        for u in (-2, 0, 1, 8):
            h, basis = hubbard(u)
            np.testing.assert_allclose((kinetic + u * density)[np.ix_(basis, basis)], h)

    def test_rotations_match_complex_state_pauli_expectations(self):
        # Non-real amplitudes detect an incorrect Y-basis rotation or bit order.
        rng = np.random.default_rng(73)
        wave = rng.normal(size=16) + 1j * rng.normal(size=16)
        wave /= np.linalg.norm(wave)
        for setting in SETTINGS[1:]:
            probabilities = measurement_probabilities(wave, setting)
            parity = np.ones(16)
            for q, symbol in enumerate(setting):
                if symbol != "I":
                    parity *= 1 - 2 * ((np.arange(16) >> q) & 1)
            self.assertAlmostEqual(probabilities @ parity,
                                   np.vdot(wave, pauli_matrix(setting) @ wave).real)

    def test_exact_ground_observables_and_legacy_api(self):
        for u in (0, 2, 8):
            wave, metrics = vqe_state(u)
            np.testing.assert_allclose(vqe(u), metrics)
            _, basis = hubbard(u)
            measured = measure_hubbard(wave, basis, u, 4096, 17)
            expected_d = (1 - u / np.sqrt(u * u + 16)) / 2
            obs = measured["observables"]
            self.assertAlmostEqual(obs["double_occupancy"]["exact_prepared_state"], expected_d, places=7)
            self.assertAlmostEqual(obs["energy"]["exact_prepared_state"], metrics[0], places=12)
            self.assertAlmostEqual(obs["kinetic_energy"]["exact_prepared_state"], -8 / np.sqrt(u*u + 16), places=7)

    def test_counts_reconstruct_estimators_and_reproduce(self):
        wave, _ = vqe_state(4)
        _, basis = hubbard(4)
        run = measure_hubbard(wave, basis, 4, 256, 51)
        self.assertEqual(run, measure_hubbard(wave, basis, 4, 256, 51))
        self.assertNotEqual(run["settings"], measure_hubbard(wave, basis, 4, 256, 52)["settings"])
        self.assertEqual(run["total_shots"], 1280)
        d = 0
        k = 0
        for group in run["settings"]:
            self.assertEqual(sum(group["counts"]), 256)
            for outcome, count in enumerate(group["counts"]):
                if group["basis"] == "ZZZZ":
                    d += count * (int(outcome & 3 == 3) + int(outcome & 12 == 12)) / 256
                else:
                    parity = (-1) ** sum((outcome >> q) & 1 for q, s in enumerate(group["basis"]) if s != "I")
                    k += -0.5 * count * parity / 256
        self.assertAlmostEqual(run["observables"]["energy"]["estimate"], k + 4*d)
        self.assertAlmostEqual(run["observables"]["double_occupancy"]["estimate"], d)
        json.dumps(run, allow_nan=False)

    def test_ci_does_not_collapse_on_deterministic_counts_and_scales(self):
        _, basis = hubbard(8)
        wave = np.array([1, 0, 0, 0])
        low = measure_hubbard(wave, basis, 8, 256, 12)
        high = measure_hubbard(wave, basis, 8, 4096, 12)
        self.assertEqual(low["observables"]["double_occupancy"]["standard_error"], 0)
        for name, o in low["observables"].items():
            width = o["ci"][1] - o["ci"][0]
            other = high["observables"][name]["ci"]
            self.assertGreater(width, 0)
            self.assertAlmostEqual(width / (other[1] - other[0]), 4)

    def test_repeated_sampling_agrees_with_exact_mean_and_interval_coverage(self):
        h, basis = hubbard(4)
        wave = np.linalg.eigh(h)[1][:, 0]
        estimates, variances, covered = [], [], 0
        for seed in range(200):
            o = measure_hubbard(wave, basis, 4, 256, seed)["observables"]["energy"]
            estimates.append(o["estimate"])
            variances.append(o["standard_error"]**2)
            covered += o["ci"][0] <= o["exact_prepared_state"] <= o["ci"][1]
        self.assertLess(abs(np.mean(estimates) - np.linalg.eigvalsh(h)[0]),
                        4 * np.sqrt(np.mean(variances) / 200))
        self.assertGreater(covered, 190)
        self.assertAlmostEqual(np.var(estimates, ddof=1) / np.mean(variances), 1, delta=0.3)

    def test_invalid_inputs(self):
        _, basis = hubbard(4)
        for shots in (0, 1, -4, 2.5, True):
            with self.assertRaises(ValueError):
                measure_hubbard([1, 0, 0, 0], basis, 4, shots)
        for wave in ([0, 0, 0, 0], [1, 1, 0, 0], [np.nan, 0, 0, 0], [1, 0]):
            with self.assertRaises(ValueError):
                measure_hubbard(wave, basis, 4)
        for confidence in (0, 1, np.nan):
            with self.assertRaises(ValueError):
                measure_hubbard([1, 0, 0, 0], basis, 4, confidence=confidence)

    def test_study_design_and_references(self):
        study = finite_shot_study()
        self.assertEqual(len(study["runs"]), 21)
        for r in study["runs"]:
            u = r["interaction"]
            self.assertAlmostEqual(r["exact_ground_state"]["energy"], (u - np.sqrt(u*u + 16))/2)
            self.assertEqual(r["total_shots"], 5*r["shots_per_setting"])
        for grouped, ungrouped in zip(study["runs"], study["ungrouped_runs"]):
            self.assertEqual(grouped["total_shots"], ungrouped["total_shots"])
            self.assertEqual(len(ungrouped["settings"]), 10)
            self.assertEqual(grouped["exact_ground_state"], ungrouped["exact_ground_state"])

    def test_ungrouped_counts_reconstruct_all_observables(self):
        h, basis = hubbard(-2)
        wave = np.linalg.eigh(h)[1][:, 0]
        run = measure_hubbard(wave, basis, -2, 128, 71, grouping="ungrouped")
        density, kinetic = 0.5, 0.
        for index, group in enumerate(run["settings"]):
            self.assertEqual(sum(group["counts"]), 128)
            parity_mean = sum(count * (-1)**sum((outcome >> q) & 1
                              for q, s in enumerate(group["basis"]) if s != "I")
                              for outcome, count in enumerate(group["counts"])) / 128
            if index < 6:
                density += (-0.25 if index < 4 else 0.25) * parity_mean
            else:
                kinetic -= 0.5 * parity_mean
        obs = run["observables"]
        self.assertAlmostEqual(obs["double_occupancy"]["estimate"], density)
        self.assertAlmostEqual(obs["kinetic_energy"]["estimate"], kinetic)
        self.assertAlmostEqual(obs["energy"]["estimate"], kinetic - 2*density)
        self.assertEqual(run, measure_hubbard(wave, basis, -2, 128, 71, grouping="ungrouped"))
        with self.assertRaises(ValueError):
            measure_hubbard(wave, basis, -2, grouping="unknown")

    def test_matched_budget_variance_and_ungrouped_coverage(self):
        h, basis = hubbard(8)
        wave = np.linalg.eigh(h)[1][:, 0]
        grouped = measure_hubbard(wave, basis, 8, 256, 0)
        baseline = measure_hubbard(wave, basis, 8, 128, 0, grouping="ungrouped")
        # Exact marginal density variance in this half-filled symmetric dimer.
        d = (1 - 8 / np.sqrt(80)) / 2
        self.assertAlmostEqual(grouped["observables"]["double_occupancy"]["exact_sampling_standard_error"]**2,
                               d * (1-d) / 256)
        self.assertLess(grouped["observables"]["energy"]["exact_sampling_standard_error"],
                        baseline["observables"]["energy"]["exact_sampling_standard_error"])
        values, covered = [], 0
        for seed in range(200):
            obs = measure_hubbard(wave, basis, 8, 128, seed,
                                  grouping="ungrouped")["observables"]["energy"]
            values.append(obs["estimate"])
            covered += obs["ci"][0] <= obs["exact_prepared_state"] <= obs["ci"][1]
        predicted = baseline["observables"]["energy"]["exact_sampling_standard_error"]
        self.assertAlmostEqual(np.var(values, ddof=1) / predicted**2, 1, delta=0.3)
        self.assertLess(abs(np.mean(values) - np.linalg.eigvalsh(h)[0]), 4*predicted/np.sqrt(200))
        self.assertGreater(covered, 190)

    def test_studio_views_expose_matched_budgets_and_uncertainty(self):
        cases = studio_experiment()["cases"]
        self.assertEqual([c["id"] for c in cases], ["allocation", "electrons", "hubbard-shots"])
        case = cases[-1]
        self.assertEqual(case["default_view"], "4096")
        self.assertEqual(len(case["table"]["rows"]), 126)
        self.assertEqual({row[0] for row in case["table"]["rows"]}, {"grouped", "ungrouped"})
        for view, budget in zip(case["views"], (256, 4096, 65536)):
            self.assertEqual(view["value"], str(budget))
            self.assertEqual(view["metrics"][2]["value"], 5*budget)
            self.assertEqual(len(view["charts"]), 5)
            for chart in view["charts"]:
                self.assertEqual(chart["x"], [0, 0.5, 1, 2, 4, 6, 8])
                for series in chart["series"]:
                    self.assertEqual(len(series["values"]), 7)
                    self.assertTrue(np.all(np.isfinite(series["values"])))
            energy = view["charts"][0]
            self.assertEqual(energy["series"][0]["name"], "Exact ground state")
            self.assertAlmostEqual(energy["series"][0]["values"][-1], (8-np.sqrt(80))/2)
            self.assertEqual(len(view["metrics"]), 8)
            self.assertIn("95% CI", view["metrics"][4]["note"])
            self.assertIn("exact ground-state difference", view["metrics"][5]["note"])
            self.assertIn("binomial", view["metrics"][6]["label"])
            self.assertIn("same counts", view["metrics"][7]["note"])

    def test_binomial_intervals_match_scipy_and_have_exact_coverage(self):
        from math import comb
        from scipy.stats import binomtest
        n = 16
        intervals = [binomial_interval(k, n) for k in range(n+1)]
        for k, interval in enumerate(intervals):
            expected = binomtest(k, n).proportion_ci(method="exact")
            np.testing.assert_allclose(interval, [expected.low, expected.high], atol=1e-12)
        for p in (0., 0.001, 0.01, 0.05, 0.2, 0.5, 0.99, 1.):
            coverage = sum(comb(n, k)*p**k*(1-p)**(n-k)
                           for k, (low, high) in enumerate(intervals) if low <= p <= high)
            self.assertGreaterEqual(coverage, 0.95 - 1e-12)
        self.assertGreater(intervals[0][1], 0)
        self.assertLess(intervals[-1][0], 1)
        for k, shots, confidence in ((-1, 5, .95), (6, 5, .95), (1, 0, .95),
                                      (1.5, 5, .95), (True, 5, .95), (1, 5, np.nan)):
            with self.assertRaises(ValueError):
                binomial_interval(k, shots, confidence)

    def test_binomial_contrast_uses_count_only_adjusted_components(self):
        study = finite_shot_study()
        for contrast in study["contrasts"]:
            strategy = contrast["grouping"]
            rows = study["runs" if strategy == "grouped" else "ungrouped_runs"]
            pair = [next(r for r in rows if r["interaction"] == u and
                         r["total_shots"] * 2 == contrast["total_shots"]) for u in (6, 8)]
            a, b = [density_binomial_interval(r["settings"], strategy, 0.975) for r in pair]
            np.testing.assert_allclose(contrast["binomial_ci"], [a[0]-b[1], a[1]-b[0]])
            self.assertEqual(contrast["binomial_resolved_decrease"], a[0]-b[1] > 0)
        # Preselected intermediate-budget grouped finding; retain Hoeffding's negative result.
        grouped = next(c for c in study["contrasts"] if c["grouping"] == "grouped"
                       and c["grouped_shot_budget"] == 4096)
        self.assertTrue(grouped["binomial_resolved_decrease"])
        self.assertFalse(grouped["resolved_decrease"])

    def test_ungrouped_binomial_interval_transforms_signed_parity_bounds(self):
        _, basis = hubbard(8)
        run = measure_hubbard([1, 0, 0, 0], basis, 8, 128, grouping="ungrouped")
        # All six Z parities are deterministic in a number basis state.
        # Four singles cancel in D; both pair parities are +1 for this state.
        error = 1 - (0.05/12)**(1/128)
        np.testing.assert_allclose(run["observables"]["double_occupancy"]["binomial_ci"],
                                   [1 - 2*error, 1 + error], atol=1e-12)

    def test_weighted_shot_bound_matches_independent_range_calculation(self):
        # Independently derive the ranges, including negative U and U=0.
        _, basis = hubbard(0)
        for strategy in ("grouped", "ungrouped"):
            for u in (-2, 0, 8):
                run = measure_hubbard([1, 0, 0, 0], basis, u, 64,
                                      confidence=0.9, grouping=strategy)
                proxies = ({"double_occupancy": 1/64, "kinetic_energy": 4/64,
                            "energy": (u*u+4)/64} if strategy == "grouped" else
                           {"double_occupancy": 1.5/64, "kinetic_energy": 4/64,
                            "energy": (1.5*u*u+4)/64})
                for name, proxy in proxies.items():
                    obs = run["observables"][name]
                    self.assertAlmostEqual(obs["squared_range_over_shots"], proxy)
                    radius = np.sqrt(np.log(20)*proxy/2)
                    np.testing.assert_allclose(obs["ci"], [obs["estimate"]-radius,
                                                           obs["estimate"]+radius])
                    self.assertGreaterEqual(obs["ci"][0], obs["union_bound_ci"][0]-1e-14)
                    self.assertLessEqual(obs["ci"][1], obs["union_bound_ci"][1]+1e-14)

    def test_preselected_contrast_uses_both_independent_measurement_budgets(self):
        study = finite_shot_study()
        self.assertEqual(len(study["contrasts"]), 6)
        for contrast in study["contrasts"]:
            n = contrast["grouped_shot_budget"]
            strategy = contrast["grouping"]
            rows = study["runs" if strategy == "grouped" else "ungrouped_runs"]
            pair = [next(r for r in rows if r["interaction"] == u
                         and r["total_shots"] == 5*n) for u in (6, 8)]
            self.assertNotEqual(pair[0]["seed"], pair[1]["seed"])
            a, b = [r["observables"]["double_occupancy"] for r in pair]
            self.assertEqual(contrast["total_shots"], 10*n)
            self.assertAlmostEqual(contrast["estimate"], a["estimate"]-b["estimate"])
            # Six independent parity terms at n/2 give three times the
            # range-squared proxy of one density measurement at n.
            radius = np.sqrt(np.log(40)*(1 if strategy == "grouped" else 3)/n)
            self.assertAlmostEqual(contrast["ci"][1]-contrast["estimate"], radius)
            exact = (8/np.sqrt(80)-6/np.sqrt(52))/2
            self.assertAlmostEqual(contrast["exact_ground_state"], exact)
            self.assertEqual(contrast["resolved_decrease"], contrast["ci"][0] > 0)

    def test_density_bound_exact_binomial_coverage_at_small_shot_count(self):
        # Sum the exact coverage probability, rather than relying on a
        # favorable seed. The sector density measurement is Bernoulli.
        from math import comb
        n, confidence = 16, 0.95
        _, basis = hubbard(0)
        for d in (0., 0.01, 0.2, 0.5, 0.99, 1.):
            wave = np.array([np.sqrt(d), np.sqrt(1-d), 0, 0])
            run = measure_hubbard(wave, basis, 0, n, confidence=confidence)
            obs = run["observables"]["double_occupancy"]
            radius = (obs["ci"][1]-obs["ci"][0])/2
            coverage = sum(comb(n, k)*d**k*(1-d)**(n-k) for k in range(n+1)
                           if abs(k/n-d) <= radius)
            self.assertGreaterEqual(coverage, confidence)


if __name__ == "__main__":
    unittest.main()
