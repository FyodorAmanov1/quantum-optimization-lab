import json
import unittest

import numpy as np

from src.hubbard_measurement import (
    SETTINGS, finite_shot_study, measure_hubbard, measurement_probabilities, pauli_matrix,
)
from src.studio import hubbard, vqe, vqe_state


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


if __name__ == "__main__":
    unittest.main()
