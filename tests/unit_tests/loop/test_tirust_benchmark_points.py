"""Physical and reproducibility contracts for saved benchmark momenta."""

import copy
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
SPEC = importlib.util.spec_from_file_location(
    "tirust_benchmark_points", ROOT / "tools/tirust_benchmark_points.py")
points = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(points)


class BenchmarkPhaseSpace(unittest.TestCase):
    def test_matrix_of_massive_and_massless_multiplicities_is_physical(self):
        for masses in ([91.188], [80.419], [173, 173], [0, 0],
                       [91.188, 0, 0], [173, 173, 0],
                       [91.188, 91.188, 0, 0], [0, 0, 0, 0], [125, 125]):
            with self.subTest(masses=masses):
                sample = points.generate_points(masses)
                self.assertEqual(len(sample), 100)
                self.assertTrue(points.validate_points(sample, masses, tolerance=1e-12))
                self.assertEqual(sample, json.loads(json.dumps(sample, allow_nan=False)))
                self.assertEqual(len({tuple(point["momenta"][2]) for point in sample}), 100)

    def test_local_random_sequence_is_reproducible_and_does_not_change_global_rng(self):
        before = random.getstate()
        first = points.generate_points([91.188, 0, 0], seed=123)
        self.assertEqual(random.getstate(), before)
        self.assertEqual(first, points.generate_points([91.188, 0, 0], seed=123))
        self.assertNotEqual(first, points.generate_points([91.188, 0, 0], seed=124))

    def test_one_body_is_on_shell_and_has_varied_longitudinal_boosts(self):
        sample = points.generate_points([91.188], count=12, energy=1000)
        for point in sample:
            self.assertEqual(point["sqrt_s"], 91.188)
            self.assertAlmostEqual(point["momenta"][2][0], 91.188 * math.cosh(point["rapidity"]))
            self.assertAlmostEqual(point["momenta"][2][3], 91.188 * math.sinh(point["rapidity"]))
        self.assertEqual(sample[0]["rapidity"], 0)
        self.assertGreater(len({point["rapidity"] for point in sample}), 1)

    def test_two_body_energies_match_exact_unequal_mass_solution(self):
        m1, m2, energy = 91.188, 173.0, 1000.0
        expected_energy = (energy * energy + m1 * m1 - m2 * m2) / (2 * energy)
        expected_momentum = math.sqrt(expected_energy * expected_energy - m1 * m1)
        for point in points.generate_points([m1, m2], count=12):
            first, second = point["momenta"][2:]
            self.assertAlmostEqual(first[0], expected_energy, places=10)
            self.assertAlmostEqual(second[0], energy - expected_energy, places=10)
            self.assertAlmostEqual(math.sqrt(sum(component * component for component in first[1:])),
                                   expected_momentum, places=10)
            for axis in range(1, 4):
                self.assertAlmostEqual(first[axis], -second[axis], places=10)

    def test_massive_incoming_beams_match_exact_unequal_mass_solution(self):
        masses = [91.188, 0, 0]
        incoming = [40, 25]
        sample = points.generate_points(masses, count=3, incoming_masses=incoming)
        self.assertTrue(points.validate_points(sample, masses, tolerance=1e-12,
                                              incoming_masses=incoming))
        expected_energy = (1000 ** 2 + 40 ** 2 - 25 ** 2) / 2000
        expected_momentum = math.sqrt(expected_energy ** 2 - 40 ** 2)
        for point in sample:
            self.assertEqual(point["incoming_masses"], incoming)
            first, second = point["momenta"][:2]
            self.assertAlmostEqual(first[0], expected_energy, places=10)
            self.assertAlmostEqual(second[0], 1000 - expected_energy, places=10)
            self.assertAlmostEqual(first[3], expected_momentum, places=10)
            self.assertEqual(first[3], -second[3])
        with self.assertRaises(ValueError):
            points.validate_points(sample, incoming_masses=[0, 0])

    def test_boosted_massive_beams_remain_physical_even_when_both_travel_forward(self):
        sample = points.generate_points([91.188], count=100, incoming_masses=[40, 40])
        self.assertTrue(points.validate_points(sample, tolerance=1e-12))
        self.assertTrue(any(point["momenta"][0][3] > 0 and point["momenta"][1][3] > 0
                            for point in sample))
        for point in sample:
            rapidity = point["rapidity"]
            incoming_total_energy = sum(vector[0] for vector in point["momenta"][:2])
            incoming_total_z = sum(vector[3] for vector in point["momenta"][:2])
            self.assertAlmostEqual(incoming_total_energy, 91.188 * math.cosh(rapidity), places=10)
            self.assertAlmostEqual(incoming_total_z, 91.188 * math.sinh(rapidity), places=10)

    def test_incoming_threshold_is_respected_by_ordinary_and_stress_samples(self):
        for generate in (points.generate_points, points.generate_stress_points):
            with self.assertRaises(ValueError):
                generate([0, 0], energy=100, incoming_masses=[60, 60])
            with self.assertRaises(ValueError):
                generate([91.188], incoming_masses=[60, 60])
            with self.assertRaises(ValueError):
                generate([0, 0], incoming_masses=[1])
        stress = points.generate_stress_points([0, 0, 0], incoming_masses=[40, 25])
        threshold = next(point for point in stress if point["kind"] == "threshold")
        self.assertAlmostEqual(threshold["sqrt_s"], 65 * (1 + 1e-7))
        self.assertTrue(points.validate_points(stress, tolerance=1e-12, incoming_masses=[40, 25]))

    def test_stress_points_reach_named_regions_without_unphysical_momenta(self):
        sample = points.generate_stress_points([173, 173, 0])
        self.assertTrue(points.validate_points(sample, tolerance=1e-12))
        by_kind = {point["kind"]: point for point in sample}
        self.assertEqual(set(by_kind), {"threshold", "soft-collinear", "small-gram"})
        self.assertLess(by_kind["threshold"]["sqrt_s"] - 346, 0.0001)
        soft = by_kind["soft-collinear"]
        emitted = soft["momenta"][soft["soft_leg"]]
        self.assertLess(emitted[0] / soft["sqrt_s"], 1e-6)
        self.assertLess(abs(emitted[1] / emitted[0]), 2e-6)
        planar = by_kind["small-gram"]["momenta"][2:]
        self.assertLess(max(abs(vector[2]) for vector in planar), 1e-3)
        self.assertEqual(sample, points.generate_stress_points([173, 173, 0]))

    def test_stress_regions_are_only_included_when_applicable(self):
        self.assertEqual(points.generate_stress_points([91.188]), [])
        self.assertEqual([point["kind"] for point in points.generate_stress_points([0, 0])], ["small-gram"])
        self.assertEqual([point["kind"] for point in points.generate_stress_points([173, 173])],
                         ["threshold", "small-gram"])
        self.assertEqual([point["kind"] for point in points.generate_stress_points([0, 0, 0, 0])],
                         ["soft-collinear", "small-gram"])

    def test_threshold_mass_solve_remains_physical_with_small_available_energy(self):
        sample = points.generate_points([173, 173, 0], count=4, energy=346.0000001)
        self.assertTrue(points.validate_points(sample, tolerance=1e-12))
        self.assertTrue(points.validate_points(points.generate_stress_points([173, 173, 0], energy=346.0000001)))

    def test_validation_rejects_nonphysical_or_mislabeled_points(self):
        original = points.generate_points([173, 173], count=1)
        changes = (
            lambda p: p[0]["momenta"][2].__setitem__(0, -1),
            lambda p: p[0]["momenta"][2].__setitem__(1, float("nan")),
            lambda p: p[0]["momenta"][2].__setitem__(0, p[0]["momenta"][2][0] + 1),
            lambda p: p[0].__setitem__("sqrt_s", 900),
            lambda p: p[0].__setitem__("masses", [0, 0]),
            lambda p: p.append(copy.deepcopy(p[0])),
        )
        for mutate in changes:
            sample = copy.deepcopy(original)
            mutate(sample)
            with self.assertRaises(ValueError):
                points.validate_points(sample, masses=[173, 173])

    def test_invalid_generation_parameters_are_rejected(self):
        for masses, kwargs in (([], {}), ([0], {}), ([-1, 0], {}), ([float("nan"), 0], {}),
                               ([173, 173], {"energy": 346}), ([0, 0], {"energy": float("inf")}),
                               ([0, 0], {"count": 0}), ([0, 0], {"count": 1.5}), ([0, 0], {"count": True})):
            with self.subTest(masses=masses, kwargs=kwargs), self.assertRaises(ValueError):
                points.generate_points(masses, **kwargs)

    def test_plain_fortran_output_round_trips_every_component(self):
        sample = points.generate_points([91.188, 0, 0], count=3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "points.dat"
            points.write_points(path, sample)
            lines = path.read_text().splitlines()
        self.assertEqual(lines[0], "3")
        self.assertEqual(len(lines), 1 + 3 * 5)
        actual = [[float(value) for value in line.split()] for line in lines[1:]]
        expected = [vector for point in sample for vector in point["momenta"]]
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
