"""Fixed stress inputs, generic worker adaptation and source attribution."""
import copy
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_stress_v2 as stress


class TestNinjaRSStress(unittest.TestCase):
    @staticmethod
    def signature(archive="archive"):
        names = stress.policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}
        return stress.policy.seal({"schema": "reducer-runtime-signature-v2",
            "libraries": {"ninjars_lib/libninjars.a": archive},
            "metadata": {name: name for name in names}, "link_inputs": {}})

    @staticmethod
    def expected(signature):
        return {"core_dataset_identity_sha256": "a"*64,
                "literal_baseline_revision": stress.bench.BASELINE_REVISION,
                "candidate_generator_identity_sha256": "b"*64,
                "candidate_runtime_identity_sha256": signature["identity_sha256"]}

    @staticmethod
    def events():
        # Four massless outgoing momenta, exactly conserving energy/momentum;
        # five copies supply enough fixed inputs without a numerical engine.
        event = [[10., 0., 0., 10.], [10., 0., 0., -10.],
                 [5., 3., 4., 0.], [5., -3., -4., 0.],
                 [5., 0., 0., 5.], [5., 0., 0., -5.]]
        return [copy.deepcopy(event) for _ in range(20)]

    @staticmethod
    def rows(count=20, value=1., backend=8):
        status = {1: 211, 6: 216, 7: 217, 8: 218}[backend]
        return [{"point": point, "order": order, "status": status,
                 "coefficients": [value+point, value*point, 0., 0.], "accuracy": 1e-14}
                for point in range(1, count+1) for order in (0, 1)]

    def test_fixed_processes_counts_and_no_new_performance_gate(self):
        ordinary = stress.configuration("ordinary")
        self.assertEqual(ordinary["points_per_case"], 20)
        self.assertEqual(ordinary["seed"], 1940083)
        self.assertEqual([c["process"] for c in ordinary["cases"]],
                         ["g g > t t~ g g [virt=QCD]", "u u~ > e+ e- mu+ mu- [virt=QED]"])
        self.assertEqual([c["nfinal"] for c in ordinary["cases"]], [4, 4])
        self.assertEqual(stress.configuration("rotation")["points_per_case"], 4)
        self.assertEqual(stress.configuration("scale")["points_per_case"], 8)
        self.assertNotIn("max_runtime_ratio", ordinary)
        self.assertFalse(stress.preregistration()["runtime_reporting"]["performance_acceptance_gate"])
        with self.assertRaises(ValueError):
            stress.configuration("unknown")

    def test_final_core_identity_cannot_be_relabelled(self):
        signature = self.signature()
        expected = self.expected(signature)
        stress.check_expected(expected, signature)
        for key, value in (("literal_baseline_revision", "wrong"),
                           ("candidate_runtime_identity_sha256", "wrong")):
            with self.subTest(key=key), self.assertRaises(ValueError):
                stress.check_expected({**expected, key: value}, signature)
        with self.assertRaises(ValueError):
            stress.check_expected({k: v for k, v in expected.items() if k != "core_dataset_identity_sha256"}, signature)
        with self.assertRaises(RuntimeError):
            stress.check_expected(expected, {**signature, "libraries": {"libninjars.a": "stale"}})

    def test_actual_stale_archive_or_module_cannot_borrow_final_identity(self):
        signature = self.signature()
        expected = self.expected(signature)
        with mock.patch.object(stress.supplement, "digest", return_value="stale"):
            with self.assertRaises(ValueError):
                stress.check_installed(expected, signature)

    def test_rational_rotation_preserves_serialized_physical_inputs(self):
        events = self.events()
        before = copy.deepcopy(events)
        result = stress.rotated(events)
        self.assertEqual(events, before)
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0][2][0], events[0][2][0])
        self.assertNotEqual(result[0][2][1:], events[0][2][1:])
        with tempfile.TemporaryDirectory(dir=ROOT / ".benchmark-work") as directory:
            path = Path(directory) / "points.dat"
            path.write_text(stress.serialize(result))
            metadata = stress.supplement.point_metadata(path, [0.]*6, 4)
            self.assertLess(metadata["max_relative_conservation_residual"], 1e-14)
            self.assertLess(metadata["max_relative_mass_shell_residual"], 1e-14)
        with self.assertRaises(ValueError):
            stress.rotated(events[:3])

    def test_scale_points_and_restoration_sequence_are_fixed(self):
        events = self.events()
        values = stress.scale_events(events)
        self.assertEqual(len(values), 8)
        self.assertEqual(stress.scale_schedule(), [1., 2., .5, 1., 1., 2., .5, 1.])
        for index, value in enumerate(values):
            self.assertEqual(value, events[index//4])
        values[0][0][0] = 999.
        self.assertEqual(events[0][0][0], 10.)
        with self.assertRaises(ValueError):
            stress.scale_events(events[:1])

    def test_scale_adapter_changes_only_three_scale_assignments_and_reader(self):
        source = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
        result = stress.scale_worker_source(source)
        original = "MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))"
        indices = ("1", "1+MOD(K-1,NPOINTS)", "K")
        for index in indices:
            self.assertEqual(result.count(original+"*STRESS_SCALE("+index+")"), 1)
        self.assertEqual(result.count("CALL UPDATE_AS_PARAM()"), source.count("CALL UPDATE_AS_PARAM()"))
        self.assertEqual(result.count("SLOOPMATRIX_THRES"), source.count("SLOOPMATRIX_THRES"))
        self.assertNotIn("MDL_MT", result)
        prefix = result.split("C     Stress-driver input schedule only;", 1)[0]
        prefix = prefix.replace("      REAL*8 ZERO,STRESS_SCALE\n", "      REAL*8 ZERO\n")
        for index in indices:
            prefix = prefix.replace(original+"*STRESS_SCALE("+index+")", original)
        self.assertEqual(prefix, source+"\n")

    def test_changed_frozen_worker_is_rejected_before_build(self):
        source = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
        with self.assertRaises(ValueError):
            stress.scale_worker_source(source.replace("MU_R=SQRT", "MU_R=2*SQRT", 1))

    def test_rotation_compares_matching_summed_physical_rows(self):
        ordinary = {b: self.rows(backend=b) for b in stress.BACKENDS}
        rotation = {b: [r.copy() for r in ordinary[b] if r["point"] <= 4] for b in stress.BACKENDS}
        self.assertTrue(stress.invariant_comparisons(ordinary, rotation, "rotation")["passed"])
        rotation[8][0]["coefficients"] = [100., 100., 0., 0.]
        result = stress.invariant_comparisons(ordinary, rotation, "rotation")
        self.assertFalse(result["passed"])
        self.assertGreater(result["max_tolerance_fraction"], 1.)

    def test_scale_restoration_does_not_demand_changed_scale_equality(self):
        ordinary = {b: self.rows(backend=b) for b in stress.BACKENDS}
        scale = {}
        for b in stress.BACKENDS:
            rows = []
            for p in range(1, 9):
                original_point = 1 if p <= 4 else 2
                for source in ordinary[b]:
                    if source["point"] == original_point:
                        row = {**source, "point": p}
                        if p in (2, 3, 6, 7):
                            row["coefficients"] = [1000., 2000., 0., 0.]
                        rows.append(row)
            scale[b] = rows
        self.assertTrue(stress.invariant_comparisons(ordinary, scale, "scale")["passed"])
        scale[8][-1]["coefficients"] = [100., 100., 0., 0.]
        self.assertFalse(stress.invariant_comparisons(ordinary, scale, "scale")["passed"])

    def test_missing_or_duplicate_invariant_rows_are_rejected(self):
        ordinary = {b: self.rows(backend=b) for b in stress.BACKENDS}
        rotation = {b: [r.copy() for r in ordinary[b] if r["point"] <= 4] for b in stress.BACKENDS}
        rotation[8].pop()
        with self.assertRaises(ValueError):
            stress.invariant_comparisons(ordinary, rotation, "rotation")
        with self.assertRaises(ValueError):
            stress.invariant_comparisons(ordinary, rotation, "not-a-control")


if __name__ == "__main__":
    unittest.main()
