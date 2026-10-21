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

    def test_driver_adapter_cannot_hide_numerical_source_changes(self):
        frozen = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
        worker = frozen.replace("@PREFIX@", "ML5_0_").replace("@NEXTERNAL@", "6").replace("@NBORN@", "1")
        driver = "SubProcesses/P0_stress/check_sa.f"
        code = "SubProcesses/P0_stress/loop_matrix.f"
        origin = {"files": {driver: {"sha256": "original-driver"}, code: {"sha256": "numerics"}}}
        checksum = stress.hashlib.sha256(worker.encode()).hexdigest()
        compiled = {"files": {driver: {"sha256": checksum}, code: {"sha256": "numerics"}}}
        stress.verify_worker_source(worker, frozen, origin, compiled, checksum)
        altered = copy.deepcopy(compiled)
        altered["files"][code]["sha256"] = "changed-numerics"
        with self.assertRaises(ValueError):
            stress.verify_worker_source(worker, frozen, origin, altered, checksum)
        changed_worker = worker.replace("CALL UPDATE_AS_PARAM()", "CALL OMITTED_UPDATE()", 1)
        with self.assertRaises(ValueError):
            stress.verify_worker_source(changed_worker, frozen, origin, compiled, checksum)
        scale_checksum = stress.hashlib.sha256(stress.scale_worker_source(worker).encode()).hexdigest()
        scale_compiled = copy.deepcopy(compiled)
        scale_compiled["files"][driver]["sha256"] = scale_checksum
        stress.verify_worker_source(worker, frozen, origin, scale_compiled, scale_checksum, scale=True)

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

    def test_optional_failure_labels_only_actual_attempt_not_unattempted_case(self):
        ids = [c["id"] for c in stress.configuration("ordinary")["cases"]]
        failed = {"variant": "ordinary", "rejected": [], "status": {
            "status": "failed", "requested_cases": ids, "completed_cases": []}}
        self.assertEqual(stress.failed_case_ids(failed), {ids[0]})
        failed["status"]["completed_cases"] = [ids[0]]
        self.assertEqual(stress.failed_case_ids(failed), {ids[1]})
        failed["rejected"] = [{"case": ids[0]+"-rotation"}]
        failed["variant"] = "rotation"
        self.assertEqual(stress.failed_case_ids(failed), {ids[0]})

    @classmethod
    def campaign_records(cls, variant):
        records = {}
        for case in stress.configuration(variant)["cases"]:
            cid = case["id"]
            for backend in stress.BACKENDS:
                rows = cls.rows(stress.VARIANTS[variant], backend=backend)
                if variant == "scale":
                    rows = []
                    ordinary = cls.rows(backend=backend)
                    for point, factor in enumerate(stress.scale_schedule(), 1):
                        physical = 1 if point <= 4 else 2
                        for source in ordinary:
                            if source["point"] == physical:
                                rows.append({**source, "point": point,
                                    "coefficients": [x*factor for x in source["coefficients"]]})
                count = stress.VARIANTS[variant]
                for repetition in (0, 1):
                    row = {"case": cid, "source": "candidate", "backend": backend,
                           "repetition": repetition, "rows": copy.deepcopy(rows), "cpu": 5,
                           "seconds_per_point": .001, "cold_seconds": 1., "loops": 1,
                           "points_sha256": cid, "parameter_card_sha256": "physical-card",
                           "shared_numerical_settings": {"mlstabthres": 1e-9},
                           "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
                           "status_counts": {str(rows[0]["status"]): count},
                           "timed_status_counts": {str(rows[0]["status"]): count},
                           "executable_sha256": f"executable-{backend}-{variant}"}
                    records[cid, "candidate", backend, repetition] = (Path("unused"), row)
        return records

    def snapshot(self, groups, failed=None):
        signature = self.signature()
        expected = self.expected(signature)
        directory = tempfile.TemporaryDirectory(dir=ROOT / ".benchmark-work")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        preparation = path / "preparation.json"
        stress.bench.dump(preparation, {})
        failures = []
        if failed:
            status = path / "failed-status.json"
            stress.bench.dump(status, failed["status"])
            failures = [{"variant": "ordinary", "optional_helper": failed["optional_helper"],
                         "run_status": stress.supplement.bound(status), "rejected": []}]
        dataset = stress.policy.seal({"schema": stress.SCHEMA, "expected": expected,
            "runtime_signature": signature, "performance_acceptance_gate": False,
            "preparation": stress.supplement.bound(preparation),
            "groups": [{"variant": v} for v in groups], "failed_groups": failures})
        filename = path / "dataset.json"
        stress.bench.dump(filename, dataset)
        def validate(entry, *args, **kwargs):
            return {}, {}, groups[entry["variant"]]
        with mock.patch.object(stress, "validate_preparation", return_value={}), \
                mock.patch.object(stress, "validate_group", side_effect=validate):
            return stress.report_snapshot(filename, expected, runtime_signature=signature)

    def test_complete_raw_candidate_cohorts_and_controls_are_required(self):
        groups = {v: self.campaign_records(v) for v in stress.VARIANTS}
        result = self.snapshot(groups)
        self.assertTrue(result["complete_campaign_passed"])
        self.assertEqual(len(result["cases"]), 2)
        self.assertTrue(all(c["ordinary"]["agreement"]["pairs"] == 6 for c in result["cases"]))
        self.assertTrue(all(c["optional_reference"]["status"] == "pending" for c in result["cases"]))
        incomplete = self.snapshot({"ordinary": groups["ordinary"], "rotation": groups["rotation"]})
        self.assertFalse(incomplete["complete_campaign_passed"])
        self.assertEqual(incomplete["cases"][0]["scale"]["agreement"]["status"], "pending")
        missing_backend = copy.deepcopy(groups)
        for key in [k for k in missing_backend["ordinary"] if k[2] == 8]:
            del missing_backend["ordinary"][key]
        self.assertFalse(self.snapshot(missing_backend)["complete_campaign_passed"])

    def test_raw_mismatch_and_restore_failure_do_not_hide_behind_summary(self):
        groups = {v: self.campaign_records(v) for v in stress.VARIANTS}
        for key, (_, row) in groups["scale"].items():
            if key[2] == 8:
                row["rows"][-1]["coefficients"] = [999., 999., 0., 0.]
        result = self.snapshot(groups)
        self.assertFalse(result["complete_campaign_passed"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["cases"][0]["scale"]["restoration"]["status"], "failed")

    def test_optional_helper_failure_is_visible_without_failing_candidate_gate(self):
        groups = {v: self.campaign_records(v) for v in stress.VARIANTS}
        ids = [c["id"] for c in stress.configuration("ordinary")["cases"]]
        failure = {"optional_helper": True, "status": {"status": "failed", "requested_cases": ids,
                   "completed_cases": [], "error": "Retained optional reference failure"}}
        result = self.snapshot(groups, failure)
        self.assertTrue(result["complete_campaign_passed"])
        self.assertEqual(result["cases"][0]["optional_reference"]["status"], "failed")
        self.assertEqual(len(result["cases"][0]["optional_reference"]["failures"]), 1)
        self.assertEqual(result["cases"][1]["optional_reference"]["status"], "pending")
        failure["optional_helper"] = False
        self.assertFalse(self.snapshot(groups, failure)["complete_campaign_passed"])

    def test_duplicate_repetitions_or_different_cards_are_rejected(self):
        groups = {v: self.campaign_records(v) for v in stress.VARIANTS}
        first = next(iter(groups["ordinary"]))
        groups["ordinary"][first][1]["repetition"] = 1
        with self.assertRaises(ValueError):
            self.snapshot(groups)
        groups = {v: self.campaign_records(v) for v in stress.VARIANTS}
        for _, row in groups["rotation"].values():
            row["parameter_card_sha256"] = "different-rotation-card"
        with self.assertRaises(ValueError):
            self.snapshot(groups)
        groups = {v: self.campaign_records(v) for v in stress.VARIANTS}
        groups["ordinary"][next(iter(groups["ordinary"]))][1]["parameter_card_sha256"] = "changed-card"
        with self.assertRaises(ValueError):
            self.snapshot(groups)

    def test_group_reader_rejects_missing_process_or_backend_before_accepting_rows(self):
        signature = self.signature()
        expected = self.expected(signature)
        config = stress.configuration("ordinary")
        targets = {f"{c['id']}-candidate-{b}": {"case": c["id"], "backend": b}
                   for c in config["cases"] for b in stress.BACKENDS}
        targets.pop(next(iter(targets)))
        binding = stress.policy.seal({"schema": "supplement-source-bindings-v1", "report": ".benchmark-work",
            "role_map": {"candidate": "candidate_corrected"}, "targets": targets})
        with tempfile.TemporaryDirectory(dir=ROOT / ".benchmark-work") as directory:
            filename = Path(directory) / "binding.json"
            stress.bench.dump(filename, binding)
            protocol = {"configuration": config, "performance_acceptance_gate": False,
                        "role_map": binding["role_map"], "repetitions": 2}
            with mock.patch.object(stress.workers, "checked_series", return_value=(protocol, {})):
                with self.assertRaises(ValueError):
                    stress.validate_group({"variant": "ordinary", "bindings": stress.supplement.bound(filename)},
                                          expected, signature, {})


if __name__ == "__main__":
    unittest.main()
