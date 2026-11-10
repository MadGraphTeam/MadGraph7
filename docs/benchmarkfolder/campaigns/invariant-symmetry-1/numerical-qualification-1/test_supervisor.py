"""Synthetic source-only checks: never launch native, make, ldd, or compilers."""
from contextlib import ExitStack, redirect_stdout
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("invariant_numerical_supervisor", HERE / "run.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


class Fixture(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.report = self.root / "report"
        self.report.mkdir()
        self.home = self.root / "supervisor"
        self.home.mkdir()
        self.patch(run, "ROOT", self.root)
        self.patch(run, "HERE", self.home)
        self.patch(run, "REPORT", self.report)
        self.patch(run, "MANIFEST", self.report / "manifest.json")
        self.patch(run.bench, "LEGACY_REPORT", self.root)
        self.config = {"cases": [{"id": value} for value in run.CASE_ORDER],
                       "points_per_case": 100, "relative_tolerance": 3e-6,
                       "madloop_parameters": {"CTModeRun": -1, "CTModeInit": 1},
                       "candidate_parameters": {}}
        self.origins = {"cards": {name: {"param_card_sha256": run.hashlib.sha256(b"physical-card\n").hexdigest()}
                                  for name in run.CASE_ORDER}}
        files = {"lib/libmadloop_tirust.so": "dso", "lib/libtirust.a": "archive",
                 "lib/tirust-config.json": "config", "lib/tirust-link-before.txt": "before",
                 "lib/tirust-native-libs.txt": "native", "Cargo.lock": "source-lock",
                 "include/tirust.h": "source-header"}
        self.native = {"prefix": str(self.root / "native"), "files": files}
        prefix = Path(self.native["prefix"])
        self.runtime = {"libraries": {str(prefix / "lib/libmadloop_tirust.so"): "dso",
                            "/nix/store/one/lib/ld-linux-x86-64.so.2": "loader-one",
                            "/nix/store/two/lib/ld-linux-x86-64.so.2": "loader-two"},
                        "archives": {str(prefix / "lib/libtirust.a"): "archive"},
                        "metadata": {str(prefix / name): value for name, value in files.items()
                                     if name.endswith((".json", ".txt"))}}
        self.dependencies = {key: self.runtime[key] for key in ("archives", "metadata")}
        self.manifest = {"candidate": {"native": self.native, "exporter_identity": {"files_sha256": "source-v1"}},
                         "shared_inputs": {"files": {}}, "predecessor": {"report": "previous"}}
        self.baseline_record = {"revision": run.bench.BASELINE_REVISION,
                                "files": {"baseline.f": "baseline-source"}, "archive_sha256": "archive-source"}
        run.dump(self.report / "inputs/baseline.json", self.baseline_record)
        self.manifest["baseline"] = {"revision": run.bench.BASELINE_REVISION,
                                      "record_sha256": run.digest(self.report / "inputs/baseline.json")}
        for name in run.CASE_ORDER:
            points = self.report / "points" / (name + ".dat")
            points.parent.mkdir(exist_ok=True)
            points.write_text("synthetic points\n")
            run.dump(points.with_suffix(".json"), {"sha256": run.digest(points),
                "inputs": {"incoming_masses": [0., 0.], "masses": [1.], "count": 100}})
            for path in (points, points.with_suffix(".json")):
                self.manifest["shared_inputs"]["files"]["points/" + path.name] = run.digest(path)
        run.dump(run.MANIFEST, self.manifest)
        self.patch(run, "EXPECTED_MANIFEST", run.digest(run.MANIFEST))
        self.binding = {"id": run.CAMPAIGN, "manifest_sha256": run.EXPECTED_MANIFEST}
        self.sources = {"Source/MODEL/model.f": "model-source"}
        self.objects = {"Source/MODEL/model.o": "model-object", "SubProcesses/P0/check_sa.o": "driver-object"}
        for case in self.config["cases"]:
            for source in ("baseline", "candidate"):
                output = self.output(source, case)
                directory = self.directory(source, case)
                (output / "Cards").mkdir(parents=True)
                directory.mkdir(parents=True)
                (output / "Cards/param_card.dat").write_text("physical-card\n")
                (directory / "check").write_text("synthetic executable bytes; never executed\n")
                generation = {"case": case, "source": source, "campaign_binding": self.binding,
                              "parameter_card_sha256": self.origins["cards"][case["id"]]["param_card_sha256"],
                              "source_identity": {"files_sha256": "source-v1"}, "command_sha256": "cli-command"}
                compiled = {"campaign_binding": self.binding, "sources": self.sources,
                    "executable_sha256": run.digest(directory / "check"), "masses": [0., 0., 1.],
                    "numerical_build_from_sources": True,
                    "full_numerical_rebuild_this_invocation": source == "candidate",
                    "link_inputs_before": self.dependencies, "link_inputs_after": self.dependencies,
                    "effective_compiler_flags": {"policy": "same"}}
                evidence = self.report / "inputs" / source / case["id"]
                evidence.mkdir(parents=True)
                if source == "baseline":
                    generation["source_identity"] = {"revision": run.bench.BASELINE_REVISION,
                        "files_sha256": run.bench.content_digest(self.baseline_record["files"]),
                        "archive_sha256": self.baseline_record["archive_sha256"]}
                    previous = self.root / "previous/inputs/baseline" / case["id"]
                    run.dump(previous / "generation.json", generation)
                    run.dump(previous / "compilation.json", compiled)
                    (evidence / "tirust_benchmark_relink.py").write_text("synthetic relink proof protocol\n")
                    proof = {"passed": True, "objects_unchanged": True, "original_executable_unchanged": True,
                        "expected_executable_sha256": compiled["executable_sha256"],
                        "relinked_executable_sha256": compiled["executable_sha256"],
                        "protocol_sha256": run.digest(evidence / "tirust_benchmark_relink.py")}
                    run.dump(evidence / "object-relink-proof.json", proof)
                    generation["reused_from"] = {"worker_only_relink_required": True,
                        "generation_sha256": run.digest(previous / "generation.json"),
                        "compilation_sha256": run.digest(previous / "compilation.json"),
                        "object_relink_proof": proof, "numerical_objects": self.objects}
                    compiled["previous_compilation_sha256"] = run.bench.content_digest(run.read(previous / "compilation.json"))
                run.dump(output / "generation.json", generation)
                run.dump(output / "compilation.json", compiled)
                run.dump(evidence / "generation.json", generation)
                run.dump(evidence / "compilation.json", compiled)
                shutil.copy2(output / "Cards/param_card.dat", evidence / "param_card.dat")
        (self.root / "tools").mkdir()
        for name in run.PROTOCOLS:
            (self.root / "tools" / name).write_text("synthetic frozen protocol " + name + "\n")
        run.dump(self.report / "cases.json", self.config)
        self.patch(run.bench, "output_dir", side_effect=self.output)
        self.patch(run.bench, "subprocess_dir", side_effect=self.directory)
        self.patch(run.bench, "generated_sources", return_value=self.sources)
        self.patch(run.bench, "numerical_objects", return_value=self.objects)
        self.patch(run.bench, "verify_worker_template")
        self.patch(run.bench, "resolved_parameter_card", side_effect=lambda directory: directory.parents[1] / "Cards/param_card.dat")
        self.patch(run.bench, "dependency_inputs", return_value=self.dependencies)
        self.patch(run.bench, "runtime_identity", return_value={"ldd": "variable addresses", **self.runtime})
        self.flags = self.patch(run, "capture_effective_flags", return_value={"policy": "same"})
        self.patch(run.bench, "same_compiler_policy", side_effect=lambda first, second: first == second)
        self.pairs = self.patch(run, "validate_compiler_equality")
        self.patch(run.campaign, "validate_generation_environment")
        self.origin_check = self.patch(run.campaign, "verify_physical_card_origins", return_value=self.origins)
        self.patch(run.bench, "case_worker_timeout", return_value=3600.)

    def patch(self, target, name, *args, **kwargs):
        return self.stack.enter_context(patch.object(target, name, *args, **kwargs))

    def output(self, source, case):
        return self.root / "outputs" / source / case["id"]

    def directory(self, source, case):
        return self.output(source, case) / "SubProcesses/P0"

    def change_json(self, path, **changes):
        value = run.read(path)
        value.update(changes)
        run.dump(path, value)

    def make_result(self, *, qp=False):
        self.records = run.verify_builds(self.config)
        self.item = run.commands(9)[0]
        case = self.config["cases"][0]
        report = self.report / "experiments/numerical" / self.item["label"]
        report.mkdir(parents=True)
        shutil.copy2(run.MANIFEST, report / "campaign-manifest.json")
        shutil.copy2(self.report / "cases.json", report / "cases.json")
        for name in run.PROTOCOLS:
            shutil.copy2(self.root / "tools" / name, report / name)
        run.dump(report / "specification.json", {"campaign_binding": self.binding, "case": case,
            "performance_eligible": False, "measurement_phase": "steady-warmup",
            "worker_timeout_seconds": 3600., "configuration_sha256": run.digest(report / "cases.json"),
            "protocols": {name: run.digest(report / name) for name in run.PROTOCOLS}})
        results, reference = [], None
        for source, backend in run.EXPECTED_LANES:
            directory = report / f"{source}-{backend}"
            raw = directory / "raw"
            raw.mkdir(parents=True)
            code = 323 if qp and backend == 9 else 210 + backend
            # Wrapped call window proves the parser binds call-to-point rather
            # than accepting sorted point IDs alone; every split column retained.
            rows = [f"{call} {1 + (call - 1) % 100} {order} {code} 1 2 3 4 1e-10\n"
                    for call in range(124, 224) for order in (0, 1)]
            (raw / "warmup.dat").write_text("".join(rows))
            (raw / "dimensions.dat").write_text("1 1 1\n")
            ready = {"points": 100, "ndim": 1, "nso": 1, "nborn": 1, "cold_seconds": 2.}
            run.dump(raw / "ready.json", ready)
            observation = run.parse_warmup(raw, [backend], 100, ready)
            if source == "baseline":
                reference = observation
            value = {"source": source, "backend": backend, "status": "complete",
                     "performance_eligible": False, "observation": observation,
                     "comparison": run.bench.compare_rows(reference["rows"], observation["rows"], 3e-6)}
            initial = directory / (source + "-initialization")
            initial.mkdir()
            for name in ("warmup.dat", "dimensions.dat"):
                shutil.copy2(raw / name, initial / name)
            run.dump(raw / "observation.json", observation)
            run.dump(directory / "result.json", value)
            record = self.records[source + "/" + case["id"]]
            inputs = {"campaign_binding": self.binding,
                "generation": run.read(self.output(source, case) / "generation.json"),
                "compilation": run.read(self.output(source, case) / "compilation.json"),
                "runtime": self.runtime, "numerical_cards": {"Cards/MadLoopParams.dat": "policy"},
                "settings": {"mlreductionlib": str(backend), "ctmoderun": -1, "ctmodeinit": 1},
                **{key: record[key] for key in ("param_card_sha256", "resolved_parameter_card", "points_sha256", "points_metadata_sha256")}}
            after = {"runtime": self.runtime, "executable_sha256": record["executable_sha256"],
                     "points_sha256": record["points_sha256"], "numerical_cards": inputs["numerical_cards"]}
            run.dump(directory / (source + "-inputs.json"), inputs)
            run.dump(directory / "inputs-after.json", after)
            results.append(value)
        run.dump(report / "results.json", {"case": case, "performance_eligible": False, "results": results})
        return report


class AdmissionTests(Fixture):
    def test_all_22_builds_and_duplicate_loader_basenames(self):
        records = run.verify_builds(self.config)
        self.assertEqual(len(records), 22)
        self.assertEqual(self.flags.call_count, 22)
        self.assertEqual(self.pairs.call_count, 11)
        self.assertNotIn("Cargo.lock", records["candidate/" + run.CASE_ORDER[0]]["runtime"]["metadata"])

    def test_shadowed_physical_card_rejected(self):
        self.patch(run.bench, "resolved_parameter_card", return_value=self.report / "cases.json")
        with self.assertRaisesRegex(ValueError, "shadowed"):
            run.verify_builds(self.config)

    def test_explicit_ancestry_required(self):
        self.origin_check.return_value = None
        with self.assertRaisesRegex(ValueError, "explicitly declared"):
            run.verify_builds(self.config)

    def test_compiler_change_rejected(self):
        self.flags.return_value = {"policy": "changed"}
        with self.assertRaisesRegex(ValueError, "compiler policy"):
            run.verify_builds(self.config)

    def test_link_inputs_change_rejected(self):
        path = self.output("baseline", self.config["cases"][0]) / "compilation.json"
        self.change_json(path, link_inputs_before={})
        shutil.copy2(path, self.report / "inputs/baseline" / run.CASE_ORDER[0] / "compilation.json")
        with self.assertRaisesRegex(ValueError, "link-time"):
            run.verify_builds(self.config)

    def test_candidate_source_identity_rejected(self):
        path = self.output("candidate", self.config["cases"][0]) / "generation.json"
        self.change_json(path, source_identity={"files_sha256": "other"})
        with self.assertRaisesRegex(ValueError, "fresh numerical source"):
            run.verify_builds(self.config)

    def test_wrong_baseline_identity_rejected_even_with_matching_retained_copy(self):
        path = self.output("baseline", self.config["cases"][0]) / "generation.json"
        value = run.read(path)
        value["source_identity"]["revision"] = "wrong-baseline"
        run.dump(path, value)
        shutil.copy2(path, self.report / "inputs/baseline" / run.CASE_ORDER[0] / "generation.json")
        with self.assertRaisesRegex(ValueError, "frozen starting revision"):
            run.verify_builds(self.config)

    def test_relink_proof_failure_rejected(self):
        path = self.report / "inputs/baseline" / run.CASE_ORDER[0] / "object-relink-proof.json"
        self.change_json(path, passed=False)
        with self.assertRaisesRegex(ValueError, "object-relink proof"):
            run.verify_builds(self.config)

    def test_wrong_runtime_prefix_cannot_be_replaced_by_matching_basename(self):
        runtime = copy.deepcopy(self.runtime)
        runtime["libraries"].pop(str(Path(self.native["prefix"]) / "lib/libmadloop_tirust.so"))
        runtime["libraries"]["/other/libmadloop_tirust.so"] = "dso"
        with self.assertRaisesRegex(ValueError, "registered prefix"):
            run.verify_native_runtime(runtime, self.manifest)

    def test_default_is_dry_run_without_campaign_activation_or_worker(self):
        self.patch(run, "verify_seal")
        self.patch(run.bench, "ROOT", self.root)
        self.patch(run.os, "sched_getaffinity", return_value={9})
        activate = self.patch(run.campaign, "activate_campaign", side_effect=AssertionError("activation forbidden"))
        launch = self.patch(run.subprocess, "Popen", side_effect=AssertionError("launch forbidden"))
        self.patch(sys, "argv", ["run.py"])
        stream = io.StringIO()
        with redirect_stdout(stream):
            run.main()
        value = json.loads(stream.getvalue())
        self.assertEqual(value["status"], "dry-run-no-execution")
        self.assertEqual([item["case"] for item in value["commands"]], run.CASE_ORDER)
        self.assertTrue(all("--warmup-only" in item["command"] for item in value["commands"]))
        self.assertFalse(value["performance_acceptance_eligible"])
        activate.assert_not_called()
        launch.assert_not_called()

    def test_execute_requires_passed_preparation_before_activation(self):
        self.patch(run, "verify_seal")
        self.patch(run.bench, "ROOT", self.root)
        self.patch(run.os, "sched_getaffinity", return_value={9})
        self.patch(run, "verify_preparation", side_effect=ValueError("preparation incomplete"))
        activate = self.patch(run.campaign, "activate_campaign", side_effect=AssertionError("activation forbidden"))
        self.patch(sys, "argv", ["run.py", "--execute"])
        with self.assertRaisesRegex(ValueError, "preparation incomplete"):
            run.main()
        activate.assert_not_called()


class ReplayTests(Fixture):
    def test_complete_wrapped_window_and_tirust_qp_alias(self):
        self.make_result(qp=True)
        value = run.observed_result(self.item, 0, self.config, self.records)
        self.assertEqual(value["status"], "passed")
        self.assertTrue(value["raw_replay_passed"])
        self.assertEqual(value["lanes"][-1]["accepted_backend_precision_counts"], {"9:QP": 100})
        self.assertTrue(all(row["fpe_flags"] is None and row["internal_retry_counts"] is None for row in value["lanes"]))

    def test_raw_row_tampering_rejected(self):
        report = self.make_result()
        path = report / "candidate-9/raw/warmup.dat"
        path.write_text(path.read_text().replace(" 1 2 3 4 ", " 1 8 3 4 ", 1))
        with self.assertRaisesRegex(ValueError, "exact final-warmup replay"):
            run.observed_result(self.item, 0, self.config, self.records)

    def test_postcheck_runtime_tampering_rejected(self):
        report = self.make_result()
        path = report / "candidate-9/inputs-after.json"
        value = run.read(path)
        value["runtime"]["libraries"]["unexpected"] = "new"
        run.dump(path, value)
        with self.assertRaisesRegex(ValueError, "build/runtime"):
            run.observed_result(self.item, 0, self.config, self.records)

    def test_retained_failed_lane_never_promoted(self):
        report = self.make_result()
        path = report / "results.json"
        value = run.read(path)
        failed = {"source": "candidate", "backend": 9, "status": "failed",
                  "performance_eligible": False, "error": "retained synthetic failure"}
        value["results"][-1] = failed
        run.dump(path, value)
        run.dump(report / "candidate-9/failure.json", failed)
        result = run.observed_result(self.item, 1, self.config, self.records)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["lanes"][-1]["matched_saved_points"], 0)

    def test_partial_controller_results_remain_failed(self):
        report = self.make_result()
        path = report / "results.json"
        value = run.read(path)
        value["results"] = value["results"][:2]
        run.dump(path, value)
        self.assertEqual(run.observed_result(self.item, 1, self.config, self.records)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
