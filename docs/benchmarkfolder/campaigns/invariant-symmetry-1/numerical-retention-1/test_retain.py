"""Synthetic portable checkpoint tests. No native programs or live case reads."""
from contextlib import ExitStack, redirect_stdout
import copy
import gzip
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


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RET = module(HERE / "retain.py", "numerical_retention_source_test")
FIX = module(HERE / "synthetic_fixture.py", "numerical_retention_fixture_test")


class CheckpointFixture:
    def __init__(self, parent, offset=0.):
        self.root = parent / "repository"
        self.docs = self.root / RET.DOCS
        self.fixture = FIX.Fixture(self.docs)
        self.case = next(case for case in self.fixture.config["cases"] if case["id"] == RET.CASE_ORDER[0])
        self.campaign = self.fixture.campaign
        self.supervisor = self.campaign / "numerical-qualification-1"
        self.package = parent / "proposal"
        self.package.mkdir()
        for name in RET.PACKAGE_SOURCES:
            if name != "source-index.json":
                shutil.copyfile(HERE / name, self.package / name)
        self._ancestry()
        self.manifest_sha = FIX.dump(self.campaign / "manifest.json", self.fixture.manifest)
        self.fixture.binding = {"id": "final", "manifest_sha256": self.manifest_sha}
        self.fixture.complete_case(self.case, offset=offset)
        self._supervisor()
        source = (self.package / "retain.py").read_text()
        source = source.replace('CAMPAIGN = "invariant-symmetry-1"', 'CAMPAIGN = "final"')
        source = source.replace(RET.MANIFEST_SHA, self.manifest_sha).replace(RET.SUPERVISOR_INDEX_SHA, self.supervisor_index_sha)
        (self.package / "retain.py").write_text(source)
        self.ret = module(self.package / "retain.py", "synthetic_numerical_retention")
        self.confirmation = {"schema": RET.CONFIRMATION_SCHEMA, "campaign_binding": self.fixture.binding,
            "supervisor_session_id": RET.SESSION, "supervisor_source_index_sha256": self.supervisor_index_sha,
            "supervisor_specification_sha256": FIX.sha((self.supervisor / "specification.json").read_bytes()),
            "closed_cases": [self._closed_entry()], "terminal_confirmation": None}
        self.confirmation_file = parent / "closed-confirmation.json"
        FIX.dump(self.confirmation_file, self.confirmation)
        self._seal_proposal()
        (self.root / "tools").mkdir()
        (self.root / "tools/tirust_benchmark.py").write_text("synthetic repository marker; never imported\n")

    def _ancestry(self):
        fixture = self.fixture
        previous = {"schema": "madloop-tirust-campaign-v1", "id": "previous",
                    "shared_inputs": fixture.manifest["shared_inputs"]}
        previous_sha = FIX.dump(self.docs / "campaigns/previous/manifest.json", previous)
        origin = {"id": "previous", "report": "campaigns/previous", "manifest_sha256": previous_sha}
        fixture.manifest["predecessor"] = origin
        origins = {"scope": "physical-input-ancestry-only", "origin": origin, "cards": {}, "observations": {}}
        for case in fixture.config["cases"]:
            card = {"param_card_sha256": FIX.sha(b"physical card\n"), "files": {}}
            for source, backend in (("baseline", 1), ("candidate", 9)):
                name = f"campaigns/previous/inputs/{source}/{case['id']}/param_card.dat"
                card["files"][name] = FIX.write(self.docs / name, "physical card\n")
                binding = {"id": origin["id"], "manifest_sha256": origin["manifest_sha256"]}
                observed = {"campaign_binding": binding, "param_card_sha256": card["param_card_sha256"],
                    "generation": {"case": case, "source": source, "campaign_binding": binding,
                                   "parameter_card_sha256": card["param_card_sha256"]},
                    "points_sha256": fixture.manifest["shared_inputs"]["files"][f"points/{case['id']}.dat"],
                    "points_metadata_sha256": fixture.manifest["shared_inputs"]["files"][f"points/{case['id']}.json"]}
                name = f"campaigns/previous/experiments/numerical/full100-warmup-{case['id']}-1/{source}-{backend}/{source}-inputs.json"
                origins["observations"][name] = FIX.dump(self.docs / name, observed)
            origins["cards"][case["id"]] = card
        fixture.manifest["physical_card_origins"] = origins

    def _supervisor(self):
        supervisor_raw = (HERE / "numerical_supervisor.py").read_bytes()
        sources = {"run.py": supervisor_raw, "process_cleanup.py": b"# synthetic source-only cleanup placeholder\n",
                   "README.md": b"Synthetic supervisor metadata only\n", "test_supervisor.py": b"# synthetic\n"}
        for name, raw in sources.items():
            FIX.write(self.supervisor / name, raw)
        protocols = {name: FIX.sha((HERE / ("protocol-" + name)).read_bytes()) for name in RET.PROTOCOLS}
        seal = {"files": {name: FIX.sha(raw) for name, raw in sources.items()},
                "inputs": {"tools/" + name: checksum for name, checksum in protocols.items()}, "validation_files": {}}
        self.supervisor_index_sha = FIX.dump(self.supervisor / "source-index.json", seal)
        shutil.copyfile(self.supervisor / "source-index.json", self.package / "numerical-supervisor-source-index.json")
        FIX.dump(self.supervisor / "configuration-initial.json", {"synthetic": True})
        prep = self.campaign / "preparation-1"
        prep_source = b"# synthetic preparation metadata; no native invocation\n"
        FIX.write(prep / "run.py", prep_source)
        prepared = {"source-index.json": FIX.dump(prep / "source-index.json", {"files": {"run.py": FIX.sha(prep_source)}}),
                    "specification.json": FIX.dump(prep / "specification.json", {"synthetic": True}),
                    "status.json": FIX.dump(prep / "status.json", {"status": "passed", "amplitude_evaluations": 0})}
        records = {}
        folder = self.fixture.folder(self.case)
        for source, backend in FIX.COLLECTOR.LANES:
            lane = folder / f"{source}-{backend}"
            value = json.loads((lane / (source + "-inputs.json")).read_text())
            value["resolved_parameter_card"] = f"/nonexistent-recorded-native/{source}/param_card.dat"
            FIX.dump(lane / (source + "-inputs.json"), value)
            for name in ("warmup.dat", "dimensions.dat"):
                FIX.write(lane / (source + "-initialization") / name, (lane / "raw" / name).read_bytes())
            retained = self.campaign / "inputs" / source / self.case["id"]
            records[source + "/" + self.case["id"]] = {
                "generation_sha256": FIX.sha((retained / "generation.json").read_bytes()),
                "compilation_sha256": FIX.sha((retained / "compilation.json").read_bytes()),
                "generation_content_sha256": FIX.COLLECTOR._canonical(value["generation"]),
                "compilation_content_sha256": FIX.COLLECTOR._canonical(value["compilation"]),
                "executable_sha256": value["compilation"]["executable_sha256"], "runtime": value["runtime"],
                **{key: value[key] for key in ("param_card_sha256", "resolved_parameter_card", "points_sha256", "points_metadata_sha256")}}
        schedule = []
        for case_id in RET.CASE_ORDER:
            label = "full100-warmup-" + case_id + "-1"
            schedule.append({"case": case_id, "label": label, "command": ["/synthetic/python", "-B",
                "tools/tirust_benchmark_numerical.py", "--campaign", "final", "--case", case_id,
                "--label", label, "--warmup-only", "--backends", "1", "6", "7", "8", "9", "--cpu", "9"]})
        self.specification = {"manifest_sha256": self.manifest_sha, "campaign": "final", "cpu": 9,
            "supervisor_sha256": FIX.sha(supervisor_raw),
            "source_index_sha256": self.supervisor_index_sha, "protocols": protocols,
            "schedule": schedule, "preparation": prepared, "compiled_inputs": records,
            "worker_timeouts": self.fixture.manifest["administrative_worker_timeout_seconds"]}
        FIX.dump(self.supervisor / "specification.json", self.specification)
        FIX.write(self.supervisor / (self.case["id"] + ".log"), "synthetic closed controller output\n")

    def _closed_entry(self):
        folder = self.fixture.folder(self.case)
        results = json.loads((folder / "results.json").read_text())
        lanes = []
        for value in results["results"]:
            observation, comparison = value.get("observation"), value.get("comparison")
            lanes.append({"source": value["source"], "backend": value["backend"], "status": value["status"],
                "error": value.get("error"), "numerical_agreement_passed": comparison.get("passed") if comparison else None,
                "ordinary_target_passed": comparison.get("ordinary_target_passed") if comparison else None,
                "status_counts": observation.get("status_counts") if observation else None,
                "accepted_backend_precision_counts": observation.get("backend_precision_counts") if observation else None,
                "matched_saved_points": 100 if observation else 0, "fpe_flags": None, "internal_retry_counts": None,
                "performance_acceptance_eligible": False})
        passed = len(lanes) == 6 and all(row["status"] == "complete" and row["numerical_agreement_passed"] for row in lanes)
        return {"case": self.case["id"], "status": "passed" if passed else "failed", "returncode": 0 if passed else 1,
            "results_path": (folder / "results.json").relative_to(self.campaign).as_posix(),
            "results_sha256": FIX.sha((folder / "results.json").read_bytes()), "lanes": lanes,
            "raw_replay_passed": True, "finished_utc": "2026-11-10T00:00:00+00:00",
            "console_sha256": FIX.sha((self.supervisor / (self.case["id"] + ".log")).read_bytes())}

    def _seal_proposal(self):
        FIX.dump(self.package / "source-index.json", {"files": {name: RET.identity((self.package / name).read_bytes())
                 for name in RET.PACKAGE_SOURCES if name != "source-index.json"}})

    def seal(self, destination):
        return self.ret.seal(self.root, self.confirmation_file, destination, self.package)

    def refresh_confirmation(self):
        self.confirmation["closed_cases"] = [self._closed_entry()]
        FIX.dump(self.confirmation_file, self.confirmation)


class PortableTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))

    def test_default_does_not_read_execution_evidence(self):
        with patch.object(sys, "argv", ["retain.py"]), patch.object(RET.Store, "add", side_effect=AssertionError("read forbidden")):
            stream = io.StringIO()
            with redirect_stdout(stream):
                RET.main()
        self.assertEqual(json.loads(stream.getvalue())["status"], "source-only-no-ingestion")

    def test_relocated_package_replays_after_source_repository_removed(self):
        fixture = CheckpointFixture(self.temp)
        destination = self.temp / "sealed"
        index = fixture.seal(destination)
        self.assertEqual(index["case_ids"], [RET.CASE_ORDER[0]])
        self.assertNotIn("docs/benchmarkfolder/campaigns/final/numerical-qualification-1/status.json", index["retained_files"])
        relocated = self.temp / "relocated"
        shutil.copytree(destination, relocated)
        shutil.rmtree(fixture.root)
        replay = module(relocated / "retain.py", "relocated_checkpoint")
        result = replay.replay_completed(relocated, RET.sha((relocated / "index.json").read_bytes()))
        self.assertEqual(result[RET.CASE_ORDER[0]]["status"], "passed")
        self.assertEqual(len(result[RET.CASE_ORDER[0]]["lanes"]), 6)

    def test_unselected_active_tree_is_never_opened(self):
        fixture = CheckpointFixture(self.temp)
        active = fixture.campaign / ("experiments/numerical/full100-warmup-" + RET.CASE_ORDER[1] + "-1")
        active.mkdir(parents=True)
        (active / "forbidden.bin").write_bytes(b"\x7fELF\x00active")
        (fixture.supervisor / "status.json").write_bytes(b"\x00mutable status must never be read")
        index = fixture.seal(self.temp / "sealed")
        self.assertFalse(any("full100-warmup-" + RET.CASE_ORDER[1] + "-1" in name for name in index["retained_files"]))

    def test_content_addressed_payload_is_deterministic(self):
        fixture = CheckpointFixture(self.temp)
        first, second = fixture.seal(self.temp / "one"), fixture.seal(self.temp / "two")
        self.assertEqual(first["payload"], second["payload"])
        data = (self.temp / "one" / first["payload"]).read_bytes()
        self.assertEqual(data[4:8], b"\x00\x00\x00\x00")

    def test_regression_failure_retained_without_promotion(self):
        fixture = CheckpointFixture(self.temp, offset=1e-3)
        destination = self.temp / "failed"
        fixture.seal(destination)
        summary = json.loads((destination / "summary.json").read_text())
        self.assertEqual(summary["selected_status"], "failed")
        self.assertFalse(summary["full_matrix_regression_passed"])
        self.assertFalse(summary["numerical"]["cases"][0]["regression_pass"])

    def test_partial_failed_controller_rows_remain_partial(self):
        fixture = CheckpointFixture(self.temp)
        folder = fixture.fixture.folder(fixture.case)
        value = json.loads((folder / "results.json").read_text())
        value["results"] = value["results"][:2]
        FIX.dump(folder / "results.json", value)
        for source, backend in FIX.COLLECTOR.LANES[2:]:
            shutil.rmtree(folder / f"{source}-{backend}")
        fixture.refresh_confirmation()
        destination = self.temp / "partial"
        fixture.seal(destination)
        result = fixture.ret.replay_completed(destination)
        self.assertEqual(result[fixture.case["id"]]["status"], "failed")
        self.assertEqual(len(result[fixture.case["id"]]["lanes"]), 2)

    def test_failed_lane_marker_and_raw_are_both_retained(self):
        fixture = CheckpointFixture(self.temp)
        folder = fixture.fixture.folder(fixture.case)
        value = json.loads((folder / "results.json").read_text())
        failed = {"source": "candidate", "backend": 9, "status": "failed", "performance_eligible": False,
                  "error": "synthetic postverification failure with complete raw rows"}
        value["results"][-1] = failed
        FIX.dump(folder / "results.json", value)
        FIX.dump(folder / "candidate-9/failure.json", failed)
        fixture.refresh_confirmation()
        destination = self.temp / "failure"
        index = fixture.seal(destination)
        self.assertTrue(any(name.endswith("candidate-9/raw/warmup.dat") for name in index["retained_files"]))
        self.assertTrue(any(name.endswith("candidate-9/failure.json") for name in index["retained_files"]))
        self.assertEqual(fixture.ret.replay_completed(destination)[fixture.case["id"]]["status"], "failed")

    def test_failed_baseline_eagerly_retains_inherited_generation_and_cli(self):
        fixture = CheckpointFixture(self.temp)
        case_id = fixture.case["id"]
        folder = fixture.fixture.folder(fixture.case)
        retained = fixture.campaign / "inputs/baseline" / case_id
        generation = json.loads((retained / "generation.json").read_text())
        compiled = json.loads((retained / "compilation.json").read_text())
        old = fixture.docs / "campaigns/previous/inputs/baseline" / case_id
        old_generation = copy.deepcopy(generation)
        old_generation["campaign_binding"] = {"id": "previous", "manifest_sha256": fixture.fixture.manifest["predecessor"]["manifest_sha256"]}
        generation_sha = FIX.dump(old / "generation.json", old_generation)
        compiled_sha = FIX.dump(old / "compilation.json", compiled)
        cli = fixture.docs / "campaigns/previous/commands" / ("baseline-" + case_id + ".mg5")
        FIX.write(cli, (fixture.campaign / "commands" / cli.name).read_bytes())
        protocol = b"# synthetic retained relink protocol\n"
        proof = {"passed": True, "objects_unchanged": True, "original_executable_unchanged": True,
                 "expected_executable_sha256": compiled["executable_sha256"],
                 "relinked_executable_sha256": compiled["executable_sha256"], "protocol_sha256": FIX.sha(protocol)}
        generation["reused_from"] = {"generation_sha256": generation_sha, "compilation_sha256": compiled_sha,
            "object_relink_proof": proof, "worker_only_relink_required": True,
            "source_generation_repeated": False, "source_export_environment_recorded": True,
            "source_export_environment": generation["export_environment"],
            "reuse_environment_binding": {"export_environment_binding_version": 1, "export_environment": {"PYTHONHASHSEED": "0"}}}
        FIX.dump(retained / "generation.json", generation)
        FIX.dump(retained / "object-relink-proof.json", proof)
        FIX.write(retained / "tirust_benchmark_relink.py", protocol)
        binding = fixture.specification["compiled_inputs"]["baseline/" + case_id]
        binding["generation_sha256"] = FIX.sha((retained / "generation.json").read_bytes())
        binding["generation_content_sha256"] = FIX.COLLECTOR._canonical(generation)
        FIX.dump(fixture.supervisor / "specification.json", fixture.specification)
        results = json.loads((folder / "results.json").read_text())
        failure = {"source": "baseline", "backend": 1, "status": "failed", "performance_eligible": False,
                   "error": "synthetic baseline startup failure"}
        results["results"][0] = failure
        FIX.dump(folder / "baseline-1/failure.json", failure)
        for result in results["results"][1:]:
            result["comparison"] = None
            FIX.dump(folder / f"candidate-{result['backend']}/result.json", result)
        FIX.dump(folder / "results.json", results)
        fixture.refresh_confirmation()
        fixture.confirmation["supervisor_specification_sha256"] = FIX.sha((fixture.supervisor / "specification.json").read_bytes())
        FIX.dump(fixture.confirmation_file, fixture.confirmation)
        destination = self.temp / "baseline-failed"
        index = fixture.seal(destination)
        for path in (old / "generation.json", old / "compilation.json", cli):
            self.assertIn(path.relative_to(fixture.root).as_posix(), index["retained_files"])
        result = fixture.ret.replay_completed(destination)[case_id]
        self.assertEqual(result["status"], "failed")
        self.assertTrue(all(lane["numerical_agreement_passed"] is None for lane in result["lanes"]))

    def test_running_case_rejected_before_source_ingestion(self):
        fixture = CheckpointFixture(self.temp)
        fixture.confirmation["closed_cases"][0]["status"] = "running"
        FIX.dump(fixture.confirmation_file, fixture.confirmation)
        with patch.object(fixture.ret.Store, "add", side_effect=AssertionError("case read forbidden")):
            with self.assertRaisesRegex(ValueError, "explicitly closed"):
                fixture.seal(self.temp / "refused")
        self.assertFalse((self.temp / "refused").exists())

    def test_failed_terminal_supervisor_cannot_qualify_full_matrix(self):
        confirmation = {"schema": RET.CONFIRMATION_SCHEMA,
            "campaign_binding": {"id": RET.CAMPAIGN, "manifest_sha256": RET.MANIFEST_SHA},
            "supervisor_session_id": RET.SESSION, "supervisor_source_index_sha256": RET.SUPERVISOR_INDEX_SHA,
            "closed_cases": [{"case": name, "status": "passed", "returncode": 0, "finished_utc": "closed",
                              "raw_replay_passed": True} for name in RET.CASE_ORDER],
            "terminal_confirmation": {"session_id": RET.SESSION, "observed_terminal": True, "exit_code": 1}}
        flags = RET.qualification_flags(confirmation)
        self.assertTrue(flags["full_matrix_closed"])
        self.assertFalse(flags["full_matrix_regression_passed"])
        self.assertEqual(flags["supervisor_terminal"]["exit_code"], 1)
        confirmation["terminal_confirmation"]["exit_code"] = 0
        self.assertTrue(RET.qualification_flags(confirmation)["full_matrix_regression_passed"])

    def test_no_overwrite_or_retry(self):
        fixture = CheckpointFixture(self.temp)
        destination = self.temp / "sealed"
        fixture.seal(destination)
        before = (destination / "index.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "no overwrite"):
            fixture.seal(destination)
        self.assertEqual((destination / "index.json").read_bytes(), before)

    def test_raw_mutation_fails_and_keeps_sealing_failure(self):
        fixture = CheckpointFixture(self.temp)
        path = fixture.fixture.folder(fixture.case) / "candidate-9/raw/warmup.dat"
        path.write_text(path.read_text().replace(" 2.0 ", " 8.0 ", 1))
        destination = self.temp / "changed"
        with self.assertRaisesRegex(ValueError, "replay"):
            fixture.seal(destination)
        self.assertTrue((destination / "sealing-failure.json").is_file())

    def test_symlink_and_native_payload_are_rejected(self):
        for mode in ("symlink", "native"):
            with self.subTest(mode=mode):
                parent = self.temp / mode
                parent.mkdir()
                fixture = CheckpointFixture(parent)
                file = fixture.fixture.folder(fixture.case) / "candidate-9/raw/extra.dat"
                if mode == "symlink":
                    file.symlink_to(fixture.confirmation_file)
                else:
                    file.write_bytes(b"\x7fELF\x00native")
                with self.assertRaises(ValueError):
                    fixture.seal(parent / "rejected")

    def test_payload_member_digest_checked_before_accessor_returns(self):
        fixture = CheckpointFixture(self.temp)
        destination = self.temp / "sealed"
        index = fixture.seal(destination)
        name = next(name for name in index["retained_files"] if name.endswith("raw/ready.json"))
        index["retained_files"][name]["bytes"] += 1
        FIX.dump(destination / "index.json", index)
        with self.assertRaisesRegex(ValueError, "member identity"):
            fixture.ret.read_retained_json(name, destination)


if __name__ == "__main__":
    unittest.main()
