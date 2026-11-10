"""Synthetic-only closed-evidence tests; never read the active timing attempt."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


retain = module("closed_provisional_retention", HERE / "retain.py")
fixture_module = module("provisional_fixture_tests", ROOT / "tests/unit_tests/loop/test_tirust_benchmark_provisional.py")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(retain.encoded(value))


class RetentionTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the existing fixture constructor; no controller/native test.
        self.fixture = fixture_module.PortableReplayTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.source = self.fixture.report
        self.destination = self.source.parent / (self.source.name + "-sealed")
        self.addCleanup(lambda: shutil.rmtree(self.destination, ignore_errors=True))
        p = fixture_module.provisional
        self.config = {"relative_tolerance": 3e-6, "madloop_parameters": {}, "candidate_parameters": {},
                       "cases": [{"id": case, "role": "main"} for case in p.CASE_ORDER]}
        self.schedule = p.schedule(self.config, {case: "1" for case in p.CASE_ORDER})
        self.row = self.schedule[0]
        self.run = self.source / "comparisons" / self.row["id"]
        self.fixture.run.rename(self.run)
        self.case = self.row["case"]
        (self.source / "points/case.dat").rename(self.source / ("points/" + self.case + ".dat"))
        (self.source / "points/case.json").rename(self.source / ("points/" + self.case + ".json"))
        native_files = {name: "native-" + name for name in ("lib/libmadloop_tirust.so", "lib/libtirust.a",
                        "lib/tirust-config.json", "lib/tirust-link-before.txt", "lib/tirust-native-libs.txt")}
        self.manifest = {"candidate": {"exporter_identity": {"files_sha256": "frozen-source"},
                                      "native": {"prefix": "/fixture", "files": native_files}},
                         "baseline": {"revision": "baseline-revision"}}
        write(self.source / "campaign-manifest.json", self.manifest)
        self.manifest_sha = retain.sha((self.source / "campaign-manifest.json").read_bytes())
        self.binding = {"id": retain.CAMPAIGN, "manifest_sha256": self.manifest_sha}
        write(self.source / "cases.json", self.config)
        self.result = self.fixture.result
        self.result.update(status="complete", comparison=self.row,
                           initialization_and_warmup_seconds={"baseline": 1., "candidate": 1.})
        self.preparation = {"clones": {self.case: {}}, "native_evaluation": False, "compiled_or_relinked": False}
        self.prerequisites = {"ordinary_numerical_qualification_passed": True, "campaign": retain.CAMPAIGN,
                              "manifest_sha256": self.manifest_sha, "runtime": {}}
        for source in ("baseline", "candidate"):
            inputs = retain.parsed((self.run / (source + "-inputs.json")).read_bytes())
            inputs["campaign_binding"] = self.binding
            inputs["settings"]["mlreductionlib"] = "|".join(map(str, self.row["chains"][source]))
            inputs["compilation"].update(campaign_binding=self.binding, numerical_build_from_sources=True, sources={"kernel.f": "source"})
            inputs["generation"].update(campaign_binding=self.binding, case=self.config["cases"][0], default_chain="1",
                source_identity={"files_sha256": "frozen-source", "revision": "baseline-revision"})
            for kind, names in {"libraries": ("lib/libmadloop_tirust.so",), "archives": ("lib/libtirust.a",),
                                "metadata": ("lib/tirust-config.json", "lib/tirust-link-before.txt", "lib/tirust-native-libs.txt")}.items():
                inputs["runtime"][kind] = {"/fixture/" + name: native_files[name] for name in names}
            self.result["inputs_after"][source]["runtime"] = inputs["runtime"]
            for name in ("compilation", "generation"):
                write(self.run / (source + "-" + name + ".json"), inputs[name])
            write(self.run / (source + "-inputs.json"), inputs)
            self.preparation["clones"][self.case][source] = {name + "_sha256": retain.sha((self.run / (source + "-" + name + ".json")).read_bytes()) for name in ("compilation", "generation")}
            self.preparation["clones"][self.case][source].update(runtime=inputs["runtime"], generated_sources=inputs["compilation"]["sources"], executable_sha256="executable")
            self.prerequisites["runtime"][source + "/" + self.case] = inputs["runtime"]
        self.metadata_root = Path(tempfile.mkdtemp(prefix="synthetic-baseline-metadata-"))
        self.addCleanup(lambda: shutil.rmtree(self.metadata_root, ignore_errors=True))
        for case in self.config["cases"]:
            directory = self.metadata_root / case["id"]
            generation = (retain.parsed((self.run / "baseline-generation.json").read_bytes()) if case["id"] == self.case else
                          {"source": "baseline", "case": case, "default_chain": "1"})
            write(directory / "generation.json", generation)
            clone = self.preparation["clones"].setdefault(case["id"], {}).setdefault("baseline", {})
            clone.update(private_output=str(directory), generation_sha256=retain.sha((directory / "generation.json").read_bytes()))
        write(self.source / "preparation.json", self.preparation)
        write(self.source / "prerequisites.json", self.prerequisites)
        self.result["host_load"] = []
        phases = [(source, phase) for source in ("baseline", "candidate") for phase in ("calibrate-0", "calibrate-1")]
        phases += [(source, "repetition-" + str(number)) for number in range(3)
                   for source in (("baseline", "candidate") if number % 2 == 0 else ("candidate", "baseline"))]
        for source, phase in phases:
            value = {"source": source, "phase": phase, "before": {}, "after": {}}
            write(self.run / "host-snapshots" / (source + "-" + phase + ".json"), value)
            self.result["host_load"].append(value)
        protocols = self.source / "protocol-sources"
        protocols.mkdir()
        for name in retain.PROTOCOLS:
            original = (ROOT / "tools" / name if name != "process_cleanup.py" else
                        ROOT / "docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-qualification-2/process_cleanup.py")
            shutil.copyfile(original, protocols / name)
        frozen = {str(path.relative_to(self.source)): retain.sha(path.read_bytes())
                  for path in self.source.rglob("*") if path.is_file() and path.parts[-2] == "points"}
        frozen.update({name: retain.sha((self.source / name).read_bytes()) for name in
                       ("campaign-manifest.json", "cases.json", "preparation.json", "prerequisites.json")})
        self.specification = {"schema": retain.RUNTIME_SCHEMA, "campaign_binding": self.binding,
            "performance_acceptance_eligible": False, "pairs": 3, "points": 100, "point_subset": None,
            "cpu_reservation": False, "total_cap_seconds": 1800, "target_batch_seconds": 1,
            "cleanup_reserve_seconds": 10, "schedule": self.schedule, "frozen_files": frozen,
            "protocols": {name: retain.sha((protocols / name).read_bytes()) for name in retain.PROTOCOLS}}
        write(self.source / "specification.json", self.specification)
        self.rows = [{"comparison": row, "status": "not_run_budget", "performance_acceptance_eligible": False} for row in self.schedule]
        self.rows[0].update(status="complete", returncode=0, elapsed_seconds_including_cleanup=1.2)
        self.closure = {"schema": retain.RUNTIME_SCHEMA, "status": "closed_budget", "rows": self.rows,
                        "performance_acceptance_eligible": False, "precise_acceptance": "deferred_to_local",
                        "budget_seconds": 1800, "elapsed_seconds": 1800.01}
        write(self.source / "run-start.json", {"performance_acceptance_eligible": False})
        self.finalize()

    def finalize(self, closure="summary.json"):
        write(self.run / "completion.json", self.rows[0])
        self.result["evidence_files"] = {str(path.relative_to(self.run)): retain.sha(path.read_bytes())
            for path in self.run.rglob("*") if path.is_file() and path.name not in ("result.json", "completion.json")}
        write(self.run / "result.json", self.result)
        write(self.source / closure, self.closure)
        self.confirmation = self.source.parent / (self.source.name + "-terminal.json")
        self.addCleanup(lambda: self.confirmation.unlink(missing_ok=True))
        write(self.confirmation, {"terminal_session_id": 82678, "terminal": True, "seal_authorized": True,
              "baseline_generation_metadata_authorized": True,
              "source_directory": str(self.source), "closure_sha256": retain.sha((self.source / closure).read_bytes())})

    def seal(self):
        return retain.seal(self.source, self.confirmation, self.destination, expected_manifest_sha256=self.manifest_sha)

    def test_seal_relocates_and_replays_without_native_calls(self):
        with patch("subprocess.Popen", side_effect=AssertionError("native execution forbidden")), patch.object(fixture_module.bench, "campaign_guard", side_effect=AssertionError("live guard forbidden")):
            result = self.seal()
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            copy = Path(shutil.copytree(self.destination, Path(temporary.name) / "portable"))
            replay = retain.replay_completed(copy, result["sha256"])
            self.assertEqual(replay, {self.row["id"]: self.result})
            self.assertEqual(retain.read_retained_json("cases.json", copy), self.config)
        with self.assertRaisesRegex(ValueError, "Append-once"):
            self.seal()

    def test_no_source_read_before_terminal_authorization(self):
        value = retain.parsed(self.confirmation.read_bytes())
        value["terminal"] = False
        write(self.confirmation, value)
        with patch.object(retain, "snapshot", side_effect=AssertionError("source must remain unread")):
            with self.assertRaisesRegex(ValueError, "terminal/seal authorization"):
                self.seal()

    def test_result_present_but_unavailable_is_retained_without_ratio(self):
        self.rows[0]["status"] = "unavailable_deadline"
        self.finalize()
        self.seal()
        self.assertEqual(retain.replay_completed(self.destination), {})
        index = retain.parsed((self.destination / "index.json").read_bytes())
        self.assertEqual(len(index["comparisons"]), 66)
        self.assertEqual(retain.read_retained_json(index["comparisons"][self.row["id"]]["result"], self.destination), self.result)

    def test_administrative_closure_preserves_partial_summary(self):
        (self.source / "summary.json").write_text('{"partial":')
        self.finalize("administrative-closure.json")
        self.seal()
        index, retained = retain._load(self.destination)
        self.assertEqual(index["paths"]["closure"], "administrative-closure.json")
        self.assertEqual(retained["summary.json"], b'{"partial":')

    def test_missing_or_duplicated_schedule_rejected(self):
        self.closure["rows"] = self.rows[:-1]
        self.finalize()
        with self.assertRaisesRegex(ValueError, "66-row"):
            self.seal()
        self.assertEqual(retain.parsed((self.destination / "seal-failure.json").read_bytes())["status"], "failed")
        shutil.rmtree(self.destination)  # synthetic fixture only; production never retries
        self.closure["rows"] = [*self.rows[:-1], self.rows[0]]
        self.finalize()
        with self.assertRaisesRegex(ValueError, "66-row"):
            self.seal()

    def test_forged_initialization_rejected(self):
        self.result["initialization_and_warmup_seconds"]["baseline"] = 2.
        self.finalize()
        with self.assertRaisesRegex(ValueError, "Initialization/READY"):
            self.seal()

    def test_forged_host_snapshot_rejected(self):
        self.result["host_load"] = []
        self.finalize()
        with self.assertRaisesRegex(ValueError, "host snapshots"):
            self.seal()

    def test_partial_completion_preserved_but_never_accepted(self):
        self.rows[0] = {"comparison": self.row, "status": "not_completed_budget", "performance_acceptance_eligible": False}
        self.closure["rows"] = self.rows
        self.finalize()
        (self.run / "completion.json").write_text('{"partial":')
        self.seal()
        index, raw = retain._load(self.destination)
        binding = index["comparisons"][self.row["id"]]
        self.assertIsNone(binding["completion"])
        self.assertIn(binding["completion_retained"], raw)
        self.assertEqual(retain.replay_completed(self.destination), {})

    def test_payload_corruption_rejected(self):
        self.seal()
        index = retain.parsed((self.destination / "index.json").read_bytes())
        payload = self.destination / index["payload"]
        payload.write_bytes(payload.read_bytes() + b"corruption")
        with self.assertRaisesRegex(ValueError, "outer inventory"):
            retain.replay_completed(self.destination)

    def test_symlink_and_unlisted_native_file_rejected(self):
        alias = self.source / "points/alias.dat"
        alias.symlink_to(self.source / "cases.json")
        with self.assertRaisesRegex(ValueError, "Symlink"):
            self.seal()
        shutil.rmtree(self.destination)  # independent synthetic rejection fixture
        alias.unlink()
        (self.run / "library.so").write_bytes(b"\x7fELF")
        with self.assertRaisesRegex(ValueError, "allowlist"):
            self.seal()

    def test_missing_baseline_metadata_is_retained_sealing_failure(self):
        (self.metadata_root / self.case / "generation.json").unlink()
        with self.assertRaisesRegex(ValueError, "missing or symlinked"):
            self.seal()
        self.assertTrue((self.destination / "seal-failure.json").is_file())
        with self.assertRaisesRegex(ValueError, "Append-once"):
            self.seal()

    def test_complete_over_cap_is_rejected(self):
        self.rows[0]["elapsed_seconds_including_cleanup"] = 121.
        self.finalize()
        with self.assertRaisesRegex(ValueError, "all-in cap"):
            self.seal()


class NativeRuntimeBindingTests(unittest.TestCase):
    def setUp(self):
        self.fixture = retain.parsed((HERE / "closed-runtime-shape.json").read_bytes())
        self.runtime = copy.deepcopy(self.fixture["runtime"])
        self.native = self.fixture["native"]

    def test_closed_mixed_loader_closure_and_source_only_pins(self):
        loaders = [(name, checksum) for name, checksum in self.runtime["libraries"].items()
                   if name.endswith("/ld-linux-x86-64.so.2")]
        self.assertEqual(len(loaders), 2)
        self.assertNotEqual(loaders[0][1], loaders[1][1])
        for name in ("Cargo.lock", "Cargo.toml", "include/mtirust.f90", "include/tirust.h"):
            self.assertIn(name, self.native["files"])
            self.assertTrue(all(self.native["prefix"] + "/" + name not in files for files in self.runtime.values()))
        retain.verify_native_runtime(self.runtime, self.native)

    def test_each_required_runtime_artifact_cannot_be_omitted(self):
        for kind, name in (("libraries", "lib/libmadloop_tirust.so"), ("archives", "lib/libtirust.a"),
                           ("metadata", "lib/tirust-config.json"), ("metadata", "lib/tirust-link-before.txt"),
                           ("metadata", "lib/tirust-native-libs.txt")):
            with self.subTest(name=name):
                runtime = copy.deepcopy(self.runtime)
                del runtime[kind][self.native["prefix"] + "/" + name]
                with self.assertRaisesRegex(ValueError, "Required recorded TIRust"):
                    retain.verify_native_runtime(runtime, self.native)

    def test_identical_basename_at_other_prefix_cannot_substitute(self):
        path = self.native["prefix"] + "/lib/libmadloop_tirust.so"
        self.runtime["libraries"]["/other/libmadloop_tirust.so"] = self.runtime["libraries"].pop(path)
        with self.assertRaisesRegex(ValueError, "Required recorded TIRust"):
            retain.verify_native_runtime(self.runtime, self.native)

    def test_changed_hash_under_recorded_prefix_is_rejected(self):
        self.runtime["metadata"][self.native["prefix"] + "/lib/tirust-config.json"] = "changed"
        with self.assertRaisesRegex(ValueError, "artifact hash differs"):
            retain.verify_native_runtime(self.runtime, self.native)


if __name__ == "__main__":
    unittest.main()
