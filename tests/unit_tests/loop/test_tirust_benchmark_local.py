"""Local replicas preserve origin authority without relaxing campaign guards."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark_local as local


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n" if not isinstance(value, str) else value)


class TestLocalMeasurement(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.report = self.root / "docs/benchmarkfolder"
        self.work = self.root / ".benchmark-work/tirust"
        self.patchers = []
        for name, value in (("ROOT", self.root), ("LEGACY_REPORT", self.report), ("LEGACY_WORK", self.work)):
            self.mock(local.bench, name, value)
        self.config = {"baseline_revision": local.bench.BASELINE_REVISION, "points_per_case": 100,
                       "repetitions": 7, "relative_tolerance": 3e-6,
                       "madloop_parameters": {"CTModeRun": -1},
                       "candidate_parameters": {"TIRustComputeFinite": True},
                       "cases": [{"id": "example", "role": "main"}]}
        # Use the real tracked launcher bytes, so mocking subprocess.run cannot
        # conceal a nonexistent entrypoint in the generated installation argv.
        launcher = self.root / "bin/madgraph"
        launcher.parent.mkdir(parents=True)
        shutil.copy2(ROOT / "bin/madgraph", launcher)
        self.source = {"files_sha256": "frozen-exporter", "revision": "frozen",
                       "files": {"bin/madgraph": local.bench.digest(launcher)}}
        self.native = {"native": {"prefix": str(self.root / "HEPTools/tirust"),
                       "files": {name: "native-" + name for name in local.campaign.NATIVE_NAMES}},
                       "dependencies": {"HEPTools/ninja/lib/libninja.so": "ninja",
                                        "HEPTools/ninjars/lib/ninjars-config.json": "metadata"},
                       "compilers": {"FC": {"path": "/compiler", "sha256": "compiler"}}, "flags": {}}
        self.host = {"machine_id_sha256": "host-a", "node": "host-a"}
        origin = self.report / "campaigns/canonical"
        write(origin / "cases.json", self.config)
        write(origin / "inputs/baseline.json", {"revision": local.bench.BASELINE_REVISION, "files": {}})
        write(origin / "inputs/loop_qcd_qed_sm_Gmu.json", {"origin": "https://example.test/model.tar.gz", "files": {}})
        write(origin / "points/example.dat", "100\nfrozen physical points\n")
        write(origin / "points/example.json", {"sha256": local.bench.digest(origin / "points/example.dat")})
        shared = {p.relative_to(origin).as_posix(): local.bench.digest(p) for p in origin.rglob("*") if p.is_file()}
        self.manifest = {"id": "canonical", "schema": local.campaign.SCHEMA,
                         "candidate": {"exporter_identity": self.source,
                                       "native": copy.deepcopy(self.native["native"]),
                                       "numerical_sources": {"kernel.f": {"numerical_sha256": "arithmetic"}}},
                         "baseline": {"record_sha256": local.bench.digest(origin / "inputs/baseline.json")},
                         "shared_inputs": {"files": shared, "model_tree": {}, "policy_sha256": "policy"},
                         local.campaign.TIMEOUT_FIELD: {"example": 14400}}
        write(origin / "manifest.json", self.manifest)
        write(self.report / "runs/old/tirust-first/failure-evidence.dat", "retained qualified failure\n")
        inventory = {p.relative_to(self.report).as_posix(): local.bench.digest(p)
                     for p in self.report.rglob("*") if p.is_file()}
        self.origin = {"campaign": "canonical", "manifest": self.manifest,
                       "manifest_sha256": local.bench.digest(origin / "manifest.json"),
                       "predecessor": {}, "qualified_failures": [{"case": "old"}],
                       "physical_cards": {"example": "physical-card"}, "files": inventory}
        for name in local.PROTOCOLS:
            write(self.root / "tools" / name, "# frozen protocol " + name + "\n")
        self.origin_mock = self.mock(local, "origin_inventory", return_value=self.origin)
        self.deployment_mock = self.mock(local, "deployment", side_effect=lambda prefix: copy.deepcopy(self.native))
        self.mock(local, "verify_recipe", return_value={"lock_sha256": "pinned-resolved-lock"})
        self.mock(local, "host_identity", side_effect=lambda: dict(self.host))
        self.mock(local.bench, "source_identity", return_value=self.source)
        self.mock(local.os, "sched_getaffinity", return_value={0, 1})

        def baseline(work, record):
            (work / "baseline").mkdir()
            shutil.copy2(record, work / "baseline.json")
        self.mock(local, "materialize_baseline", side_effect=baseline)

    def mock(self, obj, name, *args, **kwargs):
        patcher = patch.object(obj, name, *args, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def prepare(self, name="local-one", cpu=0):
        return local.prepare(name, "canonical", self.root / "HEPTools/tirust", cpu)

    def test_prepare_isolated_copies_preserve_failures_and_pending_state(self):
        before = {name: local.bench.digest(self.report / name) for name in self.origin["files"]}
        status = self.prepare()
        report, work = local.roots("local-one")
        self.assertEqual(status["historical_qualified_failures"], 1)
        self.assertFalse(status["overall_goal_qualified"])
        self.assertEqual(len(status["pending_prerequisites"]), 4)
        self.assertEqual(before, {name: local.bench.digest(self.report / name) for name in before})
        self.assertEqual((report / "origin/runs/old/tirust-first/failure-evidence.dat").read_text(),
                         "retained qualified failure\n")
        self.assertEqual(local.bench.digest(work / "baseline.json"), self.manifest["baseline"]["record_sha256"])

    def test_label_cpu_and_metadata_changes_cannot_replace_same_computation(self):
        self.prepare()
        self.native["native"]["files"]["lib/tirust-config.json"] = "new metadata"
        self.native["dependencies"]["HEPTools/ninjars/lib/ninjars-config.json"] = "new metadata"
        with self.assertRaisesRegex(ValueError, "already has a local measurement"):
            self.prepare("renamed", cpu=1)

    def test_deleted_measurement_does_not_clear_registry_authority(self):
        self.prepare()
        report, work = local.roots("local-one")
        shutil.rmtree(report)
        shutil.rmtree(work)
        with self.assertRaisesRegex(ValueError, "already has a local measurement"):
            self.prepare("renamed")

    def test_another_host_is_distinct_but_retains_origin_failures(self):
        self.prepare()
        self.host["machine_id_sha256"] = "host-b"
        status = self.prepare("host-two")
        self.assertEqual(status["historical_qualified_failures"], 1)
        self.assertFalse(status["overall_goal_qualified"])

    def test_manifest_edit_is_caught_by_external_registry_seal(self):
        self.prepare()
        report, _ = local.roots("local-one")
        value = local.read(report / "measurement.json")
        value["cpu"] = 1
        write(report / "measurement.json", value)
        with self.assertRaisesRegex(ValueError, "registry seal"):
            local.verify("local-one")

    def test_old_failure_tampering_and_new_origin_failures_are_rejected(self):
        self.prepare()
        report, _ = local.roots("local-one")
        evidence = report / "origin/runs/old/tirust-first/failure-evidence.dat"
        original = evidence.read_bytes()
        evidence.write_text("now passes")
        with self.assertRaisesRegex(ValueError, "historical failure changed"):
            local.verify("local-one")
        evidence.write_bytes(original)
        changed = copy.deepcopy(self.origin)
        changed["files"]["runs/new/failure.json"] = "new-failure"
        self.origin_mock.return_value = changed
        with self.assertRaisesRegex(ValueError, "gained failures"):
            local.verify("local-one")

    def test_changed_native_protocol_and_source_rejected(self):
        self.prepare()
        self.native["native"]["files"]["lib/libmadloop_tirust.so"] = "changed arithmetic"
        with self.assertRaisesRegex(ValueError, "deployment changed"):
            local.verify("local-one")
        self.native["native"]["files"]["lib/libmadloop_tirust.so"] = "native-lib/libmadloop_tirust.so"
        write(self.root / "tools/tirust_benchmark_worker.f", "changed protocol")
        with self.assertRaisesRegex(ValueError, "protocol changed"):
            local.verify("local-one")

    def test_new_native_bytes_require_source_receipt_before_registration(self):
        self.native["native"]["files"]["lib/libmadloop_tirust.so"] = "locally rebuilt"
        with self.assertRaisesRegex(ValueError, "actual-CLI source receipt"):
            self.prepare()

    def test_install_records_real_cli_command_and_verifiable_source_receipt(self):
        def fake_run(command, **kwargs):
            self.assertEqual(command, [sys.executable, "-B", "bin/madgraph",
                             str(self.report / "local-installations/source-install/commands.mg5")])
            self.assertEqual(kwargs["cwd"], self.root)
            launcher = kwargs["cwd"] / command[2]
            self.assertTrue(launcher.is_file())
            self.assertFalse(launcher.is_symlink())
            self.assertEqual(local.bench.digest(launcher), self.source["files"]["bin/madgraph"])
            self.assertIn("install tirust --prefix=", Path(command[3]).read_text())
            kwargs["stdout"].write("TIRust successfully installed in local prefix.\n")
        process = self.mock(local.subprocess, "run", side_effect=fake_run)
        result = local.install("canonical", "source-install", self.root / "HEPTools/tirust")
        process.assert_called_once()
        receipt = Path(result["install_receipt"])
        value = local.read(receipt)
        self.assertEqual(value["command"], process.call_args.args[0])
        self.assertEqual(value["source_before"], self.source)
        self.assertEqual(value["source_after"], self.source)
        self.assertEqual(value["native"], self.native["native"])
        self.assertEqual(value["status"], "complete")
        bound = local.verify_install_receipt(receipt, self.manifest, self.native["native"])
        self.assertEqual(bound["sha256"], local.bench.digest(receipt))
        (receipt.parent / "console.log").write_text("different build log")
        with self.assertRaisesRegex(ValueError, "source receipt"):
            local.verify_install_receipt(receipt, self.manifest, self.native["native"])

    def test_failed_cli_install_retains_failure_without_receipt(self):
        def fail(command, **kwargs):
            self.assertEqual(command[1:3], ["-B", "bin/madgraph"])
            raise subprocess.CalledProcessError(1, command)
        process = self.mock(local.subprocess, "run", side_effect=fail)
        with self.assertRaises(subprocess.CalledProcessError):
            local.install("canonical", "failed-install", self.root / "HEPTools/tirust")
        directory = self.report / "local-installations/failed-install"
        self.assertTrue((directory / "failure.json").is_file())
        self.assertFalse((directory / "receipt.json").exists())
        value = local.read(directory / "failure.json")
        self.assertEqual(value["command"], process.call_args.args[0])
        self.assertEqual(value["status"], "failed")
        self.assertEqual(value["source_before"], self.source)
        self.assertNotIn("native", value)

    def test_missing_cli_retains_exact_intent_and_failure_without_receipt(self):
        self.mock(local.subprocess, "run", side_effect=FileNotFoundError("launcher unavailable"))
        with self.assertRaises(FileNotFoundError):
            local.install("canonical", "missing-cli", self.root / "HEPTools/tirust")
        directory = self.report / "local-installations/missing-cli"
        intent = local.read(directory / "intent.json")
        failure = local.read(directory / "failure.json")
        self.assertEqual(intent["command"], [sys.executable, "-B", "bin/madgraph",
                         str(directory / "commands.mg5")])
        self.assertEqual(failure["command"], intent["command"])
        self.assertEqual(failure["status"], "failed")
        self.assertIn("FileNotFoundError", failure["error"])
        self.assertTrue((directory / "console.log").is_file())
        self.assertFalse((directory / "receipt.json").exists())

    def test_cli_exit_zero_without_success_marker_has_no_receipt(self):
        self.mock(local.subprocess, "run")
        with self.assertRaisesRegex(ValueError, "did not report a successful"):
            local.install("canonical", "no-publication", self.root / "HEPTools/tirust")
        directory = self.report / "local-installations/no-publication"
        self.assertEqual(local.read(directory / "failure.json")["status"], "failed")
        self.assertFalse((directory / "receipt.json").exists())

    def test_source_change_during_cli_install_has_no_receipt(self):
        changed = dict(self.source, files_sha256="changed-exporter")
        self.mock(local.bench, "source_identity", side_effect=[self.source, changed])
        def fake_run(command, **kwargs):
            kwargs["stdout"].write("TIRust successfully installed in local prefix.\n")
        self.mock(local.subprocess, "run", side_effect=fake_run)
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            local.install("canonical", "changed-source", self.root / "HEPTools/tirust")
        directory = self.report / "local-installations/changed-source"
        failure = local.read(directory / "failure.json")
        self.assertEqual(failure["source_after"], changed)
        self.assertEqual(failure["status"], "failed")
        self.assertFalse((directory / "receipt.json").exists())

    def test_isolated_adapter_restores_globals_on_exception_and_binds_cards(self):
        self.prepare()
        originals = {name: getattr(local.bench, name) for name in local.GLOBALS}
        with self.assertRaisesRegex(ValueError, "physical card"):
            with local.isolated("local-one"):
                self.assertEqual(local.bench.case_worker_timeout({"id": "example"}), 14400)
                self.assertIsNone(local.bench.CAMPAIGN_ID)
                self.assertNotEqual(local.bench.REPORT, self.report)
                local.bench.campaign_guard("worker", {"generation": {"case": {"id": "example"},
                    "parameter_card_sha256": "different", "source": "baseline"}})
        for name, value in originals.items():
            self.assertEqual(getattr(local.bench, name), value, name)

    def test_forward_has_no_arbitrary_label_chain_or_case_escape(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            local.forward("local-one", "pair")
        before = list(sys.argv)
        seen = []

        def core_main():
            seen.append((list(sys.argv), local.bench.REPORT))
        self.mock(local.bench, "main", side_effect=core_main)
        local.forward("local-one", "init")
        self.assertEqual(seen[0][0], ["tirust_benchmark.py", "init", "--cpu", "0", "--jobs", "4"])
        self.assertEqual(sys.argv, before)
        report, _ = local.roots("local-one")
        self.assertEqual(local.read(report / "commands/0000-init.json")["status"], "complete")

    def test_campaign_requires_reparsed_numerical_prerequisites(self):
        self.prepare()
        core = self.mock(local.bench, "main")
        with self.assertRaises(FileNotFoundError):
            local.forward("local-one", "campaign")
        core.assert_not_called()
        report, _ = local.roots("local-one")
        self.assertEqual(local.read(report / "commands/0000-campaign.json")["status"], "failed")

    def numerical_fixture(self):
        """Six actual parser streams; compiler/runtime work is never executed."""
        from tirust_benchmark_warmup import parse_warmup
        report, work = local.roots("local-one")
        directory = report / "report/experiments/numerical/ordinary-example"
        directory.mkdir(parents=True)
        names = set(local.PROTOCOLS) - {"tirust_benchmark_local.py", "tirust_benchmark_campaign.py"}
        for name in names:
            shutil.copy2(self.root / "tools" / name, directory / name)
        case = self.config["cases"][0]
        write(directory / "specification.json", {"case": case, "measurement_phase": "steady-warmup",
            "performance_eligible": False, "machine": {"selected_cpu": 0},
            "configuration_sha256": local.bench.digest(report / "report/cases.json"),
            "protocols": {name: local.bench.digest(directory / name) for name in names}})
        results, reference = [], None
        for source, backend in [("baseline", 1), *(("candidate", b) for b in (1, 6, 7, 8, 9))]:
            lane = directory / f"{source}-{backend}"
            raw = lane / "raw"
            status = 210 + backend
            write(raw / "dimensions.dat", "1 1 1\n")
            write(raw / "warmup.dat", "".join(f"{call} {call} {order} {status} 1 2 3 4 1e-10\n"
                  for call in range(1, 101) for order in (0, 1)))
            ready = {"points": 100, "ndim": 1, "nso": 1, "nborn": 1, "cold_seconds": 2.0}
            write(raw / "ready.json", ready)
            observation = parse_warmup(raw, [backend], 100, ready)
            if reference is None:
                reference = observation
            output = work / "outputs" / source / "example"
            write(output / "SubProcesses/P0_example/check", "fake compiled bytes")
            compilation = {"numerical_build_from_sources": True, "sources": {},
                "executable_sha256": local.bench.digest(output / "SubProcesses/P0_example/check")}
            generation = {"case": case, "source": source, "parameter_card_sha256": "physical-card",
                          "source_identity": self.source}
            write(output / "compilation.json", compilation)
            write(output / "generation.json", generation)
            settings = {"mlreductionlib": str(backend), "ctmoderun": -1}
            if source == "candidate":
                settings["tirustcomputefinite"] = True
            inputs = {"generation": generation, "compilation": compilation, "settings": settings,
                      "param_card_sha256": "physical-card", "worker_timeout_seconds": 14400.,
                      "points_sha256": local.bench.digest(report / "report/points/example.dat"),
                      "numerical_cards": {}, "runtime": {"libraries": {}, "archives": {}, "metadata": {}}}
            write(lane / (source + "-inputs.json"), inputs)
            write(lane / "inputs-after.json", {"executable_sha256": compilation["executable_sha256"],
                  "points_sha256": inputs["points_sha256"], "numerical_cards": {}, "runtime": inputs["runtime"]})
            item = {"source": source, "backend": backend, "status": "complete", "performance_eligible": False,
                    "observation": observation,
                    "comparison": local.bench.compare_rows(reference["rows"], observation["rows"], 3e-6)}
            write(lane / "result.json", item)
            results.append(item)
        write(directory / "results.json", {"results": results})
        self.mock(local.bench, "generated_sources", return_value={})
        self.mock(local.bench, "verify_worker_template")
        return directory

    def test_numerical_prerequisite_replays_all_six_actual_raw_streams(self):
        from tirust_benchmark_worker import MeasurementError
        self.prepare()
        directory = self.numerical_fixture()
        with local.isolated("local-one"):
            proof = local.verify_numerical(0)
            self.assertIn("candidate-9/raw/warmup.dat", proof["example"])
            path = directory / "candidate-9/raw/warmup.dat"
            path.write_text(path.read_text().replace(" 219 ", " 218 "))
            with self.assertRaises(MeasurementError):
                local.verify_numerical(0)

    def test_numerical_prerequisite_checks_postbindings_and_current_build(self):
        self.prepare()
        directory = self.numerical_fixture()
        after = directory / "candidate-8/inputs-after.json"
        original = after.read_bytes()
        changed = local.read(after)
        changed["executable_sha256"] = "wrong executable"
        write(after, changed)
        with local.isolated("local-one"):
            with self.assertRaisesRegex(ValueError, "postcheck binding"):
                local.verify_numerical(0)
            after.write_bytes(original)
            executable = local.bench.WORK / "outputs/candidate/example/SubProcesses/P0_example/check"
            executable.write_text("changed compiled bytes")
            with self.assertRaisesRegex(ValueError, "current compiled"):
                local.verify_numerical(0)

    def test_paths_reject_symlinks_and_traversal(self):
        with self.assertRaises(ValueError):
            local.within(self.root, "../escape")
        (self.root / "alias").symlink_to(self.report, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            local.within(self.root, "alias/file")


class TestInstalledRecipe(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch.object(local.bench, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.prefix = self.root / "HEPTools/tirust"
        self.git = {"name": "oneloop", "version": "0.1", "source": "git+https://example.test/oneloop?rev=frozen#frozen"}
        lock = ('version = 4\n[[package]]\nname="oneloop"\nversion="0.1"\n'
                'source="git+https://example.test/oneloop?rev=frozen#frozen"\n')
        write(self.root / "vendor/tirust/Cargo.lock", lock)
        write(self.root / "vendor/tirust/patches/manifest.json", {"upstream_revision": "tirust-revision", "patch_sha256": "patch"})
        sources = {}
        for installed, source in (("Cargo.toml", "vendor/tirust/Cargo.toml"),
                                  ("include/mtirust.f90", "vendor/tirust/mtirust.f90"),
                                  ("include/tirust.h", "vendor/tirust/tirust.h")):
            write(self.prefix / installed, "source " + source)
            sources[source] = local.bench.digest(self.prefix / installed)
        write(self.prefix / "Cargo.lock", lock)
        artifacts = {}
        for name in ("libmadloop_tirust.so", "libtirust.a", "mtirust.mod"):
            write(self.prefix / "lib" / name, "native " + name)
            artifacts[name] = local.bench.digest(self.prefix / "lib" / name)
        self.canonical = {"candidate": {"exporter_identity": {"files": sources},
                                       "native": {"files": {"Cargo.lock": local.bench.digest(self.prefix / "Cargo.lock")}}}}
        self.config = {"abi_version": 1, "quad_precision": True, "finite_terms": True, "poles_only": True,
                       "artifact_sha256": artifacts,
                       "local_sources": {"tirust": {"revision": "tirust-revision", "patch_sha256": "patch"}},
                       "build_provenance": {"manifest_sha256": sources["vendor/tirust/Cargo.toml"],
                           "lock_sha256": local.bench.digest(self.prefix / "Cargo.lock"), "git_dependencies": [self.git]}}
        self.save()

    def save(self):
        write(self.prefix / "lib/tirust-config.json", self.config)

    def test_recipe_matches_source_interfaces_lock_and_git_pins(self):
        self.assertEqual(local.verify_recipe(self.prefix, self.canonical), self.config["build_provenance"])

    def test_changed_interface_module_or_resolved_lock_is_rejected(self):
        for name, message in (("include/mtirust.f90", "canonical bridge source"),
                              ("lib/mtirust.mod", "native/module bytes"),
                              ("Cargo.lock", "resolved Cargo.lock")):
            with self.subTest(name=name):
                path = self.prefix / name
                original = path.read_bytes()
                path.write_text("changed")
                with self.assertRaisesRegex(ValueError, message):
                    local.verify_recipe(self.prefix, self.canonical)
                path.write_bytes(original)

    def test_false_precision_or_wrong_revision_patch_and_git_provenance_rejected(self):
        pristine = copy.deepcopy(self.config)
        mutations = [lambda v: v.update(quad_precision=False),
                     lambda v: v["local_sources"]["tirust"].update(revision="other"),
                     lambda v: v["local_sources"]["tirust"].update(patch_sha256="other"),
                     lambda v: v["build_provenance"]["git_dependencies"][0].update(source="different")]
        for mutate in mutations:
            self.config = copy.deepcopy(pristine)
            mutate(self.config)
            self.save()
            with self.assertRaises(ValueError):
                local.verify_recipe(self.prefix, self.canonical)


class TestLocalInstallEntrypoint(unittest.TestCase):
    def test_real_launcher_is_tracked_and_hash_bound_by_source_inventory(self):
        tracked = subprocess.check_output(["git", "ls-files", "-z", "--", "bin/madgraph"], cwd=ROOT)
        self.assertEqual(tracked, b"bin/madgraph\0")
        launcher = ROOT / "bin/madgraph"
        self.assertTrue(launcher.is_file())
        self.assertFalse(launcher.is_symlink())
        source = local.bench.source_identity(ROOT, persist=False)
        self.assertEqual(source["files"]["bin/madgraph"], local.bench.digest(launcher))


if __name__ == "__main__":
    unittest.main()
