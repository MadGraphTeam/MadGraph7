"""A new namespace must preserve evidence and require a changed computation."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import tirust_benchmark_campaign as campaign


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n" if not isinstance(value, str) else value)


class TestCampaign(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        report, work = root / "docs/benchmarkfolder", root / ".benchmark-work/tirust"
        self.bench = SimpleNamespace(ROOT=root, LEGACY_REPORT=report, LEGACY_WORK=work,
            REPORT=report, WORK=work, BASELINE=work / "baseline", BASELINE_RECORD=work / "baseline.json",
            BASELINE_REVISION="frozen-baseline", MODEL_ROOT=work / "models", CAMPAIGN_ID=None,
            CAMPAIGN_MANIFEST=None, CAMPAIGN_MANIFEST_SHA256=None, digest=digest,
            content_digest=content_digest, utc_now=lambda: "2026-11-09T00:00:00Z")
        self.bench.BASELINE.mkdir(parents=True)
        self.source_names = ["vendor/tirust/src/lib.rs", "vendor/tirust/mtirust.f90",
                             "Template/loop_material/StandAlone/SubProcesses/kernel.f",
                             "madgraph/loop/loop_exporters.py"]
        for name, source in zip(self.source_names, ("fn reduce() { 1 }\n", "subroutine bridge\nend\n",
                                                     "C generated banner\n      X=1D0\n", "exporter = 1\n")):
            write(root / name, source)
        self.revision = "old-git-metadata"
        self.source_calls = []

        def source_identity(tree, persist=True):
            self.source_calls.append(persist)
            files = {name: digest(root / name) for name in self.source_names}
            return {"revision": self.revision, "files": files, "files_sha256": content_digest(files), "dirty": False}
        self.bench.source_identity = source_identity
        self.legacy_identity = source_identity(root, persist=False)
        self.original_sources = {name: (root / name).read_bytes() for name in self.source_names}
        baseline = {"revision": "frozen-baseline", "files": {}, "archive_sha256": "base-archive"}
        write(self.bench.BASELINE_RECORD, baseline)
        write(report / "inputs/baseline.json", baseline)
        write(report / "cases.json", {"cases": [{"id": "example"}], "points_per_case": 100})
        write(report / "points/example.dat", "100\nfrozen points\n")
        write(report / "points/example.json", {"sha256": digest(report / "points/example.dat")})
        model = work / "models" / campaign.MODEL
        write(model / "particles.py", "frozen UFO\n")
        write(report / f"inputs/{campaign.MODEL}.json", {"files": {"particles.py": digest(model / "particles.py")}})
        self.prefix = root / "HEPTools/optimized-tirust"
        self._install("optimized DSO\n")
        self.legacy_dso = hashlib.sha256(b"original DSO\n").hexdigest()
        self.legacy_archive = digest(self.prefix / "lib/libtirust.a")
        old_inputs = {"generation": {"source": "candidate", "source_identity": self.legacy_identity},
                      "runtime": {"libraries": {"/old/libmadloop_tirust.so": self.legacy_dso},
                                  "archives": {"/old/libtirust.a": self.legacy_archive}}}
        self.failure = report / "runs/example/tirust-first"
        write(self.failure / "candidate-inputs.json", old_inputs)
        write(self.failure / "summary.json", {"label": "tirust-first", "acceptance_eligible": True, "passed": False})
        write(self.failure / "raw/values.dat", "failed numerical evidence\n")
        write(report / "summary.json", {"gate": {"passed": False}})
        self.bench.TIRUST_PREFIX = self.prefix

    def _install(self, dso):
        write(self.prefix / "lib/libmadloop_tirust.so", dso)
        write(self.prefix / "lib/libtirust.a", "same archive\n")
        write(self.prefix / "lib/tirust-config.json", {
            "artifact_sha256": {name: digest(self.prefix / "lib" / name)
                                for name in ("libmadloop_tirust.so", "libtirust.a")}})

    def register(self, name="optimized", **kwargs):
        return campaign.register_campaign(self.bench, name, self.prefix,
                                          reason="Changed numerical implementation", **kwargs)

    def test_registration_seeds_private_inputs_without_legacy_writes(self):
        legacy = {str(path): digest(path) for path in self.bench.LEGACY_REPORT.rglob("*") if path.is_file()}
        baseline = self.bench.BASELINE
        manifest = self.register()
        self.assertEqual(manifest[campaign.TIMEOUT_FIELD], {"example": 3600.0})
        self.assertEqual(manifest["predecessor"]["id"], "legacy")
        self.assertEqual(len(manifest["predecessor"]["qualified_failures"]), 1)
        context = campaign.activate_campaign(self.bench, "optimized")
        self.assertEqual(self.bench.BASELINE, baseline)
        self.assertEqual(self.bench.BASELINE_RECORD, self.bench.LEGACY_WORK / "baseline.json")
        self.assertEqual(self.bench.WORK, self.bench.LEGACY_WORK / "campaigns/optimized")
        self.assertEqual(context["manifest_sha256"], digest(self.bench.REPORT / "manifest.json"))
        self.assertFalse((self.bench.REPORT / "runs").exists())
        self.assertFalse((self.bench.REPORT / "summary.json").exists())
        self.assertFalse((self.bench.REPORT / "qualification.json").exists())
        self.assertFalse((self.bench.MODEL_ROOT / campaign.MODEL).is_symlink())
        self.assertEqual(legacy, {path: digest(path) for path in legacy})
        self.assertFalse(any(self.source_calls))
        self.assertEqual(campaign.verify_campaign(self.bench)["campaign_binding"]["id"], "optimized")
        with self.assertRaisesRegex(ValueError, "append-only"):
            self.register()

    def test_namespace_and_symlink_escapes_are_rejected(self):
        for name in ("../escape", "/escape", "a/b", "", "legacy", ".", "a..b"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.register(name)
        directory = self.bench.LEGACY_REPORT / "campaigns"
        directory.mkdir()
        (directory / "alias").symlink_to(self.bench.ROOT)
        with self.assertRaisesRegex(ValueError, "symlink|escapes"):
            self.register("alias")

    def test_metadata_only_changes_do_not_reset_a_qualified_failure(self):
        self._install("original DSO\n")
        self.revision = "new-exporter-git-metadata"
        write(self.bench.ROOT / "madgraph/loop/loop_exporters.py", "new metadata only\n")
        write(self.bench.ROOT / self.source_names[2], "C different generated banner\n      X=1D0 ! comment\n")
        config = self.prefix / "lib/tirust-config.json"
        value = json.loads(config.read_text())
        value["new_timestamp"] = "new install metadata"
        write(config, value)
        with patch.object(campaign.subprocess, "check_output", side_effect=lambda cmd, **kw:
                          self.original_sources[cmd[-1].split(":", 1)[1]]):
            with self.assertRaisesRegex(ValueError, "unchanged numerical identity"):
                self.register()
        self.assertFalse((self.bench.LEGACY_REPORT / "campaigns/optimized").exists())

    def test_unbuilt_bridge_edit_is_not_a_new_native_computation(self):
        self._install("original DSO\n")
        write(self.bench.ROOT / "vendor/tirust/src/lib.rs", "fn reduce() { 2 }\n")
        with patch.object(campaign.subprocess, "check_output", side_effect=lambda cmd, **kw:
                          self.original_sources[cmd[-1].split(":", 1)[1]]):
            with self.assertRaisesRegex(ValueError, "unchanged numerical identity"):
                self.register()

    def test_fortran_numerical_change_qualifies_without_native_rebuild(self):
        self._install("original DSO\n")
        write(self.bench.ROOT / self.source_names[2], "      X=2D0\n")
        with patch.object(campaign.subprocess, "check_output", side_effect=lambda cmd, **kw:
                          self.original_sources[cmd[-1].split(":", 1)[1]]):
            manifest = self.register()
        self.assertNotEqual(manifest["candidate"]["numerical_identity_sha256"],
                            manifest["predecessor"]["numerical_identity_sha256"])

    def test_registered_input_and_predecessor_mutations_fail_closed(self):
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        point = self.bench.REPORT / "points/example.dat"
        original = point.read_bytes()
        point.write_text("changed points")
        with self.assertRaisesRegex(ValueError, "Frozen campaign input"):
            campaign.verify_campaign(self.bench)
        point.write_bytes(original)
        evidence = self.failure / "raw/values.dat"
        evidence.write_text("changed old failure")
        with self.assertRaisesRegex(ValueError, "qualified failures"):
            campaign.verify_campaign(self.bench, stage="summarize")

    def test_live_guards_reject_source_and_library_changes_but_history_remains_readable(self):
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        write(self.bench.ROOT / "madgraph/loop/loop_exporters.py", "future implementation\n")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            campaign.verify_campaign(self.bench, stage="worker")
        self.assertEqual(campaign.verify_campaign(self.bench, stage="summarize")["campaign_binding"]["id"], "optimized")
        campaign.activate_campaign(self.bench, "optimized")
        write(self.bench.ROOT / "madgraph/loop/loop_exporters.py", "exporter = 1\n")
        self._install("future native DSO\n")
        with self.assertRaisesRegex(ValueError, "Native TIRust installation changed"):
            campaign.verify_campaign(self.bench)
        campaign.verify_campaign(self.bench, stage="validate-retained")

    def test_manifest_tampering_and_native_config_mismatch_are_rejected(self):
        config = self.prefix / "lib/tirust-config.json"
        original = config.read_bytes()
        value = json.loads(original)
        value["artifact_sha256"]["libtirust.a"] = "wrong"
        write(config, value)
        with self.assertRaisesRegex(ValueError, "config disagrees"):
            self.register()
        config.write_bytes(original)
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        path = self.bench.REPORT / "manifest.json"
        path.write_text(path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "manifest"):
            campaign.verify_campaign(self.bench)

    def test_recorded_runtime_and_campaign_binding_are_verified(self):
        manifest = self.register()
        campaign.activate_campaign(self.bench, "optimized")
        files = manifest["candidate"]["native"]["files"]
        inputs = {"generation": {"source": "candidate", "source_identity": manifest["candidate"]["exporter_identity"]},
                  "campaign_binding": campaign.campaign_binding(self.bench),
                  "runtime": {"libraries": {"/private/libmadloop_tirust.so": files[campaign.NATIVE_NAMES[0]],
                                            "/v1/libc.so": "one", "/v2/libc.so": "two"},
                              "archives": {"/private/libtirust.a": files[campaign.NATIVE_NAMES[1]]},
                              "metadata": {"/private/tirust-config.json": files[campaign.NATIVE_NAMES[2]]}}}
        campaign.verify_campaign(self.bench, stage="validate-retained", inputs=inputs)
        inputs["runtime"]["libraries"]["/private/libmadloop_tirust.so"] = "wrong"
        with self.assertRaisesRegex(ValueError, "Recorded TIRust runtime"):
            campaign.verify_campaign(self.bench, stage="validate-retained", inputs=inputs)
        inputs["campaign_binding"] = {"id": "different", "manifest_sha256": "wrong"}
        with self.assertRaisesRegex(ValueError, "different campaign"):
            campaign.verify_campaign(self.bench, stage="validate-retained", inputs=inputs)

    def test_missing_failure_attempt_is_detected(self):
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        (self.failure / "summary.json").unlink()
        with self.assertRaisesRegex(ValueError, "qualified failures"):
            campaign.verify_campaign(self.bench, stage="summarize")

    def test_predecessor_headline_is_snapshotted_and_volatile_caches_are_ignored(self):
        manifest = self.register()
        campaign.activate_campaign(self.bench, "optimized")
        snapshot = self.bench.LEGACY_REPORT / manifest["predecessor"]["summary_snapshot"]
        self.assertEqual(digest(snapshot), manifest["predecessor"]["summary_sha256"])
        write(self.bench.LEGACY_REPORT / "summary.json", {"updated": "late run status"})
        write(self.failure / "protocol-sources/__pycache__/worker.cpython-313.pyc", "runtime cache")
        write(self.failure / "raw/copy.tmp", "temporary copy")
        campaign.verify_campaign(self.bench, stage="summarize")
        write(snapshot, {"tampered": True})
        with self.assertRaisesRegex(ValueError, "summary snapshot changed"):
            campaign.verify_campaign(self.bench, stage="summarize")

    def test_late_failures_are_discovered_and_remain_immutable(self):
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        late = self.failure.with_name("tirust-first-attempt-002")
        shutil.copytree(self.failure, late)
        fields = campaign.verify_campaign(self.bench, stage="summarize")
        self.assertEqual(len(fields["campaign_predecessor"]["qualified_failures"]), 2)
        discoveries = list((self.bench.REPORT / "inputs/predecessor/discoveries").glob("*.json"))
        self.assertEqual(len(discoveries), 1)
        self.assertEqual(discoveries[0].stem, content_digest(json.loads(discoveries[0].read_text())))
        fields_again = campaign.verify_campaign(self.bench, stage="summarize")
        self.assertEqual(fields_again, fields)
        (late / "raw/values.dat").unlink()
        with self.assertRaisesRegex(ValueError, "changed or disappeared"):
            campaign.verify_campaign(self.bench, stage="summarize")

    def test_discovery_tampering_is_rejected(self):
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        shutil.copytree(self.failure, self.failure.with_name("late-failure"))
        campaign.verify_campaign(self.bench, stage="summarize")
        discovery = next((self.bench.REPORT / "inputs/predecessor/discoveries").glob("*.json"))
        value = json.loads(discovery.read_text())
        value["failure"]["label"] = "backend-1"
        write(discovery, value)
        with self.assertRaisesRegex(ValueError, "discovery record changed"):
            campaign.verify_campaign(self.bench, stage="summarize")

    def test_same_identity_late_failure_is_retained_and_rejected_repeatedly(self):
        manifest = self.register()
        campaign.activate_campaign(self.bench, "optimized")
        late = self.failure.with_name("same-identity-late-failure")
        shutil.copytree(self.failure, late)
        inputs_path = late / "candidate-inputs.json"
        inputs = json.loads(inputs_path.read_text())
        inputs["runtime"]["libraries"]["/old/libmadloop_tirust.so"] = manifest["candidate"]["native"]["files"][campaign.NATIVE_NAMES[0]]
        write(inputs_path, inputs)
        for unused in range(2):
            with self.assertRaisesRegex(ValueError, "this campaign's numerical identity"):
                campaign.verify_campaign(self.bench, stage="summarize")
        self.assertEqual(len(list((self.bench.REPORT / "inputs/predecessor/discoveries").glob("*.json"))), 1)

    def test_all_initial_failures_participate_in_identity_gate(self):
        other = self.failure.with_name("z-matching-candidate")
        shutil.copytree(self.failure, other)
        inputs_path = other / "candidate-inputs.json"
        inputs = json.loads(inputs_path.read_text())
        inputs["runtime"]["libraries"]["/old/libmadloop_tirust.so"] = digest(self.prefix / "lib/libmadloop_tirust.so")
        write(inputs_path, inputs)
        with self.assertRaisesRegex(ValueError, "predecessor failure already has"):
            self.register()

    def test_named_predecessor_carries_ancestral_failure_evidence(self):
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        prior_report = self.bench.REPORT
        write(prior_report / "summary.json", {"pending": True})
        self._install("second optimization DSO\n")
        manifest = campaign.register_campaign(self.bench, "second", self.prefix,
                                              predecessor="optimized", reason="Another optimization")
        self.assertEqual(manifest["predecessor"]["ancestor"]["id"], "legacy")
        campaign.activate_campaign(self.bench, "second")
        write(prior_report / "summary.json", {"updated": True})
        fields = campaign.verify_campaign(self.bench, stage="summarize")
        failures = fields["campaign_predecessor"]["qualified_failures"]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]["predecessor_id"], "legacy")
        (self.failure / "raw/values.dat").write_text("lost original failure")
        with self.assertRaisesRegex(ValueError, "qualified failures"):
            campaign.verify_campaign(self.bench, stage="summarize")

    def test_named_successor_cannot_omit_discovered_ancestral_failure(self):
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        late = self.failure.with_name("late-failure")
        shutil.copytree(self.failure, late)
        campaign.verify_campaign(self.bench, stage="summarize")
        write(self.bench.REPORT / "summary.json", {"pending": True})
        shutil.rmtree(late)
        self._install("second optimization DSO\n")
        with self.assertRaisesRegex(ValueError, "Inherited predecessor discovery"):
            campaign.register_campaign(self.bench, "second", self.prefix,
                                       predecessor="optimized", reason="Another optimization")
        self.assertFalse((self.bench.LEGACY_REPORT / "campaigns/second").exists())

    def test_exact_source_snapshots_preserve_dirty_historical_identity(self):
        write(self.bench.ROOT / self.source_names[2], "      X=3D0\n")
        with patch.object(campaign.subprocess, "check_output", side_effect=lambda cmd, **kw:
                          self.original_sources[cmd[-1].split(":", 1)[1]]):
            manifest = self.register()
            campaign.activate_campaign(self.bench, "optimized")
        dirty_identity = manifest["candidate"]["exporter_identity"]
        snapshot = self.bench.REPORT / manifest["candidate"]["source_snapshot"] / self.source_names[2]
        self.assertEqual(snapshot.read_text(), "      X=3D0\n")
        write(self.bench.ROOT / self.source_names[2], "      X=4D0\n")
        with patch.object(campaign.subprocess, "check_output", side_effect=AssertionError("git cannot supply dirty bytes")):
            historical = campaign._sources(self.bench, dirty_identity, historical=True)
        self.assertEqual(historical, manifest["candidate"]["numerical_sources"])
        snapshot.write_text("mutated snapshot")
        with self.assertRaisesRegex(ValueError, "source snapshot changed"):
            campaign.verify_campaign(self.bench, stage="summarize")

    def test_duplicate_identity_is_rejected_before_either_campaign_runs(self):
        self.register("first")
        self.revision = "new metadata cannot create another experiment"
        with self.assertRaisesRegex(ValueError, "already registered this numerical identity/policy"):
            self.register("second")
        self.assertFalse((self.bench.LEGACY_REPORT / "campaigns/second").exists())
        self.assertFalse((self.bench.LEGACY_WORK / "campaigns/second").exists())

    def test_live_guard_rejects_qualified_failure_in_older_duplicate_namespace(self):
        manifest = self.register("first")
        campaign.activate_campaign(self.bench, "first")
        sibling = self.bench.LEGACY_REPORT / "campaigns/old-duplicate"
        old_manifest = {**manifest, "id": "old-duplicate"}
        write(sibling / "manifest.json", old_manifest)
        failed = sibling / "runs/example/tirust-first"
        write(failed / "summary.json", {"label": "tirust-first", "acceptance_eligible": True, "passed": False})
        for stage in ("generate", "build", "worker", "pair-before", "pair-after"):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError, "sibling campaign"):
                campaign.verify_campaign(self.bench, stage=stage)
        # Historical evidence remains readable after another campaign fails.
        campaign.verify_campaign(self.bench, stage="summarize")

    def test_administrative_timeout_overrides_bind_all_cases_without_changing_policy(self):
        config_path = self.bench.LEGACY_REPORT / "cases.json"
        config = json.loads(config_path.read_text())
        config["cases"].append({"id": "heavy", "nfinal": 4})
        write(config_path, config)
        for suffix in (".dat", ".json"):
            shutil.copy2(self.bench.LEGACY_REPORT / ("points/example" + suffix),
                         self.bench.LEGACY_REPORT / ("points/heavy" + suffix))
        original_config = config_path.read_bytes()
        manifest = self.register(worker_timeouts={"heavy": 14400})
        self.assertEqual(manifest[campaign.TIMEOUT_FIELD], {"example": 3600.0, "heavy": 14400.0})
        self.assertEqual(manifest["shared_inputs"]["policy_sha256"], content_digest(config))
        campaign.activate_campaign(self.bench, "optimized")
        self.assertEqual((self.bench.REPORT / "cases.json").read_bytes(), original_config)
        self.assertEqual(config_path.read_bytes(), original_config)
        self.assertEqual(campaign.verify_campaign(self.bench)[campaign.TIMEOUT_FIELD],
                         manifest[campaign.TIMEOUT_FIELD])

    def test_timeout_overrides_reject_unknown_nonfinite_nonpositive_and_boolean_values(self):
        invalid = [{"unknown": 1}, {"example": True}, {"example": False}, {"example": 0},
                   {"example": -1}, {"example": float("nan")}, {"example": float("inf")},
                   {"example": -float("inf")}, {"example": "3600"}, {"example": None},
                   {"example": 10**1000}, [], 3600]
        for index, value in enumerate(invalid):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "worker timeout"):
                self.register("invalid_" + str(index), worker_timeouts=value)
            self.assertFalse((self.bench.LEGACY_REPORT / "campaigns" / ("invalid_" + str(index))).exists())

    def test_timeout_map_is_inherited_and_individual_values_may_be_overridden(self):
        first = self.register("first", worker_timeouts={"example": 14400})
        first_report = self.bench.LEGACY_REPORT / "campaigns/first"
        write(first_report / "summary.json", {"pending": True})
        self._install("second optimization DSO\n")
        second = self.register("second", predecessor="first")
        self.assertEqual(second[campaign.TIMEOUT_FIELD], first[campaign.TIMEOUT_FIELD])
        write(self.bench.LEGACY_REPORT / "campaigns/second/summary.json", {"pending": True})
        self._install("third optimization DSO\n")
        third = self.register("third", predecessor="second", worker_timeouts={"example": 7200.5})
        self.assertEqual(third[campaign.TIMEOUT_FIELD], {"example": 7200.5})
        self.assertEqual(first["shared_inputs"]["policy_sha256"], third["shared_inputs"]["policy_sha256"])
        campaign.activate_campaign(self.bench, "third")

    def test_missing_or_invalid_timeout_map_is_rejected_on_activation(self):
        self.register()
        path = self.bench.LEGACY_REPORT / "campaigns/optimized/manifest.json"
        original = json.loads(path.read_text())
        missing = {key: value for key, value in original.items() if key != campaign.TIMEOUT_FIELD}
        invalid = [missing, *({**original, campaign.TIMEOUT_FIELD: value}
                              for value in ({}, {"example": True}, {"example": 0},
                                            {"example": 3600, "unknown": 1}, None))]
        for value in invalid:
            write(path, value)
            with self.subTest(value=value.get(campaign.TIMEOUT_FIELD)), self.assertRaisesRegex(ValueError, "worker timeout"):
                campaign.activate_campaign(self.bench, "optimized")
        write(path, original)
        campaign.activate_campaign(self.bench, "optimized")

    def test_changed_timeout_map_invalidates_the_active_manifest_binding(self):
        self.register()
        campaign.activate_campaign(self.bench, "optimized")
        path = self.bench.REPORT / "manifest.json"
        value = json.loads(path.read_text())
        value[campaign.TIMEOUT_FIELD]["example"] = 14400
        write(path, value)
        with self.assertRaisesRegex(ValueError, "manifest"):
            campaign.verify_campaign(self.bench, stage="worker")

    def test_timeout_only_changes_cannot_reset_failed_or_duplicate_identity(self):
        self._install("original DSO\n")
        with self.assertRaisesRegex(ValueError, "unchanged numerical identity"):
            self.register(worker_timeouts={"example": 14400})
        self._install("optimized DSO\n")
        self.register("first", worker_timeouts={"example": 3600})
        with self.assertRaisesRegex(ValueError, "already registered this numerical identity/policy"):
            self.register("second", worker_timeouts={"example": 14400})

    def test_legacy_helpers_are_noops(self):
        self.assertIsNone(campaign.campaign_binding(self.bench))
        self.assertEqual(campaign.campaign_summary_fields(self.bench), {})
        self.assertEqual(campaign.verify_campaign(self.bench), {})


if __name__ == "__main__":
    unittest.main()
