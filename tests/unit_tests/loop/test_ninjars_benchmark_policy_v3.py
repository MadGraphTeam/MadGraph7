"""NinjaRS-first evidence boundaries; no model, build or numerical engine runs."""
import copy
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v3 as policy


def configuration():
    return json.loads((ROOT / "docs/benchmarkfolder/policy-v2/cases.json").read_text())


def observation(backend=8, case="qcd-1", source=None, repetition="calibration"):
    code = {1: 211, 6: 216, 8: 218}[backend]
    return {"case": case, "source": source or ("candidate" if backend == 8 else "baseline"),
        "backend": backend, "repetition": repetition, "loops": 1, "cpu": 207,
        "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
        "rows": [{"point": p, "order": o, "coefficients": [1., .2, -.03, .04],
                  "status": code, "accuracy": 1e-12} for p in range(1, 101) for o in (0, 1)],
        "status_counts": {str(code): 100}, "timed_status_counts": {str(code): 100},
        "seconds": 1., "seconds_per_point": .01, "cold_seconds": .1,
        "shared_numerical_settings": policy.expected_settings(),
        "points_sha256": "1" * 64, "parameter_card_sha256": "2" * 64,
        "madloop_card_sha256": "3" * 64, "executable_sha256": "4" * 64,
        "generated_sources_sha256": "5" * 64, "measurement_run_id": "test-pair",
        "paired_baseline_candidate": True}


class RowsAndGatesTest(unittest.TestCase):
    def test_exact_core_inventory_and_numerical_policy(self):
        config = configuration()
        policy.validate_configuration(config)
        for key, value in (("points_per_case", 99), ("relative_tolerance", 1e-6),
                           ("scale_tolerance", 1e-9), ("max_runtime_ratio", 1.1), ("repetitions", 6)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                policy.validate_configuration(dict(config, **{key: value}))
        changed = copy.deepcopy(config)
        changed["cases"][0]["process"] = "u u~ > z z [virt=QCD]"
        with self.assertRaises(RuntimeError):
            policy.validate_configuration(changed)
        changed = copy.deepcopy(config)
        changed["madloop_parameters"]["CTStabThres"] = .1
        with self.assertRaisesRegex(RuntimeError, "additional"):
            policy.validate_configuration(changed)

    def test_correctness_two_commands_and_different_case_order_are_not_seven_timings(self):
        config = configuration()
        config["cases"].reverse()
        config["repetitions"] = 2
        policy.validate_configuration(config, correctness=True)
        with self.assertRaises(RuntimeError):
            policy.validate_configuration(config)
        config["repetitions"] = 1
        with self.assertRaisesRegex(RuntimeError, "two actual"):
            policy.validate_configuration(config, correctness=True)

    def test_missing_or_duplicate_core_case_is_not_complete(self):
        for mutation in ("missing", "duplicate", "extra"):
            config = configuration()
            if mutation == "missing": config["cases"].pop()
            elif mutation == "duplicate": config["cases"][-1] = config["cases"][0]
            else: config["cases"].append(dict(config["cases"][0], id="extra"))
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                policy.validate_configuration(config)

    def test_all_normal_backend_codes_keep_original_meanings(self):
        for backend in (1, 6, 8):
            record = observation(backend)
            policy.validate_rows(record, backend)
        record = observation()
        for row in record["rows"]:
            if row["point"] == 100: row["status"] = 322
        record["status_counts"] = {"218": 99, "322": 1}
        record["timed_status_counts"] = dict(record["status_counts"])
        policy.validate_rows(record, 8)

    def test_failure_direct_qp_wrong_backend_and_initialization_codes_rejected(self):
        for code in (410, 120, 216, 238, 318, 328):
            record = observation()
            record["rows"][0]["status"] = code
            with self.subTest(code=code), self.assertRaises(RuntimeError):
                policy.validate_rows(record, 8)

    def test_all_points_orders_and_coefficients_are_required(self):
        for mutation in ("missing", "duplicate", "order", "short", "nan", "infinity", "index-bool"):
            record = observation()
            if mutation == "missing": record["rows"].pop()
            elif mutation == "duplicate": record["rows"][-1] = record["rows"][-2]
            elif mutation == "order": record["rows"].reverse()
            elif mutation == "short": record["rows"][0]["coefficients"].pop()
            elif mutation == "nan": record["rows"][0]["coefficients"][2] = float("nan")
            elif mutation == "infinity": record["rows"][0]["coefficients"][0] = float("inf")
            else: record["rows"][0]["point"] = True
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                policy.validate_rows(record, 8)

    def test_estimates_dimensions_and_status_histograms_are_not_optional(self):
        mutations = [lambda r: r["rows"][0].update(accuracy=-1),
                     lambda r: r["rows"][0].update(accuracy=float("nan")),
                     lambda r: r["answer_dimensions"].update(allocated=2),
                     lambda r: r.update(status_counts={"218": 99}),
                     lambda r: r.update(timed_status_counts={"218": 99}),
                     lambda r: r.update(timed_status_counts={"218": True}),
                     lambda r: r.update(seconds=float("inf")),
                     lambda r: r.update(seconds_per_point=.02)]
        for index, change in enumerate(mutations):
            record = observation(); change(record)
            with self.subTest(index=index), self.assertRaises(RuntimeError):
                policy.validate_rows(record, 8)

    def test_threshold_pole_loop_filter_and_other_numeric_drift_rejected(self):
        for key, value in (("mlstabthres", 1e-3), ("collierrequiredaccuracy", 1e-8),
                           ("colliercomputeuvpoles", False), ("colliercomputeirpoles", False),
                           ("useloopfilter", True), ("selected_backend_qp_integrand", False),
                           ("osthres", 1e-3), ("nrotations_qp", 1), ("ctstabthres", .1)):
            record = observation(); record["shared_numerical_settings"][key] = value
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                policy.validate_rows(record, 8)

    def test_missing_downstream_evidence_never_becomes_milestone_or_publication_pass(self):
        result = policy.combine_gates(True, True)
        self.assertTrue(result["nrs_core_validation_passed"])
        self.assertFalse(result["nrs_validation_milestone_passed"])
        self.assertFalse(result["final_publication_passed"])
        self.assertEqual(result["external_review_status"], "pending")
        self.assertIsNone(result["external_failures_recorded"])

    def test_known_external_failure_stays_failed_and_does_not_block_nrs_core(self):
        result = policy.combine_gates(True, True, {k: True for k in policy.QUALIFICATION_KEYS},
                                     {k: True for k in policy.PUBLICATION_KEYS},
                                     [{"status": "failed", "backend": 7}])
        self.assertTrue(result["nrs_core_validation_passed"])
        self.assertTrue(result["nrs_validation_milestone_passed"])
        self.assertTrue(result["final_publication_passed"])
        self.assertTrue(result["external_failures_recorded"])
        self.assertFalse(result["external_backend_comparison_passed"])
        for missing in policy.QUALIFICATION_KEYS:
            qualifiers = {k: True for k in policy.QUALIFICATION_KEYS if k != missing}
            self.assertFalse(policy.combine_gates(True, True, qualifiers)["nrs_validation_milestone_passed"])
        with self.assertRaisesRegex(RuntimeError, "labelled passed"):
            policy.combine_gates(True, True, external_findings=[{"status": "passed", "backend": 7}])

    def test_a_nrs_reference_disagreement_is_not_waived_by_external_failures(self):
        result = policy.combine_gates(False, True, external_findings=[{"status": "failed", "backend": 7}])
        self.assertFalse(result["nrs_core_validation_passed"])
        a, b = observation(), observation()
        b["rows"][-1]["coefficients"][1] += 1e-4
        groups = {"candidate_corrected/8": a["rows"], "original_helper_corrected/1": b["rows"],
                  "original_helper_corrected/6": a["rows"]}
        self.assertFalse(policy.v2.compare_rows(groups)["passed"])


class PrivateFilesMixin:
    def setUp(self):
        (ROOT / ".benchmark-work").mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(prefix="policy-v3-test-", dir=ROOT / ".benchmark-work")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def put(self, filename, data):
        filename = Path(filename); filename.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes): filename.write_bytes(data)
        else: filename.write_text(data)
        return filename

    def dump(self, filename, data):
        return self.put(filename, json.dumps(data, indent=2, allow_nan=False) + "\n")


class PrivateFilesTest(PrivateFilesMixin, unittest.TestCase):

    def release_fixture(self):
        """A real tiny Git source tree and install, with no numerical engine."""
        def git(*args):
            return subprocess.check_output(["git", *args], cwd=self.root, stderr=subprocess.DEVNULL).decode().strip()
        git("init", "--quiet")
        self.put(self.root / "madgraph/source.py", "committed generator\n")
        self.put(self.root / "LICENSE", "symlink target outside generator paths\n")
        (self.root / "madgraph/LICENSE").symlink_to("../LICENSE")
        pins = {"ninjars": "a" * 40, "oneloopmaster": "b" * 40, "symbolica": "c" * 40}
        packages = [{"name": name, "version": "1.0.0", "source": "git+https://example.invalid/" + pin + "#" + pin}
                    for name, pin in (("ninjars", pins["ninjars"]), ("ninjars-symbolica", pins["ninjars"]),
                                      ("oneloop", pins["oneloopmaster"]), ("symbolica", pins["symbolica"]),
                                      ("numerica", pins["symbolica"]), ("graphica", pins["symbolica"]))]
        lock = "version = 4\n" + "\n".join("[[package]]\n" + "\n".join(f"{k} = {json.dumps(v)}" for k, v in package.items()) for package in packages)
        self.put(self.root / "vendor/ninjars/Cargo.toml", '[package]\nname = "test-cohort"\nversion = "0.1.0"\n')
        self.put(self.root / "vendor/ninjars/Cargo.lock", lock)
        git("add", ".")
        git("-c", "user.name=Audit test", "-c", "user.email=audit@example.invalid", "commit", "--quiet", "-m", "fixture")
        revision = git("rev-parse", "HEAD")
        install = self.root / "installed"
        manifest = self.put(install / "Cargo.toml", (self.root / "vendor/ninjars/Cargo.toml").read_bytes())
        lockfile = self.put(install / "Cargo.lock", lock)
        archive = self.put(install / "libninjars.a", b"independent archive fixture\n")
        config = self.dump(install / "config.json", {"abi_version": 3, "quad_precision": True,
            "build_provenance": {"manifest_sha256": policy.workers.digest(manifest), "lock_sha256": policy.workers.digest(lockfile),
                                 "git_dependencies": sorted(packages, key=lambda item: (item["name"], item["source"]))}})
        state = policy._committed_generator_tree(revision)
        self.dump(self.root / "generation/status.json", {"complete": True, "git_revision": revision,
            "generator_tree_state": state, "generator_tree_state_sha256": policy.bench.content_digest(state)})
        value = {"schema": policy.COHORT_SCHEMA, "numerical_revision": revision, "generation_revision": revision,
            "generator_identity_sha256": policy.bench.content_digest(state), "runtime_identity_sha256": "d" * 64,
            "archive_sha256": policy.workers.digest(archive), "generation_root": "generation", "dependency_pins": pins,
            "installation": {name: policy.v2.bound_file(path) for name, path in
                (("archive", archive), ("configuration", config), ("cargo_manifest", manifest), ("cargo_lock", lockfile))}}
        return git, value

    def release_environment(self):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(policy, "ROOT", self.root))
        stack.enter_context(mock.patch.object(policy.v2, "ROOT", self.root))
        stack.enter_context(mock.patch.object(policy.bench, "GENERATOR_PATHS", ("madgraph",)))
        return stack

    def test_explicit_release_is_bound_to_real_committed_symlinks_install_and_pins(self):
        with self.release_environment():
            _, value = self.release_fixture()
            path = self.dump(self.root / "cohort.json", policy.v2.seal(value))
            self.assertEqual(policy.validate_cohort(path), policy.v2.seal(value))
            actual = policy.candidate_cohort({"candidate_cohort": policy.v2.bound_file(path)})
            self.assertEqual(actual["dependency_pins"], value["dependency_pins"])
            state = policy._committed_generator_tree(value["generation_revision"])
            self.assertEqual(state["tracked"]["madgraph/LICENSE"], policy.workers.digest(self.root / "LICENSE"))
        self.assertEqual(policy.candidate_cohort()["archive_sha256"], policy.ARCHIVE_SHA256)

    def test_release_self_consistent_forged_source_pins_and_install_are_rejected(self):
        with self.release_environment():
            _, good = self.release_fixture()
            for mutation in ("short-revision", "tree", "pin", "archive", "missing-install", "configuration"):
                value = copy.deepcopy(good)
                if mutation == "short-revision": value["generation_revision"] = "a" * 8
                elif mutation == "tree": value["generator_identity_sha256"] = "0" * 64
                elif mutation == "pin": value["dependency_pins"]["oneloopmaster"] = "0" * 40
                elif mutation == "archive": value["archive_sha256"] = "0" * 64
                elif mutation == "missing-install": value["installation"].pop("cargo_lock")
                else:
                    config = policy.checked(value["installation"]["configuration"])
                    data = json.loads(config.read_text()); data["build_provenance"]["git_dependencies"].pop()
                    self.dump(config, data); value["installation"]["configuration"] = policy.v2.bound_file(config)
                path = self.dump(self.root / "cohort.json", policy.v2.seal(value))
                with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                    policy.validate_cohort(path)

    def test_release_changed_lock_and_different_production_commit_are_rejected(self):
        with self.release_environment():
            git, good = self.release_fixture()
            lock = policy.checked(good["installation"]["cargo_lock"])
            original = lock.read_bytes()
            lock.write_text(lock.read_text().replace('version = 4', 'version = 3'))
            value = copy.deepcopy(good); value["installation"]["cargo_lock"] = policy.v2.bound_file(lock)
            path = self.dump(self.root / "cohort.json", policy.v2.seal(value))
            with self.assertRaisesRegex(RuntimeError, "Cargo source"):
                policy.validate_cohort(path)
            lock.write_bytes(original)
            self.put(self.root / "vendor/new.rs", "changed numerical source\n")
            git("add", "vendor/new.rs")
            git("-c", "user.name=Audit test", "-c", "user.email=audit@example.invalid", "commit", "--quiet", "-m", "different source")
            value = copy.deepcopy(good); value["numerical_revision"] = git("rev-parse", "HEAD")
            path = self.dump(self.root / "cohort.json", policy.v2.seal(value))
            with self.assertRaisesRegex(RuntimeError, "different production source"):
                policy.validate_cohort(path)

    def test_release_collector_cannot_inherit_old_or_mismatched_policy_cohort(self):
        cohort = policy.candidate_cohort()
        identities = {name: cohort[name] for name in ("numerical_revision", "generation_revision",
            "generator_identity_sha256", "runtime_identity_sha256", "archive_sha256")}
        report = self.root / "release-correctness"
        binding = self.dump(report / "bindings.json", policy.v2.seal({"schema": "supplement-source-bindings-v1",
            "role_map": {"candidate": "candidate_corrected"}, "report": policy.v2.relative(report), "targets": {}}))
        for mutation in ("missing-policy-cohort", "wrong-spec-runtime", "wrong-protocol-archive"):
            spec = {"schema": "final-pins-core-correctness-spec-v1", "cohort": {"candidate_cohort": dict(identities)}}
            protocol = {"candidate_cohort": dict(identities)}
            rules = {"candidate_cohort": {"path": "not-read", "sha256": "0" * 64}}
            if mutation == "missing-policy-cohort": rules.pop("candidate_cohort")
            elif mutation == "wrong-spec-runtime": spec["cohort"]["candidate_cohort"]["runtime_identity_sha256"] = "f" * 64
            else: protocol["candidate_cohort"]["archive_sha256"] = "f" * 64
            self.dump(report / "input-spec.json", policy.v2.seal(spec))
            protocol["input_spec"] = policy.v2.bound_file(report / "input-spec.json")
            with self.subTest(mutation=mutation), ExitStack() as context:
                context.enter_context(mock.patch.object(policy, "validate_policy", return_value=(rules, {}, {}, {})))
                context.enter_context(mock.patch.object(policy, "candidate_cohort", return_value=cohort))
                context.enter_context(mock.patch.object(policy, "_check_protocol", return_value=protocol))
                with self.assertRaisesRegex(RuntimeError, "explicit cohort|differs from qualified"):
                    policy.checked_correctness(binding, self.root / "unused-policy.json")

    def actual_release_spec(self):
        source = ROOT / "docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/core-correctness/core_collect.py"
        descriptor = importlib.util.spec_from_file_location("policy_test_actual_release_core", source)
        core = importlib.util.module_from_spec(descriptor); descriptor.loader.exec_module(core)
        cohort = policy.candidate_cohort()
        identities = {name: cohort[name] for name in ("numerical_revision", "generation_revision",
            "generator_identity_sha256", "runtime_identity_sha256", "archive_sha256")}
        output = self.root / "actual-collector-spec.json"
        # Only the runtime inspection is replaced. Serialize through the actual
        # collector so this test catches cross-component schema mismatches.
        with mock.patch.object(core, "cohort", return_value={"candidate_cohort": identities}):
            core.write_spec(SimpleNamespace(report=self.root / "report", work=self.root / "work", output=output))
        return json.loads(output.read_text()), {"candidate_cohort": identities}, cohort

    def test_actual_release_collector_spec_matches_reader_without_top_level_spec_ids(self):
        spec, protocol, cohort = self.actual_release_spec()
        self.assertNotIn("candidate_cohort", spec)
        self.assertTrue(policy._check_correctness_release(spec, protocol, {"candidate_cohort": {"bound": True}}, cohort))

    def test_actual_release_collector_missing_nested_or_contradictory_ids_rejected(self):
        good, protocol, cohort = self.actual_release_spec()
        for mutation in ("missing-nested", "missing-id", "wrong-nested", "wrong-top-level", "wrong-protocol", "finalized"):
            spec, observed = copy.deepcopy(good), copy.deepcopy(protocol)
            if mutation == "missing-nested": spec["cohort"].pop("candidate_cohort")
            elif mutation == "missing-id": spec["cohort"]["candidate_cohort"].pop("archive_sha256")
            elif mutation == "wrong-nested": spec["cohort"]["candidate_cohort"]["runtime_identity_sha256"] = "f" * 64
            elif mutation == "wrong-top-level": spec["candidate_cohort"] = {"archive_sha256": "f" * 64}
            elif mutation == "wrong-protocol": observed["candidate_cohort"]["generation_revision"] = "f" * 40
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                policy._check_correctness_release(policy.v2.seal(spec), observed,
                    {"candidate_cohort": {"bound": True}}, cohort, finalization=mutation == "finalized")

    def test_literal_card_parser_rejects_missing_values_and_wrong_or_false_selector(self):
        card = self.put(self.root / "card.dat", "#MLReductionLib\n8\n#UseQPIntegrandForNinjaRS\n.true.\n")
        self.assertEqual(policy.card_settings(card, 8), {"selected_backend_qp_integrand": True})
        for text in ("#MLReductionLib\n6\n#UseQPIntegrandForNinjaRS\n.true.\n",
                     "#MLReductionLib\n8\n#UseQPIntegrandForNinjaRS\n.false.\n",
                     "#MLReductionLib\n#UseQPIntegrandForNinjaRS\n.true.\n"):
            card.write_text(text)
            with self.assertRaises(RuntimeError): policy.card_settings(card, 8)

    def test_bound_file_change_and_repository_escape_are_rejected(self):
        filename = self.put(self.root / "evidence", "first")
        bound = policy.v2.bound_file(filename)
        filename.write_text("second")
        with self.assertRaisesRegex(RuntimeError, "Changed bound"):
            policy.checked(bound)
        with self.assertRaisesRegex(RuntimeError, "inside repository"):
            policy.retain("/etc/passwd")

    def test_selected_reference_failure_rejected_but_later_collier_failure_is_context(self):
        report = self.root / "report"
        failure = report / "rejected/qcd-4-failure.json"
        self.dump(failure, {"case": "qcd-4", "backend": 7, "stage": "calibration", "error": "410"})
        inputs = {}
        policy._selected_failure(report, "qcd-4", 1, inputs)
        self.assertIn(policy.v2.relative(failure), inputs)
        self.dump(failure, {"case": "qcd-4", "backend": 1, "error": "bad CT"})
        with self.assertRaisesRegex(RuntimeError, "Selected"):
            policy._selected_failure(report, "qcd-4", 1, {})

    def test_fixed_point_reader_checks_frozen_bytes_not_just_self_consistent_hash(self):
        frozen = self.root / "generation"
        points = self.put(self.root / "points.dat", "100\n" + "1 0 0 1\n" * 300)
        self.put(frozen / "points/qcd-1.dat", points.read_bytes())
        record = observation(); record["points_sha256"] = policy.workers.digest(points)
        with mock.patch.object(policy, "GENERATION_ROOT", frozen):
            policy._check_points(points, record, {"id": "qcd-1", "nfinal": 1}, {})
            points.write_text(points.read_text().replace("1 0 0 1", "2 0 0 2", 1))
            record["points_sha256"] = policy.workers.digest(points)
            with self.assertRaisesRegex(RuntimeError, "Changed bound"):
                policy._check_points(points, record, {"id": "qcd-1", "nfinal": 1}, {})

    def finalization(self):
        original, view = self.root / "original", self.root / "view"
        cases = configuration()["cases"]
        config = dict(configuration(), repetitions=2)
        self.dump(original / "run-status.json", {"status": "failed", "completed_cases": [c["id"] for c in cases],
            "error_type": "RuntimeError", "error": "Worker series is incomplete or failed"})
        self.dump(original / "protocol.json", {"configuration": config})
        self.dump(original / "input-spec.json", {"configuration": config})
        self.dump(original / "root-handoff.json", {"actual": "handoff"})
        self.put(original / "process-ledger.jsonl", "{}\n")
        self.put(original / "sources/worker.f", "retained source\n")
        self.put(original / "points/qcd-1.dat", "retained point bytes\n")
        self.put(original / "builds/qcd-1/build.json", "retained build\n")
        self.put(original / "source-proofs/qcd-1.json", "retained source proof\n")
        for case in cases:
            for rep in ("calibration", 0, 1):
                self.dump(original / "data" / f"{case['id']}-candidate-8-{rep}.json", {"actual": [case["id"], rep]})
        copies = []
        for source in sorted(original.rglob("*")):
            if not source.is_file() or source.name in ("run-status.json", "root-handoff.json"): continue
            destination = view / source.relative_to(original)
            self.put(destination, source.read_bytes())
            copies.append({"source": policy.v2.bound_file(source), "destination": policy.v2.bound_file(destination)})
        self.dump(view / "run-status.json", {"status": "complete", "requested_cases": [c["id"] for c in cases],
            "completed_cases": [c["id"] for c in cases]})
        self.dump(view / "source-bindings.json", {"actual": "bindings"})
        attempt = self.root / "launch-attempt"
        self.dump(attempt / "launch.json", {"specification": policy.v2.bound_file(original / "input-spec.json"),
            "handoff": policy.v2.bound_file(original / "root-handoff.json")})
        self.dump(attempt / "result.json", {"returncode": 1, "interruptions": []})
        self.put(attempt / "stdout", "retained stdout\n")
        self.put(attempt / "stderr", "retained metadata error\n")
        finalizer = self.put(self.root / "finalize.py", "# Synthetic finalizer fixture; not executed.\n")
        receipt = policy.v2.seal({"schema": policy.FINALIZATION,
            "original_failure_promoted": False, "fresh_parity": False,
            "numerical_retries": 0, "new_amplitude_evaluations": 0,
            "performance_acceptance_gate": False, "seven_repetition_timing_gate": False,
            "final_dataset_claim": False,
            "original_root_handoff": policy.v2.bound_file(original / "root-handoff.json"),
            "original_launch_record": policy.v2.bound_file(attempt / "launch.json"),
            "original_launcher_result": policy.v2.bound_file(attempt / "result.json"),
            "original_launcher_stdout": policy.v2.bound_file(attempt / "stdout"),
            "original_launcher_stderr": policy.v2.bound_file(attempt / "stderr"),
            "finalizer_source": policy.v2.bound_file(finalizer),
            "bindings": policy.v2.bound_file(view / "source-bindings.json"),
            "original_status": policy.v2.bound_file(original / "run-status.json"),
            "original_protocol": policy.v2.bound_file(original / "protocol.json"),
            "original_specification": policy.v2.bound_file(original / "input-spec.json"),
            "view_status": policy.v2.bound_file(view / "run-status.json"), "copies": copies})
        return original, view, receipt

    def test_metadata_only_finalization_preserves_failed_original(self):
        original, view, receipt = self.finalization()
        before = (original / "run-status.json").read_bytes()
        bound, actual = policy._check_finalization(receipt, {})
        self.assertEqual(actual, view)
        self.assertEqual(policy.checked(bound), view / "source-bindings.json")
        self.assertEqual(before, (original / "run-status.json").read_bytes())
        self.assertEqual(json.loads(before)["status"], "failed")

    def test_finalization_missing_extra_remapped_or_changed_copy_rejected(self):
        _, _, good = self.finalization()
        for mutation in ("missing", "extra", "remap", "changed"):
            receipt = copy.deepcopy(good)
            if mutation == "missing": receipt["copies"].pop()
            elif mutation == "extra": receipt["copies"].append(receipt["copies"][0])
            elif mutation == "remap": receipt["copies"][0]["destination"] = receipt["copies"][1]["destination"]
            else: receipt["copies"][0]["source"]["sha256"] = "0" * 64
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                policy._check_finalization(policy.v2.seal(receipt), {})

    def test_unrelated_status_protocol_or_uncompleted_original_cannot_be_finalized(self):
        original, _, good = self.finalization()
        for field in ("original_protocol", "original_specification", "original_status"):
            other = self.put(self.root / (field + ".json"), policy.checked(good[field]).read_bytes())
            value = copy.deepcopy(good); value[field] = policy.v2.bound_file(other)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                policy._check_finalization(policy.v2.seal(value), {})
        status = json.loads((original / "run-status.json").read_text()); status["completed_cases"].pop()
        self.dump(original / "run-status.json", status)
        good["original_status"] = policy.v2.bound_file(original / "run-status.json")
        with self.assertRaisesRegex(RuntimeError, "all actual"):
            policy._check_finalization(policy.v2.seal(good), {})

    def test_finalization_cannot_invent_retry_fresh_or_acceptance_claims(self):
        _, _, receipt = self.finalization()
        for field in ("original_failure_promoted", "fresh_parity", "performance_acceptance_gate",
                      "seven_repetition_timing_gate", "final_dataset_claim", "numerical_retries", "new_amplitude_evaluations"):
            altered = dict(receipt, **{field: True if isinstance(receipt[field], bool) else 1})
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "metadata-only"):
                policy._check_finalization(policy.v2.seal(altered), {})

    def test_original_failed_launch_logs_and_result_remain_required(self):
        _, _, receipt = self.finalization()
        for field in ("original_launcher_stdout", "original_launcher_stderr", "original_launch_record",
                      "original_launcher_result", "original_root_handoff"):
            filename = policy.checked(receipt[field]); original = filename.read_bytes()
            filename.write_text("changed after finalization")
            with self.subTest(field=field), self.assertRaisesRegex(RuntimeError, "Changed bound"):
                policy._check_finalization(receipt, {})
            filename.write_bytes(original)
        result = policy.checked(receipt["original_launcher_result"])
        for bad in ({"returncode": 0, "interruptions": []}, {"returncode": 1, "interruptions": ["signal"]}):
            self.dump(result, bad)
            receipt["original_launcher_result"] = policy.v2.bound_file(result)
            with self.assertRaisesRegex(RuntimeError, "interrupted"):
                policy._check_finalization(policy.v2.seal(receipt), {})

    def supplement_target(self):
        directory = self.root / "target"
        case = configuration()["cases"][0]
        card = self.put(directory / "param_card.dat", "fixed physical parameters\n")
        physical_sha = policy.workers.digest(card)
        driver = (ROOT / "tools/ninjars_benchmark_worker.f").read_text().replace("@PREFIX@", "P0_").replace("@NEXTERNAL@", "3").replace("@NBORN@", "1")
        fresh = (ROOT / "tools/ninjars_benchmark_driver.f").read_text().replace("@PREFIX@", "P0_").replace("@NEXTERNAL@", "3").replace("@NBORN@", "1")
        self.put(directory / "worker.f", driver)
        origin_files = {"SubProcesses/P0/check_sa.f": "original-driver", "Source/MODEL/couplings.f": "frozen-model"}
        origin = {"files": origin_files, "files_sha256": policy.bench.content_digest(origin_files)}
        files = dict(origin_files, **{"SubProcesses/P0/check_sa.f": policy.workers.digest(directory / "worker.f")})
        compiled = {"files": files, "files_sha256": policy.bench.content_digest(files)}
        self.dump(directory / "compiled-sources.json", compiled)
        tree = {"tracked": {}, "untracked": {}}
        tree_sha = policy.bench.content_digest(tree)
        identity = {"revision": policy.GENERATION_REVISION, "tree_state": tree, "tree_state_sha256": tree_sha,
                    "dirty_patch_sha256": None}
        generation = {"source_identity": identity, "parameter_card_sha256": physical_sha}
        compilation = {"generated_sources_sha256": origin["files_sha256"]}
        origin_path = self.dump(directory / "origin-manifest.json", origin)
        generation_path = self.dump(directory / "generation.json", generation)
        compilation_path = self.dump(directory / "compilation.json", compilation)
        build = {"generated_sources_sha256": compiled["files_sha256"], "original_generated_sources_sha256": origin["files_sha256"],
                 "generation_source_identity": identity, "parameter_card_sha256": physical_sha,
                 "executable_sha256": "executable", "fresh_executable_sha256": "actual-fresh",
                 "driver_sha256": policy.workers.digest(directory / "worker.f"),
                 "fresh_driver_sha256": hashlib.sha256(fresh.encode()).hexdigest(), "origin_compilation": compilation}
        build_path = self.dump(directory / "build.json", build)
        source_proof = {"case": case, "full_runtime_identity_sha256": policy.RUNTIME_ID,
                        "generation": policy.v2.bound_file(generation_path), "compilation": policy.v2.bound_file(compilation_path),
                        "compiled_manifest": policy.v2.bound_file(origin_path), "physical_card": policy.v2.bound_file(card)}
        source_path = self.dump(directory / "source-proof.json", source_proof)
        runtime = {"identity_sha256": policy.RUNTIME_ID}
        item = {"role": "candidate_corrected", "case": case["id"], "source": "candidate", "backend": 8,
                "build": policy.v2.bound_file(build_path), "compiled_manifest": compiled, "generation_source_identity": identity,
                "executable_sha256": build["executable_sha256"], "generated_sources_sha256": compiled["files_sha256"],
                "origin_manifest": policy.v2.bound_file(origin_path), "origin_generation": policy.v2.bound_file(generation_path),
                "origin_compilation": policy.v2.bound_file(compilation_path), "physical_card": policy.v2.bound_file(card),
                "source_proof": policy.v2.bound_file(source_path), "runtime_metadata": {}, "reducer_runtime": runtime,
                "libraries": {"libninjars.a": policy.ARCHIVE_SHA256}}
        frozen = self.root / "frozen"
        self.dump(frozen / "inputs/candidate" / case["id"] / "compiled-sources.json", origin)
        correction = {"cases": {case["id"]: {"case_definition": policy.v2.physical_definition(case)}}}
        context = ExitStack(); self.addCleanup(context.close)
        context.enter_context(mock.patch.object(policy, "GENERATION_ROOT", frozen))
        context.enter_context(mock.patch.object(policy, "GENERATOR_ID", tree_sha))
        # Existing v2 independently tests source-role/runtime proofs; here the
        # real v3 build/driver/card/model/schema checks operate on small files.
        context.enter_context(mock.patch.object(policy.v2, "validate_source_role"))
        context.enter_context(mock.patch.object(policy.v2, "reducer_signature", return_value=runtime))
        context.enter_context(mock.patch.object(policy.v2, "proof_runtime_signatures", return_value={
            "candidate_corrected/8": {policy.RUNTIME_ID: {"signature": runtime}}}))
        return item, build, build_path, correction

    def test_only_actual_supplement_v1_shape_may_omit_target_fresh_field(self):
        item, build, path, correction = self.supplement_target()
        actual, _ = policy._check_target(item, correction, {"entries": []}, {}, "supplement-source-bindings-v1")
        self.assertEqual(actual["fresh_executable_sha256"], "actual-fresh")
        self.assertNotIn("fresh_executable_sha256", item)
        with self.assertRaisesRegex(RuntimeError, "Missing fresh"):
            policy._check_target(item, correction, {"entries": []}, {})
        item["fresh_executable_sha256"] = "invented-fresh"
        with self.assertRaisesRegex(RuntimeError, "Fresh executable"):
            policy._check_target(item, correction, {"entries": []}, {}, "supplement-source-bindings-v1")

    def test_supplement_source_model_card_and_driver_drift_rejected(self):
        item, build, path, correction = self.supplement_target()
        before = copy.deepcopy(item)
        for mutation in ("executable", "driver", "model", "card", "source"):
            with self.subTest(mutation=mutation):
                item = copy.deepcopy(before); altered = copy.deepcopy(build)
                card = policy.checked(item["physical_card"])
                original_card = card.read_bytes()
                if mutation == "executable": item["executable_sha256"] = "other"
                elif mutation == "driver": altered["fresh_driver_sha256"] = "other"
                elif mutation == "card": card.write_text("different physical input\n")
                elif mutation == "source":
                    item["generation_source_identity"]["revision"] = "other"
                    altered["generation_source_identity"] = item["generation_source_identity"]
                else:
                    compiled = item["compiled_manifest"]
                    compiled["files"]["Source/MODEL/couplings.f"] = "changed-model"
                    compiled["files_sha256"] = policy.bench.content_digest(compiled["files"])
                    item["generated_sources_sha256"] = altered["generated_sources_sha256"] = compiled["files_sha256"]
                self.dump(path.parent / "compiled-sources.json", item["compiled_manifest"])
                self.dump(path, altered); item["build"] = policy.v2.bound_file(path)
                with self.assertRaises(RuntimeError):
                    policy._check_target(item, correction, {"entries": []}, {}, "supplement-source-bindings-v1")
                card.write_bytes(original_card)

    def test_changed_literal_denominator_vetoes_performance_without_erasing_nrs_correctness(self):
        config = configuration()
        names = {name: self.dump(self.root / (name + ".json"), {}) for name in ("policy", "references", "pair", "measurement")}
        old = {"configuration": policy.v2.bound_file(names["policy"]), "correction": policy.v2.bound_file(names["policy"]),
               "representative_proof": policy.v2.bound_file(names["policy"])}
        refs = {(cid, backend): observation(backend, cid) for cid in policy.CASE_IDS for backend in (1, 6)}
        timed = {(cid, source, backend, rep): (names["measurement"], observation(backend, cid, source, rep))
                 for cid in policy.CASE_IDS for source, backend in policy.PAIR_TARGETS for rep in range(7)}
        binding = {"report": str(self.root), "targets": {f"{cid}-{source}-{backend}": {
            "case": cid, "role": policy.ROLE_MAP[source], "backend": backend}
            for cid in policy.CASE_IDS for source, backend in policy.PAIR_TARGETS}}
        correction = {"cases": {cid: {"physical_cards": [{"sha256": "2" * 64}]} for cid in policy.CASE_IDS}}
        with ExitStack() as context:
            context.enter_context(mock.patch.object(policy, "validate_policy", return_value=(old, correction, config, {})))
            context.enter_context(mock.patch.object(policy, "checked_references", return_value=({}, refs, {})))
            context.enter_context(mock.patch.object(policy.v2, "checked_bindings", return_value=binding))
            context.enter_context(mock.patch.object(policy, "checked_pair_series", return_value=({}, timed)))
            context.enter_context(mock.patch.object(policy, "_check_target", return_value=({}, names["measurement"])))
            context.enter_context(mock.patch.object(policy, "_check_record_target"))
            context.enter_context(mock.patch.object(policy.bench, "runtime_statistics", return_value={"status": "passed", "passed": True, "ratio": .9}))
            passed = policy.build_dataset(names["policy"], names["references"], pair_bindings_path=names["pair"])
            self.assertTrue(passed["nrs_core_validation_passed"])
            changed = timed["qcd-4", "baseline", 6, 6][1]
            changed["rows"][-1]["coefficients"][1] += 1e-4
            policy.validate_rows(changed, 6)
            failed = policy.build_dataset(names["policy"], names["references"], pair_bindings_path=names["pair"])
            self.assertTrue(failed["nrs_correctness_passed"])
            self.assertFalse(failed["nrs_performance_passed"])
            self.assertFalse(failed["nrs_core_validation_passed"])
            self.assertTrue(failed["cases"]["qcd-4"]["performance"]["passed"])
            self.assertFalse(failed["cases"]["qcd-4"]["timing_agreement"]["passed"])


class PairSeriesTest(PrivateFilesMixin, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.report = self.root / "pair"
        self.config = configuration(); self.config["cases"] = [self.config["cases"][0]]
        self.case = self.config["cases"][0]["id"]
        self.ledger = []
        descriptor = policy.workers.protocol_descriptor()
        for name in descriptor["sources"]:
            self.put(self.report / "sources" / name, (ROOT / "tools" / name).read_bytes())
        collector = ROOT / "tools/ninjars_benchmark_pair_v3.py"
        self.put(self.report / "sources" / collector.name, collector.read_bytes())
        proof = json.loads((ROOT / "docs/benchmarkfolder/policy-v2/final-representative-v1/representative-proof.json").read_text())
        runtime = proof["reducer_runtimes"]["candidate_corrected/8"][policy.RUNTIME_ID]["signature"]
        config_file = self.dump(self.root / "config.json", self.config)
        self.spec = policy.v2.seal({"schema": policy.PAIR_SPEC, "configuration": policy.v2.bound_file(config_file),
            "pair_targets": policy.PAIR_TARGETS, "role_map": policy.ROLE_MAP, "cpu": 207, "repetitions": 7,
            "minimum_seconds": 1., "timeout": 10., "run_id": "test-pair",
            "candidate_generator_identity_sha256": policy.GENERATOR_ID, "candidate_runtime_signature": runtime,
            "literal_baseline_revision": policy.bench.BASELINE_REVISION, "production_sources": {"fixture": "bound"},
            "sources": {"collector": policy.v2.bound_file(collector), **{n: policy.v2.bound_file(ROOT / "tools" / n) for n in descriptor["sources"]}}})
        self.dump(self.report / "input-spec.json", self.spec)
        self.protocol = {"schema": policy.workers.SCHEME, "pair_schema": policy.PAIR_SCHEMA,
            "pair_targets": copy.deepcopy(policy.PAIR_TARGETS), "role_map": policy.ROLE_MAP,
            "configuration": self.config, "cpu": 207, "repetitions": 7, "sources": "both", "backends": [6, 8],
            "protocol_descriptor": descriptor, "protocol_identity_sha256": policy.bench.content_digest(descriptor),
            "input_spec": policy.v2.bound_file(self.report / "input-spec.json"),
            "input_adapter": policy.v2.bound_file(self.report / "sources" / collector.name), "fresh_parity": False,
            "thread_environment": {n: "1" for n in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS")}}
        for source, backend in policy.PAIR_TARGETS:
            for phase in ("prepare", "initialize"):
                self.event(phase, source, backend)
            self.add_record(source, backend, "calibration", "calibration", 1)
        for rep in range(7):
            for source, backend in (policy.PAIR_TARGETS if rep % 2 == 0 else list(reversed(policy.PAIR_TARGETS))):
                self.add_record(source, backend, rep, "measurement", rep + 2)
        for source, backend in reversed(policy.PAIR_TARGETS): self.event("cleanup", source, backend)
        self.save_ledger()
        self.dump(self.report / "run-status.json", {"status": "complete", "requested_cases": [self.case], "completed_cases": [self.case]})
        patcher = mock.patch.object(policy, "validate_configuration", return_value=self.config)
        patcher.start(); self.addCleanup(patcher.stop)

    def event(self, phase, source, backend):
        for event in ("start", "complete"):
            self.ledger.append({"index": len(self.ledger), "case": self.case, "phase": phase, "event": event,
                "source": source, "backend": backend, "repetition": None, "loops": None, "worker_sequence": None})

    def raw(self, record, suffix):
        directory = self.report / "raw" / suffix
        self.put(directory / "dimensions.dat", "1 1 1\n")
        self.put(directory / "timing.dat", ".1 1\n")
        self.put(directory / "values.dat", "".join(" ".join(map(str, [row["point"], row["order"], row["status"],
            *row["coefficients"], row["accuracy"]])) + "\n" for row in record["rows"]))
        self.put(directory / "timed_status.dat", "".join(f"{code} {count}\n" for code, count in record["timed_status_counts"].items()))
        return {name: policy.v2.bound_file(directory / name) for name in (*policy.workers.OUTPUTS, "dimensions.dat")}

    def add_record(self, source, backend, rep, phase, sequence):
        record = observation(backend, self.case, source, rep)
        record.update(protocol_identity_sha256=self.protocol["protocol_identity_sha256"], worker_sequence=sequence, dispatch_index=len(self.ledger))
        prefix = f"{self.case}-{source}-{backend}"
        record["raw_capture"] = self.raw(record, f"{prefix}-{rep}")
        filename = self.dump(self.report / "data" / f"{prefix}-{rep}.json", record)
        self.dump(self.report / "builds" / prefix / "build.json", {"executable_sha256": record["executable_sha256"],
            "generated_sources_sha256": record["generated_sources_sha256"]})
        for event in ("start", "complete"):
            value = {"index": len(self.ledger), "case": self.case, "phase": phase, "event": event,
                "source": source, "backend": backend, "repetition": rep, "loops": 1, "worker_sequence": sequence}
            if event == "complete": value["record"] = policy.v2.bound_file(filename)
            self.ledger.append(value)

    def save_ledger(self):
        self.put(self.report / "dispatch.jsonl", "".join(json.dumps(e) + "\n" for e in self.ledger))
        self.protocol["dispatch_ledger"] = policy.v2.bound_file(self.report / "dispatch.jsonl")
        self.dump(self.report / "protocol.json", self.protocol)

    def test_complete_explicit_pair_is_read_without_v2_cartesian_product(self):
        protocol, records = policy.checked_pair_series(self.report)
        self.assertEqual(len(records), 14)
        self.assertEqual({(key[1], key[2]) for key in records}, {("baseline", 6), ("candidate", 8)})
        self.assertEqual(protocol["repetitions"], 7)

    def test_pair_missing_repetition_calibration_or_extra_target_rejected(self):
        for mutation in ("missing-rep", "extra-target", "bad-calibration"):
            with self.subTest(mutation=mutation):
                target = self.report / "data" / f"{self.case}-candidate-8-0.json"
                original = target.read_bytes()
                extra = self.report / "data" / f"{self.case}-candidate-6-0.json"
                calibration = self.report / "data" / f"{self.case}-candidate-8-calibration.json"
                old_calibration = calibration.read_bytes()
                if mutation == "missing-rep": target.unlink()
                elif mutation == "extra-target": extra.write_bytes(original)
                else:
                    item = json.loads(old_calibration); item["repetition"] = 0; self.dump(calibration, item)
                    event = next(e for e in self.ledger if e["phase"] == "calibration" and e["event"] == "complete" and e["source"] == "candidate")
                    event["record"] = policy.v2.bound_file(calibration); self.save_ledger()
                with self.assertRaises((RuntimeError, FileNotFoundError)):
                    policy.checked_pair_series(self.report)
                target.write_bytes(original); extra.unlink(missing_ok=True); calibration.write_bytes(old_calibration)
                for event in self.ledger:
                    if event["phase"] == "calibration" and event["event"] == "complete" and event["source"] == "candidate":
                        event["record"] = policy.v2.bound_file(calibration)
                self.save_ledger()

    def test_wrong_dispatch_order_or_unclosed_event_rejected(self):
        original = copy.deepcopy(self.ledger)
        starts = [i for i, e in enumerate(self.ledger) if e["phase"] == "measurement" and e["event"] == "start"]
        self.ledger[starts[0]]["source"] = "candidate"
        self.save_ledger()
        with self.assertRaises(RuntimeError): policy.checked_pair_series(self.report)
        self.ledger = original[:-1]; self.save_ledger()
        with self.assertRaises(RuntimeError): policy.checked_pair_series(self.report)

    def test_thread_cpu_parity_and_collector_are_independently_checked(self):
        original = copy.deepcopy(self.protocol)
        for field, value in (("thread_environment", {}), ("cpu", 4), ("fresh_parity", True),
                             ("input_adapter", policy.v2.bound_file(ROOT / "tools/ninjars_benchmark.py"))):
            self.protocol = dict(copy.deepcopy(original), **{field: value}); self.save_ledger()
            with self.subTest(field=field), self.assertRaises(RuntimeError): policy.checked_pair_series(self.report)
        self.protocol = original

    def test_raw_capture_tamper_rejected_even_if_json_and_ledger_are_self_consistent(self):
        filename = self.report / "data" / f"{self.case}-candidate-8-0.json"
        record = json.loads(filename.read_text())
        path = policy.checked(record["raw_capture"]["values.dat"])
        path.write_text(path.read_text().replace("1 0 218 1.0", "1 0 218 1.1", 1))
        record["raw_capture"]["values.dat"] = policy.v2.bound_file(path)
        self.dump(filename, record)
        end = next(e for e in self.ledger if e["phase"] == "measurement" and e["event"] == "complete" and e["source"] == "candidate" and e["repetition"] == 0)
        end["record"] = policy.v2.bound_file(filename); self.save_ledger()
        with self.assertRaisesRegex(RuntimeError, "Raw output differs"):
            policy.checked_pair_series(self.report)


if __name__ == "__main__":
    unittest.main()
