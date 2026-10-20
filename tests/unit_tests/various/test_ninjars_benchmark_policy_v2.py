"""Policy failures are exercised without physics evaluations or timing builds."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_references_v2 as references


class PolicyV2Test(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="policy-v2-test-", dir=ROOT / ".benchmark-work")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.identity = {"revision": policy.bench.BASELINE_REVISION,
                         "status": "recorded-before-generation",
                         "verification": "provenance.json:baseline_source_integrity"}
        self.before = self.root / "before"
        self.after = self.root / "after"
        self.original = self.root / "original"
        for output in (self.original, self.before, self.after):
            self.put(output / "Source/MODEL/model_functions.f", "original helpers\n")
            self.put(output / "Source/MODEL/mp_intparam_definition.inc", "original MP constants\n")
            self.put(output / "Source/MODEL/other.f", "unchanged model\n")
            self.put(output / "Source/make_opts", "FC=gfortran\n")
            self.put(output / "SubProcesses/P0_test/matrix.f", "original process\n")
            self.put(output / "SubProcesses/P0_test/check_sa.f", "fresh driver\n")
            self.put(output / "SubProcesses/MadLoop_makefile_definitions", "same numerical linker flags\n")
            self.put(output / "Cards/param_card.dat", "same physics\n")
            self.dump(output / "generation.json", {"source_identity": self.identity})
        self.put(self.after / policy.MODEL_FILES[0], "corrected helpers\n")
        self.put(self.after / policy.MODEL_FILES[1], "corrected MP constants\n")
        generator_before, generator_after = {}, {}
        for name in ("madgraph/iolibs/export_v4.py", "madgraph/iolibs/ufo_expression_parsers.py"):
            a, b = self.root / "generator-before" / name, self.root / "generator-after" / name
            self.put(a, "old\n"); self.put(b, "correct\n")
            generator_before[name], generator_after[name] = str(a), str(b)
        evidence = self.root / "helper-test.json"
        self.dump(evidence, {"fixture_only": True})
        self.spec = {"generator_before": generator_before, "generator_after": generator_after,
                     "evidence": [str(evidence)], "cases": {"toy": {
                         "original_output": str(self.original), "fresh_before_output": str(self.before),
                         "fresh_after_output": str(self.after), "ufo_evidence": str(evidence),
                         "cli_evidence": [str(evidence)]}}}
        tree = {"tracked": {name: policy.workers.digest(Path(value)) for name, value in generator_after.items()}, "untracked": {}}
        after_identity = policy.seal({"status": "recorded-before-generation", "revision": "corrected-fixture",
                                      "tree_state": tree, "tree_state_sha256": policy.bench.content_digest(tree),
                                      "dirty_patch_sha256": None})
        self.dump(self.after / "generation.json", {"source_identity": after_identity})
        self.dump(self.after / "compilation.json", {
            "source_identity": after_identity,
            "generated_sources_sha256": policy.manifest(self.after)["files_sha256"],
            "source_manifest_recording": "recorded-before-and-verified-after-build"})
        anchored = {"generated-sources-observed.json": json.dumps(policy.manifest(self.original)).encode(),
                    "generation.json": (self.original / "generation.json").read_bytes(),
                    "param_card.dat": (self.original / "Cards/param_card.dat").read_bytes()}
        def git_bytes(revision, name):
            if revision == policy.ORIGINAL_EVIDENCE_REVISION:
                if name.endswith("/cases.json"):
                    return json.dumps(self.config).encode()
                return anchored[Path(name).name]
            sources = generator_before if revision == policy.bench.BASELINE_REVISION else generator_after
            return Path(sources[name]).read_bytes()
        patcher = mock.patch.object(policy, "git_bytes", side_effect=git_bytes)
        patcher.start(); self.addCleanup(patcher.stop)
        self.config = {"seed": 410083, "points_per_case": 100, "repetitions": 7,
                       "relative_tolerance": 1e-7, "scale_tolerance": 1e-10, "max_runtime_ratio": 1.05,
                       "madloop_parameters": {"MLStabThres": 1e-9, "COLLIERRequiredAccuracy": 1e-12},
                       "cases": [{"id": "toy", "nfinal": 1, "process": "u u~ > z [virt=QCD]"}]}
        self.correction_path = self.root / "correction.json"
        self.correction = policy.create_correction(self.spec, self.correction_path)
        self.config_path = self.root / "config.json"
        self.dump(self.config_path, self.config)
        self.policy_path = self.root / "policy.json"
        policy.create_policy(self.correction_path, self.config_path, self.policy_path)

    @staticmethod
    def put(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    @staticmethod
    def dump(path, value):
        policy.bench.dump(path, value)

    def test_retained_exact_correction_survives_disposable_fresh_tree(self):
        self.assertEqual(policy.validate_correction(self.correction_path), self.correction)
        self.put(self.after / policy.MODEL_FILES[0], "disposable output changed\n")
        self.assertEqual(policy.validate_correction(self.correction_path), self.correction)
        retained = self.correction["cases"]["toy"]["postimages"][policy.MODEL_FILES[0]]
        self.put(policy.path(retained["path"]), "corruption\n")
        with self.assertRaisesRegex(RuntimeError, "Changed bound"):
            policy.validate_correction(self.correction_path)

    def test_full_delta_rejects_third_file_deleted_or_extra_source(self):
        before = policy.manifest(self.before)
        expected = {name: policy.manifest(self.after)["files"][name] for name in policy.MODEL_FILES}
        for mutation in ("third", "delete", "extra", "wrong-postimage", "missing-correction"):
            after = copy.deepcopy(policy.manifest(self.after))
            if mutation == "third":
                after["files"]["Source/MODEL/other.f"]["sha256"] = "bad"
            elif mutation == "delete":
                del after["files"]["Source/MODEL/other.f"]
            elif mutation == "extra":
                after["files"]["Source/MODEL/new.f"] = {"sha256": "extra", "bytes": 5}
            elif mutation == "wrong-postimage":
                after["files"][policy.MODEL_FILES[0]]["sha256"] = "arbitrary replacement"
            else:
                after["files"][policy.MODEL_FILES[1]] = before["files"][policy.MODEL_FILES[1]]
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                policy.verify_delta(before, after, expected)

    def test_original_parent_changed_after_sealing_rejected(self):
        self.put(self.original / "SubProcesses/P0_test/matrix.f", "candidate optimization\n")
        with self.assertRaisesRegex(RuntimeError, "parent sources changed"):
            policy.validate_correction(self.correction_path, live=True)

    def test_original_and_corrected_roles_cannot_exchange_model_or_process(self):
        parent = self.correction["cases"]["toy"]["parent_manifest"]
        corrected = self.correction["cases"]["toy"]["fresh_after_manifest"]
        policy.validate_source_role("original", "toy", parent, self.identity, self.correction)
        policy.validate_source_role("original_helper_corrected", "toy", corrected, self.identity, self.correction)
        policy.validate_source_role("candidate_corrected", "toy", corrected, {}, self.correction)
        for role, source in (("original", corrected), ("original_helper_corrected", parent),
                             ("candidate_corrected", parent)):
            with self.subTest(role=role), self.assertRaises(RuntimeError):
                policy.validate_source_role(role, "toy", source, self.identity, self.correction)
        spoof = copy.deepcopy(corrected)
        spoof["files"]["SubProcesses/P0_test/matrix.f"]["sha256"] = "candidate"
        with self.assertRaisesRegex(RuntimeError, "non-model/non-driver"):
            policy.validate_source_role("original_helper_corrected", "toy", spoof, self.identity, self.correction)
        with self.assertRaisesRegex(RuntimeError, "original archive"):
            policy.validate_source_role("original", "toy", parent, {"revision": "current"}, self.correction)

    def test_model_write_cannot_escape_isolated_copy(self):
        shared = self.root / "shared.f"
        self.put(shared, "never mutate")
        link = self.before / "Source/MODEL/link.f"
        link.symlink_to(shared)
        with self.assertRaisesRegex(RuntimeError, "shared/external"):
            references.local_write(link, self.before)
        self.assertEqual(shared.read_text(), "never mutate")

    def test_missing_required_config_or_relaxed_case_threshold_rejected(self):
        policy.validate_config(self.config)
        for key, value in (("points_per_case", 99), ("relative_tolerance", 1e-6),
                           ("scale_tolerance", 1e-9), ("max_runtime_ratio", 1.1), ("repetitions", 6)):
            with self.subTest(key=key), self.assertRaises(RuntimeError):
                policy.validate_config(dict(self.config, **{key: value}))
        changed = copy.deepcopy(self.config)
        changed["cases"][0]["madloop_parameters"] = {"COLLIERRequiredAccuracy": 1e-10}
        with self.assertRaisesRegex(RuntimeError, "overrides"):
            policy.validate_config(changed)

    def test_policy_rejects_changed_denominator_even_if_resealed(self):
        data = policy.read(self.policy_path)
        data["performance_pair"][0] = "original_helper_corrected/6"
        self.dump(self.policy_path, policy.seal(data))
        with self.assertRaisesRegex(RuntimeError, "denominator"):
            policy.validate_policy(self.policy_path)

    def test_old_or_changed_representative_proof_rejected(self):
        proof = self.root / "proof.json"
        self.dump(proof, policy.seal({"schema": "persistent-worker-v1"}))
        with self.assertRaisesRegex(RuntimeError, "representative-policy-validation-v2"):
            policy.validate_proof(proof, self.policy_path)
        value = policy.seal({"schema": policy.PROOF, "policy_identity_sha256": "old",
                             "correction_identity_sha256": self.correction["identity_sha256"]})
        self.dump(proof, value)
        with self.assertRaisesRegex(RuntimeError, "different policy"):
            policy.validate_proof(proof, self.policy_path)

    @staticmethod
    def rows(value=1.):
        return [{"point": 1, "order": 0, "coefficients": [value, .2, .03, -.04]}]

    def test_all_corrected_pairs_required_original_failures_stay_visible(self):
        corrected = {f"candidate_corrected/{b}": self.rows() for b in (1, 6, 7, 8)}
        corrected.update({f"original_helper_corrected/{b}": self.rows() for b in (1, 6, 7)})
        result = policy.compare_rows(corrected)
        self.assertTrue(result["passed"])
        self.assertEqual(result["pairs"], 21)
        all_rows = dict(corrected, **{"original/6": self.rows(1000.)})
        diagnostic = policy.compare_rows(all_rows, [("original/6", "candidate_corrected/8")])
        self.assertFalse(diagnostic["passed"])
        self.assertEqual(len(diagnostic["failures"]), 1)
        corrected["original_helper_corrected/7"] = self.rows(1.01)
        self.assertFalse(policy.compare_rows(corrected)["passed"])

    def test_wrong_original_magnitude_cannot_inflate_corrected_scale(self):
        corrected = {"candidate/8": self.rows(), "reference/6": self.rows(1.0001)}
        self.assertFalse(policy.compare_rows(corrected)["passed"])
        # Original is deliberately absent from the corrected-cohort comparison.
        original_diagnostic = policy.compare_rows({**corrected, "original/6": self.rows(1e20)},
                                                   [("original/6", "candidate/8")])
        self.assertFalse(original_diagnostic["passed"])

    def test_comparison_rejects_missing_rows_and_nonfinite_values(self):
        with self.assertRaisesRegex(RuntimeError, "streams"):
            policy.compare_rows({"a": self.rows(), "b": []})
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            policy.compare_rows({"a": self.rows(), "b": self.rows(float("nan"))})

    def test_forced_rebuild_removes_stale_model_and_process_objects(self):
        work = self.root / "rebuild"
        target = work / "outputs/baseline/toy"
        references.clone_output(self.original, target)
        self.put(target / "lib/libmodel.a", "stale model")
        self.put(target / "lib/libdhelas.a", "unchanged dhelas")
        self.put(target / "Source/MODEL/makeinc.inc", "MODEL = couplings.o model_functions.o")
        for name in ("couplings.o", "model_functions.o"):
            self.put(target / "Source/MODEL" / name, "stale object")
        self.put(target / "SubProcesses/P0_test/check", "stale executable")
        self.put(target / "SubProcesses/P0_test/matrix.o", "stale object")
        report = self.root / "rebuild-report"
        def run(command, cwd, log):
            self.assertFalse((target / "lib/libmodel.a").exists())
            self.assertFalse(list((target / "Source/MODEL").glob("*.o")))
            for name in ("couplings.o", "model_functions.o"):
                self.put(target / "Source/MODEL" / name, "rebuilt object")
            self.put(target / "lib/libmodel.a", "rebuilt model")
            self.put(log, "exact make command")
            return .01
        def build(source, case):
            self.assertFalse((target / "SubProcesses/P0_test/check").exists())
            self.assertFalse(list((target / "SubProcesses/P0_test").glob("*.o")))
            self.put(target / "SubProcesses/P0_test/check", "rebuilt executable")
            self.put(work / "logs/compile-baseline-toy.log", "link command")
        with mock.patch.object(references.bench, "run", side_effect=run), mock.patch.object(references.bench, "build", side_effect=build):
            result = references.force_rebuild(target, self.config["cases"][0], self.correction["identity_sha256"], work, report)
        self.assertFalse(result["prebuild_model_objects"])
        self.put(target / "lib/libmodel.a", "stale archive restored")
        with self.assertRaisesRegex(RuntimeError, "Stale model archive"):
            policy.verify_rebuild(target / "correction-build.json", target, self.correction["identity_sha256"])

    def test_input_adapter_resolves_outer_inputs_but_keeps_inner_clones_private(self):
        outer = self.root / "sentinel"
        inner = self.root / "job"
        original_function = policy.bench.output_dir
        roots = {"baseline": self.root / "original-root", "candidate": self.root / "candidate-root"}
        with policy.workers.isolated_harness(outer, self.root / "report"), references.input_roots(roots, outer):
            self.assertEqual(policy.bench.output_dir("baseline", {"id": "toy"}), roots["baseline"] / "toy")
            with policy.workers.isolated_harness(inner, self.root / "inner-report"):
                self.assertEqual(policy.bench.output_dir("baseline", {"id": "toy"}), inner / "outputs/baseline/toy")
        self.assertIs(policy.bench.output_dir, original_function)

    def make_series(self, name, sources, backends, repetitions):
        report = self.root / name
        descriptor = policy.workers.protocol_descriptor()
        identity = policy.bench.content_digest(descriptor)
        for filename in policy.workers.SOURCE_NAMES:
            self.put(report / "sources" / filename, (ROOT / "tools" / filename).read_text())
        self.dump(report / "protocol.json", {
            "protocol_descriptor": descriptor, "protocol_identity_sha256": identity,
            "repetitions": repetitions, "configuration": self.config,
            "input_adapter": policy.bound_file(ROOT / "tools/ninjars_benchmark_references_v2.py"),
            "sources": sources, "backends": backends})
        self.dump(report / "run-status.json", {"status": "complete", "requested_cases": ["toy"],
                                                "completed_cases": ["toy"]})
        bindings = {"report": str(report), "targets": {}}
        for source in (("baseline", "candidate") if sources == "both" else (sources,)):
            for backend in backends:
                if source == "baseline" and backend == 8:
                    continue
                key = f"toy-{source}-{backend}"
                role = "original_helper_corrected" if name == "refs" else (
                    "original" if source == "baseline" else "candidate_corrected")
                bindings["targets"][key] = self.runtime_item(role, backend)
                bindings["targets"][key].update(executable_sha256=key,
                    generation_source_identity={"tree_state_sha256": "frozen-final-candidate"})
                settings = {"collierrequiredaccuracy": 1e-12, "mlstabthres": 1e-9,
                            "colliercomputeuvpoles": True, "colliercomputeirpoles": True}
                if backend != 7:
                    settings["selected_backend_qp_integrand"] = True
                code = 210 + backend
                data = {"case": "toy", "source": source, "backend": backend,
                        "benchmark_protocol": policy.workers.SCHEME, "protocol_identity_sha256": identity,
                        "executable_sha256": key, "generated_sources_sha256": "sources",
                        "points_sha256": "same100points", "parameter_card_sha256": "samecard", "cpu": 0,
                        "shared_numerical_settings": settings,
                        "answer_dimensions": {"allocated": 1, "loop_orders": 1, "born_orders": 1},
                        "loops": 1, "seconds": 100., "seconds_per_point": 1.,
                        "status_counts": {str(code): 100}, "timed_status_counts": {str(code): 100},
                        "measurement_run_id": "same-paired-run", "paired_baseline_candidate": sources == "both",
                        "rows": [{"point": point, "order": order, "status": code, "accuracy": 1e-12,
                                  "coefficients": [1., .2, .03, -.04]}
                                 for point in range(1, 101) for order in (0, 1)]}
                self.dump(report / "builds" / key / "build.json", {
                    "executable_sha256": key, "generated_sources_sha256": "sources"})
                for rep in ("calibration", *range(repetitions)):
                    self.dump(report / "data" / f"{key}-{rep}.json", dict(data, repetition=rep))
        return bindings

    def runtime_item(self, role="candidate_corrected", backend=8):
        metadata = {}
        for name in ("mninjars.mod", *sorted(policy.RUNTIME_METADATA_NAMES)):
            filename = self.root / "runtime" / name
            self.put(filename, "same module or link metadata " + name)
            metadata["ninjars_lib/" + name] = policy.bound_file(filename)
        return {"role": role, "backend": backend,
                "libraries": {"libmodel.a": "process-model", "libdhelas.a": "process-dhelas",
                              "libcts.a": "cts", "ninja_lib/libninja.a": "ninja",
                              "ninja_lib/libavh_olo.a": "olo", "collier_lib/libcollier.a": "collier",
                              "ninjars_lib/libninjars.a": "ninjars"},
                "runtime_metadata": metadata,
                "compiled_manifest": {"files": {name: {"sha256": name, "bytes": 1}
                                                 for name in policy.RUNTIME_LINK_INPUTS}}}

    def publish_fixture(self, mutate=None):
        timing = self.make_series("timed", "both", [1, 6, 7, 8], 7)
        refs = self.make_series("refs", "baseline", [1, 6, 7], 2)
        proof = {"entries": [{"role": item["role"], "binding": copy.deepcopy(item)}
                            for group in (timing, refs) for item in group["targets"].values()]}
        if mutate:
            mutate(timing, refs)
        # Exercise real frozen series, row and pairing validation. Source binding
        # checks are tested independently above; avoid fake physics executables.
        timing_path, refs_path, proof_path = (self.root / name for name in ("timing-bindings.json", "ref-bindings.json", "proof.json"))
        self.dump(timing_path, timing); self.dump(refs_path, refs); self.dump(proof_path, {})
        with mock.patch.object(policy, "checked_bindings", side_effect=lambda filename, correction: policy.read(filename)), \
             mock.patch.object(policy, "validate_proof", return_value=proof):
            return policy.publish(self.policy_path, proof_path, timing_path, refs_path, self.root / "published")

    def test_identical_generator_with_changed_ninjars_archive_is_rejected(self):
        def mutate(timing, refs):
            timing["targets"]["toy-candidate-8"]["libraries"]["ninjars_lib/libninjars.a"] = "rebuilt-rust"
        with self.assertRaisesRegex(RuntimeError, "Reduction runtime differs"):
            self.publish_fixture(mutate)

    def test_corrected_reference_runtime_change_also_requires_fresh_proof(self):
        def mutate(timing, refs):
            refs["targets"]["toy-baseline-6"]["libraries"]["ninja_lib/libavh_olo.a"] = "different-master"
        with self.assertRaisesRegex(RuntimeError, "Reduction runtime differs"):
            self.publish_fixture(mutate)

    def test_runtime_signature_excludes_only_process_model_and_dhelas(self):
        item = self.runtime_item()
        expected = policy.reducer_signature(item)
        item["libraries"]["libmodel.a"] = "another physical model"
        item["libraries"]["libdhelas.a"] = "another process helicity library"
        self.assertEqual(policy.reducer_signature(item), expected)
        item["libraries"]["libiregi.a"] = "unexpected linked reduction library"
        self.assertNotEqual(policy.reducer_signature(item), expected)

    def test_metadata_and_mixed_representative_runtime_are_rejected(self):
        item = self.runtime_item()
        other = copy.deepcopy(item)
        other["libraries"]["ninjars_lib/libninjars.a"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "mixes reduction runtime"):
            policy.proof_runtime_signatures([{"binding": item}, {"binding": other}])
        metadata = next(iter(item["runtime_metadata"].values()))
        self.put(policy.path(metadata["path"]), "corruption")
        with self.assertRaisesRegex(RuntimeError, "Changed bound evidence"):
            policy.reducer_signature(item)

    def test_missing_ninjars_interface_or_runtime_refresh_is_rejected(self):
        item = self.runtime_item()
        del item["runtime_metadata"]["ninjars_lib/mninjars.mod"]
        with self.assertRaisesRegex(RuntimeError, "metadata is incomplete"):
            policy.reducer_signature(item)
        del item["runtime_metadata"]
        with self.assertRaisesRegex(RuntimeError, "refresh is required"):
            policy.reducer_signature(item)

    def test_complete_failed_physics_data_is_published_honestly_without_mutating_inputs(self):
        def mutate(timing, refs):
            report = Path(timing["report"])
            for filename in (report / "data").glob("toy-baseline-6-*.json"):
                data = policy.read(filename)
                data["rows"][0]["coefficients"][1] = .3
                self.dump(filename, data)
            self.before_publication = {str(filename): filename.read_bytes()
                for group in (timing, refs) for filename in Path(group["report"]).rglob("*") if filename.is_file()}
        result = self.publish_fixture(mutate)
        self.assertTrue(result["cases"]["toy"]["corrected_agreement"]["passed"])
        self.assertFalse(result["cases"]["toy"]["original_diagnostics"]["passed"])
        self.assertTrue(result["cases"]["toy"]["performance"]["passed"])
        self.assertEqual(result["performance_denominator"], "original/6")
        for filename, original in self.before_publication.items():
            self.assertEqual(Path(filename).read_bytes(), original)

    def test_corrected_reference_disagreement_cannot_be_reclassified_as_original_bug(self):
        def mutate(timing, refs):
            for filename in (Path(refs["report"]) / "data").glob("toy-baseline-7-*.json"):
                data = policy.read(filename); data["rows"][0]["coefficients"][1] = .3
                self.dump(filename, data)
        result = self.publish_fixture(mutate)
        self.assertFalse(result["passed"])
        self.assertFalse(result["cases"]["toy"]["corrected_agreement"]["passed"])

    def test_runtime_failure_status_cannot_be_exempted(self):
        def mutate(timing, refs):
            filename = Path(timing["report"]) / "data/toy-baseline-7-0.json"
            data = policy.read(filename); data["rows"][0]["status"] = 410
            self.dump(filename, data)
        with self.assertRaisesRegex(RuntimeError, "unexpected status"):
            self.publish_fixture(mutate)

    def test_missing_reference_command_is_structural_failure(self):
        def mutate(timing, refs):
            (Path(refs["report"]) / "data/toy-baseline-6-1.json").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing or noncontiguous"):
            self.publish_fixture(mutate)

    def test_unpaired_original_timing_is_rejected(self):
        def mutate(timing, refs):
            filename = Path(timing["report"]) / "data/toy-baseline-6-0.json"
            data = policy.read(filename); data["measurement_run_id"] = "unpaired"
            self.dump(filename, data)
        with self.assertRaisesRegex(RuntimeError, "Paired worker inputs differ"):
            self.publish_fixture(mutate)

    def test_changed_case_point_count_is_rejected(self):
        def mutate(timing, refs):
            filename = Path(refs["report"]) / "data/toy-baseline-6-0.json"
            data = policy.read(filename); data["rows"].pop()
            self.dump(filename, data)
        with self.assertRaisesRegex(RuntimeError, "missing/duplicated/reordered"):
            self.publish_fixture(mutate)

    def test_offline_bound_source_validation_survives_removed_outputs_but_rejects_retained_tampering(self):
        import shutil
        for name, value in (("libmodel.a", "original model"), ("libdhelas.a", "original DHELAS"),
                            ("ninja_lib/libninja.a", "original Ninja"), ("ninja_lib/libavh_olo.a", "original masters"),
                            ("mpmodule.mod", "original module")):
            self.put(self.original / "lib" / name, value)
        output = self.root / "disposable-worker"
        references.clone_output(self.original, output)
        directory = output / "SubProcesses/P0_test"
        worker_source = (ROOT / "tools/ninjars_benchmark_worker.f").read_text().replace(
            "@PREFIX@", "ML5_0_").replace("@NEXTERNAL@", "3").replace("@NBORN@", "1")
        fresh_source = (ROOT / "tools/ninjars_benchmark_driver.f").read_text().replace(
            "@PREFIX@", "ML5_0_").replace("@NEXTERNAL@", "3").replace("@NBORN@", "1")
        self.put(directory / "check_sa.f", worker_source)
        self.put(directory / "check", "worker executable fixture")
        self.put(directory / "check_fresh", "fresh executable fixture")
        series = self.make_series("offline", "baseline", [6], 2)
        report = Path(series["report"])
        compiled = policy.manifest(output)
        build = {"generation_source_identity": self.identity,
                 "generated_sources_sha256": compiled["files_sha256"],
                 "driver_sha256": policy.workers.digest(directory / "check_sa.f"),
                 "executable_sha256": policy.workers.digest(directory / "check"),
                 "fresh_executable_sha256": policy.workers.digest(directory / "check_fresh"),
                 "fresh_driver_sha256": policy.hashlib.sha256(fresh_source.encode()).hexdigest()}
        retained = report / "builds/toy-baseline-6"
        self.dump(retained / "build.json", build)
        self.dump(retained / "compiled-sources.json", compiled)
        self.put(retained / "worker.f", worker_source)
        for filename in (report / "data").glob("*.json"):
            data = policy.read(filename)
            data.update(executable_sha256=build["executable_sha256"], generated_sources_sha256=compiled["files_sha256"])
            self.dump(filename, data)
        bindings_path = self.root / "offline-bindings.json"
        policy.bind_run(report, {"baseline": "original"}, {"toy-baseline-6": str(output)},
                        self.correction_path, bindings_path)
        live = policy.checked_bindings(bindings_path, self.correction, live=True)
        original_bytes = bindings_path.read_bytes()
        refreshed_path = self.root / "runtime-bindings.json"
        refreshed = policy.refresh_runtime_bindings(bindings_path, self.correction_path, refreshed_path)
        self.assertEqual(bindings_path.read_bytes(), original_bytes)
        self.assertEqual(policy.checked_bindings(refreshed_path, self.correction, live=True), refreshed)
        self.put(output / "lib/mpmodule.mod", "changed compiler interface")
        with self.assertRaisesRegex(RuntimeError, "module/link metadata changed"):
            policy.checked_bindings(refreshed_path, self.correction, live=True)
        self.put(output / "lib/mpmodule.mod", "original module")
        for disposable in (output, self.original, self.before, self.after):
            shutil.rmtree(disposable)
        correction = policy.validate_correction(self.correction_path)
        self.assertEqual(policy.checked_bindings(bindings_path, correction), live)
        self.assertEqual(policy.checked_bindings(refreshed_path, correction), refreshed)
        self.put(retained / "worker.f", worker_source + "unexpected mutation")
        with self.assertRaisesRegex(RuntimeError, "Bound run evidence changed"):
            policy.checked_bindings(bindings_path, correction)

    def test_wrong_generator_snapshot_cannot_authorize_arbitrary_model_postimages(self):
        spec = copy.deepcopy(self.spec)
        wrong = self.root / "unrelated-exporter.py"
        self.put(wrong, "unrelated generator")
        spec["generator_after"]["madgraph/iolibs/export_v4.py"] = str(wrong)
        with self.assertRaisesRegex(RuntimeError, "not bound to corrected exporter"):
            policy.create_correction(spec, self.root / "wrong-correction.json")

    def test_forged_parent_generation_label_does_not_replace_committed_original_anchor(self):
        self.put(self.original / "SubProcesses/P0_test/matrix.f", "changed candidate implementation")
        with self.assertRaisesRegex(RuntimeError, "committed untouched-original"):
            policy.create_correction(self.spec, self.root / "wrong-parent.json")

    def test_offline_proof_revalidation_needs_no_temporary_work_directory(self):
        policy_record = policy.read(self.policy_path)
        recorded = policy.seal({"schema": policy.PROOF,
            "policy_identity_sha256": policy_record["identity_sha256"],
            "correction_identity_sha256": self.correction["identity_sha256"],
            "files_sha256": {}, "bindings": []})
        proof = self.root / "retained-proof.json"
        self.dump(proof, recorded)
        def recompute(policy_path, bindings, output=None):
            self.assertIsNone(output)
            return recorded
        with mock.patch.object(policy, "create_proof", side_effect=recompute), \
             mock.patch("tempfile.TemporaryDirectory", side_effect=AssertionError("Disposable work tree is absent")):
            self.assertEqual(policy.validate_proof(proof, self.policy_path), recorded)


if __name__ == "__main__":
    unittest.main()
