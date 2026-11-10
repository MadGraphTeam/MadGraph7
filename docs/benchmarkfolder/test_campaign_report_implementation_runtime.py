"""Synthetic closed evidence: no compiler, loader, reducer or frozen imports."""
import gzip
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location(
    "implementation_runtime_under_test", Path(__file__).with_name("campaign_report_implementation_runtime.py"))
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)


def encode(value):
    return json.dumps(value, sort_keys=True).encode("utf-8")


class Package:
    def __init__(self, schema):
        self.index = {"schema": schema, "files": {}}
        self.files = {}

    def put(self, name, value, raw=False):
        data = value if raw else encode(value)
        self.files[name] = data
        self.index["files"][name] = {"sha256": M.sha(data), "bytes": len(data)}

    def raw(self, name):
        identity = self.index["files"][name]
        data = self.files[name]
        assert M.sha(data) == identity["sha256"] and len(data) == identity["bytes"]
        return data

    def json(self, name):
        data = self.raw(name)
        return json.loads(gzip.decompress(data) if name.endswith(".gz") else data)


def mapping(executable, library, identity):
    text = "retained process-map text\n"
    expected = {library: identity}
    return {"executable": executable, "expected_runtime_sha256": expected,
            "mapped_files_sha256_after_exit": dict(expected),
            "mapped_elf_paths": [executable, library], "maps_text": text,
            "maps_sha256": M.sha(text.encode()), "selected_library": library}


def adaptive_fixture():
    package = Package(1)
    package.put("frozen/build1/build.json", {"build": "closed"})
    package.put("frozen/source-index.json", {"source": "closed"})
    keys = ("evaluations_256", "evaluations_512", "cache_hits_256", "cache_hits_512",
            "check_evaluations_256", "check_evaluations_512", "eligible_refusals_256",
            "successes_512", "failures_512")
    actions = []
    for number in range(1, 11):
        delta = dict.fromkeys(keys, 0)
        if number < 10:
            delta["evaluations_256" if number < 4 else "evaluations_512"] = 1
        actions.append({"action": number, "delta": delta,
                        "tensor_status": 2 if number == 10 else 0,
                        "last_dispatched_tensor_bits": 256 if number < 4 else 512,
                        "accepted_amplitude_tier": None, "successes_256": None,
                        "attribution_gap": None})
    legacy = {"action": 1, "tensor_status": 0, "attribution_gap": "getter_unavailable",
              "delta": None, "last_dispatched_tensor_bits": None,
              "accepted_amplitude_tier": None, "successes_256": None}
    result = {"schema": 1, "passed": True, "genuine_binary128_input_components": 13,
              "accepted_amplitude_tier": None, "successes_256": None,
              "partial_finite": "unevaluated; zero ABI placeholder only", "workers": {}}
    execution = {"complete": True, "workers": {},
                 "build_sha256": M.sha(package.raw("frozen/build1/build.json")),
                 "source_index_sha256": M.sha(package.raw("frozen/source-index.json"))}
    for name, worker_actions in (("adaptive", actions), ("legacy", [legacy])):
        result["workers"][name] = {"actions": worker_actions, "tensor_calls": len(worker_actions),
                                    "cache_boundary_preserves_precision_counters": True,
                                    "recreation_attribution_gap": "context_generation_changed"}
        maps = mapping("/old/" + name, "/old/lib" + name, M.sha(name.encode()))
        execution["workers"][name] = {"exit_code": 0, "loader": {"exit_code": 0},
                                        "progress": {"process_reaped": True},
                                        "argv": [maps["executable"]], "mapping": maps,
                                        "runtime_sha256": maps["expected_runtime_sha256"]}
    return package, execution, result


def seal_adaptive(package, execution, result):
    package.put("results.json", result)
    execution["results_sha256"] = M.sha(package.raw("results.json"))
    package.put("execution.json", execution)
    package.index["execution_sha256"] = M.sha(package.raw("execution.json"))
    return package


def test_adaptive_preserves_narrow_scope_and_real_counter_totals():
    result = M.adaptive(seal_adaptive(*adaptive_fixture()))
    assert result["precision_counters"]["evaluations_256"] == 3
    assert result["precision_counters"]["evaluations_512"] == 6
    assert result["invalid_mode_status"] == 2
    assert result["accepted_amplitude_tier"] is None
    assert result["amplitude_qualification"] is False
    assert result["adaptive_library_sha256"] == M.sha(b"adaptive")


@pytest.mark.parametrize("change", [
    lambda e, r: e.update(complete=False),
    lambda e, r: e["workers"]["adaptive"].update(exit_code=1),
    lambda e, r: e["workers"]["adaptive"].update(exit_code=False),
    lambda e, r: e["workers"]["adaptive"]["mapping"]["mapped_files_sha256_after_exit"].update({"/old/libadaptive": "a" * 64}),
    lambda e, r: e["workers"]["adaptive"]["mapping"].update(maps_text="different"),
    lambda e, r: e["workers"]["adaptive"]["mapping"]["mapped_elf_paths"].append("/unbound/library"),
    lambda e, r: r.update(accepted_amplitude_tier=512),
    lambda e, r: r.update(successes_256=1),
    lambda e, r: r.update(genuine_binary128_input_components=True),
    lambda e, r: r["workers"]["adaptive"]["actions"][9].update(tensor_status=0),
    lambda e, r: r["workers"]["adaptive"]["actions"][9]["delta"].update(evaluations_512=1),
    lambda e, r: r["workers"]["adaptive"]["actions"][0]["delta"].update(evaluations_256=True),
    lambda e, r: r["workers"]["legacy"]["actions"][0].update(delta={}),
])
def test_adaptive_rejects_changed_qualification(change):
    package, execution, result = adaptive_fixture()
    change(execution, result)
    with pytest.raises(ValueError):
        M.adaptive(seal_adaptive(package, execution, result))


WORK = ".tirust-build/born-effective-runtime/native89-controller2"


class BornFixture:
    def __init__(self):
        self.package = Package("portable-real-z-born-native89-qualification-v1")
        self.embedded, self.prepared, self.summaries = {}, {"commands": []}, {}
        identity = M.sha(b"native dependency label, never opened")
        for stage, mode, policies in (("ordinary", -1, (1,)), ("fixed-qp", 4, (0, 1))):
            summary = {"stage": stage, "state": "complete", "changed_bound_files": {},
                       "runs": [], "comparisons": []}
            for backend in (8, 9):
                for policy in policies:
                    for arm in ("original", "mass", "graph", "full"):
                        key = f"real_uux_z/{arm}/backend{backend}-mode{mode}-policy{policy}-warm4-order0"
                        exe = "/historical/" + key
                        command = {"id": key, "case": "real_uux_z", "stage": stage,
                                   "backend": backend, "mode": mode, "policy": policy,
                                   "arm": arm, "argv": [exe], "executable_sha256": identity}
                        self.prepared["commands"].append(command)
                        base = WORK + "/execution-" + stage + "/runs/" + key
                        maps = mapping(exe, "/historical/lib", identity)
                        event = {"command": [exe], "returncode": 0, "timed_out": False}
                        runtime = {exe: {"sha256": identity}, "/historical/lib": {"sha256": identity}}
                        rows, checks = [], []
                        for call, point in enumerate((1, 2, 1), 1):
                            rows.append({"call": call, "point": point, "ret": 210 + backend if mode == -1 else 120,
                                         "finite": True, "rounding_unchanged": True,
                                         "coefficients": {"0": [1.0, 2.0, 3.0, 4.0]},
                                         "stability_checked": mode == -1, "actual_tier": "DP" if mode == -1 else "QP",
                                         "actual_integrand_precision": "QP" if mode == 4 and policy else "DP",
                                         "actual_integrand_input_view": "host DP point P" if mode == -1 else ("C (MP_P)" if policy else "REAL(C,8)"),
                                         "flags": [], "mp_done": False, "mp_done_once": False})
                            for label in ("published Born exact actual-view", "DP view exact rounded C"):
                                checks.append({"name": f"{label} call{call}", "pass": True})
                        parsed = {"passed": True, "parse_errors": [], "unstable_contribution_warnings": [],
                                  "rows": rows, "checks": checks}
                        for name, value in (("execution", event), ("mapping", maps), ("runtime", runtime), ("parsed", parsed)):
                            self.embedded[base + "/" + name + ".json"] = value
                        self.embedded[base + "/process-maps.txt"] = maps["maps_text"]
                        self.embedded[base + "/output.log"] = "retained ordinary diagnostic output\n"
                        summary["runs"].append({"id": key, "passed": True, "maps_passed": True, "command": event,
                                                "maps_sha256": maps["maps_sha256"]})
                        if arm != "original":
                            summary["comparisons"].append({"backend": backend, "policy": policy, "arm": arm,
                                "case": "real_uux_z", "mode": mode, "passed": True, "criterion": 3e-6,
                                "unchecked": mode == 4,
                                "coefficients": [{"point": p, "split": "0", "slot": s, "relative": 0.0}
                                                 for p in (1, 2, 1) for s in range(4)],
                                "states": [{"point": p, "ret_equal": True, "flags_equal": True,
                                            "rounding_equal": True, "mp_state_equal": True} for p in (1, 2, 1)]})
            self.summaries[stage] = summary

    def seal(self):
        self.package.put("native89-prepared.json", self.prepared)
        self.embedded[WORK + "/prepared.json"] = self.prepared
        for stage, summary in self.summaries.items():
            summary["prepared_sha256"] = M.sha(self.package.raw("native89-prepared.json"))
            for run in summary["runs"]:
                base = WORK + "/execution-" + stage + "/runs/" + run["id"]
                for key in ("parsed", "runtime"):
                    run[key + "_sha256"] = M.sha(encode(self.embedded[base + "/" + key + ".json"]))
            self.package.put(stage + "-summary.json", summary)
            self.embedded[WORK + "/execution-" + stage + "/summary.json"] = summary
        files, blobs = {}, {}
        for name, value in self.embedded.items():
            data = value.encode() if isinstance(value, str) else encode(value)
            checksum = M.sha(data)
            files[name] = {"sha256": checksum, "bytes": len(data)}
            blobs[checksum] = data.decode()
        raw = encode({"schema": "real-z-born-native89-utf8-v1", "files": files, "blobs": blobs})
        packed = gzip.compress(raw, mtime=0)
        self.package.put("payload.json.gz", packed, raw=True)
        self.package.put("payload-inventory.json", {"files": files, "gzip_sha256": M.sha(packed),
                                                    "uncompressed_bytes": len(raw), "unique_text_contents": len(blobs)})
        return self.package

    def first(self, stage="fixed-qp", arm="original"):
        return next(value for name, value in self.embedded.items()
                    if "/execution-" + stage + "/" in name and "/" + arm + "/" in name
                    and name.endswith("/parsed.json"))


def test_born_derives_counts_without_inventing_stability_or_stress():
    result = M.born_backend89(BornFixture().seal())
    assert (result["ordinary_runs"], result["fixed_qp_runs"], result["output_logs"]) == (8, 16, 24)
    assert result["status_counts"] == {"120": 48, "218": 12, "219": 12}
    assert result["unchecked_status120_rows"] == 48
    assert result["W03_lines"] == 0
    assert result["full_fixed_qp_exact_rows"] == 12
    assert result["fixed_qp_stability_acceptance"] is False
    assert result["stress_qualification"] is False


@pytest.mark.parametrize("change", [
    lambda f: f.summaries["ordinary"].update(state="running"),
    lambda f: f.summaries["ordinary"]["runs"].pop(),
    lambda f: f.summaries["ordinary"]["runs"][0].update(maps_passed=False),
    lambda f: f.first()["rows"][0].update(stability_checked=True),
    lambda f: f.first()["rows"][0].update(ret=219),
    lambda f: f.first()["rows"][0].update(actual_integrand_input_view="host DP point P"),
    lambda f: f.first("ordinary")["rows"][0].update(point=2),
    lambda f: f.first()["rows"][0]["coefficients"]["0"].__setitem__(1, 99.0),
    lambda f: f.first("fixed-qp", "full")["checks"].pop(),
    lambda f: f.summaries["fixed-qp"]["comparisons"][0].update(unchecked=False),
    lambda f: f.summaries["ordinary"]["comparisons"][0]["coefficients"][0].update(relative=1e-10),
])
def test_born_rejects_invalid_or_overqualified_records(change):
    fixture = BornFixture()
    change(fixture)
    with pytest.raises(ValueError):
        M.born_backend89(fixture.seal())


def test_born_checks_raw_w03_even_when_parser_claims_none():
    fixture = BornFixture()
    name = next(n for n in fixture.embedded if n.endswith("/output.log"))
    fixture.embedded[name] = "##W03 unstable contribution\n"
    with pytest.raises(ValueError, match="warning"):
        M.born_backend89(fixture.seal())


def test_born_rejects_corrupt_embedded_blob_with_resealed_outer_package():
    package = BornFixture().seal()
    payload = package.json("payload.json.gz")
    checksum = next(iter(payload["blobs"]))
    payload["blobs"][checksum] += " "
    raw = encode(payload)
    packed = gzip.compress(raw, mtime=0)
    package.put("payload.json.gz", packed, raw=True)
    inventory = package.json("payload-inventory.json")
    inventory.update(gzip_sha256=M.sha(packed), uncompressed_bytes=len(raw))
    package.put("payload-inventory.json", inventory)
    with pytest.raises(ValueError, match="text hash"):
        M.born_backend89(package)


class RepositoryEvidence:
    def __init__(self, files):
        self.files, self.reads = files, []

    def add(self, name, expected):
        self.reads.append(name)
        M.need(name in self.files and M.sha(self.files[name]) == expected, "Repository identity differs")
        return self.files[name]


class SymmetryFixture:
    def __init__(self):
        self.package = Package("ew-v2-production-promotion-qualification-v1")
        prefix = "experiments/ew-invariant-symmetry-prototype/v2/"
        names = ["experiments/dp-rescue-cause-observer/controller/run.py"] + [prefix + name for name in (
            "TIRUST_interface.f", "build-qualification-1/index.json", "fortran-fixtures/qualification-1/index.json",
            "index.json", "promotion-proposal/compare_source.py", "promotion-proposal/promotion.patch",
            "promotion-proposal/source-comparison.json", "runtime-1/qualification-1/index.json")]
        self.repo = {name: ("closed source identity: " + name).encode() for name in names}
        self.package.evidence = RepositoryEvidence(self.repo)
        identities = {"docs/benchmarkfolder/" + name: {"sha256": M.sha(raw), "bytes": len(raw)}
                      for name, raw in self.repo.items()}
        template = "madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc"
        tests = "tests/unit_tests/loop/test_tirust_fortran.py"
        for production, retained, raw in ((template, "source/TIRUST_interface.inc", b"template snapshot"),
                                           (tests, "source/test_tirust_fortran.py", b"tests snapshot")):
            self.package.put(retained, raw, raw=True)
            identities[production] = {"sha256": M.sha(raw), "bytes": len(raw)}
        source_hashes = {name: identities[name]["sha256"] for name in (template, tests)}
        self.package.index.update(repository_bindings=identities,
            old_campaign_or_other_exports_requalified=False, performance_claim=False,
            recording_tests={"passed": 4, "skipped": 0, "returncode": 0, "reaped": True, "real_reducer_invoked": False},
            source_comparison={"canonical_statement_count": 316,
                               "original_recovered_by_removing_four_exact_blocks": True,
                               "qualified_statements_equal": True})
        self.package.put("rendered-TIRUST_interface.f", b"rendered EW fixture", raw=True)
        self.comparison = {"schema": "ew-v2-applied-production-source-comparison-v1",
            "comparison_was_source_only": True, "qualified_canonical_statements_equal": True,
            "original_template_recovered_by_removing_four_exact_blocks": True,
            "old_campaign_or_other_exports_requalified": False, "actual_recording_tests": "tests.json",
            "canonical_statement_count": 316, "canonical_statements_sha256": M.sha(b"canonical identity"),
            "applied_template_sha256": source_hashes[template], "applied_tests_sha256": source_hashes[tests],
            "rendered_sha256": M.sha(self.package.raw("rendered-TIRUST_interface.f")),
            "render_context": {"AmplitudeReduction": False, "tirust_available": True, "tirust_supports_quad_prec": True}}
        for field, relative in (("qualified_adapter_sha256", "TIRUST_interface.f"),
                                ("comparison_helper_sha256", "promotion-proposal/compare_source.py"),
                                ("reviewed_patch_sha256", "promotion-proposal/promotion.patch"),
                                ("reviewed_source_comparison_sha256", "promotion-proposal/source-comparison.json")):
            self.comparison[field] = identities["docs/benchmarkfolder/" + prefix + relative]["sha256"]
        methods = [("test_invariant_mirroring_preserves_words_and_ieee_state", "TensorAdapterTest"),
                   ("test_recorded_dp_qp_tensors_and_failure_lifecycle", "TensorAdapterTest"),
                   ("test_sparse_qp_rescue_list_wraps_at_the_zero_sentinel", "TensorAdapterTest"),
                   ("test_all_components_through_rank_six", "TensorPermutationTest")]
        self.log = "\n".join(f"{name} (tests.unit_tests.loop.test_tirust_fortran.{cls}.{name}) ... ok"
                             for name, cls in methods) + "\n\n-------------------\nRan 4 tests in 2.817s\n\nOK\n"
        self.tests = {"schema": "ew-v2-production-recording-tests-v1", "returncode": 0, "test_process_exit": 0,
            "process_spawned": True, "process_reaped": True, "sources_unchanged_by_tests": True,
            "shared_numerical_lock": True, "real_reducer_invoked": False, "full_numerical_replay_repeated": False,
            "source_sha256_before": dict(source_hashes), "source_sha256_after": dict(source_hashes),
            "helper_sha256": identities["docs/benchmarkfolder/experiments/dp-rescue-cause-observer/controller/run.py"]["sha256"],
            "command": ["/historical/python", "-B", "-m", "unittest", "-v", "tests.unit_tests.loop.test_tirust_fortran"],
            "compiler": "/historical/gfortran"}

    def seal(self):
        self.package.put("unittest.log", self.log.encode(), raw=True)
        self.tests["log_sha256"] = M.sha(self.package.raw("unittest.log"))
        self.package.put("tests.json", self.tests)
        self.package.put("source-comparison.json", self.comparison)
        return self.package


def test_symmetry_promotion_binds_closed_references_and_preserves_scope():
    fixture = SymmetryFixture()
    result = M.symmetry_promotion(fixture.seal())
    assert (result["passed"], result["skipped"]) == (4, 0)
    assert (result["compiled_fixture_test_methods"], result["pure_python_test_methods"]) == (3, 1)
    assert result["statement_equivalence_basis"] == "bound source-comparison receipt"
    assert result["canonical_statement_count"] == 316
    assert result["adaptive_bridge_promoted"] is False
    assert result["old_campaign_or_other_exports_requalified"] is False
    assert result["performance_qualified"] is False
    assert len(fixture.package.evidence.reads) == 9
    assert all(name.startswith("experiments/") for name in fixture.package.evidence.reads)


@pytest.mark.parametrize("change", [
    lambda f: f.package.index.update(performance_claim=True),
    lambda f: f.package.index.update(old_campaign_or_other_exports_requalified=True),
    lambda f: f.package.index["recording_tests"].update(returncode=False),
    lambda f: f.package.index["recording_tests"].update(skipped=1),
    lambda f: f.comparison.update(qualified_canonical_statements_equal=False),
    lambda f: f.comparison.update(applied_template_sha256="0" * 64),
    lambda f: f.comparison.update(qualified_adapter_sha256="0" * 64),
    lambda f: f.comparison["render_context"].update(AmplitudeReduction=True),
    lambda f: f.tests.update(real_reducer_invoked=True),
    lambda f: f.tests.update(full_numerical_replay_repeated=True),
    lambda f: f.tests.update(process_reaped=False),
    lambda f: f.tests["source_sha256_after"].update({"madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc": "0" * 64}),
    lambda f: f.tests["command"].__setitem__(-1, "other.tests"),
])
def test_symmetry_promotion_rejects_unbound_or_overqualified_receipts(change):
    fixture = SymmetryFixture()
    change(fixture)
    with pytest.raises(ValueError):
        M.symmetry_promotion(fixture.seal())


@pytest.mark.parametrize("change", [
    lambda s: s.replace(" ... ok", " ... skipped 'compiler missing'", 1),
    lambda s: s.replace("Ran 4 tests", "Ran 3 tests"),
    lambda s: s.replace("\nOK\n", "\nFAILED (failures=1)\n"),
    lambda s: s.replace("test_all_components_through_rank_six", "test_unrelated_method"),
])
def test_symmetry_promotion_parses_actual_log_instead_of_summary_count(change):
    fixture = SymmetryFixture()
    fixture.log = change(fixture.log)
    with pytest.raises(ValueError):
        M.symmetry_promotion(fixture.seal())


def test_symmetry_rejects_unknown_repository_scope_before_reading_it():
    fixture = SymmetryFixture()
    fixture.package.index["repository_bindings"]["docs/benchmarkfolder/active/output.log"] = {
        "sha256": "0" * 64, "bytes": 1}
    with pytest.raises(ValueError, match="reference scope"):
        M.symmetry_promotion(fixture.seal())
    assert fixture.package.evidence.reads == []


def test_symmetry_rejects_changed_historical_repository_bytes():
    fixture = SymmetryFixture()
    name = next(iter(fixture.repo))
    fixture.repo[name] = b"changed"
    with pytest.raises(ValueError, match="Repository identity"):
        M.symmetry_promotion(fixture.seal())


class FiniteAuditFixture:
    def __init__(self):
        self.package = Package(1)
        path = "experiments/generated-rank6-poles-proposal/attempt1/common-input-proposal/runtime1/index.json"
        raw = b'{"closed":"runtime identity"}'
        self.package.evidence = RepositoryEvidence({path: raw})
        self.package.index.update(
            status="read-only audit; finite discrepancy open; comparator missed requested accuracy",
            native_evaluations=0, implementation_changes=0, acceptance_threshold_changes=0,
            retained_runtime_index_sha256=M.sha(raw))
        self.bindings = {"schema": 1, "live_verification_optional": True,
                         "retained_runtime_index": {"path": "docs/benchmarkfolder/" + path, "sha256": M.sha(raw)}}
        ranks, offset = [], 0
        for rank, count in enumerate((1, 4, 10, 20, 35, 56, 84)):
            parts = [{"part": part, "mismatched_at_3e6": 0,
                      "maximum_symmetric_residual": "0", "worst_index": offset}
                     for part in ("real", "imaginary")]
            ranks.append({"rank": rank, "coefficients": count, "parts": parts})
            offset += count
        ranks[-1]["parts"][0].update(mismatched_at_3e6=1, maximum_symmetric_residual="0.0001", worst_index=160)
        ranks[-1]["parts"][1].update(mismatched_at_3e6=2, maximum_symmetric_residual="0.0013285038", worst_index=161)
        suspects = [{"index": i, "part": part, "rank": 6, "powers": powers,
                     "tirust_action": 3007, "collier_action": 3013, "symmetric_residual": residual}
                    for i, part, powers, residual in ((160, "real", [6, 0, 0, 0], "0.0001"),
                                                       (161, "imaginary", [3, 1, 1, 1], "0.0013285038"),
                                                       (162, "imaginary", [2, 2, 1, 1], "0.0002"))]
        self.analysis = {"schema": 1, "status": "open; neither finite truth nor implementation blame established",
            "fixture": 108, "n": 6, "rank": 6, "coefficient_count": 210,
            "same_finite_binary128_words_across_pole_selectors": {"tirust": True, "collier": True},
            "qp_full_partial_pole_binary128_words_equal": True, "rank_summaries": ranks,
            "finite_mismatched_parts": 3, "finite_mismatched_real_parts": 1, "finite_mismatched_imaginary_parts": 2,
            "suspect_rows": suspects, "maximum_finite_symmetric_residual": "0.0013285038",
            "pole_summaries": [{"slot": "simple", "maximum_symmetric_residual": "1",
                                "maximum_pole_to_finite_ratio": "7.3e-7", "tirust_nonzero_complex_coefficients": 0,
                                "collier_nonzero_complex_coefficients": 210},
                               {"slot": "double", "maximum_symmetric_residual": "0",
                                "maximum_pole_to_finite_ratio": "0", "tirust_nonzero_complex_coefficients": 0,
                                "collier_nonzero_complex_coefficients": 0}],
            "n6_row_states": [{"action": a, "state": [32, 0, 32, 0, 1, 0, -1 if a >= 3013 else 0],
                               "adapter_stable": a >= 3007} for a in range(3001, 3016)]}

    def seal(self):
        self.package.put("bindings.json", self.bindings)
        self.package.put("analysis.json", self.analysis)
        return self.package


def test_finite_audit_keeps_finite_maximum_separate_from_zero_pole_ratio():
    fixture = FiniteAuditFixture()
    result = M.finite_audit(fixture.seal())
    assert result["maximum_finite_symmetric_residual"] == "0.0013285038"
    assert result["maximum_aggregate_symmetric_residual"] == "1"
    assert result["maximum_aggregate_slot"] == "simple"
    assert result["worst_finite_index"] == 161
    assert result["finite_mismatched_parts"] == 3
    assert result["comparator_requested_accuracy_miss_rows"] == 3
    assert result["cause_established"] is result["physical_accuracy_qualified"] is result["performance_qualified"] is False
    assert fixture.package.evidence.reads == [result["retained_runtime_index"]]


@pytest.mark.parametrize("change", [
    lambda f: f.package.index.update(native_evaluations=1),
    lambda f: f.package.index.update(acceptance_threshold_changes=True),
    lambda f: f.bindings["retained_runtime_index"].update(path="/active/runtime/index.json"),
    lambda f: f.analysis.update(maximum_finite_symmetric_residual="1"),
    lambda f: f.analysis.update(maximum_finite_symmetric_residual="NaN"),
    lambda f: f.analysis.update(finite_mismatched_parts=122),
    lambda f: f.analysis["suspect_rows"].append(dict(f.analysis["suspect_rows"][0])),
    lambda f: f.analysis["suspect_rows"][0].update(tirust_action=3001),
    lambda f: f.analysis["rank_summaries"][-1]["parts"][1].update(maximum_symmetric_residual="1"),
    lambda f: f.analysis["pole_summaries"][0].update(maximum_symmetric_residual="Infinity"),
    lambda f: f.analysis["n6_row_states"][-1]["state"].__setitem__(-1, 0),
])
def test_finite_audit_rejects_changed_scope_or_mislabeled_maximum(change):
    fixture = FiniteAuditFixture()
    change(fixture)
    with pytest.raises(ValueError):
        M.finite_audit(fixture.seal())
