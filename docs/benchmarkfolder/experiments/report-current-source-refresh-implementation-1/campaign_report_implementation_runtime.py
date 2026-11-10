"""Read selected, closed implementation evidence without importing its code.

``package`` supplies an already hash-bound index and allowlisted raw/json reads.
Historical absolute paths are evidence labels only; this module never opens them.
The returned records describe narrow implementation checks, not campaign gates.
"""
from collections import Counter
from decimal import Decimal, InvalidOperation
import gzip
import hashlib
import json
import math
from pathlib import PurePosixPath
import re


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def integer(value):
    return type(value) is int and value >= 0


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def safe_label(name):
    need(isinstance(name, str) and name and "\\" not in name,
         "Invalid embedded evidence path")
    path = PurePosixPath(name)
    need(not path.is_absolute() and ".." not in path.parts and str(path) == name,
         "Unsafe embedded evidence path")


def _json(text):
    def invalid(value):
        raise ValueError("Nonfinite embedded JSON: " + value)
    return json.loads(text, parse_constant=invalid)


def _mapping(mapping, expected, executable):
    """Check recorded mappings; do not load or rehash native library paths."""
    need(isinstance(expected, dict) and expected
         and all(digest(v) for v in expected.values()), "Invalid mapped dependency hashes")
    need(mapping["executable"] == executable
         and mapping["expected_runtime_sha256"] == expected
         and mapping["mapped_files_sha256_after_exit"] == expected,
         "Recorded dependency identities differ")
    need(set(mapping["mapped_elf_paths"]) == set(expected) | {executable}
         and len(mapping["mapped_elf_paths"]) == len(expected) + 1,
         "Recorded ELF mapping closure differs")
    need(sha(mapping["maps_text"].encode("utf-8")) == mapping["maps_sha256"],
         "Recorded process-map bytes differ")


def adaptive(package):
    """Extract an actual standalone adaptive/legacy Fortran tensor diagnostic."""
    try:
        return _adaptive(package)
    except (KeyError, TypeError, UnicodeError) as error:
        raise ValueError("Malformed adaptive implementation evidence") from error


def _adaptive(package):
    execution = package.json("execution.json")
    result = package.json("results.json")
    need(type(package.index.get("schema")) is int and package.index["schema"] == 1
         and execution.get("complete") is True,
         "Adaptive execution is not complete")
    need(package.index["execution_sha256"] == sha(package.raw("execution.json"))
         and execution["results_sha256"] == sha(package.raw("results.json"))
         and execution["build_sha256"] == sha(package.raw("frozen/build1/build.json"))
         and execution["source_index_sha256"] == sha(package.raw("frozen/source-index.json")),
         "Adaptive execution/source/result binding differs")
    need(type(result.get("schema")) is int and result["schema"] == 1 and result.get("passed") is True
         and type(result.get("genuine_binary128_input_components")) is int
         and result["genuine_binary128_input_components"] == 13,
         "Adaptive tensor/transport result differs")
    need(result.get("accepted_amplitude_tier", "missing") is None
         and result.get("successes_256", "missing") is None
         and result.get("partial_finite") == "unevaluated; zero ABI placeholder only",
         "Adaptive result overstates amplitude or finite qualification")
    need(set(execution["workers"]) == set(result["workers"]) == {"adaptive", "legacy"},
         "Adaptive worker scope differs")
    libraries = {}
    for name, worker in execution["workers"].items():
        need(type(worker["exit_code"]) is int and worker["exit_code"] == 0
             and type(worker["loader"]["exit_code"]) is int and worker["loader"]["exit_code"] == 0
             and worker["progress"]["process_reaped"] is True,
             "Adaptive native worker did not exit successfully")
        mapping = worker["mapping"]
        _mapping(mapping, worker["runtime_sha256"], worker["argv"][0])
        selected = mapping["selected_library"]
        need(selected in worker["runtime_sha256"], "Selected adaptive library is not mapped")
        libraries[name] = worker["runtime_sha256"][selected]

    counters = {name: 0 for name in (
        "evaluations_256", "evaluations_512", "cache_hits_256", "cache_hits_512",
        "check_evaluations_256", "check_evaluations_512", "eligible_refusals_256",
        "successes_512", "failures_512")}
    adaptive_worker = result["workers"]["adaptive"]
    legacy_worker = result["workers"]["legacy"]
    need(adaptive_worker["tensor_calls"] == len(adaptive_worker["actions"]) == 10
         and legacy_worker["tensor_calls"] == len(legacy_worker["actions"]) == 1,
         "Adaptive tensor-call coverage differs")
    for number, action in enumerate(adaptive_worker["actions"], 1):
        need(type(action["action"]) is int and action["action"] == number
             and action.get("accepted_amplitude_tier", "missing") is None
             and action.get("successes_256", "missing") is None
             and action["attribution_gap"] is None,
             "Adaptive call attribution differs")
        delta = action["delta"]
        need(set(delta) == set(counters) and all(integer(v) for v in delta.values()),
             "Adaptive precision counters are malformed")
        need(type(action["tensor_status"]) is int and action["tensor_status"] == (2 if number == 10 else 0)
             and type(action["last_dispatched_tensor_bits"]) is int
             and action["last_dispatched_tensor_bits"] in (256, 512),
             "Adaptive diagnostic status differs")
        if number == 10:
            need(not any(delta.values()), "Invalid-mode rejection performed tensor arithmetic")
        for key, value in delta.items():
            counters[key] += value
    legacy = legacy_worker["actions"][0]
    need(legacy["action"] == 1 and legacy["tensor_status"] == 0
         and legacy["attribution_gap"] == "getter_unavailable"
         and all(legacy.get(key, "missing") is None for key in
                 ("delta", "last_dispatched_tensor_bits", "accepted_amplitude_tier", "successes_256")),
         "Legacy optional-getter compatibility differs")
    need(all(w["cache_boundary_preserves_precision_counters"] is True
             and w["recreation_attribution_gap"] == "context_generation_changed"
             for w in result["workers"].values()), "Adaptive context/cache evidence differs")
    need(counters["evaluations_256"] > 0 and counters["evaluations_512"] > 0,
         "Adaptive record does not exercise both tensor precisions")
    return {
        "kind": "standalone_adaptive_fortran_tensor", "passed": True,
        "genuine_binary128_input_components": result["genuine_binary128_input_components"],
        "adaptive_tensor_calls": adaptive_worker["tensor_calls"],
        "legacy_tensor_calls": legacy_worker["tensor_calls"],
        "precision_counters": counters, "invalid_mode_status": 2,
        "mapped_library_sha256": libraries,
        "adaptive_library_sha256": libraries["adaptive"],
        "accepted_amplitude_tier": None, "successes_256": None,
        "partial_finite": "unevaluated", "amplitude_qualification": False,
        "performance_qualification": False,
    }


def _payload(package):
    compressed = package.raw("payload.json.gz")
    inventory = package.json("payload-inventory.json")
    data = package.json("payload.json.gz")
    need(data.get("schema") == "real-z-born-native89-utf8-v1"
         and inventory["gzip_sha256"] == sha(compressed)
         and inventory["uncompressed_bytes"] == len(gzip.decompress(compressed)),
         "Born payload envelope differs")
    files, blobs = data["files"], data["blobs"]
    need(isinstance(files, dict) and files and isinstance(blobs, dict) and blobs
         and inventory["files"] == files and inventory["unique_text_contents"] == len(blobs),
         "Born payload inventory differs")
    for identity, text in blobs.items():
        need(digest(identity) and isinstance(text, str) and "\0" not in text
             and sha(text.encode("utf-8")) == identity,
             "Born embedded text hash differs")
    for name, identity in files.items():
        safe_label(name)
        need(set(identity) == {"sha256", "bytes"} and integer(identity["bytes"])
             and identity["sha256"] in blobs
             and len(blobs[identity["sha256"]].encode("utf-8")) == identity["bytes"],
             "Born embedded file identity differs")
    need(set(blobs) == {v["sha256"] for v in files.values()}, "Unreferenced Born text blob")
    return files, lambda name: blobs[files[name]["sha256"]]


def born_backend89(package):
    """Extract retained real-Z controls without rerunning their frozen parser."""
    try:
        return _born_backend89(package)
    except (KeyError, TypeError, UnicodeError, OSError) as error:
        raise ValueError("Malformed Born implementation evidence") from error


def _born_backend89(package):
    need(package.index.get("schema") == "portable-real-z-born-native89-qualification-v1",
         "Wrong Born package schema")
    files, text = _payload(package)
    work = ".tirust-build/born-effective-runtime/native89-controller2"
    doc = lambda name: _json(text(name))
    prepared = package.json("native89-prepared.json")
    prepared_hash = sha(package.raw("native89-prepared.json"))
    need(prepared == doc(work + "/prepared.json"), "Born prepared workload differs")
    commands = {command["id"]: command for command in prepared["commands"]}
    need(len(commands) == len(prepared["commands"]) == 24, "Born command coverage differs")
    counts, observed, maxima, runs_per_stage = Counter(), {}, {}, {}
    raw_streams = full_exact = unchecked = 0
    for stage, mode, policies in (("ordinary", -1, (1,)), ("fixed-qp", 4, (0, 1))):
        expected = {f"real_uux_z/{arm}/backend{backend}-mode{mode}-policy{policy}-warm4-order0"
                    for arm in ("original", "mass", "graph", "full")
                    for backend in (8, 9) for policy in policies}
        summary = package.json(stage + "-summary.json")
        prefix = work + "/execution-" + stage
        need(summary == doc(prefix + "/summary.json") and summary["state"] == "complete"
             and summary["stage"] == stage and summary["changed_bound_files"] == {}
             and summary["prepared_sha256"] == prepared_hash,
             "Born stage is incomplete or differently bound")
        need(len(summary["runs"]) == len(expected)
             and {run["id"] for run in summary["runs"]} == expected,
             "Born stage run coverage differs")
        runs_per_stage[stage] = len(summary["runs"])
        for run in summary["runs"]:
            key, command = run["id"], commands[run["id"]]
            need(command["case"] == "real_uux_z" and command["stage"] == stage
                 and command["backend"] in (8, 9) and command["mode"] == mode
                 and command["policy"] in policies
                 and key == f'real_uux_z/{command["arm"]}/backend{command["backend"]}'
                            f'-mode{mode}-policy{command["policy"]}-warm4-order0',
                 "Born prepared command scope differs")
            base = prefix + "/runs/" + key
            event, mapping, runtime, parsed = (doc(base + "/" + name + ".json")
                                               for name in ("execution", "mapping", "runtime", "parsed"))
            need(run["passed"] is True and run["maps_passed"] is True
                 and event == run["command"] and type(event["returncode"]) is int and event["returncode"] == 0
                 and event["timed_out"] is False and event["command"][-1] == command["argv"][0],
                 "Born native execution did not close successfully")
            executable = command["argv"][0]
            _mapping(mapping, {n: rec["sha256"] for n, rec in runtime.items() if n != executable}, executable)
            need(runtime[executable]["sha256"] == command["executable_sha256"]
                 and files[base + "/runtime.json"]["sha256"] == run["runtime_sha256"]
                 and files[base + "/parsed.json"]["sha256"] == run["parsed_sha256"]
                 and mapping["maps_sha256"] == run["maps_sha256"]
                 == sha(text(base + "/process-maps.txt").encode("utf-8")),
                 "Born execution/map/result identities differ")
            need(parsed["passed"] is True and parsed["parse_errors"] == []
                 and parsed["unstable_contribution_warnings"] == []
                 and not re.search(r"##W03\b", text(base + "/output.log"), re.IGNORECASE),
                 "Born raw stream has a parse or unstable-contribution warning")
            raw_streams += 1
            rows = parsed["rows"]
            need(len(rows) == 3 and [r["point"] for r in rows] == [1, 2, 1]
                 and [r["call"] for r in rows] == [1, 2, 3], "Born point/call coverage differs")
            for row in rows:
                counts[row["ret"]] += 1
                need(row["finite"] is True and row["rounding_unchanged"] is True
                     and row["coefficients"] and all(len(v) == 4 and all(finite(x) for x in v)
                                                      for v in row["coefficients"].values()),
                     "Invalid Born coefficients or rounding")
                if stage == "ordinary":
                    need(row["ret"] == 210 + command["backend"] and row["stability_checked"] is True
                         and row["actual_tier"] == "DP"
                         and row["actual_integrand_precision"] == "DP"
                         and row["actual_integrand_input_view"] == "host DP point P",
                         "Ordinary Born precision status differs")
                else:
                    need(row["ret"] == 120 and row["stability_checked"] is False
                         and row["actual_tier"] == "QP"
                         and row["actual_integrand_precision"] == ("QP" if command["policy"] else "DP")
                         and row["actual_integrand_input_view"] == ("C (MP_P)" if command["policy"] else "REAL(C,8)"),
                         "Fixed-QP Born result is incorrectly qualified")
                    unchecked += 1
                    if command["arm"] == "full":
                        checks = parsed["checks"]
                        for label in ("published Born exact actual-view", "DP view exact rounded C"):
                            matches = [c for c in checks if c["name"] == f'{label} call{row["call"]}']
                            need(len(matches) == 1 and matches[0]["pass"] is True,
                                 "Full-arm exact Born/coordinate assertion missing")
                        full_exact += 1
            observed[(stage, command["backend"], command["policy"], command["arm"])] = rows
        comparisons = summary["comparisons"]
        comparison_keys = {(b, p, arm) for b in (8, 9) for p in policies for arm in ("mass", "graph", "full")}
        need(len(comparisons) == len(comparison_keys)
             and {(c["backend"], c["policy"], c["arm"]) for c in comparisons} == comparison_keys,
             "Born comparison coverage differs")
        maximum = 0.0
        for comparison in comparisons:
            need(comparison["case"] == "real_uux_z" and comparison["mode"] == mode
                 and comparison["passed"] is True and comparison["criterion"] == 3e-6
                 and comparison["unchecked"] is (stage == "fixed-qp"),
                 "Born comparison scope or acceptance differs")
            key = (stage, comparison["backend"], comparison["policy"])
            diffs, states = [], []
            for old, new in zip(observed[key + ("original",)], observed[key + (comparison["arm"],)], strict=True):
                need(set(old["coefficients"]) == set(new["coefficients"]), "Born split orders differ")
                for split, coefficients in old["coefficients"].items():
                    for slot, (a, b) in enumerate(zip(coefficients, new["coefficients"][split], strict=True)):
                        need(finite(abs(a) + abs(b)), "Born comparison denominator overflow")
                        relative = abs(a - b) / (abs(a) + abs(b)) if a or b else 0.0
                        need(finite(relative) and relative <= 3e-6, "Born coefficient comparison failed")
                        diffs.append({"point": old["point"], "split": split, "slot": slot, "relative": relative})
                        maximum = max(maximum, relative)
                states.append({"point": old["point"], "ret_equal": old["ret"] == new["ret"],
                               "flags_equal": old["flags"] == new["flags"],
                               "rounding_equal": old["rounding_unchanged"] == new["rounding_unchanged"],
                               "mp_state_equal": (old["mp_done"], old["mp_done_once"]) == (new["mp_done"], new["mp_done_once"])})
            need(diffs == comparison["coefficients"] and states == comparison["states"],
                 "Born retained comparison does not match recorded rows")
            if stage == "ordinary":
                need(all(all(v for k, v in state.items() if k != "point") for state in states),
                     "Ordinary Born state changed")
        maxima[stage] = maximum
    need(counts == {218: 12, 219: 12, 120: 48} and full_exact == 12,
         "Born observation coverage differs")
    return {"kind": "real_z_born_backend89_controls", "passed": True,
            "ordinary_runs": runs_per_stage["ordinary"], "fixed_qp_runs": runs_per_stage["fixed-qp"],
            "output_logs": raw_streams, "status_counts": {str(k): v for k, v in sorted(counts.items())},
            "unchecked_status120_rows": unchecked, "W03_lines": 0,
            "full_fixed_qp_exact_rows": full_exact, "maximum_relative": maxima,
            "backends": [8, 9], "physical_case": "real_uux_z", "saved_points": 2,
            "point_sequence": [1, 2, 1], "fixed_qp_stability_acceptance": False,
            "stress_qualification": False, "performance_qualification": False,
            "independent_per_helicity_trace": False}


def symmetry_promotion(package):
    """Read the applied-template receipt and actual recording-test log.

    Statement equivalence is the retained comparison receipt's assertion.
    Historical repository references are explicitly allowlisted; mutable live
    production sources are represented by their retained snapshots instead.
    """
    try:
        return _symmetry_promotion(package)
    except (KeyError, TypeError, UnicodeError) as error:
        raise ValueError("Malformed symmetry template promotion evidence") from error


def _symmetry_promotion(package):
    index = package.index
    need(index.get("schema") == "ew-v2-production-promotion-qualification-v1"
         and index.get("old_campaign_or_other_exports_requalified") is False
         and index.get("performance_claim") is False,
         "Symmetry promotion scope differs")
    recording = index["recording_tests"]
    need(set(recording) == {"passed", "skipped", "returncode", "reaped", "real_reducer_invoked"}
         and type(recording["passed"]) is int and recording["passed"] == 4
         and type(recording["skipped"]) is int and recording["skipped"] == 0
         and type(recording["returncode"]) is int and recording["returncode"] == 0
         and recording["reaped"] is True and recording["real_reducer_invoked"] is False,
         "Symmetry recording tests did not complete successfully")
    template = "madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc"
    test_source = "tests/unit_tests/loop/test_tirust_fortran.py"
    repository_prefix = "docs/benchmarkfolder/"
    prototype = "experiments/ew-invariant-symmetry-prototype/v2/"
    docs = {
        "experiments/dp-rescue-cause-observer/controller/run.py",
        prototype + "TIRUST_interface.f",
        prototype + "build-qualification-1/index.json",
        prototype + "fortran-fixtures/qualification-1/index.json",
        prototype + "index.json",
        prototype + "promotion-proposal/compare_source.py",
        prototype + "promotion-proposal/promotion.patch",
        prototype + "promotion-proposal/source-comparison.json",
        prototype + "runtime-1/qualification-1/index.json",
    }
    identities = index["repository_bindings"]
    need(set(identities) == {repository_prefix + name for name in docs} | {template, test_source},
         "Symmetry repository reference scope differs")
    for name, identity in identities.items():
        need(set(identity) == {"sha256", "bytes"} and digest(identity["sha256"])
             and integer(identity["bytes"]), "Invalid symmetry repository identity")
        if name.startswith(repository_prefix):
            relative = name.removeprefix(repository_prefix)
            safe_label(relative)
            raw = package.evidence.add(relative, identity["sha256"])
        else:
            raw = package.raw("source/TIRUST_interface.inc" if name == template
                              else "source/test_tirust_fortran.py")
        need(len(raw) == identity["bytes"] and sha(raw) == identity["sha256"],
             "Symmetry repository/snapshot identity differs")
    source_hashes = {name: identities[name]["sha256"] for name in (template, test_source)}
    comparison = package.json("source-comparison.json")
    need(comparison.get("schema") == "ew-v2-applied-production-source-comparison-v1"
         and comparison["comparison_was_source_only"] is True
         and comparison["qualified_canonical_statements_equal"] is True
         and comparison["original_template_recovered_by_removing_four_exact_blocks"] is True
         and comparison["old_campaign_or_other_exports_requalified"] is False
         and comparison["actual_recording_tests"] == "tests.json",
         "Symmetry source-comparison receipt differs")
    need(type(comparison["canonical_statement_count"]) is int
         and comparison["canonical_statement_count"] == 316
         and digest(comparison["canonical_statements_sha256"])
         and index["source_comparison"] == {
             "canonical_statement_count": 316,
             "original_recovered_by_removing_four_exact_blocks": True,
             "qualified_statements_equal": True},
         "Symmetry statement comparison was not qualified")
    need(comparison["applied_template_sha256"] == source_hashes[template]
         and comparison["applied_tests_sha256"] == source_hashes[test_source]
         and comparison["rendered_sha256"] == sha(package.raw("rendered-TIRUST_interface.f")),
         "Symmetry rendered/applied source identity differs")
    for field, relative in (
            ("qualified_adapter_sha256", prototype + "TIRUST_interface.f"),
            ("comparison_helper_sha256", prototype + "promotion-proposal/compare_source.py"),
            ("reviewed_patch_sha256", prototype + "promotion-proposal/promotion.patch"),
            ("reviewed_source_comparison_sha256", prototype + "promotion-proposal/source-comparison.json")):
        need(comparison[field] == identities[repository_prefix + relative]["sha256"],
             "Symmetry reviewed-source linkage differs")
    need(comparison["render_context"] == {"AmplitudeReduction": False, "tirust_available": True,
                                           "tirust_supports_quad_prec": True},
         "Symmetry render context differs from the qualified adapter")
    tests = package.json("tests.json")
    need(tests.get("schema") == "ew-v2-production-recording-tests-v1"
         and type(tests["returncode"]) is int and tests["returncode"] == 0
         and type(tests["test_process_exit"]) is int and tests["test_process_exit"] == 0
         and all(tests[key] is True for key in (
             "process_spawned", "process_reaped", "sources_unchanged_by_tests", "shared_numerical_lock"))
         and tests["real_reducer_invoked"] is False
         and tests["full_numerical_replay_repeated"] is False,
         "Symmetry recording execution is incomplete or overqualified")
    need(tests["source_sha256_before"] == tests["source_sha256_after"] == source_hashes
         and tests["helper_sha256"] == identities[
             repository_prefix + "experiments/dp-rescue-cause-observer/controller/run.py"]["sha256"]
         and tests["command"][-5:] == ["-B", "-m", "unittest", "-v", "tests.unit_tests.loop.test_tirust_fortran"]
         and isinstance(tests["compiler"], str) and PurePosixPath(tests["compiler"]).name == "gfortran",
         "Symmetry test source/command binding differs")
    log = package.raw("unittest.log")
    need(tests["log_sha256"] == sha(log), "Symmetry test log identity differs")
    lines = [line for line in log.decode("utf-8").splitlines() if line and set(line) != {"-"}]
    successes = {
        ("test_invariant_mirroring_preserves_words_and_ieee_state", "TensorAdapterTest"),
        ("test_recorded_dp_qp_tensors_and_failure_lifecycle", "TensorAdapterTest"),
        ("test_sparse_qp_rescue_list_wraps_at_the_zero_sentinel", "TensorAdapterTest"),
        ("test_all_components_through_rank_six", "TensorPermutationTest"),
    }
    need(len(lines) == 6 and lines[-1] == "OK"
         and re.fullmatch(r"Ran 4 tests in [0-9]+(?:\.[0-9]+)?s", lines[-2]) is not None,
         "Symmetry test log contains a failure, skip or different test count")
    parsed = []
    for line in lines[:4]:
        match = re.fullmatch(
            r"(test_[a-z0-9_]+) \(tests\.unit_tests\.loop\.test_tirust_fortran\."
            r"(TensorAdapterTest|TensorPermutationTest)\.(test_[a-z0-9_]+)\) \.\.\. ok", line)
        need(match is not None and match[1] == match[3], "Unexpected symmetry test success record")
        parsed.append((match[1], match[2]))
    need(len(set(parsed)) == 4 and set(parsed) == successes, "Symmetry test-method coverage differs")
    return {
        "kind": "fortran_symmetry_template_promotion",
        "applied_template_sha256": source_hashes[template],
        "applied_tests_sha256": source_hashes[test_source],
        "rendered_sha256": comparison["rendered_sha256"],
        "canonical_statement_count": comparison["canonical_statement_count"],
        "statement_equivalence_basis": "bound source-comparison receipt",
        "passed": len(parsed), "skipped": 0,
        "compiled_fixture_test_methods": sum(cls == "TensorAdapterTest" for _, cls in parsed),
        "pure_python_test_methods": sum(cls == "TensorPermutationTest" for _, cls in parsed),
        "real_reducer_invoked": False, "full_numerical_replay_repeated": False,
        "old_campaign_or_other_exports_requalified": False, "performance_qualified": False,
        "adaptive_bridge_promoted": False,
    }


def finite_audit(package):
    """Separate finite and pole maxima in the sealed, additive tensor audit.

    This validates recorded summary primitives; it does not import the frozen
    analyzer, rerun reductions, or promote a comparator into numerical truth.
    """
    try:
        return _finite_audit(package)
    except (KeyError, TypeError, UnicodeError, InvalidOperation) as error:
        raise ValueError("Malformed common-input finite audit") from error


def _finite_audit(package):
    def number(value):
        need(isinstance(value, str), "Audit residual must retain its decimal text")
        result = Decimal(value)
        need(result.is_finite() and result >= 0, "Invalid finite-audit residual")
        return result

    index = package.index
    need(type(index.get("schema")) is int and index["schema"] == 1
         and index.get("status") == "read-only audit; finite discrepancy open; comparator missed requested accuracy"
         and all(type(index.get(key)) is int and index[key] == 0 for key in
                 ("native_evaluations", "implementation_changes", "acceptance_threshold_changes")),
         "Finite audit changed the implementation, threshold or execution scope")
    bindings = package.json("bindings.json")
    source = bindings["retained_runtime_index"]
    runtime_index = "experiments/generated-rank6-poles-proposal/attempt1/common-input-proposal/runtime1/index.json"
    need(bindings.get("schema") == 1 and bindings["live_verification_optional"] is True
         and source["path"] == "docs/benchmarkfolder/" + runtime_index
         and digest(source["sha256"]) and source["sha256"] == index["retained_runtime_index_sha256"],
         "Finite audit names a different retained runtime")
    raw_index = package.evidence.add(runtime_index, source["sha256"])
    need(sha(raw_index) == source["sha256"], "Finite audit runtime index changed")
    value = package.json("analysis.json")
    need(type(value.get("schema")) is int and value["schema"] == 1
         and value["status"] == "open; neither finite truth nor implementation blame established"
         and all(type(value.get(k)) is int and value[k] == expected for k, expected in
                 (("fixture", 108), ("n", 6), ("rank", 6), ("coefficient_count", 210)))
         and value["same_finite_binary128_words_across_pole_selectors"] == {"collier": True, "tirust": True}
         and value["qp_full_partial_pole_binary128_words_equal"] is True,
         "Finite audit fixture, selector or qualification scope differs")
    ranks = value["rank_summaries"]
    need(len(ranks) == 7 and [r["rank"] for r in ranks] == list(range(7))
         and [r["coefficients"] for r in ranks] == [1, 4, 10, 20, 35, 56, 84],
         "Finite audit rank coverage differs")
    rank_maxima, counts, offset = [], Counter(), 0
    for rank in ranks:
        parts = rank["parts"]
        need(len(parts) == 2 and {p["part"] for p in parts} == {"real", "imaginary"},
             "Finite audit real/imaginary coverage differs")
        for part in parts:
            count = part["mismatched_at_3e6"]
            need(integer(count) and count <= rank["coefficients"]
                 and type(part["worst_index"]) is int
                 and offset <= part["worst_index"] < offset + rank["coefficients"],
                 "Finite audit rank-part count/index differs")
            counts[part["part"]] += count
            rank_maxima.append(number(part["maximum_symmetric_residual"]))
        offset += rank["coefficients"]
    suspects = value["suspect_rows"]
    need(suspects and len({(r["index"], r["part"]) for r in suspects}) == len(suspects),
         "Finite audit suspect parts are empty or duplicated")
    for row in suspects:
        need(row["part"] in ("real", "imaginary") and type(row["index"]) is int
             and 0 <= row["index"] < 210 and len(row["powers"]) == 4
             and all(integer(p) for p in row["powers"]) and sum(row["powers"]) == row["rank"]
             and row["tirust_action"] == 3007 and row["collier_action"] == 3013
             and number(row["symmetric_residual"]) >= Decimal("3e-6"),
             "Finite audit suspect-part attribution differs")
    observed_counts = Counter(r["part"] for r in suspects)
    need(observed_counts == counts
         and type(value["finite_mismatched_parts"]) is int and value["finite_mismatched_parts"] == len(suspects)
         and value["finite_mismatched_real_parts"] == counts["real"]
         and value["finite_mismatched_imaginary_parts"] == counts["imaginary"],
         "Finite audit mismatch counts differ")
    worst = max(suspects, key=lambda r: number(r["symmetric_residual"]))
    finite_max = number(worst["symmetric_residual"])
    need(finite_max == max(rank_maxima) == number(value["maximum_finite_symmetric_residual"]),
         "Finite audit finite maximum differs from rank/part evidence")
    poles = value["pole_summaries"]
    need(len(poles) == 2 and {p["slot"] for p in poles} == {"simple", "double"},
         "Finite audit pole coverage differs")
    maxima = {"finite": finite_max}
    pole_view = {}
    for pole in poles:
        maxima[pole["slot"]] = number(pole["maximum_symmetric_residual"])
        ratio = number(pole["maximum_pole_to_finite_ratio"])
        need(all(integer(pole[key]) and pole[key] <= 210 for key in
                 ("tirust_nonzero_complex_coefficients", "collier_nonzero_complex_coefficients")),
             "Finite audit pole coefficient count differs")
        pole_view[pole["slot"]] = {
            "maximum_symmetric_residual": str(maxima[pole["slot"]]),
            "maximum_pole_to_finite_ratio": str(ratio),
            "tirust_nonzero_complex_coefficients": pole["tirust_nonzero_complex_coefficients"],
            "collier_nonzero_complex_coefficients": pole["collier_nonzero_complex_coefficients"],
        }
    aggregate_slot = max(maxima, key=maxima.get)
    states = value["n6_row_states"]
    need(len(states) == 15 and [r["action"] for r in states] == list(range(3001, 3016))
         and all(len(r["state"]) == 7 and all(type(v) is int for v in r["state"])
                 and type(r["adapter_stable"]) is bool for r in states),
         "Finite audit comparator state coverage differs")
    comparator = states[-3:]
    need(all(r["state"][-2:] == [0, -1] for r in comparator),
         "Finite audit comparator requested-accuracy observations differ")
    return {
        "kind": "common_input_finite_audit", "retained_runtime_index": runtime_index,
        "retained_runtime_index_sha256": source["sha256"], "fixture": 108, "n": 6, "rank": 6,
        "coefficient_count": offset, "finite_mismatched_parts": len(suspects),
        "finite_mismatched_real_parts": counts["real"], "finite_mismatched_imaginary_parts": counts["imaginary"],
        "maximum_finite_symmetric_residual": str(finite_max),
        "maximum_aggregate_symmetric_residual": str(maxima[aggregate_slot]),
        "maximum_aggregate_slot": aggregate_slot,
        "worst_finite_index": worst["index"], "worst_finite_part": worst["part"],
        "worst_finite_powers": worst["powers"], "pole_summaries": pole_view,
        "comparator_requested_accuracy_miss_rows": len(comparator),
        "summary_basis": "bound finite/part and pole audit summaries",
        "cause_established": False, "physical_accuracy_qualified": False,
        "performance_qualified": False, "native_evaluations": 0,
    }
