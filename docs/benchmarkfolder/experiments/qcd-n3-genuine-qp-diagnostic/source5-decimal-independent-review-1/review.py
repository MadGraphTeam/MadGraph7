"""Independent source5 decimal-only draft audit; controller remains out of scope.

Load only the stable parser/replay/tests. Synthetic fixture helpers are extracted
from the immutable source4 failure capsule, without importing any controller.
No compiler, loader or actual numerical executable may run.
"""
import ast
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import types
import unittest
from unittest import mock
from fractions import Fraction

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / "source-proposal5"
FAILED = BASE / "failed-run4-retention-1"
FAILURE_SEAL = "fcd4f75340f7db9261e91d6f0c37e7a1ba85b3ff68b8b45a639ecbcc49ddf361"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    require(not (HERE / "review.json").exists(), "Refuse to overwrite review")
    stable = [SOURCE / p for p in ("analyze.py", "test_decimal_outputs.py", "closed_run4_replay.py", "closed-run4-preflight.json")]
    pins = {str(p): sha(p) for p in stable}
    require(sha(FAILED / "evidence-index.json") == FAILURE_SEAL, "Closed failed run4 seal mismatch")
    index = json.loads((FAILED / "evidence-index.json").read_bytes())
    for name, item in index["files"].items():
        path = FAILED / name
        require(sha(path) == item["sha256"] and path.stat().st_size == item["bytes"], "Retained payload changed")
        pins[str(path)] = item["sha256"]
    pins[str(FAILED / "evidence-index.json")] = FAILURE_SEAL
    transport = BASE / "run4-static-transport-analysis-1"
    require(sha(transport / "index.json") == "53027747d94b197022b5033376b231aa90f7a92d7496d330c6879d8356c9b0f8", "Formatter proof seal mismatch")
    formatter_index = json.loads((transport / "index.json").read_bytes())
    for name, item in formatter_index["files"].items():
        require(sha(transport / name) == item["sha256"], "Formatter proof payload changed")
        pins[str(transport / name)] = item["sha256"]
    old_parser = (FAILED / "controller/analyze.py").read_bytes()
    new_parser = (SOURCE / "analyze.py").read_bytes()
    functions = lambda raw: {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)}
    before, after = functions(old_parser), functions(new_parser)
    require(set(after) - set(before) == {"ascii_zero_signs"}, "Unexpected new parser functions")
    changed = sorted(name for name in before if before[name] != after[name])
    require(changed == ["ascii113", "native_calls", "rounded113"], "Unrelated parser/input gate changed")
    guard = {"actual_native_or_loader_attempts": [], "source_or_capture_write_attempts": []}
    protected = (SOURCE, FAILED, BASE / "source-proposal4", transport)

    def audit(event, args):
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.system", "ctypes.dlopen"):
            guard["actual_native_or_loader_attempts"].append(event)
            raise AssertionError("No actual numerical/compiler/loader process or inspected library may run")
        path = None
        if event == "open":
            p, mode, flags = args
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                path = p
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate"):
            path = args[0]
        elif event in ("os.rename", "os.link", "os.symlink"):
            path = args[1]
        if isinstance(path, (str, bytes, os.PathLike)):
            absolute = Path(os.fsdecode(path)).absolute()
            if any(absolute == p or p in absolute.parents for p in protected):
                guard["source_or_capture_write_attempts"].append({"event": event, "path": str(absolute)})
                raise AssertionError("Source/capture modifications forbidden")

    sys.addaudithook(audit)
    started = time.monotonic()
    with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("No actual subprocess")), mock.patch.object(os, "execve", side_effect=AssertionError("No real execve")):
        parser = load_module("analyze", SOURCE / "analyze.py")
        replay_module = load_module("independent_closed_run4_candidate_replay", SOURCE / "closed_run4_replay.py")
        replayed = json.loads(json.dumps(replay_module.replay(FAILED), allow_nan=False))
        expected = json.loads((SOURCE / "closed-run4-preflight.json").read_bytes())
        require(replayed == expected, "Ten-worker preflight JSON replay differs")
        require(replayed["candidate_parser_full_workers_passed"] == 10 and replayed["matrix_workers_not_replayed"] == 6 and
                len(replayed["partial_common_input_checks"]) == 18 and replayed["all_partial_common_inputs_equal"] is True and
                replayed["scientific_acceptance"] is False and replayed["adjudication_executed"] is False,
                "Partial prefix qualification scope changed")
        helper_source = (FAILED / "controller/test_protocol.py").read_bytes()
        helper_tree = ast.parse(helper_source)
        helper_nodes = []
        for node in helper_tree.body:
            if isinstance(node, ast.ClassDef):
                break
            if isinstance(node, ast.Import) and any(alias.name in ("run", "shapes") for alias in node.names):
                continue
            helper_nodes.append(node)
        fixture = types.ModuleType("test_protocol")
        fixture.__dict__["__file__"] = str(FAILED / "controller/test_protocol.py")
        sys.modules["test_protocol"] = fixture
        exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), "immutable-source4-fixture-helpers", "exec"), fixture.__dict__)
        decimal_tests = load_module("independent_decimal_output_tests", SOURCE / "test_decimal_outputs.py")
        log = io.StringIO()
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(decimal_tests))
        require(tests.testsRun == 10 and tests.wasSuccessful(), log.getvalue())

        def record(text):
            raw = text.encode("ascii")
            require(len(raw) <= 64, "Synthetic record too wide")
            return raw.ljust(64, b" ")

        controls = []
        for kind, tokens, output in [
            ("exponentless input", ("0", "-0", "+0", "1", "-1", "0.0"), False),
            ("noncanonical/exponentless output", ("+0", "00", "-00", "0.", "0.0", "-0.0", "1", "-1", "1.5"), True),
            ("nonfinite/malformed output", ("NaN", "Inf", "Infinity", "-Inf", "1E", "1E+", ""), True),
            ("unqualified nonzero range", ("1E-4950", "-1E-4950", "1E-6000", "1E6000"), True)]:
            for token in tokens:
                try:
                    parser.ascii113(record(token), canonical_output_zero=output)
                except ValueError as error:
                    controls.append({"class": kind, "token": token, "rejected": True, "message": str(error)})
                else:
                    raise AssertionError("Malformed or out-of-domain record accepted")
        for raw in (b"0", record("0")[:-1], record("0") + b" "):
            try:
                parser.ascii113(raw, canonical_output_zero=True)
            except ValueError:
                controls.append({"class": "truncation/stride", "bytes": len(raw), "rejected": True})
            else:
                raise AssertionError("Wrong record width accepted")
        zeros = record("0") + record("-0")
        require(parser.ascii113(zeros, canonical_output_zero=True) == [0, 0], "Canonical output zeros not decoded")
        signs = parser.ascii_zero_signs(zeros)
        require(signs["positive_zero_indices"] == [0] and signs["negative_zero_indices"] == [1], "Lexical zero signs lost")
        beyond = parser.ascii113(record("1.000000000000000000000000000000788860905E0"))[0]
        require(beyond == Fraction(1) + Fraction(1, 2 ** 100) and beyond != Fraction(float(beyond)), "Decimal narrowed to DP")
        exact_edges = [(Fraction(1) + 3 * Fraction(2) ** -113, Fraction(1) + Fraction(2) ** -111),
                       (-(Fraction(1) + 3 * Fraction(2) ** -113), -(Fraction(1) + Fraction(2) ** -111)),
                       (Fraction(2) - Fraction(2) ** -113, Fraction(2)),
                       (Fraction(2) ** 16384 - 3 * Fraction(2) ** 16269, Fraction(2) ** 16384 - Fraction(2) ** 16271)]
        for value, rounded in exact_edges:
            require(parser.rounded113(value) == rounded, "Nearest-even edge mismatch")
        for sign in (1, -1):
            try:
                parser.rounded113(sign * (Fraction(2) ** 16384 - Fraction(2) ** 16270))
            except ValueError as error:
                require("rounded value outside qualified normal range" in str(error), "Unexpected rounded-overflow rejection")
            else:
                raise AssertionError("Rounded overflow accepted")
        failure_controls = []
        for backend in (8, 9):
            rows = fixture.sample(backend, 4, 1)
            fixture.find(rows, 100 + backend, 9)["data"] = fixture.J(2)
            rows.remove(fixture.find(rows, 100 + backend, 10))
            original_decode = parser.ascii113
            decode_options = []

            def spy(data, **options):
                decode_options.append(options.get("canonical_output_zero", False))
                return original_decode(data, **options)

            with mock.patch.object(parser, "ascii113", side_effect=spy):
                try:
                    fixture.parse(rows, backend, 4, 1)
                except ValueError as error:
                    require("refusal" in str(error), "Failure did not remain a native refusal")
                else:
                    raise AssertionError("Failed native output accepted")
            require(not any(decode_options), "Output-zero allowance used on failed native evaluation")
            failure_controls.append({"backend": backend, "native_refusal_rejected": True, "canonical_output_zero_decoder_called": False})
    require(all(sha(path) == expected for path, expected in pins.items()) and not any(guard.values()), "Review pin/guard failure")
    review = {"schema": "source5-decimal-independent-draft-review-1", "passed": True,
              "scope": "Stable parser/closed-prefix replay/decimal tests only; unsealed source5 controller and continuation are outside this review.",
              "stable_parser_sha256": sha(SOURCE / "analyze.py"), "stable_reviewed_files": {str(p): pins[str(p)] for p in stable},
              "failed_run4_retention_index_sha256": FAILURE_SEAL, "source4_attempt_remains_failed": True,
              "production_formatter_contract_index_sha256": "53027747d94b197022b5033376b231aa90f7a92d7496d330c6879d8356c9b0f8",
              "changed_existing_parser_functions": changed, "new_helper": "ascii_zero_signs",
              "all_other_parser_input_protocol_common_digest_and_adjudication_functions_unchanged": True,
              "ten_closed_worker_preflight_replayed_exactly": True, "full_actual_workers_reparsed": 10,
              "actual_public_parity_pairs": 5, "actual_partial_common_input_checks": len(replayed["partial_common_input_checks"]),
              "all_actual_partial_common_inputs_equal": True, "unexecuted_matrix_workers": 6,
              "decimal_unit_tests": tests.testsRun, "decimal_unit_test_failures": len(tests.failures), "decimal_unit_test_errors": len(tests.errors),
              "focused_rejection_controls": controls, "signed_zero_evidence": signs,
              "beyond_DP_exact_fraction": str(beyond), "nearest_even_extra_edges_checked": len(exact_edges),
              "positive_and_negative_rounded_overflow_rejected": True,
              "failed_native_output_controls": failure_controls,
              "guard": guard, "native_or_compiler_or_loader_calls": 0, "source5_sealed_by_this_review": False,
              "checked_pins_unchanged_before_after": pins, "seconds": time.monotonic() - started,
              "limitations": ["Only canonical bare0/-0 successful outputs8/9 are admitted; every input and nonzero decimal retains the exponent-required grammar.",
                              "Fraction comparisons cannot carry the sign of zero; lexical signs are retained separately as evidence.",
                              "Nonzero subnormal/underflow inputs and binary128 overflow remain unsupported explicit failures; no edge-domain qualification is claimed.",
                              "Six native suffix arms remain unexecuted. No sixteen-worker adjudication, coefficient comparison or scientific acceptance is claimed.",
                              "Controller continuation needs its separate audit and eventual source5 index binding."],
              "adjudication_executed": False, "coefficient_comparison_qualified": False, "scientific_acceptance": False}
    (HERE / "test.log").write_text(log.getvalue())
    (HERE / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    packet = {"schema": "source5-decimal-independent-draft-review-index-1", "scope": review["scope"],
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "index.json"}}
    (HERE / "index.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"index_sha256": sha(HERE / "index.json"), "parser_sha256": sha(SOURCE / "analyze.py"),
                      "actual_closed_workers": 10, "partial_common_checks": 18, "decimal_tests": tests.testsRun,
                      "focused_rejections": len(controls), "failed_native_output_controls": 2, "passed": True}, indent=2))


if __name__ == "__main__":
    main()
