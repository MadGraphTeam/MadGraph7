"""Read-only sealed parser review and focused faults on closed CT4 bytes.

No source or actual attempt is changed. Numerical workers, compilers and loaders
are forbidden; temporary fixture tests use only their existing synthetic bytes.
"""
import argparse
import ast
import copy
import gzip
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
import unittest
from unittest import mock

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SOURCE = BASE / "source-proposal4"
FAILED = BASE / "failed-run3-retention-1"
ROOT = HERE.parents[4]
WORK = ROOT / ".tirust-build/qcd-n3-genuine-qp-diagnostic"
FAILED_SEAL = "50fcb62f7d2b8b725e409755caf396f5a5f682e08edb07640d90a246a3201b0c"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def index_check(directory, name, pin):
    path = directory / name
    require(sha(path) == pin, "Index seal mismatch")
    value = json.loads(path.read_bytes())
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file() or p.is_symlink()}
    require(actual == set(value["files"]) | {name}, "Index membership mismatch")
    for member, item in value["files"].items():
        p = directory / member
        require(not p.is_symlink() and p.stat().st_size == item["bytes"] and sha(p) == item["sha256"], "Indexed payload mismatch")
    return value


def functions(raw):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(raw).body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-index-sha", required=True)
    args = parser.parse_args()
    require(not (HERE / "review.json").exists(), "Refuse to overwrite review")
    candidate = index_check(SOURCE, "source-index.json", args.source_index_sha)
    frozen = index_check(FAILED, "evidence-index.json", FAILED_SEAL)
    pins = {str(SOURCE / name): item["sha256"] for name, item in candidate["files"].items()}
    pins[str(SOURCE / "source-index.json")] = args.source_index_sha
    pins.update({str(FAILED / name): item["sha256"] for name, item in frozen["files"].items()})
    pins[str(FAILED / "evidence-index.json")] = FAILED_SEAL
    binding = json.loads((SOURCE / "bindings.json").read_bytes())
    pins.update(binding["dependencies"])
    for path, wanted in pins.items():
        require(sha(path) == wanted, "Before-review pin mismatch: " + path)
    inventory = json.loads((FAILED / "retention-inventory.json").read_bytes())
    payload = {}
    for name, record in inventory["original_files"].items():
        raw = (FAILED / record["member"]).read_bytes()
        raw = gzip.decompress(raw) if record["codec"] == "gzip" else raw
        require(hashlib.sha256(raw).hexdigest() == record["raw_sha256"], "Retained raw bytes changed")
        payload[name] = raw
        original_path = Path(record["original_path"])
        original_path = original_path if original_path.is_absolute() else ROOT / original_path
        pins[str(original_path)] = record["raw_sha256"]
        require(sha(original_path) == record["raw_sha256"], "Actual original source3/run3 bytes changed against closed capture")
    actual_run_files = {p.relative_to(WORK / "run3").as_posix() for p in (WORK / "run3").rglob("*")
                        if p.is_file() and not p.is_symlink()}
    retained_run_files = {name[len("actual-run3/"):] for name in inventory["original_files"] if name.startswith("actual-run3/")}
    require(actual_run_files == retained_run_files, "Actual closed run3 regular-file membership changed")
    actual_run_links = {p.relative_to(WORK / "run3").as_posix(): str(p.readlink())
                       for p in (WORK / "run3").rglob("*") if p.is_symlink()}
    require(actual_run_links == inventory["actual_attempt_symlinks"], "Actual closed run3 alias inventory changed")
    old_review_dir = BASE / "source3-independent-review-1"
    old_review = index_check(old_review_dir, "index.json", "b015e81c91a758e90bb2022712774d52062d914b2864b5785cb4af6ece2ead1f")
    old_record = json.loads((old_review_dir / "review.json").read_bytes())
    for path, wanted in old_record["closed_build2_run2_reference_hashes"].items():
        require(sha(path) == wanted, "Actual inherited build2/run2 proof input changed")
        pins[path] = wanted
    prior_ast = functions(payload["controller/analyze.py"])
    now_ast = functions((SOURCE / "analyze.py").read_bytes())
    changed = sorted(name for name in set(prior_ast) | set(now_ast) if prior_ast.get(name) != now_ast.get(name))
    require(changed == ["observed_inputs", "widen_binary64"], "Unexpected parser function changes")
    prior_run = functions(payload["controller/run.py"])
    now_run = functions((SOURCE / "run.py").read_bytes())
    run_changed = sorted(name for name in set(prior_run) | set(now_run) if prior_run.get(name) != now_run.get(name))
    require(run_changed == ["build", "run", "validate_target_closure"], "Unexpected run controller changes")
    source3_observed = next(n for n in ast.parse(payload["controller/analyze.py"]).body if isinstance(n, ast.FunctionDef) and n.name == "observed_inputs")
    source4_observed = next(n for n in ast.parse((SOURCE / "analyze.py").read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == "observed_inputs")

    def pieces(node):
        return [ast.dump(n, include_attributes=False) for n in node.body
                if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pieces" for t in n.targets)
                or isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name) and n.target.id == "pieces"]

    require(pieces(source3_observed) == pieces(source4_observed), "Cross-worker common-input pieces changed")
    unchanged_scientific = ["observer.f90", "observer.c", "driver.f", "supervise.py", "exec_target.py", "donor_closure.py", "shapes.py"]
    for name in unchanged_scientific:
        require((SOURCE / name).read_bytes() == payload["controller/" + name], "Native/helper payload changed")
    for name in candidate["files"]:
        if name.startswith("fixtures/"):
            require((SOURCE / name).read_bytes() == payload["controller/" + name], "Physical fixture changed")
    guard = {"actual_process_or_loader_attempts": [], "protected_input_write_attempts": []}
    protected = (SOURCE, BASE / "source-proposal3", FAILED, WORK / "build2", WORK / "run3", WORK / "run4")

    def audit(event, arguments):
        if event in ("subprocess.Popen", "os.exec", "os.posix_spawn", "os.system", "ctypes.dlopen"):
            guard["actual_process_or_loader_attempts"].append(event)
            raise AssertionError("No actual process or inspected library may run")
        path = None
        if event == "open":
            p, mode, flags = arguments
            if (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                path = p
        elif event in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.truncate"):
            path = arguments[0]
        elif event in ("os.rename", "os.link", "os.symlink"):
            path = arguments[1]
        if isinstance(path, (str, bytes, os.PathLike)):
            absolute = Path(os.fsdecode(path)).absolute()
            if any(absolute == p or p in absolute.parents for p in protected):
                guard["protected_input_write_attempts"].append({"event": event, "path": str(absolute)})
                raise AssertionError("Actual source/attempt input writes forbidden")

    sys.addaudithook(audit)
    sys.path.insert(0, str(SOURCE))
    import analyze as m
    import run as controller
    import closed_replay
    selected = json.loads(payload["controller/fixtures/selected-inputs.json"])
    points = {p: b"".join(struct.pack("<d", float.fromhex(x)) for row in selected["points"][str(p)]["binary64_hex"] for x in row)
              for p in (1, 93, 39)}
    tests_log = io.StringIO()
    started = time.monotonic()
    with mock.patch.object(subprocess, "Popen", side_effect=AssertionError("No native subprocess")), mock.patch.object(os, "execve", side_effect=AssertionError("No real execve")):
        binding_checked = controller.source_check()
        controller.inherited_build(binding_checked)
        spec = importlib.util.spec_from_file_location("original_failed_replay", FAILED / "verify.py")
        original_verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original_verifier)
        original = original_verifier.verify(FAILED)
        require(original["overall_native_attempt_passed"] is False, "Original failure was reclassified")
        workers = []
        for observed in (0, 1):
            prefix = "actual-run3/b1-m4-o" + str(observed)
            workers.append(m.parse_worker(payload[prefix + "/trace.txt"].decode(), payload[prefix + "/worker.log"].decode(), 1, 4, observed, points))
        require(workers[0]["public_raw"] == workers[1]["public_raw"], "Actual two-worker public parity failed")
        regenerated_preflight = json.loads(json.dumps(closed_replay.replay(FAILED), allow_nan=False))
        retained_preflight = json.loads((SOURCE / "closed-run3-preflight.json").read_bytes())
        require(regenerated_preflight == retained_preflight, "Complete closed-run3 preflight JSON replay differs")
        require(regenerated_preflight["scientific_acceptance"] is False and regenerated_preflight["adjudication_executed"] is False,
                "Two-worker replay was overstated as scientific/matrix acceptance")
        _, rows = m.parse_trace(payload["actual-run3/b1-m4-o1/trace.txt"].decode())
        group = [r for r in rows if r["call"] == 101]
        base_input = workers[1]["points"][0]["inputs"]
        controls = []

        def modified(kind, part, offset, *, replacement=None):
            out = list(group)
            index = next(i for i, r in enumerate(out) if r["kind"] == kind and r["part"] == part)
            row = dict(out[index]); data = bytearray(row["data"])
            if replacement is None:
                data[offset] ^= 1
            else:
                data[offset:offset + len(replacement)] = replacement
            row["data"] = bytes(data); out[index] = row
            return out

        def rejected(name, mutated):
            try:
                m.observed_inputs(mutated, 1, 4)
            except ValueError as error:
                controls.append({"name": name, "rejected": True, "message": str(error)})
            else:
                raise AssertionError("Tampered closed state accepted: " + name)

        rejected("Born entry differs from original public point", modified(10, 0, 0))
        rejected("DP repair entry differs from original public point", modified(20, 0, 0))
        rejected("QP entry low bit differs from exact widening", modified(21, 0, 0))
        rejected("QP entry signed zero differs from exact widening", modified(21, 0, 31, replacement=b"\x80"))
        rejected("Born input mutated on return", modified(10, 1, 0))
        rejected("QP repair result differs from MP construction", modified(21, 1, 0))
        rejected("MP construction input mutated on return", modified(30, 1, 0))
        rejected("Helicity construction path input differs", modified(34, 0, 0))
        rejected("MP model after coupling update differs", modified(32, 1, 0))
        rejected("Native numerator differs from canonical QP construction", modified(110, 0, 0))
        rejected("Native reducer failure cannot use returned finite remnants", modified(101, 9, 0, replacement=struct.pack("<i", 0)))
        omitted = base_input["nonzero_coefficient_groups"][0]
        rejected("Omitted nonzero canonical coefficient group", [r for r in group if not (r["kind"] >= 100 and r["group"] == omitted)])
        rejected("Unstable host group flag", modified(40, 1, 248 + 696 * 8, replacement=struct.pack("<q", 0)))
        changed_dp = m.observed_inputs(modified(20, 1, 0), 1, 4)
        require(changed_dp["shared_input_sha256"] != base_input["shared_input_sha256"], "Changed DP repair after-state was dropped from cross-worker digest")
        controls.append({"name": "Changed independent DP after-state remains retained and changes common-input digest", "retained_separately": True, "digest_changed": True})
        try:
            m.parse_worker(payload["actual-run3/b1-m4-o0/trace.txt"].decode(), payload["actual-run3/b1-m4-o0/worker.log"].decode() + "\n##w03 omitted group\n", 1, 4, 0, points)
        except ValueError as error:
            controls.append({"name": "Case-insensitive W03 rejection", "rejected": True, "message": str(error)})
        else:
            raise AssertionError("W03 accepted")
        suite = unittest.defaultTestLoader.discover(str(SOURCE), pattern="test_*.py")
        tests = unittest.TextTestRunner(stream=tests_log, verbosity=2).run(suite)
        require(tests.testsRun == 68 and tests.wasSuccessful(), tests_log.getvalue())
    after = {p: sha(p) for p in pins}
    require(after == pins and not any(guard.values()), "Guard/input immutability failure")
    index_check(SOURCE, "source-index.json", args.source_index_sha)
    index_check(FAILED, "evidence-index.json", FAILED_SEAL)
    summaries = []
    for worker in workers:
        summaries.append({"observed": worker["observed"], "public_calls": worker["public_calls"], "points": [
            {"point": p["point"], "status": p["status"], "host_checked": p["host_checked"],
             "native_evaluations": p["inputs"]["native_evaluations"] if p["inputs"] else None,
             "QP_numerator_callbacks": p["inputs"]["qp_numerator_callbacks"] if p["inputs"] else None,
             "helicities": p["inputs"]["helicities"] if p["inputs"] else None,
             "changed_DP_components": p["inputs"]["input_transitions"]["dp_repair_changed_components"] if p["inputs"] else None,
             "shared_input_sha256": p["inputs"]["shared_input_sha256"] if p["inputs"] else None} for p in worker["points"]]})
    review = {"schema": "source4-independent-input-gate-review-1", "passed": True,
              "source4_index_sha256": args.source_index_sha, "source4_payload_count": len(candidate["files"]),
              "parser_sha256": sha(SOURCE / "analyze.py"), "closed_failure_index_sha256": FAILED_SEAL,
              "frozen_source3_attempt_remains_failed": True, "original_failure": original["overall_failure"],
              "changed_parser_functions": changed, "changed_run_functions": run_changed,
              "cross_worker_common_input_pieces_AST_unchanged": True,
              "other_parser_functions_including_native_routing_parseworker_and_adjudicate_unchanged": True,
              "native_Fortran_driver_observers_and_helpers_unchanged": unchanged_scientific,
              "actual_closed_run3_file_and_alias_membership_unchanged": True,
              "source3_and_generated_sources_match_closed_raw_capture": True,
              "build2_and_run2_match_prior_independent_review_pins": True,
              "closed_run3_preflight_JSON_replayed_exactly": True,
              "closed_run3_preflight_sha256": sha(SOURCE / "closed-run3-preflight.json"),
              "reviewed_source_relationships": [
                  "loop_matrix.f807 Born receives original P_USER before loop_matrix.f922 copies it to PS and f937 repairs the independent DP copy.",
                  "loop_matrix.f978 SET_MP_PS receives original P_USER; f2147 promotes to MP_PS and f2151 calls QP repair. MP construction receives that repaired QP state.",
                  "mp_compute_loop_coefs.f185 updates MP couplings, f192 copies PS, f274 traverses48helicities, f309/f398 call the observed amplitude/coefficient paths.",
                  "observer.f90 payload declarations and fixed-array extents match Born, independent repairs, MP construction, canonical common-block numerator and host group output boundaries.",
                  "All unchanged adjacent strict gates passed on the two actual complete CTmode4 workers, including native group success/coverage, IEEE/public parity, model consistency, canonical numerator and status120 treatment."],
              "closed_worker_replays": summaries, "public_plain_observed_bytes_equal": True,
              "focused_closed_byte_controls": controls, "pure_fixture_test_count": tests.testsRun,
              "pure_fixture_test_failures": len(tests.failures), "pure_fixture_test_errors": len(tests.errors),
              "checked_pin_count": len(pins), "all_checked_pins_unchanged": pins,
              "guards": guard, "native_or_compiler_or_loader_calls": 0,
              "adjudicate_sixteen_worker_matrix_invoked_on_actual_data": False,
              "coefficient_comparison_qualified": False, "scientific_acceptance": False,
              "unexecuted_matrix_workers": 14, "seconds": time.monotonic() - started,
              "limitations": ["Two closed CTmode4 workers were replayed, not the absent full sixteen-worker matrix.",
                              "Native actual-map evidence remains an observational sample, not a before-main barrier or continuous monitor.",
                              "All fixed diagnostic amplitudes remain unchecked status120; this is parser/source consistency review, not numerical acceptance."]}
    (HERE / "test.log").write_text(tests_log.getvalue())
    (HERE / "review.json").write_text(json.dumps(review, indent=2, sort_keys=True) + "\n")
    packet = {"schema": "source4-independent-input-gate-review-index-1", "scope": "Pure source/gate audit, retained two-worker replay and focused tamper controls; no native or scientific-acceptance claim.",
              "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in sorted(HERE.iterdir()) if p.is_file() and p.name != "index.json"}}
    (HERE / "index.json").write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"index_sha256": sha(HERE / "index.json"), "source4_index_sha256": args.source_index_sha,
                      "focused_control_count": len(controls), "fixture_tests": tests.testsRun,
                      "actual_closed_workers_replayed": 2, "checked_pin_count": len(pins), "passed": True}, indent=2))


if __name__ == "__main__":
    main()
