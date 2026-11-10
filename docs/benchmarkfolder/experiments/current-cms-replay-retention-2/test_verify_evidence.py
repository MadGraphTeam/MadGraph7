"""Pure byte-reader tests of the closed successor startup evidence.

Only temporary or in-memory evidence is mutated. No driver is imported and no
compiler, loader, inherited binary, or native child is launched.
"""
import copy
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
INDEX = "evidence-index.json"
SUMMARY = "summary.json"
CASE = "cms_ew_n2_epem_mupmum"
IDS = ["baseline-1-auto", "candidate-1-auto", "candidate-6-auto", "candidate-7-auto", "candidate-8-auto", "candidate-9-auto"]
SPEC = importlib.util.spec_from_file_location("cms_retention2_verifier_tests", HERE / "verify_evidence.py")
V = importlib.util.module_from_spec(SPEC)
with patch.object(subprocess, "Popen", side_effect=AssertionError("Verifier import must not launch commands")):
    SPEC.loader.exec_module(V)


def encoded(value, *, allow_nan=False):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=allow_nan) + "\n").encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class MemoryBytes:
    def __init__(self, files):
        self.files, self.calls = files, []

    @property
    def paths(self):
        return tuple(sorted(self.files))

    def __call__(self, name):
        if not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts:
            raise AssertionError("Nonrelative evidence access")
        self.calls.append(name)
        return self.files[name]


def seal(files):
    records = {}
    for name, raw in sorted(files.items()):
        if name == INDEX:
            continue
        decoded = gzip.decompress(raw) if name.endswith(".gz") else raw
        decoded.decode("utf8", errors="strict")
        records[name] = {"bytes": len(raw), "sha256": sha(raw),
                         "codec": "gzip-utf8" if name.endswith(".gz") else "utf8",
                         "decoded_bytes": len(decoded), "decoded_sha256": sha(decoded)}
    files[INDEX] = encoded({"schema_version": 1, "role": "current-cms-replay", "closed": True,
                            "stage": "native_terminal", "binding": copy.deepcopy(V.FIXED_BINDING),
                            "files": records, "summary": SUMMARY})
    return MemoryBytes(files)


def document(files, name):
    return json.loads(files[name])


def alter_json(files, name, change):
    value = document(files, name)
    change(value)
    files[name] = encoded(value)


def closed_bytes(files, name):
    item = document(files, "retention.json")["closed_files"][name]
    return gzip.decompress(files[item["path"]])


def replace_closed(files, name, raw):
    retention = document(files, "retention.json")
    item = retention["closed_files"][name]
    files[item["path"]] = gzip.compress(raw, mtime=0)
    item.update(bytes=len(raw), sha256=sha(raw))
    retention["original_bytes"] = sum(record["bytes"] for record in retention["closed_files"].values())
    files["retention.json"] = encoded(retention)


def alter_closed(files, name, change, *, allow_nan=False):
    value = json.loads(closed_bytes(files, name))
    change(value)
    replace_closed(files, name, encoded(value, allow_nan=allow_nan))


def result_bundle(files, request_id, change, *, allow_nan=False):
    """Keep duplicate result copies coherent to exercise derivation from raw data."""
    position = IDS.index(request_id)
    prefix = f"runs/{CASE}/{request_id}/"
    alter_closed(files, prefix + "result.json", change, allow_nan=allow_nan)
    alter_closed(files, prefix + "completion.json", lambda value: change(value["result"]), allow_nan=allow_nan)
    alter_closed(files, f"request-completions/{position + 1:02d}.json", lambda value: change(value["result"]), allow_nan=allow_nan)
    alter_closed(files, "summary.json", lambda value: change(value["requests"][position]["result"]), allow_nan=allow_nan)


def request_bundle(files, request_id, change):
    position = IDS.index(request_id)
    alter_closed(files, f"runs/{CASE}/{request_id}/specification.json", lambda value: change(value["request"]))
    result_bundle(files, request_id, lambda value: change(value["request"]))
    alter_closed(files, f"request-completions/{position + 1:02d}.json", lambda value: change(value["request"]))
    alter_closed(files, "summary.json", lambda value: change(value["requests"][position]["request"]))


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = {str(path.relative_to(HERE)): path.read_bytes() for path in HERE.rglob("*")
                 if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
                 and path.relative_to(HERE).as_posix() not in (INDEX, SUMMARY)}
        with patch.object(subprocess, "Popen", side_effect=AssertionError("Summary derivation must not launch commands")):
            cls.expected_summary = V.build_summary(MemoryBytes(files))
        files[SUMMARY] = encoded(cls.expected_summary)
        seal(files)
        cls.original = files

    def setUp(self):
        self.block = patch.object(subprocess, "Popen", side_effect=AssertionError("No subprocess permitted"))
        self.block.start()
        self.addCleanup(self.block.stop)
        self.files = dict(self.original)

    def rejected(self, reader=None):
        with self.assertRaises(V.EvidenceError):
            V.verify(read_bytes=reader or MemoryBytes(self.files))

    def semantic_rejection(self):
        reader = seal(self.files)
        with self.assertRaises(V.EvidenceError):
            V.build_summary(reader)
        self.rejected(reader)

    def test_complete_capsule_is_incomplete_numerically_despite_terminal_zero(self):
        result = V.verify(read_bytes=MemoryBytes(self.files))
        self.assertTrue(result["verified"])
        self.assertEqual(result["index_sha256"], sha(self.files[INDEX]))
        summary = result["summary"]
        self.assertEqual(summary, self.expected_summary)
        self.assertEqual(summary["status"], "incomplete")
        self.assertEqual(summary["overall_status"], "incomplete")
        self.assertEqual(summary["terminal_exit_code"], 0)
        self.assertIsNone(summary["amplitude_evaluations"])
        self.assertEqual(summary["failure_kind"], "warmup_incomplete_or_rejected")
        self.assertEqual([summary[key] for key in ("completed_requests", "amplitude_launch_count", "worker_launch_count", "physics_loader_launch_count")], [6, 6, 6, 6])
        self.assertEqual(summary["bootstrap_rows"], 12)
        self.assertEqual(summary["warmup_observed_rows"], 242)
        self.assertEqual(summary["W03_count"], 5)
        self.assertEqual(summary["complete_checked_streams"], 0)
        self.assertEqual(summary["full_final_streams"], 0)
        self.assertEqual(summary["comparisons"], [])
        self.assertIsNone(summary["primary_policy_agreement"])
        self.assertIs(summary["independent_finite_term_oracle"], False)
        self.assertIs(summary["performance_eligible"], False)
        self.assertIs(summary["numerically_qualified"], False)
        self.assertEqual(summary["preflight_origin"], "inherited")
        self.assertEqual(summary["setup_provenance"]["kind"], "inherited")
        self.assertEqual([summary["setup_provenance"][key] for key in
                          ("successor_setup_subprocess_launch_count", "successor_mass_probe_launch_count",
                           "successor_CMS_model_probe_launch_count")], [0, 0, 0])
        self.assertEqual(summary["gates"], {"preparation_passed": True, "preflight_passed": True,
                                            "native_run_passed": True, "numerically_qualified": False,
                                            "independent_finite_term_oracle": False})
        self.assertEqual(summary["stages"]["inherit"], {"status": "passed", "exit_code": 0})
        self.assertEqual(summary["stages"]["run"], {"status": "passed", "exit_code": 0})

    def test_delivered_seal_and_summary_verify_without_resealing(self):
        if not (HERE / INDEX).is_file() or not (HERE / SUMMARY).is_file():
            self.skipTest("The final capsule seal has not been written yet")
        result = V.verify(HERE)
        self.assertTrue(result["verified"])
        self.assertEqual(result["summary"], json.loads((HERE / SUMMARY).read_bytes()))

    def test_callback_boundary_includes_nested_ancestor_verification(self):
        reader = MemoryBytes(self.files)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Filesystem escape from byte callback")):
            result = V.verify(read_bytes=reader)
        self.assertTrue(result["verified"])
        self.assertIn("ancestor1/verify_evidence.py", reader.calls)
        self.assertIn("ancestor1/evidence-index.json", reader.calls)

    def test_every_request_preserves_warmup_and_empty_final_coverage(self):
        summary = V.verify(read_bytes=MemoryBytes(self.files))["summary"]
        expected = {"baseline-1-auto": (40, 0, 2), "candidate-1-auto": (40, 0, 2),
                    "candidate-6-auto": (38, 2, 2), "candidate-7-auto": (40, 2, 0),
                    "candidate-8-auto": (38, 2, 2), "candidate-9-auto": (38, 2, 2)}
        self.assertEqual([row["id"] for row in summary["requests"]], IDS)
        self.assertEqual([row["W03_count"] for row in summary["requests"]], [0, 0, 1, 2, 1, 1])
        self.assertEqual([row["bootstrap_rows"] for row in summary["requests"]], [2] * 6)
        self.assertEqual([row["warmup_observed_rows"] for row in summary["requests"]], [40, 40, 40, 42, 40, 40])
        self.assertEqual([row["native_cleanup_returncode"] for row in summary["requests"]], [-15, -15, 3, 3, 3, 3])
        self.assertTrue(all(row["worker_closed"] is True for row in summary["requests"]))
        for row in summary["requests"]:
            accepted, rejected, missing = expected[row["id"]]
            self.assertIs(row["execution_started"], True)
            self.assertIs(row["amplitude_observed"], True)
            self.assertIs(row["numerically_qualified"], False)
            self.assertEqual(row["state"], "startup_timeout" if row["backend"] == 1 else "warmup_rejected")
            self.assertEqual(row["phase_counts"]["warmup"],
                             {"accepted": accepted, "rejected": rejected, "unchecked": 0, "unexecuted": missing})
            self.assertEqual(row["phase_counts"]["final"],
                             {"accepted": 0, "rejected": 0, "unchecked": 0, "unexecuted": 42})
            for phase in ("warmup", "final"):
                self.assertEqual([(item["point"], item["order"]) for item in row["phase_outcomes"][phase]],
                                 [(point, order) for point in range(1, 22) for order in (0, 1)])
        self.assertEqual({state: sum(row["phase_counts"]["warmup"][state] for row in summary["requests"])
                          for state in ("accepted", "rejected", "unchecked", "unexecuted")},
                         {"accepted": 234, "rejected": 8, "unchecked": 0, "unexecuted": 10})
        self.assertEqual(sum(row["phase_counts"]["final"]["unexecuted"] for row in summary["requests"]), 252)

    def test_exact_inventory_and_raw_byte_digests_are_required(self):
        self.files["unindexed.txt"] = b"unbound\n"
        self.rejected()
        self.files = dict(self.original)
        self.files.pop("closure.json")
        self.rejected()
        self.files = dict(self.original)
        self.files["closed2/stage-run.json.gz"] += b"changed"
        self.rejected()
        self.files = dict(self.original)
        self.rejected(lambda name: self.files[name])

    def test_ancestor_verifier_and_sealed_successor_sources_cannot_be_replaced(self):
        for name in ("ancestor1/verify_evidence.py", "source/supervisor2/run.py"):
            self.files = dict(self.original)
            self.files[name] += b"\n# changed retained code\n"
            self.semantic_rejection()

    def test_ancestor_failed_summary_cannot_be_promoted(self):
        alter_json(self.files, "ancestor1/summary.json", lambda value: value.update(overall_status="passed"))
        self.semantic_rejection()

    def test_root_terminal_attestation_and_exit_must_match_closed_stages(self):
        for mutate in (lambda value: value.update(root_observed_terminal=False),
                       lambda value: value["sessions"]["run"].update(exit_code=1),
                       lambda value: value.update(inherited_checkpoint_sha256="0" * 64)):
            self.files = dict(self.original)
            alter_json(self.files, "closure.json", mutate)
            self.semantic_rejection()

    def test_six_controller_completions_do_not_permit_missing_or_changed_accounting(self):
        for mutate in (lambda value: value.update(completed=5),
                       lambda value: value.update(requested=5),
                       lambda value: value.update(not_completed=[{"id": IDS[-1]}])):
            self.files = dict(self.original)
            alter_closed(self.files, "request-accounting.json", mutate)
            self.semantic_rejection()
        self.files = dict(self.original)
        alter_closed(self.files, "request-completions/06.json", lambda value: value["request"].update(id=IDS[0]))
        self.semantic_rejection()

    def test_passed_stages_require_no_exception_and_correct_native_boundary(self):
        for mutate in (lambda value: value.update(status="failed"),
                       lambda value: value.update(exception="RuntimeError('unexpected')"),
                       lambda value: value.update(native_execution=False)):
            self.files = dict(self.original)
            alter_closed(self.files, "stage-run.json", mutate)
            self.semantic_rejection()

    def test_coherent_result_copies_cannot_promote_startup_to_final_success(self):
        for change in (lambda value: value.update(numerically_qualified=True),
                       lambda value: value.update(failure_phase="traversal"),
                       lambda value: value.update(fatal_controller_failure=True)):
            self.files = dict(self.original)
            result_bundle(self.files, "candidate-9-auto", change)
            self.semantic_rejection()

    def test_warmup_bootstrap_rejection_and_missing_rows_are_rederived_from_raw(self):
        changes = [lambda value: value["diagnostics"]["bootstrap_rows"][0].update(eligible=True),
                   lambda value: value["diagnostics"]["phase_outcomes"]["warmup"][38].update(state="accepted"),
                   lambda value: value["diagnostics"]["phase_outcomes"]["traversal"][0].update(state="accepted"),
                   lambda value: value["diagnostics"]["warmup_rows"][-1].update(accuracy=0.0, eligible=True)]
        for change in changes:
            self.files = dict(self.original)
            result_bundle(self.files, "candidate-9-auto", change)
            self.semantic_rejection()

    def test_protocol_failure_point_and_status_must_match_retained_warmup(self):
        result_bundle(self.files, "candidate-9-auto",
                      lambda value: value["protocol_messages"][0].update(message=["ERROR", "warmup-status", "21", "410"]))
        self.semantic_rejection()

    def test_raw_warmup_cannot_change_under_coherently_updated_result_hashes(self):
        name = f"runs/{CASE}/candidate-9-auto/warmup.dat"
        raw = closed_bytes(self.files, name).replace(b"         420", b"         219")
        replace_closed(self.files, name, raw)
        result_bundle(self.files, "candidate-9-auto", lambda value: value["raw_outputs_sha256"].update({"warmup.dat": sha(raw)}))
        self.semantic_rejection()

    def test_w03_cannot_disappear_from_all_duplicate_result_copies(self):
        result_bundle(self.files, "candidate-7-auto", lambda value: value.update(incomplete_warnings=[]))
        self.semantic_rejection()

    def test_w03_context_cannot_be_empty_or_truncated_in_coherent_copies(self):
        for empty in (False, True):
            self.files = dict(self.original)
            def change(value):
                warning = value["incomplete_warnings"][0]
                warning["context"] = "" if empty else warning["context"].splitlines()[0]
            result_bundle(self.files, "candidate-6-auto", change)
            self.semantic_rejection()

    def test_explicit_nonfinite_diagnostics_and_bare_diagnostic_nan_normalize_equally(self):
        def change(value):
            value["diagnostics"]["warmup_rows"][38]["coefficients"][1] = math.nan
            value["diagnostics"]["warmup_rows"][38]["coefficients"][3] = math.inf
        result_bundle(self.files, "candidate-6-auto", change, allow_nan=True)
        result = V.verify(read_bytes=seal(self.files))
        self.assertTrue(result["verified"])
        self.assertEqual(result["summary"], self.expected_summary)

    def test_nonfinite_outside_diagnostics_and_in_own_summary_is_rejected(self):
        result_bundle(self.files, "candidate-6-auto", lambda value: value.update(accuracy_limit=math.nan), allow_nan=True)
        self.semantic_rejection()
        self.files = dict(self.original)
        summary = document(self.files, SUMMARY)
        summary["amplitude_evaluations"] = math.nan
        self.files[SUMMARY] = encoded(summary, allow_nan=True)
        self.rejected(seal(self.files))

    def test_nonfinite_diagnostic_labels_and_positions_must_match_raw_evidence(self):
        for change in (lambda value: value["diagnostics"]["warmup_rows"][38]["coefficients"].__setitem__(3, {"nonfinite": "NaN"}),
                       lambda value: value["diagnostics"]["warmup_rows"][0].update(accuracy=math.nan)):
            self.files = dict(self.original)
            result_bundle(self.files, "candidate-6-auto", change, allow_nan=True)
            self.semantic_rejection()

    def test_duplicate_json_keys_are_rejected(self):
        original = self.files["closure.json"]
        self.files["closure.json"] = b'{"schema_version": 1,' + original.lstrip()[1:]
        self.rejected(seal(self.files))
        self.files = dict(self.original)
        name = f"runs/{CASE}/candidate-9-auto/result.json"
        raw = closed_bytes(self.files, name)
        replace_closed(self.files, name, b'{"orchestration_complete": true,' + raw.lstrip()[1:])
        self.semantic_rejection()

    def test_cleanup_and_owned_controller_returncode_must_prove_completion(self):
        result_bundle(self.files, "candidate-9-auto", lambda value: value["cleanup_receipts"][0].update(complete=False))
        self.semantic_rejection()
        self.files = dict(self.original)
        name = f"execution/request-{CASE}-candidate-9-auto.json"
        alter_closed(self.files, name, lambda value: value.update(returncode=1))
        self.semantic_rejection()
        self.files = dict(self.original)
        result_bundle(self.files, "candidate-9-auto", lambda value: value["cleanup_receipts"][0].update(pid=9999999))
        self.semantic_rejection()

    def test_owned_environment_cannot_escape_private_paths_or_inject_libraries(self):
        name = f"execution/request-{CASE}-candidate-9-auto.json"
        for key, value in (("PWD", "/tmp/wrong-worker"), ("LD_PRELOAD", "/tmp/unbound-library.so")):
            self.files = dict(self.original)
            alter_closed(self.files, name, lambda receipt: receipt["environment"].update({key: value}))
            self.semantic_rejection()

    def test_owned_python_and_adapter_paths_must_match_sealed_invocation(self):
        name = f"execution/request-{CASE}-candidate-9-auto.json"
        for index, changed in ((3, "/tmp/unbound-python3"),
                               (5, "/tmp/fake/current-cms-replay-supervisor-1/native_job.py")):
            self.files = dict(self.original)
            alter_closed(self.files, name, lambda value: value["argv"].__setitem__(index, changed))
            self.semantic_rejection()

    def test_native_specification_and_coherent_request_copies_require_exact_worker_paths(self):
        name = f"runs/{CASE}/candidate-9-auto/specification.json"
        mutations = [("argv_without_fifo", ["taskset", "-c", "12", "./check", "/wrong/points.dat"]),
                     ("cwd", "/tmp/unbound-worker/SubProcesses/P0_epem_mupmum"),
                     ("fifo_argument_appended_by_worker", "/tmp/worker-protocol.fifo")]
        for key, value in mutations:
            self.files = dict(self.original)
            request_bundle(self.files, "candidate-9-auto", lambda request: request.update({key: value}))
            if key != "fifo_argument_appended_by_worker":
                alter_closed(self.files, name, lambda spec: spec.update({key: value}))
            self.semantic_rejection()

    def test_frozen_policy_and_baseline_ignored_fields_cannot_change_in_metadata(self):
        for request_id, change in (("candidate-9-auto", lambda value: value["settings"]["requested"].update(mlstabthres=1e-6)),
                                   ("candidate-9-auto", lambda value: value["settings"]["requested"].update(osthres=1e-6)),
                                   ("baseline-1-auto", lambda value: value["settings"].update(not_applicable_to_baseline={})),
                                   ("candidate-9-auto", lambda value: value["numerical_cards"].__setitem__(
                                       "unbound/MadLoopParams.dat", value["numerical_cards"].pop("Cards/MadLoopParams.dat")))):
            self.files = dict(self.original)
            alter_closed(self.files, f"runs/{CASE}/{request_id}/specification.json", change)
            self.semantic_rejection()

    def test_runtime_capture_metadata_cannot_disagree_with_expected_specification(self):
        result_bundle(self.files, "candidate-9-auto",
                      lambda value: value["actual_map_captures"][0]["elf_files"].pop(next(iter(value["actual_map_captures"][0]["elf_files"]))))
        self.semantic_rejection()

    def test_rebound_raw_maps_and_stdout_hashes_still_require_matching_parsed_evidence(self):
        name = f"runs/{CASE}/candidate-9-auto/maps-000.txt"
        raw = closed_bytes(self.files, name).replace(b"libmadloop_tirust.so", b"libimpostor.so")
        replace_closed(self.files, name, raw)
        result_bundle(self.files, "candidate-9-auto", lambda value: value["actual_map_captures"][0].update(sha256=sha(raw)))
        self.semantic_rejection()
        self.files = dict(self.original)
        name = f"runs/{CASE}/candidate-6-auto/stdout.txt"
        raw = b"\n".join(line for line in closed_bytes(self.files, name).split(b"\n") if b"##W03" not in line)
        replace_closed(self.files, name, raw)
        result_bundle(self.files, "candidate-6-auto", lambda value: value.update(stdout_sha256=sha(raw)))
        self.semantic_rejection()

    def test_coherently_rebound_nonlocale_data_mapping_is_not_admitted(self):
        prefix = f"runs/{CASE}/candidate-9-auto/"
        result = json.loads(closed_bytes(self.files, prefix + "result.json"))
        old = next(iter(result["actual_map_captures"][0]["data_files"]))
        new = "/tmp/unbound-locale-data"
        raw = closed_bytes(self.files, prefix + "maps-000.txt").replace(old.encode(), new.encode())
        replace_closed(self.files, prefix + "maps-000.txt", raw)
        def change(value):
            capture = value["actual_map_captures"][0]
            capture["data_files"][new] = capture["data_files"].pop(old)
            capture["sha256"] = sha(raw)
        result_bundle(self.files, "candidate-9-auto", change)
        self.semantic_rejection()

    def test_finite_outer_summary_cannot_invent_qualified_oracle_or_comparison(self):
        alter_json(self.files, SUMMARY, lambda value: value.update(status="passed", overall_status="passed",
                                                                  primary_policy_agreement=True, comparisons=[{"passed": True}]))
        self.rejected(seal(self.files))

    def test_filesystem_reader_rejects_symlinked_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, raw in self.files.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            path = root / "closure.json"
            path.unlink()
            path.symlink_to(HERE / "closure.json")
            with self.assertRaises(V.EvidenceError):
                V.verify(root)


if __name__ == "__main__":
    unittest.main()
