"""Pure retention checks against in-memory mutations of the closed capsule.

These tests read retained bytes. They never import a driver or launch commands.
"""
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent
INDEX = "evidence-index.json"
SUMMARY = "summary.json"
SPEC = importlib.util.spec_from_file_location("cms_retention_verifier_tests", HERE / "verify_evidence.py")
V = importlib.util.module_from_spec(SPEC)
with patch.object(subprocess, "Popen", side_effect=AssertionError("Verifier import must not launch subprocesses")):
    SPEC.loader.exec_module(V)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


class MemoryBytes:
    def __init__(self, files):
        self.files = files
        self.calls = []

    @property
    def paths(self):
        return tuple(sorted(self.files))

    def __call__(self, name):
        if not isinstance(name, str) or name.startswith("/") or ".." in Path(name).parts:
            raise AssertionError("Reader received a nonrelative capsule path")
        self.calls.append(name)
        return self.files[name]


def seal(files):
    """Seal only an in-memory test copy; no original index is modified."""
    records = {}
    for name, raw in sorted(files.items()):
        if name == INDEX:
            continue
        decoded = gzip.decompress(raw) if name.endswith(".gz") else raw
        decoded.decode("utf8", errors="strict")
        records[name] = {"sha256": digest(raw), "bytes": len(raw),
                         "codec": "gzip-utf8" if name.endswith(".gz") else "utf8",
                         "decoded_sha256": digest(decoded), "decoded_bytes": len(decoded)}
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


def alter_closed(files, name, change):
    retention = document(files, "retention.json")
    item = retention["closed_files"][name]
    value = json.loads(gzip.decompress(files[item["path"]]))
    change(value)
    raw = encoded(value)
    files[item["path"]] = gzip.compress(raw, mtime=0)
    item.update(sha256=digest(raw), bytes=len(raw))
    # Keep byte totals coherent so the targeted contradiction reaches its gate.
    retention["original_bytes"] = sum(row["bytes"] for row in retention["closed_files"].values())
    files["retention.json"] = encoded(retention)


class RetentionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = {str(path.relative_to(HERE)): path.read_bytes()
                 for path in HERE.rglob("*") if path.is_file() and not path.is_symlink()
                 and "__pycache__" not in path.parts and path.name != INDEX}
        files.pop(SUMMARY, None)
        with patch.object(subprocess, "Popen", side_effect=AssertionError("Summary derivation must not launch subprocesses")):
            cls.expected_summary = V.build_summary(MemoryBytes(files))
        files[SUMMARY] = encoded(cls.expected_summary)
        seal(files)
        cls.original = files

    def setUp(self):
        self.no_spawn = patch.object(subprocess, "Popen", side_effect=AssertionError("No subprocess permitted"))
        self.no_spawn.start()
        self.addCleanup(self.no_spawn.stop)
        self.files = dict(self.original)

    def rejected(self, reader=None):
        with self.assertRaises(V.EvidenceError):
            V.verify(read_bytes=reader or MemoryBytes(self.files))

    def semantic_rejection(self):
        reader = seal(self.files)
        with self.assertRaises(V.EvidenceError):
            V.build_summary(reader)
        self.rejected(reader)

    def test_complete_retained_capsule_verifies_with_exact_derived_summary(self):
        result = V.verify(read_bytes=MemoryBytes(self.files))
        self.assertTrue(result["verified"])
        self.assertEqual(result["index_sha256"], digest(self.files[INDEX]))
        self.assertEqual(result["summary"], self.expected_summary)
        self.assertEqual(result["summary"]["role"], "current-cms-replay")
        self.assertIs(result["summary"]["closed"], True)
        self.assertEqual(result["summary"]["stage"], "native_terminal")
        self.assertEqual(result["summary"]["overall_status"], "failed")
        self.assertEqual(result["summary"]["stages"]["run"]["exit_code"], 1)

    def test_delivered_capsule_seal_and_summary_verify_without_resealing(self):
        if not (HERE / INDEX).is_file() or not (HERE / SUMMARY).is_file():
            self.skipTest("The final capsule seal has not been written yet")
        result = V.verify(HERE)
        self.assertTrue(result["verified"])
        self.assertEqual(result["index_sha256"], digest((HERE / INDEX).read_bytes()))
        self.assertEqual(result["summary"], json.loads((HERE / SUMMARY).read_bytes()))

    def test_memory_callback_is_complete_io_boundary(self):
        reader = MemoryBytes(self.files)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("Unexpected filesystem read")):
            result = V.verify(read_bytes=reader)
        self.assertTrue(result["verified"])
        self.assertIn(INDEX, reader.calls)
        self.assertIn("closure.json", reader.calls)
        self.assertTrue(all(not name.startswith("/") for name in reader.calls))

    def test_summary_marks_all_requested_amplitude_rows_unexecuted(self):
        summary = V.verify(read_bytes=MemoryBytes(self.files))["summary"]
        requested = document(self.files, "source/proposal1/proposal.json")["requests"]
        rows = summary["requests"]
        self.assertEqual([(row["id"], row["source"], row["backend"]) for row in rows],
                         [(row["id"], row["source"], row["backend"]) for row in requested])
        self.assertEqual([row["controller_setup_started"] for row in rows], [True, False, False, False, False, False])
        self.assertTrue(all(row["execution_started"] is False for row in rows))
        self.assertEqual(summary["amplitude_evaluations"], 0)
        self.assertEqual(summary["comparisons"], [])
        self.assertIsNone(summary["primary_policy_agreement"])
        self.assertIs(summary["performance_eligible"], False)
        self.assertEqual(summary["gates"], {"preparation_passed": True, "preflight_passed": True,
                                            "native_run_passed": False, "numerically_qualified": False,
                                            "independent_finite_term_oracle": False})
        for row in rows:
            self.assertIs(row["amplitude_observed"], False)
            self.assertIs(row["numerically_qualified"], False)
            for phase in ("warmup", "final"):
                outcomes = row["phase_outcomes"][phase]
                self.assertEqual([(item["point"], item["order"]) for item in outcomes],
                                 [(point, order) for point in range(1, 22) for order in (0, 1)])
                self.assertTrue(all(item["state"] == "unexecuted" for item in outcomes))
                self.assertEqual(row["phase_counts"][phase],
                                 {"accepted": 0, "rejected": 0, "unchecked": 0, "unexecuted": 42})

    def test_callback_must_supply_exact_inventory(self):
        self.files["unindexed-extra.txt"] = b"extra\n"
        self.rejected()
        self.files = dict(self.original)
        self.files.pop("closure.json")
        self.rejected()
        self.files = dict(self.original)
        self.rejected(lambda name: self.files[name])

    def test_raw_and_compressed_byte_tampering_are_rejected(self):
        for name in ("source/supervisor1/run.py", "closed/stage-run.json.gz"):
            with self.subTest(name=name):
                self.files = dict(self.original)
                self.files[name] += b"tampered"
                self.rejected()

    def test_bad_codec_decoded_hash_and_length_are_rejected(self):
        for field, value in (("codec", "binary"), ("decoded_sha256", "0" * 64), ("decoded_bytes", 0)):
            with self.subTest(field=field):
                self.files = dict(self.original)
                alter_json(self.files, INDEX, lambda index: index["files"]["closure.json"].update({field: value}))
                self.rejected()

    def test_corrupt_gzip_and_invalid_utf8_reject_even_with_matching_raw_digest(self):
        for name, raw, codec in (("closed/stage-run.json.gz", b"not a gzip stream", "gzip-utf8"),
                                 ("closure.json", b"\xff\xfe", "utf8")):
            with self.subTest(name=name):
                self.files = dict(self.original)
                self.files[name] = raw
                def update(index):
                    index["files"][name].update(codec=codec, bytes=len(raw), sha256=digest(raw))
                alter_json(self.files, INDEX, update)
                self.rejected()

    def test_index_binding_and_relative_paths_are_enforced(self):
        for mutate in (lambda index: index.update(binding={}),
                       lambda index: index["files"].update({"../outside.txt": index["files"]["closure.json"]}),
                       lambda index: index["files"].update({"/absolute.txt": index["files"]["closure.json"]})):
            self.files = dict(self.original)
            alter_json(self.files, INDEX, mutate)
            self.rejected()

    def test_duplicate_keys_and_nonfinite_json_cannot_enter_the_capsule(self):
        original = self.original["closure.json"]
        malformed = [b'{"schema_version": 1,' + original.lstrip()[1:],
                     original.replace(b'"amplitude_launch_count": 0', b'"amplitude_launch_count": NaN'),
                     original.replace(b'"amplitude_launch_count": 0', b'"amplitude_launch_count": 1e999')]
        for raw in malformed:
            self.files = dict(self.original)
            self.files["closure.json"] = raw
            self.rejected(seal(self.files))

    def test_resealed_source_and_ancestry_changes_cannot_inherit_frozen_identity(self):
        for name in ("source/supervisor1/run.py", "source/proposal1/proposal.json",
                     "source/ancestry/tools/tirust_benchmark_worker.py"):
            with self.subTest(name=name):
                self.files = dict(self.original)
                if name.endswith(".json"):
                    alter_json(self.files, name, lambda value: value.update(strict_host_accuracy_limit=0.1))
                else:
                    self.files[name] += b"\n# altered retained source\n"
                self.semantic_rejection()

    def test_stored_summary_cannot_claim_numerical_success(self):
        alter_json(self.files, SUMMARY, lambda value: value.update(overall_status="passed", numerical_success=True))
        self.rejected(seal(self.files))

    def test_closure_requires_observed_terminal_and_zero_amplitude_counters(self):
        for field, value in (("root_observed_terminal", False), ("amplitude_launch_count", 1),
                             ("worker_launch_count", 1), ("physics_loader_launch_count", 1),
                             ("completed_requests", 1)):
            with self.subTest(field=field):
                self.files = dict(self.original)
                alter_json(self.files, "closure.json", lambda row: row.update({field: value}))
                self.semantic_rejection()

    def test_run_status_and_session_exit_cannot_be_promoted_to_success(self):
        alter_closed(self.files, "stage-run.json", lambda value: value.update(status="passed", exception=None))
        self.semantic_rejection()
        self.files = dict(self.original)
        alter_json(self.files, "closure.json", lambda value: value["sessions"]["run"].update(exit_code=0))
        self.semantic_rejection()

    def test_accounting_requires_all_six_distinct_uncompleted_requests(self):
        for mutate in (lambda value: value["not_completed"].pop(),
                       lambda value: value["not_completed"].__setitem__(1, value["not_completed"][0]),
                       lambda value: value.update(completed=1),
                       lambda value: value.update(requested=5),
                       lambda value: value["not_completed"][0]["point_indices"].pop()):
            self.files = dict(self.original)
            alter_closed(self.files, "request-accounting.json", mutate)
            self.semantic_rejection()

    def test_abort_receipt_identifies_this_unqualified_first_request(self):
        for mutate in (lambda value: value.update(state="completed"),
                       lambda value: value.update(numerically_qualified=True),
                       lambda value: value["request"].update(id="candidate-9-auto", backend=9, source="candidate")):
            self.files = dict(self.original)
            alter_closed(self.files, "aborted-request.json", mutate)
            self.semantic_rejection()

    def test_stage_accounting_and_abort_failure_receipts_must_agree(self):
        for name, key in (("stage-run.json", "exception"),
                          ("request-accounting.json", "native_stage_failure"),
                          ("aborted-request.json", "exception")):
            with self.subTest(name=name):
                self.files = dict(self.original)
                alter_closed(self.files, name, lambda value: value.update({key: "RuntimeError('different failure')"}))
                self.semantic_rejection()

    def test_failed_prepare_or_preflight_cannot_qualify_setup(self):
        for stage in ("prepare", "preflight"):
            with self.subTest(stage=stage):
                self.files = dict(self.original)
                alter_closed(self.files, "stage-" + stage + ".json", lambda value: value.update(status="failed"))
                self.semantic_rejection()
                self.files = dict(self.original)
                alter_json(self.files, "closure.json", lambda value: value["sessions"][stage].update(exit_code=1))
                self.semantic_rejection()

    def test_passed_setup_stage_cannot_record_an_exception(self):
        for stage in ("prepare", "preflight"):
            with self.subTest(stage=stage):
                self.files = dict(self.original)
                alter_closed(self.files, "stage-" + stage + ".json",
                             lambda value: value.update(exception="RuntimeError('setup failed')"))
                self.semantic_rejection()

    def test_model_proof_requires_cmsparam_card_widths_and_complex_sign(self):
        name = "builds/candidate-cms_ew_n2_epem_mupmum/preflight.json"
        changes = [lambda value: value["complex_model"].update(CMSParam=0.0),
                   lambda value: value["complex_model"]["masses"]["W"].update(width=-2.09291),
                   lambda value: value["complex_model"]["masses"]["W"].update(width=2.0),
                   lambda value: value["complex_model"]["masses"]["Z"]["squared_mass"].__setitem__(1, 227.85783876)]
        for change in changes:
            self.files = dict(self.original)
            alter_closed(self.files, name, change)
            self.semantic_rejection()

    def test_prelink_receipt_requires_every_loaded_input_to_be_bound(self):
        name = "builds/candidate-cms_ew_n2_epem_mupmum/build.json"
        alter_closed(self.files, name, lambda value: value["actual_link_loads"].update({"/unbound/libextra.a": "0" * 64}))
        build_hash = document(self.files, "retention.json")["closed_files"][name]["sha256"]
        alter_closed(self.files, "builds/candidate-cms_ew_n2_epem_mupmum/preflight.json",
                     lambda value: value.update(build_sha256=build_hash))
        self.semantic_rejection()

    def test_filesystem_reader_refuses_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, raw in self.files.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
            target = root / "closure.json"
            target.unlink()
            target.symlink_to(HERE / "closure.json")
            with self.assertRaises(V.EvidenceError):
                V.verify(root)


if __name__ == "__main__":
    unittest.main()
