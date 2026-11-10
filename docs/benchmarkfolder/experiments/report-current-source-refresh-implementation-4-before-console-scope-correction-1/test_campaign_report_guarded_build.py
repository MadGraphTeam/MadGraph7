"""Synthetic/memory admission tests; no real native, build or Typst execution.

Invented source, receipts and byte strings test report validation boundaries.
The stub verifier is deliberately patched into a synthetic seal in each test;
it is not numerical evidence and never executes a retained build/ELF program.
"""
from contextlib import contextmanager
from copy import deepcopy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("candidate_guarded_build", Path(__file__).with_name("campaign_report_guarded_build.py"))
G = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(G)

STUB_VERIFIER = b'''from pathlib import Path
import json
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
def verify():
    assert __debug__, "Assertions were optimized away"
    assert __name__ != "__main__", "Do not invoke the retained CLI"
    assert HERE.is_relative_to(Path("/tmp")), "Reconstruct only in /tmp"
    assert sys.dont_write_bytecode is True
    index = json.loads((HERE / "evidence-index.json").read_bytes())
    paths = {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()}
    assert paths == set(index["files"]) | {"evidence-index.json"}
    assert not any("__pycache__" in p for p in paths)
    return {"verification":"pass", "payload_files":len(index["files"]),
        "protected_bindings":1193, "retained_code_texts":133, "source_identities":499,
        "newly_compiled":4, "reused":140,
        "private_library_sha256":"69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b",
        "native_artifact_read_or_loaded":False, "historical_native_paths_required":False,
        "physical_or_performance_acceptance":False}
if __name__ == "__main__":
    raise AssertionError("Synthetic CLI must never run")
'''


def encoded(value):
    return json.dumps(value, sort_keys=True).encode("utf-8")


class MemoryEvidence:
    def __init__(self, entries):
        self.entries, self.reads = entries, []

    def add(self, name, expected=None):
        self.reads.append(name)
        raw = self.entries[name]
        if expected is not None and G.sha(raw) != expected:
            raise ValueError("Synthetic evidence identity differs")
        return raw


class SyntheticPacket:
    def __init__(self, verifier=STUB_VERIFIER):
        library = {"path": "/unavailable/historical/private/libmadloop_tirust_adaptive_private.so",
                   "sha256": G.LIBRARY_SHA256, "bytes": 69162336}
        self.receipt = {"schema": "cayley-guarded-adaptive-build-v1", "complete": True,
            "loader_or_native_test_executed": False, "private_library_not_loaded": library}
        self.receipt_raw = encoded(self.receipt)
        self.log = b"synthetic Cargo log; no command executed"
        qualification = {"schema": "guarded-adaptive-build-qualification-v1", "scope": "Private release compilation only",
            "library_qualified_only_as": "A newly compiled ELF shared object; not loaded or numerically exercised",
            "source_index_sha256": G.SOURCE_SHA256, "execution_sha256": G.sha(self.receipt_raw), "build_log_sha256": G.sha(self.log),
            "cargo_artifacts": 144, "newly_compiled": 4, "reused": 140, "warning_diagnostics": 41,
            "protected_binding_count": 1193, "copied_source_identity_count": 499, "retained_code_text_count": 133,
            "sole_changed_source": G.GUARD, "library": library, **{k: False for k in G.SCIENTIFIC_FLAGS}}
        main_path = "/unavailable/.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so"
        old = {"complete": True, "private_library_not_loaded": {"sha256": G.OLD_LIBRARY_SHA256},
               "protected_inputs_before": {main_path: G.MAIN_LIBRARY_SHA256},
               "protected_inputs_after": {main_path: G.MAIN_LIBRARY_SHA256}}
        self.payloads = {"verify.py": verifier, "qualification.json": encoded(qualification),
            "attempt1/execution.json.gz": gzip.compress(self.receipt_raw), "attempt1/release-build.log.gz": gzip.compress(self.log),
            "proposal3/old-release-execution.json": encoded(old),
            "retention-failure-1/failure.json": encoded({"schema": "portable-verifier-preparation-failure-v1",
                "DSO_read_again": False, "native_or_build_rerun": False}),
            "elf_metadata.py": b"raise AssertionError('Synthetic ELF reader must never execute')\n",
            "proposal3/build.py": b"raise AssertionError('Synthetic build must never execute')\n",
            "retention-script.py": b"raise AssertionError('Synthetic retention must never execute')\n"}
        for number, count, code in ((1, 1163, 101), (2, 1186, -9)):
            protected = {"synthetic-source-" + str(i): "a" * 64 for i in range(count)}
            failed = {"complete": False, "protected_input_closure_complete": False, "loader_or_native_test_executed": False,
                "protected_inputs_before": protected, "commands": [{"progress": {"returncode": code}}]}
            failed_raw = encoded(failed)
            closure = {"read_hash_only": True, "failed_execution_unchanged": True, "protected_input_drift": False,
                "new_shared_library_emitted": False, "native_reducer_or_amplitude_executed": False,
                "protected_inputs_after": protected, "failed_execution_sha256": G.sha(failed_raw)}
            prefix = f"proposal3/failed-attempt{number}/"
            self.payloads[prefix + "execution.json.gz"] = gzip.compress(failed_raw)
            self.payloads[prefix + "closure.json"] = encoded(closure)

    def document(self, name):
        raw = self.payloads[name]
        return G.decode(gzip.decompress(raw) if name.endswith(".gz") else raw)

    def update_document(self, name, value):
        raw = encoded(value)
        self.payloads[name] = gzip.compress(raw) if name.endswith(".gz") else raw

    @contextmanager
    def opened(self, index_edit=None, verifier_pin=None):
        index = {"schema": G.INDEX_SCHEMA, "scope": G.INDEX_SCOPE, "files": {
            name: {"bytes": len(raw), "sha256": G.sha(raw)} for name, raw in self.payloads.items()}}
        if index_edit is not None:
            index_edit(index)
        index_raw = encoded(index)
        prefix = str(Path(G.INDEX).parent)
        entries = {prefix + "/" + name: raw for name, raw in self.payloads.items()}
        entries[G.INDEX] = index_raw
        evidence = MemoryEvidence(entries)
        with patch.multiple(G, INDEX_SHA256=G.sha(index_raw),
                VERIFIER_SHA256=verifier_pin or G.sha(self.payloads["verify.py"]),
                RECEIPT_SHA256=G.sha(self.receipt_raw), LOG_SHA256=G.sha(self.log)):
            yield evidence, G.default_selection()


class SyntheticGuardedBuildTests(unittest.TestCase):
    def test_synthetic_compile_success_is_not_scientific_acceptance(self):
        packet = SyntheticPacket()
        with packet.opened() as (evidence, selected):
            result = G.collect(evidence, selected)
        primitive = result["primitives"][G.ROLE]
        self.assertTrue(primitive["build_complete"])
        self.assertEqual(primitive["newly_compiled"], 4)
        self.assertEqual(primitive["reused"], 140)
        self.assertTrue(primitive["libraries_distinct"])
        self.assertEqual(len(primitive["prior_failed_builds"]), 2)
        self.assertEqual([row["original_complete"] for row in primitive["prior_failed_builds"]], [False, False])
        for key in G.SCIENTIFIC_FLAGS:
            self.assertIs(primitive[key], False)
        self.assertFalse(result["main_campaign_modified"])
        self.assertFalse(result["performance_acceptance_eligible"])
        self.assertNotIn("compile_seconds_administrative_only", primitive)
        self.assertIn("trap behavior", result["notes"][0]["text"])

    def test_index_and_verifier_substitution_rejected(self):
        evidence = MemoryEvidence({})
        pin = G.default_selection()[G.ROLE]
        pin["sha256"] = "0" * 64
        with self.assertRaises(ValueError): G.Package(evidence, pin)
        self.assertEqual(evidence.reads, [])
        packet = SyntheticPacket()
        with packet.opened(verifier_pin="0" * 64) as (evidence, selected):
            with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_index_and_source_drift_are_detected_without_historical_reads(self):
        for drift in ("index", "source"):
            with self.subTest(drift=drift):
                packet = SyntheticPacket()
                with packet.opened() as (evidence, selected):
                    name = G.INDEX if drift == "index" else str(Path(G.INDEX).parent / "proposal3/build.py")
                    evidence.entries[name] += b"\n# synthetic drift\n"
                    with self.assertRaises(ValueError): G.collect(evidence, selected)
                    self.assertTrue(all(name in evidence.entries and not name.startswith("/") for name in evidence.reads))
        packet = SyntheticPacket()
        with packet.opened() as (evidence, selected):
            G.collect(evidence, selected)
            self.assertEqual(set(evidence.reads), set(evidence.entries))
            self.assertFalse(any(Path(name).suffix == ".so" or name.startswith("/") for name in evidence.reads))

    def test_strict_index_paths_and_file_identities(self):
        mutations = [lambda i: i.update(unrecognized=True), lambda i: i.update(schema="future"),
            lambda i: i["files"]["verify.py"].update(bytes=True),
            lambda i: i["files"]["verify.py"].update(bytes=G.MAX_TEXT_BYTES + 1),
            lambda i: i["files"]["verify.py"].update(sha256="f" * 64),
            lambda i: i["files"].update({"../outside": {"bytes": 0, "sha256": G.sha(b"")}})]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                packet = SyntheticPacket()
                with packet.opened(index_edit=mutation) as (evidence, selected):
                    with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_native_binary_and_bytecode_payloads_are_forbidden(self):
        for name, raw in (("native.so", b"synthetic"), ("native.txt", b"\x7fELF synthetic"),
                          ("compressed.txt.gz", gzip.compress(b"\x7fELF synthetic")),
                          ("__pycache__/x.pyc", b"synthetic"), ("x.pyc", b"synthetic")):
            with self.subTest(name=name):
                packet = SyntheticPacket();packet.payloads[name] = raw
                with packet.opened() as (evidence, selected):
                    with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_scientific_or_promotion_flag_cannot_follow_compile_success(self):
        for key in G.SCIENTIFIC_FLAGS:
            with self.subTest(key=key):
                packet = SyntheticPacket();value = packet.document("qualification.json");value[key] = True
                packet.update_document("qualification.json", value)
                with packet.opened() as (evidence, selected):
                    with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_library_identities_and_original_failures_remain_distinct(self):
        for defect in ("new_is_old", "old_is_new", "main_is_new", "failed_complete", "closure_drift", "failure_hash", "retention_rerun"):
            with self.subTest(defect=defect):
                packet = SyntheticPacket()
                if defect == "new_is_old":
                    name = "qualification.json";value = packet.document(name);value["library"]["sha256"] = G.OLD_LIBRARY_SHA256
                elif defect in ("old_is_new", "main_is_new"):
                    name = "proposal3/old-release-execution.json";value = packet.document(name)
                    if defect == "old_is_new": value["private_library_not_loaded"]["sha256"] = G.LIBRARY_SHA256
                    else:
                        key = next(iter(value["protected_inputs_before"]))
                        value["protected_inputs_before"][key] = value["protected_inputs_after"][key] = G.LIBRARY_SHA256
                elif defect == "failed_complete":
                    name = "proposal3/failed-attempt1/execution.json.gz";value = packet.document(name);value["complete"] = True
                elif defect in ("closure_drift", "failure_hash"):
                    name = "proposal3/failed-attempt2/closure.json";value = packet.document(name)
                    value["protected_input_drift" if defect == "closure_drift" else "failed_execution_sha256"] = True if defect == "closure_drift" else "0" * 64
                else:
                    name = "retention-failure-1/failure.json";value = packet.document(name);value["native_or_build_rerun"] = True
                packet.update_document(name, value)
                with packet.opened() as (evidence, selected):
                    with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_bytecode_flag_restored_after_success_and_assertion_failure(self):
        original = sys.dont_write_bytecode
        try:
            for initial in (True, False):
                for source, succeeds in ((STUB_VERIFIER, True),
                        (STUB_VERIFIER.replace(b'assert __debug__, "Assertions were optimized away"', b'assert False, "synthetic failure"'), False)):
                    with self.subTest(initial=initial, succeeds=succeeds):
                        sys.dont_write_bytecode = initial
                        packet = SyntheticPacket(source)
                        with packet.opened() as (evidence, selected):
                            if succeeds: G.collect(evidence, selected)
                            else:
                                with self.assertRaises(ValueError): G.collect(evidence, selected)
                        self.assertIs(sys.dont_write_bytecode, initial)
        finally:
            sys.dont_write_bytecode = original

    def test_compile_explicitly_preserves_assertions_and_does_not_invoke_cli(self):
        packet = SyntheticPacket()
        calls = []
        native_compile = compile
        def recording_compile(source, filename, mode, **kwargs):
            calls.append(kwargs)
            return native_compile(source, filename, mode, **kwargs)
        with packet.opened() as (evidence, selected), patch.object(G, "compile", recording_compile, create=True):
            G.collect(evidence, selected)
        self.assertEqual(calls, [{"dont_inherit": True, "optimize": 0}])

    def test_verifier_cannot_pollute_exact_reconstructed_inventory(self):
        polluted = STUB_VERIFIER.replace(b'    return {"verification":"pass"',
            b'    (HERE / "unexpected.pyc").write_bytes(b"synthetic pollution")\n    return {"verification":"pass"')
        packet = SyntheticPacket(polluted)
        with packet.opened() as (evidence, selected):
            with self.assertRaises(ValueError): G.collect(evidence, selected)

    def test_replay_rejects_untrusted_source_even_for_package_compatible_objects(self):
        class Untrusted:
            def raw(self, name): return b"raise AssertionError('must not execute')"
        with self.assertRaises(ValueError): G.replay(Untrusted())

    def test_unknown_empty_role_and_strict_json(self):
        for selected in ({}, {"adaptive_future": {}}, {G.ROLE: {}, "extra": {}}):
            with self.subTest(selected=selected):
                with self.assertRaises(ValueError): G.collect(MemoryEvidence({}), selected)
        for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":1e999}'):
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError): G.decode(raw)


if __name__ == "__main__":
    unittest.main()
