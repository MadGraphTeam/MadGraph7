"""Explicit closed guarded-adaptive compilation role; portable text replay only.

The authenticated verifier is reconstructed in /tmp and called without its CLI.
It reads retained UTF-8/gzip records, never a historical native path. Build
success does not qualify a physical replay, precision, traps or promotion.
"""
import gzip
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tempfile


ROLE = "guarded_adaptive_build"
ROLES = frozenset({ROLE})
INDEX = "experiments/cayley-guarded-adaptive-build-qualification-1/evidence-index.json"
INDEX_SHA256 = "94f7a23d4c10373353b04e876da32585bc0274852fc573e8d94b740c9fa0ae3e"
VERIFIER_SHA256 = "7135aeba1c9b1737e8235cf567012fff8a6bc92d271cedd75b007fd74b0e9bd5"
INDEX_SCHEMA = "portable-guarded-adaptive-build-evidence-v1"
INDEX_SCOPE = "Closed private release build, source provenance and pure ELF metadata; no numerical, precision, stability or performance acceptance"
LIBRARY_SHA256 = "69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b"
OLD_LIBRARY_SHA256 = "c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7"
MAIN_LIBRARY_SHA256 = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
SOURCE_SHA256 = "39d6ba2bd3c7aef5c8f5008360661c988a93d86ab1265cb3bc815707233b9e72"
RECEIPT_SHA256 = "59167eec84a767f0712b6905285a99564458ff0667427de87ce1921534b46632"
LOG_SHA256 = "7ea452558263c98b91781e979f843c8cf236d951b89450cbbc436275945e8a3d"
GUARD = "sources/pole-development/crates/iregi-rust/src/direct/exec/gram.rs"
SCIENTIFIC_FLAGS = ("native_library_loaded", "native_tests_executed", "installed_or_promoted", "precision_qualified",
                    "physical_accuracy_qualified", "stability_qualified", "runtime_performance_qualified")
MAX_TEXT_BYTES = 1048576


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def safe(name):
    require(isinstance(name, str) and name and "\\" not in name, "Invalid guarded-build evidence path")
    path = PurePosixPath(name)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == name
            and "__pycache__" not in path.parts, "Unsafe guarded-build evidence path")
    return name


def decode(raw):
    def invalid(value):
        raise ValueError("Nonfinite guarded-build JSON: " + value)
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "Duplicate guarded-build JSON key: " + key)
            result[key] = value
        return result
    def finite(value):
        result = float(value)
        require(math.isfinite(result), "Nonfinite guarded-build JSON number")
        return result
    return json.loads(raw, parse_constant=invalid, parse_float=finite, object_pairs_hook=pairs)


def fields(value, expected, message):
    require(isinstance(value, dict), message)
    for key, wanted in expected.items():
        actual = value.get(key)
        require(type(actual) is type(wanted) and actual == wanted, message + ": " + key)


def default_selection():
    return {ROLE: {"index": INDEX, "sha256": INDEX_SHA256}}


class Package:
    """Bind all indexed bytes before any authenticated Python is executed."""
    def __init__(self, evidence, pin):
        require(isinstance(pin, dict) and pin == {"index": INDEX, "sha256": INDEX_SHA256}
                and set(pin) == {"index", "sha256"}, "Guarded build requires its exact closed package pin")
        self.evidence, self.name, self.checksum = evidence, safe(pin["index"]), pin["sha256"]
        self.prefix = str(PurePosixPath(self.name).parent)
        self.index_raw = evidence.add(self.name, self.checksum)
        require(sha(self.index_raw) == self.checksum, "Guarded-build index identity differs")
        self.index = decode(self.index_raw)
        require(isinstance(self.index, dict) and set(self.index) == {"schema", "scope", "files"},
                "Unexpected guarded-build index fields")
        fields(self.index, {"schema": INDEX_SCHEMA, "scope": INDEX_SCOPE}, "Unknown guarded-build index")
        self.files = self.index["files"]
        require(isinstance(self.files, dict) and self.files and "verify.py" in self.files,
                "Empty guarded-build payload index")
        self._raw = {}
        for name, identity in self.files.items():
            safe(name)
            require(isinstance(identity, dict) and set(identity) == {"bytes", "sha256"}
                    and type(identity["bytes"]) is int and 0 <= identity["bytes"] <= MAX_TEXT_BYTES
                    and digest(identity["sha256"]), "Invalid guarded-build payload identity")
            require(PurePosixPath(name).suffix not in {".so", ".rlib", ".o", ".a", ".mod", ".pyc", ".dll", ".dylib"},
                    "Guarded-build collector accepts retained text only")
            raw = evidence.add(self.prefix + "/" + name, identity["sha256"])
            require(len(raw) == identity["bytes"] and sha(raw) == identity["sha256"],
                    "Guarded-build payload bytes differ")
            unpacked = gzip.decompress(raw) if name.endswith(".gz") else raw
            require(not raw.startswith(b"\x7fELF") and not unpacked.startswith(b"\x7fELF"), "Native guarded-build payload is forbidden")
            unpacked.decode("utf-8")
            self._raw[name] = raw
        require(sha(self.raw("verify.py")) == VERIFIER_SHA256, "Guarded-build verifier identity differs")

    def raw(self, name):
        require(safe(name) in self._raw, "Unindexed guarded-build claim input: " + name)
        return self._raw[name]

    def data(self, name):
        raw = self.raw(name)
        return gzip.decompress(raw) if name.endswith(".gz") else raw

    def json(self, name):
        return decode(self.data(name))


def replay(package):
    """Execute only the separately pinned, reviewed, text-only verifier.

    Explicit optimize=0 retains its assertions even under a Python -O caller.
    No module import writes into the exact reconstructed inventory. The global
    bytecode setting is restored on successful and failing verifier calls.
    """
    source = package.raw("verify.py")
    require(sha(source) == VERIFIER_SHA256, "Untrusted guarded-build verifier")
    try:
        with tempfile.TemporaryDirectory(prefix="report-guarded-build-", dir="/tmp") as temporary:
            root = Path(temporary)
            entries = {name: package.raw(name) for name in package.files}
            require("evidence-index.json" not in entries, "Guarded-build index may not index itself")
            entries["evidence-index.json"] = package.index_raw
            for name, raw in entries.items():
                path = root / safe(name)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            filename = str(root / "verify.py")
            namespace = {"__file__": filename, "__name__": "_closed_guarded_build_verifier_",
                         "__package__": None, "__cached__": None}
            previous = sys.dont_write_bytecode
            try:
                sys.dont_write_bytecode = True
                exec(compile(source, filename, "exec", dont_inherit=True, optimize=0), namespace)
                require(callable(namespace.get("verify")), "Guarded-build verifier has no entry point")
                result = namespace["verify"]()
            finally:
                sys.dont_write_bytecode = previous
            paths = list(root.rglob("*"))
            require(not any(path.is_symlink() for path in paths)
                    and {path.relative_to(root).as_posix() for path in paths if path.is_file()} == set(entries),
                    "Guarded-build replay changed its exact text inventory")
            require(all((root / name).read_bytes() == raw for name, raw in entries.items()),
                    "Guarded-build replay modified retained evidence")
            return result
    except (AssertionError, KeyError, TypeError, IndexError, UnicodeError, OSError, SyntaxError) as error:
        raise ValueError("Closed guarded-build portable verification failed") from error


def derive(package, verification):
    """Derive report claims only after the frozen verifier has passed."""
    fields(verification, {"verification": "pass", "payload_files": len(package.files), "protected_bindings": 1193,
        "retained_code_texts": 133, "source_identities": 499, "newly_compiled": 4, "reused": 140,
        "private_library_sha256": LIBRARY_SHA256, "native_artifact_read_or_loaded": False,
        "historical_native_paths_required": False, "physical_or_performance_acceptance": False},
        "Guarded-build verifier outcome differs")
    qualification = package.json("qualification.json")
    fields(qualification, {"schema": "guarded-adaptive-build-qualification-v1", "scope": "Private release compilation only",
        "library_qualified_only_as": "A newly compiled ELF shared object; not loaded or numerically exercised",
        "source_index_sha256": SOURCE_SHA256, "execution_sha256": RECEIPT_SHA256, "build_log_sha256": LOG_SHA256,
        "cargo_artifacts": 144, "newly_compiled": 4, "reused": 140, "warning_diagnostics": 41,
        "protected_binding_count": 1193, "copied_source_identity_count": 499, "retained_code_text_count": 133,
        "sole_changed_source": GUARD, **{key: False for key in SCIENTIFIC_FLAGS}}, "Guarded-build qualification scope differs")
    library = qualification["library"]
    require(set(library) == {"bytes", "path", "sha256"}, "Invalid guarded-build library identity")
    fields(library, {"sha256": LIBRARY_SHA256, "bytes": 69162336}, "Guarded-build library identity differs")
    require(len({LIBRARY_SHA256, OLD_LIBRARY_SHA256, MAIN_LIBRARY_SHA256}) == 3,
            "Guarded, old private and campaign libraries must remain distinct")
    receipt = package.json("attempt1/execution.json.gz")
    fields(receipt, {"schema": "cayley-guarded-adaptive-build-v1", "complete": True,
                    "loader_or_native_test_executed": False}, "Guarded-build execution scope differs")
    require(receipt["private_library_not_loaded"] == library
            and sha(package.data("attempt1/execution.json.gz")) == RECEIPT_SHA256
            and sha(package.data("attempt1/release-build.log.gz")) == LOG_SHA256,
            "Guarded-build summary/receipt binding differs")
    old = package.json("proposal3/old-release-execution.json")
    fields(old, {"complete": True}, "Old adaptive release receipt changed")
    fields(old["private_library_not_loaded"], {"sha256": OLD_LIBRARY_SHA256}, "Old adaptive library identity changed")
    require(old["protected_inputs_before"] == old["protected_inputs_after"], "Old adaptive protected closure changed")
    main = [value for name, value in old["protected_inputs_before"].items()
            if name.endswith("/.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so")]
    require(main == [MAIN_LIBRARY_SHA256], "Campaign library historical binding differs")
    failed_attempts = []
    for number, count, code, label in ((1, 1163, 101, "LLVM allocation failure under 4 GiB address-space limit"),
                                     (2, 1186, -9, "600-second build deadline expired")):
        prefix = f"proposal3/failed-attempt{number}/"
        failed = package.json(prefix + "execution.json.gz")
        closure = package.json(prefix + "closure.json")
        fields(failed, {"complete": False, "protected_input_closure_complete": False,
                       "loader_or_native_test_executed": False}, "Prior failed build was promoted")
        fields(closure, {"read_hash_only": True, "failed_execution_unchanged": True,
            "protected_input_drift": False, "new_shared_library_emitted": False,
            "native_reducer_or_amplitude_executed": False}, "Prior failure closure differs")
        require(len(closure["protected_inputs_after"]) == count
                and sha(package.data(prefix + "execution.json.gz")) == closure["failed_execution_sha256"]
                and failed["protected_inputs_before"] == closure["protected_inputs_after"]
                and failed["commands"][-1]["progress"]["returncode"] == code,
                "Prior failed build binding differs")
        failed_attempts.append({"attempt": number, "outcome": label, "original_complete": False,
                                "closed_unchanged_inputs": count, "new_shared_library_emitted": False})
    retention = package.json("retention-failure-1/failure.json")
    fields(retention, {"schema": "portable-verifier-preparation-failure-v1", "DSO_read_again": False,
                      "native_or_build_rerun": False}, "Retained metadata correction changed scope")
    return {"build_complete": True, "qualification_scope": "private release compilation only",
        "library_sha256": LIBRARY_SHA256, "library_bytes": library["bytes"], "old_private_library_sha256": OLD_LIBRARY_SHA256,
        "campaign_library_sha256": MAIN_LIBRARY_SHA256, "libraries_distinct": True,
        "source_index_sha256": SOURCE_SHA256, "sole_changed_source": GUARD,
        "protected_bindings": 1193, "source_identities": 499, "retained_code_texts": 133,
        "cargo_artifacts": 144, "newly_compiled": 4, "reused": 140, "warning_diagnostics": 41,
        "prior_failed_builds": failed_attempts, "retained_verifier_metadata_failure": True,
        "portable_text_replay_passed": True, "native_artifact_read_or_loaded_by_collector": False,
        **{key: False for key in SCIENTIFIC_FLAGS}}


def note(primitive):
    return {"title": "Guarded adaptive bridge: closed private compilation",
        "text": ("The guarded private release build completes with four newly compiled crates and 140 reused dependency artifacts. "
            "All 1,193 protected bindings remain unchanged; 499 source identities and 133 retained code texts bind the sole reviewed signed-zero denominator guard. "
            "The new library 69722452c234f494 is distinct from the retained old private c5de24e6c41a90ca library and the campaign's eb3d67fdc99b742b bridge. "
            "Both prior failed builds remain retained: LLVM allocation failure under a 4 GiB address-space limit, then a separate 600-second deadline. "
            "The Cargo-banner verifier correction is retained without another build or DSO read. "
            "This package qualifies compilation and source provenance only. The new library was not loaded, installed or numerically exercised in this build package; "
            "physical accuracy, precision, stability, trap behavior, production promotion and runtime performance remain unqualified here. "
            "Recorded build durations are administrative and supply no performance claim."),
        "proof_role": ROLE, "primitives": primitive}


def collect(evidence, selected):
    require(isinstance(selected, dict) and set(selected) == ROLES, "Unknown or empty guarded-build role selection")
    package = Package(evidence, selected[ROLE])
    primitive = derive(package, replay(package))
    row = note(primitive)
    row["evidence"] = package.name
    return {"packages": {ROLE: {"index": package.name, "sha256": package.checksum}},
            "primitives": {ROLE: primitive}, "notes": [row], "main_campaign_modified": False,
            "performance_acceptance_eligible": False}
