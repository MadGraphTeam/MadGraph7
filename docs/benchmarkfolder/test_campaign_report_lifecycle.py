"""Lifecycle report acceptance is bound to raw evidence and exact source/native identities."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("campaign_report_lifecycle", HERE / "campaign_report_lifecycle.py")
LIFECYCLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LIFECYCLE)
REPORT_SPEC = importlib.util.spec_from_file_location("campaign_report", HERE / "campaign_report.py")
REPORT = importlib.util.module_from_spec(REPORT_SPEC)
REPORT_SPEC.loader.exec_module(REPORT)


@pytest.fixture
def identities():
    sources = {"madgraph/iolibs/helas_call_writers.py": "exporter", "Template/cache.f": "cache"}
    names = ("include/mtirust.f90", "include/tirust.h", "lib/libmadloop_tirust.so",
             "lib/libtirust.a", "lib/tirust-config.json", "lib/tirust-link-before.txt",
             "lib/tirust-native-libs.txt")
    native = {name: name + "-hash" for name in names}
    native.update({"Cargo.lock": "lock-hash", "Cargo.toml": "manifest-hash"})
    sources.update({"vendor/tirust/" + name: native[name] for name in ("Cargo.lock", "Cargo.toml")})
    artifacts = {"/recorded/installation/" + name: {"sha256": checksum, "size": 10}
                 for name, checksum in native.items()}
    artifacts["/recorded/installation/lib/mtirust.mod"] = {"sha256": "module-hash", "size": 20}
    installed = {"native_artifacts": {name: {"sha256": artifacts["/recorded/installation/lib/" + name]["sha256"],
                                              "bytes": artifacts["/recorded/installation/lib/" + name]["size"]}
                                       for name in ("libmadloop_tirust.so", "libtirust.a", "mtirust.mod")}}
    installed["build_provenance"] = {"lock_sha256": "lock-hash", "manifest_sha256": "manifest-hash"}
    source = {"files": {name: {"kind": "file", "sha256": checksum} for name, checksum in sources.items()},
              "revision": "implementation"}
    freeze = {"implementation_revision": "implementation", "installation": artifacts}
    manifest = {"candidate": {"exporter_identity": {"files": sources},
                               "numerical_sources": {"Template/cache.f": {"sha256": "cache"}},
                               "native": {"files": native}}}
    return source, freeze, installed, manifest


def test_lifecycle_matches_sources_and_native_bytes_without_machine_paths(identities):
    result = LIFECYCLE.validate_candidate(*identities)
    assert result["registered_production_entries"] == 4
    assert result["registered_production_content_hashes"] == 4
    assert result["fortran_module_sha256"] == "module-hash"
    assert result["implementation_revision"] == "implementation"


def test_registered_source_link_metadata_is_never_followed(identities):
    source, freeze, installed, manifest = identities
    source["files"]["Template/cache.f"].update(kind="link", target="../unavailable/cache.f")
    source["files"]["unused/dangling"] = {"kind": "link", "target": "unavailable"}
    manifest["candidate"]["exporter_identity"]["files"]["unused/dangling"] = None
    result = LIFECYCLE.validate_candidate(*identities)
    assert result["registered_production_entries"] == 5
    assert result["registered_production_content_hashes"] == 4


@pytest.mark.parametrize("change", [
    lambda source, freeze, installed, manifest: source["files"]["Template/cache.f"].update(sha256="changed"),
    lambda source, freeze, installed, manifest: source["files"].pop("Template/cache.f"),
    lambda source, freeze, installed, manifest: source["files"]["Template/cache.f"].update(kind="symlink"),
    lambda source, freeze, installed, manifest: source.update(revision="unrelated"),
    lambda source, freeze, installed, manifest: manifest["candidate"]["numerical_sources"]["Template/cache.f"].update(sha256="changed"),
    lambda source, freeze, installed, manifest: freeze["installation"]["/recorded/installation/lib/libmadloop_tirust.so"].update(sha256="old"),
    lambda source, freeze, installed, manifest: freeze["installation"].pop("/recorded/installation/include/mtirust.f90"),
    lambda source, freeze, installed, manifest: installed["native_artifacts"]["mtirust.mod"].update(sha256="other"),
    lambda source, freeze, installed, manifest: installed["native_artifacts"]["libtirust.a"].update(bytes=999),
    lambda source, freeze, installed, manifest: freeze["installation"].update({"/other/lib/libmadloop_tirust.so": {"sha256": "other"}}),
    lambda source, freeze, installed, manifest: manifest["candidate"]["native"]["files"].update({"Cargo.lock": "changed"}),
    lambda source, freeze, installed, manifest: manifest["candidate"]["native"]["files"].update({"Cargo.toml": "changed"}),
    lambda source, freeze, installed, manifest: installed["build_provenance"].update(lock_sha256="changed"),
    lambda source, freeze, installed, manifest: manifest["candidate"]["native"]["files"].update({"uncovered": "changed"}),
])
def test_lifecycle_cannot_qualify_other_source_or_installation(identities, change):
    change(*identities)
    with pytest.raises(ValueError):
        LIFECYCLE.validate_candidate(*identities)


@pytest.fixture
def corpus(tmp_path, identities):
    """Exercise the reader boundary with a pure fixture verifier, without physics.

    The separately frozen real verifier owns raw-record and trace adjudication;
    its full package is also replayed when this report change is qualified.
    """
    source, freeze, installed, manifest = identities
    def write(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True) + "\n")
        return path
    summary = {"schema_version": 1, "kind": "tirust-generated-subprocess-cache-lifecycle",
               "implementation_revision": "implementation", "passed": True}
    write(LIFECYCLE.INSTALLER, installed)
    for name, value in (("materialized-source.json", source), ("source-freeze.json", freeze),
                        ("attempt5/qualification-summary.json", summary)):
        write(LIFECYCLE.PREFIX + name, value)
    protocol = tmp_path / LIFECYCLE.PREFIX / "verify_evidence.py"
    protocol.write_text('''import hashlib
import json
def verify(root=None, *, read_bytes=None, read_link=None):
    index_raw = read_bytes("evidence-index.json")
    index = json.loads(index_raw)
    for name in index["files"]:
        read_bytes(name)
    return {"passed": True, "index_sha256": hashlib.sha256(index_raw).hexdigest(),
            "physical_workers": 4, "measured_physical_calls": 76, "tensor_key_calls": 96,
            "card_smoke_passed": True, "historical_symlinks_are_metadata_only": True,
            "summary": json.loads(read_bytes("attempt5/qualification-summary.json"))}
''')
    def seal():
        paths = sorted((tmp_path / LIFECYCLE.PREFIX).rglob("*"))
        files = {path.relative_to(tmp_path / LIFECYCLE.PREFIX).as_posix():
                 {"kind": "file", "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "size": len(path.read_bytes())}
                 for path in paths if path.is_file() and path.name != "evidence-index.json"}
        write(LIFECYCLE.INDEX, {"schema_version": 1, "files": files})
    seal()
    return tmp_path, manifest, seal


def test_missing_final_index_remains_pending(tmp_path):
    result = LIFECYCLE.collect_lifecycle(REPORT.Evidence(tmp_path), {})
    assert result == LIFECYCLE.pending()
    assert result["evidence"] is None


def test_frozen_pending_snapshot_cannot_promote_from_later_index(corpus):
    root, manifest, _ = corpus
    frozen = REPORT.Evidence(root, {})
    assert LIFECYCLE.collect_lifecycle(frozen, manifest) == LIFECYCLE.pending()


def test_complete_indexed_proof_binds_identity_without_claiming_timing(corpus):
    root, manifest, _ = corpus
    evidence = REPORT.Evidence(root)
    result = LIFECYCLE.collect_lifecycle(evidence, manifest)
    assert result["status"] == "passed"
    assert result["evidence"] == LIFECYCLE.INDEX
    assert result["verification"]["measured_physical_calls"] == 76
    assert "QP coverage is confined to tensor-key diagnostics" in result["scope"]
    assert "timing claim" in result["scope"]
    assert LIFECYCLE.PREFIX + "verify_evidence.py" in evidence.files
    assert LIFECYCLE.INSTALLER in evidence.files
    frozen = REPORT.Evidence(root, deepcopy(evidence.files))
    (root / LIFECYCLE.PREFIX / "future-result.json").write_text('{"passed":true}')
    assert LIFECYCLE.collect_lifecycle(frozen, manifest) == result


def test_changed_or_missing_indexed_evidence_cannot_qualify(corpus):
    root, manifest, _ = corpus
    path = root / LIFECYCLE.PREFIX / "attempt5/qualification-summary.json"
    path.write_text('{"passed":true}')
    with pytest.raises(ValueError, match="hash differs"):
        LIFECYCLE.collect_lifecycle(REPORT.Evidence(root), manifest)
    path.unlink()
    with pytest.raises(FileNotFoundError):
        LIFECYCLE.collect_lifecycle(REPORT.Evidence(root), manifest)


def test_resealed_summary_cannot_change_implementation(corpus):
    root, manifest, seal = corpus
    path = root / LIFECYCLE.PREFIX / "attempt5/qualification-summary.json"
    value = json.loads(path.read_text())
    value["implementation_revision"] = "unrelated"
    path.write_text(json.dumps(value))
    seal()
    with pytest.raises(ValueError, match="different implementation"):
        LIFECYCLE.collect_lifecycle(REPORT.Evidence(root), manifest)


def test_indexed_symlink_is_not_followed(corpus):
    root, manifest, _ = corpus
    path = root / LIFECYCLE.PREFIX / "attempt5/qualification-summary.json"
    target = root / "outside-index.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symlinks"):
        LIFECYCLE.collect_lifecycle(REPORT.Evidence(root), manifest)


@pytest.mark.parametrize("requested_name", ["../outside", "/absolute", "unbound.json", "a\\b", "./source-freeze.json"])
def test_verifier_cannot_read_outside_declared_files(corpus, requested_name):
    root, manifest, seal = corpus
    path = root / LIFECYCLE.PREFIX / "verify_evidence.py"
    path.write_text("def verify(root=None, *, read_bytes=None, read_link=None):\n    return read_bytes(" + repr(requested_name) + ")\n")
    seal()
    with pytest.raises(ValueError):
        LIFECYCLE.collect_lifecycle(REPORT.Evidence(root), manifest)


@pytest.mark.parametrize("source", [
    "open('/tmp/unapproved', 'w')\ndef verify(): pass",
    "import subprocess\ndef verify(): pass",
    "@decorator\ndef verify(): pass",
    "if True:\n    print('unapproved')\ndef verify(): pass",
    "VALUE = open('unapproved')\ndef verify(): pass",
    "def verify(root=open('unapproved')): pass",
    "def verify(root: open('unapproved')): pass",
    "from .json import loads\ndef verify(): pass",
])
def test_verifier_loader_rejects_unexpected_top_level_code(source):
    with pytest.raises((ValueError, TypeError)):
        LIFECYCLE.retained_verifier(source.encode())
