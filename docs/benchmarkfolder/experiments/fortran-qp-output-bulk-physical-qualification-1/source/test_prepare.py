"""Pure preparation contract tests; never run prepare or a discovery tool."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("physical_preparation_sources", HERE/"prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_module_import_is_source_only():
    tree = ast.parse((HERE/"prepare.py").read_text())
    imports = {node.names[0].name for node in tree.body if isinstance(node, ast.Import)}
    assert not imports & {"subprocess", "ctypes", "controller", "os"}
    assert "controller.py" not in (HERE/"prepare.py").read_text()
    assert "shutil.which(\"make\", path=environment[\"PATH\"])" in (HERE/"prepare.py").read_text()


def test_consistent_input_merging_and_conflicts():
    files = {}
    checksum = "1"*64
    prepare.merge(files, {"source.f": checksum})
    prepare.merge(files, {"source.f": checksum})
    assert files == {"source.f": checksum}
    with pytest.raises(ValueError, match="Conflicting"):
        prepare.merge(files, {"source.f": "2"*64})


class RecordedPath:
    """In-memory filesystem-state fixture; no test creates files or links."""
    def __init__(self, exists=False, link=None):
        self.present = exists
        self.link = link

    def exists(self):
        return self.present

    def is_symlink(self):
        return self.link is not None

    def readlink(self):
        assert self.link is not None
        return self.link

    def __str__(self):
        return "recorded/input"


def test_declared_dangling_link_and_plain_absence_preserved():
    target = "../../SubProcesses/pineappl_maxproc.inc"
    prepare.verify_intentional_absence(RecordedPath(link=target), target)
    prepare.verify_intentional_absence(RecordedPath())
    files = {}
    prepare.merge(files, {"missing.inc": None})
    prepare.merge(files, {"missing.inc": None})
    assert files == {"missing.inc": None}
    with pytest.raises(ValueError, match="Conflicting"):
        prepare.merge(files, {"missing.inc": "1"*64})


@pytest.mark.parametrize("state,target", [
    (RecordedPath(link="different-target"), "expected-target"),
    (RecordedPath(exists=True, link="expected-target"), "expected-target"),
    (RecordedPath(exists=True), None),
    (RecordedPath(), "expected-target"),
    (RecordedPath(link="undeclared-target"), None),
    (RecordedPath(exists=True), "expected-target"),
])
def test_changed_absence_link_regular_file_or_appeared_target_rejected(state, target):
    with pytest.raises(ValueError):
        prepare.verify_intentional_absence(state, target)


class RecordedShell:
    def __init__(self, present=True, raw=b"changed shell bytes"):
        self.present = present
        self.raw = raw

    def is_file(self):
        return self.present

    def read_bytes(self):
        assert self.present
        return self.raw

    def __str__(self):
        return prepare.MAKE_SHELL


@pytest.mark.parametrize("shell", [RecordedShell(present=False), RecordedShell()])
def test_missing_or_changed_fixed_make_shell_rejected(monkeypatch, shell):
    monkeypatch.setattr(prepare, "Path", lambda name: shell)
    with pytest.raises(ValueError):
        prepare.pin_fixed_make_shell({}, HERE)


def test_reviewed_fixed_make_shell_is_pinned_by_known_hash():
    assert prepare.MAKE_SHELL == "/nix/store/lw117lsr8d585xs63kx5k233impyrq7q-bash-5.3p3/bin/bash"
    assert prepare.MAKE_SHELL_SHA == "03b13000ff5011a38d8a3f9575b6807d78cf9936c64a0be28f02abf9be8c9c25"
    files = {}
    assert prepare.pin_fixed_make_shell(files, Path.cwd()) == prepare.MAKE_SHELL
    assert files[prepare.MAKE_SHELL] == prepare.MAKE_SHELL_SHA


@pytest.mark.parametrize("checksum", ["", "a"*63, "g"*64, "A"*64, 1])
def test_invalid_input_checksum_rejected(checksum):
    with pytest.raises(ValueError):
        prepare.merge({}, {"source.f": checksum})


@pytest.mark.parametrize("unsafe", ["/tmp/x", "../x", "a/../x", "a//x", "a\\x", ""])
def test_unsafe_index_name_rejected_before_payload_read(unsafe):
    called = []
    index = {"files": {unsafe: {"bytes": 1, "sha256": "0"*64}}}
    with pytest.raises(ValueError):
        prepare.verify_index_payloads(index, lambda name: called.append(name), "index.json")
    assert called == []


def test_self_index_rejected_before_payload_read():
    with pytest.raises(ValueError, match="Self-indexed"):
        prepare.verify_index_payloads({"files": {"index.json": {}}},
                                      lambda name: pytest.fail("Unexpected read"), "index.json")


def test_source_payload_size_and_digest_are_both_checked():
    raw = b"source text\n"
    meta = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    prepare.verify_index_payloads({"files": {"source.f": meta}}, lambda name: raw, "index.json")
    for value in [raw+b"x", b"SOURCE text\n"]:
        with pytest.raises(ValueError):
            prepare.verify_index_payloads({"files": {"source.f": meta}}, lambda name: value, "index.json")


def point_bytes(count=100, fields="1 0 0 -1", rows=500):
    return (str(count)+"\n"+(fields+"\n")*rows).encode()


def test_full100_saved_point_shape():
    prepare.validate_points(point_bytes())


@pytest.mark.parametrize("raw", [point_bytes(99), point_bytes(rows=499),
    point_bytes(fields="1 0 0"), point_bytes(fields="NaN 0 0 1"),
    point_bytes(fields="Infinity 0 0 1"), point_bytes(fields="bad 0 0 1")])
def test_changed_or_nonfinite_point_data_rejected(raw):
    with pytest.raises(ValueError):
        prepare.validate_points(raw)


def test_timing_plan_freezes_common_counts_and_seven_alternating_pairs():
    plan = prepare.timing_plan()
    assert plan["anchors"] == [1,16]
    assert plan["target_seconds"] == 1.0 and plan["max_loops"] == 1000000
    assert plan["calibration_pairs"][0] == {"id":"initial", "order":["original","candidate"], "loops":1}
    assert "max(initial_elapsed_original,initial_elapsed_candidate)" in plan["calibration_pairs"][1]["loops_rule"]
    assert "min(pilot_elapsed_original/pilot_loops,pilot_elapsed_candidate/pilot_loops)" in plan["fixed_loops_rule"]
    assert plan["timed_pairs"] == [{"pair":i+1,"order":(["original","candidate"] if i%2==0 else ["candidate","original"])} for i in range(7)]
    assert plan["performance_acceptance"] is False and plan["partial_or_forced_QP"] is False
    assert json.loads(prepare.encoded(plan)) == plan


def test_verify_only_never_calls_writer(monkeypatch):
    pins = {"files": {}, "original_sources": {}, "original_objects": {}, "expected_runtime": {}}
    monkeypatch.setattr(prepare, "collect_inputs", lambda *a, **k: (pins, {"preview.f":b"source"}))
    monkeypatch.setattr(prepare, "write_new", lambda *a: pytest.fail("Writer called in verify-only mode"))
    result = prepare.prepare(verify_only=True)
    assert result["generated_sources_written"] == 0
    assert result["compiler_executions"] == result["native_executions"] == 0
