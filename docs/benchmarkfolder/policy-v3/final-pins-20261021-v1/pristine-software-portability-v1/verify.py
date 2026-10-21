#!/usr/bin/env python3
"""Verify the retained software observations; never qualify numerical results."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile


if not __debug__:
    raise RuntimeError("Verification requires Python assertions; optimized Python (-O/-OO) is unsupported")


HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def verify(live=None):
    manifest = read("manifest.json")
    actual = {p.name for p in HERE.iterdir() if p.is_file()} - {"manifest.json"}
    assert actual == set(manifest["payloads"]), "Payload inventory mismatch"
    for name, row in manifest["payloads"].items():
        data = (HERE / name).read_bytes()
        assert len(data) == row["bytes"] and digest(data) == row["sha256"], name

    result = read("result.json")
    state = read("source-state.json")
    preparation = read("archive-preparation.json")
    assert result["schema"] == "ninjars-pristine-software-portability-v1"
    assert result["archive_base_revision"] == preparation["source_revision"] == state["base_revision"]
    assert preparation["exit_code"] == 0 and not preparation["working_tree_edits_copied"]
    assert set(preparation["ignored_parents_absent"]) == {".benchmark-work", ".ninjars-build"}
    assert result["native_library_or_install_calls"] == 0
    assert result["cpu_affinity"] == [180]
    expected_sources = {**state["tests"], **state["base_tracked_tool_helpers"]}
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress((HERE / "software-sources.tar.gz").read_bytes())), mode="r:") as archive:
        members = archive.getmembers()
        assert len(members) == len(expected_sources) == result["full_source_files"] == 42
        assert {member.name for member in members} == set(expected_sources)
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
            assert digest(archive.extractfile(member).read()) == expected_sources[member.name], member.name

    runs = json.loads(gzip.decompress((HERE / "runs.json.gz").read_bytes()))
    logs = json.loads(gzip.decompress((HERE / "raw-logs.json.gz").read_bytes()))
    assert set(runs) == set(logs) == {"before", "before-new-loop", "after", "after-final", "after-current"}
    for phase, run in runs.items():
        assert len(logs[phase]) == len(run["records"])
        for row in run["records"]:
            name = row["module"].rsplit(".", 1)[1] + ".log"
            text = logs[phase][name]
            assert len(text.encode()) == row["log_bytes"] and digest(text.encode()) == row["log_sha256"]
            footers = re.findall(r"(?m)^Ran (\d+) tests? in ([0-9.]+)s$", text)
            assert footers and int(footers[-1][0]) == row["tests_run"], (phase, name)
            statuses = re.findall(r"(?m)^(OK(?: \(skipped=\d+\))?|FAILED \([^\n]+\))$", text)
            assert statuses, (phase, name)
            counts = {key: int(value) for key, value in re.findall(r"(failures|errors|skipped)=(\d+)", statuses[-1])}
            assert counts.get("failures", 0) == len(row["failures"])
            assert counts.get("errors", 0) == len(row["errors"])
            assert counts.get("skipped", 0) == len(row["skipped"])
            assert statuses[-1].startswith("OK") == row["passed"]
            assert row["ignored_parents_before"] == {".benchmark-work": False, ".ninjars-build": False}
            assert row["source_unchanged"]
            relative = row["module"].replace(".", "/") + ".py"
            assert row["loaded_from"] == str(Path(row["private_source"]) / relative)
            if phase == "after-current":
                assert row["source_sha256"] == state["tests"][relative]
                assert row["exit_code"] == 0 and row["elapsed_seconds"] > 0
                assert row["command"][1] == "-B" and row["command"][4] == row["module"]
                assert row["cwd"] == result["reproduction"]["cwd"]
        if "tests_run" in run:
            assert run["tests_run"] == sum(row["tests_run"] for row in run["records"])
            assert run["errors"] == sum(len(row["errors"]) for row in run["records"])
            assert run["failures"] == sum(len(row["failures"]) for row in run["records"])
            assert run["skips"] == sum(len(row["skipped"]) for row in run["records"])

    before, final = runs["before"], runs["after-current"]
    assert (before["modules"], before["tests_run"], before["errors"], before["failures"]) == (14, 276, 212, 2)
    assert len(runs["before-new-loop"]["records"][0]["errors"]) == 2
    assert (runs["after"]["tests_run"], runs["after"]["errors"], runs["after"]["failures"]) == (278, 5, 2)
    assert (runs["after-final"]["tests_run"], runs["after-final"]["errors"], runs["after-final"]["failures"]) == (278, 0, 0)
    assert (final["modules"], final["tests_run"], final["errors"], final["failures"], final["skips"]) == (16, 308, 0, 0, 1)
    assert final["cpu_affinity"] == [180] and final["passed"]
    assert (result["modules"], result["tests_run"], result["tests_passed"], result["errors"], result["failures"], result["skips"]) == (16, 308, 307, 0, 0, 1)
    skipped = [item for row in final["records"] for item in row["skipped"]]
    assert skipped == [{"reason": "Actual final generated source is not retained in this checkout", "test": "test_actual_generated_qcd2_count_definitions_agree (tests.unit_tests.loop.test_ninjars_paper_report_v3.NinjaRSPaperReportV3Test.test_actual_generated_qcd2_count_definitions_agree)"}]
    assert [row["module"] for row in final["records"]] == read("modules-current.json")
    assert len(read("fixture-changes.json")) == result["fixture_changed_test_files"] == 13
    for row in read("fixture-changes.json"):
        assert state["tests"][row["path"]] == row["after_sha256"]
    overlay = read("dp-checkpoint-overlay.json")["files"]
    assert all(expected_sources[name] == value for name, value in overlay.items())
    production = result["production_regression"]
    assert production["manifest_sha256"] == "e4752a6a3d0476008470b0997c193cee7bc0842edfcf87452f5cf82621508da1"
    if live is not None:
        live = Path(live).resolve()
        assert digest((live / production["manifest_path"]).read_bytes()) == production["manifest_sha256"]
        for name, value in {**expected_sources, **production["unchanged_bound_sources"]}.items():
            assert digest((live / name).read_bytes()) == value, name
    return {"status": "verified", "scope": "additive pristine-checkout software portability", "modules": 16, "tests_run": 308, "tests_passed": 307, "skipped": 1, "live_source_check": live is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, metavar="ROOT")
    args = parser.parse_args()
    print(json.dumps(verify(args.live), sort_keys=True))
