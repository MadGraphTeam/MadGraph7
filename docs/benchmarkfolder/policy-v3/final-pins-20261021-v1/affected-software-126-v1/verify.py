#!/usr/bin/env python3
"""Verify the retained affected-software checkpoint, never numerical results."""
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
    seal = read("manifest.json")
    actual = {p.name for p in HERE.iterdir() if p.is_file()} - {"manifest.json"}
    assert actual == set(seal["payloads"]), "Payload inventory mismatch"
    for name, row in seal["payloads"].items():
        raw = (HERE / name).read_bytes()
        assert len(raw) == row["bytes"] and digest(raw) == row["sha256"], name
    before, after = read("source-state-before.json"), read("source-state-after.json")
    bindings, result = read("bindings.json"), read("run-result.json")
    files = bindings["files"]
    sources = before["software_source_sha256_before"]
    combined = {**before["explicit_overlays"], **sources}
    assert combined == files and len(files) == 84 and len(sources) == 40
    assert set(files) == set(after["files"])
    assert bindings["software_source_files"] == 40 and bindings["new_point_authority_files"] == 44
    assert before["ignored_parents_after_overlay"] == {".benchmark-work": False, ".ninjars-build": False}
    assert before["native_calls"] == after["native_calls"] == result["native_calls"] == 0
    assert before["fixture_edits"] == 0 and after["all_unchanged"]
    assert after["all_six_process_groups_reaped"]
    for name, checksum in files.items():
        row = after["files"][name]
        assert row["expected"] == row["private_after"] == row["handed_over_root_after"] == checksum, name

    with tarfile.open(fileobj=io.BytesIO(gzip.decompress((HERE / "software-sources.tar.gz").read_bytes())), mode="r:") as archive:
        members = archive.getmembers()
        assert len(members) == 40 and {item.name for item in members} == set(sources)
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts
            assert digest(archive.extractfile(member).read()) == sources[member.name], member.name

    modules = read("modules.json")
    records = result["records"]
    assert len(modules) == len(records) == result["modules"] == 6
    assert [row["module"] for row in records] == modules
    assert [row["tests_run"] for row in records] == before["expected_test_counts"] == [44, 31, 20, 19, 10, 2]
    assert (result["tests_run"], result["failures"], result["errors"], result["skips"]) == (126, 0, 0, 1)
    assert result["passed"] and result["cpu_affinity"] == [180]
    assert sum(row["tests_run"] for row in records) == 126
    logs = json.loads(gzip.decompress((HERE / "raw-logs.json.gz").read_bytes()))
    assert len(logs) == 6
    for row in records:
        name = row["module"].rsplit(".", 1)[1] + ".log"
        text = logs[name]
        assert digest(text.encode()) == row["log_sha256"] and len(text.encode()) == row["log_bytes"]
        footers = re.findall(r"(?m)^Ran (\d+) tests? in ([0-9.]+)s$", text)
        assert footers and int(footers[-1][0]) == row["tests_run"]
        statuses = re.findall(r"(?m)^(OK(?: \(skipped=\d+\))?|FAILED \([^\n]+\))$", text)
        assert statuses and statuses[-1].startswith("OK")
        skip = re.search(r"skipped=(\d+)", statuses[-1])
        assert (int(skip[1]) if skip else 0) == len(row["skipped"])
        assert row["exit_code"] == 0 and row["passed"] and row["child_process_group_reaped"]
        assert not row["failures"] and not row["errors"] and row["elapsed_seconds"] > 0
        assert row["ignored_parents_before"] == {".benchmark-work": False, ".ninjars-build": False}
        relative = row["module"].replace(".", "/") + ".py"
        assert row["source_unchanged"] and row["source_sha256"] == sources[relative]
        assert row["loaded_from"] == str(Path(row["private_source"]) / relative)
        assert row["command"][1] == "-B" and row["command"][4] == row["module"]
    skipped = [skip for row in records for skip in row["skipped"]]
    assert skipped == [{"reason": "Actual final generated source is not retained in this checkout", "test": "test_actual_generated_qcd2_count_definitions_agree (tests.unit_tests.loop.test_ninjars_paper_report_v3.NinjaRSPaperReportV3Test.test_actual_generated_qcd2_count_definitions_agree)"}]

    archive, shared = read("archive-preparation.json"), read("archive-deduplication.json")
    assert archive["source_revision"] == before["archive_base_revision"] == "aec2e50b8fc714729287871b345861485a2bb837"
    assert archive["exit_code"] == 0 and archive["ignored_parents_before_overlay"] == {".benchmark-work": False, ".ninjars-build": False}
    assert shared["byte_verified_hardlinked_files"] == 32879 and shared["byte_verified_hardlinked_bytes"] == 1575285249
    root = Path(live).resolve() if live is not None else HERE.parents[4]
    prefix = "docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/dp-random-points-v2/"
    data_files = {name: value for name, value in files.items() if name.startswith(prefix)}
    assert len(data_files) == 44
    for name, checksum in (files if live is not None else data_files).items():
        assert digest((root / name).read_bytes()) == checksum, name
    return {"status": "verified", "scope": "affected software checkpoint", "tests_run": 126,
            "tests_passed": 125, "skipped": 1, "modules": 6, "bound_files": 84,
            "live_source_check": live is not None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, metavar="ROOT")
    args = parser.parse_args()
    print(json.dumps(verify(args.live), sort_keys=True))
