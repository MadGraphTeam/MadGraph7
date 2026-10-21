"""Read-only verification of the DP62/report40 software checkpoint."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

if not __debug__:
    raise RuntimeError("Optimized Python (-O/-OO) is unsupported for evidence verification")

HERE = Path(__file__).resolve().parent
OVERLAYS = {
    "tools/ninjars_dp_stability.py": "af91583643ae64e7922290cd08cf823eb5133ceafb07bf6d8fc23f6e60f2d0dd",
    "tests/unit_tests/loop/test_ninjars_dp_stability.py": "3d4f394bcdafe4117c3b5a8d86d74df58d189b00907ba0c28090f9725d82abb2",
    "tools/ninjars_paper_report_v3.py": "8a137adbd5d629ab45d6aa091d9ef4ec9f788ae68dd59900a76757f810dac4a2",
    "tests/unit_tests/loop/test_ninjars_paper_report_v3.py": "65efec48ce6f91a13d424d8bff3460bb8cafcaddafdaac1fbe0bfc4ffcf22e29"}
MODULES = {"tests.unit_tests.loop.test_ninjars_dp_stability": 62,
           "tests.unit_tests.loop.test_ninjars_paper_report_v3": 40}
SKIP = {"test_id": "tests.unit_tests.loop.test_ninjars_paper_report_v3.NinjaRSPaperReportV3Test.test_actual_generated_qcd2_count_definitions_agree",
        "status": "skipped", "reason": "Actual final generated source is not retained in this checkout"}


def require(condition, message):
    if not condition: raise ValueError(message)


def digest(raw): return hashlib.sha256(raw).hexdigest()
def read(name): return json.loads((HERE/name).read_text())


def verify(live=None):
    manifest = read("manifest.json")
    require(manifest["schema"] == "ninjars-affected-software-102-v1", "Wrong software checkpoint schema")
    require({p.name for p in HERE.iterdir() if p.is_file()} - {"manifest.json"} == set(manifest["payloads"]),
            "Software payload inventory differs")
    for name, record in manifest["payloads"].items():
        raw = (HERE/name).read_bytes()
        require(len(raw) == record["bytes"] and digest(raw) == record["sha256"], "Changed software payload: "+name)
    before, after = read("source-before.json"), read("source-after.json")
    require(before == after == OVERLAYS, "The four tested overlays changed")
    preparation = read("archive-preparation.json")
    require(preparation["source_revision"] == "9b5abb345452becddd37ee603c9bed17aed7db04" and
            preparation["explicit_overlays"] == OVERLAYS and
            preparation["ignored_parents_before_overlay"] == preparation["ignored_parents_after_overlay"] ==
                {".benchmark-work": False, ".ninjars-build": False} and
            preparation["untracked_cli_helpers_or_tests"] is False, "Archive/overlay preparation differs")
    sources = read("tested-sources.json")
    require(len(sources) == 10 and all(sources[name] == value for name, value in OVERLAYS.items()) and
            not any("profile_cli" in name or "cli_report" in name for name in sources), "Wrong tested source selection")
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress((HERE/"software-sources.tar.gz").read_bytes())), mode="r:") as archive:
        members = archive.getmembers()
        require(len(members) == 10 and {member.name for member in members} == set(sources), "Source snapshot inventory differs")
        for member in members:
            require(member.isfile() and not Path(member.name).is_absolute() and ".." not in Path(member.name).parts and
                    digest(archive.extractfile(member).read()) == sources[member.name], "Changed source snapshot: "+member.name)
    inventory, result, command = read("test-inventory.json"), read("run-result.json"), read("command.json")
    ids = inventory["test_ids"]
    require(inventory["modules"] == MODULES and inventory["total"] == len(ids) == len(set(ids)) == 102 and
            dict(Counter(name.rsplit(".", 2)[0] for name in ids)) == MODULES, "Discovery is not exactly62+40 tests")
    log = (HERE/"test.log").read_text()
    observed = re.findall(r"(?m)^test_\w+ \(([^()\n]+)\) \.\.\. (ok|skipped '[^\n]*'|FAIL|ERROR)$", log)
    rows = [{"test_id": name, "status": "passed" if state == "ok" else "skipped",
             **({"reason": state[9:-1]} if state.startswith("skipped '") else {})} for name, state in observed]
    require([row["test_id"] for row in rows] == ids and rows == result["records"] and
            Counter(row["status"] for row in rows) == {"passed": 101, "skipped": 1}, "Raw log/outcomes do not bind102 tests")
    require(re.search(r"(?m)^Ran 102 tests in [0-9.]+s$", log) is not None and log.rstrip().endswith("OK (skipped=1)") and
            result["tests_run"] == 102 and result["passed"] == 101 and result["failures"] == result["errors"] == 0 and
            result["skipped"] == [SKIP] and result["source_unchanged"] is True and result["cpu_affinity"] == [5] and
            result["ignored_parents_before_discovery"] == {".benchmark-work": False, ".ninjars-build": False},
            "Unexpected test/skip/source/cpu result")
    require(command["exit_code"] == 0 and command["process_group_reaped"] is True and
            command["command"][:3] == ["taskset", "-c", "5"] and "-B" in command["command"] and
            (HERE/"stderr.log").read_bytes() == b"", "Software command failed or left its process group")
    for name, value in OVERLAYS.items():
        row = command["source_state_after"][name]
        require(row["expected"] == row["archive_after"] == row["root_after"] == value, "Source changed after execution")
    for name, expected, count in (("dp-independent-source-review.json", "d62d8b3c27954138d4779cea165b206df98939836fa47d8ddbf16f42d074728c", 62),
                                   ("report-independent-source-review.json", "33c2684dc243f62b03c413efee3faaa65ff68ea26cbe41be3a5a9e928078eb24", 40)):
        require(digest((HERE/name).read_bytes()) == expected, "Independent receipt is not the reviewed snapshot")
        receipt = read(name)
        require(receipt["owner_software_tests"] == count and receipt["audit_reran_tests"] is False and
                receipt["native_calls"] == receipt["compiler_calls"] == 0, "Wrong independent source-review scope")
    if live is not None:
        root = Path(live).resolve()
        for name, value in sources.items(): require(digest((root/name).read_bytes()) == value, "Live source differs: "+name)
    return {"status": "verified", "tests_run": 102, "passed": 101, "skipped": 1,
            "failures": 0, "errors": 0, "source_images": 10, "cpu_affinity": [5],
            "live_source_check": live is not None, "scope": "Software checkpoint only; no numerical qualification"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", type=Path, help="Additionally compare the exact tested source images to this checkout")
    print(json.dumps(verify(parser.parse_args().live), sort_keys=True))
