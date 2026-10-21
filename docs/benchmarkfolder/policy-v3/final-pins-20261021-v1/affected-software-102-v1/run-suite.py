"""Run only the frozen DP62 and report40 software tests in a clean archive."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

EXPECTED = {
    "tools/ninjars_dp_stability.py": "af91583643ae64e7922290cd08cf823eb5133ceafb07bf6d8fc23f6e60f2d0dd",
    "tests/unit_tests/loop/test_ninjars_dp_stability.py": "3d4f394bcdafe4117c3b5a8d86d74df58d189b00907ba0c28090f9725d82abb2",
    "tools/ninjars_paper_report_v3.py": "8a137adbd5d629ab45d6aa091d9ef4ec9f788ae68dd59900a76757f810dac4a2",
    "tests/unit_tests/loop/test_ninjars_paper_report_v3.py": "65efec48ce6f91a13d424d8bff3460bb8cafcaddafdaac1fbe0bfc4ffcf22e29"}
MODULES = {"tests.unit_tests.loop.test_ninjars_dp_stability": 62,
           "tests.unit_tests.loop.test_ninjars_paper_report_v3": 40}


def require(condition, message):
    if not condition: raise ValueError(message)


def dump(path, value): path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n")
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(suite):
    for item in suite:
        yield from walk(item) if isinstance(item, unittest.TestSuite) else [item]


class Recording(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs); self.records = []
    def addSuccess(self, test):
        super().addSuccess(test); self.records.append({"test_id": test.id(), "status": "passed"})
    def addSkip(self, test, reason):
        super().addSkip(test, reason); self.records.append({"test_id": test.id(), "status": "skipped", "reason": reason})
    def addFailure(self, test, err):
        super().addFailure(test, err); self.records.append({"test_id": test.id(), "status": "failed"})
    def addError(self, test, err):
        super().addError(test, err); self.records.append({"test_id": test.id(), "status": "error"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); root = args.archive.resolve(); out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    require(sys.dont_write_bytecode and os.sched_getaffinity(0) == {5}, "Use Python -B pinned to CPU5")
    require(Path.cwd() == root, "Run from the clean source archive")
    ignored = {name: (root/name).exists() for name in (".ninjars-build", ".benchmark-work")}
    require(not any(ignored.values()), "Ignored parents must be absent before discovery/tests")
    require(not (root/"tools/ninjars_profile_cli_v3.py").exists() and
            not (root/"tests/unit_tests/loop/test_ninjars_cli_report_v3.py").exists(), "Do not overlay CLI helpers/tests")
    before = {name: digest(root/name) for name in EXPECTED}
    require(before == EXPECTED, "Frozen software overlay differs")
    dump(out/"source-before.json", before)
    sys.path.insert(0, str(root))
    suite = unittest.defaultTestLoader.loadTestsFromNames(list(MODULES))
    ids = [test.id() for test in walk(suite)]
    counts = dict(Counter(test.rsplit(".", 2)[0] for test in ids))
    require(counts == MODULES and len(ids) == len(set(ids)) == 102, "Discovery must contain exactly the expected62+40 tests")
    for name in MODULES:
        module = sys.modules[name]
        require(Path(module.__file__).resolve() == root/(name.replace(".", "/")+".py"), "Test imported outside clean archive")
    dump(out/"test-inventory.json", {"modules": MODULES, "test_ids": ids, "total": len(ids)})
    start = time.monotonic()
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2, resultclass=Recording).run(suite)
    sources = {}
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            filename = Path(filename).resolve()
            if filename.suffix == ".py" and filename.is_relative_to(root) and filename.is_file():
                sources[str(filename.relative_to(root))] = digest(filename)
    for name in [*EXPECTED, "tools/ninjars_dp_stability_driver.f", "tools/ninjars_paper_timing_v3.py", "tools/ninjars_paper_numerator_v3.py"]:
        sources[name] = digest(root/name)
    after = {name: digest(root/name) for name in EXPECTED}
    dump(out/"source-after.json", after); dump(out/"tested-sources.json", sources)
    report = {"tests_run": result.testsRun, "passed": sum(row["status"] == "passed" for row in result.records),
        "skipped": [row for row in result.records if row["status"] == "skipped"],
        "failures": len(result.failures), "errors": len(result.errors), "records": result.records,
        "elapsed_seconds": time.monotonic()-start, "source_unchanged": before == after,
        "ignored_parents_before_discovery": ignored, "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "python": sys.version, "loaded_test_files": {name: sys.modules[name].__file__ for name in MODULES},
        "scope": "Software fixtures only; no native evaluations, generated-process smoke, or scientific render"}
    dump(out/"run-result.json", report)
    require(result.wasSuccessful() and before == after and result.testsRun == 102 and report["passed"] == 101 and
            report["skipped"] == [{"test_id": "tests.unit_tests.loop.test_ninjars_paper_report_v3.NinjaRSPaperReportV3Test.test_actual_generated_qcd2_count_definitions_agree",
                "status": "skipped", "reason": "Actual final generated source is not retained in this checkout"}], "Unexpected102-test outcome")


if __name__ == "__main__": main()
