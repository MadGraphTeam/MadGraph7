#!/usr/bin/env python3
"""Run and enumerate focused tests against the archived final checkout."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
MODULES = (
    "tests.unit_tests.loop.test_ninjars_fortran",
    "tests.unit_tests.various.test_ninjars_install",
    "tests.unit_tests.various.test_ninjars_statistics",
    "tests.unit_tests.various.test_ninjars_integration_boundaries",
    "tests.unit_tests.various.test_ninjars_upstream",
)


def test_ids(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from test_ids(item)
        else:
            yield item.id()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    assert Path.cwd() == source
    sys.path.insert(0, str(source))
    suite = unittest.defaultTestLoader.loadTestsFromNames(MODULES)
    names = list(test_ids(suite))
    assert names and len(set(names)) == len(names)
    files = {}
    for name in ("madgraph", *MODULES):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(source), path
        files[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = {"passed": result.wasSuccessful() and not result.skipped,
        "tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "test_ids": names, "modules": list(MODULES),
        "archived_source_files_sha256": files, "source": str(source)}
    assert result.testsRun == len(names)
    with args.result.open("x") as stream:
        json.dump(record, stream, indent=2, sort_keys=True); stream.write("\n")
    if not record["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
