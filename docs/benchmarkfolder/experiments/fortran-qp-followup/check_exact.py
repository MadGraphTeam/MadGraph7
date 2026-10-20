#!/usr/bin/env python3
"""Apply the isolated transform to the existing independent compiled oracle."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(HERE)]
from prototype import logical_lines, transform_fixed
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines
from tests.unit_tests.various.test_q_polynomial_compiled import (
    SparseQPWavefunctionUpdateTest)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    targets = {
        name: getattr(FortranPolynomialRoutines, name)
        for name in ("write_compact_wl_updater", "write_expanded_wl_updater")}
    def wrapped(name):
        def generate(self, *args, **kwargs):
            source = targets[name](self, *args, **kwargs)
            if "WL_NEAREST" not in source:
                return source
            # Exercise the fixed-form path used on actual copied exports too.
            with tempfile.TemporaryDirectory(prefix="qp-source-roundtrip-") as temp:
                filename = Path(temp) / "routine.f"
                with FortranWriter(str(filename)) as writer:
                    writer.writelines(source)
                changed, count = transform_fixed(filename.read_text())
                assert count == 1
                return logical_lines(changed)
        return generate
    start = time.perf_counter()
    with patch.object(FortranPolynomialRoutines, "write_compact_wl_updater",
                      wrapped("write_compact_wl_updater")), \
         patch.object(FortranPolynomialRoutines, "write_expanded_wl_updater",
                      wrapped("write_expanded_wl_updater")):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(SparseQPWavefunctionUpdateTest)
        with (HERE / "exact-tests.log").open("w") as stream:
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    data = {
        "passed": result.wasSuccessful(),
        "tests": result.testsRun,
        "seconds": time.perf_counter()-start,
        "prototype_sha256": digest(HERE / "prototype.py"),
        "production_generator_sha256": digest(ROOT / "madgraph/various/q_polynomial.py"),
        "independent_regression_sha256": digest(
            ROOT / "tests/unit_tests/various/test_q_polynomial_compiled.py"),
        "scope": "30,000 comparisons of complete output/input bits and IEEE exception flags "
                 "against independent frozen dense references; both updater modes, "
                 "all four rounding modes, finite/extreme/subnormal/nonfinite/signaling-NaN "
                 "inputs, signed zeros, empty dimensions, and forced layout rejection.",
    }
    (HERE / "exact-tests.json").write_text(json.dumps(data, indent=2)+"\n")
    print(json.dumps(data, indent=2))
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
