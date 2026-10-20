#!/usr/bin/env python3
"""Read-only verification of the completed production-helper oracle binding."""
import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())
REPORT = HERE / "exact-replay-v1"
MANIFEST = HERE / "helper-validation.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_oracle_binding():
    result = json.loads((REPORT / "result.json").read_text())
    assert result["complete"] and len(result["results"]) == 20
    assert sha(REPORT / "q_polynomial.py") == result["production_generator_sha256"]
    assert sha(REPORT / "replay_oracle.py") == result["replay_script_sha256"]
    fixtures, outputs = 0, 0
    for entry in result["results"]:
        work = REPORT / entry["key"]
        exact_path = ROOT / entry["original_exact_oracle"]
        assert sha(exact_path) == entry["original_exact_oracle_sha256"]
        exact = json.loads(exact_path.read_text())
        assert exact["complete"] and exact["passed"]
        old = exact_path.parent
        assert sha(old / "dump.txt.gz") == exact["dump_sha256"]
        assert sha(old.parent / "run.json") == entry["original_run_sha256"]
        current_bytes = gzip.decompress((work / "dump.txt.gz").read_bytes())
        previous_bytes = gzip.decompress((old / "dump.txt.gz").read_bytes())
        assert current_bytes == previous_bytes
        assert hashlib.sha256(current_bytes).hexdigest() == entry["raw_dump_sha256"]
        assert entry["raw_dump_sha256"] == entry["original_raw_dump_sha256"]
        assert entry["returncode"] == 0 and entry["dump_identical"]
        for name, digest in entry["sources"].items():
            assert sha(work / name) == digest
            assert (work / name).read_bytes() == (old / name).read_bytes()
        fixtures += len(exact["cases"])
        outputs += sum(case["output_coefficients"] for case in exact["cases"])
    assert fixtures == 200 and outputs == 250880
    return dict(streams=20, fixtures=fixtures, complex_outputs=outputs,
                all_source_and_raw_stream_bytes_identical=True)


def retained_files():
    result = {}
    for path in sorted(HERE.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        if "__pycache__" in path.parts or path.suffix in (".pyc", ".o", ".mod"):
            continue
        data = path.read_bytes()
        if data.startswith(b"\x7fELF"):
            continue
        result[str(path.relative_to(HERE))] = dict(
            sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    parser.add_argument("--current-source", action="store_true")
    args = parser.parse_args()
    binding = check_oracle_binding()
    if args.current_source:
        assert (ROOT / "madgraph/various/q_polynomial.py").read_bytes() == (
            REPORT / "q_polynomial.py").read_bytes()
        assert (ROOT / "tests/unit_tests/various/test_q_polynomial_factorized.py"
                ).read_bytes() == (HERE / "test_q_polynomial_factorized.py").read_bytes()
    if args.seal:
        record = dict(
            schema="production-qp-factor-helper-validation-v1",
            binding=binding, files=retained_files(),
            scope="Only the explicitly listed completed helper sources/tests/"
                  "oracle replay evidence. Separate future exporter/process "
                  "controls are not implied by this manifest.")
        with MANIFEST.open("x") as stream:
            stream.write(json.dumps(record, indent=2) + "\n")
    record = json.loads(MANIFEST.read_text())
    assert record["binding"] == binding
    for relative, expected in record["files"].items():
        path = HERE / relative
        assert sha(path) == expected["sha256"]
        assert path.stat().st_size == expected["bytes"]
    print(json.dumps(dict(
        verified_files=len(record["files"]), binding=binding,
        manifest_sha256=sha(MANIFEST)), indent=2))


if __name__ == "__main__":
    main()
