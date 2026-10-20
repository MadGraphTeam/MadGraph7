#!/usr/bin/env python3
"""Retain actual binary64 input/output bit dumps for independent exact checks."""
import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import support

HERE = Path(__file__).resolve().parent


def load_driver(path):
    spec = importlib.util.spec_from_file_location("binary64_oracle_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--vertices", nargs="+", default=["FFV1L1_2", "FFV1L2_1"])
    parser.add_argument("--flags", nargs="+", choices=support.FLAGS, default=["nlo"])
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    report = HERE / args.run_id
    report.mkdir(exist_ok=False)
    source = HERE / "oracle/driver.py"
    (report / "oracle-driver.py").write_bytes(source.read_bytes())
    driver = load_driver(report / "oracle-driver.py")
    driver.HERE = HERE / "oracle"
    driver.PROOF = driver.HERE / "source-parity.json"
    for name in ("factor.py", "support.py", "run_oracle.py"):
        (report / name).write_bytes((HERE / name).read_bytes())
    (report / "source-parity.json").write_bytes(driver.PROOF.read_bytes())
    result = dict(
        complete=False, numerical_validation_pending=True,
        acceptance_eligible=False, cpu=args.cpu, results=[],
        inherited_proof_sha256=support.factor.PROOF_SHA256,
        source_parity_sha256=support.sha(driver.PROOF),
        factor_sha256=support.sha(HERE / "factor.py"),
        oracle_driver_sha256=support.sha(source),
        runner_sha256=support.sha(__file__),
        support_sha256=support.sha(HERE / "support.py"),
        production_generator_sha256=support.sha(
            support.ROOT / "madgraph/various/q_polynomial.py"),
        compiler=support.run(["gfortran", "--version"], report, "compiler.log"),
        scope="Compiled stored-binary64 full-updater input/output dumps. "
              "Completion alone is not numerical validation; the independent "
              "exact Fraction oracle must validate every retained stream. "
              "No DP terminal or process/performance claim.")
    def retain():
        (report / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    retain()
    try:
        for flag_name in args.flags:
            for vertex in args.vertices:
                for rank in args.ranks:
                    key = f"{flag_name}-{vertex}-rank{rank}-full"
                    work = report / key
                    sources = support.write_sources(work, rank, vertex)
                    (work / "oracle.f90").write_text(driver.driver(rank, vertex, False))
                    built = support.run(
                        ["gfortran", *support.FLAGS[flag_name], *sources,
                         "oracle.f90", "-o", "oracle-check"], work, "build.log")
                    start = time.monotonic()
                    ran = subprocess.run([str(work / "oracle-check")],
                                         cwd=work, capture_output=True)
                    (work / "run.stderr").write_bytes(ran.stderr)
                    with (work / "dump.txt.gz").open("wb") as output:
                        with gzip.GzipFile(filename="", mode="wb", fileobj=output,
                                           mtime=0) as stream:
                            stream.write(ran.stdout)
                    result["results"].append(dict(
                        key=key, rank=rank, terminal=False, vertex=vertex,
                        flags=support.FLAGS[flag_name], build=built,
                        returncode=ran.returncode,
                        seconds=time.monotonic() - start,
                        raw_dump_sha256=hashlib.sha256(ran.stdout).hexdigest(),
                        gzip_sha256=support.sha(work / "dump.txt.gz"),
                        executable_sha256=support.sha(work / "oracle-check"),
                        sources={p.name: support.sha(p) for p in work.iterdir()
                                 if p.suffix in (".f", ".f90", ".inc")}))
                    retain()
                    if ran.returncode or not ran.stdout.rstrip().endswith(b"PASS_DUMP"):
                        raise RuntimeError(f"{key}: dump failed; retained raw stream/stderr")
                    print(key, "dump complete", flush=True)
        result["complete"] = True
        retain()
    except Exception as error:
        result["error"] = repr(error)
        retain()
        raise


if __name__ == "__main__":
    main()
