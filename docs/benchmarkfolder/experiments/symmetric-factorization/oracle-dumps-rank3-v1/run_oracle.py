#!/usr/bin/env python3
"""Build isolated kernels and retain independent oracle input/output bit dumps."""
import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import check_fallback as check

HERE = Path(__file__).resolve().parent


def load_driver(path):
    spec = importlib.util.spec_from_file_location("factor_oracle_driver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--vertices", nargs="+", default=["MP_FFV1L1_2", "MP_FFV1L2_1"])
    parser.add_argument("--flags", nargs="+", choices=check.FLAGS, default=["nlo"])
    parser.add_argument("--cpu", type=int, default=3)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    report = HERE/args.run_id
    report.mkdir(exist_ok=False)
    source = HERE/"oracle/driver.py"
    (report/"oracle-driver.py").write_bytes(source.read_bytes())
    driver = load_driver(report/"oracle-driver.py")
    # Module uses its own location to locate immutable source evidence. Use the
    # actual evidence directory while retaining the exact imported source bytes.
    driver.HERE = HERE/"oracle"
    driver.PROOF = HERE/"analysis.json"
    for name in ("factor.py", "check_fallback.py", "run_oracle.py"):
        (report/name).write_bytes((HERE/name).read_bytes())
    result = dict(complete=False, numerical_validation_pending=True,
        acceptance_eligible=False, cpu=args.cpu, results=[],
        proof_sha256=check.factor.PROOF_SHA256,
        factor_sha256=check.sha(HERE/"factor.py"),
        oracle_driver_sha256=check.sha(source),
        runner_sha256=check.sha(__file__),
        scope="Compiled stored-binary128 input/output dumps. Completion alone is "
              "not numerical validation; the independent exact Fraction oracle "
              "must validate each retained stream.")
    (report/"run.json").write_text(json.dumps(result, indent=2)+"\n")
    for flag_name in args.flags:
      for vertex in args.vertices:
       for rank in args.ranks:
        for terminal in (False, True):
            key = f"{flag_name}-{vertex}-rank{rank}-{'terminal' if terminal else 'full'}"
            work = report/key
            sources = check.write_sources(work, rank, terminal, vertex)
            (work/"oracle.f90").write_text(driver.driver(rank, vertex, terminal))
            command = ["gfortran", *check.FLAGS[flag_name],
                       *sources, "oracle.f90", "-o", "oracle-check"]
            built = check.run(command, work, "build.log")
            start = time.monotonic()
            ran = subprocess.run([str(work/"oracle-check")], cwd=work, capture_output=True)
            (work/"run.stderr").write_bytes(ran.stderr)
            with (work/"dump.txt.gz").open("wb") as output:
                with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as stream:
                    stream.write(ran.stdout)
            record = dict(key=key, rank=rank, terminal=terminal, vertex=vertex,
                flags=check.FLAGS[flag_name], build=built, returncode=ran.returncode,
                seconds=time.monotonic()-start,
                raw_dump_sha256=__import__("hashlib").sha256(ran.stdout).hexdigest(),
                gzip_sha256=check.sha(work/"dump.txt.gz"),
                executable_sha256=check.sha(work/"oracle-check"),
                sources={p.name:check.sha(p) for p in work.iterdir()
                         if p.suffix in (".f", ".f90", ".inc")})
            result["results"].append(record)
            (report/"run.json").write_text(json.dumps(result, indent=2)+"\n")
            if ran.returncode or not ran.stdout.rstrip().endswith(b"PASS_DUMP"):
                raise RuntimeError(f"{key}: dump failed; see retained stderr/raw stream")
            print(key, "dump complete", flush=True)
    result["complete"] = True
    (report/"run.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
