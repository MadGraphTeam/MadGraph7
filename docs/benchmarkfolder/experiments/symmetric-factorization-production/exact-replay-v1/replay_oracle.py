#!/usr/bin/env python3
"""Bind production-generated QP helpers to the sealed exact-oracle campaign."""
import sys
sys.dont_write_bytecode = True

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())
SEALED = HERE.parent / "symmetric-factorization"
sys.path[:0] = [str(ROOT), str(SEALED)]
import check_fallback as check
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    report = HERE / args.run_id
    report.mkdir(exist_ok=False)
    generator_path = ROOT / "madgraph/various/q_polynomial.py"
    (report / "q_polynomial.py").write_bytes(generator_path.read_bytes())
    (report / "replay_oracle.py").write_bytes(Path(__file__).read_bytes())
    result = dict(
        complete=False, acceptance_eligible=False,
        production_generator_sha256=sha(generator_path),
        replay_script_sha256=sha(Path(__file__)),
        sealed_campaign_sha256=sha(SEALED / "campaign-seal.json"),
        scope="Current production-generated helpers/public routines/constants "
              "must equal the sealed independently validated sources, after "
              "renaming only the private helper declaration for the unchanged "
              "oracle driver. Compile with each original actual SA/NLO flag set "
              "and require every raw binary128 input/output/flag record identical "
              "to the independently exact-validated stream.", results=[])
    def retain():
        (report / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    retain()
    try:
        for campaign in ("oracle-dumps-nlo-v1", "oracle-dumps-sa-v1",
                         "oracle-dumps-rank3-v1"):
            metadata = json.loads((SEALED / campaign / "run.json").read_text())
            assert metadata["complete"]
            for entry in metadata["results"]:
                key = campaign + "-" + entry["key"]
                previous = SEALED / campaign / entry["key"]
                exact = previous / "exact-oracle.json"
                assert json.loads(exact.read_text())["passed"]
                work = report / key
                work.mkdir()
                rank, terminal = entry["rank"], entry["terminal"]
                vertex = entry["vertex"].removeprefix("MP_")
                generator = FortranPolynomialRoutines(
                    6, updater_max_rank=2, proc_prefix="TEST_", sub_prefix="TEST_",
                    coef_format="complex*32")
                private = generator.write_factorized_wl_updater(rank, vertex, terminal)
                name = "TEST_%s_WL_%d_1_FACTOR" % (
                    "TERMINAL" if terminal else "UPDATE", rank)
                declaration = "SUBROUTINE " + name + "_" + vertex + "("
                assert private.count(declaration) == 1
                private = private.replace(declaration, "SUBROUTINE " + name + "(", 1)
                original = generator.write_compact_wl_updater(rank, 1)
                if terminal:
                    original += "\n\n" + generator.write_compact_wl_updater(
                        rank, 1, terminal=True)
                generated = {
                    "constants.f": generator.write_polynomial_constant_module(),
                    "original.f": original,
                    "factor.f": check.instrument(private),
                }
                for filename, text in generated.items():
                    with FortranWriter(str(work / filename)) as writer:
                        writer.writelines(text)
                    assert (work / filename).read_bytes() == (previous / filename).read_bytes()
                for filename in ("coef_specs.inc", "loop_max_coefs.inc", "oracle.f90"):
                    (work / filename).write_bytes((previous / filename).read_bytes())
                build = check.run(
                    ["gfortran", *entry["flags"], *generated, "oracle.f90",
                     "-o", "oracle-check"], work, "build.log")
                start = time.monotonic()
                ran = subprocess.run([str(work / "oracle-check")], cwd=work,
                                     capture_output=True)
                (work / "run.stderr").write_bytes(ran.stderr)
                with (work / "dump.txt.gz").open("wb") as target:
                    with gzip.GzipFile(filename="", mode="wb", fileobj=target,
                                       mtime=0) as stream:
                        stream.write(ran.stdout)
                expected = gzip.decompress((previous / "dump.txt.gz").read_bytes())
                record = dict(
                    key=key, returncode=ran.returncode,
                    dump_identical=ran.stdout == expected,
                    seconds=time.monotonic() - start, build=build,
                    original_exact_oracle=str(exact.relative_to(ROOT)),
                    original_exact_oracle_sha256=sha(exact),
                    original_run_sha256=sha(SEALED / campaign / "run.json"),
                    raw_dump_sha256=hashlib.sha256(ran.stdout).hexdigest(),
                    original_raw_dump_sha256=hashlib.sha256(expected).hexdigest(),
                    executable_sha256=sha(work / "oracle-check"),
                    sources={p.name: sha(p) for p in work.iterdir()
                             if p.suffix in (".f", ".f90", ".inc")})
                result["results"].append(record)
                retain()
                assert ran.returncode == 0 and ran.stdout.rstrip().endswith(b"PASS_DUMP")
                assert record["dump_identical"], key
                print(key, "exact validated stream reproduced", flush=True)
        assert len(result["results"]) == 20
        result["complete"] = True
        retain()
    except Exception as error:
        result["error"] = repr(error)
        retain()
        raise


if __name__ == "__main__":
    main()
