#!/usr/bin/env python3
"""Compile isolated factor guards with actual SA/NLO flags; no production edits."""
import sys
sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import factor

ROOT = factor.gather.ROOT
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines
from tests.unit_tests.various.test_q_polynomial_compiled import _SPARSE_DRIVER

FLAGS = {
    "standalone": ["-O3", "-w", "-fPIC", "-ffixed-line-length-132"],
    "nlo": ["-O3", "-w", "-fPIC", "-ffixed-line-length-132", "-fno-automatic"],
}

TRAPS = """program traps
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
implicit none
complex(16) a(4,0:209,4),b(4,0:14,4),out(4,0:209,4)
integer flag,checks,fast,fallback
common/FACTOR_TRACE/fast,fallback
logical raised(size(ieee_all))
checks=0
do flag=1,size(ieee_all)
 if(.not.ieee_support_halting(ieee_all(flag))) cycle
 call ieee_set_halting_mode(ieee_all,.false.)
 a=0
 b=0
 fast=0
 fallback=0
 call ieee_set_flag(ieee_all,.false.)
 call ieee_set_halting_mode(ieee_all(flag),.true.)
 call NEW_ROUTINE(a,4,b,4,4,out,.true.)
 call ieee_set_halting_mode(ieee_all,.false.)
 call ieee_get_flag(ieee_all,raised)
 if(fast/=0.or.fallback/=1.or.any(raised).or.any(out/=0)) stop 1
 checks=checks+1
enddo
if(checks/=size(ieee_all)) stop 2
write(*,*) 'PASS',checks
end program
"""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def instrument(source):
    source = source.replace("  implicit none", "  implicit none\n"
        "  INTEGER FAST,FALLBACK\n  COMMON/FACTOR_TRACE/FAST,FALLBACK", 1)
    assert source.count("C FACTOR_FAST_PATH") == 1
    source = source.replace("C FACTOR_FAST_PATH", "C FACTOR_FAST_PATH\nFAST=FAST+1")
    import re
    source, count = re.subn(
        r"(?m)^(\s*CALL TEST_(?:UPDATE|TERMINAL)_WL_\d_1(?:_DENSE)?\(.*)$",
        r"  FALLBACK=FALLBACK+1\n\1", source)
    assert count >= 5
    return source


def write_sources(work, rank, terminal, vertex):
    work.mkdir(parents=True, exist_ok=True)
    (work/"coef_specs.inc").write_text("      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n"
        "      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n")
    (work/"loop_max_coefs.inc").write_text("      INTEGER LOOPMAXCOEFS\n"
        "      PARAMETER(LOOPMAXCOEFS=210)\n")
    gen = FortranPolynomialRoutines(6, updater_max_rank=2, proc_prefix="TEST_",
        sub_prefix="TEST_", coef_format="complex*32")
    old = f"TEST_{'TERMINAL' if terminal else 'UPDATE'}_WL_{rank}_1"
    sources = {
        "constants.f": gen.write_polynomial_constant_module(),
        "original.f": gen.write_compact_wl_updater(rank, 1),
        "factor.f": instrument(factor.generate(gen, rank, vertex, terminal)),
    }
    if terminal:
        sources["original.f"] += gen.write_compact_wl_updater(rank, 1, terminal=True)
    for name, source in sources.items():
        with FortranWriter(str(work/name)) as writer:
            writer.writelines(source)
    driver = _SPARSE_DRIVER.replace("OLD_ROUTINE", old).replace("NEW_ROUTINE", old+"_FACTOR")
    driver = driver.replace("integer :: i,j,k,ls,ins,outs,trial,rounding,checks",
        "integer :: i,j,k,ls,ins,outs,trial,rounding,checks,fast,fallback\n"
        "common/FACTOR_TRACE/fast,fallback")
    driver = driver.replace("checks=0", "checks=0\nfast=0\nfallback=0", 1)
    if terminal:
        driver = driver.replace("outs,expected)", "outs,expected,.false.)")
    driver = driver.replace("outs,actual)", "outs,actual,.false.)")
    driver = driver.replace("if(checks/=6000) stop 4",
        "if(checks/=6000) stop 4\nif(fast/=0.or.fallback/=checks) stop 5")
    (work/"fallback.f90").write_text(driver)
    (work/"traps.f90").write_text(TRAPS.replace("NEW_ROUTINE", old+"_FACTOR"))
    return sources


def run(command, work, log):
    start = time.monotonic()
    result = subprocess.run(command, cwd=work, capture_output=True, text=True)
    (work/log).write_text(result.stdout+result.stderr)
    record = dict(command=command, returncode=result.returncode,
                  seconds=time.monotonic()-start, output=result.stdout+result.stderr)
    if result.returncode:
        raise RuntimeError(f"{work/log}: {record['output']}")
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--flags", nargs="+", choices=FLAGS, default=list(FLAGS))
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    destination = HERE/args.run_id
    destination.mkdir(parents=True, exist_ok=False)
    result = dict(complete=False, acceptance_eligible=False, cpu=args.cpu,
                  source_sha256=sha(__file__), factor_sha256=sha(HERE/"factor.py"),
                  proof_sha256=factor.PROOF_SHA256,
                  generator_sha256=sha(ROOT/"madgraph/various/q_polynomial.py"),
                  compiler=run(["gfortran", "--version"], destination, "compiler.log"),
                  scope="Exact public fallback with false entry-INEXACT, 6000 shape/"
                        "rounding/exception fixtures per configuration; each supported "
                        "enabled IEEE trap also forces fallback. No fast-path numerical "
                        "or process/performance claim.", results=[])
    (destination/"factor-generator.py").write_bytes((HERE/"factor.py").read_bytes())
    (destination/"check-fallback.py").write_bytes(Path(__file__).read_bytes())
    for flag_name in args.flags:
        for rank in args.ranks:
            for terminal in (False, True):
                name = f"{flag_name}-rank{rank}-{'terminal' if terminal else 'full'}"
                work = destination/name
                sources = write_sources(work, rank, terminal, "MP_FFV1L2_1")
                build = run(["gfortran", *FLAGS[flag_name], "-c", *sources],
                            work, "build.log")
                records = {}
                for driver in ("fallback", "traps"):
                    records[driver+"_build"] = run(
                        ["gfortran", *FLAGS[flag_name], "constants.o", "original.o",
                         "factor.o", driver+".f90", "-o", driver], work, driver+"-build.log")
                    records[driver] = run([str(work/driver)], work, driver+".log")
                    records[driver]["executable_sha256"] = sha(work/driver)
                result["results"].append(dict(name=name, build=build, tests=records,
                    sources={p.name:sha(p) for p in work.iterdir()
                             if p.suffix in (".f", ".f90", ".inc")}))
                (destination/"results.json").write_text(json.dumps(result, indent=2)+"\n")
                print(name, records["fallback"]["output"].strip(),
                      records["traps"]["output"].strip(), flush=True)
    result["complete"] = True
    result["fallback_comparisons"] = len(result["results"])*6000
    result["trap_comparisons"] = len(result["results"])*5
    (destination/"results.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
