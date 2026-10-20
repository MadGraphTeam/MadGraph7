#!/usr/bin/env python3
"""Compile terminal diagonal/flag contracts against frozen dense references."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/qp-terminal-exact-v2"
sys.path[:0] = [str(ROOT), str(HERE)]
from terminal import fixed, helper
from prototype import logical_lines
from madgraph.various.q_polynomial import FortranPolynomialRoutines
from tests.unit_tests.various.test_q_polynomial_compiled import (
    _SPARSE_DRIVER, _SPARSE_DENSE_REFERENCES)


def run(mode, rounding, trials=12, no_automatic=False):
    work = WORK / (mode+("-rounding" if rounding else "")+
                   ("-static" if no_automatic else ""))
    work.mkdir(parents=True, exist_ok=False)
    (work/"coef_specs.inc").write_text(
        "      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n"
        "      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n")
    (work/"loop_max_coefs.inc").write_text(
        "      INTEGER LOOPMAXCOEFS\n      PARAMETER(LOOPMAXCOEFS=210)\n")
    constants = FortranPolynomialRoutines(6, updater_max_rank=2, sub_prefix="TEST_")
    (work/"constants.f").write_text(fixed(constants.write_polynomial_constant_module()))
    generator = FortranPolynomialRoutines(
        6, updater_max_rank=2, proc_prefix="TEST_", sub_prefix="MP_FULL_",
        coef_format="complex*32")
    a,b = (1,1) if mode == "expanded" else (2,1)
    base = fixed(getattr(generator, "write_"+mode+"_wl_updater")(a,b))
    (work/"full.f").write_text(base)
    routine = re.search(r"(?m)^      SUBROUTINE MP_FULL_UPDATE_WL_\d+_\d+\("
                        r".*?^      END\s*$", base, flags=re.S|re.M)[0]
    private = helper(logical_lines(routine))
    # Test-only counters prove the guarded path actually executes.
    private = private.replace("LOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT",
        "LOGICAL WL_ALREADY,WL_HALT,WL_ENTRY_INEXACT\n"
        "INTEGER FAST,FALLBACK\nCOMMON/TRACE_COUNTS/FAST,FALLBACK")
    private = private.replace(f"CALL MP_FULL_UPDATE_WL_{a}_{b}(A,",
                              f"FALLBACK=FALLBACK+1\nCALL MP_FULL_UPDATE_WL_{a}_{b}(A,",1)
    private = private.replace("RETURN\nENDIF\n", "RETURN\nENDIF\nFAST=FAST+1\n",1)
    (work/"terminal.f").write_text(fixed(private))
    (work/"old.f").write_text(_SPARSE_DENSE_REFERENCES[mode])
    driver = _SPARSE_DRIVER.replace("OLD_ROUTINE",f"MP_OLD_UPDATE_WL_{a}_{b}").replace(
        "NEW_ROUTINE",f"MP_FULL_TERMINAL_WL_{a}_{b}")
    driver = driver.replace("integer :: i,j,k,ls,ins,outs,trial,rounding,checks",
        "integer :: i,j,k,ls,ins,outs,trial,rounding,checks,flagstate,before_fast\n"
        "integer :: fast,fallback\ncommon/TRACE_COUNTS/fast,fallback")
    driver = driver.replace("checks=0\n", "checks=0\nfast=0\nfallback=0\ndo flagstate=0,1\n")
    driver = driver.replace("call ieee_set_flag(ieee_all,.false.)",
        "call ieee_set_flag(ieee_all,.false.)\n"
        "if(flagstate==1) call ieee_set_flag(ieee_inexact,.true.)")
    driver = driver.replace(f"call MP_FULL_TERMINAL_WL_{a}_{b}",
                            f"before_fast=fast\ncall MP_FULL_TERMINAL_WL_{a}_{b}")
    driver = driver.replace(f"call MP_FULL_TERMINAL_WL_{a}_{b}(a,ls,b,ins,outs,actual)",
                            f"call MP_FULL_TERMINAL_WL_{a}_{b}(a,ls,b,ins,outs,actual,flagstate==1)")
    driver = driver.replace("if(any(old_bits/=new_bits)) then",
        "if(fast>before_fast) then\n"
        "  do i=1,4\n"
        "    if(any(transfer(expected(i,:,i),old_bits,210*4)/="
        "transfer(actual(i,:,i),new_bits,210*4))) then\n"
        "      write(*,*) 'DIAGONAL MISMATCH',ls,ins,outs,trial,rounding,flagstate\n"
        "      stop 6\n"
        "    endif\n"
        "  enddo\n"
        "else if(any(old_bits/=new_bits)) then")
    driver = driver.replace("call ieee_set_rounding_mode(ieee_nearest)\nif(checks/=6000)",
                            "enddo\ncall ieee_set_rounding_mode(ieee_nearest)\nif(checks/=12000)")
    driver = driver.replace("write(*,*) 'PASS',checks",
        "if(fast==0.or.fallback==0) stop 7\nwrite(*,*) 'PASS',checks,fast,fallback")
    driver = driver.replace("if(checks/=12000)",f"if(checks/={1000*trials})")
    (work/"driver.f90").write_text(driver)
    flags = ["-O3","-fcheck=all","-ffixed-line-length-132"]
    if rounding:
        flags.append("-frounding-math")
    if no_automatic:
        flags.append("-fno-automatic")
    command = ["gfortran",*flags,"constants.f","old.f","full.f","terminal.f","driver.f90","-o","check"]
    build = subprocess.run(command,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (work/"build.log").write_text(build.stdout)
    if build.returncode:
        raise RuntimeError(build.stdout)
    result = subprocess.run([str(work/"check")],cwd=work,text=True,
                            stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (work/"run.log").write_text(result.stdout)
    if result.returncode or "PASS" not in result.stdout:
        raise RuntimeError(result.stdout)
    return {"mode":mode,"rounding_math":rounding,"no_automatic":no_automatic,
            "stdout":result.stdout.strip(),
            "command":command,"sources_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                for p in work.glob("*") if p.suffix in (".f",".f90",".inc")}}


def main():
    start=time.perf_counter()
    results=[]
    try:
        for mode in ("expanded","compact"):
            for rounding in (False,True):
                results.append(run(mode,rounding))
        report={"passed":True,"comparisons":48000,"results":results,
                "contract":"Consumed diagonal bits, input bits and all cumulative IEEE flags. "
                           "Full output bits on fallback. Fast path requires pre-set INEXACT.",
                "seconds":time.perf_counter()-start}
    except Exception as error:
        report={"passed":False,"results":results,"error":str(error)}
        raise
    finally:
        (HERE/"terminal-exact.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
