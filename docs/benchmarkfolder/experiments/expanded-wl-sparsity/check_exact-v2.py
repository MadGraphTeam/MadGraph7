#!/usr/bin/env python3
"""Check isolated QP zero-skip prototypes against the unchanged generator."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/expanded-wl-sparsity/exact-v2"
sys.path[:0] = [str(ROOT), str(HERE)]
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines
from prototype import transform

PAIRS = {"expanded": [(i, j) for i in range(5) for j in range(3)],
         "compact": [(i, j) for i in range(2, 5) for j in range(1, 3)]}
TRIALS = 10
FLAGS = ["-O3", "-fcheck=all", "-ffixed-line-length-132"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(mode, rounding_math):
    work = WORK / (mode + ("-rounding" if rounding_math else "-default"))
    work.mkdir(parents=True, exist_ok=False)
    (work / "coef_specs.inc").write_text(
        "      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n"
        "      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n")
    (work / "loop_max_coefs.inc").write_text(
        "      INTEGER LOOPMAXCOEFS\n"
        "      PARAMETER(LOOPMAXCOEFS=210)\n")
    generator = FortranPolynomialRoutines(
        6, updater_max_rank=2, proc_prefix="TEST_", sub_prefix="MP_OLD_",
        coef_format="complex*32")
    constants = FortranPolynomialRoutines(
        6, updater_max_rank=2, proc_prefix="TEST_", sub_prefix="TEST_",
        coef_format="complex*32")
    with FortranWriter(str(work / "constants.f")) as writer:
        writer.writelines(constants.write_polynomial_constant_module())
    method = getattr(generator, f"write_{mode}_wl_updater")
    with FortranWriter(str(work / "old.f")) as writer:
        writer.writelines("\n".join(method(*ranks) for ranks in PAIRS[mode]))
    transformed, count = transform((work / "old.f").read_text(), mode)
    assert count == len(PAIRS[mode])
    (work / "new.f").write_text(transformed.replace("MP_OLD_", "MP_NEW_"))
    calls = []
    for left, right in PAIRS[mode]:
        calls.append(f"""
expected=sentinel
actual=sentinel
call ieee_set_flag(ieee_all,.false.)
call MP_OLD_UPDATE_WL_{left}_{right}(a,ls,b,ins,outs,expected)
call ieee_get_flag(ieee_all,old_flags)
call ieee_set_flag(ieee_all,.false.)
call MP_NEW_UPDATE_WL_{left}_{right}(a,ls,b,ins,outs,actual)
call ieee_get_flag(ieee_all,new_flags)
old_bits=transfer(expected,old_bits)
new_bits=transfer(actual,new_bits)
if(any(old_bits/=new_bits)) then
  write(*,*) 'BIT MISMATCH', {left},{right},ls,ins,outs,trial,rounding
  stop 1
endif
if(any(old_flags.neqv.new_flags)) then
  write(*,*) 'FLAGS MISMATCH', {left},{right},ls,ins,outs,trial,rounding
  write(*,*) old_flags,new_flags
  stop 2
endif
if(any(transfer(a,old_bits)/=saved_a).or.any(transfer(b,b_bits)/=saved_b)) stop 3
checks=checks+1
""")
    expected_checks = len(PAIRS[mode]) * 5**3 * TRIALS * 4
    (work / "driver.f90").write_text(f"""
program exact_sparse
use iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
use TEST_POLYNOMIAL_CONSTANTS
implicit none
complex(16) :: a(4,0:209,4),b(4,0:14,4),expected(4,0:209,4),actual(4,0:209,4)
complex(16) :: sentinel
integer(int64) :: old_bits(4*4*210*4),new_bits(4*4*210*4),saved_a(4*4*210*4)
integer(int64) :: b_bits(4*4*15*4),saved_b(4*4*15*4)
real(16) :: scale,inf,nan,small
logical :: old_flags(size(ieee_all)),new_flags(size(ieee_all))
type(ieee_round_type) :: modes(4)
integer :: i,j,k,ls,ins,outs,trial,rounding,checks
modes=[ieee_nearest,ieee_up,ieee_down,ieee_to_zero]
sentinel=cmplx(-123.25_16,456.125_16,kind=16)
inf=ieee_value(0.0_16,ieee_positive_inf)
nan=ieee_value(0.0_16,ieee_quiet_nan)
small=nearest(0.0_16,1.0_16)
checks=0
do rounding=1,4
  call ieee_set_rounding_mode(modes(rounding))
  do trial=1,{TRIALS}
    do k=1,4
      do j=0,209
        do i=1,4
          scale=1.0_16
          if(trial==3.and.mod(j,3)==0) scale=1.e2000_16
          if(trial==3.and.mod(j,3)==1) scale=1.e-2000_16
          a(i,j,k)=cmplx(real(mod(17*i+31*j+13*k,43)-21,kind=16)/7, &
            real(mod(37*i+11*j+7*k,31)-15,kind=16)/13,kind=16)*scale
          if(trial==2.and.mod(i+j+k,3)/=0) a(i,j,k)=0
          if(trial==4) a(i,j,k)=0
          if(trial==5) a(i,j,k)=cmplx((-1.0_16)**i,(-1.0_16)**(i+1),kind=16)
          if(trial==6.and.mod(i+j+k,2)==0) a(i,j,k)=cmplx(-0.0_16,0.0_16,kind=16)
          if(trial==7) a(i,j,k)=cmplx(0.0_16,-0.0_16,kind=16)
          if(trial==7.and.mod(i+j+k,5)==0) a(i,j,k)=cmplx(inf,nan,kind=16)
          if(trial==8.and.mod(i+j+k,3)/=0) a(i,j,k)=0
          if(trial==9) a(i,j,k)=cmplx(small,-small,kind=16)
          if(trial==10.and.mod(i+j+k,2)==0) a(i,j,k)=0
          if(trial==10.and.mod(i+j+k,2)==1) a(i,j,k)=cmplx(huge(scale),-huge(scale),kind=16)
        enddo
      enddo
      do j=0,14
        do i=1,4
          b(i,j,k)=cmplx(real(mod(19*i+29*j+3*k,17)-8,kind=16)/3, &
            real(mod(11*i+17*j+13*k,23)-11,kind=16)/11,kind=16)
          if(trial==2.and.mod(i+j+k,3)/=0) b(i,j,k)=0
          if(trial==4.or.trial==7) b(i,j,k)=0
          if(trial==5) b(i,j,k)=cmplx(1.0_16,-1.0_16,kind=16)
          if(trial==6.and.mod(i+j+k,2)==1) b(i,j,k)=cmplx(0.0_16,-0.0_16,kind=16)
          if(trial==8.and.mod(i+j+k,5)==0) b(i,j,k)=cmplx(inf,nan,kind=16)
          if(trial==9) b(i,j,k)=cmplx(0.25_16,-0.25_16,kind=16)
          if(trial==10.and.mod(i+j+k,2)==1) b(i,j,k)=0
        enddo
      enddo
    enddo
    saved_a=transfer(a,saved_a)
    saved_b=transfer(b,saved_b)
    do ls=0,4
      do ins=0,4
        do outs=0,4
{''.join(calls)}
        enddo
      enddo
    enddo
  enddo
enddo
call ieee_set_rounding_mode(ieee_nearest)
if(checks/={expected_checks}) stop 4
write(*,*) 'PASS',checks
end program
""")
    flags = FLAGS + (["-frounding-math"] if rounding_math else [])
    command = ["gfortran", *flags, "constants.f", "old.f", "new.f", "driver.f90",
               "-o", "check"]
    compiled = subprocess.run(command, cwd=work, capture_output=True, text=True)
    (work / "build.log").write_text(compiled.stdout + compiled.stderr)
    result = {"mode": mode, "flags": flags, "expected_checks": expected_checks,
              "source_hashes": {p.name: sha(p) for p in work.glob("*.f*")},
              "build_passed": compiled.returncode == 0}
    if compiled.returncode == 0:
        checked = subprocess.run(["taskset", "-c", "3", "./check"], cwd=work,
                                 capture_output=True, text=True)
        result.update(returncode=checked.returncode, stdout=checked.stdout,
                      stderr=checked.stderr,
                      passed=checked.returncode == 0 and "PASS" in checked.stdout)
        (work / "run.log").write_text(checked.stdout + checked.stderr)
    else:
        result.update(passed=False, stderr=compiled.stderr)
    print(json.dumps(result), flush=True)
    return result


def main():
    os.sched_setaffinity(0, {3})
    results = []
    report = {"scope": "Isolated generated QP prototypes only; no production edit",
              "comparison": "Entire output bit patterns, input preservation and all IEEE flags",
              "ranks": PAIRS, "dimensions": list(range(5)), "trials": TRIALS,
              "rounding_modes": ["nearest", "up", "down", "toward zero"],
              "inputs": ["dense", "sparse", "1e±2000", "all zero", "cancellation",
                         "signed zero", "nonfinite A and zero B", "nonfinite B",
                         "subnormal", "overflow"],
              "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
              "prototype_sha256": sha(HERE / "prototype.py"),
              "test_sha256": sha(Path(__file__)),
              "generator_sha256": sha(ROOT / "madgraph/various/q_polynomial.py"),
              "results": results}
    try:
        for mode in PAIRS:
            for rounding_math in (False, True):
                results.append(check(mode, rounding_math))
    finally:
        (HERE / "exact-regression.json").write_text(json.dumps(report, indent=2) + "\n")
    if not all(r["passed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
