#!/usr/bin/env python3
"""Compiled bit-pattern regression for the isolated expanded UPDATE_WL edit.

Run from the repository root after sourcing .ninjars-build/dev-env.sh.
This does not edit the generator or any benchmark output.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types

ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "madgraph/__init__.py").is_file())
sys.path.insert(0, str(ROOT))
from madgraph.iolibs.file_writers import FortranWriter

# Keep the original side reproducible after the production generator changes.
BASE_REVISION = "d47f0bc4f3fd47f8bf6e62c50447cbd5300d9561"
BASE_SOURCE = subprocess.check_output(
    ["git", "show", BASE_REVISION + ":madgraph/various/q_polynomial.py"], cwd=ROOT, text=True)
legacy = types.ModuleType("original_q_polynomial")
exec(compile(BASE_SOURCE, BASE_REVISION + "/q_polynomial.py", "exec"), legacy.__dict__)
FortranPolynomialRoutines = legacy.FortranPolynomialRoutines

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("expanded_wl_experiment", HERE / "run.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)

FLAGS = ["-O3", "-fcheck=all", "-ffixed-line-length-132"]
TRIALS = 6


def precision(work, kind):
    generator = FortranPolynomialRoutines(
        6, updater_max_rank=2, proc_prefix="TEST_", sub_prefix="OLD_",
        coef_format=f"complex*{2*kind}")
    original, calls = [], []
    for left in range(5):
        for right in range(3):
            original.append(generator.write_expanded_wl_updater(left, right))
            calls.append(f"""
expected=sentinel
actual=sentinel
call OLD_UPDATE_WL_{left}_{right}(a,ls,b,ins,outs,expected)
call NEW_UPDATE_WL_{left}_{right}(a,ls,b,ins,outs,actual)
old_bits=transfer(expected,old_bits)
new_bits=transfer(actual,new_bits)
if(any(old_bits/=new_bits)) then
  write(*,*) 'BIT MISMATCH', {kind},{left},{right},ls,ins,outs,trial
  stop 1
endif
if(any(a/=saved_a).or.any(b/=saved_b)) stop 3
checks=checks+1
""")
    old_path = work / f"original-{kind}.f"
    with FortranWriter(str(old_path)) as writer:
        writer.writelines("\n".join(original))
    old_text = old_path.read_text()
    transformed, count = experiment.interchange(old_text)
    if count != 15:
        raise RuntimeError(f"Expected exactly 15 expanded routines, transformed {count}")
    new_path = work / f"transformed-{kind}.f"
    new_path.write_text(transformed.replace("OLD_UPDATE_WL_", "NEW_UPDATE_WL_"))
    expected_checks = 15 * 5**3 * TRIALS
    driver = work / f"driver-{kind}.f90"
    driver.write_text(f"""
program exact_expanded
use iso_fortran_env, only: int64
use TEST_POLYNOMIAL_CONSTANTS
implicit none
complex({kind}) :: a(4,0:209,4), b(4,0:14,4)
complex({kind}) :: saved_a(4,0:209,4), saved_b(4,0:14,4)
complex({kind}) :: expected(4,0:209,4), actual(4,0:209,4), sentinel
real({kind}) :: scale
integer(int64) :: old_bits({2*kind//8*4*210*4}), new_bits({2*kind//8*4*210*4})
integer :: i,j,k,ls,ins,outs,trial,checks
sentinel=cmplx(-123.25_{kind},456.125_{kind},kind={kind})
checks=0
do trial=1,{TRIALS}
  do k=1,4
    do j=0,209
      do i=1,4
        scale=1.0_{kind}
        if(trial==3.and.mod(j,3)==0) scale=1.e100_{kind}
        if(trial==3.and.mod(j,3)==1) scale=1.e-100_{kind}
        a(i,j,k)=cmplx(real(mod(17*i+31*j+13*k,43)-21,kind={kind})/7, &
          real(mod(37*i+11*j+7*k,31)-15,kind={kind})/13,kind={kind})*scale
        if(trial==2.and.mod(i+j+k,3)/=0) a(i,j,k)=0
        if(trial==4) a(i,j,k)=0
        if(trial==5) a(i,j,k)=cmplx((-1.0_{kind})**i, &
                                      (-1.0_{kind})**(i+1),kind={kind})
        if(trial==6.and.mod(i+j+k,2)==0) a(i,j,k)=cmplx(-0.0_{kind},0.0_{kind},kind={kind})
      enddo
    enddo
    do j=0,14
      do i=1,4
        b(i,j,k)=cmplx(real(mod(19*i+29*j+3*k,17)-8,kind={kind})/3, &
          real(mod(11*i+17*j+13*k,23)-11,kind={kind})/11,kind={kind})
        if(trial==2.and.mod(i+j+k,3)/=0) b(i,j,k)=0
        if(trial==4) b(i,j,k)=0
        if(trial==5) b(i,j,k)=cmplx(1.0_{kind},-1.0_{kind},kind={kind})
        if(trial==6.and.mod(i+j+k,2)==1) b(i,j,k)=cmplx(0.0_{kind},-0.0_{kind},kind={kind})
      enddo
    enddo
  enddo
  saved_a=a
  saved_b=b
  do ls=0,4
    do ins=0,4
      do outs=0,4
{''.join(calls)}
      enddo
    enddo
  enddo
enddo
if(checks/={expected_checks}) stop 2
write(*,*) 'PASS', {kind},checks
end program
""")
    binary = work / f"check-{kind}"
    command = ["gfortran", *FLAGS, "constants.f", old_path.name,
               new_path.name, driver.name, "-o", str(binary)]
    compiled = subprocess.run(command, cwd=work, capture_output=True, text=True)
    if compiled.returncode:
        raise RuntimeError(compiled.stdout + compiled.stderr)
    checked = subprocess.run([str(binary)], cwd=work, capture_output=True, text=True)
    if checked.returncode or "PASS" not in checked.stdout:
        raise RuntimeError(checked.stdout + checked.stderr)
    print(checked.stdout.strip(), flush=True)
    return {"kind": kind, "comparisons": expected_checks,
            "transformed_routines": count, "passed": True,
            "original_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
            "transformed_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest(),
            "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest(),
            "stdout": checked.stdout.strip()}


def main():
    with tempfile.TemporaryDirectory(prefix="expanded-wl-exact-") as tmp:
        work = Path(tmp)
        (work / "coef_specs.inc").write_text(
            "      INTEGER MAXLWFSIZE,VERTEXMAXCOEFS\n"
            "      PARAMETER(MAXLWFSIZE=4,VERTEXMAXCOEFS=15)\n")
        (work / "loop_max_coefs.inc").write_text(
            "      INTEGER LOOPMAXCOEFS\n"
            "      PARAMETER(LOOPMAXCOEFS=210)\n")
        generator = FortranPolynomialRoutines(
            6, updater_max_rank=2, proc_prefix="TEST_", sub_prefix="TEST_")
        with FortranWriter(str(work / "constants.f")) as writer:
            writer.writelines(generator.write_polynomial_constant_module())
        result = {
            "scope": "Isolated expanded UPDATE_WL transformation; no production source edit",
            "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
            "flags": FLAGS, "comparison": "Full output array bit patterns, including untouched sentinels",
            "original_generator_revision": BASE_REVISION,
            "original_generator_sha256": hashlib.sha256(BASE_SOURCE.encode()).hexdigest(),
            "left_ranks": list(range(5)), "right_ranks": list(range(3)),
            "wavefunction_dimensions": list(range(5)),
            "inputs": ["dense", "sparse", "1e-100 to 1e100", "all zero",
                       "alternating cancellation", "signed zero"],
            "transform_sha256": hashlib.sha256((HERE / "run.py").read_bytes()).hexdigest(),
            "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "precisions": [precision(work, kind) for kind in (8, 16)],
        }
    (HERE / "exact-regression.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
