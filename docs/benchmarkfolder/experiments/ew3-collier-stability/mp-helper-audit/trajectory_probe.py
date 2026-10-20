#!/usr/bin/env python3
"""Isolate trajectory constructor rounding at the equal-mass threshold."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent / "trajectory"
WORK = ROOT / ".benchmark-work/experiments/ew3-collier-stability/mp-helper-trajectory"
SOURCE = ROOT / ".benchmark-work/outputs/candidate/ew-3/Source/MODEL/model_functions.f"


def main():
    os.sched_setaffinity(0, {2})
    HERE.mkdir(exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=False)
    text = SOURCE.read_text()
    names = ["MP_SQRT_TRAJECTORY", "MP_LOG_TRAJECTORY"]
    routines = []
    for name in names:
        match = re.search(rf"(?m)^ +COMPLEX\*32 FUNCTION {name}\(.*?^ +END\s*$",
                          text, flags=re.S | re.M)
        assert match, name
        routines.append(match[0])
    original = "\n".join(routines) + "\n"
    fixed = original.replace("CMPLX(M**2,-GAI*M)",
                             "CMPLX(M**2,-GAI*M,KIND=16)")
    assert original.count("CMPLX(M**2,-GAI*M)") == 4
    # Deliberately leave pi and all other arithmetic identical, isolating
    # only the four variable-argument trajectory constructors.
    for name in names:
        fixed = re.sub(rf"\b{name}\b", "FIX_" + name, fixed)
    (WORK / "original.f").write_text(original)
    (WORK / "fixed.f").write_text(fixed)
    (WORK / "probe.f90").write_text("""
program trajectory_precision
implicit none
complex(16), external :: MP_SQRT_TRAJECTORY, FIX_MP_SQRT_TRAJECTORY
complex(16), external :: MP_LOG_TRAJECTORY, FIX_MP_LOG_TRAJECTORY
complex(16) :: p2, m2, reference, old, new, discriminant
real(16) :: s, x
integer i,n
m2=cmplx(1.0_16,0.0_16,kind=16)
do i=-1,1,2
  s=4.0_16+i*2.0_16**(-30)
  p2=cmplx(s,-1.0_16,kind=16)
  discriminant=1.0_16-4.0_16*(m2-cmplx(0.0_16,1.e-24_16,kind=16))/p2
  ! Below threshold, continuation crosses the negative real discriminant
  ! axis from above to below: the continued root is minus principal sqrt.
  reference=sqrt(discriminant)
  if(i<0) reference=-reference
  do n=100,400,100
    old=MP_SQRT_TRAJECTORY(n,p2,m2,m2)
    new=FIX_MP_SQRT_TRAJECTORY(n,p2,m2,m2)
    write(*,'(A,2I5,6ES48.36E4)') 'SQRT',i,n,real(old),aimag(old), &
      real(new),aimag(new),abs(old-reference),abs(new-reference)
    if(abs(new-reference)>1.e-31_16) stop 1
    old=MP_LOG_TRAJECTORY(n,p2,m2,m2)
    new=FIX_MP_LOG_TRAJECTORY(n,p2,m2,m2)
    write(*,'(A,2I5,5ES48.36E4)') 'LOG',i,n,real(old),aimag(old), &
      real(new),aimag(new),abs(old-new)
  enddo
enddo
x=1.0_16+2.0_16**(-100)
write(*,'(A,2ES48.36E4)') 'REAL_CAST',real(real(x),16)-x,real(x,16)-x
if(real(real(x),16)==x) stop 2
end program
""")
    command = ["gfortran", "-O3", "-ffixed-line-length-132", "original.f",
               "fixed.f", "probe.f90", "-o", "probe"]
    build = subprocess.run(command, cwd=WORK, capture_output=True, text=True)
    (HERE / "build.log").write_text(build.stdout + build.stderr)
    build.check_returncode()
    run = subprocess.run(["./probe"], cwd=WORK, capture_output=True, text=True)
    (HERE / "values.dat").write_text(run.stdout)
    (HERE / "run.log").write_text(run.stderr)
    result = {
        "scope": "Copied generated routines; only four variable-argument CMPLX constructors changed.",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "command": command, "passed": run.returncode == 0,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "files_sha256": {},
        "cases": [],
        "analytic_reference": (
            "m12=m22=1, p2=4 +/- 2^-30 - i. Discriminant "
            "1-4*(1-i*1e-24)/p2; continuation is negative principal square "
            "root below threshold and principal square root above threshold."
        ),
        "log_scope": "Old/new diagnostic only; no independent log-trajectory oracle asserted.",
    }
    for line in run.stdout.splitlines():
        fields = line.split()
        if fields[0] == "SQRT":
            result["cases"].append(dict(zip(
                ["side", "segments", "old_real", "old_imag", "fixed_real",
                 "fixed_imag", "old_error", "fixed_error"], fields[1:])))
    for name in ("original.f", "fixed.f", "probe.f90", "values.dat"):
        data = (WORK / name if name != "values.dat" else HERE / name).read_bytes()
        if name != "values.dat":
            (HERE / name).write_bytes(data)
        result["files_sha256"][name] = hashlib.sha256(data).hexdigest()
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(run.stdout)
    run.check_returncode()


if __name__ == "__main__":
    main()
