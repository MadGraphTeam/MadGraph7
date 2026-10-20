#!/usr/bin/env python3
"""Compile copied generated MP helpers and isolate lost KIND and pi defects."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/ew3-collier-stability/mp-helper-audit"
SOURCE = ROOT / ".benchmark-work/outputs/candidate/ew-3/Source/MODEL/model_functions.f"


def main():
    os.sched_setaffinity(0, {2})
    WORK.mkdir(parents=True, exist_ok=False)
    text = SOURCE.read_text()
    names = ["MP_REGLOG", "MP_REGLOGP", "MP_REGLOGM", "MP_REGSQRT", "MP_GRREGLOG"]
    routines = []
    for name in names:
        match = re.search(rf"(?m)^ +COMPLEX\*32 FUNCTION {name}\(.*?^ +END\s*$",
                          text, flags=re.S | re.M)
        assert match, name
        routines.append(match[0])
    original = "\n".join(routines) + "\n"
    fixed = original
    replacements = 0
    for variable in ("ARG", "EXPR1", "EXPR2"):
        for old, new in (
            (f"CMPLX(REAL({variable},KIND=16),0.0E0_16)",
             f"CMPLX(REAL({variable},KIND=16),0.0E0_16,KIND=16)"),
            (f"CMPLX(0.0E0_16,IMAGPART({variable}))",
             f"CMPLX(0.0E0_16,IMAGPART({variable}),KIND=16)"),
        ):
            replacements += fixed.count(old)
            fixed = fixed.replace(old, new)
    assert replacements == 12
    fixed = fixed.replace("3.14169258478796109557151794433593750",
                          "3.14159265358979323846264338327950288")
    for name in names:
        fixed = re.sub(rf"\b{name}\b", "FIX_" + name, fixed)
    (WORK / "original.f").write_text(original)
    (WORK / "fixed.f").write_text(fixed)
    (WORK / "probe.f90").write_text("""
program helper_precision
implicit none
complex(16),external :: MP_REGLOG,MP_REGLOGP,MP_REGLOGM,MP_REGSQRT,MP_GRREGLOG
complex(16),external :: FIX_MP_REGLOG,FIX_MP_REGLOGP,FIX_MP_REGLOGM,FIX_MP_REGSQRT,FIX_MP_GRREGLOG
complex(16) :: z,reference,old,new
real(16) :: pi,x
integer i
x=1.234567890123456789012345678901_16
pi=acos(-1.0_16)
do i=1,2
  z=cmplx(x,0.0_16,kind=16)
  if(i==2) z=cmplx(0.0_16,x,kind=16)
  reference=log(z)
  old=MP_REGLOG(z)
  new=FIX_MP_REGLOG(z)
  write(*,'(A,I1,2ES48.36E4)') 'REGLOG_',i,abs(old-reference),abs(new-reference)
  if(abs(new-reference)>1.e-31_16) stop 1
  reference=sqrt(z)
  old=MP_REGSQRT(z)
  new=FIX_MP_REGSQRT(z)
  write(*,'(A,I1,2ES48.36E4)') 'REGSQRT_',i,abs(old-reference),abs(new-reference)
  if(abs(new-reference)>1.e-31_16) stop 2
enddo
z=cmplx(-1.0_16,-1.0_16,kind=16)
reference=log(z)+cmplx(0.0_16,2*pi,kind=16)
old=MP_REGLOGP(z)
new=FIX_MP_REGLOGP(z)
write(*,'(A,2ES48.36E4)') 'REGLOGP_SHEET',abs(old-reference),abs(new-reference)
if(abs(new-reference)>1.e-31_16) stop 3
z=cmplx(-1.0_16,1.0_16,kind=16)
reference=log(z)-cmplx(0.0_16,2*pi,kind=16)
old=MP_REGLOGM(z)
new=FIX_MP_REGLOGM(z)
write(*,'(A,2ES48.36E4)') 'REGLOGM_SHEET',abs(old-reference),abs(new-reference)
if(abs(new-reference)>1.e-31_16) stop 4
old=MP_GRREGLOG(1.0_16,z,conjg(z))
new=FIX_MP_GRREGLOG(1.0_16,z,conjg(z))
write(*,'(A,2ES48.36E4)') 'GRREGLOG_SHEET',abs(old-reference),abs(new-reference)
if(abs(new-reference)>1.e-31_16) stop 5
end program
""")
    command = ["gfortran", "-O3", "-ffixed-line-length-132", "original.f", "fixed.f",
               "probe.f90", "-o", "probe"]
    compiled = subprocess.run(command, cwd=WORK, capture_output=True, text=True)
    (HERE / "build.log").write_text(compiled.stdout+compiled.stderr)
    compiled.check_returncode()
    run = subprocess.run(["./probe"], cwd=WORK, capture_output=True, text=True)
    (HERE / "values.dat").write_text(run.stdout)
    (HERE / "run.log").write_text(run.stderr)
    result = {
        "scope": "Copied generated helper functions only; no production/model change",
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "command": command, "kind_fixes": replacements,
        "passed": run.returncode == 0,
        "tests": [{"name": fields[0], "old_absolute_error": fields[1],
                   "fixed_absolute_error": fields[2]}
                  for fields in (line.split() for line in run.stdout.splitlines())],
        "files_sha256": {},
        "origins": ["madgraph/iolibs/export_v4.py:13049",
                    "madgraph/iolibs/export_v4.py:13052",
                    "madgraph/iolibs/export_v4.py:13044"],
    }
    for name in ("original.f", "fixed.f", "probe.f90"):
        data = (WORK / name).read_bytes()
        (HERE / name).write_bytes(data)
        result["files_sha256"][name] = hashlib.sha256(data).hexdigest()
    (HERE / "result.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result["tests"], indent=2))
    run.check_returncode()


if __name__ == "__main__":
    main()
