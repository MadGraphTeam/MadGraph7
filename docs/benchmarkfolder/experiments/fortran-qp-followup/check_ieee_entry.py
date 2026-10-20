#!/usr/bin/env python3
"""Check actual GNU Fortran sticky-flag/trap behavior at new procedure boundaries."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
WORK = ROOT/".benchmark-work/experiments/qp-terminal-ieee-entry"
SOURCE = """
program probe
use, intrinsic :: ieee_exceptions
implicit none
logical :: before, batch_value, helper_value, after, flags(4)
call ieee_set_halting_mode(ieee_inexact,.false.)
call ieee_set_flag(ieee_all,.false.)
call ieee_set_halting_mode(ieee_inexact,.true.)
call ieee_get_halting_mode(ieee_inexact,before)
call batch(batch_value,helper_value)
call ieee_get_halting_mode(ieee_inexact,after)
call ieee_set_halting_mode(ieee_inexact,.false.)
if (.not.(before.and.batch_value.and.helper_value.and.after)) stop 1
write(*,*) 'HALTING_PRESERVED',before,batch_value,helper_value,after
call ieee_set_flag(ieee_inexact,.true.)
call ieee_get_flag(ieee_inexact,flags(1))
call batch_flags(flags(2),flags(3))
call ieee_get_flag(ieee_inexact,flags(4))
if (.not.flags(1).or..not.flags(4)) stop 2
write(*,*) 'STICKY_FLAGS',flags
end program
subroutine batch(b,h)
use, intrinsic :: ieee_exceptions
implicit none
logical :: b,h
call ieee_get_halting_mode(ieee_inexact,b)
call helper(h)
end subroutine
subroutine helper(h)
use, intrinsic :: ieee_exceptions
implicit none
logical :: h
call ieee_get_halting_mode(ieee_inexact,h)
end subroutine
subroutine batch_flags(b,h)
use, intrinsic :: ieee_exceptions
implicit none
logical :: b,h
call ieee_get_flag(ieee_inexact,b)
call helper_flags(h)
end subroutine
subroutine helper_flags(h)
use, intrinsic :: ieee_exceptions
implicit none
logical :: h
call ieee_get_flag(ieee_inexact,h)
end subroutine
"""

if __name__=="__main__":
    WORK.mkdir(parents=True,exist_ok=False)
    (WORK/"probe.f90").write_text(SOURCE)
    subprocess.run(["gfortran","-O3","probe.f90","-o","check"],cwd=WORK,check=True)
    run=subprocess.run([str(WORK/"check")],cwd=WORK,text=True,
                       stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    library=subprocess.check_output(["gfortran","-print-file-name=libgfortran.so"],
                                    cwd=ROOT,text=True).strip()
    assembly=subprocess.check_output(["objdump","-d",
        "--disassemble=_gfortran_ieee_procedure_entry",library],cwd=ROOT,text=True)
    (HERE/"ieee-entry-assembly.txt").write_text(assembly)
    result={"passed":run.returncode==0,"returncode":run.returncode,"stdout":run.stdout,
            "source_sha256":hashlib.sha256(SOURCE.encode()).hexdigest(),
            "executable_sha256":hashlib.sha256((WORK/"check").read_bytes()).hexdigest(),
            "gfortran_runtime":library,
            "interpretation":"Nested IEEE procedures preserve enabled INEXACT halting. "
                             "They clear sticky flags internally and restore/merge them on return."}
    (HERE/"ieee-entry.json").write_text(json.dumps(result,indent=2)+"\n")
    (HERE/"ieee-entry.f90").write_text(SOURCE)
    print(json.dumps(result,indent=2))
    if run.returncode:
        raise SystemExit(1)
