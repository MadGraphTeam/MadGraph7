#!/usr/bin/env python3
"""Exercise enabled IEEE trap fallback using the actual instrumented prototype."""
import json
import os
import subprocess

import check_shared as fixture

DRIVER = """
program trap_control
use, intrinsic :: ieee_exceptions
implicit none
complex(16) a(4,0:209,4),b(4,0:14,4),out(4,0:209,4)
integer flag,checks,fast,fallback
common/SHARED_TRACE/fast,fallback
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
 call TEST_UPDATE_WL_2_1_SHARED(a,4,b,4,4,out)
 call ieee_set_halting_mode(ieee_all,.false.)
 call ieee_get_flag(ieee_all,raised)
 if(fast/=0.or.fallback/=1.or.any(raised).or.any(out/=0)) stop 1
 checks=checks+1
enddo
if(checks==0) stop 2
write(*,*) 'PASS',checks
end program
"""


def main():
    os.sched_setaffinity(0,{2})
    work=fixture.WORK/'rank2'
    (work/'traps.f90').write_text(DRIVER)
    command=['gfortran','-O3','-frounding-math','-fno-automatic',
        '-ffixed-line-length-132','constants.f','full.f','checked-shared.f',
        'traps.f90','-o','traps']
    built=subprocess.run(command,cwd=work,capture_output=True,text=True)
    (work/'traps-build.log').write_text(built.stdout+built.stderr)
    built.check_returncode()
    ran=subprocess.run([str(work/'traps')],cwd=work,capture_output=True,text=True)
    (work/'traps.log').write_text(ran.stdout+ran.stderr)
    result=dict(passed=ran.returncode==0,command=command,output=ran.stdout+ran.stderr,
        executable_sha256=fixture.check.sha(work/'traps'),
        source_sha256=fixture.check.sha(work/'traps.f90'),
        runner_sha256=fixture.check.sha(__file__))
    (fixture.HERE/'traps.json').write_text(json.dumps(result,indent=2)+'\n')
    ran.check_returncode()
    print(ran.stdout)


if __name__=='__main__':
    main()
