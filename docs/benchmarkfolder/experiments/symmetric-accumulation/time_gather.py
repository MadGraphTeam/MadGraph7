#!/usr/bin/env python3
"""Exploratory synthetic timings; deliberately not an acceptance benchmark."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

import check_gather as check

LOOP_COUNTS = {8:20000,16:1000}

DRIVER = """
program measure
use TEST_POLYNOMIAL_CONSTANTS
implicit none
complex(@KIND@) a(4,0:209,4),b(4,0:14,4),out(4,0:209,4)
integer i,j,k,pair,order,which,it,density
real(8) t0,t1,times(0:1)
do density=0,1
do k=1,4
do j=0,209
do i=1,4
 a(i,j,k)=cmplx(real(mod(i+3*j+7*k,13),kind=@KIND@)/7,&
              real(mod(11*i+j+k,17),kind=@KIND@)/11,kind=@KIND@)
 if(density==1.and.mod(i+j+k,3)/=0) a(i,j,k)=0
enddo
enddo
do j=0,14
do i=1,4
 b(i,j,k)=cmplx(real(i+j+k,kind=@KIND@)/3,real(2*i+j-k,kind=@KIND@)/7,kind=@KIND@)
enddo
enddo
enddo
@ZERO@
do pair=1,7
 do order=0,1
  which=mod(order+pair,2)
  call cpu_time(t0)
  if(which==0) then
   do it=1,@LOOPS@
    call TEST_UPDATE_WL_@RANK@_1(a,4,b,4,4,out)
   enddo
  else
   do it=1,@LOOPS@
    call TEST_UPDATE_WL_@RANK@_1_GATHER(a,4,b,4,4,out)
   enddo
  endif
  call cpu_time(t1)
  times(which)=t1-t0
 enddo
 write(*,*) density,pair,times,real(sum(out),kind=8)
enddo
enddo
end program
"""


def main():
    os.sched_setaffinity(0,{2})
    inventory=json.loads((check.HERE/'inventory.json').read_text())['qcd-3']
    shapes=inventory['routines']['MP_FFV1L2_1']['shapes']
    zero=next(s['literal_zero_entries'] for s in shapes if s['in_size']==4 and s['out_size']==4)
    results=[]
    for kind in (8,16):
        for rank in (2,3):
            work=check.WORK/f'kind{kind}-rank{rank}'
            loops=LOOP_COUNTS[kind]
            driver=(DRIVER.replace('@KIND@',str(kind)).replace('@RANK@',str(rank))
                    .replace('@LOOPS@',str(loops)).replace('@ZERO@','\n'.join(
                        f'b({j},{m},{k})=0' for j,m,k in zero)))
            (work/'timing.f90').write_text(driver)
            command=['gfortran','-O3','-frounding-math','-fno-automatic',
                '-ffixed-line-length-132','constants.f','full.f','gather.f',
                'timing.f90','-o','timing']
            built=subprocess.run(command,cwd=work,capture_output=True,text=True)
            (work/'timing-build.log').write_text(built.stdout+built.stderr)
            built.check_returncode()
            run=subprocess.run([str(work/'timing')],cwd=work,capture_output=True,text=True)
            (work/'timing.log').write_text(run.stdout+run.stderr)
            run.check_returncode()
            rows=[line.split() for line in run.stdout.splitlines()]
            assert len(rows)==14
            for density in (0,1):
                pairs=[(float(row[2]),float(row[3])) for row in rows if int(row[0])==density]
                assert len(pairs)==7
                ratio=statistics.median(t[1] for t in pairs)/statistics.median(t[0] for t in pairs)
                results.append(dict(kind=kind,rank=rank,density=density,loops=loops,
                    pairs_seconds=pairs,ratio_of_medians=ratio,
                    source_sha256=check.sha(work/'gather.f'),executable_sha256=check.sha(work/'timing')))
                print(kind,rank,density,ratio,flush=True)
    (check.HERE/'gather-timing.json').write_text(json.dumps(dict(
        scope='Synthetic kernel CPU time on CPU2 under concurrent work; not full-process '
              'or acceptance timing. Fixed source-proved FFV1L2_1 zero mask; synthetic complex '
              'values, dense A or two-thirds zero A. Seven alternating paired traversals.',
        acceptance_eligible=False,results=results,driver_sha256=check.sha(__file__)),indent=2)+'\n')


if __name__=='__main__':
    main()
