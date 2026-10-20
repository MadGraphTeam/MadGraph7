#!/usr/bin/env python3
"""True-entry scanner must not raise on sNaN A which public B-zero skips."""
import sys
sys.dont_write_bytecode=True
import argparse
import json
import os
from pathlib import Path

import check_safety
import support

HERE=Path(__file__).resolve().parent


def driver(rank,vertex):
    prefix,initialization,helper=check_safety.pieces(rank,vertex)
    return prefix+f"""
do test=1,2
 trial=1
{initialization}
 B(:,0:4,:)=cmplx(0.0_8,0.0_8,8)
 if(test==1) then
   A(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),A(1,0,1))
 else
   A(1,0,1)=transfer((/0_8,int(z'7FF0000000000001',8)/),A(1,0,1))
 endif
 ins=4
 outs=4
 entry=.true.
 call ieee_set_flag(ieee_all,.false.)
 call ieee_set_flag(ieee_inexact,.true.)
 call ieee_get_flag(ieee_all,initialflags)
{check_safety.comparison(rank,'if(fast/=0.or.fallback/=1) stop 12')}
 if(any(oldflags.neqv.initialflags)) stop 19
enddo
contains
{helper}
end program
"""


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--cpu',type=int,default=2)
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    report=HERE/args.run_id
    report.mkdir(exist_ok=False)
    for name in ('check_zero_work_snan.py','check_safety.py','support.py','factor.py'):
        (report/name).write_bytes((HERE/name).read_bytes())
    result=dict(complete=False,passed=False,results=[],runner_sha256=support.sha(__file__),
                scope='Original B-zero skips every product; true-entry bounded scanner must preserve sNaN bits and must not add invalid.')
    def retain():
        (report/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    retain()
    try:
      for flags in support.FLAGS:
       for vertex in ('FFV1L1_2','FFV1L2_1'):
        for rank in (2,3,4):
          work=report/f'{flags}-{vertex}-rank{rank}'
          sources=support.write_sources(work,rank,vertex)
          (work/'zero-work.f90').write_text(driver(rank,vertex))
          build=support.run(['gfortran',*support.FLAGS[flags],*sources,'zero-work.f90','-o','zero-work'],work,'build.log')
          run=support.run([str(work/'zero-work')],work,'run.log')
          assert [row.split() for row in run['output'].splitlines()]==[['PASS','1','0','1'],['PASS','2','0','1']]
          result['results'].append(dict(key=work.name,build=build,run=run,
              sources={p.name:support.sha(p) for p in work.iterdir() if p.suffix in ('.f','.f90','.inc')}))
          retain()
      result.update(complete=True,passed=True)
    except Exception as error:
      result['failure']=repr(error)
      raise
    finally:
      retain()
