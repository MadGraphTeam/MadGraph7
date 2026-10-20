#!/usr/bin/env python3
"""DP-specific guard and IEEE procedure-boundary controls; isolated only."""
import sys
sys.dont_write_bytecode = True
import argparse
import json
import os
from pathlib import Path
import resource
import subprocess

import support
from oracle import driver as dump_driver

HERE = Path(__file__).resolve().parent


def pieces(rank, vertex):
    source = dump_driver.driver(rank, vertex)
    prefix, rest = source.split("do trial=1,10\n", 1)
    initialization = rest.split("  entry=trial/=8\n", 1)[0]
    prefix = prefix.replace("integer :: trial,i,j,k,l,m,n,fast,fallback,diag",
        "integer :: trial,i,j,k,l,m,n,fast,fallback,diag,test,ins,outs,mask,flag\n"
        "integer*8 oldbits(2*4*MAXC*4),newbits(2*4*MAXC*4)\n"
        "logical initialflags(size(ieee_all)),halting(size(ieee_all))\n"
        "type(ieee_round_type) observed_rounding,initial_rounding")
    prefix = prefix[:prefix.index("write(*,'(A)') 'FORMAT")]
    helper = source.split("\ncontains\n",1)[1].split("\nend program",1)[0]
    return prefix, initialization, helper


def comparison(rank, expect):
    old = f"TEST_UPDATE_WL_{rank}_1"
    return f"""
  saved_a=transfer(A,saved_a)
  saved_b=transfer(B,saved_b)
  call ieee_get_rounding_mode(initial_rounding)
  call ieee_set_flag(ieee_all,initialflags)
  call {old}(A,4,B,ins,outs,oldout)
  call ieee_get_flag(ieee_all,oldflags)
  if(any(saved_a/=transfer(A,saved_a)).or.any(saved_b/=transfer(B,saved_b))) stop 13
  call ieee_get_rounding_mode(observed_rounding)
  if(observed_rounding/=initial_rounding) stop 14
  fast=0
  fallback=0
  call ieee_set_flag(ieee_all,initialflags)
  call {old}_FACTOR(A,4,B,ins,outs,newout,entry)
  call ieee_get_flag(ieee_all,newflags)
  if(any(oldflags.neqv.newflags)) then
    write(*,*) 'FLAGS',test,oldflags,newflags
    stop 11
  endif
  if(any(initialflags.and..not.newflags)) stop 15
  call ieee_get_rounding_mode(observed_rounding)
  if(observed_rounding/=initial_rounding) stop 16
  call ieee_get_halting_mode(ieee_all,halting)
  if(any(halting)) stop 17
  if(any(saved_a/=transfer(A,saved_a)).or.any(saved_b/=transfer(B,saved_b))) stop 18
  {expect}
  if(fallback==1) then
    oldbits=transfer(oldout,oldbits)
    newbits=transfer(newout,newbits)
    if(any(oldbits/=newbits)) then
      write(*,*) 'BITS',test
      stop 10
    endif
  endif
  write(*,*) 'PASS',test,fast,fallback
"""


def guards(rank, vertex):
    prefix, initialization, helper = pieces(rank, vertex)
    return prefix+f"""
do test=1,20
  trial=1
  call ieee_set_rounding_mode(ieee_nearest)
{initialization}
  ins=4
  outs=4
  entry=.true.
  select case(test)
  case(1)
    call ieee_set_rounding_mode(ieee_up)
  case(2)
    call ieee_set_rounding_mode(ieee_down)
  case(3)
    call ieee_set_rounding_mode(ieee_to_zero)
  case(4)
    A(1,0,1)=cmplx(scale(1.0_8,201),0.0_8,8)
  case(5)
    A(1,0,1)=cmplx(scale(1.0_8,-201),0.0_8,8)
  case(6)
    A(1,0,1)=cmplx(transfer(1_8,0.0_8),0.0_8,8)
  case(7)
    A(1,0,1)=cmplx(ieee_value(0.0_8,ieee_positive_inf),0.0_8,8)
  case(8)
    A(1,0,1)=cmplx(ieee_value(0.0_8,ieee_quiet_nan),0.0_8,8)
  case(9)
    A(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),A(1,0,1))
  case(10)
    B(1,0,1)=cmplx(scale(1.0_8,201),0.0_8,8)
  case(11)
    B(1,0,1)=cmplx(transfer(1_8,0.0_8),0.0_8,8)
  case(12)
    B(1,0,1)=cmplx(ieee_value(0.0_8,ieee_positive_inf),0.0_8,8)
  case(13)
    B(1,0,1)=cmplx(ieee_value(0.0_8,ieee_quiet_nan),0.0_8,8)
  case(14)
    B(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),B(1,0,1))
  case(15)
    B(1,2,1)=B(1,2,1)+cmplx(1.0_8,0.0_8,8)
  case(16)
    ins=3
  case(17)
    outs=3
  case(18)
    ins=0
  case(19)
    outs=0
  case(20)
    B(1,0,1)=cmplx(scale(1.0_8,-201),0.0_8,8)
  end select
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,.true.)
  call ieee_get_flag(ieee_all,initialflags)
{comparison(rank, 'if(fast/=0.or.fallback/=1) stop 12')}
enddo
call ieee_set_rounding_mode(ieee_nearest)
contains
{helper}
end program
"""


def sticky(rank, vertex):
    prefix, initialization, helper = pieces(rank, vertex)
    return prefix+f"""
do mask=0,2**size(ieee_all)-1
  trial=1
  test=mask+1
{initialization}
  ins=4
  outs=4
  do flag=1,size(ieee_all)
    initialflags(flag)=btest(mask,flag-1)
  enddo
  call ieee_set_flag(ieee_all,initialflags)
  call ieee_get_flag(ieee_inexact,entry)
{comparison(rank, 'if(fast/=merge(1,0,entry).or.fallback/=merge(0,1,entry)) stop 12')}
enddo
contains
{helper}
end program
"""


def traps(rank):
    old = f"TEST_UPDATE_WL_{rank}_1"
    return f"""program boundary_traps
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
use, intrinsic :: iso_fortran_env,only:output_unit
implicit none
complex*16 a(4,0:209,4),b(4,0:14,4),out(4,0:209,4)
integer flag,backend,scenario,fast,fallback,i,initial_mask
common/FACTOR_TRACE/fast,fallback
logical masks(size(ieee_all)),observed(size(ieee_all)),raised(size(ieee_all)),initial_flags(size(ieee_all))
character(32) arg
call get_command_argument(1,arg)
read(arg,*) backend
call get_command_argument(2,arg)
read(arg,*) scenario
call get_command_argument(3,arg)
read(arg,*) flag
call get_command_argument(4,arg)
read(arg,*) initial_mask
do i=1,size(ieee_all)
 if(.not.ieee_support_halting(ieee_all(i))) stop 1
enddo
call ieee_set_halting_mode(ieee_all,.false.)
call ieee_set_rounding_mode(ieee_nearest)
a=0
b=0
select case(scenario)
case(0)
 a(1,0,1)=cmplx(2.0_8,0.0_8,8)
 b(1,0,1)=cmplx(4.0_8,0.0_8,8)
case(1)
 a(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),a(1,0,1))
 b(1,0,1)=1
case(2)
 a(1,0,1)=1
 b(1,0,1)=transfer((/int(z'7FF0000000000001',8),0_8/),b(1,0,1))
case(3)
 a(1,0,1)=cmplx(huge(1.0_8),0.0_8,8)
 b(1,0,1)=4
case(4)
 a(1,0,1)=cmplx(transfer(1_8,0.0_8),0.0_8,8)
 b(1,0,1)=cmplx(0.5_8,0.0_8,8)
case(5)
 a(1,0,1)=cmplx(1.0_8+epsilon(1.0_8),0.0_8,8)
 b(1,0,1)=a(1,0,1)
case default
 stop 2
end select
do i=1,size(ieee_all)
 initial_flags(i)=btest(initial_mask,i-1)
enddo
call ieee_set_flag(ieee_all,initial_flags)
masks=.false.
select case(scenario)
case(0)
 masks(flag)=.true.
case(1,2)
 call ieee_set_halting_mode(ieee_invalid,.true.)
case(3)
 call ieee_set_halting_mode(ieee_overflow,.true.)
case(4)
 call ieee_set_halting_mode(ieee_underflow,.true.)
case(5)
 call ieee_set_halting_mode(ieee_inexact,.true.)
end select
if(scenario==0) call ieee_set_halting_mode(ieee_all,masks)
call ieee_get_halting_mode(ieee_all,masks)
fast=0
fallback=0
write(output_unit,*) 'ENTER',backend,scenario,flag
flush(output_unit)
if(backend==0) then
 call {old}(a,4,b,4,4,out)
else
 call {old}_FACTOR(a,4,b,4,4,out,.true.)
endif
call ieee_get_halting_mode(ieee_all,observed)
call ieee_get_flag(ieee_all,raised)
call ieee_set_halting_mode(ieee_all,.false.)
if(any(masks.neqv.observed)) stop 3
if(scenario/=0) stop 4
if(any(raised.neqv.initial_flags)) stop 5
if(out(1,0,1)/=cmplx(8.0_8,0.0_8,8)) stop 6
if(backend==1.and.(fast/=0.or.fallback/=1)) stop 7
write(*,*) 'PASS',backend,scenario,flag
end program
"""


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--cpu',type=int,default=2)
    parser.add_argument('--ranks',type=int,nargs='+',default=[2,3,4])
    parser.add_argument('--vertices',nargs='+',default=['FFV1L1_2','FFV1L2_1'])
    parser.add_argument('--flags',nargs='+',choices=support.FLAGS,default=list(support.FLAGS))
    parser.add_argument('--sticky-traps-only',action='store_true')
    args=parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    report=HERE/args.run_id
    report.mkdir(exist_ok=False)
    for file in ('check_safety.py','factor.py','support.py','oracle/driver.py','oracle/source-parity.json'):
        (report/Path(file).name).write_bytes((HERE/file).read_bytes())
    result=dict(complete=False,passed=False,cpu=args.cpu,acceptance_eligible=False,
                runner_sha256=support.sha(__file__),factor_sha256=support.sha(HERE/'factor.py'),
                compiler=support.run(['gfortran','--version'],report,'compiler.log'),
                sticky_traps_only=args.sticky_traps_only,results=[])
    def retain():
        (report/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    retain()
    try:
      for flags in args.flags:
       for vertex in args.vertices:
        for rank in args.ranks:
          key=f'{flags}-{vertex}-rank{rank}-full'
          work=report/key
          sources=support.write_sources(work,rank,vertex)
          records={}
          programs=[('traps',traps(rank))] if args.sticky_traps_only else [('guards',guards(rank,vertex)),('sticky',sticky(rank,vertex)),('traps',traps(rank))]
          for name,source in programs:
            (work/(name+'.f90')).write_text(source)
            records[name+'_build']=support.run(['gfortran',*support.FLAGS[flags],*sources,name+'.f90','-o',name],work,name+'-build.log')
            if name!='traps':
              records[name]=support.run([str(work/name)],work,name+'.log')
              rows=[line.split() for line in records[name]['output'].splitlines()]
              count=20 if name=='guards' else 32
              assert len(rows)==count and [int(row[1]) for row in rows]==list(range(1,count+1))
              assert all(len(row)==4 and row[0]=='PASS' for row in rows)
          records['trap_subprocesses']=[]
          trap_cases=[(0,i,mask) for i in range(1,6) for mask in range(32) if not mask & (1<<(i-1))] if args.sticky_traps_only else [(0,i,0) for i in range(1,6)]+[(i,1,0) for i in range(1,6)]
          for scenario,flag,initial_mask in trap_cases:
            paired=[]
            for backend in (0,1):
              command=[str(work/'traps'),str(backend),str(scenario),str(flag),str(initial_mask)]
              ran=subprocess.run(command,cwd=work,capture_output=True,text=True)
              name=f'trap-{backend}-{scenario}-{flag}-{initial_mask}'
              (work/(name+'.stdout')).write_text(ran.stdout)
              (work/(name+'.stderr')).write_text(ran.stderr)
              record=dict(command=command,backend=backend,scenario=scenario,flag=flag,initial_mask=initial_mask,returncode=ran.returncode)
              records['trap_subprocesses'].append(record)
              paired.append(ran.returncode)
              assert ['ENTER',str(backend),str(scenario),str(flag)]==ran.stdout.splitlines()[0].split()
              assert ran.returncode==(0 if scenario==0 else -8), record
            assert paired[0]==paired[1]
          result['results'].append(dict(key=key,tests=records,sources={p.name:support.sha(p) for p in work.iterdir() if p.suffix in ('.f','.f90','.inc')},executables={n:support.sha(work/n) for n,_ in programs}))
          retain()
          print(key,('80 compatible sticky/trap mask pairs' if args.sticky_traps_only else '20 guards,32 sticky masks,10 trap pairs'),'passed',flush=True)
      result.update(complete=True,passed=True)
    except Exception as error:
      result['failure']=repr(error)
      raise
    finally:
      retain()


if __name__=='__main__':
    main()
