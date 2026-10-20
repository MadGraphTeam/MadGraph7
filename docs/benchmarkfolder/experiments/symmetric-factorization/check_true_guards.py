#!/usr/bin/env python3
"""Exercise later factor guards with a true sticky entry-INEXACT flag."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path

import check_fallback as check
import run_oracle

HERE = Path(__file__).resolve().parent


def driver(source, rank, terminal):
    old = f"TEST_{'TERMINAL' if terminal else 'UPDATE'}_WL_{rank}_1"
    prefix, rest = source.split("do trial=1,10\n",1)
    initialization = rest.split("  entry=trial/=8\n",1)[0]
    prefix = prefix.replace("integer :: trial,i,j,k,l,m,n,hi,fast,fallback,diag",
        "integer :: trial,i,j,k,l,m,n,hi,fast,fallback,diag,test,ins,outs\n"
        "integer*8 oldbits(4*4*MAXC*4),newbits(4*4*MAXC*4)")
    prefix = prefix[:prefix.index("write(*,'(A)') 'FORMAT")]
    helper = source.split("\ncontains\n",1)[1].split("\nend program",1)[0]
    extra = ",entry" if terminal else ""
    return prefix+f"""
do test=1,19
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
    A(1,0,1)=cmplx(scale(1.0_16,1001),0.0_16,16)
  case(5)
    A(1,0,1)=cmplx(scale(1.0_16,-1001),0.0_16,16)
  case(6)
    A(1,0,1)=cmplx(nearest(0.0_16,1.0_16),0.0_16,16)
  case(7)
    A(1,0,1)=cmplx(ieee_value(0.0_16,ieee_positive_inf),0.0_16,16)
  case(8)
    A(1,0,1)=cmplx(ieee_value(0.0_16,ieee_quiet_nan),0.0_16,16)
  case(9)
    A(1,0,1)=cmplx(ieee_value(0.0_16,ieee_signaling_nan),0.0_16,16)
  case(10)
    B(1,0,1)=cmplx(scale(1.0_16,1001),0.0_16,16)
  case(11)
    B(1,0,1)=cmplx(nearest(0.0_16,1.0_16),0.0_16,16)
  case(12)
    B(1,0,1)=cmplx(ieee_value(0.0_16,ieee_positive_inf),0.0_16,16)
  case(13)
    B(1,0,1)=cmplx(ieee_value(0.0_16,ieee_quiet_nan),0.0_16,16)
  case(14)
    B(1,0,1)=cmplx(ieee_value(0.0_16,ieee_signaling_nan),0.0_16,16)
  case(15)
    B(1,2,1)=B(1,2,1)+cmplx(1.0_16,0.0_16,16)
  case(16)
    ins=3
  case(17)
    outs=3
  case(18)
    ins=0
  case(19)
    outs=0
  end select
  saved_a=transfer(A,saved_a)
  saved_b=transfer(B,saved_b)
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,.true.)
  call {old}(A,4,B,ins,outs,oldout{extra})
  call ieee_get_flag(ieee_all,oldflags)
  fast=0
  fallback=0
  call ieee_set_flag(ieee_all,.false.)
  call ieee_set_flag(ieee_inexact,.true.)
  call {old}_FACTOR(A,4,B,ins,outs,newout,entry)
  call ieee_get_flag(ieee_all,newflags)
  oldbits=transfer(oldout,oldbits)
  newbits=transfer(newout,newbits)
  if(any(oldbits/=newbits)) then
    write(*,*) 'BITS',test
    stop 10
  endif
  if(any(oldflags.neqv.newflags)) then
    write(*,*) 'FLAGS',test,oldflags,newflags
    stop 11
  endif
  if(fast/=0.or.fallback/=1) then
    write(*,*) 'GUARD',test,fast,fallback
    stop 12
  endif
  if(any(saved_a/=transfer(A,saved_a)).or.any(saved_b/=transfer(B,saved_b))) stop 13
  write(*,*) 'PASS',test
enddo
call ieee_set_rounding_mode(ieee_nearest)
contains
{helper}
end program
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--cpu",type=int,default=3)
    args = parser.parse_args()
    os.sched_setaffinity(0,{args.cpu})
    report = HERE/args.run_id
    report.mkdir(exist_ok=False)
    oracle = run_oracle.load_driver(HERE/"oracle/driver.py")
    for name in ("factor.py","check_fallback.py","check_true_guards.py"):
        (report/name).write_bytes((HERE/name).read_bytes())
    result = dict(complete=False,acceptance_eligible=False,cpu=args.cpu,results=[],
        factor_sha256=check.sha(HERE/"factor.py"),
        oracle_driver_sha256=check.sha(HERE/"oracle/driver.py"),
        runner_sha256=check.sha(__file__),
        scope="True sticky entry-INEXACT, otherwise valid phase-related inputs, "
              "with one nonnearest/range/subnormal/nonfinite/shape/row violation. "
              "Require one fallback and exact public output/input bits plus flags.")
    for flag_name in check.FLAGS:
      for rank in (2,4):
       for terminal in (False,True):
        key = f"{flag_name}-rank{rank}-{'terminal' if terminal else 'full'}"
        work = report/key
        sources = check.write_sources(work,rank,terminal,"MP_FFV1L2_1")
        (work/"guards.f90").write_text(driver(
            oracle.driver(rank,"MP_FFV1L2_1",terminal),rank,terminal))
        built = check.run(["gfortran",*check.FLAGS[flag_name],*sources,
                           "guards.f90","-o","guards"],work,"build.log")
        ran = check.run([str(work/"guards")],work,"run.log")
        assert [line.split() for line in ran["output"].splitlines()] == [
            ["PASS",str(i)] for i in range(1,20)]
        result["results"].append(dict(key=key,build=built,run=ran,
            executable_sha256=check.sha(work/"guards"),
            sources={p.name:check.sha(p) for p in work.iterdir()
                     if p.suffix in (".f",".f90",".inc")}))
        (report/"results.json").write_text(json.dumps(result,indent=2)+"\n")
        print(key,"19 passed",flush=True)
    result.update(complete=True,comparisons=19*len(result["results"]))
    (report/"results.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__ == "__main__":
    main()
