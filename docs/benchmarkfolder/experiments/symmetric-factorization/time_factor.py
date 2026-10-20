#!/usr/bin/env python3
"""Exploratory original/factored full guarded QP kernel CPU-time comparison."""
import sys
sys.dont_write_bytecode = True

import argparse
import json
import os
from pathlib import Path
import statistics

import check_fallback as check
import run_oracle

HERE = Path(__file__).resolve().parent


def timing_driver(driver_source, rank, terminal, loops):
    old = f"TEST_{'TERMINAL' if terminal else 'UPDATE'}_WL_{rank}_1"
    prefix, rest = driver_source.split("do trial=1,10\n", 1)
    initialization, _ = rest.split("  entry=trial/=8\n", 1)
    prefix = prefix.replace("integer :: trial,i,j,k,l,m,n,hi,fast,fallback,diag",
        "integer :: trial,i,j,k,l,m,n,hi,fast,fallback,diag,pair,order,which,it\n"
        "complex*32 measured(4,0:MAXC-1,4)\n"
        "integer*8 oldbits(4*4*MAXC*4),newbits(4*4*MAXC*4)\n"
        "real*8 t0,t1,times(0:1),checksums(0:1)")
    prefix = prefix[:prefix.index("write(*,'(A)') 'FORMAT")]
    extra = ",entry" if terminal else ""
    body = f"""
open(73,file='initial-values.dat',status='replace')
do trial=1,2
{initialization}
  entry=.true.
  call ieee_set_flag(ieee_inexact,.true.)
  call {old}(A,4,B,4,4,oldout{extra})
  call {old}_FACTOR(A,4,B,4,4,newout,entry)
  oldbits=transfer(oldout,oldbits)
  newbits=transfer(newout,newbits)
  write(73,*) 'TRIAL',trial
  do i=1,4
    do l=0,NC-1
      do k=1,4
        call dump_complex('A',k,l,i,A(k,l,i))
      enddo
    enddo
  enddo
  do k=1,4
    do m=0,4
      do j=1,4
        call dump_complex('B',j,m,k,B(j,m,k))
      enddo
    enddo
  enddo
  do i=1,4
    do n=0,NO-1
      do j=1,4
        call dump_complex('OLD',j,n,i,oldout(j,n,i))
        call dump_complex('NEW',j,n,i,newout(j,n,i))
      enddo
    enddo
  enddo
  do pair=1,7
    do order=0,1
      which=mod(order+pair,2)
      call ieee_set_flag(ieee_inexact,.true.)
      call cpu_time(t0)
      if(which==0) then
        do it=1,{loops}
          call {old}(A,4,B,4,4,measured{extra})
        enddo
      else
        do it=1,{loops}
          call {old}_FACTOR(A,4,B,4,4,measured,entry)
        enddo
      endif
      call cpu_time(t1)
      times(which)=t1-t0
      if(which==0) then
        if(any(transfer(measured,oldbits)/=oldbits)) stop 10
      else
        if(any(transfer(measured,newbits)/=newbits)) stop 11
      endif
      checksums(which)=real(sum(measured),8)
    enddo
    write(*,*) trial,pair,times,checksums
  enddo
enddo
close(73)
end program
"""
    # The fixture generator includes these calls only in trials9/10; the timing
    # loop executes trials1/2. Keep the same initialization source self-contained.
    helper = driver_source.split("\ncontains\n", 1)[1].split("\nend program", 1)[0]
    helper = helper.replace("write(*,'(A,3(1X,I0),4(1X,Z16.16))')",
                            "write(73,'(A,3(1X,I0),4(1X,Z16.16))')")
    return prefix+body.replace("\nend program\n", "\ncontains\n"+helper+"\nend program\n")


def check_initial_bits(path, oracle_module, cases):
    expected = {case["number"]:case["bits"] for case in cases[:2]}
    actual = {}
    trial = None
    for line in path.read_text().splitlines():
        values = line.split()
        if values[0] == "TRIAL":
            trial = int(values[1])
            assert trial in expected and trial not in actual
            actual[trial] = {}
            continue
        assert trial is not None and values[0] in ("A", "B", "OLD", "NEW")
        key = tuple(map(int, values[1:4]))
        array = actual[trial].setdefault(values[0], {})
        assert key not in array
        _, array[key] = oracle_module.complex128(values[4:])
    assert actual == expected, "Timed initialization/warm outputs differ from validated stored bits"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--oracle-run", required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[2, 4])
    parser.add_argument("--vertex", default="MP_FFV1L2_1")
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--loops", type=int, default=300)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    report = HERE/args.run_id
    report.mkdir(exist_ok=False)
    source_run = HERE/args.oracle_run
    run_record = json.loads((source_run/"run.json").read_text())
    assert run_record["complete"]
    assert run_record["factor_sha256"] == check.sha(HERE/"factor.py")
    assert run_record["oracle_driver_sha256"] == check.sha(source_run/"oracle-driver.py")
    driver = run_oracle.load_driver(source_run/"oracle-driver.py")
    driver.HERE = HERE/"oracle"
    driver.PROOF = HERE/"analysis.json"
    for name in ("factor.py", "check_fallback.py", "time_factor.py"):
        (report/name).write_bytes((HERE/name).read_bytes())
    result = dict(complete=False, acceptance_eligible=False, cpu=args.cpu,
        loops=args.loops, flags=check.FLAGS["nlo"], results=[],
        factor_sha256=check.sha(HERE/"factor.py"),
        oracle_driver_sha256=check.sha(source_run/"oracle-driver.py"),
        runner_sha256=check.sha(__file__),
        scope="Synthetic complete guarded QP kernels, seven alternating pairs; "
              "dense/two-thirds-zero A with actual source-derived rounded B "
              "replaced by exactly phase-related stored derivative rows. "
              "Ordinary CPU-time diagnostics during concurrent host work, "
              "not a full-process speed or acceptance claim. No trace counters "
              "inside timed kernels. Each timed output is bit-identical to its "
              "own independently oracle-validated variant, not to the other variant.")
    (report/"timing.json").write_text(json.dumps(result, indent=2)+"\n")
    for rank in args.ranks:
      for terminal in (False, True):
        key = f"nlo-{args.vertex}-rank{rank}-{'terminal' if terminal else 'full'}"
        evidence = source_run/key
        oracle = json.loads((evidence/"exact-oracle.json").read_text())
        assert oracle["complete"] and oracle["passed"]
        assert oracle["dump_sha256"] == check.sha(evidence/"dump.txt.gz")
        assert oracle["proof_sha256"] == check.factor.PROOF_SHA256
        assert len(oracle["cases"]) == 10
        assert all(c["fast"] == 1 for c in oracle["cases"][:2])
        recorded = next(r for r in run_record["results"] if r["key"] == key)
        for name, digest in recorded["sources"].items():
            assert check.sha(evidence/name) == digest, (key,name)
        assert check.sha(HERE/"oracle/oracle.py") == oracle["oracle_sha256"]
        oracle_module = run_oracle.load_driver(HERE/"oracle/oracle.py")
        work = report/key
        sources = check.write_sources(work, rank, terminal, args.vertex)
        # Bind regenerated constants/reference/instrumented candidate to the
        # exact oracle-run source before removing only the trace counters.
        for name in sources:
            assert check.sha(work/name) == recorded["sources"][name], (key,name)
        oracle_source = driver.driver(rank,args.vertex,terminal)
        assert __import__("hashlib").sha256(oracle_source.encode()).hexdigest() == \
            recorded["sources"]["oracle.f90"]
        gen = check.FortranPolynomialRoutines(6, updater_max_rank=2,
            proc_prefix="TEST_", sub_prefix="TEST_", coef_format="complex*32")
        uninstrumented = check.factor.generate(gen, rank, args.vertex, terminal)
        with check.FortranWriter(str(work/"factor.f")) as writer:
            writer.writelines(uninstrumented)
        text = timing_driver(oracle_source,rank,terminal,args.loops)
        (work/"timing.f90").write_text(text)
        command = ["gfortran", *check.FLAGS["nlo"], *sources, "timing.f90", "-o", "timing"]
        built = check.run(command, work, "build.log")
        ran = check.run([str(work/"timing")], work, "timing.log")
        check_initial_bits(work/"initial-values.dat",oracle_module,
                           oracle_module.read_cases(evidence/"dump.txt.gz"))
        rows = [line.split() for line in ran["output"].splitlines()]
        assert len(rows) == 14 and all(len(row) == 6 for row in rows)
        for trial in (1,2):
            selected = [row for row in rows if int(row[0])==trial]
            assert [int(row[1]) for row in selected] == list(range(1,8))
            seconds = dict(original=[float(row[2]) for row in selected],
                           factor=[float(row[3]) for row in selected])
            medians = {name:statistics.median(values) for name,values in seconds.items()}
            record = dict(key=key, trial=trial, seconds=seconds, medians=medians,
                ratio=medians["factor"]/medians["original"],
                oracle_sha256=check.sha(evidence/"exact-oracle.json"),
                initialization_matches_oracle_bits=True,
                initialization_sha256=check.sha(work/"initial-values.dat"),
                executable_sha256=check.sha(work/"timing"), build=built,
                sources={p.name:check.sha(p) for p in work.iterdir()
                         if p.suffix in (".f", ".f90", ".inc")})
            result["results"].append(record)
            print(key,trial,record["ratio"],flush=True)
        (report/"timing.json").write_text(json.dumps(result, indent=2)+"\n")
    result["complete"] = True
    (report/"timing.json").write_text(json.dumps(result, indent=2)+"\n")


if __name__ == "__main__":
    main()
