#!/usr/bin/env python3
"""Exploratory complete guarded-kernel comparison; not acceptance timing."""
import argparse
import json
import os
from pathlib import Path
import shutil
import statistics

import check_static as check
import static_owner

HERE = Path(__file__).resolve().parent
DRIVER = """
program measure
use TEST_POLYNOMIAL_CONSTANTS
use, intrinsic :: ieee_exceptions
implicit none
complex(16) a(4,0:209,4),b(4,0:14,4),out(4,0:209,4),reference(4,0:209,4)
integer(8) bits(4*4*210*4)
integer i,j,k,pair,order,which,it,density
real(8) t0,t1,times(0:3),checksums(0:3)
do density=0,1
 do k=1,4
  do j=0,209
   do i=1,4
    a(i,j,k)=cmplx(real(mod(i+3*j+7*k,13),kind=16)/7,&
                  real(mod(11*i+j+k,17),kind=16)/11,kind=16)
    if(density==1.and.mod(i+j+k,3)/=0) a(i,j,k)=0
   enddo
  enddo
  do j=0,14
   do i=1,4
    b(i,j,k)=cmplx(real(i+j+k,kind=16)/3,real(2*i+j-k,kind=16)/7,kind=16)
   enddo
  enddo
 enddo
 @INPUT@
 call ieee_set_halting_mode(ieee_all,.false.)
 call ieee_set_flag(ieee_inexact,.true.)
 call @NAME@(a,4,b,4,4,reference@EXTRA@)
 do pair=1,7
  do order=0,3
   which=mod(order+pair,4)
   call ieee_set_flag(ieee_inexact,.true.)
   call cpu_time(t0)
   select case(which)
   @CALLS@
   end select
   call cpu_time(t1)
   times(which)=t1-t0
   if(any(transfer(reference,bits)/=transfer(out,bits))) stop 10
   checksums(which)=real(sum(out),kind=8)
  enddo
  write(*,*) density,pair,times,checksums
 enddo
enddo
end program
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ranks", default="2,3,4")
    parser.add_argument("--loops", type=int, default=300)
    parser.add_argument("--cpu", type=int, default=2)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    report = HERE / args.tag
    report.mkdir(exist_ok=False)
    inventory = static_owner.PREVIOUS / "final-inventory.json"
    shape = next(s for s in json.loads(inventory.read_text())["qcd-4"]
        ["routines"]["MP_FFV1L2_1"]["shapes"]
        if s["in_size"] == s["out_size"] == 4)
    groups = static_owner.shared.groups(inventory_path=inventory, case="qcd-4")
    copies = [f"b({j},{m},{k})=0" for j, m, k in shape["literal_zero_entries"]]
    for group in groups:
        ref = ",".join(map(str, group[0]))
        copies += ["b(" + ",".join(map(str, entry)) + ")=b(" + ref + ")"
                   for entry in group[1:]]
    names = ("original", "lazy", "owner-only", "folded")
    results = []
    for rank in (int(r) for r in args.ranks.split(",")):
      for terminal in (False, True):
        key = f"rank{rank}-" + ("terminal" if terminal else "full")
        work = check.check.ROOT / ".benchmark-work/experiments/symmetric-static-owner" / args.tag / key
        check.write_sources(work, rank, "MP_FFV1L2_1", terminal, "qcd-4",
                            inventory, True)
        gen = check.check.FortranPolynomialRoutines(6, updater_max_rank=2,
            proc_prefix="TEST_", sub_prefix="TEST_", coef_format="complex*32")
        name = "TEST_" + ("TERMINAL" if terminal else "UPDATE") + f"_WL_{rank}_1"
        sources = {
            "lazy.f": static_owner.LEGACY_GENERATE(gen, rank, terminal=terminal,
                suffix="_LAZY", case="qcd-4", inventory_path=inventory),
            "owner-only.f": static_owner.generate(gen, rank, terminal=terminal,
                suffix="_OWNER", case="qcd-4", inventory_path=inventory, fold_guards=False),
            "folded.f": static_owner.generate(gen, rank, terminal=terminal,
                suffix="_FOLDED", case="qcd-4", inventory_path=inventory, fold_guards=True),
        }
        for filename, source in sources.items():
            with check.check.FortranWriter(str(work / filename)) as writer:
                writer.writelines(source)
        extra = ",.true." if terminal else ""
        calls = "\n".join(
            f"case({which})\n do it=1,{args.loops}\n"
            f"  call {name}{suffix}(a,4,b,4,4,out{extra})\n enddo"
            for which, suffix in enumerate(("", "_LAZY", "_OWNER", "_FOLDED")))
        text = DRIVER.replace("@INPUT@", "\n".join(copies)).replace(
            "@NAME@", name).replace("@EXTRA@", extra).replace("@CALLS@", calls)
        (work / "timing.f90").write_text(text)
        command = ["gfortran", *check.FLAGS["nlo"], "constants.f", "full.f",
                   *sources, "timing.f90", "-o", "timing"]
        check.run(command, work, "build.log")
        output = check.run([str(work / "timing")], work, "timing.log")
        rows = [line.split() for line in output.splitlines()]
        assert len(rows) == 14 and all(len(row) == 10 for row in rows)
        for density in (0, 1):
            seconds = {name: [float(row[i + 2]) for row in rows if int(row[0]) == density]
                       for i, name in enumerate(names)}
            medians = {name: statistics.median(values) for name, values in seconds.items()}
            record = dict(rank=rank, terminal=terminal, density=density,
                samples_seconds=seconds, median_seconds=medians,
                ratios_to_original={name: value / medians["original"]
                                    for name, value in medians.items()},
                ratios_to_lazy={name: value / medians["lazy"]
                                for name, value in medians.items()},
                executable_sha256=check.check.sha(work / "timing"),
                sources={p.name: check.check.sha(p) for p in work.iterdir()
                         if p.suffix in (".f", ".f90", ".inc")})
            results.append(record)
            print(key, density, record["ratios_to_original"], flush=True)
        destination = report / key
        destination.mkdir()
        for path in work.iterdir():
            if path.suffix in (".f", ".f90", ".inc", ".log"):
                shutil.copy2(path, destination / path.name)
        (report / "timing.json").write_text(json.dumps(dict(complete=False,
            acceptance_eligible=False, results=results), indent=2) + "\n")
    for path in (Path(__file__), HERE / "static_owner.py", HERE / "check_static.py"):
        shutil.copy2(path, report / path.name)
    (report / "timing.json").write_text(json.dumps(dict(complete=True,
        acceptance_eligible=False, results=results, cpu=args.cpu, loops=args.loops,
        flags=check.FLAGS["nlo"], variants=names,
        scope="Synthetic whole guarded QP kernels, seven rotated repetitions "
              "per four variants, including dispatch and all bounds/identity/"
              "trap checks. Dense or two-thirds-zero A, actual vertex zero/group "
              "patterns and synthetic finite complex operands. Timed outputs "
              "are bit-identical. Concurrent host work: diagnostic kernel CPU "
              "time, not full-process or acceptance performance.",
        runner_sha256=check.check.sha(__file__)), indent=2) + "\n")


if __name__ == "__main__":
    main()
