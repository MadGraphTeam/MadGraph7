#!/usr/bin/env python3
"""Bitwise tests of the actual mixed-order CREATE routines with a bounded cache."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prototype import polynomial


def main():
    os.sched_setaffinity(0, {2})
    source = next((ROOT / ".benchmark-work/outputs/candidate/control-mixed").glob("SubProcesses/P*_*"))
    text = (source / "polynomial.f").read_text()
    creates = re.findall(r"(?m)^      SUBROUTINE (?:MP_)?ML5_0_CREATE_LOOP_COEFS\(.*?^      END\s*$",
                         text, flags=re.S | re.M)
    merges = re.findall(r"(?m)^      SUBROUTINE (?:MP_)?ML5_0_MERGE_WL\(.*?^      END\s*$",
                        text, flags=re.S | re.M)
    assert len(creates) == len(merges) == 2
    original = "\n".join(creates)
    modified, dims = polynomial(original)
    all_dims = {name: int(re.search(rf"\b{name}=(\d+)", original)[1])
                for name in ("NLOOPGROUPS", "NCOMB", "NSQUAREDSO")}
    dims.update(all_dims)
    work = ROOT / ".benchmark-work/experiments/color-contraction/exact-v2"
    work.mkdir(parents=True, exist_ok=False)
    for name in ("coef_specs.inc", "loop_max_coefs.inc"):
        shutil.copy2(source / name, work / name)
    constants = re.search(r"(?m)^      MODULE .*?^      END MODULE[^\n]*", text, flags=re.S | re.M)[0]
    (work / "constants.f").write_text(constants + "\n")
    (work / "merges.f").write_text("\n".join(merges) + "\n")
    (work / "original.f").write_text(re.sub(r"SUBROUTINE ((?:MP_)?ML5_0_CREATE)",
                                          r"SUBROUTINE OLD_\1", original) + "\n")
    (work / "cached.f").write_text(modified + "\n")
    result = {"scope": "Actual generated mixed-order CREATE/MERGE routines; synthetic shared inputs",
              "production_modified": False, "dimensions": dims,
              "comparison": "Whole LOOPCOEFS bit patterns and cumulative batch IEEE exception flags",
              "contract": "IEEE flags are not cleared by user code during an active cache batch",
              "source_polynomial_sha256": hashlib.sha256(text.encode()).hexdigest(),
              "prototype_sha256": hashlib.sha256((HERE / "prototype.py").read_bytes()).hexdigest(),
              "tests": []}
    for kind in (8, 16):
        precision, mp = ("DP", "") if kind == 8 else ("QP", "MP_")
        nb, nr, ns = dims["NBORNAMPS"], dims["NCOLORROWS"], dims["NAMPSO"]
        nc, ng, nq = dims["NCOMB"], dims["NLOOPGROUPS"], dims["NSQUAREDSO"]
        driver = work / f"driver-{kind}.f90"
        driver.write_text(f"""
program exact_color
use iso_fortran_env, only: int64
use, intrinsic :: ieee_arithmetic
use, intrinsic :: ieee_exceptions
use ML5_0_POLYNOMIAL_CONSTANTS
implicit none
complex({kind}) :: amp({nb}),wf(MAXLWFSIZE,0:LOOPMAXCOEFS-1,MAXLWFSIZE)
complex({kind}) :: coefs(0:LOOPMAXCOEFS-1,{nq},{ng}),saved(0:LOOPMAXCOEFS-1,{nq},{ng})
complex({kind}) :: expected(0:LOOPMAXCOEFS-1,{nq},{ng})
integer :: cf_d({nr},{nb}),cf_n({nr},{nb}),goodhel({nc}),heloffset,helpicked
logical :: checkphase,heldoublechecked,goodamp({nq},{ng})
common /ML5_0_{mp}AMPS/amp
common /ML5_0_{mp}LCOEFS/coefs
common /ML5_0_CF/cf_d,cf_n
common /ML5_0_INIT/checkphase,heldoublechecked
common /ML5_0_FILTERS/goodamp,goodhel,heloffset
common /ML5_0_HELCHOICE/helpicked
integer(int64) :: oldbits(LOOPMAXCOEFS*{nq}*{ng}*{kind//4})
integer(int64) :: newbits(LOOPMAXCOEFS*{nq}*{ng}*{kind//4})
logical :: oldflags(size(ieee_all)),newflags(size(ieee_all))
integer :: epoch,trial,i,j,k,row,group,hel,rank,ls,sym,mult,checks
real({kind}) :: nan
nan=ieee_value(0.0_{kind},ieee_quiet_nan)
checks=0
do i=1,{nc}
  goodhel(i)=mod(i,3)+1
enddo
goodamp=.true.
heloffset=0
heldoublechecked=.true.
do epoch=1,8
  do i=1,{nb}
    amp(i)=cmplx(real(3*i-epoch,kind={kind})/7,real(5*i+epoch,kind={kind})/11,kind={kind})
  enddo
  if(epoch==7) amp(1)=cmplx(nan,1.0_{kind},kind={kind})
  if(epoch==8) amp=cmplx(-0.0_{kind},0.0_{kind},kind={kind})
  do i=1,{nr}
    do j=1,{nb}
      cf_n(i,j)=mod(i,3)*(j-2)
      cf_d(i,j)=(-1)**(mod(i,3)+j)*(mod(i,3)+j)
      if(epoch>=4.and.mod(i,7)==0) cf_n(i,j)=j+epoch
    enddo
  enddo
  call ML5_0_COLOR_CACHE_INVALIDATE()
  checkphase=mod(epoch,2)==0
  helpicked=-1
  if(epoch==3) helpicked=1
  do k=1,MAXLWFSIZE
    do j=0,LOOPMAXCOEFS-1
      do i=1,MAXLWFSIZE
        wf(i,j,k)=cmplx(real(i+j-k,kind={kind})/13,real(3*i-j+k,kind={kind})/17,kind={kind})
      enddo
    enddo
  enddo
  call ML5_0_COLOR_CACHE_BEGIN_{precision}()
  oldflags=.false.
  newflags=.false.
  do trial=1,160
    if(trial==81) call ML5_0_COLOR_CACHE_END_{precision}()
    if(trial>80) amp=amp+cmplx(0.125_{kind},-0.0625_{kind},kind={kind})
    row=mod(trial*17,{nr})+1
    group=mod(trial,{ng})+1
    hel=mod(trial,{nc})+1
    rank=mod(trial,3)
    ls=mod(trial,MAXLWFSIZE+1)
    sym=mod(trial,3)+1
    mult=mod(trial,7)-3
    coefs=cmplx(0.125_{kind},-0.25_{kind},kind={kind})
    saved=coefs
    call ieee_set_flag(ieee_all,oldflags)
    call OLD_{mp}ML5_0_CREATE_LOOP_COEFS(wf,rank,ls,group,sym,mult,row,hel)
    call ieee_get_flag(ieee_all,oldflags)
    expected=coefs
    coefs=saved
    call ieee_set_flag(ieee_all,newflags)
    call {mp}ML5_0_CREATE_LOOP_COEFS(wf,rank,ls,group,sym,mult,row,hel)
    call ieee_get_flag(ieee_all,newflags)
    oldbits=transfer(expected,oldbits)
    newbits=transfer(coefs,newbits)
    if(any(oldbits/=newbits)) then
      write(*,*) 'BIT MISMATCH',{kind},epoch,trial
      stop 1
    endif
    if(any(oldflags.neqv.newflags)) then
      write(*,*) 'FLAGS MISMATCH',{kind},epoch,trial,oldflags,newflags
      stop 2
    endif
    checks=checks+1
  enddo
enddo
write(*,*) 'PASS',{kind},checks
end program

integer function ML5_0_ML5SOINDEX_FOR_BORN_AMP(i)
integer i
ML5_0_ML5SOINDEX_FOR_BORN_AMP=mod(i-1,{ns})+1
end function
integer function ML5_0_ML5SOINDEX_FOR_LOOP_AMP(i)
integer i
ML5_0_ML5SOINDEX_FOR_LOOP_AMP=mod(i-1,{ns})+1
end function
integer function ML5_0_ML5SQSOINDEX(i,j)
integer i,j
ML5_0_ML5SQSOINDEX=mod(i+j,{nq})+1
end function
""")
        command = ["gfortran", "-O3", "-fcheck=all", "-ffixed-line-length-132",
                   "constants.f", "merges.f", "original.f", "cached.f",
                   driver.name, "-o", f"check-{kind}"]
        compiled = subprocess.run(command, cwd=work, capture_output=True, text=True)
        (work / f"build-{kind}.log").write_text(compiled.stdout + compiled.stderr)
        item = {"kind": kind, "build_passed": compiled.returncode == 0,
                "driver_sha256": hashlib.sha256(driver.read_bytes()).hexdigest()}
        if compiled.returncode == 0:
            checked = subprocess.run([f"./check-{kind}"], cwd=work, capture_output=True, text=True)
            item.update(passed=checked.returncode == 0, stdout=checked.stdout, stderr=checked.stderr)
        else:
            item.update(passed=False, stderr=compiled.stderr)
        result["tests"].append(item)
        print(json.dumps(item), flush=True)
    (HERE / "exact-regression.json").write_text(json.dumps(result, indent=2)+"\n")
    if not all(t["passed"] for t in result["tests"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
