#!/usr/bin/env python3
"""Prepare buffers, or explicitly build/run the isolated installed-ABI3 test."""
import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import os
import shutil
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/policy-v2/final-focused-audit-fixed/reversal-physical"
PREFIX = ROOT / "HEPTools/ninjars"
EXPECTED_ARCHIVE = "1199bc84570bee76a6b162a86fd5c191479236b2c0617fe4bf250d5de0d9df32"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ncoeff(rank):
    return (rank+1)*(rank+2)*(rank+3)*(rank+4)//24


def prepare():
    assert not (HERE / "result.json").exists(), "A completed attempt is immutable"
    cases=[]
    dat=["2"]
    for filename in ("madloop-gg-hh-top-triangle-0.json", "madloop-gg-hh-top-box-2.json"):
        path=ROOT/"vendor/ninjars/tests/data/cases"/filename
        fixture=json.loads(path.read_text())
        lane=fixture["lanes"][1]
        n,rank=fixture["n"],fixture["rank"]
        c=[[Fraction.from_float(float(x)) for x in z] for z in lane["tensor"]]
        p=[[Fraction.from_float(float(x)) for x in v] for v in lane["shifts"]]
        s=[[Fraction.from_float(float(x)) for x in row] for row in lane["s_matrix"]]
        # Keep the validated physical real squared masses; add distinct, exactly
        # representable negative widths so propagator/mass permutation is tested.
        masses=[[Fraction.from_float(float(z[0])),-Fraction(j+1,4)]
                for j,z in enumerate(lane["masses_squared"])]
        scale=Fraction.from_float(float(lane["mu_squared"]))
        assert len(c)==ncoeff(rank)
        cr=[row[:] for row in c]
        for degree in range(1,rank+1,2):
            for k in range(ncoeff(degree-1),ncoeff(degree)):
                cr[k]=[-x for x in cr[k]]
        assert any(any(x for x in z) for z in c[1:5])
        def form(coefficients,shifts,matrix,mass):
            # Values after promotion/multiplication in Fortran are rounded once
            # to binary128. The exact mathematical factor is recorded here.
            return dict(coefficients_dp=[[str(x) for x in z] for z in coefficients],
                coefficients_mp_source="binary64 values widened exactly, then multiplied by 1+2^-100 in binary128",
                shifts=[[str(x) for x in v] for v in shifts],
                s_matrix=[[str(x) for x in v] for v in matrix],
                masses_squared=[[str(x) for x in v] for v in mass],scale_squared=str(scale))
        cases.append(dict(n=n,rank=rank,fixture=str(path.relative_to(ROOT)),
            fixture_sha256=sha(path),lane=lane["id"],
            modification="Keep captured tensors, shifts, exact S and real masses; set squared-mass imaginary components to -(index+1)/4.",
            original=form(c,p,s,masses),
            reversed=form(cr,[[-x for x in v] for v in reversed(p)],
                [list(reversed(v)) for v in reversed(s)],list(reversed(masses))),
            negative_unflipped_numerator=form(c,[[-x for x in v] for v in reversed(p)],
                [list(reversed(v)) for v in reversed(s)],list(reversed(masses)))))
        def row(values): return " ".join(format(float(x),".39e") for x in values)
        dat.append(f"{n} {rank} {len(c)} {format(float(scale),'.39e')}")
        dat.extend(row(z) for z in c)
        dat.extend(row(v) for v in p)
        dat.extend(row(z) for z in masses)
        dat.extend(row([s[i][j] for i in range(n)]) for j in range(n))
    (HERE/"inputs.dat").write_text("\n".join(dat)+"\n")
    (HERE/"inputs.json").write_text(json.dumps({"exact_rational_dp_buffers":cases,
        "driver_sha256":sha(HERE/"driver.f90"),"data_sha256":sha(HERE/"inputs.dat"),
        "convention":"D_i=(q+V_i)^2-m_i^2-mu^2; N'(q)=N(-q), reversed propagator order.",
        "negative_control_bound":"10 * (1e-7 + 1e-10) * common largest Laurent magnitude",
        "prepared_only":True},indent=2)+"\n")
    print("Prepared physical buffers; no compile or evaluation.")


def run(cpu):
    assert (HERE/"inputs.json").is_file(), "Run prepare first"
    assert not (HERE/"result.json").exists(), "Preserve previous attempts"
    assert not WORK.exists(), "Use a new namespace after any prior attempt"
    assert json.loads((HERE/"inputs.json").read_text())["driver_sha256"]==sha(HERE/"driver.f90")
    assert sha(PREFIX/"lib/libninjars.a")==EXPECTED_ARCHIVE
    config=json.loads((PREFIX/"lib/ninjars-config.json").read_text())
    assert config["abi_version"]==3
    assert json.loads((HERE/"inputs.json").read_text())["data_sha256"]==sha(HERE/"inputs.dat")
    WORK.mkdir(parents=True)
    shutil.copy2(HERE/"inputs.dat",WORK/"inputs.dat")
    compiler=shutil.which("gfortran")
    assert compiler
    prefix_flags=config.get("link_before",[])
    assert prefix_flags, "Missing validated native-runtime prefix metadata"
    command=[compiler,"-O2","-ffree-line-length-none","-I",str(PREFIX/"lib"),
        str(HERE/"driver.f90"),*prefix_flags,str(PREFIX/"lib/libninjars.a"),
        *config["native_libs"],"-o",str(WORK/"check")]
    result=dict(command=command,driver_sha256=sha(HERE/"driver.f90"),
        inputs_sha256=sha(HERE/"inputs.json"),runner_sha256=sha(__file__),
        archive_sha256=EXPECTED_ARCHIVE,config_sha256=sha(PREFIX/"lib/ninjars-config.json"),
        module_sha256=sha(PREFIX/"lib/mninjars.mod"),passed=False)
    try:
        built=subprocess.run(command,cwd=WORK,capture_output=True,text=True)
        (HERE/"build.log").write_text(built.stdout+built.stderr)
        result["build_exit_code"]=built.returncode
        built.check_returncode()
        result["executable_sha256"]=sha(WORK/"check")
        execution=["taskset","-c",str(cpu),str(WORK/"check")]
        result["execution"]=execution
        ran=subprocess.run(execution,cwd=WORK,capture_output=True,text=True,env=os.environ)
        output=ran.stdout+ran.stderr
        (HERE/"run.log").write_text(output)
        result["exit_code"]=ran.returncode
        result["output_sha256"]=sha(HERE/"run.log")
        ran.check_returncode()
        rows=[line for line in output.splitlines() if line.startswith("ROW ")]
        assert len(rows)==30
        assert [tuple(map(int,line.split()[1:4])) for line in rows]==[
            (case,phase,mode) for case in (1,2) for phase in range(1,6) for mode in (1,2,3)]
        assert all(int(line.split()[4])==0 for line in rows)
        result["rows"]=rows
        assert "PASS_REVERSAL_ABI3" in output
        assert sum(line.startswith("PASS_CASE ") for line in output.splitlines())==2
        assert sum(line.startswith("NEGATIVE_CASE ") for line in output.splitlines())==2
        result["mp_reversal_tolerance_fractions"]=[
            line for line in output.splitlines() if line.startswith("PASS_CASE ")]
        result["negative_controls"]=[
            line for line in output.splitlines() if line.startswith("NEGATIVE_CASE ")]
        result["passed"]=True
    finally:
        (HERE/"result.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Installed ABI3 reversal control passed.")


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("action",choices=("prepare","run"))
    parser.add_argument("--cpu",type=int,default=5)
    args=parser.parse_args()
    prepare() if args.action=="prepare" else run(args.cpu)
