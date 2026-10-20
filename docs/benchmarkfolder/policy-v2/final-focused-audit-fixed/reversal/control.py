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
WORK = ROOT / ".benchmark-work/policy-v2/final-focused-audit-fixed/reversal"
PREFIX = ROOT / "HEPTools/ninjars"
EXPECTED_ARCHIVE = "1199bc84570bee76a6b162a86fd5c191479236b2c0617fe4bf250d5de0d9df32"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ncoeff(rank):
    return (rank+1)*(rank+2)*(rank+3)*(rank+4)//24


def prepare():
    assert not (HERE / "result.json").exists(), "A completed attempt is immutable"
    cases = []
    for n in (3, 4):
        rank=n
        c = [[Fraction((7*k)%13-6,16),Fraction((11*k)%17-8,32)]
             for k in range(ncoeff(rank))]
        cr=[row[:] for row in c]
        for degree in range(1,rank+1,2):
            for k in range(ncoeff(degree-1),ncoeff(degree)):
                cr[k]=[-x for x in cr[k]]
        p=[[Fraction(0) for _ in range(4)] for _ in range(n)]
        for j in range(1,n):
            p[j][j]=Fraction(j)
        s=[[(p[i][0]-p[j][0])**2-sum((p[i][k]-p[j][k])**2 for k in range(1,4))
            for j in range(n)] for i in range(n)]
        masses=[[Fraction((j+2)**2),-Fraction((j+2)**2,128)] for j in range(n)]
        def form(coefficients, shifts, matrix, mass):
            factor=1+Fraction(1,2**100)
            return dict(coefficients_dp=[[str(x) for x in z] for z in coefficients],
                coefficients_mp=[[str(x*factor) for x in z] for z in coefficients],
                shifts=[[str(x) for x in v] for v in shifts],
                s_matrix=[[str(x) for x in v] for v in matrix],
                masses_squared=[[str(x) for x in v] for v in mass],
                scale_squared="49")
        cases.append(dict(n=n,rank=rank,
            original=form(c,p,s,masses),
            reversed=form(cr,[[-x for x in v] for v in reversed(p)],
                [list(reversed(v)) for v in reversed(s)],list(reversed(masses))),
            negative_unflipped_numerator=form(c,[[-x for x in v] for v in reversed(p)],
                [list(reversed(v)) for v in reversed(s)],list(reversed(masses)))))
    (HERE/"inputs.json").write_text(json.dumps({"exact_rational_buffers":cases,
        "driver_sha256":sha(HERE/"driver.f90"),
        "convention":"D_i=(q+V_i)^2-m_i^2-mu^2; N'(q)=N(-q), reversed propagator order.",
        "negative_control_bound":"10 * (1e-7 + 1e-10) * common largest Laurent magnitude",
        "prepared_only":True},indent=2)+"\n")
    print("Prepared exact buffers; no compile or evaluation.")


def run(cpu):
    assert (HERE/"inputs.json").is_file(), "Run prepare first"
    assert not (HERE/"result.json").exists(), "Preserve previous attempts"
    assert not WORK.exists(), "Use a new namespace after any prior attempt"
    assert json.loads((HERE/"inputs.json").read_text())["driver_sha256"]==sha(HERE/"driver.f90")
    assert sha(PREFIX/"lib/libninjars.a")==EXPECTED_ARCHIVE
    config=json.loads((PREFIX/"lib/ninjars-config.json").read_text())
    assert config["abi_version"]==3
    WORK.mkdir(parents=True)
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
        result["rows"]=rows
        assert "PASS_REVERSAL_ABI3" in output
        assert sum(line.startswith("PASS_CASE ") for line in output.splitlines())==2
        assert sum(line.startswith("NEGATIVE_CASE ") for line in output.splitlines())==2
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
