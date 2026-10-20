#!/usr/bin/env python3
"""Build an isolated ABI2 bridge and run the normal DP decision before rescue."""
import argparse, gzip, hashlib, importlib.util, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[4]
DOC = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/experiments/qcd3-point67-t3"
BRIDGE = WORK / "guarded-bridge"
OUTPUT = WORK / "guarded-qcd3"
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, v): p.write_text(json.dumps(v, indent=2) + "\n")
def prepare():
    if not BRIDGE.exists():
        shutil.copytree(ROOT/"vendor/ninjars", BRIDGE,
            ignore=shutil.ignore_patterns("target","install","*.o","*.a","*.mod","*.log","smoke"))
    for f in ["guarded.rs", "guarded_fenv.c", "guarded_tests.rs"]:
        if (DOC/f).exists(): shutil.copy2(DOC/f, BRIDGE/"src"/f)
    shutil.copy2(DOC/"guarded-build.rs", BRIDGE/"build.rs")
    text = (ROOT/"vendor/ninjars/src/lib.rs").read_text()
    text = text.replace("mod tensor;", "mod tensor;\nmod guarded;\nuse guarded::GuardedTensor as TensorNumerator;")
    text = text.replace("FastStatus, TensorNumerator}", "FastStatus}")
    if (DOC/"guarded_tests.rs").exists(): text += "\n#[cfg(test)]\nmod guarded_tests;\n"
    (BRIDGE/"src/lib.rs").write_text(text)
    cargo = (ROOT/"vendor/ninjars/Cargo.toml").read_text()
    cargo += '\n[build-dependencies]\ncc = "1"\n'
    cargo = cargo.replace('serde_json = "1"', 'serde_json = {version="1",features=["float_roundtrip"]}')
    (BRIDGE/"Cargo.toml").write_text(cargo)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(WORK/"guarded-target")
    subprocess.run(["cargo","metadata","--format-version=1","--manifest-path",str(BRIDGE/"Cargo.toml")],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, check=True)
    save(DOC/"guarded-source.json", {"production_revision": subprocess.check_output(
        ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "files": {str(p.relative_to(BRIDGE)):sha(p) for p in
            [BRIDGE/"Cargo.toml",BRIDGE/"Cargo.lock",BRIDGE/"build.rs",*sorted((BRIDGE/"src").glob("*"))]},
        "note":"Isolated ABI2 bridge. No installed production source or libraries changed."})
def build():
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(WORK/"guarded-target")
    env["MAKEFLAGS"] = "SHELL="+shutil.which("bash")
    command = ["make","-C",str(BRIDGE),"-j2","all"]
    with (DOC/"guarded-build.log").open("w") as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(DOC/"guarded-build.json",{"command":command,"archive_sha256":sha(BRIDGE/"libninjars.a"),
        "source_manifest_sha256":sha(DOC/"guarded-source.json")})
def process(cpu, existing=False):
    source = ROOT/".benchmark-work/experiments/qcd3-point67-dp/qcd3"
    if existing:
        archive=DOC/"guarded-initial-run"
        archive.mkdir()
        for name in ["guarded-process.json","guarded-process-reductions.json",
            "guarded-process-comparison.json","guarded-normal-dp-decision.dat",
            "guarded-process-build.log","guarded-process.log"]:
            shutil.copy2(DOC/name,archive/name)
    else:
        if OUTPUT.exists(): raise RuntimeError(f"Refusing overwrite {OUTPUT}")
        shutil.copytree(source,OUTPUT,symlinks=True)
    # Resolve internal absolute symlinks into the copied output, retaining
    # reference library links read-only. Only the local libninjars is replaced.
    for p in OUTPUT.rglob("*"):
        if p.is_symlink() and str(p.readlink()).startswith(str(source)):
            target=str(p.readlink()).replace(str(source),str(OUTPUT),1)
            p.unlink();p.symlink_to(target)
    lib=OUTPUT/"lib/libninjars.a"
    if lib.is_symlink() or lib.exists(): lib.unlink()
    shutil.copy2(BRIDGE/"libninjars.a",lib)
    proc=next(OUTPUT.glob("SubProcesses/P*_*"))
    spec=importlib.util.spec_from_file_location("paired_probe",DOC.parent/"qcd3-point67-dp/probe.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    card=module.MadLoopParam(str(OUTPUT/"Cards/MadLoopParams.dat"))
    card["MLReductionLib"]="8";card["CTModeRun"]=-1;card["MLStabThres"]=1e-9
    for p in [OUTPUT/"Cards/MadLoopParams.dat",*OUTPUT.glob("SubProcesses/**/MadLoopParams.dat")]:
        if not p.resolve().is_relative_to(OUTPUT): raise RuntimeError(f"External card {p}")
        card.write(str(p),template=str(ROOT/"Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    # This inherited diagnostic driver explicitly STOPs immediately after the
    # normal DP decision. It cannot invoke an MP callback.
    assert "FILE='normal-dp-decision.dat'" in (proc/"loop_matrix.f").read_text()
    env=os.environ.copy();env["MAKEFLAGS"]="SHELL="+shutil.which("bash")
    (proc/"check").unlink(missing_ok=True)
    with (DOC/"guarded-process-build.log").open("w")as log:
        subprocess.run(["make","-j2","check"],cwd=proc,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    env["OMP_NUM_THREADS"]=env["OPENBLAS_NUM_THREADS"]="1"
    env["LD_LIBRARY_PATH"]=":".join([str(ROOT/".benchmark-work/original/HEPTools/ninja/lib"),
        str(ROOT/".benchmark-work/original/HEPTools/collier"),env.get("LD_LIBRARY_PATH","")])
    command=["taskset","-c",str(cpu),"./check",str(ROOT/"docs/benchmarkfolder/points/qcd-3.dat")]
    with (DOC/"guarded-process.log").open("w")as log:
        subprocess.run(command,cwd=proc,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=module.parse_trace(proc/"paired-reductions.dat")
    if existing:
        assert rows==json.loads((archive/"guarded-process-reductions.json").read_text())
    save(DOC/"guarded-process-reductions.json",rows)
    shutil.copy2(proc/"normal-dp-decision.dat",DOC/"guarded-normal-dp-decision.dat")
    shutil.copy2(OUTPUT/"Cards/MadLoopParams.dat",DOC/"guarded-MadLoopParams.dat")
    save(DOC/"guarded-process.json",{"command":command,"executable_sha256":sha(proc/"check"),
        "archive_sha256":sha(lib),"driver_sha256":sha(proc/"check_sa.f"),
        "loop_matrix_sha256":sha(proc/"loop_matrix.f"),"interface_sha256":sha(proc/"CT_interface.f"),
        "card_sha256":sha(DOC/"guarded-MadLoopParams.dat"),
        "point_file_sha256":sha(ROOT/"docs/benchmarkfolder/points/qcd-3.dat"),
        "reductions":len(rows),"failed_ninjars":sum(r["status_ninjars"]!=0 for r in rows),
        "failed_ninja":sum(r["status_ninja"]!=0 for r in rows)})
    print((DOC/"guarded-normal-dp-decision.dat").read_text())
    analyze()
def analyze():
    actual=json.loads((DOC/"guarded-process-reductions.json").read_text())
    original=json.load(gzip.open(DOC.parent/"qcd3-point67-dp/normal/reductions.json.gz","rt"))
    assert len(actual)==len(original)==232
    identity_fields=["loop","direction","n","rank","coefficients","s_matrix_column_major",
        "momenta_column_major","masses_squared","scale_squared","ninja"]
    for a,b in zip(actual,original):
        for key in identity_fields: assert a[key]==b[key],(key,a["loop"],a["direction"])
    lines=(DOC/"guarded-normal-dp-decision.dat").read_text().splitlines()
    directions=[list(map(float,line.split()[2:]))for line in lines[3:]if line.split()[0]=="0"]
    mean=[sum(v)/2 for v in zip(*directions)]
    comparisons=[]
    for backend in [1,6,7]:
        path=ROOT/f"docs/benchmarkfolder/data/baseline-qcd-3-{backend}-calibration.json"
        reference=json.loads(path.read_text())
        rows=[r for r in reference["rows"]if r["point"]==67 and r["order"]==0]
        assert len(rows)==1
        row=rows[0];coeff=[row["coefficients"][0],*mean]
        scale=max(map(abs,coeff+row["coefficients"]))
        components=[]
        for k,(a,b)in enumerate(zip(coeff,row["coefficients"])):
            tolerance=1e-7*max(abs(a),abs(b))+1e-10*scale
            components.append({"component":k,"value":a,"reference":b,
                "difference":a-b,"tolerance":tolerance,"fraction":abs(a-b)/tolerance,
                "passed":abs(a-b)<=tolerance})
        comparisons.append({"backend":backend,"reference_path":str(path.relative_to(ROOT)),
            "reference_sha256":sha(path),"reference_status":row["status"],"components":components})
    save(DOC/"guarded-process-comparison.json",{"all_232_inputs_and_paired_ninja_outputs_identical":True,
        "directions":directions,"averaged_laurent":mean,"stability_estimate":float(lines[1]),
        "mixed_tolerance":{"relative":1e-7,"scale":1e-10},"comparisons":comparisons,
        "all_components_pass":all(c["passed"]for r in comparisons for c in r["components"]),
        "note":"Point67 only, using retained historical identical-input legacy coefficients. Normal driver stops before rescue/final return code. This is not the required full100point/current-source validation or runtime acceptance."})
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","build","process","relink","analyze"])
    p.add_argument("--cpu",type=int,default=4);a=p.parse_args()
    if a.action=="relink": process(a.cpu,True)
    elif a.action=="process": process(a.cpu)
    else: globals()[a.action]()
