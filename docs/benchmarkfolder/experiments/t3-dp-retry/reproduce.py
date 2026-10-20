#!/usr/bin/env python3
"""Isolated explicit compensated-DP retry; production source is read-only."""
import argparse, difflib, hashlib, json, os, shutil, subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
DOC=Path(__file__).resolve().parent
KERNEL=DOC.parent/"qcd3-point67-t3"
WORK=ROOT/".benchmark-work/experiments/t3-dp-retry"
BRIDGE=WORK/"bridge"
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2)+"\n")
def prepare():
    if BRIDGE.exists():raise RuntimeError(f"Refusing overwrite {BRIDGE}")
    shutil.copytree(ROOT/"vendor/ninjars",BRIDGE,
        ignore=shutil.ignore_patterns("target","install","*.o","*.a","*.mod","*.log","smoke"))
    for name in ["guarded.rs","guarded_fenv.c"]:
        shutil.copy2(KERNEL/name,BRIDGE/"src"/name)
    shutil.copy2(KERNEL/"guarded-build.rs",BRIDGE/"build.rs")
    source=(BRIDGE/"src/lib.rs").read_text()
    # The complete ordinary DP implementation remains textually unchanged.
    start=source.index("struct DpEntry {")
    stop=source.index("/// Evaluate one DP numerator.",start)
    state=source[start:stop].replace("DpEntry","CompensatedDpEntry").replace(
        "DpState","CompensatedDpState").replace("TensorNumerator","guarded::GuardedTensor")
    first=source.index('#[unsafe(no_mangle)]\npub unsafe extern "C" fn ninjars_evaluate_dp(')
    last=source.index("#[cfg(test)]\nfn parse_record",first)
    entry=source[first:last].replace("fn ninjars_evaluate_dp(","fn ninjars_evaluate_dp_compensated(").replace(
        "context.dp.lock()","context.compensated_dp.lock()").replace("DpState::new()","CompensatedDpState::new()")
    source=source[:last]+entry+source[last:]
    source=source[:stop]+state+source[stop:]
    source=source.replace("mod tensor;","mod tensor;\nmod guarded;")
    source=source.replace("    mp: Mutex<Option<MpContext>>,\n","    mp: Mutex<Option<MpContext>>,\n    compensated_dp: Mutex<Option<CompensatedDpState>>,\n",1)
    source=source.replace("            mp: Mutex::new(None),\n","            mp: Mutex::new(None),\n            compensated_dp: Mutex::new(None),\n",1)
    source+="\n#[cfg(test)]\nmod retry_tests;\n"
    (BRIDGE/"src/lib.rs").write_text(source)
    if (DOC/"retry_tests.rs").exists():shutil.copy2(DOC/"retry_tests.rs",BRIDGE/"src/retry_tests.rs")
    else:(BRIDGE/"src/retry_tests.rs").write_text("// Tests are supplied before the test build.\n")
    module=(BRIDGE/"mninjars.f90").read_text()
    i=module.index("    function evaluate_dp(");j=module.index("    function evaluate_mp(",i)
    interface=module[i:j].replace("function evaluate_dp(","function evaluate_dp_compensated(").replace(
        'name="ninjars_evaluate_dp"','name="ninjars_evaluate_dp_compensated"')
    module=module[:j]+interface+module[j:]
    i=module.index("  subroutine tensor_dp(");j=module.index("  subroutine pack(",i)
    routine=module[i:j].replace("subroutine tensor_dp(","subroutine ninjars_tensor_evaluate_compensated(").replace(
        "status = evaluate_dp(","status = evaluate_dp_compensated(")
    module=module[:j]+routine+module[j:]
    module=module.replace("  public :: ninjars_tensor_evaluate,","  public :: ninjars_tensor_evaluate_compensated\n  public :: ninjars_tensor_evaluate,",1)
    (BRIDGE/"mninjars.f90").write_text(module)
    header=(BRIDGE/"ninjars.h").read_text()
    i=header.index("int32_t ninjars_evaluate_dp(");j=header.index(";",i)+1
    header=header[:j]+"\n\n/* EXPERIMENTAL: explicit DP retry, not installed ABI2. */\n"+header[i:j].replace(
        "ninjars_evaluate_dp(","ninjars_evaluate_dp_compensated(")+header[j:]
    (BRIDGE/"ninjars.h").write_text(header)
    cargo=(BRIDGE/"Cargo.toml").read_text()+'\n[build-dependencies]\ncc = "1"\n'
    cargo=cargo.replace('serde_json = "1"','serde_json = {version="1",features=["float_roundtrip"]}')
    (BRIDGE/"Cargo.toml").write_text(cargo)
    env=os.environ.copy();env["CARGO_TARGET_DIR"]=str(WORK/"target")
    subprocess.run(["cargo","metadata","--format-version=1","--manifest-path",str(BRIDGE/"Cargo.toml")],
        cwd=ROOT,env=env,stdout=subprocess.DEVNULL,check=True)
    patch=""
    for name in ["src/lib.rs","mninjars.f90","ninjars.h","Cargo.toml","Cargo.lock"]:
        base=ROOT/"vendor/ninjars"/name;new=BRIDGE/name
        patch+="".join(difflib.unified_diff(base.read_text().splitlines(True),new.read_text().splitlines(True),
            fromfile="a/"+name,tofile="b/"+name))
    (DOC/"bridge.patch").write_text(patch)
    save(DOC/"source.json",{"base_git_revision":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "base_files":{str(p.relative_to(ROOT/"vendor/ninjars")):sha(p)for p in [
            ROOT/"vendor/ninjars"/name for name in ["src/lib.rs","src/tensor.rs","mninjars.f90","ninjars.h","Cargo.toml","Cargo.lock"]]},
        "files":{str(p.relative_to(BRIDGE)):sha(p)for p in [BRIDGE/name for name in [
            "src/lib.rs","src/guarded.rs","src/guarded_fenv.c","src/tensor.rs","mninjars.f90",
            "ninjars.h","Cargo.toml","Cargo.lock","build.rs"]]},
        "kernel_provenance":str(KERNEL.relative_to(ROOT)),
        "note":"Experimental additive entry only; ordinary DP function and DpState body are unchanged. Separate compensated state is lazy. Production promotion requires version/capability bump."})
def build():
    env=os.environ.copy();env["CARGO_TARGET_DIR"]=str(WORK/"target")
    env["MAKEFLAGS"]="SHELL="+shutil.which("bash")
    command=["make","-C",str(BRIDGE),"-j2","all"]
    with (DOC/"build.log").open("w")as log:subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(DOC/"build.json",{"command":command,"archive_sha256":sha(BRIDGE/"libninjars.a"),
        "module_sha256":sha(BRIDGE/"mninjars.mod"),"source_manifest_sha256":sha(DOC/"source.json")})
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","build"])
    a=p.parse_args();globals()[a.action]()
