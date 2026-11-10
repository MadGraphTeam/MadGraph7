#!/usr/bin/env python3
"""Explicitly compile/run isolated installed-library workers and bind evidence.

Run only after the coordinating installer agent authorizes native execution.
Binaries are created under --output, evidence under --report; installed prefixes
are read-only inputs.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import time

from validate import qualify

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")


def identity(prefix):
    paths=[prefix/"lib"/name for name in ("libmadloop_tirust.so","libtirust.a","mtirust.mod",
        "tirust-native-libs.txt","tirust-link-before.txt")]
    paths += [prefix/"lib/tirust-config.json",prefix/"include/mtirust.f90",
              prefix/"Cargo.toml",prefix/"Cargo.lock"]
    for path in paths:
        if not path.is_file():
            raise RuntimeError(f"Missing installed input: {path}")
    return {str(path):dict(sha256=digest(path),bytes=path.stat().st_size) for path in paths}


def runtime_identity(ldd_text):
    paths=[]
    for line in ldd_text.splitlines():
        text=line.strip()
        if not text:
            continue
        candidate=text.split("=>",1)[1].strip().split()[0] if "=>" in text else text.split()[0]
        if candidate.startswith("/"):
            path=Path(candidate)
            if not path.is_file():
                raise RuntimeError("Missing resolved runtime dependency: "+candidate)
            paths.append(path)
        elif not candidate.startswith("linux-vdso"):
            raise RuntimeError("Unresolved or unknown ldd entry: "+text)
    return {str(p):dict(resolved=str(p.resolve()),sha256=digest(p),bytes=p.stat().st_size) for p in paths}


def command(argv, log, env, timeout):
    start=now()
    monotonic=time.monotonic()
    timed_out=False
    with log.open("wb") as output:
        process=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            returncode=process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out=True
            os.killpg(process.pid,signal.SIGTERM)
            try:
                returncode=process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                returncode=process.wait()
    record=dict(argv=list(map(str,argv)),cwd=str(ROOT),started_utc=start,finished_utc=now(),
        elapsed_seconds=time.monotonic()-monotonic,timeout_seconds=timeout,timed_out=timed_out,
        returncode=returncode,log=str(log),log_sha256=digest(log))
    dump(log.with_suffix(log.suffix+".json"),record)
    if returncode or timed_out:
        raise RuntimeError(f"Command failed or timed out; see {log}")
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-prefix",type=Path,required=True)
    parser.add_argument("--new-prefix",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--report",type=Path,default=ROOT/"docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache/success-parity")
    parser.add_argument("--compiler",default="gfortran")
    parser.add_argument("--timeout",type=int,default=300,help="Hard per-worker deadline in seconds")
    parser.add_argument("--lock",type=Path,default=ROOT/".tirust-build/numerical-tests.lock")
    args=parser.parse_args()
    if Path.cwd().resolve()!=ROOT:
        parser.error("Run from the main MadGraph repository root")
    if args.timeout<1:
        parser.error("timeout must be positive")
    prefixes={"old":args.old_prefix.resolve(),"new":args.new_prefix.resolve()}
    if prefixes["old"]==prefixes["new"]:
        parser.error("Old and new installations must be distinct")
    work=args.output.resolve()
    if not work.is_relative_to(HERE) or any(work.is_relative_to(p) for p in prefixes.values()):
        parser.error("Output must be a new directory under this isolated protocol directory")
    output=args.report.resolve()
    expected_report=ROOT/"docs/benchmarkfolder/qualification/tirust-install/qp-transport-cache/success-parity"
    if output!=expected_report:
        parser.error("Report must be the assigned new success-parity evidence directory")
    work.mkdir(parents=True,exist_ok=False)
    output.mkdir(parents=True,exist_ok=False)
    specification=json.loads((HERE/"specification.json").read_text())
    source_fixture=ROOT/specification["source_fixture_path"]
    if digest(source_fixture)!=specification["source_fixture_sha256"]:
        raise RuntimeError("External frozen fixture source changed")
    if digest(HERE/"fixtures.dat")!=specification["input_sha256"]:
        raise RuntimeError("Frozen fixture stream changed")
    protocols={name:digest(HERE/name) for name in ("driver.f90","prepare.py","validate.py","run.py",
                                                "fixtures.dat","specification.json","README.md")}
    frozen=json.loads((HERE/"freeze.json").read_text())
    if protocols!=frozen["files"]:
        raise RuntimeError("Protocol differs from the reviewed source freeze")
    (output/"protocol-sources").mkdir()
    for name in (*protocols,"freeze.json"):
        shutil.copy2(HERE/name,output/"protocol-sources"/name)
    shutil.copy2(source_fixture,output/"protocol-sources"/source_fixture.name)
    before={label:identity(prefix) for label,prefix in prefixes.items()}
    dump(output/"inputs-before.json",dict(protocols=protocols,installations=before))
    env=os.environ.copy()
    # SONAME resolution must come from each explicit installed prefix.
    for name in ("LD_LIBRARY_PATH","LD_PRELOAD","LD_AUDIT","GFORTRAN_CONVERT_UNIT"):
        env.pop(name,None)
    compiler=shutil.which(args.compiler, path=env.get("PATH"))
    if not compiler:
        raise RuntimeError("Fortran compiler not found")
    metadata=dict(scope=specification["scope"],started_utc=now(),platform=platform.platform(),
        python=sys.version,byteorder=sys.byteorder,cpu_affinity=sorted(os.sched_getaffinity(0)),
        compiler=str(Path(compiler).resolve()),compiler_sha256=digest(Path(compiler)),
        prefixes={k:str(v) for k,v in prefixes.items()},timeout_seconds=args.timeout,
        native_lock=str(args.lock.resolve()),environment={key:env.get(key) for key in
            ("PATH","LIBRARY_PATH","NIX_LDFLAGS","NIX_CFLAGS_COMPILE","LD_LIBRARY_PATH")},workers={})
    dump(output/"execution.json",metadata)
    try:
        command([compiler,"--version"],output/"compiler-version.txt",env,30)
        for label,prefix in prefixes.items():
            directory=output/label
            directory.mkdir()
            (work/label).mkdir()
            executable=work/label/"driver"
            lib=prefix/"lib"
            before_flags=shlex.split((lib/"tirust-link-before.txt").read_text())
            native_flags=shlex.split((lib/"tirust-native-libs.txt").read_text())
            if before_flags!=["-lgcc_s"] or native_flags!=["-lmadloop_tirust"]:
                raise RuntimeError("Unexpected installed link contract")
            argv=[compiler,"-O2","-fno-fast-math","-frounding-math","-I"+str(lib),
                str(HERE/"driver.f90"),*before_flags,str(lib/"libtirust.a"),
                str(lib/"libmadloop_tirust.so"),"-Wl,-rpath,"+str(lib),"-o",str(executable)]
            build=command(argv,directory/"build.log",env,120)
            linkage=command(["ldd",str(executable)],directory/"ldd.log",env,30)
            ldd=(directory/"ldd.log").read_text()
            resolved=[line.split("=>",1)[1].split("(",1)[0].strip()
                      for line in ldd.splitlines() if "libmadloop_tirust.so =>" in line]
            if resolved!=[str(lib/"libmadloop_tirust.so")]:
                raise RuntimeError("Worker does not resolve its own installed TIRust DSO")
            metadata["workers"][label]=dict(build=build,linkage=linkage,executable_sha256=digest(executable),
                resolved_tirust_dso=resolved[0],resolved_tirust_dso_sha256=digest(Path(resolved[0])),
                runtime_before=runtime_identity(ldd))
            dump(output/"execution.json",metadata)
        metadata["lock_wait_started_utc"]=now()
        dump(output/"execution.json",metadata)
        with args.lock.open("a+") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            metadata["lock_acquired_utc"]=now()
            dump(output/"execution.json",metadata)
            for label in prefixes:
                directory=output/label
                if identity(prefixes[label])!=before[label]:
                    raise RuntimeError("Installed inputs changed before execution")
                run=command([str(work/label/"driver"),str(HERE/"fixtures.dat"),str(directory/"frames.txt")],
                    directory/"run.log",env,args.timeout)
                text=(directory/"frames.txt").read_text()
                qualification=qualify(text,specification)
                dump(directory/"qualification.json",qualification)
                metadata["workers"][label].update(run=run,frames_sha256=digest(directory/"frames.txt"),
                    qualification_sha256=digest(directory/"qualification.json"))
                dump(output/"execution.json",metadata)
            metadata["lock_released_utc"]=now()
        after={label:identity(prefix) for label,prefix in prefixes.items()}
        current_protocols={name:digest(HERE/name) for name in protocols}
        executables_unchanged=True
        runtimes_unchanged=True
        for label in prefixes:
            worker=metadata["workers"][label]
            worker["executable_after_sha256"]=digest(work/label/"driver")
            executables_unchanged &= worker["executable_after_sha256"]==worker["executable_sha256"]
            worker["runtime_after"]=runtime_identity((output/label/"ldd.log").read_text())
            runtimes_unchanged &= worker["runtime_before"]==worker["runtime_after"]
        dump(output/"inputs-after.json",dict(protocols=current_protocols,installations=after))
        old_data=(output/"old/frames.txt").read_bytes()
        new_data=(output/"new/frames.txt").read_bytes()
        exact=old_data==new_data
        qualifications={label:json.loads((output/label/"qualification.json").read_text()) for label in prefixes}
        source_fixture_unchanged=digest(source_fixture)==specification["source_fixture_sha256"]
        results=dict(passed=exact and before==after and protocols==current_protocols and executables_unchanged and
            runtimes_unchanged and source_fixture_unchanged and
            all(q["passed"] for q in qualifications.values()),scope=specification["scope"],
            calls_per_installation=specification["expected_calls"],exact_frame_bytes_equal=exact,
            installations_unchanged=before==after,protocols_unchanged=protocols==current_protocols,
            executables_unchanged=executables_unchanged,
            resolved_runtimes_unchanged=runtimes_unchanged,source_fixture_unchanged=source_fixture_unchanged,
            per_installation={label:{key:q[key] for key in ("passed","calls","status_counts")}
                              for label,q in qualifications.items()},
            old_frames_sha256=digest(output/"old/frames.txt"),new_frames_sha256=digest(output/"new/frames.txt"))
        if not exact:
            old_lines=old_data.decode().splitlines()
            new_lines=new_data.decode().splitlines()
            results["first_different_line"]=next((i+1 for i,(a,b) in enumerate(zip(old_lines,new_lines)) if a!=b),
                                                 min(len(old_lines),len(new_lines))+1)
        results["state"]="complete" if results["passed"] else "failed"
        dump(output/"results.json",results)
        metadata["finished_utc"]=now()
        metadata["passed"]=results["passed"]
        dump(output/"execution.json",metadata)
        if not results["passed"]:
            raise RuntimeError("Installed-library parity qualification failed")
        print(json.dumps(results,indent=2))
    except BaseException as exc:
        metadata["finished_utc"]=now()
        metadata["failure"]=f"{type(exc).__name__}: {exc}"
        dump(output/"execution.json",metadata)
        raise


if __name__=="__main__":
    main()
