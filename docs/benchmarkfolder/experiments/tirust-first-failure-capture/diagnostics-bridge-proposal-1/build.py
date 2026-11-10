"""One private, bounded build. Does not load any DSO or run the replay.

Running this entry point requires the root agent's explicit build admission.
The source packet is safe to inspect, import and test without admission.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import shlex
import shutil
import signal
import subprocess
import time

import bridge_patch
import driver_patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / ".tirust-build/tirust-private-diagnostics-1"
IGNORE = {".git", "target", "references", "__pycache__", ".direnv"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(root):
    files = {}
    for directory, names, leaves in os.walk(root):
        names[:] = sorted(x for x in names if x not in IGNORE)
        for name in sorted(leaves):
            path = Path(directory) / name
            if path.is_file():files[path.relative_to(root).as_posix()] = digest(path)
    return files


def tree_hash(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def checked_bindings():
    bindings = json.loads((HERE / "bindings.json").read_text())
    for filename, expected in bindings["file_pins"].items():
        path = ROOT / filename
        if digest(path) != expected["sha256"] or path.stat().st_size != expected["bytes"]:
            raise ValueError("Pinned input changed: " + filename)
    for name, info in bindings["source_trees"].items():
        actual = inventory(ROOT / info["path"])
        if len(actual) != info["files"] or tree_hash(actual) != info["sha256"]:
            raise ValueError("Pinned source tree changed: " + name)
    for filename, expected in bindings["original_runtime_providers"].items():
        if digest(Path(filename)) != expected:
            raise ValueError("Original runtime provider changed: " + filename)
    return bindings


def checked_source_packet():
    index = json.loads((HERE / "source-index.json").read_text())
    for name, expected in index["files"].items():
        path = HERE / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or digest(path) != expected["sha256"]:
            raise ValueError("Private proposal payload changed: " + name)


def write_json(path, value):
    with path.open("x") as stream:json.dump(value, stream, indent=2, sort_keys=True);stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build"])
    parser.parse_args()
    started = time.monotonic()
    checked_source_packet()
    bindings = checked_bindings()
    cpus = {16, 17}
    if not cpus <= os.sched_getaffinity(0):raise ValueError("Proposed nonexclusive CPUs16–17 unavailable")
    if any(name.startswith("LD_") for name in os.environ):raise ValueError("Outer LD_* environment is forbidden")
    for name, value in os.environ.items():
        if value and (name in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS", "CARGO_BUILD_TARGET",
                              "RUSTC_WRAPPER", "RUSTC_WORKSPACE_WRAPPER", "CARGO_BUILD_RUSTC_WRAPPER",
                              "CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER", "CFLAGS", "CXXFLAGS", "CPPFLAGS",
                              "FFLAGS", "FCFLAGS", "LDFLAGS", "NIX_CFLAGS_COMPILE", "NIX_LDFLAGS") or
                      name.startswith("CARGO_PROFILE_") or (name.startswith("CARGO_TARGET_") and name != "CARGO_TARGET_DIR")):
            raise ValueError("Unreviewed compiler option: " + name)
    WORK.mkdir(parents=True, exist_ok=False)
    result = {"status": "started", "native_replay_started": False, "private_dso_loaded": False,
              "maximum_wall_seconds": 900, "memory_bytes_per_build_process": 4294967296,
              "cpus_nonexclusive": sorted(cpus), "source_packet_index_sha256": digest(HERE / "source-index.json"),
              "commands": []}
    write_json(WORK / "build-started.json", result)
    deadline = started + 900
    environment = os.environ.copy()
    environment.update(CARGO_BUILD_JOBS="2", CARGO_HOME=str(ROOT / ".tirust-build/cargo-home"),
                       CARGO_TARGET_DIR=str(WORK / "target"), CARGO_BUILD_BUILD_DIR=str(WORK / "cargo-build"),
                       CARGO_NET_OFFLINE="true", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       CC=bindings["tool_paths"]["cc"], FC=bindings["tool_paths"]["fc"],
                       RUSTC=bindings["tool_paths"]["rustc"], RUSTDOC=bindings["tool_paths"]["rustdoc"])

    def limit():
        os.setsid()
        os.sched_setaffinity(0, cpus)
        resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def run(argv, cwd):
        receipt = {"argv": list(map(str, argv)), "cwd": str(cwd)}
        result["commands"].append(receipt)
        remaining = deadline-time.monotonic()
        if remaining <= 15:raise TimeoutError("Private build deadline before next command")
        with (WORK / "build.log").open("a") as log:
            log.write("\n$ " + shlex.join(receipt["argv"]) + "\n");log.flush()
            process = subprocess.Popen(receipt["argv"], cwd=cwd, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, preexec_fn=limit)
            def signal_group(which):
                try:os.killpg(process.pid, which)
                except ProcessLookupError:pass
            try:receipt["returncode"] = process.wait(timeout=remaining-15)
            except BaseException:
                signal_group(signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:pass
                # The leader may exit before an owned compiler grandchild.
                # Always signal the owned group, then reap its leader.
                signal_group(signal.SIGKILL)
                process.wait(timeout=5)
                raise
            finally:
                signal_group(signal.SIGKILL)
                receipt["owned_process_group_cleanup_attempted"] = True
        if receipt["returncode"]:raise RuntimeError("Private build command failed")

    try:
        # Preserve a complete source-cache inventory before/after build. Cargo
        # may create its own locks; no cached compiled artifacts are reused.
        cache_before = {name: inventory(ROOT / info["path"]) for name, info in bindings["cache_source_roots"].items()}
        for name, files in cache_before.items():
            if tree_hash(files) != bindings["cache_source_roots"][name]["sha256"]:
                raise ValueError("Locked source cache changed: " + name)
        write_json(WORK / "cache-sources-before.json", cache_before)
        source = WORK / "sources/tirust"
        shutil.copytree(ROOT / bindings["source_trees"]["tirust_pristine"]["path"], source,
                        ignore=shutil.ignore_patterns(*IGNORE))
        kernels = WORK / "sources/tirust/madloop-pole-kernels"
        shutil.copytree(ROOT / bindings["source_trees"]["pole_kernels"]["path"], kernels,
                        ignore=shutil.ignore_patterns(*IGNORE))
        run([bindings["tool_paths"]["patch"], "-p1", "--batch", "--forward", "--fuzz=0", "-i",
             str(ROOT / "vendor/tirust/patches/0001-direct-poles-only.patch")], source)
        bridge = WORK / "bridge"
        (bridge / "src").mkdir(parents=True)
        (bridge / ".cargo").mkdir()
        (bridge / "src/lib.rs").write_bytes(bridge_patch.prepare((ROOT / bindings["base_bridge_source"]).read_bytes()))
        (bridge / "Cargo.toml").write_bytes((HERE / "private-Cargo.toml").read_bytes())
        shutil.copy2(ROOT / bindings["resolved_cargo_lock"], bridge / "Cargo.lock")
        (bridge / ".cargo/config.toml").write_text('[patch."https://github.com/huashengshao/TIRust"]\n'+
            "".join(name+' = { path = '+json.dumps(str(source / "crates" / name))+' }\n' for name in ("iregi-rust", "iregi-masters")))
        staged_before = {"tirust": inventory(source), "bridge": inventory(bridge)}
        write_json(WORK / "staged-source-files.json", staged_before)
        run([bindings["tool_paths"]["cargo"], "build", "--locked", "--offline", "--release", "--features", "poles-only"], bridge)
        if digest(bridge / "Cargo.lock") != bindings["file_pins"][bindings["resolved_cargo_lock"]]["sha256"]:
            raise ValueError("Cargo changed the frozen resolved lock")
        private_lib = WORK / "target/release/libmadloop_tirust_private_diag.so"
        if not private_lib.is_file():raise ValueError("Unique private DSO was not produced")
        capture = WORK / "capture"
        capture.mkdir()
        (capture / "replay.c").write_bytes(driver_patch.prepare((ROOT / bindings["base_driver_source"]).read_bytes()))
        shutil.copy2(ROOT / "vendor/tirust/tirust.h", capture / "tirust.h")
        run([bindings["tool_paths"]["cc"], "-std=c11", "-O2", "-fPIC", "-I.", "-I"+str(ROOT),
             "-c", "replay.c", "-o", "capture-replay.o"], capture)
        old = shlex.split(bindings["original_link_command"])
        new = [bindings["tool_paths"]["fc"]]
        old = old[1:]
        old[old.index("-o")+1] = str(capture / "stability-check")
        old[old.index("capture-replay.o")] = str(capture / "capture-replay.o")
        for arg in old:
            if arg == "-lmadloop_tirust":new.extend(["-L"+str(private_lib.parent), "-lmadloop_tirust_private_diag"])
            elif arg.startswith("-Wl,-rpath,") and "tirust" in arg:new.append("-Wl,-rpath,"+str(private_lib.parent))
            else:new.append(arg)
        new.append("-Wl,-Map,"+str(capture / "link.map"))
        run(new, capture)
        staged_after = {"tirust": inventory(source), "bridge": inventory(bridge)}
        write_json(WORK / "staged-source-files-after.json", staged_after)
        if staged_before != staged_after:raise ValueError("Staged numerical sources changed during private build")
        cache_after = {name: inventory(ROOT / info["path"]) for name, info in bindings["cache_source_roots"].items()}
        write_json(WORK / "cache-sources-after.json", cache_after)
        if cache_after != cache_before:raise ValueError("Locked source cache changed during private build")
        checked_bindings()
        if time.monotonic() > deadline:raise TimeoutError("Private build/postverification exceeded 900-second deadline")
        result.update(status="build_complete_unloaded", postlink_artifacts={
            str(path.relative_to(WORK)): {"bytes": path.stat().st_size, "sha256": digest(path)}
            for path in (private_lib, capture / "stability-check", capture / "capture-replay.o", capture / "link.map")})
    except BaseException as error:
        result.update(status="failed", error={"class": type(error).__name__, "message": str(error)})
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        write_json(WORK / "build-terminal.json", result)


if __name__ == "__main__":main()
