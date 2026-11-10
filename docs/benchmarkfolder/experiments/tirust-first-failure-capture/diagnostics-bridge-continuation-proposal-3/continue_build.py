"""One conditional cached build only; explicit root admission is required.

Original V2 sources/config/native outputs are immutable referenced inputs.
Only independently copied caches and new V3 outputs may be written.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import select
import shlex
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
OLD = ROOT / ".tirust-build/tirust-private-diagnostics-2"
WORK = ROOT / ".tirust-build/tirust-private-diagnostics-3"
V2 = HERE.parent / "diagnostics-bridge-proposal-2"
V2_INDEX = "9c962ee3f7117eba4399d5db3253533ecf92dbcf138964b18b446c363f4ca1d4"
CAP = 17179869184
CPUS = {16, 17}


def require(value, message):
    if not value:raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def packet_check(directory, index_name, expected=None):
    index_path = directory / index_name
    if expected:require(digest(index_path) == expected, "Source index changed")
    index = json.loads(index_path.read_text())
    for name, item in index["files"].items():
        require(Path(name).name == name, "Unsafe source payload")
        path = directory / name
        require(not path.is_symlink() and path.is_file() and path.stat().st_size == item["bytes"] and
                digest(path) == item["sha256"], "Source payload changed: " + name)
    return index


def v2_helpers():
    packet_check(V2, "source-index.json", V2_INDEX)
    sys.path.insert(0, str(V2))
    spec = importlib.util.spec_from_file_location("frozen_tirust_v2_build", V2 / "build.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tree_state(directory):
    """All entries; atime excluded because verification itself reads files."""
    result = {}
    for path in [directory] + sorted(directory.rglob("*")):
        stat = path.lstat()
        entry = {"mode": stat.st_mode, "uid": stat.st_uid, "gid": stat.st_gid,
                 "dev": stat.st_dev, "ino": stat.st_ino, "nlink": stat.st_nlink,
                 "size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns}
        if path.is_symlink():entry.update(kind="symlink", target=os.readlink(path))
        elif path.is_file():entry.update(kind="file", sha256=digest(path))
        elif path.is_dir():entry.update(kind="directory")
        else:raise ValueError("Unexpected original V2 entry")
        result[path.relative_to(directory).as_posix()] = entry
    return result


def import_outputs(inventory, destination):
    for name, expected in inventory.items():
        relative = Path(name)
        require(relative.parts and relative.parts[0] in ("target", "cargo-build") and ".." not in relative.parts,
                "Unsafe imported output")
        source, target = OLD / relative, destination / relative
        require(source.is_file() and not source.is_symlink() and digest(source) == expected, "Imported V2 artifact changed")
        require(not target.exists() and not target.is_symlink(), "Imported destination already exists")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        a, b = source.stat(), target.stat()
        require((a.st_dev, a.st_ino) != (b.st_dev, b.st_ino), "Hard-linked cache import forbidden")
        require(digest(target) == expected and a.st_mtime_ns == b.st_mtime_ns and
                (a.st_mode & 0o7777) == (b.st_mode & 0o7777), "Imported bytes/metadata differ")


def unexpected_activity(line):
    text = line.strip()
    match = re.match(r"(?:Dirty|Compiling)\s+([^\s:]+)", text)
    if match and match.group(1) != "madloop-tirust":return "Dependency rebuild: " + text
    if text.startswith("Running ") and "--crate-name madloop_tirust_private_diag " not in text:
        return "Unexpected Cargo child command: " + text
    return None


def write_json(path, value):
    with path.open("x") as stream:json.dump(value, stream, indent=2, sort_keys=True);stream.write("\n")


def source_cache_check(base, bindings):
    for name, info in bindings["cache_source_roots"].items():
        files = base.inventory(ROOT / info["path"])
        require(len(files) == info["files"] and base.tree_hash(files) == info["sha256"],
                "Pinned dependency source cache changed: " + name)


def main():
    require(sys.argv[1:] == ["build"], "Use: continue_build.py build; root admission required")
    started = time.monotonic();deadline = started + 900
    packet_check(HERE, "source-index.json")
    config = json.loads((HERE / "bindings.json").read_text())
    base = v2_helpers();bindings = base.checked_bindings();source_cache_check(base, bindings)
    require(CPUS <= os.sched_getaffinity(0), "CPUs 16–17 unavailable")
    require(not any(name.startswith("LD_") for name in os.environ), "Outer LD_* injection forbidden")
    for name, expected in config["required_environment"].items():
        require(os.environ.get(name) == expected, "Selected build environment changed: " + name)
    for name, value in os.environ.items():
        if value and (name in config["forbidden_environment"] or name.startswith("CARGO_PROFILE_") or
                      (name.startswith("CARGO_TARGET_") and name != "CARGO_TARGET_DIR")):
            raise ValueError("Unreviewed compiler option: " + name)
    for name, expected in config["additional_file_pins"].items():
        require(digest(Path(name)) == expected, "Compiler/native input changed: " + name)
    before = tree_state(OLD)
    require(canonical_hash(before) == config["original_v2_tree_state_sha256"], "Closed original V2 tree changed")
    inventory_path = ROOT / config["imported_inventory"]["path"]
    require(digest(inventory_path) == config["imported_inventory"]["sha256"], "Imported inventory changed")
    imported = json.loads(inventory_path.read_text())
    require(len(imported) == 1161, "Exactly 1161 imported output files required")
    WORK.mkdir(parents=True, exist_ok=False)
    result = {"status": "started", "source_index_sha256": digest(HERE / "source-index.json"),
              "original_v2_source_index_sha256": V2_INDEX, "imported_files": len(imported),
              "memory_bytes_per_child": CAP, "cpus_nonexclusive": sorted(CPUS), "jobs": 2,
              "maximum_wall_seconds": 900, "private_dso_loaded": False, "tensor_evaluations": 0, "commands": []}
    write_json(WORK / "build-started.json", result)
    environment = os.environ.copy()
    environment.update(CARGO_BUILD_JOBS="2", CARGO_HOME=str(ROOT / ".tirust-build/cargo-home"),
        CARGO_TARGET_DIR=str(WORK / "target"), CARGO_BUILD_BUILD_DIR=str(WORK / "cargo-build"),
        CARGO_NET_OFFLINE="true", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
        CC=bindings["tool_paths"]["cc"], FC=bindings["tool_paths"]["fc"],
        RUSTC=bindings["tool_paths"]["rustc"], RUSTDOC=bindings["tool_paths"]["rustdoc"])

    def limit():
        os.setsid();os.sched_setaffinity(0, CPUS)
        resource.setrlimit(resource.RLIMIT_AS, (CAP, CAP));resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    def run(argv, cwd, cargo=False):
        receipt = {"argv": list(map(str, argv)), "cwd": str(cwd), "cargo_rebuild_guard": cargo}
        result["commands"].append(receipt)
        require(time.monotonic() < deadline-15, "Build deadline before command")
        with (WORK / "build.log").open("ab") as log:
            log.write(("\n$ "+shlex.join(receipt["argv"])+"\n").encode());log.flush()
            process = subprocess.Popen(receipt["argv"], cwd=cwd, env=environment, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, preexec_fn=limit)
            receipt["owned_child_pid_and_group"] = process.pid
            pending = b""
            def signal_group(which):
                try:os.killpg(process.pid, which)
                except ProcessLookupError:pass
            def inspect(raw):
                if cargo:
                    reason = unexpected_activity(raw.decode("utf-8", errors="replace"))
                    if reason:receipt["unexpected_activity"] = reason;raise RuntimeError(reason)
            try:
                while True:
                    remaining = deadline-15-time.monotonic()
                    if remaining <= 0:raise TimeoutError("Cached build deadline")
                    ready, _, _ = select.select([process.stdout], [], [], min(0.25, remaining))
                    if not ready:continue
                    chunk = os.read(process.stdout.fileno(), 65536)
                    if not chunk:break
                    log.write(chunk);log.flush();pending += chunk
                    require(len(pending) <= 1048576, "Compiler output line exceeds 1MiB bound")
                    while b"\n" in pending:
                        line, pending = pending.split(b"\n", 1);inspect(line)
                if pending:inspect(pending)
                receipt["returncode"] = process.wait(timeout=max(0.1, deadline-15-time.monotonic()))
                require(receipt["returncode"] == 0, "Build command failed")
            finally:
                if process.poll() is None:
                    signal_group(signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:pass
                signal_group(signal.SIGKILL)
                process.wait(timeout=5);process.stdout.close()
                receipt["child_terminal_returncode"] = process.returncode
                receipt["owned_process_group_cleanup_attempted"] = True

    try:
        write_json(WORK / "original-v2-before.json", before)
        import_outputs(imported, WORK)
        write_json(WORK / "imported-files.json", imported)
        # Original manifest/config/source paths deliberately stay unchanged.
        run([bindings["tool_paths"]["cargo"], "build", "--locked", "--offline", "--release",
             "--features", "poles-only", "-vv", "--color", "never"], OLD / "bridge", cargo=True)
        private_lib = WORK / "target/release/libmadloop_tirust_private_diag.so"
        require(private_lib.is_file(), "Complete private DSO not produced")
        capture = WORK / "capture";capture.mkdir()
        shutil.copy2(V2 / "private-replay.c", capture / "replay.c")
        shutil.copy2(ROOT / "vendor/tirust/tirust.h", capture / "tirust.h")
        run([bindings["tool_paths"]["cc"], "-std=c11", "-O2", "-fPIC", "-I.", "-I"+str(ROOT),
             "-c", "replay.c", "-o", "capture-replay.o"], capture)
        link = shlex.split(bindings["original_link_command"])[1:]
        link[link.index("-o")+1] = str(capture / "stability-check")
        link[link.index("capture-replay.o")] = str(capture / "capture-replay.o")
        argv = [bindings["tool_paths"]["fc"]]
        for arg in link:
            if arg == "-lmadloop_tirust":argv.extend(["-L"+str(private_lib.parent), "-lmadloop_tirust_private_diag"])
            elif arg.startswith("-Wl,-rpath,") and "tirust" in arg:argv.append("-Wl,-rpath,"+str(private_lib.parent))
            else:argv.append(arg)
        argv.append("-Wl,-Map,"+str(capture / "link.map"));run(argv, capture)
        result.update(status="build_complete_unloaded", artifacts={str(q.relative_to(WORK)):
            {"bytes": q.stat().st_size, "sha256": digest(q)} for q in
            [private_lib, capture / "stability-check", capture / "capture-replay.o", capture / "link.map"]})
    except BaseException as error:
        result.update(status="failed", error={"class": type(error).__name__, "message": str(error)})
        raise
    finally:
        try:
            after = tree_state(OLD);write_json(WORK / "original-v2-after.json", after)
            require(before == after, "Original closed V2 tree changed")
            base.checked_bindings();source_cache_check(base, bindings)
            for name, expected in config["additional_file_pins"].items():
                require(digest(Path(name)) == expected, "Compiler/native input changed")
            result["original_v2_unchanged"] = True
            require(time.monotonic() <= deadline, "Postverification exceeded 900-second deadline")
        except BaseException as error:
            result.update(status="failed", postverification_error={"class": type(error).__name__, "message": str(error)})
            raise
        finally:
            result["elapsed_seconds"] = time.monotonic()-started
            write_json(WORK / "build-terminal.json", result)


if __name__ == "__main__":main()
