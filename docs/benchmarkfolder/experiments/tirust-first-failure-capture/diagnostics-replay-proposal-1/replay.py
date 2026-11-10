"""One separately admitted private diagnostic replay; default is pure preflight."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types

import static_audit as audit

HERE, ROOT = audit.HERE, audit.ROOT
WORK = ROOT / ".tirust-build/tirust-private-diagnostic-replay-1"
PILOT = ROOT / "docs/benchmarkfolder/stability_appendix_c/source5/pilot.py"
PILOT_SHA = "22b06cda2303d9d2fd42aed418aa4518db0d9e834678862330ffa0d6600e9eef"
CLEANUP = ROOT / "docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1/ownership/process_cleanup.py"
CLEANUP_SHA = "240d1067a7b7098f056f0f367097def8f374d7979c3001d249c8b1f7cb0bbd84"
PROTOCOL = audit.V2 / "protocol.py"
PROTOCOL_SHA = "2a1fbd8a68780742ca7f741a7452f5dfd6c29a8f3a1febae029dc84446bd5dea"
require, digest = audit.require, audit.digest


def load(path, expected):
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected, "Frozen helper changed")
    module = types.ModuleType("private_diagnostic_" + path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


def adapted_pilot():
    """Only CPU 12->16 and the required sampled bridge basename change."""
    pilot = load(PILOT, PILOT_SHA)
    source = PILOT.read_text()
    nodes = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    changes = {
        "command": [("'taskset', '-c', '12'", "'taskset', '-c', '16'"),
                    ("'cpu': 12", "'cpu': 16"),
                    ("'libmadloop_tirust.so'", "'libmadloop_tirust_private_diag.so'")],
        "runtime_binding": [("'libmadloop_tirust.so'", "'libmadloop_tirust_private_diag.so'")],
    }
    for name, edits in changes.items():
        code = ast.get_source_segment(source, nodes[name])
        for old, new in edits:
            require(code.count(old) == 1, "Ambiguous private helper adaptation")
            code = code.replace(old, new)
        exec(compile(code, str(HERE / "replay.py") + ":adapted:" + name, "exec"), pilot.__dict__)
    pilot.WORK = WORK
    return pilot


def preflight():
    index = json.loads((HERE / "source-index.json").read_text())
    for name, item in index["files"].items():
        require(Path(name).name == name, "Unsafe proposal payload")
        path = HERE / name
        require(not path.is_symlink() and path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], "Proposal payload changed: " + name)
    audit.packet(audit.V2, "source-index.json", audit.V2_SHA)
    audit.packet(audit.V3, "index.json", audit.V3_SHA)
    observation = json.loads((HERE / "static-observation.json").read_text())
    qualified = audit.qualify(observation)
    require(qualified == json.loads((HERE / "static-qualification.json").read_text()), "Static qualification changed")
    config = json.loads((HERE / "bindings.json").read_text())
    for path, expected in config["additional_file_pins"].items():
        require(digest(path) == expected, "Launch helper or source changed: " + path)
    require(str(Path(sys.executable).resolve()) == config["python"] and
            str(Path(shutil.which("taskset")).resolve()) == config["taskset"], "Launch executable selection changed")
    for name, item in observation["objects"].items():
        require(digest(name) == item["sha256"], "Static artifact/provider changed")
    for name, item in config["v3_artifacts"].items():
        require(digest(audit.WORK / name) == item["sha256"], "V3 artifact changed")
    bindings = json.loads((audit.V2 / "bindings.json").read_text())
    protocol = load(PROTOCOL, PROTOCOL_SHA)
    for item in bindings["inputs"]:
        raw = (ROOT / item["path"]).read_bytes()
        require(len(raw) == item["bytes"] and hashlib.sha256(raw).hexdigest() == item["sha256"], "Frozen captured input changed")
        parsed = protocol.input_record(raw)
        require(all(parsed[key] == item[key] for key in parsed), "Captured operand words changed")
    require(len(bindings["inputs"]) == 6, "Six input ceiling changed")
    require(16 in os.sched_getaffinity(0), "CPU16 unavailable")
    require(WORK.parent.resolve() == WORK.parent and ROOT.resolve() == ROOT, "Private output ancestor traverses an alias")
    require(not any(name.startswith("LD_") for name in os.environ), "Outer LD injection forbidden")
    require(os.environ.get("RUST_MIN_STACK") == "134217728", "Rust stack selection changed")
    for path, expected in ((PILOT, PILOT_SHA), (CLEANUP, CLEANUP_SHA), (PROTOCOL, PROTOCOL_SHA)):
        require(digest(path) == expected, "Frozen execution/parser helper changed")
    adapted_pilot()
    return {"source_index_sha256": digest(HERE / "source-index.json"), "inputs": bindings["inputs"],
            "qualified": qualified, "loader": observation["loader"], "driver": observation["driver"],
            "maximum_tensor_evaluations": 12, "native_execution_started": False}


def verify_aliases(directory, aliases):
    require(directory.is_dir() and not directory.is_symlink(), "Alias directory changed")
    require({path.name for path in directory.iterdir()} == set(aliases), "Alias membership changed")
    for name, target in aliases.items():
        path = directory / name
        require(path.is_symlink() and os.readlink(path) == target and str(path.resolve()) == target, "Exact provider alias changed")


def loader_paths(text):
    require("not found" not in text.lower(), "Loader reported unresolved provider")
    paths = set()
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0].startswith("linux-vdso.so."):
            continue
        if len(fields) >= 3 and fields[1] == "=>" and fields[2].startswith("/"):
            paths.add(str(Path(fields[2]).resolve()))
        elif fields[0].startswith("/"):
            paths.add(str(Path(fields[0]).resolve()))
        else:
            raise ValueError("Unclassified loader output: " + line)
    return paths


def verify_copied_inputs(files, manifest, expected_sha):
    directory = manifest.parent / "inputs"
    require(directory.is_dir() and not directory.is_symlink(), "Input copy directory changed")
    require({str(path) for path in directory.iterdir()} == set(files), "Input copy membership changed")
    for name, expected in files.items():
        path = Path(name)
        require(path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1 and
                digest(path) == expected, "Exact input copy changed")
    require(manifest.is_file() and not manifest.is_symlink() and manifest.stat().st_nlink == 1 and
            digest(manifest) == expected_sha and
            manifest.read_bytes() == "".join(name + "\n" for name in files).encode("ascii"),
            "Exact ordered input manifest changed")


def execute():
    admitted = preflight()
    require(not WORK.exists(), "A closed or partial replay cannot be retried")
    pilot = adapted_pilot()
    cleanup = load(CLEANUP, CLEANUP_SHA)
    protocol = load(PROTOCOL, PROTOCOL_SHA)
    lock_path = ROOT / ".tirust-build/numerical-tests.lock"
    with lock_path.open("a+") as lock:
        # EW owns priority. A busy lock causes no workspace creation or launch.
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        WORK.mkdir(exist_ok=False)
        state = {"status": "started", "admission": admitted, "maximum_wall_seconds": 60,
                 "maximum_tensor_evaluations": 12, "memory_bytes": 4294967296,
                 "cpu_nonexclusive": 16, "lock_path": str(lock_path), "automatic_retry": False,
                 "loader_command": None, "driver_command": None, "numerical_acceptance_claimed": False}
        pilot.save(WORK / "started.json", state)
        aliases = WORK / "providers"
        copied_inputs = {}
        manifest_sha = None
        old_symbolica = os.environ.get("SYMBOLICA_LOCK_DIR")
        try:
            aliases.mkdir()
            for name, target in admitted["qualified"]["aliases"].items():
                (aliases / name).symlink_to(target)
            verify_aliases(aliases, admitted["qualified"]["aliases"])
            inputs_dir = WORK / "inputs"
            inputs_dir.mkdir()
            for item in admitted["inputs"]:
                path = inputs_dir / item["basename"]
                with path.open("xb") as stream:
                    stream.write((ROOT / item["path"]).read_bytes())
                require(digest(path) == item["sha256"], "Copied input changed")
                copied_inputs[str(path)] = item["sha256"]
            manifest = WORK / "request-manifest.txt"
            with manifest.open("x") as stream:
                stream.write("".join(name + "\n" for name in copied_inputs))
            manifest_sha = digest(manifest)
            pilot.save(WORK / "derived-inputs.json", {"aliases": admitted["qualified"]["aliases"],
                       "input_copies": copied_inputs, "manifest_sha256": digest(manifest)})
            os.environ["SYMBOLICA_LOCK_DIR"] = str(WORK / "environment/symbolica-locks")
            Path(os.environ["SYMBOLICA_LOCK_DIR"]).mkdir(parents=True)
            target_environment = {"LD_LIBRARY_PATH": str(aliases)}
            preflight()
            with pilot.signal_requests(cleanup) as signals:
                remaining = 60 - (time.monotonic() - started)
                require(remaining > 20, "Insufficient time before loader check")
                state["loader_command"] = pilot.command(
                    [admitted["loader"], "--inhibit-cache", "--library-path", aliases, "--list", admitted["driver"]],
                    WORK, WORK / "loader", min(20, remaining), cleanup, signals,
                    target_environment=target_environment)
                require(state["loader_command"]["error"] is None, "Strict loader check failed")
                resolved = loader_paths((WORK / "loader/stdout.txt").read_text())
                require(resolved == set(admitted["qualified"]["providers"]), "Actual loader provider set differs")
                state["loader_exact_provider_set_verified"] = True
                preflight()
                verify_aliases(aliases, admitted["qualified"]["aliases"])
                verify_copied_inputs(copied_inputs, manifest, manifest_sha)
                remaining = 60 - (time.monotonic() - started)
                require(remaining > 10, "Insufficient time before numerical launch")
                state["driver_command"] = pilot.command([admitted["driver"], manifest], WORK,
                    WORK / "driver", remaining, cleanup, signals, target_environment=target_environment)
                require(state["driver_command"]["error"] is None, "Diagnostic driver failed")
                state["runtime_binding"] = pilot.runtime_binding(WORK / "driver", admitted["driver"], admitted["qualified"]["providers"])
                require(state["runtime_binding"]["verified"], "No exact sampled runtime closure")
                raw = (WORK / "tirust-replay.jsonl").read_text()
                state["diagnostics"] = protocol.replay_summary(raw, admitted["inputs"])
                state["status"] = "diagnostic_protocol_passed"
        except BaseException as error:
            state.update(status="failed", error={"class": type(error).__name__, "message": str(error)})
            raise
        finally:
            try:
                preflight()
                verify_aliases(aliases, admitted["qualified"]["aliases"])
                if manifest_sha is not None:
                    verify_copied_inputs(copied_inputs, WORK / "request-manifest.txt", manifest_sha)
                state["postverification_passed"] = True
                require(time.monotonic() - started <= 60, "Inclusive replay acceptance deadline exceeded")
            except BaseException as error:
                state.update(status="failed", postverification_error={"class": type(error).__name__, "message": str(error)})
                raise
            finally:
                if old_symbolica is None:
                    os.environ.pop("SYMBOLICA_LOCK_DIR", None)
                else:
                    os.environ["SYMBOLICA_LOCK_DIR"] = old_symbolica
                state["elapsed_seconds"] = time.monotonic() - started
                output = WORK / "tirust-replay.jsonl"
                if output.exists():
                    state["raw_diagnostics"] = {"bytes": output.stat().st_size, "sha256": digest(output)}
                pilot.save(WORK / "terminal.json", state)
                fcntl.flock(lock, fcntl.LOCK_UN)


if __name__ == "__main__":
    require(sys.argv[1:] in ([], ["--execute"]), "Use replay.py [--execute]; root admission required for execution")
    if sys.argv[1:]:
        execute()
    else:
        result = preflight()
        print(json.dumps({"preflight_passed": True, "source_index_sha256": result["source_index_sha256"],
                          "inputs": 6, "maximum_tensor_evaluations": 12, "native_execution_started": False}, sort_keys=True))
