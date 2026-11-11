"""Portable data-only verification; no live artifacts, loader or native calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent / "diagnostics-replay-proposal-1"
PROPOSAL_SHA = "6cc9b0f96e2493fcd3f967d3c39f1ae273ab461ae7917bf220ccb504f1387fb5"
V2 = HERE.parent / "diagnostics-bridge-proposal-2"
V2_SHA = "9c962ee3f7117eba4399d5db3253533ecf92dbcf138964b18b446c363f4ca1d4"
PROTOCOL_SHA = "2a1fbd8a68780742ca7f741a7452f5dfd6c29a8f3a1febae029dc84446bd5dea"
WORK = "/shared/external/madloop_tirust/madgraph7_madloop_tirust/.tirust-build/tirust-private-diagnostic-replay-1"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def packet(directory, index_name, expected=None):
    path = directory / index_name
    if expected:
        require(digest(path) == expected, "Referenced packet changed")
    data = json.loads(path.read_text())
    for name, item in data["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe payload name")
        target = directory / name
        require(not target.is_symlink() and target.is_file() and target.stat().st_size == item["bytes"] and
                digest(target) == item["sha256"], "Packet payload changed: " + name)
    return data


def protocol():
    path = V2 / "protocol.py"
    require(digest(path) == PROTOCOL_SHA, "Frozen protocol changed")
    module = types.ModuleType("retained_private_diagnostics_protocol")
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def verify():
    index = packet(HERE, "index.json")
    require({p.relative_to(HERE).as_posix() for p in HERE.rglob("*") if p.is_file()} ==
            set(index["files"]) | {"index.json"}, "Retention membership changed")
    require(index["payload_files"] == len(index["files"]) and
            index["payload_bytes"] == sum(x["bytes"] for x in index["files"].values()), "Payload totals changed")
    packet(PROPOSAL, "source-index.json", PROPOSAL_SHA)
    packet(V2, "source-index.json", V2_SHA)
    terminal, started = read("run/terminal.json"), read("run/started.json")
    tree, identities, observed = read("retained-tree.json"), read("mapped-file-identities.json"), read("postterminal-observation.json")
    require(terminal["status"] == "diagnostic_protocol_passed" and started["status"] == "started", "Terminal outcome changed")
    require(started["admission"] == terminal["admission"] and
            terminal["admission"]["source_index_sha256"] == PROPOSAL_SHA, "Execution admission changed")
    require(terminal["postverification_passed"] and terminal["loader_exact_provider_set_verified"], "Guard or loader failure")
    require(0 < terminal["elapsed_seconds"] == observed["controller_elapsed_seconds"] < 60, "Observed elapsed time changed")
    require(observed["actual_tool_session_id"] == index["actual_tool_session_id"] == 90460 and
            observed["actual_tool_exit_code"] == index["actual_tool_exit_code"] == 0, "Actual tool outcome changed")
    require(terminal["automatic_retry"] is False and observed["native_retries"] == 0, "Retry claim changed")
    for name, record in tree["files"].items():
        target = HERE / "run" / name
        require(digest(target) == record["sha256"] and target.stat().st_size == record["bytes"], "Run tree copy differs")
    require(set(tree["files"]) == {p.relative_to(HERE / "run").as_posix() for p in (HERE / "run").rglob("*") if p.is_file()}, "Run retention incomplete")
    require(tree["files"]["started.json"]["mtime_ns"] < tree["files"]["terminal.json"]["mtime_ns"], "File timestamp order changed")
    qualified = json.loads((PROPOSAL / "static-qualification.json").read_text())
    require(terminal["admission"]["qualified"] == qualified, "Runtime provider admission changed")
    driver = terminal["admission"]["driver"]
    original = json.loads((V2 / "bindings.json").read_text())
    old_bridge = next(name for name in original["original_runtime_providers"] if Path(name).name == "libmadloop_tirust.so")
    require(observed["original_production_dso_sha256"] == original["original_runtime_providers"][old_bridge] and
            old_bridge not in qualified["providers"], "Production bridge replacement or preservation changed")
    expected = dict(qualified["providers"])
    driver_sha = json.loads((PROPOSAL / "bindings.json").read_text())["v3_artifacts"]["capture/stability-check"]["sha256"]
    expected[driver] = driver_sha
    require(len(expected) == 15 and {name: item["sha256"] for name, item in identities.items()} == expected, "Mapped file identities differ")
    require(all(name == item["resolved"] for name, item in identities.items()), "Mapped identity canonical path changed")
    expected_links = {"providers/" + name: {"target": target, "resolved": target, "sha256": qualified["providers"][target]}
                      for name, target in qualified["aliases"].items()}
    require(tree["symlinks"] == expected_links, "Exact aliases changed")
    derived = read("run/derived-inputs.json")
    require(derived["aliases"] == qualified["aliases"], "Derived aliases changed")
    ordered = [WORK + "/inputs/" + item["basename"] for item in original["inputs"]]
    require((HERE / "run/request-manifest.txt").read_text() == "".join(name + "\n" for name in ordered) and
            digest(HERE / "run/request-manifest.txt") == derived["manifest_sha256"], "Exact input manifest changed")
    require(derived["input_copies"] == {name: item["sha256"] for name, item in zip(ordered, original["inputs"])}, "Copied input binding changed")
    parser = protocol()
    for item in original["inputs"]:
        raw = (HERE / "run/inputs" / item["basename"]).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == item["sha256"] and len(raw) == item["bytes"], "Captured input bytes changed")
        require(all(value == item[key] for key, value in parser.input_record(raw).items()), "Captured operand words changed")
    for kind in ("loader", "driver"):
        command = read("run/" + kind + "/command.json")
        require(command == terminal[kind + "_command"], "Actual command receipt changed")
        require(command["returncode"] == 0 and command["error"] is None and command["cleanup"]["survivors"] == [], "Command/cleanup failed")
        require(command["cpu"] == 16 and command["cpu_exclusive"] is False and
                command["memory_bytes"] == 4294967296 and 0 < command["elapsed_seconds"] < command["wall_cap_seconds"] <= 60, "Resource scope changed")
        require(command["target_environment"] == {"LD_LIBRARY_PATH": WORK + "/providers"}, "Target environment changed")
        require(not any(key.startswith("LD_") for key in command["setup_environment"]), "Outer loader injection observed")
        require(digest(HERE / "run" / kind / "stdout.txt") == command["stdout_sha256"], "Command log changed")
        for item in command["runtime_map_samples"]:
            require(digest(HERE / "run" / kind / item["path"]) == item["sha256"], "Map sample changed")
    require(terminal["driver_command"]["target_argv"] == [driver, WORK + "/request-manifest.txt"], "Driver invocation changed")
    require(terminal["loader_command"]["target_argv"] == [terminal["admission"]["loader"], "--inhibit-cache", "--library-path", WORK + "/providers", "--list", driver], "Loader invocation changed")
    aliases = {WORK + "/providers/" + name: target for name, target in qualified["aliases"].items()}
    resolved = set()
    for line in (HERE / "run/loader/stdout.txt").read_text().splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("linux-vdso.so."):
            continue
        if len(fields) >= 3 and fields[1] == "=>":
            require(fields[2] in aliases, "Unregistered loader alias")
            resolved.add(aliases[fields[2]])
        else:
            require(fields[0] == terminal["admission"]["loader"], "Unexpected direct loader provider")
            resolved.add(fields[0])
    require(resolved == set(qualified["providers"]), "Loader set changed")
    samples = []
    for path in sorted((HERE / "run/driver").glob("runtime-maps-*.txt")):
        lines = [line.split(maxsplit=5) for line in path.read_text().splitlines()]
        if not any(len(fields) == 6 and fields[5] == driver for fields in lines):
            continue
        actual = {}
        for fields in lines:
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            name = fields[5]
            require(name in identities and not name.endswith(" (deleted)"), "Unregistered application mapping")
            identity = identities[name]
            require(int(fields[4]) == identity["inode"] and
                    tuple(int(x, 16) for x in fields[3].split(":")) == (identity["device_major"], identity["device_minor"]), "Mapped inode/device differs")
            actual[name] = identity["sha256"]
        require(actual == expected, "Application map is not exact driver+fourteen-provider closure")
        samples.append({"path": path.name, "elf_files": actual, "read_only_locale": {}, "required_application_files_present": True})
    require(len(samples) == 4 and samples == terminal["runtime_binding"]["samples"] and terminal["runtime_binding"]["verified"], "Observed map qualification changed")
    raw = (HERE / "run/tirust-replay.jsonl").read_text()
    require(digest(HERE / "run/tirust-replay.jsonl") == "7711572f21459ab74bbf07e80f4624e6e8510f0cc7333e1f98a0a9a8299a64ce", "Raw diagnostics changed")
    diagnostics = parser.replay_summary(raw, original["inputs"])
    require(diagnostics == terminal["diagnostics"], "Frozen diagnostic reconstruction differs")
    routes = Counter(row["public_metadata"]["last_lane_routes"] for row in diagnostics["evaluations_metadata"])
    require(routes == {"Some([Refused])": 12}, "Refusal routes changed")
    for row in diagnostics["evaluations_metadata"]:
        metadata = row["public_metadata"]
        require(metadata["fallback_used"] == "[true]" and "precision_rescue: []" in metadata["engine_execution"] and
                "expansion_rows: 1, expansion_successes: 0" in metadata["engine_execution"], "Method-choice observations changed")
        for text in ("small-Gram expansion did not converge", "2.5968712461921914e-3", "2.0110026150849244e-3", "modified Cayley determinant cancels (relative rounding inf)"):
            require(text in metadata["last_diagnostics"], "Retained diagnostic string changed")
    require(all(not item["matching_identity_present"] for item in observed["process_identity_observations"]), "Owned process remained present")
    require(observed["common_lock_observation"]["free"] and observed["common_lock_observation"]["released_immediately"], "Post-terminal lock observation differs")
    require(observed["tensor_evaluations"] == 12 and observed["amplitude_evaluations"] == 0 and
            observed["failed_tensor_outputs_read"] == 0 and observed["qp_evaluations_claimed"] is False, "Diagnostic scope changed")
    return {"retained_replay_verified": True, "evaluations": 12, "original_refusals_preserved": 12,
            "full_application_map_samples": len(samples), "runtime_providers": 14,
            "failed_tensor_outputs_read": 0, "loader_or_native_invoked": False,
            "accuracy_or_performance_qualification": False}


if __name__ == "__main__":
    def deny(event, args):
        if event.startswith(("subprocess.", "ctypes.dlopen", "os.exec", "os.spawn", "os.posix_spawn")) or event in ("os.system", "os.fork", "os.forkpty"):
            raise RuntimeError("Native/process execution forbidden in portable verification")
    sys.addaudithook(deny)
    print(json.dumps(verify(), sort_keys=True))
