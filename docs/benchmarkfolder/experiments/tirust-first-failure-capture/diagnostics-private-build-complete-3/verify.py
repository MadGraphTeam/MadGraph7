"""Pure verification of retained build evidence; no native execution."""
from pathlib import Path
import hashlib
import json
import re
import sys

HERE = Path(__file__).resolve().parent
SOURCE_SHA = "22f45713e21a2d95d1e0ebd6b718c3af9d8f47e65c0c91d262d6b1eed63d7fea"
V2_SHA = "9c962ee3f7117eba4399d5db3253533ecf92dbcf138964b18b446c363f4ca1d4"
FAILED_V2_SHA = "64d35be39e7d7467bd4e2d7a1477abfee2bc6af429a65b5f3616c45810fc2003"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_packet(directory, index_name, expected=None):
    index_path = directory / index_name
    if expected:
        require(digest(index_path) == expected, "Referenced index changed")
    index = json.loads(index_path.read_text())
    for name, item in index["files"].items():
        require(Path(name).name == name, "Unsafe payload name")
        path = directory / name
        require(not path.is_symlink() and path.is_file(), "Non-file payload")
        require(path.stat().st_size == item["bytes"] and digest(path) == item["sha256"], "Payload changed: " + name)
    require(sum(item["bytes"] for item in index["files"].values()) == index["payload_bytes"], "Payload total changed")
    require(len(index["files"]) == index["payload_files"], "Payload count changed")
    return index


def main():
    index = checked_packet(HERE, "index.json")
    require({p.name for p in HERE.iterdir() if p.is_file()} == set(index["files"]) | {"index.json"}, "Packet file set changed")
    source = HERE.parent / "diagnostics-bridge-continuation-proposal-3"
    checked_packet(source, "source-index.json", SOURCE_SHA)
    old = HERE.parent / "diagnostics-private-build-failed-2"
    checked_packet(old, "index.json", FAILED_V2_SHA)
    read = lambda name: json.loads((HERE / name).read_text())
    terminal, started, observed = read("build-terminal.json"), read("build-started.json"), read("tool-observation.json")
    require(terminal["status"] == observed["controller_terminal_status"] == "build_complete_unloaded", "Build outcome changed")
    require(started["status"] == "started" and started["commands"] == [], "Started receipt changed")
    require(terminal["source_index_sha256"] == started["source_index_sha256"] == SOURCE_SHA, "V3 source pin changed")
    require(terminal["original_v2_source_index_sha256"] == started["original_v2_source_index_sha256"] == V2_SHA, "V2 source pin changed")
    require(terminal["original_v2_unchanged"] is True, "Original tree guard failed")
    before, after = read("original-v2-before.json"), read("original-v2-after.json")
    require(before == after, "Original V2 tree changed")
    canonical = hashlib.sha256(json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    bindings = json.loads((source / "bindings.json").read_text())
    require(canonical == bindings["original_v2_tree_state_sha256"], "V2 tree differs from admitted state")
    require(read("imported-files.json") == json.loads((old / "partial-build-file-hashes.json").read_text()), "Imported artifact identity changed")
    require(len(read("imported-files.json")) == terminal["imported_files"] == 1161, "Import count changed")
    for field, expected in (("jobs", 2), ("cpus_nonexclusive", [16, 17]), ("memory_bytes_per_child", 17179869184), ("maximum_wall_seconds", 900)):
        require(terminal[field] == started[field] == expected, "Resource policy changed: " + field)
    require(0 < terminal["elapsed_seconds"] == observed["controller_elapsed_seconds"] < 900, "Observed elapsed time changed")
    require(observed["actual_tool_session_id"] == index["actual_tool_session_id"] == 12260, "Tool session changed")
    require(observed["actual_tool_exit_code"] == index["actual_tool_exit_code"] == 0, "Tool exit changed")
    commands = terminal["commands"]
    require(len(commands) == 3 and [c["cargo_rebuild_guard"] for c in commands] == [True, False, False], "Command scope changed")
    require([c["returncode"] for c in commands] == observed["command_returncodes"] == [0, 0, 0], "Command outcomes changed")
    require([c["child_terminal_returncode"] for c in commands] == observed["child_terminal_returncodes"] == [0, 0, 0], "Child outcomes changed")
    require(all(c["owned_process_group_cleanup_attempted"] for c in commands), "Cleanup receipt missing")
    require(commands[0]["argv"][1:] == ["build", "--locked", "--offline", "--release", "--features", "poles-only", "-vv", "--color", "never"], "Cargo options changed")
    log = (HERE / "build.log").read_text()
    compiling = []
    running = []
    for raw in log.splitlines():
        line = raw.strip()
        match = re.match(r"(?:Dirty|Compiling)\s+([^\s:]+)", line)
        if match:
            require(match.group(1) == "madloop-tirust", "Unexpected dependency rebuild")
            compiling.append(line)
        if line.startswith("Running "):
            require("--crate-name madloop_tirust_private_diag " in line, "Unexpected Cargo child")
            running.append(line)
    require(compiling and len(running) == 1 and "Finished `release` profile" in log, "Cargo completion evidence missing")
    require(terminal["artifacts"] == observed["artifacts_at_postterminal_observation"], "Artifact observation changed")
    library = terminal["artifacts"]["target/release/libmadloop_tirust_private_diag.so"]
    require(library == {"bytes": 69161040, "sha256": "73abcfb34d01afa09c45f1acb612da0a83d05c8961ec07b3cf2f65a02e94c721"}, "Private library identity changed")
    for item in (terminal, started, observed, index):
        require(item["private_dso_loaded"] is False and item["tensor_evaluations"] == 0, "Invented native evaluation")
    require(not observed["runtime_provider_qualification_performed"] and not observed["static_ELF_qualification_performed"] and not observed["native_replay_started"] and not observed["automatic_retry_performed"], "Scope expanded")
    require(len(observed["limitations"]) == 4, "Reporting limits missing")
    print(json.dumps({"retained_build_verified": True, "command_returncodes": [0, 0, 0], "original_v2_unchanged": True,
                      "private_library_sha256": library["sha256"], "private_dso_loaded": False,
                      "tensor_evaluations": 0, "build_or_loader_invoked": False}, sort_keys=True))


if __name__ == "__main__":
    def deny(event, args):
        if event.startswith(("subprocess.", "ctypes.dlopen", "os.exec", "os.spawn", "os.posix_spawn")) or event in ("os.system", "os.fork", "os.forkpty"):
            raise RuntimeError("Native/process execution forbidden in portable verification")
    sys.addaudithook(deny)
    main()
