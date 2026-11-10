"""Portable pure replay of the retained failed build, not an accuracy test."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:raise ValueError(message)


def main():
    index = json.loads((HERE / "index.json").read_text())
    actual = {x.name for x in HERE.iterdir() if x.is_file() and x.name != "index.json"}
    if actual != set(index["files"]):raise ValueError("Failure capsule file set changed")
    for name, expected in index["files"].items():
        path = HERE / name
        if path.is_symlink():raise ValueError("Unexpected symbolic payload")
        raw = path.read_bytes()
        if len(raw) != expected["bytes"] or hashlib.sha256(raw).hexdigest() != expected["sha256"]:
            raise ValueError("Failure payload changed: " + name)
    read = lambda name:json.loads((HERE / name).read_text())
    terminal, summary = read("build-terminal.json"), read("summary.json")
    require(terminal["status"] == "failed" and terminal["commands"][-1]["returncode"] == 101, "Retained failure invariant")
    require(terminal["elapsed_seconds"] == summary["elapsed_seconds"] < 900, "Retained failure invariant")
    require(terminal["memory_bytes_per_build_process"] == 4294967296, "Retained failure invariant")
    require(terminal["source_packet_index_sha256"] == summary["source_packet_index_sha256"] == "755a2d95c1478508ee41c69f11174339440210458e97e4e1eb259de912f3c6e0", "Retained failure invariant")
    require(read("staged-source-files.json") == read("staged-source-files-after.json"), "Retained failure invariant")
    require(read("cache-sources-before.json") == read("cache-sources-after.json"), "Retained failure invariant")
    require(not any(summary["expected_private_artifacts_present"].values()), "Retained failure invariant")
    require(summary["tensor_evaluations"] == 0 and not summary["private_dso_loaded"] and not summary["native_replay_started"], "Retained failure invariant")
    log = (HERE / "build.log").read_text()
    require('failed to spawn work thread: Os { code: 11' in log and 'rustc-LLVM ERROR: out of memory' in log, "Retained failure invariant")
    require('-C opt-level=3 -C codegen-units=1' in log, "Retained failure invariant")
    partial = read("partial-build-file-hashes.json")
    require(len(partial) == summary["partial_build_artifact_files"] == 1130, "Retained failure invariant")
    require(not any(x.endswith("libmadloop_tirust_private_diag.so") for x in partial), "Retained failure invariant")
    print(json.dumps({"failure_replayed": True, "cargo_returncode": 101, "source_cache_unchanged": True,
                      "staged_sources_unchanged": True, "tensor_evaluations": 0, "build_or_loader_invoked": False}, sort_keys=True))


if __name__ == "__main__":main()
