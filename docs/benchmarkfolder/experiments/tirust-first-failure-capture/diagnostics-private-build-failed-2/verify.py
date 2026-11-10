"""Portable pure replay of V2's timeout; no compilation/loading/evaluation."""
from pathlib import Path
import hashlib
import json

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:raise ValueError(message)


def main():
    index = json.loads((HERE / "index.json").read_text())
    actual = {x.name for x in HERE.iterdir() if x.is_file() and x.name != "index.json"}
    require(actual == set(index["files"]), "Timeout capsule file set changed")
    for name, expected in index["files"].items():
        path = HERE / name
        require(not path.is_symlink(), "Unexpected symbolic payload")
        raw = path.read_bytes()
        require(len(raw) == expected["bytes"] and hashlib.sha256(raw).hexdigest() == expected["sha256"], "Payload changed: " + name)
    read = lambda name:json.loads((HERE / name).read_text())
    terminal, summary = read("build-terminal.json"), read("summary.json")
    require(terminal["status"] == "failed" and terminal["error"]["class"] == "TimeoutExpired", "Original timeout changed")
    require("returncode" not in terminal["commands"][-1] and summary["cargo_returncode_observed"] is None, "Invented Cargo return code")
    require(terminal["elapsed_seconds"] == summary["elapsed_seconds"] < 900, "Elapsed build bound changed")
    require(terminal["memory_bytes_per_build_process"] == 17179869184, "Resource amendment changed")
    require(terminal["source_packet_index_sha256"] == summary["source_packet_index_sha256"] == "9c962ee3f7117eba4399d5db3253533ecf92dbcf138964b18b446c363f4ca1d4", "Source proposal pin changed")
    require(read("staged-source-files.json") == read("staged-source-files-after.json"), "Staged sources changed")
    require(read("cache-sources-before.json") == read("cache-sources-after.json"), "Source cache changed")
    require(not any(summary["expected_private_artifacts_present"].values()), "Unexpected private artifact")
    require(summary["tensor_evaluations"] == 0 and not summary["private_dso_loaded"] and not summary["native_replay_started"], "Invented diagnostic results")
    partial = read("partial-build-file-hashes.json")
    require(len(partial) == summary["partial_build_artifact_files"] == 1161, "Partial artifact inventory changed")
    private = [x for x in partial if "madloop_tirust_private_diag" in Path(x).name]
    require(sorted(private) == summary["private_named_files"] == ["cargo-build/release/deps/madloop_tirust_private_diag.d"], "Private artifact names changed")
    observation = read("private-ELF-absence-observation.json")["files"]
    require(sorted(observation) == summary["all_partial_ELF_ET_DYN_files"], "ELF observation coverage changed")
    for name, item in observation.items():
        require(item["sha256"] == partial[name] and item["required_private_getter_name_bytes_present"] is False, "Private ABI absence observation changed")
    require(not summary["private_ELF_or_complete_DSO_found"], "Private DSO claim changed")
    native_cache = read("gmp-native-cache-observation.json")
    require(native_cache["fresh_Cargo_target_and_build_cache"] is True and native_cache["all_native_dependencies_freshly_rebuilt_claimed"] is False, "Native cache scope changed")
    require("Compiling madloop-tirust v0.1.0" in (HERE / "build.log").read_text(), "Missing private compilation stage")
    print(json.dumps({"timeout_replayed": True, "cargo_returncode": None, "source_cache_unchanged": True,
                      "staged_sources_unchanged": True, "usable_private_DSO_found": False,
                      "tensor_evaluations": 0, "build_or_loader_invoked": False}, sort_keys=True))


if __name__ == "__main__":main()
