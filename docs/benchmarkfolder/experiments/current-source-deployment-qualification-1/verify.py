"""Pure retained-evidence verification; external byte rehashing is explicit."""
import argparse
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
EXPECTED = {
    "proposal/source-index.json": "3b8ffa3555351ce73e98d1103514944bce6d1e9eff4c6d519b609526649782bd",
    "build/build-result.json": "8b54b3ecd0e8daf7f670fc883b26913ccde810dc39a3d6c819cba975c9ffbd58",
    "build/build-admitted.json": "75ab86f7a9acc9c0a0359b3aa2a7c9293a8d36ef6fd74f8f1768cfe8970da49b",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(name):
    return json.loads((HERE / name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-build-root", type=Path)
    parser.add_argument("--external-providers", action="store_true")
    args = parser.parse_args()
    require(not args.external_providers or args.external_build_root is not None,
            "Provider rehash requires the explicit external build root")
    index = read("evidence-index.json")
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()
              and p.name != "evidence-index.json"}
    require(actual == set(index["files"]), "Packet file set differs from its index")
    for name, checksum in index["files"].items():
        path = HERE / name
        require(path.resolve().is_relative_to(HERE) and not path.is_symlink(),
                "Packet path escapes or uses an unrecorded symlink: " + name)
        require(digest(path) == checksum, "Retained bytes changed: " + name)
        with path.open("rb") as stream:
            magic = stream.read(8)
        require(not magic.startswith((b"\x7fELF", b"!<arch>")), "Native binary in packet")
    for name, checksum in EXPECTED.items():
        require(digest(HERE / name) == checksum, "Original receipt/source seal changed")

    summary = read("qualification-summary.json")
    origins = read("origins.json")
    for name, record in origins.items():
        require(digest(HERE / name) == record["sha256"], "Retained origin hash mismatch")
    source = read("proposal/source-index.json")
    require(len(source["files"]) == 12 and len(source["inputs"]) == 9, "Source seal shape")
    for name, checksum in source["files"].items():
        require(digest(HERE / "proposal" / name) == checksum, "Proposal seal mismatch")
    require(set(summary["retained_inputs"]) == set(source["inputs"]), "Missing sealed input")
    omitted_inputs = []
    for name, record in summary["retained_inputs"].items():
        require(record["sha256"] == source["inputs"][name], "Input seal mismatch")
        if record["retained"]:
            require(digest(HERE / record["path"]) == record["sha256"], "Retained input mismatch")
        else:
            omitted_inputs.append(name)
    require(len(omitted_inputs) == 1 and omitted_inputs[0].endswith("/source.tar"),
            "Unexpected omitted source inputs")

    result = read("build/build-result.json")
    admitted = read("build/build-admitted.json")
    before = read("build/before-relocation.json")
    relocation = read("build/relocation.json")
    plan = read("proposal/commands.json")
    start = read("build/build-start.json")
    require(start["source_index_sha256"] == EXPECTED["proposal/source-index.json"] and
            start["commands_sha256"] == digest(HERE / "proposal/commands.json"), "Build start binding")
    require(result["passed"] and result["protected_unchanged"] and admitted["passed"],
            "Closed build did not pass")
    require(result["native_evaluations"] == admitted["native_evaluations"] ==
            summary["native_evaluations"] == 0, "Unexpected native evaluation claim")
    require(admitted["revision"] == summary["revision"] ==
            "0fcd2cd873d61d134837e1d9e87afb13a47824fd", "Wrong implementation revision")
    require(admitted["exporter_inventory"] == summary["exporter_inventory"] ==
            "afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6", "Wrong exporter")
    stages = result["stages"]
    require(len(stages) == summary["stage_count"] == 15, "Stage count")
    child_stages = [s for s in stages if "command" in s]
    require(len(child_stages) == 12 and len(stages) - len(child_stages) == 3, "Stage kinds")
    expected_commands = plan["export_commands"] + plan["build_commands"]
    for actual_stage, expected_stage in zip(child_stages[:8], expected_commands, strict=True):
        for key in ["label", "command", "cwd", "cap_seconds"]:
            require(actual_stage[key] == expected_stage[key], "Actual command differs from plan")
    for stage in stages:
        require(stage["status"] == "passed", "Failed stage")
        if "command" not in stage:
            require(stage["kind"] == "pure file validation/move", "Unknown pure stage")
            continue
        prefix = "build/stages/" + stage["label"]
        require(read(prefix + "/receipt.json") == stage, "Stage/aggregate receipt mismatch")
        require(stage["returncode"] == 0 and not stage["cleanup"]["survivors"], "Child failure")
        require(digest(HERE / prefix / "console.txt") == stage["console_sha256"], "Stage log mismatch")
    require(relocation["source_prefixes_removed"] and
            not relocation["original_native_installations_removed"] and
            relocation["outputs_moved"] == ["standalone", "nlo"], "Relocation receipt mismatch")

    observed = {}
    spellings = read("provider-spellings.json")["aliases"]
    observed_spellings = set()
    for label, export in admitted["exports"].items():
        require(export["ldd_initialized_scope"] is False, "Unproven loader initialization claim")
        inventory = export["inventory"]
        for name, artifact in before[label]["copied_native"].items():
            require(inventory["lib/" + name]["sha256"] == artifact["sha256"] and
                    inventory["lib/" + name]["bytes"] == artifact["bytes"], "Copied native identity mismatch")
        require(before[label]["copied_native"]["libninjars.a"] == {
            "sha256": "0edfa89d7d83161a6ce3fdf0bb1fbca928b73d2b5ec44211a3b2efa186a155d6",
            "bytes": 298293528}, "NinjaRS archive changed")
        require(inventory["lib/libmadloop_tirust.so"]["sha256"] ==
                "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9", "TIRust DSO changed")
        raw = (HERE / "build/stages" / (label + "-readelf") / "console.txt").read_text()
        require("$ORIGIN" in raw and "/source-prefixes/" not in raw, "Recorded RPATH mismatch")
        raw = (HERE / "build/stages" / (label + "-ldd") / "console.txt").read_text()
        require("not found" not in raw, "Unresolved recorded provider")
        for line in raw.splitlines():
            match = re.search(r"(?:=>\s*)?(/\S+)\s+\(", line)
            if match:
                # Historical ldd spelling can be an alias. Preserve it without
                # opening the original host path during portable verification.
                name = match.group(1)
                require(name in spellings, "Unbound recorded provider spelling: " + name)
                alias = spellings[name]
                require(admitted["runtime_providers"].get(alias["resolved_path"]) == alias["sha256"],
                        "Provider alias differs from bound provider")
                observed[alias["resolved_path"]] = alias["sha256"]
                observed_spellings.add(name)
        expected_card = "/Cards/MadLoopParams.dat" if label == "standalone" else "/SubProcesses/MadLoopParams.dat"
        require(export["runtime_card"].endswith(expected_card), "Wrong runtime card location")
        require(admitted["files"][export["runtime_card"]] ==
                "e782503a95c9a709fd1f7a28a737f44e016e9574c6799668e393287da05f0d92", "Initial card changed")
        for name, key in [("check", "executable_sha256"), ("check_sa.f", "check_sa_sha256")]:
            require(admitted["files"][export["process"] + "/" + name] == export[key], "Driver binding")
    require(observed == admitted["runtime_providers"], "Provider receipt does not match raw ldd")
    require(observed_spellings == set(spellings), "Unobserved provider alias")
    require(len(admitted["files"]) == summary["bound_artifact_path_count"] == 1871, "Artifact count")
    original_root = Path(summary["source_directory"])
    for retained, item in summary["selected_text"].items():
        relative = Path(item["original_path"]).relative_to(original_root / "relocated")
        label, rest = relative.parts[0], str(Path(*relative.parts[1:]))
        expected = (admitted["files"][item["original_path"]] if item["receipt_hash_source"] == "bound_artifact_map"
                    else admitted["exports"][label]["inventory"][rest]["sha256"])
        require(digest(HERE / retained) == item["sha256"] == expected, "Generated text binding")

    checked_artifacts = checked_providers = 0
    if args.external_build_root is not None:
        external = args.external_build_root.resolve(strict=True)
        files = dict(admitted["files"])
        files.update({r["original_path"]: r["sha256"] for r in summary["selected_text"].values()})
        for original, checksum in files.items():
            path = external / Path(original).relative_to(original_root)
            require(path.resolve().is_relative_to(external / "relocated"), "External artifact escape")
            require(digest(path) == checksum, "External artifact changed: " + str(path))
            checked_artifacts += 1
        if args.external_providers:
            for original, checksum in admitted["runtime_providers"].items():
                path = Path(original)
                if path.is_relative_to(original_root):
                    path = external / path.relative_to(original_root)
                else:
                    require(str(path).startswith("/nix/store/"), "Unexpected external provider")
                require(digest(path) == checksum, "External provider changed: " + str(path))
                checked_providers += 1
    print(json.dumps({"retained_receipts_verified": True, "retained_files": len(index["files"]),
                      "recorded_build_stages": 15, "recorded_native_evaluations": 0,
                      "external_artifact_paths_rehashed": checked_artifacts,
                      "external_provider_paths_rehashed": checked_providers,
                      "source_archive_bytes_rehashed": False,
                      "runtime_initialization_or_accuracy_verified": False}, indent=2))


if __name__ == "__main__":
    main()
