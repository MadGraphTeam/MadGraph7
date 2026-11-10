#!/usr/bin/env python3
"""Verify and reparse retained UTF-8 evidence; never invoke native code."""
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import runpy
import sys
import tempfile

HERE = Path(__file__).resolve().parent
EXECUTION_SHA256 = "51170427b184026ddc0a695a9509e43a647b5067a3227831bf537e184f60b1c3"
RESULTS_SHA256 = "1db3228443e5bcc37d104541e608b62aba36b17000d1c8096bff6d838f6d644f"
SOURCE_INDEX_SHA256 = "f85e28784762a70806398ab5b878f93f01c0a1ca51475d960c50036a9ad7995f"
BUILD_SHA256 = "f27a190f04804e8dc13a9d534e8558ac4eab95c2881cf97cf0aaaefb12260005"


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def path(root, name):
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and ".." not in relative.parts, "Unsafe retained path")
    result = Path(root).joinpath(*relative.parts)
    require(result.resolve() == result.absolute(), "Symlink in retained package")
    return result


def read(file):
    def reject(value):
        raise ValueError("Nonfinite retained JSON: " + value)
    return json.loads(file.read_text(encoding="utf-8"), parse_constant=reject)


def replay(root=HERE):
    root = Path(root).resolve()
    index = read(root / "evidence-index.json")
    # Nested build indices are evidence, only the package's own index is omitted.
    observed = {p.relative_to(root).as_posix(): digest(p.read_bytes())
                for p in root.rglob("*") if p.is_file() and p != root / "evidence-index.json"}
    require(not any(p.is_symlink() for p in root.rglob("*")), "Symlink in retained package")
    require(observed == index["files"], "Package file inventory or SHA-256 mismatch")
    for name, expected in (("execution.json", EXECUTION_SHA256), ("results.json", RESULTS_SHA256),
                           ("frozen/source-index.json", SOURCE_INDEX_SHA256),
                           ("frozen/build1/build.json", BUILD_SHA256)):
        require(digest(path(root, name).read_bytes()) == expected, "Frozen origin hash mismatch: " + name)
    execution = read(root / "execution.json")
    expected_result = read(root / "results.json")
    source_index = read(root / "frozen/source-index.json")
    built = read(root / "frozen/build1/build.json")
    require(execution["complete"] is True and expected_result["passed"] is True, "Original execution did not pass")
    require(execution["source_index_sha256"] == SOURCE_INDEX_SHA256 and execution["build_sha256"] == BUILD_SHA256,
            "Execution ancestry mismatch")
    require(execution["results_sha256"] == RESULTS_SHA256, "Execution result binding mismatch")
    require(built["complete"] is True and built["source_index_sha256"] == SOURCE_INDEX_SHA256, "Build ancestry mismatch")
    require(len(built["commands"]) == 1 and all(item["exit_code"] == 0 for item in built["commands"]), "Build command failure")
    for name, expected in source_index["sources"].items():
        require(digest(path(root / "frozen/source-tree", name).read_bytes()) == expected, "Frozen source mismatch: " + name)
    build_index = read(root / "frozen/build1/evidence-index.json")
    require(digest((root / "frozen/build1/evidence-index.json").read_bytes()) == "f6c4fd59be05127bfe3e98beb7480f61f2802225f31e4834d34ee661a75008a8", "Release build index mismatch")
    binding = read(root / "frozen/build1/bindings.json")
    require(binding["source_index_sha256"] == SOURCE_INDEX_SHA256 and binding["selected_release_sha256"] == "c5de24e6c41a90caf9e34684392d0193588522f26e6bbac646bdf74e5f151cb7", "Release build binding mismatch")
    for name, expected in build_index["files"].items():
        raw = path(root / "frozen/build1", name).read_bytes()
        require(digest(raw) == expected["sha256"] and len(raw) == expected["bytes"], "Retained build evidence mismatch: " + name)
    for name, item in binding["copies"].items():
        raw = path(root / "frozen/build1", name).read_bytes()
        raw = gzip.decompress(raw) if item["gzip"] else raw
        require(digest(raw) == item["raw_sha256"] and len(raw) == item["raw_bytes"], "Retained release link evidence mismatch")
    for file in (root / "frozen/ancestry").rglob("*"):
        if file.is_file():
            name = "docs/benchmarkfolder/experiments/" + file.relative_to(root / "frozen/ancestry").as_posix()
            require(digest(file.read_bytes()) == source_index["dependencies"][name], "Original ancestry changed")
    require(binding["selected_release_sha256"] in execution["workers"]["adaptive"]["runtime_sha256"].values(), "Release DSO not observed in runtime")
    copies = read(root / "copies.json")
    require(set(copies) == set(execution["retained_files"]) | {"execution.json"}, "Missing/extra original runtime files")
    with tempfile.TemporaryDirectory(prefix="adaptive-qp-evidence-") as temporary:
        extracted = Path(temporary)
        for original, item in copies.items():
            packed = path(root, item["stored_path"]).read_bytes()
            require(digest(packed) == item["stored_sha256"], "Packed stream hash mismatch")
            raw = gzip.decompress(packed) if item["encoding"] == "gzip-utf8" else packed
            require(item["encoding"] in ("gzip-utf8", "utf8"), "Unexpected retained encoding")
            raw.decode("utf-8", errors="strict")
            require(not raw.startswith(b"\x7fELF"), "Native binary is outside portable evidence scope")
            expected = EXECUTION_SHA256 if original == "execution.json" else execution["retained_files"][original]
            require(digest(raw) == expected == item["raw_sha256"] and len(raw) == item["raw_bytes"], "Raw stream mismatch: " + original)
            target = path(extracted, original)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        require(set(execution["workers"]) == {"legacy", "adaptive"}, "Unexpected runtime workers")
        for arm, worker in execution["workers"].items():
            require(worker["exit_code"] == worker["loader"]["exit_code"] == 0, "Original worker/loader failed")
            for name, expected in worker["inputs"].items():
                require(digest(path(extracted / arm, name).read_bytes()) == expected, "Runtime input binding mismatch")
            mapping = worker["mapping"]
            require(mapping["expected_runtime_sha256"] == worker["runtime_sha256"] == mapping["mapped_files_sha256_after_exit"],
                    "Original runtime closure changed")
            require(mapping["cpu_allowed_list"] == str(execution["cpu"]), "Original worker affinity mismatch")
            require(digest(mapping["maps_text"].encode("utf-8")) == mapping["maps_sha256"] ==
                    digest((extracted / (arm + "-maps.txt")).read_bytes()), "Original maps binding mismatch")
        previous = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            analyzer = runpy.run_path(str(root / "frozen/source-tree/source/analyze.py"))["analyze"]
            actual = analyzer(extracted)
        finally:
            sys.dont_write_bytecode = previous
        rendered = (json.dumps(actual, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        require(actual == expected_result and digest(rendered) == RESULTS_SHA256, "Frozen analyzer result differs")
    return {"schema": 1, "passed": True, "portable_replay_only": True, "native_execution": False, "bridge_profile": "release",
            "execution_sha256": EXECUTION_SHA256, "results_sha256": RESULTS_SHA256,
            "runtime_files": len(copies), "genuine_binary128_input_components": actual["genuine_binary128_input_components"],
            "adaptive_tensor_calls": actual["workers"]["adaptive"]["tensor_calls"],
            "legacy_tensor_calls": actual["workers"]["legacy"]["tensor_calls"],
            "scope": actual["scope"], "accepted_amplitude_tier": None, "performance_qualified": False}


if __name__ == "__main__":
    require(len(sys.argv) == 1, "Replay takes no arguments and does not execute the native controller")
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
