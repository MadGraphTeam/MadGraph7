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
EXECUTION_SHA256 = "2fe072df782064edb9fb4f435d96e17041e20c211d178a79cd094175aaf6aa48"
RESULTS_SHA256 = "1db3228443e5bcc37d104541e608b62aba36b17000d1c8096bff6d838f6d644f"
SOURCE_INDEX_SHA256 = "80f83d5384ada33b609ac94bb9f9be935eb92e56ee706f979494e4d053fcc9b5"
BUILD_SHA256 = "a213c73aadab7d898e1a3699ab7c1c4ae09dae8afa1c7916a8e6671ebfb14b15"


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
    require(len(built["commands"]) == 7 and all(item["exit_code"] == 0 for item in built["commands"]), "Build command failure")
    for name, expected in source_index["sources"].items():
        require(digest(path(root / "frozen/source-tree", name).read_bytes()) == expected, "Frozen source mismatch: " + name)
    build_index = read(root / "frozen/build1/evidence-index.json")
    require(build_index["source_index_sha256"] == SOURCE_INDEX_SHA256, "Build index ancestry mismatch")
    for name, expected in build_index["files"].items():
        require(digest(path(root / "frozen/build1", name).read_bytes()) == expected, "Retained build evidence mismatch: " + name)
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
    return {"schema": 1, "passed": True, "portable_replay_only": True, "native_execution": False,
            "execution_sha256": EXECUTION_SHA256, "results_sha256": RESULTS_SHA256,
            "runtime_files": len(copies), "genuine_binary128_input_components": actual["genuine_binary128_input_components"],
            "adaptive_tensor_calls": actual["workers"]["adaptive"]["tensor_calls"],
            "legacy_tensor_calls": actual["workers"]["legacy"]["tensor_calls"],
            "scope": actual["scope"], "accepted_amplitude_tier": None, "performance_qualified": False}


if __name__ == "__main__":
    require(len(sys.argv) == 1, "Replay takes no arguments and does not execute the native controller")
    print(json.dumps(replay(), indent=2, sort_keys=True, allow_nan=False))
