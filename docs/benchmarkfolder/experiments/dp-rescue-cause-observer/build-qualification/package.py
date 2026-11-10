#!/usr/bin/env python3
"""Package text-only build evidence, or verify its portable bytes; no workloads."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = HERE.parents[4]
WORK = ROOT / ".tirust-build/dp-rescue-cause-observer"
ATTEMPTS = {
    "cause-1": ("cause-1/build/build.json", "fdc30362ca2147775125c1d39def5e112c3168b345ce122f02fd24426d65c727"),
    "cause-2": ("cause-2/build/build.json", "7749f7d9ddc11f718297b5cd45d3f17261c245fe22e018ea95956e26cccae35e"),
    "continuation-1": ("cause-2-continuation-1/build.json", "0ef8947065e4505834e9095cec86a584fdb4e42014165cceaab709f231b3efe2"),
    "continuation-2": ("cause-2-continuation-2/build.json", "b3f71a6493ce315dfd45d9a9da2302e16ae80c155f5a147ec65cb8a20f1e71e3"),
}


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pin(path):
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def verify():
    index = json.loads((HERE / "index.json").read_text())
    for name, expected in index["files"].items():
        path = HERE / name
        if path.is_symlink() or pin(path) != expected:
            raise ValueError("Portable payload changed: " + name)
    for name, row in index["compressed_text"].items():
        value, size = hashlib.sha256(), 0
        with gzip.open(HERE / name, "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                value.update(block)
                size += len(block)
        if value.hexdigest() != row["raw_sha256"] or size != row["raw_bytes"]:
            raise ValueError("Compressed text does not reproduce original bytes: " + name)
    bindings = json.loads((HERE / "source-bindings.json").read_text())
    for name, expected in bindings.items():
        if pin(ROOT / name) != expected:
            raise ValueError("Retained controller/proposal source changed: " + name)
    actual = {str(path.relative_to(HERE)) for path in HERE.rglob("*") if path.is_file()}
    if actual != set(index["files"]) | {"index.json"}:
        raise ValueError("Unindexed portable payload")
    print(json.dumps({"portable_files": len(index["files"]),
                      "compressed_text_files": len(index["compressed_text"]),
                      "source_bindings": len(bindings), "status": "passed"}))


def package():
    if Path.cwd().resolve() != ROOT or (HERE / "index.json").exists():
        raise ValueError("Run once from repository root; existing package is immutable")
    os.sched_setaffinity(0, {11})
    compressed, summary, originals = {}, {}, {}

    def retain(source, relative):
        source = source.resolve(strict=True)
        before = pin(source)
        target = HERE / (relative + ".gz")
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as out:
            if src.read(4) in (b"\x7fELF", b"!<ar"):
                raise ValueError("Binary input is forbidden")
            src.seek(0)
            with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=out, mtime=0) as gz:
                for block in iter(lambda: src.read(1024 * 1024), b""):
                    gz.write(block)
        if pin(source) != before:
            raise ValueError("Raw evidence changed while packaging")
        name = str(target.relative_to(HERE))
        compressed[name] = {"codec": "gzip;mtime=0;filename-empty;level=6",
                            "source_path": str(source.relative_to(ROOT)),
                            "raw_sha256": before["sha256"], "raw_bytes": before["bytes"]}
        originals[str(source)] = before

    final = None
    for label, (relative, expected) in ATTEMPTS.items():
        path = WORK / relative
        if digest(path) != expected:
            raise ValueError("Prior build manifest changed")
        record = json.loads(path.read_text())
        retain(path, "attempts/" + label + "/build.json")
        retain(path.parent / "commands.json", "attempts/" + label + "/commands.json")
        for event in record["commands"]:
            source = Path(event["log"])
            if digest(source) != event["log_sha256"]:
                raise ValueError("Command log changed")
            retain(source, "attempts/" + label + "/logs/" + source.name)
        summary[label] = {"manifest_sha256": expected, "status": record["status"],
                          "error": record.get("error"), "protected_inputs_unchanged": record["protected_inputs_unchanged"],
                          "numerical_execution_performed": record["numerical_execution_performed"],
                          "command_count": len(record["commands"]), "artifact_count": len(record["artifacts"]),
                          "all_children_reaped": all(x.get("process_reaped") is True for x in record["commands"])}
        if label == "continuation-2":
            final = record
    retain(Path(final["native"]["link_map"]["path"]), "linkmaps/native.map")
    for case, row in final["cases"].items():
        for arm, value in row["arms"].items():
            retain(Path(value["link_map"]["path"]), "linkmaps/" + case + "-" + arm + ".map")
    # The unchanged execution controller requires <label>/build/build.json.
    # This is a byte-identical metadata copy, not a new build or replay result.
    admission = WORK / "cause-2-qualified/build/build.json"
    admission.parent.mkdir(parents=True, exist_ok=False)
    original = WORK / ATTEMPTS["continuation-2"][0]
    with original.open("rb") as src, admission.open("xb") as out:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            out.write(block)
    if pin(admission) != pin(original):
        raise ValueError("Execution-admission metadata copy differs")
    taskset = Path("/run/current-system/sw/bin/taskset").resolve(strict=True)
    taskset_hash = "b5d01660bcca50634a7e046ed5702be5d46417b287dfb4cf642683bd70747a15"
    if digest(taskset) != taskset_hash:
        raise ValueError("Reviewed taskset changed")
    argv = ["python3", "-I", "-B", "docs/benchmarkfolder/experiments/dp-rescue-cause-observer/controller/run.py",
            "execute", "--label", "cause-2-qualified", "--self-sha256",
            "6a0512b620f0f9723af8811bbbd128f3b1f0990aac64e3aaf1da60ec7e5cab10",
            "--build-manifest", str(admission), "--build-manifest-sha256", ATTEMPTS["continuation-2"][1],
            "--taskset", str(taskset), "--taskset-sha256", taskset_hash]
    write(HERE / "future-command.json", {"cwd": str(ROOT), "environment_setup": "source tools/tirust_env.sh",
          "argv": argv, "launched": False, "root_execution_admission_pending": True,
          "cpu": 11, "arms": 6, "seconds_per_arm": 600, "total_seconds_after_lock": 3900,
          "shared_lock": ".tirust-build/numerical-tests.lock", "outer_flock": False,
          "partial_arm_resume_supported": False, "admission_manifest": pin(admission)})
    write(HERE / "summary.json", {"attempts": summary, "build_status": final["status"],
          "same_local_mpfr_instance": final["native"]["same_local_mpfr_instance"],
          "original_dso": final["original_dso"], "observed_dso": final["observed_dso"],
          "cases": final["cases"], "artifact_binaries_included": False,
          "cause_attribution_qualified": False, "numerical_parity_tested": False,
          "performance_eligible": False})
    write(HERE / "verifier-receipt.json", {"scope": "Read-only current artifact/provenance verification; no replay",
          "function": "controller/run.py:verify_build", "controller_sha256": argv[8],
          "verified_manifest": str(original.relative_to(ROOT)),
          "manifest_sha256": ATTEMPTS["continuation-2"][1], "result": "passed", "verified_pins": 33940,
          "all_prior_failure_manifest_bytes_unchanged": True,
          "admission_copy_byte_identical": True,
          "independent_source_review": "/root/madloop_audit/cache_driver: exact module guard and six-arm lock/admission contract clear; no workloads"})
    bindings = {}
    initial = json.loads((PROPOSAL / "index.json").read_text())
    sources = [PROPOSAL / "index.json"] + [PROPOSAL / name for name in initial["files"]]
    sources += [path for path in (PROPOSAL / "controller").rglob("*") if path.is_file()]
    for source in sources:
        bindings[str(source.relative_to(ROOT))] = pin(source)
    write(HERE / "source-bindings.json", bindings)
    for path, expected in originals.items():
        if pin(Path(path)) != expected:
            raise ValueError("Original raw record changed during packaging")
    files = {str(path.relative_to(HERE)): pin(path) for path in sorted(HERE.rglob("*"))
             if path.is_file() and path != HERE / "index.json"}
    write(HERE / "index.json", {"schema_version": 1, "scope": "Portable compile/link evidence only; no native replay",
          "files": files, "compressed_text": compressed,
          "binary_payloads": 0, "dependency_cache_payloads": 0})
    verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true", help="Create once; default only verifies existing evidence")
    args = parser.parse_args()
    package() if args.package else verify()
