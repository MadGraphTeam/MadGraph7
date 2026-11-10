#!/usr/bin/env python3
"""Append compact text evidence once; default verifies retained bytes only."""
import argparse
import codecs
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = PROPOSAL.parents[4]
CAUSE = ROOT / "docs/benchmarkfolder/experiments/dp-rescue-cause-observer"
WORK = ROOT / ".tirust-build/ew-invariant-symmetry-prototype-v2/attempt1/build"
MANIFEST_SHA = "bd0b33f69a552851feac60e61573442bfefad5c17f45dd9713d40d6a9ac6094a"
BUILDER_SHA = "c442152caf1f7d29cd9511eba259cc442461730baeab17fb19ee4d3126d5593d"
PLAN_SHA = "74cc1ba21a3f2da38b6f26d7f80d6fa443a9be08b2955032aeb292d2c698e184"
TOP_SHA = "d3c1e4212669d66846195bf65af0a05153ce406c87ac12cb30cc58a612b3bb93"
BUILD_INDEX_SHA = "eefd95d7deafbcb74852e2e1ec5fcb527f0d2fd19f1dce75eb85a28063068041"
NUMERICAL_INDEX_SHA = "9f934ced6cfe06d2f01252b8a86d6c5f50cd7bd29890fd2bf78b9e5db40769e2"
MAX_RAW = 128 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def relative(base, name):
    need(isinstance(name, str), "Relative filename must be text")
    part = PurePosixPath(name)
    need(part.parts and not part.is_absolute() and ".." not in part.parts
         and part.as_posix() == name, "Unsafe retained relative name")
    path = base.joinpath(*part.parts)
    need(path.resolve() == path and not path.is_symlink(), "Retained path aliases")
    return path


def pin(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return {"sha256": value.hexdigest(), "bytes": path.stat().st_size}


def write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def original_pins(records):
    """Only packaging uses current workspace files; portable verify does not."""
    for name, expected in records.items():
        path = Path(name)
        resolved = path.resolve(strict=True)
        need(path.is_absolute() and any(path.is_relative_to(root) and resolved.is_relative_to(root)
             for root in (ROOT, Path("/nix/store"))), "Original input escaped declared roots")
        actual = pin(path)
        need(actual["sha256"] == expected["sha256"] and
             ("size_bytes" not in expected or actual["bytes"] == expected["size_bytes"]), "Original bytes changed: " + name)
        need("resolved_path" not in expected or str(resolved) == expected["resolved_path"], "Original path changed")


def bindings():
    result = {}
    for path, sha in ((PROPOSAL / "index.json", TOP_SHA), (PROPOSAL / "build-proposal/run.py", BUILDER_SHA),
                      (PROPOSAL / "build-proposal/plan.json", PLAN_SHA),
                      (CAUSE / "build-qualification/index.json", BUILD_INDEX_SHA),
                      (CAUSE / "numerical-qualification-1/index.json", NUMERICAL_INDEX_SHA)):
        value = pin(path)
        need(value["sha256"] == sha, "Frozen source/package anchor changed")
        result[str(path.relative_to(ROOT))] = value
    top = json.loads((PROPOSAL / "index.json").read_text())
    for name, expected in top["files"].items():
        path = relative(PROPOSAL, name)
        need(pin(path) == expected, "Canonical source payload changed")
        result[str(path.relative_to(ROOT))] = expected
    for name, expected in top["repository_bindings"].items():
        need(pin(relative(ROOT, name)) == expected, "Canonical repository binding changed")
        result[name] = expected
    prior = json.loads((CAUSE / "build-qualification/index.json").read_text())
    prior_name = "attempts/continuation-2/build.json.gz"
    prior_path = CAUSE / "build-qualification" / prior_name
    need(pin(prior_path) == prior["files"][prior_name]
         and prior["compressed_text"][prior_name]["raw_sha256"] ==
         "b3f71a6493ce315dfd45d9a9da2302e16ae80c155f5a147ec65cb8a20f1e71e3", "Prior cause build archive differs")
    result[str(prior_path.relative_to(ROOT))] = prior["files"][prior_name]
    numerical = json.loads((CAUSE / "numerical-qualification-1/index.json").read_text())
    row = numerical["inventory"]["execution.json"]
    need(row["raw_sha256"] == "c308d6bdf48db24bc428e0c556618e4191fe07535720708f954b62331376f8b4", "Prior numerical top differs")
    path = relative(CAUSE / "numerical-qualification-1", row["payload"])
    need(pin(path) == numerical["files"][row["payload"]], "Prior numerical payload differs")
    result[str(path.relative_to(ROOT))] = numerical["files"][row["payload"]]
    return result


def validate(record, plan, commands):
    need(record["schema"] == "ew-invariant-adapter-build-v2" and record["status"] == "complete", "Incomplete or wrong build")
    need(record["builder_sha256"] == BUILDER_SHA and record["plan_sha256"] == PLAN_SHA, "Wrong reviewed build source")
    need(record["protected_before"] == record["protected_after"] and len(record["protected_before"]) == 899,
         "Build did not preserve all selected inputs")
    need(record["dependency_listing_verified"] is True, "Dependency listing did not pass")
    for flag in ("numerical_execution_performed", "numerical_parity_qualified", "loaded_numerical_runtime_verified", "performance_eligible"):
        need(record[flag] is False, "Build evidence claims an unperformed qualification")
    need(record["commands"] == commands and len(commands) == 3 and all(row["process_spawned"] and row["process_reaped"]
         and row["returncode"] == 0 for row in commands), "Compile/link/list commands did not complete")
    need(commands[0]["argv"] == plan["compile_argv"] and commands[2]["argv"] ==
         [record["interpreter"]["path"], "--list", record["executable"]["path"]], "Actual command scope differs")
    need(record["candidate"] == plan["candidate"] and record["original_executable"] == plan["original_executable"],
         "Candidate or baseline differs")
    need(record["runtime"] == plan["runtime"] and record["interpreter"] == plan["interpreter"], "Runtime closure changed")
    need(record["link_objects"]["check_sa.o"] == plan["original_driver"], "Raw-word driver changed")
    for name, expected in plan["unchanged_observer_support"].items():
        need(record["link_objects"][name] == expected, "Noop/host support object changed")
    need(len(record["link_objects"]) == 23 and record["link_objects"]["TIRUST_interface.o"] == record["changed_object"],
         "Adapter substitution differs")
    for name, value in record["link_objects"].items():
        if name != "TIRUST_interface.o":
            need(record["protected_before"][value["path"]]["sha256"] == value["sha256"], "Another object changed")
    bridges = [path for path in record["runtime"] if PurePosixPath(path).name == "libmadloop_tirust.so"]
    need(len(bridges) == 1 and record["runtime"][bridges[0]] ==
         "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9", "Wrong installed provider")
    return {"schema": "ew-invariant-adapter-build-summary-v2", "build_status": "complete", "cpu": record["cpu"],
            "jobs": record["jobs"], "seconds_cap": record["seconds_cap"], "command_count": 3,
            "compile_commands": 1, "link_commands": 1, "loader_list_commands": 1,
            "protected_inputs_unchanged": 899, "changed_numerical_objects": ["TIRUST_interface.o"],
            "driver_and_host_noop_objects_unchanged": True, "candidate": record["candidate"],
            "changed_object": record["changed_object"], "executable": record["executable"],
            "runtime": record["runtime"], "dependency_listing_verified": True,
            "numerical_execution_performed": False, "numerical_parity_qualified": False,
            "loaded_numerical_runtime_verified": False, "performance_eligible": False,
            "binary_payloads": 0, "dependency_cache_payloads": 0,
            "qualification_pending": record["qualification_pending"]}


def verify():
    index = json.loads((HERE / "index.json").read_text())
    need(index["schema"] == "ew-invariant-portable-build-v2", "Wrong package schema")
    need({str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()} == set(index["files"]) | {"index.json"},
         "Unindexed or missing retained file")
    for name, expected in index["files"].items():
        need(pin(relative(HERE, name)) == expected, "Retained file changed")
    for name, expected in index["repository_bindings"].items():
        need(name.startswith("docs/benchmarkfolder/"), "Portable verifier cannot read workspace artifacts")
        need(pin(relative(ROOT, name)) == expected, "Repository binding changed")
    for name, row in index["evidence"].items():
        value, size = hashlib.sha256(), 0
        with gzip.open(relative(HERE, row["payload"]), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                need(size <= row["raw_bytes"] <= MAX_RAW, "Decompressed evidence exceeds bound")
                value.update(block)
        need(size == row["raw_bytes"] and value.hexdigest() == row["raw_sha256"], "Original text bytes differ: " + name)
    def read(name):
        with gzip.open(relative(HERE, index["evidence"][name]["payload"]), "rb") as stream:
            return json.load(stream)
    record, plan, commands = read("build.json"), read("source/plan.json"), read("commands.json")
    need(index["evidence"]["build.json"]["raw_sha256"] == MANIFEST_SHA, "Completed manifest hash differs")
    for name, expected in (("source/plan.json", PLAN_SHA), ("source/run.py", BUILDER_SHA),
                           ("source/TIRUST_interface.f", record["candidate"]["sha256"])):
        need(index["evidence"][name]["raw_sha256"] == expected, "Frozen retained source differs: " + name)
    need(read_json_summary() == validate(record, plan, commands), "Summary differs from retained build evidence")
    need(index["evidence"]["private-link.map"]["raw_sha256"] == record["link_map"]["sha256"], "Link map binding differs")
    for command in commands:
        name = "logs/" + PurePosixPath(command["log"]).name
        need(index["evidence"][name]["raw_sha256"] == command["log_pin"]["sha256"], "Actual command log differs")
    return {"status": "passed", "index_sha256": pin(HERE / "index.json")["sha256"], "files": len(index["files"]) + 1,
            "text_payloads": len(index["evidence"]), "scope": "Portable text bytes, build records and repository bindings",
            "native_execution": False, "numerical_qualification": False}


def read_json_summary():
    return json.loads((HERE / "summary.json").read_text())


def package():
    need(Path.cwd().resolve() == ROOT and not (HERE / "index.json").exists()
         and not (HERE / "packaging-start.json").exists(), "Package once from repository root")
    manifest = WORK / "build.json"
    need(pin(manifest)["sha256"] == MANIFEST_SHA, "Completed build changed")
    record = json.loads(manifest.read_text())
    plan = json.loads((PROPOSAL / "build-proposal/plan.json").read_text())
    commands = json.loads((WORK / "commands.json").read_text())
    summary = validate(record, plan, commands)
    retained_bindings = bindings()
    original_pins(record["protected_before"])
    original_pins({value["path"]: value for value in record["artifacts"].values()})
    sources = {"build.json": manifest, "commands.json": WORK / "commands.json",
               "private-link.map": Path(record["link_map"]["path"]),
               "source/plan.json": PROPOSAL / "build-proposal/plan.json",
               "source/run.py": PROPOSAL / "build-proposal/run.py",
               "source/TIRUST_interface.f": PROPOSAL / "TIRUST_interface.f",
               "source/derivation.patch": PROPOSAL / "derivation.patch",
               "source/v1-to-v2.patch": PROPOSAL / "v1-to-v2.patch",
               "source/controller-delta.patch": PROPOSAL / "build-proposal/controller-delta.patch"}
    sources.update({"logs/" + Path(row["log"]).name: Path(row["log"]) for row in commands})
    originals = {name: pin(path) for name, path in sources.items()}
    write(HERE / "packaging-start.json", {"manifest_sha256": MANIFEST_SHA, "packager_sha256": pin(Path(__file__))["sha256"],
          "native_execution": False, "source_only_retention": True})
    evidence = {}
    try:
        for name, source in sources.items():
            expected = originals[name]
            need(expected["bytes"] <= MAX_RAW, "Text evidence exceeds bound")
            target = relative(HERE, "evidence/" + name + ".gz")
            target.parent.mkdir(parents=True, exist_ok=True)
            decoder, copied, size = codecs.getincrementaldecoder("utf-8")("strict"), hashlib.sha256(), 0
            with source.open("rb") as src, target.open("xb") as out:
                need(src.read(4) not in (b"\x7fELF", b"!<ar"), "Binary payload forbidden")
                src.seek(0)
                with gzip.GzipFile(filename="", mode="wb", compresslevel=6, fileobj=out, mtime=0) as compressed:
                    for block in iter(lambda: src.read(1024 * 1024), b""):
                        decoder.decode(block)
                        size += len(block)
                        need(size <= expected["bytes"], "Source grew during copy")
                        copied.update(block)
                        compressed.write(block)
                    decoder.decode(b"", final=True)
            need(size == expected["bytes"] and copied.hexdigest() == expected["sha256"], "Source changed during copy")
            evidence[name] = {"payload": str(target.relative_to(HERE)), "raw_sha256": expected["sha256"], "raw_bytes": size}
        for name, source in sources.items():
            need(pin(source) == originals[name], "Raw build evidence changed")
        original_pins(record["protected_before"])
        for name, expected in retained_bindings.items():
            need(pin(relative(ROOT, name)) == expected, "Repository binding changed while packaging")
        write(HERE / "summary.json", summary)
        files = {str(p.relative_to(HERE)): pin(p) for p in sorted(HERE.rglob("*")) if p.is_file()}
        write(HERE / "index.json", {"schema": "ew-invariant-portable-build-v2", "files": files, "evidence": evidence,
              "repository_bindings": retained_bindings, "manifest_sha256": MANIFEST_SHA,
              "codec": "gzip;mtime=0;filename-empty;level=6", "binary_payloads": 0, "dependency_cache_payloads": 0})
    except BaseException as error:
        try:
            write(HERE / "packaging-failure.json", {"error": repr(error), "partial_evidence_preserved": True, "automatic_retry": False})
        except BaseException as metadata_error:
            error.add_note("Packaging metadata error: " + repr(metadata_error))
        raise
    return verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true", help="Create the text package once; default only verifies")
    args = parser.parse_args()
    print(json.dumps(package() if args.package else verify(), indent=2, sort_keys=True))
