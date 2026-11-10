#!/usr/bin/env python3
"""Read-only portable verification by default; --package appends evidence once.

No build, numerical worker, parser modification, or execution retry is provided.
Wait for root's final-completion notice and the completed supplemental audit
before invoking --package. Hash arguments bind inputs; they grant no permission.
"""
from __future__ import annotations

import argparse
import codecs
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = PROPOSAL.parents[3]
SOURCE = ROOT / ".tirust-build/dp-rescue-cause-observer/cause-2-qualified/controller-native/attempt1"
BUILD_INDEX = PROPOSAL / "build-qualification/index.json"
AUDIT_INDEX = PROPOSAL / "numerical-audit/index.json"
BUILD_SHA = "b3f71a6493ce315dfd45d9a9da2302e16ae80c155f5a147ec65cb8a20f1e71e3"
FIXED = {
    "index.json": "b0b0af81739be886420450459eeed94a839c71e8aeef22b17c79d3b510c2cd1b",
    "controller/index.json": "73980096e0603cc311b4a4a6936357e89eb75a42dd0d28a6242830f91f411af5",
    "controller/run.py": "6a0512b620f0f9723af8811bbbd128f3b1f0990aac64e3aaf1da60ec7e5cab10",
    "native/index.json": "751208a43d3dd1f01a4f6008ca7177cf8b91f69fa9212ead2b524e964efa6585",
    "analyze.py": "473475dfdde70df2f721ca782f3ac1719ab08b92b72a9eadd4e8bbe216aed10b",
    "protocol.py": "b668433b0fe6d397d3416d408b17a1f7f54f2f8b24c9cc063c1a5710423d38ec",
    "preparation/source-1/snapshot.json": "2aa5e87ead00c163d0f987dbef257654a8a22ec748f7d3f71be68aeaa64e7759",
    "build-qualification/index.json": "eefd95d7deafbcb74852e2e1ec5fcb527f0d2fd19f1dce75eb85a28063068041",
}
CASES = ("qcd_n3_uux_zgg", "ew_n3_epem_ttxa")
ARMS = ("original", "disabled", "enabled")
REQUIRED = {"cause-public.txt", "warmup.dat", "protocol.raw", "stdout.log", "process-maps.txt",
            "points.dat", "points.json", "Cards/param_card.dat", "Cards/ident_card.dat"}
MAX_RAW = 512 * 1024**2


def need(value, message):
    if not value:
        raise ValueError(message)


def relative(base, name):
    part = PurePosixPath(name)
    need(isinstance(name, str) and part.parts and not part.is_absolute() and ".." not in part.parts
         and part.as_posix() == name, "Unsafe relative name")
    path = base.joinpath(*part.parts)
    need(path.resolve() == path and not path.is_symlink(), "Linked or aliased artifact")
    return path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def pin(path):
    need(not path.is_symlink() and path.is_file(), "Expected a regular retained file: " + str(path))
    return {"sha256": digest(path), "bytes": path.stat().st_size}


def write(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def read_json(path):
    need(path.stat().st_size <= MAX_RAW, "JSON record exceeds bound")
    return json.loads(path.read_text())


def binding_inventory(audit_sha):
    """Repository text only; do not traverse recorded runtime/artifact paths."""
    bindings = {}
    for name, expected in FIXED.items():
        path = relative(PROPOSAL, name)
        actual = pin(path)
        need(actual["sha256"] == expected, "Frozen source/build binding changed: " + name)
        bindings[str(path.relative_to(ROOT))] = actual
    build_index = read_json(BUILD_INDEX)
    for name, expected in build_index["files"].items():
        path = relative(BUILD_INDEX.parent, name)
        need(pin(path) == expected, "Completed build package changed: " + name)
        bindings[str(path.relative_to(ROOT))] = expected
    source_bindings = read_json(BUILD_INDEX.parent / "source-bindings.json")
    for name, expected in source_bindings.items():
        need(name.startswith("docs/benchmarkfolder/"), "Source binding points outside retained repository evidence")
        path = relative(ROOT, name)
        need(pin(path) == expected, "Frozen source binding changed: " + name)
        bindings[name] = expected
    need(pin(AUDIT_INDEX)["sha256"] == audit_sha, "Required completed supplemental audit index differs")
    bindings[str(AUDIT_INDEX.relative_to(ROOT))] = pin(AUDIT_INDEX)
    audit_index = read_json(AUDIT_INDEX)
    need("results.json" in audit_index["files"] and "audit_native_joins.py" in audit_index["files"],
         "Supplemental audit index omits results or source")
    for name, expected in audit_index["files"].items():
        path = relative(AUDIT_INDEX.parent, name)
        normalized = {"sha256": expected["sha256"], "bytes": expected["size_bytes"]}
        need(pin(path) == normalized, "Supplemental audit payload changed: " + name)
        bindings[str(path.relative_to(ROOT))] = normalized
    return bindings


def terminal(record):
    need(record.get("schema") == "dp-rescue-six-arm-execution-v1", "Unexpected execution schema")
    need(record.get("complete") is True or record.get("error"), "Execution is still active or lacks final failure metadata")
    need(record["build_manifest_sha256"] == BUILD_SHA
         and record["controller_sha256"] == FIXED["controller/run.py"]
         and record["proposal_sha256"] == FIXED["index.json"], "Execution binds another build/controller/proposal")
    need(record["performance_eligible"] is False and record["automatic_retry_count"] == 0,
         "Unexpected execution scope")


def validate_records(top, inventory, read, origin):
    """Validate retained identity joins; this is not the semantic event parser."""
    terminal(top)
    need(isinstance(origin, str) and PurePosixPath(origin).is_absolute()
         and ".." not in PurePosixPath(origin).parts, "Invalid recorded source origin")
    completed = []
    for case in CASES:
        for arm in ARMS:
            key = case + "/" + arm
            name = key + "/execution.json"
            if name not in inventory:
                need(top["complete"] is not True and key not in top["arms"], "Missing arm metadata")
                continue
            row = json.loads(read(name))
            need(not row.get("process_spawned") or row.get("process_reaped") is True,
                 "Worker ownership is not closed; cannot package live output")
            if key in top["arms"]:
                need(row == top["arms"][key], "Outer and arm execution metadata differ")
            if top["complete"] is True:
                need(row.get("complete") is True and row.get("returncode") == 0
                     and row.get("process_spawned") is True, "Completed execution has incomplete arm")
            for output, sha in row.get("output_sha256", {}).items():
                logical = key + "/" + output
                need(logical in inventory and inventory[logical]["raw_sha256"] == sha, "Arm output binding differs: " + logical)
            for source, sha in row.get("copied_inputs", {}).items():
                prefix = origin.rstrip("/") + "/" + key + "/"
                need(source.startswith(prefix), "Copied input escaped its arm")
                logical = key + "/" + source[len(prefix):]
                need(logical in inventory and inventory[logical]["raw_sha256"] == sha, "Initial private input differs")
            if row.get("complete") is True:
                required = REQUIRED | ({"cause-native.jsonl"} if arm != "original" else set())
                need(all(key + "/" + name in inventory for name in required), "Missing complete-arm raw records")
                need(inventory[key + "/process-maps.txt"]["raw_sha256"] == row["actual_runtime"]["maps_sha256"],
                     "Actual runtime map evidence differs")
                completed.append(key)
        analysis_name = case + "/analysis.json"
        if case in top.get("case_analysis", {}):
            need(analysis_name in inventory and inventory[analysis_name]["raw_sha256"]
                 == top["case_analysis"][case]["sha256"], "Final analysis binding differs")
    if top["complete"] is True:
        need(set(completed) == {case + "/" + arm for case in CASES for arm in ARMS}
             and set(top["case_analysis"]) == set(CASES), "Successful run lacks all six arms and both analyses")
        need(top["protected_before"] == top["protected_after"] and top["numerical_parity_passed"] is True
             and top["runtime_provenance_validated"] is True, "Final controller did not qualify parity/provenance")
    return completed


def validate_audit(audit, execution_sha, bindings, inventory):
    need(audit.get("schema") == "dp-rescue-six-arm-join-audit-v1"
         and audit.get("execution_sha256") == execution_sha and set(audit["cases"]) == set(CASES),
         "Supplemental audit is not for this complete six-arm execution")
    for flag in ("supplemental_joins_passed", "all_six_completed_execution_record_bound",
                 "all_public_triplets_independently_reparsed"):
        need(audit.get(flag) is True, "Supplemental audit gate failed: " + flag)
    auditor = str((AUDIT_INDEX.parent / "audit_native_joins.py").relative_to(ROOT))
    need(audit["auditor_sha256"] == bindings[auditor]["sha256"], "Supplemental auditor source differs")
    for case, result in audit["cases"].items():
        need(result["case"] == case and result["supplemental_joins_passed"] is True
             and result["frozen_native_parser_passed"] is True and result["public_call_count"] == 101,
             "Supplemental case did not pass the complete native/public audit")
        need(result["execution_sha256"] == inventory[case + "/enabled/execution.json"]["raw_sha256"],
             "Supplemental enabled-arm execution differs")
        for name, sha in result["raw_sha256"].items():
            need(inventory[case + "/enabled/" + name]["raw_sha256"] == sha, "Supplemental raw journal binding differs")
        triplet = result["independent_public_triplet"]
        need(triplet["case"] == case and all(triplet.get(flag) is True for flag in
             ("public_parity_passed", "full101_call_order_passed", "historical_warmup_bytes_exact",
              "all_input_answer_accuracy_words_exact", "all_entry_exit_ieee_states_exact")),
             "Independent public triplet did not pass")


def measured_cases(audit, read):
    result = {}
    for case, supplemental in audit["cases"].items():
        analysis = json.loads(read(case + "/analysis.json"))
        points = supplemental["cause_observations"]["all100_points"]
        need(len(points) == 100 and {row["point"] for row in points} == set(range(1, 101)),
             "Supplemental public point membership differs")
        need(analysis["status_counts"] == supplemental["independent_public_triplet"]["status_counts"],
             "Frozen and independent public status counts differ")
        errors = [row for row in supplemental["immediate_native_errors"]
                  if row["phase"] == "points_1_to_100" and row["input_bits"] == 53]
        statuses, error_classes = Counter(), Counter()
        for row in errors:
            statuses[str(row["status"])] += row["count"]
            text = row["error"]
            label = ("ill-conditioned Gram matrix" if "ill-conditioned Gram matrix" in text else
                     "supplied invariant matrix must be symmetric" if "supplied invariant matrix must be symmetric" in text
                     else "other; see retained immediate_native_errors")
            error_classes[label] += row["count"]
        result[case] = {
            "scope": "Concurrent observations with independent supplemental joins; no unique-cause or performance claim",
            "public_points": len(points), "status_counts": analysis["status_counts"],
            "dp_accepted_points": analysis["status_counts"].get("219", 0),
            "qp_rescued_points": analysis["status_counts"].get("323", 0),
            "native_dp_failure_calls": sum(statuses.values()), "native_dp_failure_status_counts": dict(statuses),
            "immediate_native_dp_error_classes": dict(error_classes),
            "host_finite_pre_accuracy_exceed_point_count": sum(row["finite_host_accuracy_exceeds_threshold"] for row in points),
            "host_finite_pre_accuracy_exceed_points": [row["point"] for row in points
                                                       if row["finite_host_accuracy_exceeds_threshold"]],
            "post_stable_contraction_rejection_groups": sum(row["contraction_rejections"] for row in points),
            "observation_point_counts": supplemental["cause_observations"]["observation_point_counts"],
            "bootstrap_separate": supplemental["cause_observations"]["bootstrap"],
            "cause_attribution_qualified": False, "performance_eligible": False,
        }
    return result


def portable_verify():
    """Works without ignored workspace inputs, libraries, binaries or caches."""
    index = read_json(HERE / "index.json")
    need(index["schema"] == "dp-rescue-portable-numerical-evidence-v1", "Wrong portable package schema")
    actual_files = {str(p.relative_to(HERE)) for p in HERE.rglob("*") if p.is_file()}
    need(actual_files == set(index["files"]) | {"index.json"}, "Unindexed or missing portable file")
    for name, expected in index["files"].items():
        need(pin(relative(HERE, name)) == expected, "Portable file differs: " + name)
    for name, expected in index["source_bindings"].items():
        need(name.startswith("docs/benchmarkfolder/"), "Portable verification cannot read ignored workspace")
        need(pin(relative(ROOT, name)) == expected, "Repository source/build/audit binding differs: " + name)
    objects = {}
    for row in index["inventory"].values():
        old = objects.setdefault(row["payload"], row)
        need((old["raw_sha256"], old["raw_bytes"]) == (row["raw_sha256"], row["raw_bytes"]), "Conflicting payload identity")
    for name, row in objects.items():
        value, size = hashlib.sha256(), 0
        with gzip.open(relative(HERE, name), "rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                size += len(block)
                need(size <= row["raw_bytes"] <= MAX_RAW, "Decompressed payload exceeds bound")
                value.update(block)
        need(size == row["raw_bytes"] and value.hexdigest() == row["raw_sha256"], "Raw compressed bytes differ")
    def read(name):
        row = index["inventory"][name]
        need(row["raw_bytes"] <= MAX_RAW, "Oversized retained JSON")
        with gzip.open(relative(HERE, row["payload"]), "rb") as stream:
            return stream.read()
    top = json.loads(read("execution.json"))
    completed = validate_records(top, index["inventory"], read, index["source_origin"])
    need(index["execution_sha256"] == index["inventory"]["execution.json"]["raw_sha256"], "Final execution hash differs")
    need(index["inventory"]["supplemental/results.json"]["raw_sha256"] ==
         index["source_bindings"][str((AUDIT_INDEX.parent / "results.json").relative_to(ROOT))]["sha256"],
         "Supplemental results do not match required audit binding")
    audit = json.loads(read("supplemental/results.json"))
    validate_audit(audit, index["execution_sha256"], index["source_bindings"], index["inventory"])
    need(read_json(HERE / "summary.json")["cases"] == measured_cases(audit, read), "Measured summary differs from bound analyses")
    return {"status": "passed", "scope": "Portable bytes, inventory, execution/output/analysis joins and repository bindings only",
            "semantic_parser_replayed": False, "native_execution": False, "performance_eligible": False,
            "completed_arms": len(completed), "retained_names": len(index["inventory"]), "unique_text_payloads": len(objects),
            "index_sha256": digest(HERE / "index.json")}


def package(execution_sha, audit_sha):
    need(Path.cwd().resolve() == ROOT, "Package only from repository root")
    need(not (HERE / "index.json").exists() and not (HERE / "packaging-start.json").exists()
         and not (HERE / "payloads").exists(), "Packaging is append-once; retain prior partial or complete attempt")
    need(execution_sha and audit_sha, "Final execution and supplemental audit hashes are required")
    source_top = SOURCE / "execution.json"
    need(pin(source_top)["sha256"] == execution_sha, "Final execution hash differs")
    top = read_json(source_top)
    terminal(top)
    bindings = binding_inventory(audit_sha)
    raw_sources = {}
    excluded = []
    for path in sorted(SOURCE.rglob("*")):
        name = str(path.relative_to(SOURCE))
        need(not path.is_symlink(), "Linked runtime artifact: " + name)
        if path.is_dir():
            continue
        mode = path.stat().st_mode
        if stat.S_ISFIFO(mode) and path.name == "protocol.fifo":
            excluded.append({"path": name, "reason": "FIFO transport; exact bytes retained as protocol.raw"})
            continue
        if "tmp" in path.relative_to(SOURCE).parts:
            excluded.append({"path": name, "reason": "Temporary/cache workspace; not numerical evidence"})
            continue
        need(stat.S_ISREG(mode), "Nonregular runtime evidence: " + name)
        raw_sources[name] = path
    raw_sources["supplemental/results.json"] = AUDIT_INDEX.parent / "results.json"
    inventory = {name: {"raw_sha256": pin(path)["sha256"], "raw_bytes": path.stat().st_size}
                 for name, path in raw_sources.items()}
    read = lambda name: raw_sources[name].read_bytes()
    completed = validate_records(top, inventory, read, str(SOURCE))
    audit = json.loads(read("supplemental/results.json"))
    validate_audit(audit, execution_sha, bindings, inventory)
    case_summary = measured_cases(audit, read)
    write(HERE / "packaging-start.json", {"execution_sha256": execution_sha, "audit_index_sha256": audit_sha,
          "packager_sha256": digest(Path(__file__)), "scope": "Text packaging only; no numerical execution"})
    try:
        (HERE / "payloads").mkdir()
        for name, source in raw_sources.items():
            row = inventory[name]
            need(row["raw_bytes"] <= MAX_RAW, "Raw text exceeds artifact bound")
            row["payload"] = "payloads/" + row["raw_sha256"] + ".gz"
            target = relative(HERE, row["payload"])
            if not target.exists():
                decoder = codecs.getincrementaldecoder("utf-8")("strict")
                copied, size = hashlib.sha256(), 0
                with source.open("rb") as src, target.open("xb") as out:
                    head = src.read(4)
                    need(head not in (b"\x7fELF", b"!<ar"), "Binary executable/archive payload is forbidden")
                    src.seek(0)
                    with gzip.GzipFile(filename="", mode="wb", fileobj=out, compresslevel=6, mtime=0) as gz:
                        for block in iter(lambda: src.read(1024 * 1024), b""):
                            decoder.decode(block)
                            size += len(block)
                            need(size <= row["raw_bytes"], "Source grew while packaging")
                            copied.update(block)
                            gz.write(block)
                        decoder.decode(b"", final=True)
                need(size == row["raw_bytes"] and copied.hexdigest() == row["raw_sha256"], "Source changed while copying")
        for name, source in raw_sources.items():
            need(pin(source) == {"sha256": inventory[name]["raw_sha256"], "bytes": inventory[name]["raw_bytes"]},
                 "Raw evidence changed during packaging: " + name)
        for name, expected in bindings.items():
            need(pin(relative(ROOT, name)) == expected, "Source/build/audit binding changed during packaging")
        summary = {"execution_complete": top["complete"], "execution_error": top.get("error"),
                   "completed_arms": completed, "runtime_provenance_validated": top.get("runtime_provenance_validated", False),
                   "numerical_parity_passed": top.get("numerical_parity_passed", False),
                   "cause_attribution_qualified": top.get("cause_attribution_qualified", False),
                   "semantic_parser_replayed_by_packager": False, "performance_eligible": False,
                   "cases": case_summary,
                   "supplemental_audit_required_and_retained": True, "binary_payloads": 0, "dependency_cache_payloads": 0,
                   "prior_build_failures": "Preserved in the bound build-qualification package; not duplicated"}
        write(HERE / "summary.json", summary)
        files = {str(p.relative_to(HERE)): pin(p) for p in sorted(HERE.rglob("*")) if p.is_file()}
        write(HERE / "index.json", {"schema": "dp-rescue-portable-numerical-evidence-v1", "files": files,
              "inventory": inventory, "source_bindings": bindings, "excluded_non_evidence": excluded,
              "execution_sha256": execution_sha, "supplemental_audit_index_sha256": audit_sha,
              "source_origin": str(SOURCE),
              "codec": "gzip;mtime=0;filename-empty;level=6", "binary_payloads": 0, "dependency_cache_payloads": 0})
    except BaseException as error:
        try:
            write(HERE / "packaging-failure.json", {"error": repr(error), "partial_outputs_preserved": True,
                  "automatic_retry": False, "execution_sha256": execution_sha})
        except BaseException as metadata_error:
            error.add_note("Packaging failure metadata error: " + repr(metadata_error))
        raise
    return portable_verify()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", action="store_true")
    parser.add_argument("--execution-sha256")
    parser.add_argument("--audit-index-sha256")
    args = parser.parse_args()
    result = package(args.execution_sha256, args.audit_index_sha256) if args.package else portable_verify()
    print(json.dumps(result, indent=2, sort_keys=True))
