#!/usr/bin/env python3
"""Versioned correctness policy; frozen v1 collection and evidence stay untouched.

Commands consume explicit evidence specifications. No filename allowlist, role
label or recorded pass boolean substitutes for recomputed source/row validation.
"""
import argparse
import copy
import difflib
import itertools
import json
import math
from pathlib import Path
import shutil
import hashlib
import subprocess

import ninjars_benchmark as bench
import ninjars_benchmark_workers as workers

ROOT = bench.ROOT
SCHEMA = "madloop-helper-correction-policy-v2"
CORRECTION = "exact-model-correction-v2"
PROOF = "representative-policy-validation-v2"
MODEL_FILES = ("Source/MODEL/model_functions.f", "Source/MODEL/mp_intparam_definition.inc")
ROLES = {"original": (1, 6, 7), "candidate_corrected": (1, 6, 7, 8),
         "original_helper_corrected": (1, 6, 7)}
TIMING_ROLES = ("original", "candidate_corrected")
ORIGINAL_EVIDENCE_REVISION = "80fca669eb29ca507239e8a655c82c19d55818ee"
POLICY_SOURCES = ("ninjars_benchmark_policy_v2.py", "ninjars_benchmark_references_v2.py")


def read(path):
    return json.loads(Path(path).read_text())


def path(value):
    result = Path(value)
    result = (ROOT / result).resolve() if not result.is_absolute() else result.resolve()
    if not result.is_relative_to(ROOT):
        raise RuntimeError(f"Evidence must stay inside repository: {value}")
    return result


def relative(value):
    return str(path(value).relative_to(ROOT))


def seal(data, key="identity_sha256"):
    value = copy.deepcopy(data)
    value.pop(key, None)
    value[key] = bench.content_digest(value)
    return value


def verified(data, schema, key="identity_sha256"):
    if data.get("schema") != schema or seal(data, key) != data:
        raise RuntimeError(f"Invalid {schema} identity")
    return data


def bound_file(value):
    target = path(value)
    return {"path": relative(target), "sha256": workers.digest(target)}


def check_file(record):
    target = path(record["path"])
    if workers.digest(target) != record["sha256"]:
        raise RuntimeError(f"Changed bound evidence: {record['path']}")
    return target


def manifest(output):
    output = path(output)
    # rglob does not descend directory symlinks; refusing them prevents omitted
    # source trees. Individual links are included by the frozen manifest helper.
    for base in ("Source", "SubProcesses"):
        for item in (output / base).rglob("*"):
            if item.is_symlink() and item.is_dir() and not item.resolve().is_relative_to(output):
                raise RuntimeError(f"Unresolved directory source link: {item}")
    return bench.generated_source_manifest(output)


def model_subset(source_manifest):
    return {key: value for key, value in source_manifest["files"].items()
            if key.startswith("Source/MODEL/")}


def delta(before, after):
    return {key for key in before.keys() | after.keys() if before.get(key) != after.get(key)}


def verify_model_delta(before, after, expected):
    """Independent CLI exports may differ outside MODEL; model delta is exact."""
    verify_delta({"files": model_subset(before)}, {"files": model_subset(after)}, expected)


def original_identity(identity):
    if (identity.get("revision") != bench.BASELINE_REVISION or
            identity.get("status") not in ("recorded-before-generation", "reconstructed-from-verified-archive") or
            identity.get("verification") != "provenance.json:baseline_source_integrity"):
        raise RuntimeError("Source role requires the verified original archive")
    return identity


def verify_delta(before, after, expected):
    """Full source equality apart from independently specified exact postimages."""
    a, b = before["files"], after["files"]
    if set(a) != set(b):
        raise RuntimeError("Corrected reference adds or deletes generated source")
    if delta(a, b) != {name for name in MODEL_FILES if a.get(name) != expected.get(name)}:
        raise RuntimeError("Correction changed an unexpected source or omitted a model correction")
    for name in MODEL_FILES:
        if b.get(name) != expected.get(name):
            raise RuntimeError(f"Wrong exact model postimage: {name}")


def git_bytes(revision, name):
    return subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)


def physical_definition(case):
    return {key: case.get(key) for key in ("id", "model", "process", "nfinal", "complex_mass_scheme")}


def original_case_definition(case_id):
    for name in ("cases.json", "controls.json"):
        config = json.loads(git_bytes(ORIGINAL_EVIDENCE_REVISION, "docs/benchmarkfolder/"+name))
        for case in config["cases"]:
            if case["id"] == case_id:
                return physical_definition(case)
    raise RuntimeError("Case is not in committed core/control specification")


def anchor_original(case_id, output, source_manifest, live=True):
    """Use committed pre-correction evidence, never an asserted source label."""
    prefix = f"docs/benchmarkfolder/inputs/baseline/{case_id}/"
    saved = json.loads(git_bytes(ORIGINAL_EVIDENCE_REVISION, prefix+"generated-sources-observed.json"))
    generation = json.loads(git_bytes(ORIGINAL_EVIDENCE_REVISION, prefix+"generation.json"))
    card = git_bytes(ORIGINAL_EVIDENCE_REVISION, prefix+"param_card.dat")
    if (source_manifest["files"] != saved["files"] or source_manifest["files_sha256"] != saved["files_sha256"] or (live and (read(path(output) / "generation.json") != generation or
            (path(output) / "Cards/param_card.dat").read_bytes() != card))):
        raise RuntimeError("Parent differs from committed untouched-original export evidence")
    original_identity(generation["source_identity"])
    return {"evidence_revision": ORIGINAL_EVIDENCE_REVISION, "case": case_id,
            "manifest_sha256": bench.content_digest(saved),
            "generation_sha256": bench.content_digest(generation),
            "parameter_card_sha256": hashlib.sha256(card).hexdigest()}


def generation_model_binding(output, source_manifest, generator, after):
    """Bind postimages to actual clean CLI generation and before-build manifest."""
    output = path(output)
    generation = read(output / "generation.json")
    identity = generation["source_identity"]
    side = "after" if after else "before"
    if after:
        if (identity.get("status") != "recorded-before-generation" or
                identity.get("dirty_patch_sha256") is not None or
                identity.get("tree_state_sha256") != bench.content_digest(identity["tree_state"])):
            raise RuntimeError("Corrected CLI proof requires a recorded clean generator identity")
        if seal(identity) != identity:
            raise RuntimeError("Generator source identity digest is invalid")
        recorded = identity["tree_state"]["tracked"]
        for name, pair in generator.items():
            checksum = pair[side]["sha256"]
            if (recorded.get(name) != checksum or
                    hashlib.sha256(git_bytes(identity["revision"], name)).hexdigest() != checksum):
                raise RuntimeError("Fresh CLI generation is not bound to corrected exporter/parser snapshots")
        compilation = read(output / "compilation.json")
        if (compilation["generated_sources_sha256"] != source_manifest["files_sha256"] or
                compilation["source_identity"] != identity or
                compilation.get("source_manifest_recording") != "recorded-before-and-verified-after-build"):
            raise RuntimeError("Fresh postimages differ from retained CLI/build source identity")
    else:
        original_identity(identity)
        for name, pair in generator.items():
            if hashlib.sha256(git_bytes(bench.BASELINE_REVISION, name)).hexdigest() != pair[side]["sha256"]:
                raise RuntimeError("Preimage generator is not the untouched original exporter/parser")
    return generation


def create_correction(spec, destination):
    """Seal fresh CLI delta proofs; no source generation or inference occurs here.

    spec: generator_before/after maps of the two changed Python files; evidence
    list (CLI command cards/logs and independent helper tests); cases mapping to
    original_output, fresh_before_output, fresh_after_output, and ufo_evidence.
    Fresh before/after generation records and physical cards are mandatory.
    """
    destination = path(destination)
    if destination.exists():
        raise RuntimeError("Correction destination exists; evidence is immutable")
    generator_names = {"madgraph/iolibs/export_v4.py", "madgraph/iolibs/ufo_expression_parsers.py"}
    if set(spec["generator_before"]) != generator_names or set(spec["generator_after"]) != generator_names:
        raise RuntimeError("Both exact exporter/parser snapshots are required")
    if not spec.get("evidence") or not spec.get("cases"):
        raise RuntimeError("Missing CLI/helper evidence or cases")
    retained = destination.with_suffix(".files")
    def retain(value):
        source = path(value)
        checksum = workers.digest(source)
        target = retained / checksum
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(source, target)
        return {**bound_file(target), "original_path": relative(source)}
    evidence = [retain(item) for item in spec["evidence"]]
    generator = {}
    for name in sorted(generator_names):
        before, after = path(spec["generator_before"][name]), path(spec["generator_after"][name])
        if before.read_bytes() == after.read_bytes():
            raise RuntimeError("Expected exporter/parser correction is absent")
        generator[name] = {"before": retain(before), "after": retain(after),
                           "diff": "".join(difflib.unified_diff(before.read_text().splitlines(True),
                                  after.read_text().splitlines(True), fromfile=name+".before", tofile=name+".after"))}
    cases = {}
    for case_id, item in spec["cases"].items():
        parent, before, after = (path(item[key]) for key in
                                ("original_output", "fresh_before_output", "fresh_after_output"))
        parent_generation = read(parent / "generation.json")
        original_identity(parent_generation["source_identity"])
        a, b, p = manifest(before), manifest(after), manifest(parent)
        anchor = anchor_original(case_id, parent, p)
        generation_model_binding(before, a, generator, False)
        generation_model_binding(after, b, generator, True)
        expected = {name: b["files"].get(name) for name in MODEL_FILES}
        verify_model_delta(a, b, expected)
        if not delta(a["files"], b["files"]):
            raise RuntimeError("Fresh CLI proof contains no correction")
        if model_subset(p) != model_subset(a):
            raise RuntimeError("Original model differs from fresh CLI preimage")
        cards = [retain(directory / "Cards/param_card.dat") for directory in (parent, before, after)]
        if len({card["sha256"] for card in cards}) != 1:
            raise RuntimeError("Correction changes physical inputs")
        generations = [retain(directory / "generation.json") for directory in (parent, before, after)]
        cases[case_id] = {"parent_output": relative(parent), "parent_manifest": p,
                         "fresh_before_manifest": a, "fresh_after_manifest": b,
                         "original_generation": parent_generation, "original_anchor": anchor,
                         "case_definition": original_case_definition(case_id),
                         "corrected_compilation": retain(after / "compilation.json"),
                         "generation_records": generations, "physical_cards": cards,
                         "ufo_evidence": retain(item["ufo_evidence"]),
                         "cli_evidence": [retain(value) for value in item["cli_evidence"]],
                         "preimages": {name: retain(before / name) if a["files"].get(name) else None for name in MODEL_FILES},
                         "postimages": {name: retain(after / name) if expected[name] else None
                                        for name in MODEL_FILES},
                         "changed_model_files": sorted(delta(model_subset(a), model_subset(b))),
                         "other_cli_source_changes": sorted(delta(a["files"], b["files"]) - set(MODEL_FILES))}
    result = seal({"schema": CORRECTION, "generator": generator, "evidence": evidence, "cases": cases})
    destination.parent.mkdir(parents=True, exist_ok=True)
    bench.dump(destination, result)
    validate_correction(destination)
    return result


def validate_correction(filename, live=False):
    data = verified(read(filename), CORRECTION)
    for item in data["evidence"]:
        check_file(item)
    for pair in data["generator"].values():
        a, b = check_file(pair["before"]), check_file(pair["after"])
        # Diff is documentary; exact source hashes are authoritative.
        if a.read_bytes() == b.read_bytes():
            raise RuntimeError("Missing generator correction")
    for case_id, case in data["cases"].items():
        if case["case_definition"] != original_case_definition(case_id):
            raise RuntimeError("Correction case definition differs from committed process")
        original_identity(case["original_generation"]["source_identity"])
        if anchor_original(case_id, case["parent_output"], case["parent_manifest"], live=live) != case["original_anchor"]:
            raise RuntimeError("Original export evidence anchor changed")
        if (bench.content_digest(case["original_generation"]) != case["original_anchor"]["generation_sha256"] or
                read(check_file(case["generation_records"][0])) != case["original_generation"]):
            raise RuntimeError("Original generation differs from committed anchor")
        corrected_generation = read(check_file(case["generation_records"][2]))
        corrected_identity = corrected_generation["source_identity"]
        if (seal(corrected_identity) != corrected_identity or corrected_identity.get("dirty_patch_sha256") is not None or
                corrected_identity.get("tree_state_sha256") != bench.content_digest(corrected_identity["tree_state"])):
            raise RuntimeError("Corrected generator identity is invalid")
        for name, pair in data["generator"].items():
            if (corrected_identity["tree_state"]["tracked"].get(name) != pair["after"]["sha256"] or
                    hashlib.sha256(git_bytes(corrected_identity["revision"], name)).hexdigest() != pair["after"]["sha256"] or
                    hashlib.sha256(git_bytes(bench.BASELINE_REVISION, name)).hexdigest() != pair["before"]["sha256"]):
                raise RuntimeError("Retained generator snapshots do not match actual generation revision")
        compilation = read(check_file(case["corrected_compilation"]))
        if (compilation["generated_sources_sha256"] != case["fresh_after_manifest"]["files_sha256"] or
                compilation["source_identity"] != corrected_identity):
            raise RuntimeError("Retained corrected build differs from model postimages")
        if not case["cli_evidence"]:
            raise RuntimeError("Missing per-case fresh CLI command/log evidence")
        for item in case["generation_records"] + case["physical_cards"] + case["cli_evidence"] + [case["ufo_evidence"]]:
            check_file(item)
        if len({item["sha256"] for item in case["physical_cards"]}) != 1:
            raise RuntimeError("Correction physical cards differ")
        for name, preimage in case["preimages"].items():
            if preimage:
                source = check_file(preimage)
                if case["fresh_before_manifest"]["files"].get(name) != {"sha256": preimage["sha256"], "bytes": source.stat().st_size}:
                    raise RuntimeError("Incorrect retained model preimage")
        expected = {}
        for name in MODEL_FILES:
            item = case["postimages"][name]
            if item:
                target = check_file(item)
                expected[name] = {"sha256": item["sha256"], "bytes": target.stat().st_size}
            else:
                expected[name] = None
        verify_model_delta(case["fresh_before_manifest"], case["fresh_after_manifest"], expected)
        if model_subset(case["parent_manifest"]) != model_subset(case["fresh_before_manifest"]):
            raise RuntimeError("Fresh model preimage is not original")
        if live and manifest(case["parent_output"]) != case["parent_manifest"]:
            raise RuntimeError("Original parent sources changed")
    return data


def validate_config(config, core=True):
    fixed = {"relative_tolerance": 1e-7, "scale_tolerance": 1e-10, "max_runtime_ratio": 1.05}
    if core:
        anchor = json.loads(git_bytes(ORIGINAL_EVIDENCE_REVISION, "docs/benchmarkfolder/cases.json"))
        physical = lambda case: {key: case.get(key) for key in ("id", "model", "process", "nfinal", "complex_mass_scheme")}
        if ([physical(case) for case in config["cases"]] != [physical(case) for case in anchor["cases"]] or
                any(config.get(key) != anchor.get(key) for key in ("seed", "energy_gev"))):
            raise RuntimeError("Core process inventory/physical definitions differ from reviewed eight-case specification")
        fixed.update(points_per_case=100)
        if not isinstance(config.get("repetitions"), int) or config["repetitions"] < 7:
            raise RuntimeError("At least seven timing repetitions are required")
    if any(config.get(key) != value for key, value in fixed.items()):
        raise RuntimeError("Configuration changes the reviewed v2 gates")
    if core and config.get("madloop_parameters", {}).get("MLStabThres") != 1e-9:
        raise RuntimeError("Core v2 stability threshold must remain 1e-9")
    if config.get("madloop_parameters", {}).get("COLLIERRequiredAccuracy") != 1e-12:
        raise RuntimeError("All v2 roles require common Collier accuracy 1e-12")
    for case in config["cases"]:
        if core and case.get("madloop_parameters", {}).get("MLStabThres", 1e-9) != 1e-9:
            raise RuntimeError("Core case overrides common stability threshold")
        if case.get("madloop_parameters", {}).get("COLLIERRequiredAccuracy", 1e-12) != 1e-12:
            raise RuntimeError("Case overrides common Collier accuracy")


def create_policy(correction_path, config_path, output):
    correction = validate_correction(correction_path)
    config = read(config_path)
    validate_config(config)
    data = seal({"schema": SCHEMA, "correction": bound_file(correction_path),
                 "correction_identity_sha256": correction["identity_sha256"],
                 "configuration": bound_file(config_path),
                 "collection_identity_sha256": bench.content_digest(workers.protocol_descriptor()),
                 "implementation": {name: bound_file(ROOT / "tools" / name) for name in POLICY_SOURCES},
                 "timing_roles": {role: list(ROLES[role]) for role in TIMING_ROLES},
                 "correctness_roles": {role: list(ROLES[role]) for role in ("candidate_corrected", "original_helper_corrected")},
                 "performance_pair": ["original/6", "candidate_corrected/8"]})
    if path(output).exists():
        raise RuntimeError("Policy already exists")
    bench.dump(path(output), data)
    return data


def validate_policy(filename):
    data = verified(read(filename), SCHEMA)
    correction = validate_correction(check_file(data["correction"]))
    config = read(check_file(data["configuration"]))
    validate_config(config)
    if data["correction_identity_sha256"] != correction["identity_sha256"]:
        raise RuntimeError("Policy correction identity changed")
    if (data["performance_pair"] != ["original/6", "candidate_corrected/8"] or
            data["timing_roles"] != {role: list(ROLES[role]) for role in TIMING_ROLES} or
            data["correctness_roles"] != {role: list(ROLES[role]) for role in ("candidate_corrected", "original_helper_corrected")}):
        raise RuntimeError("Invalid policy roles or performance denominator")
    if data["collection_identity_sha256"] != bench.content_digest(workers.protocol_descriptor()):
        raise RuntimeError("Frozen collection protocol changed")
    for item in data["implementation"].values():
        check_file(item)
    return data, correction, config


def library_hashes(output):
    files = {}
    root = path(output) / "lib"
    targets = list(root.iterdir())
    for directory in list(targets):
        if directory.is_dir():
            targets.extend(directory.rglob("*"))
    for target in sorted(targets):
        if target.suffix in (".a", ".so", ".dylib") or ".so." in target.name:
            if not target.is_file():
                raise RuntimeError(f"Missing linked library: {target}")
            files[str(target.relative_to(root))] = workers.digest(target)
    if "libmodel.a" not in files or "libdhelas.a" not in files:
        raise RuntimeError("Model/DHELAS archive is missing")
    return files


def validate_source_role(role, case_id, compiled, generation, correction):
    if role not in ROLES:
        raise RuntimeError("Unknown source role")
    case = correction["cases"][case_id]
    files = compiled["files"]
    if role == "candidate_corrected":
        expected_model = model_subset(case["fresh_after_manifest"])
    else:
        original_identity(generation)
        expected_model = model_subset(case["parent_manifest"] if role == "original" else case["fresh_after_manifest"])
        expected = dict(case["parent_manifest"]["files"])
        if role == "original_helper_corrected":
            for name in MODEL_FILES:
                if name in expected:
                    expected[name] = case["fresh_after_manifest"]["files"][name]
        # Only the frozen driver is permitted outside the exact parent source.
        differing = delta(expected, files)
        drivers = {name for name in differing if name.endswith("/check_sa.f")}
        if differing != drivers or len(drivers) > 1:
            raise RuntimeError("Original role contains non-model/non-driver source changes")
    if model_subset(compiled) != expected_model:
        raise RuntimeError("Source role has wrong exact model postimages")


def bind_target(role, case_id, backend, output, build_path, correction):
    output = path(output)
    build = read(build_path)
    compiled = manifest(output)
    validate_source_role(role, case_id, compiled, build["generation_source_identity"], correction)
    if compiled["files_sha256"] != build["generated_sources_sha256"]:
        raise RuntimeError("Compiled source manifest differs from build")
    directories = list(output.glob("SubProcesses/P*_*"))
    if len(directories) != 1:
        raise RuntimeError("Expected one worker subprocess")
    directory = directories[0]
    for name, key in (("check", "executable_sha256"), ("check_sa.f", "driver_sha256"),
                      ("check_fresh", "fresh_executable_sha256")):
        if workers.digest(directory / name) != build[key]:
            raise RuntimeError(f"Worker {name} differs from build")
    # Derive dimensions/prefix, then demand byte-for-byte frozen driver output.
    import re
    driver = (directory / "check_sa.f").read_text()
    template = (ROOT / "tools/ninjars_benchmark_worker.f").read_text()
    prefix = re.search(r"CALL\s+(\w*)GET_ANSWER_DIMENSION", driver, re.I).group(1)
    nexternal = re.search(r"PARAMETER\s*\(NEXTERNAL\s*=\s*(\d+)\)", driver, re.I).group(1)
    nborn = re.search(r"PARAMETER\s*\(NSQSO_BORN\s*=\s*(\d+)\)", driver, re.I).group(1)
    if int(nexternal) != correction["cases"][case_id]["case_definition"]["nfinal"] + 2:
        raise RuntimeError("Worker external dimension differs from anchored physical process")
    expected_driver = template.replace("@PREFIX@", prefix).replace("@NEXTERNAL@", nexternal).replace("@NBORN@", nborn)
    if driver != expected_driver:
        raise RuntimeError("Worker driver differs from frozen protocol")
    fresh_template = (ROOT / "tools/ninjars_benchmark_driver.f").read_text()
    expected_fresh = fresh_template.replace("@PREFIX@", prefix).replace("@NEXTERNAL@", nexternal).replace("@NBORN@", nborn)
    import hashlib
    if hashlib.sha256(expected_fresh.encode()).hexdigest() != build["fresh_driver_sha256"]:
        raise RuntimeError("Fresh parity driver differs from frozen source")
    workers.validate_param_card(directory, (output / "Cards/param_card.dat").read_bytes())
    if workers.digest(output / "Cards/param_card.dat") != correction["cases"][case_id]["physical_cards"][0]["sha256"]:
        raise RuntimeError("Role changes physical card")
    libraries = library_hashes(output)
    parent_libraries = library_hashes(correction["cases"][case_id]["parent_output"])
    if role != "candidate_corrected":
        compare = {k: v for k, v in libraries.items() if k != "libmodel.a"}
        if compare != {k: v for k, v in parent_libraries.items() if k != "libmodel.a"}:
            raise RuntimeError("Original role changes numerical/DHELAS libraries")
        if role == "original" and libraries != parent_libraries:
            raise RuntimeError("Untouched original model archive changed")
    rebuild = None
    if role == "original_helper_corrected":
        original_rebuild = output / "correction-build.json"
        verify_rebuild(original_rebuild, output, correction["identity_sha256"])
        retained_rebuild = Path(build_path).parent / "correction-build.json"
        if retained_rebuild.exists() and retained_rebuild.read_bytes() != original_rebuild.read_bytes():
            raise RuntimeError("Existing retained rebuild record differs")
        if not retained_rebuild.exists():
            shutil.copy2(original_rebuild, retained_rebuild)
        rebuild = bound_file(retained_rebuild)
    return {"role": role, "case": case_id, "backend": backend, "output": relative(output),
            "build": bound_file(build_path), "compiled_manifest": compiled,
            "generation_source_identity": build["generation_source_identity"],
            "libraries": libraries, "original_libraries": parent_libraries, "correction_build": rebuild,
            "executable_sha256": build["executable_sha256"],
            "fresh_executable_sha256": build["fresh_executable_sha256"],
            "model_identity_sha256": bench.content_digest(model_subset(compiled))}


def verify_rebuild(filename, output, correction_identity):
    data = verified(read(filename), "forced-model-rebuild-v2")
    if (data["correction_identity_sha256"] != correction_identity or
            data["prebuild_model_objects"] or data["prebuild_model_archive_exists"] or
            data["prelink_executable_exists"] or not data["model_objects"]):
        raise RuntimeError("Missing genuine forced model rebuild")
    if data["model_archive_sha256"] != workers.digest(path(output) / "lib/libmodel.a"):
        raise RuntimeError("Stale model archive")
    for name, checksum in data["model_objects"].items():
        if workers.digest(path(output) / name) != checksum:
            raise RuntimeError("Stale model object")
    for log in data["logs"]:
        check_file(log)
    return data


def require_no_rejections(report):
    status = read(path(report) / "run-status.json")
    selected = set(status["requested_cases"])
    for filename in (path(report) / "rejected").glob("*.json"):
        value = read(filename)
        if value.get("case") in selected or any(filename.name.startswith(case+"-") for case in selected):
            raise RuntimeError("Selected case has retained failure/cleanup evidence")


def bind_run(report, role_map, outputs, correction_path, destination):
    correction = validate_correction(correction_path, live=True)
    require_no_rejections(report)
    _, records = workers.checked_series(path(report), minimum=2)
    bindings = {}
    for case_id, source, backend, repetition in records:
        if repetition:
            continue
        role = role_map[source]
        if backend not in ROLES[role]:
            raise RuntimeError("Backend does not belong to source role")
        key = f"{case_id}-{source}-{backend}"
        bindings[key] = bind_target(role, case_id, backend, outputs[key],
            path(report) / "builds" / key / "build.json", correction)
    run_inventory = {relative(item): workers.digest(item) for item in path(report).rglob("*")
                     if item.is_file() and item.name != "role-bindings.json"}
    result = seal({"schema": "source-role-bindings-v2", "report": relative(report),
                   "run_files_sha256": run_inventory,
                   "correction_identity_sha256": correction["identity_sha256"], "targets": bindings})
    if path(destination).exists():
        raise RuntimeError("Binding destination already exists")
    bench.dump(path(destination), result)
    return result


def checked_bindings(filename, correction, live=False):
    bindings = verified(read(filename), "source-role-bindings-v2")
    require_no_rejections(bindings["report"])
    if bindings["correction_identity_sha256"] != correction["identity_sha256"]:
        raise RuntimeError("Mixed correction identities")
    for name, checksum in bindings["run_files_sha256"].items():
        if workers.digest(path(name)) != checksum:
            raise RuntimeError("Bound run evidence changed")
    for item in bindings["targets"].values():
        build_path = check_file(item["build"])
        build = read(build_path)
        compiled = read(build_path.parent / "compiled-sources.json")
        if (compiled != item["compiled_manifest"] or compiled["files_sha256"] != bench.content_digest(compiled["files"]) or
                compiled["files_sha256"] != build["generated_sources_sha256"] or
                item["executable_sha256"] != build["executable_sha256"] or
                item["fresh_executable_sha256"] != build["fresh_executable_sha256"] or
                item["generation_source_identity"] != build["generation_source_identity"] or
                workers.digest(build_path.parent / "worker.f") != build["driver_sha256"] or
                item["model_identity_sha256"] != bench.content_digest(model_subset(compiled))):
            raise RuntimeError("Retained compiled source/runtime binding changed")
        validate_source_role(item["role"], item["case"], compiled, item["generation_source_identity"], correction)
        if item["role"] != "candidate_corrected":
            libraries, original = item["libraries"], item["original_libraries"]
            if {k: v for k, v in libraries.items() if k != "libmodel.a"} != {k: v for k, v in original.items() if k != "libmodel.a"}:
                raise RuntimeError("Original role changes non-model libraries")
            if item["role"] == "original" and libraries != original:
                raise RuntimeError("Original timing model library changed")
        if item["role"] == "original_helper_corrected":
            rebuilt = verified(read(check_file(item["correction_build"])), "forced-model-rebuild-v2")
            if (rebuilt["correction_identity_sha256"] != correction["identity_sha256"] or
                    rebuilt["prebuild_model_objects"] or rebuilt["prebuild_model_archive_exists"] or
                    rebuilt["prelink_executable_exists"] or not rebuilt["model_objects"] or
                    rebuilt["model_archive_sha256"] != item["libraries"]["libmodel.a"] or
                    rebuilt["fresh_executable_sha256"] != item["fresh_executable_sha256"]):
                raise RuntimeError("Retained forced rebuild identity is incomplete or stale")
            for log in rebuilt["logs"]:
                check_file(log)
        if live:
            actual = bind_target(item["role"], item["case"], item["backend"], item["output"], build_path, correction)
            if actual != item:
                raise RuntimeError("Bound source/runtime changed")
    return bindings


def validate_adapter(protocol, policy):
    adapter = protocol.get("input_adapter")
    if adapter is None or adapter["sha256"] != policy["implementation"]["ninjars_benchmark_references_v2.py"]["sha256"]:
        raise RuntimeError("Collection adapter differs from v2 policy implementation")
    check_file(adapter)


def create_proof(policy_path, binding_paths, output=None):
    policy, correction, _ = validate_policy(policy_path)
    coverage = {role: {"dp": set(), "qp": set(), "loop_induced": False, "split_orders": False}
                for role in ROLES}
    files, entries = {}, []
    for filename in binding_paths:
        bindings = checked_bindings(filename, correction)
        report = path(bindings["report"])
        protocol, records = workers.checked_series(report, minimum=2)
        validate_adapter(protocol, policy)
        validate_config(protocol["configuration"], core=False)
        definitions = {case["id"]: case for case in protocol["configuration"]["cases"]}
        for item in (path(filename), report / "protocol.json", report / "run-status.json"):
            files[relative(item)] = workers.digest(item)
        for (case_id, source, backend, repetition), (record_path, data) in records.items():
            files[relative(record_path)] = workers.digest(record_path)
            if physical_definition(definitions[case_id]) != correction["cases"][case_id]["case_definition"]:
                raise RuntimeError("Representative classification differs from actual anchored process")
            key = f"{case_id}-{source}-{backend}"
            binding = bindings["targets"][key]
            if data["executable_sha256"] != binding["executable_sha256"]:
                raise RuntimeError("Proof uses a different executable")
            if repetition:
                continue
            fresh_path = report / "data" / f"{key}-fresh.json"
            fresh = read(fresh_path)
            if fresh["executable_sha256"] != binding["fresh_executable_sha256"]:
                raise RuntimeError("Fresh proof executable differs")
            parity = workers.numerical_parity(data, fresh, protocol["configuration"])
            files[relative(fresh_path)] = workers.digest(fresh_path)
            role = binding["role"]
            statuses = {row["status"] // 100 for row in data["rows"]}
            for digit, kind in ((2, "dp"), (3, "qp")):
                if digit in statuses:
                    coverage[role][kind].add(backend)
            coverage[role]["loop_induced"] |= ("[sqrvirt=" in definitions[case_id]["process"] and
                                               data["answer_dimensions"]["born_orders"] == 0)
            coverage[role]["split_orders"] |= data["answer_dimensions"]["loop_orders"] > 1
            entries.append({"role": role, "case": case_id, "backend": backend, **parity,
                            "binding": binding})
    for role, covered in coverage.items():
        qp = {1, 6, 8} if role == "candidate_corrected" else {1, 6}
        if (covered["dp"] != set(ROLES[role]) or not qp.issubset(covered["qp"]) or
                not covered["loop_induced"] or not covered["split_orders"]):
            raise RuntimeError(f"Incomplete role-aware representative coverage: {role}: {covered}")
        covered["dp"], covered["qp"] = sorted(covered["dp"]), sorted(covered["qp"])
    evidence = seal({"schema": PROOF, "policy_identity_sha256": policy["identity_sha256"],
                     "correction_identity_sha256": correction["identity_sha256"],
                     "bindings": [bound_file(item) for item in binding_paths],
                     "coverage": coverage, "entries": entries, "files_sha256": files})
    if output is not None:
        if path(output).exists():
            raise RuntimeError("Proof destination exists")
        bench.dump(path(output), evidence)
    return evidence


def validate_proof(filename, policy_path):
    data = verified(read(filename), PROOF)
    policy, correction, _ = validate_policy(policy_path)
    if (data["policy_identity_sha256"] != policy["identity_sha256"] or
            data["correction_identity_sha256"] != correction["identity_sha256"]):
        raise RuntimeError("Proof uses different policy/correction")
    for name, checksum in data["files_sha256"].items():
        if workers.digest(path(name)) != checksum:
            raise RuntimeError("Representative proof evidence changed")
    # Recompute coverage and parity instead of trusting sealed booleans.
    actual = create_proof(policy_path, [check_file(item) for item in data["bindings"]])
    if actual != data:
        raise RuntimeError("Representative proof summary does not match raw evidence")
    return data


def compare_rows(groups, pairs=None, scales=None):
    first = next(iter(groups.values()))
    if any([(r["point"], r["order"]) for r in rows] !=
           [(r["point"], r["order"]) for r in first] for rows in groups.values()):
        raise RuntimeError("Different physical point/order streams")
    if scales is None:
        scales = {}
        for rows in groups.values():
            for row in rows:
                scales[row["point"]] = max(scales.get(row["point"], 0.), *map(abs, row["coefficients"]))
    failures = []
    pairs = list(pairs if pairs is not None else itertools.combinations(groups, 2))
    for left, right in pairs:
        for a, b in zip(groups[left], groups[right]):
            for coefficient, (x, y) in enumerate(zip(a["coefficients"], b["coefficients"])):
                if not math.isfinite(x) or not math.isfinite(y):
                    raise RuntimeError("Nonfinite comparison input")
                tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scales[a["point"]]
                if abs(x-y) > tolerance:
                    failures.append({"targets": [left, right], "point": a["point"],
                                     "order": a["order"], "coefficient": coefficient,
                                     "values": [x, y], "tolerance": tolerance})
    return {"passed": not failures, "pairs": len(pairs), "failures": failures}


def build_dataset(policy_path, proof_path, timing_binding_path, reference_binding_path):
    policy, correction, config = validate_policy(policy_path)
    proof = validate_proof(proof_path, policy_path)
    timing = checked_bindings(timing_binding_path, correction)
    reference = checked_bindings(reference_binding_path, correction)
    timing_protocol, timed = workers.checked_series(path(timing["report"]), minimum=7, require_core=True)
    reference_protocol, refs = workers.checked_series(path(reference["report"]), minimum=2)
    validate_adapter(timing_protocol, policy)
    validate_adapter(reference_protocol, policy)
    candidate_proof_identities = {entry["binding"]["generation_source_identity"].get("tree_state_sha256")
                                  for entry in proof["entries"] if entry["role"] == "candidate_corrected"}
    for item in timing["targets"].values():
        if item["role"] == "candidate_corrected":
            identity = item["generation_source_identity"].get("tree_state_sha256")
            if identity is None or candidate_proof_identities != {identity}:
                raise RuntimeError("Candidate generator differs from refreshed representative proof")
    if timing_protocol["configuration"] != config or reference_protocol["configuration"] != config:
        raise RuntimeError("Dataset configurations differ from v2 policy")
    timed_cases = {key[0] for key in timed}
    if timed_cases != {key[0] for key in refs}:
        raise RuntimeError("Corrected references do not cover every timed case")
    results, bound_inputs = {}, {}
    for case_id in sorted(timed_cases):
        groups, firsts = {}, {}
        for records, bindings in ((timed, timing), (refs, reference)):
            for (case, source, backend, rep), (record_path, data) in records.items():
                if case != case_id:
                    continue
                item = bindings["targets"][f"{case}-{source}-{backend}"]
                role = item["role"]
                if ((records is timed and role not in TIMING_ROLES) or
                        (records is refs and role != "original_helper_corrected")):
                    raise RuntimeError("Wrong source role in timing/reference series")
                if data["executable_sha256"] != item["executable_sha256"]:
                    raise RuntimeError("Measurement executable differs from bound role")
                numerical = data["shared_numerical_settings"]
                if (numerical.get("collierrequiredaccuracy") != 1e-12 or
                        numerical.get("mlstabthres") != 1e-9 or
                        numerical.get("colliercomputeuvpoles") is not True or
                        numerical.get("colliercomputeirpoles") is not True or
                        (backend != 7 and numerical.get("selected_backend_qp_integrand") is not True)):
                    raise RuntimeError("Measurements do not use required full precision/pole/settings policy")
                bound_inputs[relative(record_path)] = workers.digest(record_path)
                key = f"{role}/{backend}"
                groups.setdefault(key, []).append(data)
                if rep == 0:
                    firsts[key] = data
        expected = {f"{role}/{backend}" for role in ROLES for backend in ROLES[role]}
        if set(groups) != expected:
            raise RuntimeError("Incomplete or duplicate source-role cohort")
        inputs = {(d["points_sha256"], d["parameter_card_sha256"], d["cpu"],
                   json.dumps({k: v for k, v in d["shared_numerical_settings"].items()
                               if k != "selected_backend_qp_integrand"}, sort_keys=True)) for d in firsts.values()}
        if len(inputs) != 1:
            raise RuntimeError("Source roles use different physical/numerical inputs")
        corrected = {key: value["rows"] for key, value in firsts.items() if not key.startswith("original/")}
        agreement = compare_rows(corrected)
        all_rows = {key: value["rows"] for key, value in firsts.items()}
        diagnostic_pairs = [(f"original/{b}", f"original_helper_corrected/{b}") for b in (1, 6, 7)]
        diagnostic_pairs += [(f"original/{b}", f"candidate_corrected/{b}") for b in (1, 6, 7)]
        diagnostic_pairs += [("original/6", "candidate_corrected/8")]
        original = compare_rows(all_rows, diagnostic_pairs)
        baseline = sorted(groups["original/6"], key=lambda d: d["repetition"])
        candidate = sorted(groups["candidate_corrected/8"], key=lambda d: d["repetition"])
        performance = bench.runtime_statistics([d["seconds_per_point"] for d in baseline],
                                               [d["seconds_per_point"] for d in candidate], 1.05, config["seed"])
        results[case_id] = {"corrected_agreement": agreement, "original_diagnostics": original,
                           "performance": performance,
                           "passed": agreement["passed"] and performance["passed"]}
    all_cases = {case["id"] for case in config["cases"]}
    data = seal({"schema": "madloop-policy-v2-dataset", "structural_status": "complete",
                 "scope": "all-configured-cases" if timed_cases == all_cases else "complete-case-subset",
                 "completed_cases": sorted(timed_cases), "required_cases": sorted(all_cases),
                 "policy": bound_file(policy_path), "representative_proof": bound_file(proof_path),
                 "timing_bindings": bound_file(timing_binding_path),
                 "reference_bindings": bound_file(reference_binding_path),
                 "cases": results, "measurements_sha256": bound_inputs,
                 "passed": all(result["passed"] for result in results.values()),
                 "all_cases_passed": timed_cases == all_cases and all(result["passed"] for result in results.values()),
                 "original_disagreements_preserved": True,
                 "performance_denominator": "original/6"})
    return data


def validate_dataset(filename):
    """Recompute all raw-evidence gates; a sealed or edited summary is not proof."""
    recorded = verified(read(filename), "madloop-policy-v2-dataset")
    actual = build_dataset(check_file(recorded["policy"]), check_file(recorded["representative_proof"]),
                           check_file(recorded["timing_bindings"]), check_file(recorded["reference_bindings"]))
    if actual != recorded:
        raise RuntimeError("Dataset summary differs from independently recomputed raw evidence")
    return actual


def publish(policy_path, proof_path, timing_binding_path, reference_binding_path, destination):
    data = build_dataset(policy_path, proof_path, timing_binding_path, reference_binding_path)
    target = path(destination) / data["identity_sha256"]
    if target.exists():
        raise RuntimeError("Immutable dataset destination already exists")
    target.mkdir(parents=True)
    bench.dump(target / "dataset.json", data)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("correction", "policy", "bind", "proof", "publish"))
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--correction", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--proof", type=Path)
    parser.add_argument("--bindings", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "correction":
        result = create_correction(read(args.spec), args.output)
    elif args.action == "policy":
        result = create_policy(args.correction, args.config, args.output)
    elif args.action == "bind":
        spec = read(args.spec)
        result = bind_run(spec["report"], spec["role_map"], spec["outputs"], args.correction, args.output)
    elif args.action == "proof":
        result = create_proof(args.policy, args.bindings, args.output)
    else:
        if len(args.bindings or []) != 2:
            parser.error("publish requires timing then reference --bindings")
        result = publish(args.policy, args.proof, *args.bindings, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
