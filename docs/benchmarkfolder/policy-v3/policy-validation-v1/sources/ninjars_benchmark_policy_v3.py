#!/usr/bin/env python3
"""NinjaRS-first validation; the frozen four-backend v2 policy is unchanged.

Accepted numerical calibrations and paired performance observations have distinct
readers. A failure of an external backend remains evidence, never an accepted row.
All numerical/source/runtime primitives come from the existing benchmark tools.
"""
import argparse
from collections import Counter
import copy
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import ninjars_benchmark as bench
import ninjars_benchmark_workers as workers
import ninjars_benchmark_policy_v2 as v2

ROOT = bench.ROOT
SCHEMA = "madloop-nrs-first-policy-v3"
REFERENCES = "madloop-calibration-references-v3"
DATASET = "madloop-nrs-first-dataset-v3"
PAIR_SCHEMA = "madloop-explicit-timing-pair-v3"
PAIR_SPEC = "madloop-explicit-timing-pair-spec-v3"
FINALIZATION = "nrs-first-core-correctness-finalization-v1"
PAIR_TARGETS = [["baseline", 6], ["candidate", 8]]
ROLE_MAP = {"baseline": "original", "candidate": "candidate_corrected"}
CASE_IDS = {"qcd-1", "qcd-2", "qcd-3", "qcd-4", "ew-1", "ew-2", "ew-3", "ew-4"}
NUMERICAL_REVISION = "d05b59f3277ce0d727cb2816d8b7ad45107e58c4"
GENERATION_REVISION = "b0ea105a04afa34304d437d2956e4d4672360670"
GENERATOR_ID = "34aed45638a2e18b18135179e749317c3cf9a0fdd3bb404e43fe68d23837c6de"
RUNTIME_ID = "8b454519cb6818ef1fafe37378930c276f7e85394a2333ebc9fdb00e75726cff"
ARCHIVE_SHA256 = "800a941faf3cdfdab764fdffe1d5ab73f84984a786aa194596b753af6b6f5d42"
GENERATION_ROOT = ROOT / "docs/benchmarkfolder/policy-v2/final-core-generation-v1"
USER_PRIORITY = ("Continue as planned, but focus first on getting a validated MadLoop+NinjaRS version. "
                 "Then if you uncovered issues with other reduction tools (e.g. COLLIER), then just "
                 "post an issue with an MRE to the corresponding github")
QUALIFICATION_KEYS = ("installation_export", "precision_rescue", "focused_controls",
                      "loop_induced", "regressions")
PUBLICATION_KEYS = ("four_tool_report", "profiles", "reproducible_pdf", "pushed_milestones")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def retain(filename, inputs=None, checksum=None):
    filename = v2.path(filename)
    actual = workers.digest(filename)
    require(checksum is None or checksum == actual, "Changed bound evidence: " + v2.relative(filename))
    if inputs is not None:
        inputs[v2.relative(filename)] = actual
    return filename


def checked(record, inputs=None):
    return retain(record["path"], inputs, record["sha256"])


def read(filename, inputs=None):
    return v2.read(retain(filename, inputs))


def _write(filename, value):
    filename = v2.path(filename)
    require(not filename.exists(), "Preserve the existing evidence; choose a new destination")
    bench.dump(filename, value)
    return value


def validate_configuration(config, correctness=False):
    """Allow correctness ordering/two commands, without changing physical inputs."""
    definitions = config.get("cases", [])
    require(len(definitions) == 8 and {c["id"] for c in definitions} == CASE_IDS,
            "All eight reviewed core processes are required")
    normalized = copy.deepcopy(config)
    anchor = json.loads(v2.git_bytes(v2.ORIGINAL_EVIDENCE_REVISION, "docs/benchmarkfolder/cases.json"))
    by_id = {c["id"]: c for c in definitions}
    normalized["cases"] = [by_id[c["id"]] for c in anchor["cases"]]
    if correctness:
        require(type(config.get("repetitions")) is int and config["repetitions"] >= 2,
                "Correctness needs two actual commands, not a relabelled calibration")
        normalized["repetitions"] = max(7, config["repetitions"])
    v2.validate_config(normalized)
    require(config.get("madloop_parameters") == {"MLStabThres": 1e-9, "COLLIERRequiredAccuracy": 1e-12},
            "Unreviewed additional core numerical override")
    for case in definitions:
        require(set(case.get("madloop_parameters", {})).issubset({"MLStabThres", "COLLIERRequiredAccuracy"}),
                "Unreviewed case numerical override")
    require(len({c["id"] for c in definitions}) == len(definitions), "Duplicate core case")
    return config


def create_policy(v2_policy, proof, output, plan=None):
    old, correction, config = v2.validate_policy(v2_policy)
    validate_configuration(config)
    v2.validate_proof(proof, v2_policy)
    plan = v2.path(plan or ROOT / "references/PLAN.md")
    require(USER_PRIORITY in " ".join(plan.read_text().split()), "User priority amendment is absent from PLAN.md")
    value = v2.seal({"schema": SCHEMA, "v2_policy": v2.bound_file(v2_policy),
        "representative_proof": v2.bound_file(proof), "configuration": old["configuration"],
        "correction": old["correction"], "correction_identity_sha256": correction["identity_sha256"],
        "worker_protocol": workers.protocol_descriptor(), "user_priority": USER_PRIORITY,
        "amendment_source": "references/PLAN.md (quoted statement retained here)",
        "required_cases": sorted(CASE_IDS), "correctness_references": [1, 6],
        "pair_targets": PAIR_TARGETS, "performance_denominator": "original/6",
        "numerical_revision": NUMERICAL_REVISION, "generation_revision": GENERATION_REVISION,
        "candidate_generator_identity_sha256": GENERATOR_ID, "candidate_runtime_identity_sha256": RUNTIME_ID,
        "candidate_archive_sha256": ARCHIVE_SHA256,
        "candidate_generation_status": v2.bound_file(GENERATION_ROOT / "status.json"),
        "candidate_generation_inputs": {cid: {name: v2.bound_file(GENERATION_ROOT / "inputs/candidate" / cid / name)
            for name in ("generation.json", "compilation.json", "compiled-sources.json")} for cid in sorted(CASE_IDS)},
        "candidate_fixed_points": {cid: v2.bound_file(GENERATION_ROOT / "points" / (cid + ".dat")) for cid in sorted(CASE_IDS)},
        "implementation": v2.bound_file(Path(__file__))})
    return _write(output, value)


def validate_policy(filename, inputs=None):
    policy = v2.verified(read(filename, inputs), SCHEMA)
    old_path = checked(policy["v2_policy"], inputs)
    old, correction, config = v2.validate_policy(old_path)
    checked(policy["configuration"], inputs); checked(policy["correction"], inputs)
    proof = v2.validate_proof(checked(policy["representative_proof"], inputs), old_path)
    validate_configuration(config)
    require(policy["configuration"] == old["configuration"] and policy["correction"] == old["correction"]
            and policy["correction_identity_sha256"] == correction["identity_sha256"], "Mixed v3 physical/model policy")
    fixed = {"user_priority": USER_PRIORITY, "required_cases": sorted(CASE_IDS),
        "correctness_references": [1, 6], "pair_targets": PAIR_TARGETS,
        "performance_denominator": "original/6", "numerical_revision": NUMERICAL_REVISION,
        "generation_revision": GENERATION_REVISION, "candidate_generator_identity_sha256": GENERATOR_ID,
        "candidate_runtime_identity_sha256": RUNTIME_ID, "candidate_archive_sha256": ARCHIVE_SHA256,
        "worker_protocol": workers.protocol_descriptor()}
    require(all(policy.get(k) == value for k, value in fixed.items()), "Changed v3 roles, physics or runtime gate")
    generation = read(checked(policy["candidate_generation_status"], inputs), inputs)
    require(generation.get("complete") is True and generation["git_revision"] == GENERATION_REVISION
            and generation["generator_tree_state_sha256"] == GENERATOR_ID, "Wrong actual CLI generation cohort")
    require(set(policy["candidate_generation_inputs"]) == CASE_IDS, "Missing actual CLI source anchors")
    require(set(policy["candidate_fixed_points"]) == CASE_IDS, "Missing frozen core point streams")
    for cid, record in policy["candidate_fixed_points"].items():
        require(checked(record, inputs) == GENERATION_ROOT / "points" / (cid + ".dat"), "Wrong frozen point anchor")
    for cid, files in policy["candidate_generation_inputs"].items():
        require(set(files) == {"generation.json", "compilation.json", "compiled-sources.json"}, "Incomplete CLI generation binding")
        for record in files.values():
            checked(record, inputs)
        require(read(checked(files["generation.json"], inputs), inputs) == generation["cases"][cid]["generation"]
                and read(checked(files["compilation.json"], inputs), inputs) == generation["cases"][cid]["compilation"],
                "CLI source anchor differs from actual generation record")
    require(checked(policy["implementation"], inputs) == Path(__file__).resolve(), "Wrong v3 policy implementation")
    return policy, correction, config, proof


def validate_rows(record, backend, points=100):
    """Strict normal-mode observations only; calibration is not timing promotion."""
    require(record.get("backend") == backend and type(backend) is int, "Wrong numerical backend")
    require(type(record.get("loops")) is int and record["loops"] > 0, "Invalid traversal count")
    dimensions = record["answer_dimensions"]
    allocated, nso, born = (dimensions[k] for k in ("allocated", "loop_orders", "born_orders"))
    require(all(type(v) is int for v in (allocated, nso, born)) and nso >= 1 and 0 <= born <= nso
            and allocated == max(nso, born), "Incomplete split-order dimensions")
    rows = record["rows"]
    require([(r["point"], r["order"]) for r in rows] ==
            [(p, o) for p in range(1, points + 1) for o in range(nso + 1)], "Missing, duplicated or reordered point/order rows")
    counts, statuses = Counter(), {}
    for row in rows:
        require(type(row["point"]) is int and type(row["order"]) is int, "Invalid point/order index type")
        require(type(row["status"]) is int, "Invalid status type")
        bench.validate_status(row["status"], backend)
        require(statuses.setdefault(row["point"], row["status"]) == row["status"], "Inconsistent point/split status")
        values = row["coefficients"]
        require(len(values) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in values),
                "Nonfinite or missing Born/Laurent coefficient")
        accuracy = row["accuracy"]
        require(type(accuracy) in (int, float) and math.isfinite(accuracy) and accuracy >= 0,
                "Invalid or missing stability estimate")
        if row["order"] == 0:
            counts[str(row["status"])] += 1
    require(dict(counts) == record["status_counts"], "Point status histogram differs from rows")
    timed = record["timed_status_counts"]
    for code, count in timed.items():
        require(str(int(code)) == code and type(count) is int and count > 0, "Invalid timed status count")
        bench.validate_status(int(code), backend)
    require(sum(timed.values()) == points * record["loops"], "Timed histogram omits evaluations")
    for key in ("seconds", "seconds_per_point", "cold_seconds"):
        value = record[key]
        require(type(value) in (int, float) and math.isfinite(value) and value > 0, "Invalid observation time")
    require(record["seconds_per_point"] == record["seconds"] / (points * record["loops"]), "Inconsistent observation time/count")
    settings = record["shared_numerical_settings"]
    fixed = {"ctmoderun": -1, "ctmodeinit": 1, "mlstabthres": 1e-9,
             "collierrequiredaccuracy": 1e-12, "colliercomputeuvpoles": True,
             "colliercomputeirpoles": True, "useloopfilter": False,
             "selected_backend_qp_integrand": True}
    require(all(settings.get(k) == value and (type(value) is not bool or type(settings.get(k)) is bool)
                for k, value in fixed.items()), "Changed precision, stability, pole or filter settings")
    require(settings == expected_settings(), "Changed a shared frozen numerical setting")
    return rows


def _card_values(text, defaults=False):
    lines = text.splitlines()
    values = {}
    for index, line in enumerate(lines):
        if not line.startswith("#"):
            continue
        name = line[1:].strip().lower()
        following = None
        for next_line in lines[index + 1:]:
            if defaults:
                default = re.fullmatch(r"\s*!\s*Default\s*::\s*(.*?)\s*", next_line)
                if default:
                    following = default.group(1)
                    break
            if not next_line.strip() or next_line.lstrip().startswith("!"):
                continue
            if next_line.lstrip().startswith("#"):
                break
            following = next_line.strip()
            break
        require(following is not None and name not in values, "Malformed numerical card")
        if following.lower() in (".true.", ".false."):
            value = following.lower() == ".true."
        elif re.fullmatch(r"[-+]?\d+", following):
            value = int(following)
        else:
            try:
                value = float(following.lower().replace("d", "e"))
            except ValueError:
                value = following
        values[name] = value
    return values


@lru_cache(maxsize=1)
def _expected_base_settings():
    values = _card_values(v2.git_bytes(bench.BASELINE_REVISION, "Template/loop_material/StandAlone/Cards/MadLoopParams.dat").decode(), defaults=True)
    values.pop("mlreductionlib")
    values.pop("useqpintegrandforninjars", None)
    values.update(ctmoderun=-1, ctmodeinit=1, useloopfilter=False,
                  colliercomputeuvpoles=True, colliercomputeirpoles=True,
                  mlstabthres=1e-9, collierrequiredaccuracy=1e-12,
                  selected_backend_qp_integrand=True)
    return tuple(values.items())


def expected_settings():
    """Exact shared original card defaults plus the retained core overrides."""
    return dict(_expected_base_settings())


def card_settings(filename, backend):
    """Read literal MadLoop card values without importing a generated model."""
    values = _card_values(Path(filename).read_text())
    require(str(values.pop("mlreductionlib")) == str(backend), "Numerical card selects a different backend")
    nrs = values.pop("useqpintegrandforninjars", None)
    qp = {1: values.get("useqpintegrandforcuttools"), 6: values.get("useqpintegrandforninja"), 8: nrs}[backend]
    require(qp is True, "Selected QP integrand is disabled")
    values["selected_backend_qp_integrand"] = qp
    return values


def _check_points(filename, record, case, inputs):
    filename = retain(filename, inputs, record["points_sha256"])
    retain(GENERATION_ROOT / "points" / (case["id"] + ".dat"), inputs, record["points_sha256"])
    lines = filename.read_text().splitlines()
    require(lines and lines[0] == "100" and len(lines) == 1 + 100 * (case["nfinal"] + 2), "Changed fixed 100-point stream")
    for line in lines[1:]:
        numbers = line.split()
        require(len(numbers) == 4 and all(math.isfinite(float(x)) for x in numbers), "Malformed fixed momentum stream")


def _check_target(item, correction, proof, inputs, target_schema="source-role-bindings-v2"):
    """Validate retained v2 target fields, without requiring its parent run to pass."""
    role, case, backend = item["role"], item["case"], item["backend"]
    build_path = checked(item["build"], inputs)
    build = read(build_path, inputs)
    compiled = read(build_path.parent / "compiled-sources.json", inputs)
    require(compiled == item["compiled_manifest"] and bench.content_digest(compiled["files"]) == compiled["files_sha256"]
            and compiled["files_sha256"] == build["generated_sources_sha256"], "Changed compiled source/model manifest")
    v2.validate_source_role(role, case, compiled, item["generation_source_identity"], correction)
    for key in ("executable_sha256", "generation_source_identity"):
        require(item[key] == build[key], "Target/build identity changed: " + key)
    if "fresh_executable_sha256" in item:
        require(item["fresh_executable_sha256"] == build["fresh_executable_sha256"], "Fresh executable identity changed")
    else:
        require(target_schema == "supplement-source-bindings-v1" and role == "candidate_corrected"
                and item.get("source") == "candidate", "Missing fresh identity outside known supplement-v1 target shape")
    if target_schema == "supplement-source-bindings-v1":
        require(item.get("generated_sources_sha256") == compiled["files_sha256"], "Supplement source identity changed")
        origin = read(checked(item["origin_manifest"], inputs), inputs)
        generation = read(checked(item["origin_generation"], inputs), inputs)
        compilation = read(checked(item["origin_compilation"], inputs), inputs)
        require(bench.content_digest(origin["files"]) == origin["files_sha256"] == build["original_generated_sources_sha256"]
                and compilation == build["origin_compilation"]
                and compilation["generated_sources_sha256"] == origin["files_sha256"]
                and generation["source_identity"] == item["generation_source_identity"]
                and generation["parameter_card_sha256"] == build["parameter_card_sha256"]
                and workers.digest(checked(item["physical_card"], inputs)) == build["parameter_card_sha256"],
                "Actual supplement generation/source/physical anchor differs from build")
        source_proof = read(checked(item["source_proof"], inputs), inputs)
        require(v2.physical_definition(source_proof["case"]) == correction["cases"][case]["case_definition"]
                and source_proof["full_runtime_identity_sha256"] == RUNTIME_ID
                and read(checked(source_proof["generation"], inputs), inputs) == generation
                and read(checked(source_proof["compilation"], inputs), inputs) == compilation
                and read(checked(source_proof["compiled_manifest"], inputs), inputs) == origin
                and workers.digest(checked(source_proof["physical_card"], inputs)) == build["parameter_card_sha256"],
                "Actual core source proof differs from copied generation anchors")
    driver = retain(build_path.parent / "worker.f", inputs, build["driver_sha256"]).read_text()
    prefix = re.search(r"CALL\s+(\w*)GET_ANSWER_DIMENSION", driver, re.I).group(1)
    nexternal = re.search(r"PARAMETER\s*\(NEXTERNAL\s*=\s*(\d+)\)", driver, re.I).group(1)
    nborn = re.search(r"PARAMETER\s*\(NSQSO_BORN\s*=\s*(\d+)\)", driver, re.I).group(1)
    require(int(nexternal) == correction["cases"][case]["case_definition"]["nfinal"] + 2, "Wrong worker external dimension")
    def instantiated(name):
        return (ROOT / "tools" / name).read_text().replace("@PREFIX@", prefix).replace("@NEXTERNAL@", nexternal).replace("@NBORN@", nborn)
    require(driver == instantiated("ninjars_benchmark_worker.f") and
            hashlib.sha256(instantiated("ninjars_benchmark_driver.f").encode()).hexdigest() == build["fresh_driver_sha256"],
            "Worker/fresh driver differs from the frozen protocol")
    for record in item["runtime_metadata"].values():
        checked(record, inputs)
    runtime = v2.reducer_signature(item)
    require(runtime == item["reducer_runtime"], "Changed full runtime descriptor")
    groups = v2.proof_runtime_signatures(proof["entries"])
    group = groups.get(f"{role}/{backend}", {}).get(runtime["identity_sha256"])
    require(group is not None and group["signature"] == runtime, "Unqualified source-role/runtime group")
    generation = item["generation_source_identity"]
    if role == "candidate_corrected":
        require(backend == 8 and generation.get("revision") == GENERATION_REVISION
                and generation.get("tree_state_sha256") == GENERATOR_ID
                and generation.get("dirty_patch_sha256") is None and not generation["tree_state"]["untracked"]
                and bench.content_digest(generation["tree_state"]) == GENERATOR_ID
                and runtime["identity_sha256"] == RUNTIME_ID, "Candidate differs from frozen final generation/runtime")
        require({value for name, value in item["libraries"].items() if Path(name).name == "libninjars.a"} == {ARCHIVE_SHA256},
                "Candidate archive is not the validated final installation")
        anchor = read(GENERATION_ROOT / "inputs/candidate" / case / "compiled-sources.json", inputs)
        differences = v2.delta(anchor["files"], compiled["files"])
        require(bench.content_digest(anchor["files"]) == anchor["files_sha256"] == build["original_generated_sources_sha256"]
                and all(name.endswith("/check_sa.f") for name in differences) and len(differences) <= 1,
                "Candidate numerical source differs from actual frozen CLI export")
    else:
        require(role in ("original", "original_helper_corrected") and backend in (1, 6), "Wrong reference/timing role")
        original = item["original_libraries"]
        require({k: v for k, v in item["libraries"].items() if k != "libmodel.a"} ==
                {k: v for k, v in original.items() if k != "libmodel.a"}, "Legacy numerical libraries changed")
        if role == "original":
            require(item["libraries"] == original, "Literal original model archive changed")
        else:
            rebuild = v2.verified(read(checked(item["correction_build"], inputs), inputs), "forced-model-rebuild-v2")
            require(rebuild["correction_identity_sha256"] == correction["identity_sha256"]
                    and not rebuild["prebuild_model_objects"] and not rebuild["prebuild_model_archive_exists"]
                    and not rebuild["prelink_executable_exists"] and rebuild["model_objects"]
                    and rebuild["model_archive_sha256"] == item["libraries"]["libmodel.a"]
                    and rebuild["fresh_executable_sha256"] == build["fresh_executable_sha256"], "Missing/stale corrected-model rebuild")
            for record in rebuild["logs"]:
                checked(record, inputs)
    return build, build_path


def _check_record_target(record, item, build, build_path, case, points, inputs):
    validate_rows(record, item["backend"])
    require((record["case"], record["source"], record["backend"]) ==
            (item["case"], "candidate" if item["role"] == "candidate_corrected" else "baseline", item["backend"]), "Wrong row source role/case")
    for key in ("executable_sha256", "generation_source_identity"):
        require(record[key] == item[key], "Measurement differs from target: " + key)
    require(record["generated_sources_sha256"] == item["compiled_manifest"]["files_sha256"], "Measurement source manifest changed")
    physical = retain(build_path.parent / "param_card.dat", inputs, record["parameter_card_sha256"])
    require(record["parameter_card_sha256"] == build["parameter_card_sha256"] ==
            correction_physical_hash(case), "Physical parameter card changed")
    require(workers.digest(physical) == record["parameter_card_sha256"], "Physical card differs")
    card = retain(build_path.parent / "MadLoopParams.dat", inputs, record["madloop_card_sha256"])
    require(record["madloop_card_sha256"] == build["madloop_card_sha256"] and
            record["shared_numerical_settings"] == card_settings(card, item["backend"]), "Effective numerical card/settings changed")
    _check_points(points, record, case, inputs)


def correction_physical_hash(case):
    # Set explicitly by callers through a local case copy; no ambient policy state.
    return case["physical_card_sha256"]


def _case(config, correction, case_id):
    value = dict(next(c for c in config["cases"] if c["id"] == case_id))
    value["physical_card_sha256"] = correction["cases"][case_id]["physical_cards"][0]["sha256"]
    return value


def _check_protocol(report, inputs):
    protocol = read(report / "protocol.json", inputs)
    descriptor = workers.protocol_descriptor()
    require(protocol.get("schema") == workers.SCHEME and protocol.get("protocol_descriptor") == descriptor
            and protocol.get("protocol_identity_sha256") == bench.content_digest(descriptor), "Frozen worker protocol changed")
    for name, checksum in descriptor["sources"].items():
        retain(report / "sources" / name, inputs, checksum)
    checked(protocol["input_adapter"], inputs)
    return protocol


def _check_raw_capture(record, inputs):
    capture = record.get("raw_capture", {})
    require(set(capture) == {*workers.OUTPUTS, "dimensions.dat"}, "Pair observation lacks complete retained raw outputs")
    paths = {name: checked(bound, inputs) for name, bound in capture.items()}
    require(len({p.parent for p in paths.values()}) == 1 and all(p.name == n for n, p in paths.items()),
            "Mixed raw-output capture directories")
    parsed = workers.parse_outputs(next(iter(paths.values())).parent, record["backend"], 100, record["loops"])
    require(all(record[key] == value for key, value in parsed.items()), "Raw output differs from published pair observation")


def _selected_failure(report, case, backend, inputs):
    for filename in sorted((report / "rejected").glob("*.json")):
        value = read(filename, inputs)
        if value.get("case") != case and not filename.name.startswith(case + "-"):
            continue
        require(value.get("backend") not in (None, backend), "Selected numerical reference has failure/cleanup evidence")


def create_references(policy_path, specification, output):
    """Snapshot selected accepted calibrations without promoting a failed run."""
    _, correction, config, proof = validate_policy(policy_path)
    spec = v2.read(specification)
    require(spec.get("schema") == "nrs-reference-selection-v3", "Wrong numerical reference selection")
    output = v2.path(output)
    assets = output.with_name(output.stem + "-assets")
    require(not output.exists() and not assets.exists(), "Choose a new reference snapshot namespace")
    assets.mkdir(parents=True)
    entries = []
    for selected in spec["entries"]:
        cid, backend = selected["case"], selected["backend"]
        require(cid in CASE_IDS and backend in (1, 6), "Unexpected calibration target")
        report = v2.path(selected["report"])
        target_dir = assets / f"{cid}-{backend}"
        target_dir.mkdir()
        old_build = checked(selected["build"])
        for name in ("build.json", "compiled-sources.json", "worker.f", "param_card.dat", "MadLoopParams.dat"):
            shutil.copy2(old_build.parent / name, target_dir / name)
        source = v2.path(selected["output"])
        shutil.copy2(source / "correction-build.json", target_dir / "correction-build.json")
        item = v2.bind_target("original_helper_corrected", cid, backend, source, target_dir / "build.json", correction)
        metadata = {}
        for name, filename in v2.runtime_metadata_files(source).items():
            destination = assets / workers.digest(filename)
            if not destination.exists():
                shutil.copy2(filename, destination)
            metadata[name] = v2.bound_file(destination)
        item.update(runtime_metadata=metadata)
        item["reducer_runtime"] = v2.reducer_signature(item)
        entry = {"case": cid, "backend": backend, "record": selected["record"], "target": item,
                 "points": selected["points"], "parent_protocol": v2.bound_file(report / "protocol.json"),
                 "parent_status": v2.bound_file(report / "run-status.json"),
                 "parent_rejections": [v2.bound_file(p) for p in sorted((report / "rejected").glob("*.json"))]}
        entries.append(entry)
    value = v2.seal({"schema": REFERENCES, "policy": v2.bound_file(policy_path),
                    "selection": v2.bound_file(specification), "entries": entries,
                    "scope": "Accepted numerical calibrations only; parent failures retained; no timing repetition promotion"})
    # Validate before writing a successful seal, while preserving a failed attempt.
    _check_references(value, correction, config, proof, {})
    return _write(output, value)


def _check_references(data, correction, config, proof, inputs):
    records, targets = {}, {}
    for entry in data["entries"]:
        cid, backend = entry["case"], entry["backend"]
        key = cid, backend
        require(cid in CASE_IDS and backend in (1, 6) and key not in records, "Duplicate or injected reference target")
        item = entry["target"]
        require((item["role"], item["case"], item["backend"]) == ("original_helper_corrected", cid, backend), "Wrong calibration source role")
        build, build_path = _check_target(item, correction, proof, inputs)
        filename = checked(entry["record"], inputs)
        record = read(filename, inputs)
        require(record["repetition"] == "calibration" and record["loops"] == 1,
                "A numerical calibration cannot be relabelled as timing repetitions")
        protocol_path = checked(entry["parent_protocol"], inputs)
        require(filename.parent == protocol_path.parent / "data", "Calibration belongs to another parent report")
        protocol = _check_protocol(protocol_path.parent, inputs)
        actual_config = protocol["configuration"]
        for name in ("seed", "energy_gev", "points_per_case", "relative_tolerance", "scale_tolerance",
                     "max_runtime_ratio", "madloop_parameters"):
            require(actual_config.get(name) == config.get(name), "Calibration configuration changes frozen numerical/physical inputs")
        definitions = actual_config["cases"]
        known = {c["id"]: c for c in config["cases"]}
        require(len({c["id"] for c in definitions}) == len(definitions) and
                all(c["id"] in known and c == known[c["id"]] for c in definitions)
                and cid in {c["id"] for c in definitions}, "Calibration source process differs from its retained protocol")
        require(record["protocol_identity_sha256"] == protocol["protocol_identity_sha256"], "Calibration protocol differs")
        require(checked(entry["parent_status"], inputs) == protocol_path.parent / "run-status.json", "Unrelated parent run status")
        require({checked(p, inputs) for p in entry["parent_rejections"]} == set((protocol_path.parent / "rejected").glob("*.json")),
                "Omitted or injected parent rejection context")
        for bound in entry["parent_rejections"]:
            rejected = read(checked(bound, inputs), inputs)
            if rejected.get("case") == cid:
                require(rejected.get("backend") not in (None, backend), "Selected reference was rejected")
        _selected_failure(protocol_path.parent, cid, backend, inputs)
        _check_record_target(record, item, build, build_path, _case(config, correction, cid), checked(entry["points"], inputs), inputs)
        records[key], targets[key] = record, item
    return records, targets


def checked_references(filename, policy_path, inputs=None):
    inputs = {} if inputs is None else inputs
    _, correction, config, proof = validate_policy(policy_path, inputs)
    data = v2.verified(read(filename, inputs), REFERENCES)
    require(checked(data["policy"], inputs) == v2.path(policy_path), "Reference policy differs")
    checked(data["selection"], inputs)
    records, targets = _check_references(data, correction, config, proof, inputs)
    return data, records, targets


def checked_pair_series(report, inputs=None):
    """Read exactly two explicit targets; never the v2 Cartesian product."""
    inputs = {} if inputs is None else inputs
    report = v2.path(report)
    protocol = _check_protocol(report, inputs)
    config = protocol["configuration"]
    validate_configuration(config)
    require(protocol.get("pair_schema") == PAIR_SCHEMA and protocol.get("pair_targets") == PAIR_TARGETS
            and protocol.get("sources") == "both" and protocol.get("backends") == [6, 8]
            and protocol.get("role_map") == ROLE_MAP and protocol.get("cpu") == 207
            and protocol.get("thread_environment") == {name: "1" for name in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS", "MKL_NUM_THREADS")}
            and protocol.get("fresh_parity") is False, "Wrong explicit target, CPU, thread or parity policy")
    repetitions = protocol["repetitions"]
    require(type(repetitions) is int and repetitions >= 7, "At least seven paired repetitions are required")
    spec = v2.verified(read(checked(protocol["input_spec"], inputs), inputs), PAIR_SPEC)
    import ninjars_benchmark_pair_v3 as pair
    pair.validate_spec(spec)  # Pure source/spec hashes; no preflight, build or engine.
    for record in spec["sources"].values():
        checked(record, inputs)
    require(protocol["input_adapter"]["sha256"] == spec["sources"]["collector"]["sha256"],
            "Protocol uses a different pair collector")
    require(spec["pair_targets"] == PAIR_TARGETS and spec["role_map"] == ROLE_MAP
            and spec["repetitions"] == repetitions and spec["cpu"] == protocol["cpu"]
            and read(checked(spec["configuration"], inputs), inputs) == config
            and type(spec["minimum_seconds"]) in (int, float) and math.isfinite(spec["minimum_seconds"])
            and spec["minimum_seconds"] >= 1, "Pair protocol differs from its exact execution specification")
    status = read(report / "run-status.json", inputs)
    cases = [c["id"] for c in config["cases"]]
    require(status.get("status") == "complete" and status.get("requested_cases") == cases
            and status.get("completed_cases") == cases, "Pair series is incomplete or failed")
    require(not list((report / "rejected").glob("*.json")), "Pair series has retained failure/cleanup evidence")
    ledger_file = checked(protocol["dispatch_ledger"], inputs)
    ledger = [json.loads(line) for line in ledger_file.read_text().splitlines()]
    require([e.get("index") for e in ledger] == list(range(len(ledger))), "Noncontiguous dispatch ledger")
    require(all(type(e.get("index")) is int and e.get("case") in cases
                and [e.get("source"), e.get("backend")] in PAIR_TARGETS
                and e.get("phase") in ("prepare", "initialize", "calibration", "measurement", "cleanup")
                and e.get("event") in ("start", "complete") for e in ledger), "Invalid dispatch target/phase/failure event")
    require(len(ledger) % 2 == 0, "Unclosed dispatch event")
    for index in range(0, len(ledger), 2):
        start, end = ledger[index:index + 2]
        require(start["event"] == "start" and end["event"] == "complete" and
                all(start.get(k) == end.get(k) for k in ("case", "phase", "source", "backend", "repetition")),
                "Overlapping or mismatched dispatch commands")
    records, expected_paths = {}, set()
    for cid in cases:
        measurements = [e for e in ledger if e.get("case") == cid and e.get("phase") == "measurement"]
        expected = [(rep, source, backend) for rep in range(repetitions)
                    for source, backend in (PAIR_TARGETS if rep % 2 == 0 else list(reversed(PAIR_TARGETS)))]
        starts = [e for e in measurements if e.get("event") == "start"]
        require([(e.get("repetition"), e.get("source"), e.get("backend")) for e in starts] == expected,
                "Timing pair is not alternating, contiguous and complete")
        require(len(measurements) == 2 * len(expected), "Missing/extra measurement dispatch event")
        for source, backend in PAIR_TARGETS:
            prefix = f"{cid}-{source}-{backend}"
            calibration_path = report / "data" / (prefix + "-calibration.json")
            calibration = read(calibration_path, inputs)
            validate_rows(calibration, backend)
            _check_raw_capture(calibration, inputs)
            require(calibration["repetition"] == "calibration" and calibration["loops"] == 1
                    and calibration["case"] == cid and calibration["source"] == source
                    and calibration["cpu"] == protocol["cpu"] and calibration["measurement_run_id"] == spec["run_id"]
                    and calibration["paired_baseline_candidate"] is True
                    and calibration["protocol_identity_sha256"] == protocol["protocol_identity_sha256"], "Invalid pair calibration")
            for phase in ("prepare", "initialize", "calibration", "cleanup"):
                events = [e for e in ledger if (e["case"], e["source"], e["backend"], e["phase"]) == (cid, source, backend, phase)]
                require(len(events) == 2, "Missing or duplicate preparation/initialization/calibration/cleanup")
                if phase == "calibration":
                    require(events[0]["loops"] == events[1]["loops"] == 1
                            and checked(events[1]["record"], inputs) == calibration_path,
                            "Calibration dispatch does not bind its actual complete stream")
            expected_paths.add(calibration_path)
            build = read(report / "builds" / prefix / "build.json", inputs)
            for rep in range(repetitions):
                filename = report / "data" / f"{prefix}-{rep}.json"
                data = read(filename, inputs)
                validate_rows(data, backend)
                _check_raw_capture(data, inputs)
                require((data["case"], data["source"], data["backend"], data["repetition"]) == (cid, source, backend, rep)
                        and data["cpu"] == protocol["cpu"] and data["measurement_run_id"] == spec["run_id"]
                        and data["paired_baseline_candidate"] is True
                        and data["protocol_identity_sha256"] == protocol["protocol_identity_sha256"]
                        and data["executable_sha256"] == build["executable_sha256"]
                        and data["generated_sources_sha256"] == build["generated_sources_sha256"], "Pair record identity drift")
                bench.validate_repetition(data, calibration, config)
                start = next(e for e in starts if (e["source"], e["backend"], e["repetition"]) == (source, backend, rep))
                index = ledger.index(start)
                require(index + 1 < len(ledger), "Unclosed dispatch")
                end = ledger[index + 1]
                require(end.get("event") == "complete" and all(end.get(k) == start.get(k) for k in
                        ("case", "phase", "source", "backend", "repetition"))
                        and data["dispatch_index"] == start["index"]
                        and start["loops"] == end["loops"] == data["loops"]
                        and checked(end["record"], inputs) == filename, "Dispatch completion does not bind its measured row")
                require(start.get("worker_sequence") == end.get("worker_sequence") == data["worker_sequence"], "Worker sequence changed")
                records[cid, source, backend, rep] = filename, data
                expected_paths.add(filename)
        for rep in range(repetitions):
            a, b = (records[cid, source, backend, rep][1] for source, backend in PAIR_TARGETS)
            require(all(a[k] == b[k] for k in ("points_sha256", "parameter_card_sha256", "cpu",
                        "shared_numerical_settings", "measurement_run_id", "answer_dimensions")), "Paired physical/numerical inputs differ")
    require(set((report / "data").glob("*.json")) == expected_paths, "Extra/injected pair target or numerical record")
    return protocol, records


def _check_finalization(data, inputs):
    """A new exact-byte view may finalize complete raw data after metadata failure."""
    data = v2.verified(data, FINALIZATION)
    require(all(data.get(key) is False for key in ("original_failure_promoted", "fresh_parity",
                "performance_acceptance_gate", "seven_repetition_timing_gate", "final_dataset_claim"))
            and all(type(data.get(key)) is int and data[key] == 0
                    for key in ("numerical_retries", "new_amplitude_evaluations")),
            "A metadata-only view cannot invent numerical retries, fresh parity or performance claims")
    original_status_path = checked(data["original_status"], inputs)
    original = original_status_path.parent
    require(original_status_path.name == "run-status.json", "Unrelated original run status")
    failed = read(original_status_path, inputs)
    require(failed.get("status") == "failed" and failed.get("error_type") == "RuntimeError"
            and failed.get("error") == "Worker series is incomplete or failed",
            "Original metadata-finalization failure is not retained")
    view_status_path = checked(data["view_status"], inputs)
    view = view_status_path.parent
    require(view != original and not view.is_relative_to(original) and not view.is_symlink(), "Finalization must use a separate physical view")
    protocol = read(checked(data["original_protocol"], inputs), inputs)
    specification = read(checked(data["original_specification"], inputs), inputs)
    require(checked(data["original_protocol"], inputs) == original / "protocol.json"
            and checked(data["original_specification"], inputs) == original / "input-spec.json", "Unrelated original finalization inputs")
    launch = read(checked(data["original_launch_record"], inputs), inputs)
    result = read(checked(data["original_launcher_result"], inputs), inputs)
    handoff = checked(data["original_root_handoff"], inputs)
    require(handoff == original / "root-handoff.json"
            and launch["specification"]["sha256"] == data["original_specification"]["sha256"]
            and launch["handoff"]["sha256"] == data["original_root_handoff"]["sha256"]
            and type(result.get("returncode")) is int and result["returncode"] == 1
            and result.get("interruptions") == [], "Unrelated or interrupted original launch failure")
    attempt = checked(data["original_launch_record"], inputs).parent
    require(checked(data["original_launcher_result"], inputs) == attempt / "result.json"
            and checked(data["original_launcher_stdout"], inputs) == attempt / "stdout"
            and checked(data["original_launcher_stderr"], inputs) == attempt / "stderr",
            "Original launch evidence belongs to another attempt")
    checked(data["finalizer_source"], inputs)
    cases = [c["id"] for c in protocol["configuration"]["cases"]]
    status = read(view_status_path, inputs)
    require(failed["completed_cases"] == cases and status.get("status") == "complete"
            and status.get("requested_cases") == cases and status.get("completed_cases") == cases,
            "Metadata finalization does not contain all actual completed cases")
    require(specification["configuration"] == protocol["configuration"], "Finalization changed the declared configuration")
    expected = {original / name for name in ("protocol.json", "input-spec.json", "process-ledger.jsonl")}
    for subtree in ("data", "builds", "sources", "points", "source-proofs"):
        expected.update(p for p in (original / subtree).rglob("*") if p.is_file())
    sources, destinations = set(), set()
    for entry in data["copies"]:
        source, destination = checked(entry["source"], inputs), checked(entry["destination"], inputs)
        require(source.is_relative_to(original) and destination.is_relative_to(view)
                and source.relative_to(original) == destination.relative_to(view)
                and not destination.is_symlink() and source not in sources and destination not in destinations
                and entry["source"]["sha256"] == entry["destination"]["sha256"], "Changed, duplicate or remapped completion-view input")
        sources.add(source); destinations.add(destination)
    require(expected == sources, "Completion view omits or injects original raw/source/card/point inputs")
    require(checked(data["bindings"], inputs).parent == view, "Finalization bindings belong to another view")
    require((view / "protocol.json").read_bytes() == (original / "protocol.json").read_bytes()
            and (view / "input-spec.json").read_bytes() == (original / "input-spec.json").read_bytes(),
            "Finalization rewrites the original protocol/specification")
    return data["bindings"], view


def checked_correctness(filename, policy_path, inputs=None):
    inputs = {} if inputs is None else inputs
    _, correction, config, proof = validate_policy(policy_path, inputs)
    supplied = read(filename, inputs)
    finalization = supplied.get("schema") == FINALIZATION
    view = None
    if finalization:
        bound_binding, view = _check_finalization(supplied, inputs)
        supplied = read(checked(bound_binding, inputs), inputs)
    binding = v2.verified(supplied, "supplement-source-bindings-v1")
    require(binding["role_map"] == {"candidate": "candidate_corrected"}, "Wrong NRS correctness roles")
    report = v2.path(binding["report"])
    require(view is None or report == view, "Finalization view/binding mismatch")
    protocol = _check_protocol(report, inputs)
    specification = v2.verified(read(checked(protocol["input_spec"], inputs), inputs), "nrs-first-core-correctness-spec-v1")
    validate_configuration(protocol["configuration"], correctness=True)
    require(protocol["sources"] == "candidate" and protocol["backends"] == [8] and protocol["repetitions"] == 2
            and specification["configuration"] == protocol["configuration"]
            and specification["scheduled_sources"] == ["candidate"] and specification["backends"] == [8]
            and specification["cpu"] == protocol["cpu"] and specification["repetitions"] == 2
            and specification["fresh_parity"] is False and protocol.get("fresh_parity") is False
            and specification["strict_statuses"] == [218, 228, 322],
            "Wrong NRS-only correctness targets/commands")
    for record in specification["sources"].values():
        checked(record, inputs)
    status = read(report / "run-status.json", inputs)
    require(status["status"] == ("complete" if finalization else "complete-raw-correctness-only") and
            status["completed_cases"] == [c["id"] for c in protocol["configuration"]["cases"]], "NRS correctness collection is incomplete")
    require(not list((report / "rejected").glob("*.json")), "NRS correctness run has rejected observations")
    require(set(binding["targets"]) == {f"{cid}-candidate-8" for cid in CASE_IDS}, "Missing or extra NRS correctness target")
    records = {}
    expected_paths = set()
    for cid in CASE_IDS:
        item = binding["targets"][f"{cid}-candidate-8"]
        build, build_path = _check_target(item, correction, proof, inputs, target_schema="supplement-source-bindings-v1")
        points = report / "points" / f"{cid}.dat"
        calibration = None
        for rep in ["calibration", *range(protocol["repetitions"])]:
            path = report / "data" / f"{cid}-candidate-8-{rep}.json"
            record = read(path, inputs)
            require(record["repetition"] == rep and record["loops"] == 1
                    and record["cpu"] == protocol["cpu"]
                    and record["protocol_identity_sha256"] == protocol["protocol_identity_sha256"], "Correctness command identity changed")
            _check_record_target(record, item, build, build_path, _case(config, correction, cid), points, inputs)
            require(all(row["status"] in specification["strict_statuses"] for row in record["rows"])
                    and all(int(code) in specification["strict_statuses"] for code in record["timed_status_counts"]),
                    "NRS correctness status differs from the actual declared collector")
            if rep == "calibration":
                calibration = record
            else:
                bench.validate_repetition(record, calibration, config)
                records[cid, rep] = record
            expected_paths.add(path)
    require(set((report / "data").glob("*.json")) == expected_paths, "Extra/injected correctness observations")
    return binding, records


def combine_gates(correctness_passed, performance_passed, qualifications=None, publication=None, external_findings=()):
    """The report supplies separately validated gates; absent evidence stays pending."""
    qualifications, publication = qualifications or {}, publication or {}
    require(type(correctness_passed) is bool and type(performance_passed) is bool, "Gate values must be Booleans")
    require(set(qualifications).issubset(QUALIFICATION_KEYS) and set(publication).issubset(PUBLICATION_KEYS), "Unknown downstream gate")
    require(all(type(v) is bool for v in [*qualifications.values(), *publication.values()]), "Downstream gates require verified Booleans")
    require(all(e.get("status") in ("failed", "pending", "accepted-observation") for e in external_findings), "External failures cannot be labelled passed")
    core = correctness_passed and performance_passed
    milestone = core and all(qualifications.get(k) is True for k in QUALIFICATION_KEYS)
    return {"nrs_correctness_passed": correctness_passed, "nrs_performance_passed": performance_passed,
            "nrs_core_validation_passed": core, "nrs_validation_milestone_passed": milestone,
            "external_failures_recorded": any(e["status"] == "failed" for e in external_findings) if external_findings else None,
            "external_review_status": "recorded" if external_findings else "pending",
            "external_backend_comparison_passed": False,
            "final_publication_passed": milestone and all(publication.get(k) is True for k in PUBLICATION_KEYS)}


def build_dataset(policy_path, references_path, correctness_path=None, pair_bindings_path=None, external_path=None):
    inputs = {}
    policy, correction, config, proof = validate_policy(policy_path, inputs)
    references, refs, _ = checked_references(references_path, policy_path, inputs)
    correct_binding, correct = (checked_correctness(correctness_path, policy_path, inputs) if correctness_path else (None, {}))
    pair_binding, timed = None, {}
    if pair_bindings_path:
        pair_binding = v2.checked_bindings(retain(pair_bindings_path, inputs), correction)
        require(set(pair_binding["targets"]) == {f"{cid}-{s}-{b}" for cid in CASE_IDS for s, b in PAIR_TARGETS}, "Injected/missing pair binding target")
        _, timed = checked_pair_series(pair_binding["report"], inputs)
        for item in pair_binding["targets"].values():
            build, bp = _check_target(item, correction, proof, inputs)
            for (cid, source, backend, _), (_, record) in timed.items():
                if (cid, source, backend) == (item["case"], "candidate" if item["role"] == "candidate_corrected" else "baseline", item["backend"]):
                    _check_record_target(record, item, build, bp, _case(config, correction, cid), v2.path(pair_binding["report"]) / "points" / f"{cid}.dat", inputs)
    external = []
    if external_path:
        external_spec = read(external_path, inputs)
        require(external_spec.get("schema") == "nrs-external-findings-v3", "Wrong external findings schema")
        external = external_spec["findings"]
        for entry in external:
            require(entry.get("status") in ("failed", "pending", "accepted-observation") and entry.get("evidence"), "External findings need honest bound evidence")
            for evidence in entry["evidence"]:
                checked(evidence, inputs)
    cases = {}
    for cid in sorted(CASE_IDS):
        observations = [record for (case, _), record in correct.items() if case == cid]
        observations += [data for (case, source, backend, _), (_, data) in timed.items() if case == cid and source == "candidate" and backend == 8]
        agreement = {"status": "pending", "passed": False, "comparisons": []}
        if observations and all((cid, backend) in refs for backend in (1, 6)):
            comparisons = []
            for record in observations:
                groups = {"original_helper_corrected/1": refs[cid, 1]["rows"],
                          "original_helper_corrected/6": refs[cid, 6]["rows"], "candidate_corrected/8": record["rows"]}
                require(all(record["points_sha256"] == refs[cid, b]["points_sha256"]
                            and record["parameter_card_sha256"] == refs[cid, b]["parameter_card_sha256"]
                            and record["shared_numerical_settings"] == refs[cid, b]["shared_numerical_settings"] for b in (1, 6)), "Correctness reference physical/numerical drift")
                comparisons.append(v2.compare_rows(groups))
            passed = all(c["passed"] for c in comparisons)
            agreement = {"status": "passed" if passed else "failed", "passed": passed, "comparisons": comparisons}
        performance = {"status": "pending", "passed": False}
        timing_agreement = {"status": "pending", "passed": False, "comparisons": []}
        if timed:
            comparisons = []
            for (case, source, backend, rep), (_, candidate) in sorted(timed.items()):
                if (case, source, backend) != (cid, "candidate", 8):
                    continue
                original = timed[cid, "baseline", 6, rep][1]
                require(all(original[k] == candidate[k] for k in ("points_sha256", "parameter_card_sha256",
                            "shared_numerical_settings", "answer_dimensions")), "Performance denominator input drift")
                if all((cid, b) in refs for b in (1, 6)):
                    comparisons.append(v2.compare_rows({"original/6": original["rows"],
                        "candidate_corrected/8": candidate["rows"],
                        "original_helper_corrected/1": refs[cid, 1]["rows"],
                        "original_helper_corrected/6": refs[cid, 6]["rows"]}))
            if comparisons:
                passed = all(c["passed"] for c in comparisons)
                timing_agreement = {"status": "passed" if passed else "failed", "passed": passed,
                                    "comparisons": comparisons}
            a = [data["seconds_per_point"] for (case, source, backend, _), (_, data) in sorted(timed.items()) if case == cid and source == "baseline" and backend == 6]
            b = [data["seconds_per_point"] for (case, source, backend, _), (_, data) in sorted(timed.items()) if case == cid and source == "candidate" and backend == 8]
            performance = bench.runtime_statistics(a, b, 1.05, config["seed"])
        cases[cid] = {"agreement": agreement, "performance": performance, "timing_agreement": timing_agreement,
                      "calibration_reference_timing_repetitions": 0, "nrs_numerical_observations": len(observations)}
    gates = combine_gates(all(c["agreement"]["passed"] for c in cases.values()),
                          all(c["performance"]["passed"] and c["timing_agreement"]["passed"] for c in cases.values()), external_findings=external)
    status = "passed" if gates["nrs_core_validation_passed"] else "failed" if any(
        c[k]["status"] == "failed" for c in cases.values() for k in ("agreement", "performance", "timing_agreement")) else "unresolved" if any(
        c["performance"]["status"] == "unresolved" for c in cases.values()) else "pending"
    return v2.seal({"schema": DATASET, "policy": v2.bound_file(policy_path),
        "reference_bindings": v2.bound_file(references_path),
        "correctness_bindings": v2.bound_file(correctness_path) if correctness_path else None,
        "timing_bindings": v2.bound_file(pair_bindings_path) if pair_bindings_path else None,
        "external_findings": v2.bound_file(external_path) if external_path else None,
        "configuration": policy["configuration"], "correction": policy["correction"],
        "representative_proof": policy["representative_proof"],
        "cases": cases, **gates, "nrs_core_status": status,
        "complete_campaign_passed": False, "qualification_status": "pending-separate-report-validation",
        "measurements_sha256": {v2.relative(filename): workers.digest(filename) for filename, _ in timed.values()},
        "inputs_sha256": dict(sorted(inputs.items())), "performance_denominator": "original/6",
        "user_priority": USER_PRIORITY})


def validate_dataset(filename):
    recorded = v2.verified(v2.read(filename), DATASET)
    actual = build_dataset(checked(recorded["policy"]), checked(recorded["reference_bindings"]),
                          checked(recorded["correctness_bindings"]) if recorded["correctness_bindings"] else None,
                          checked(recorded["timing_bindings"]) if recorded["timing_bindings"] else None,
                          checked(recorded["external_findings"]) if recorded["external_findings"] else None)
    require(actual == recorded, "Stale or forged v3 dataset conclusion")
    return recorded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("policy", "references", "publish", "verify"))
    for name in ("policy", "v2-policy", "proof", "spec", "references", "correctness", "timing", "external", "output", "dataset"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.action == "policy":
        result = create_policy(args.v2_policy, args.proof, args.output)
    elif args.action == "references":
        result = create_references(args.policy, args.spec, args.output)
    elif args.action == "publish":
        result = _write(args.output, build_dataset(args.policy, args.references, args.correctness, args.timing, args.external))
    else:
        result = validate_dataset(args.dataset)
    print(json.dumps({k: v for k, v in result.items() if k in ("schema", "identity_sha256", "nrs_core_status")}, indent=2))


if __name__ == "__main__":
    main()
