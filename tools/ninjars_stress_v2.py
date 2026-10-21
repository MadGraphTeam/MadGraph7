#!/usr/bin/env python3
"""Additional fixed two-to-four correctness controls; no performance gate.

All generation/evaluation is parameterized by final core identities and a full
measured reducer descriptor. Existing core/loop-induced protocols stay frozen.
"""
import argparse
from contextlib import contextmanager
import copy
import difflib
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from types import SimpleNamespace

import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_workers as workers
import ninjars_precision_v2 as precision
import ninjars_supplement_v2 as supplement

ROOT = bench.ROOT
PLAN = ROOT / "docs/benchmarkfolder/policy-v2/final-stress-plan-v1"
SCHEMA = "madloop-final-stress-v1"
IDENTITIES = {"core_dataset_identity_sha256", "literal_baseline_revision",
              "candidate_generator_identity_sha256", "candidate_runtime_identity_sha256"}
VARIANTS = {"ordinary": 20, "rotation": 4, "scale": 8}
BACKENDS = (1, 6, 7, 8)


def read(path):
    return json.loads(Path(path).read_text())


def checked(record, inputs=None):
    return supplement.checked(record, inputs)


def preregistration():
    seal = read(PLAN / "manifest.json")
    if seal["status"] != "root-approved-no-execution":
        raise ValueError("Stress plan is not root approved")
    for name, record in seal["files"].items():
        checked({"path": policy.relative(PLAN / name), **record})
    config = read(PLAN / "configuration.json")
    if (config["candidate_backends"] != list(BACKENDS) or
            config["phase_space"]["ordinary_points_per_case"] != 20 or
            config["runtime_reporting"]["performance_acceptance_gate"]):
        raise ValueError("Stress preregistration scope changed")
    return config


def configuration(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown stress control")
    plan = preregistration()
    cases = [{k: c[k] for k in ("id", "model", "process", "nfinal")} for c in plan["cases"]]
    if variant != "ordinary":
        cases = [{**c, "id": c["id"]+"-"+variant} for c in cases]
    return {"schema_version": 1, "seed": plan["phase_space"]["master_seed"],
            "points_per_case": VARIANTS[variant], "repetitions": 2,
            "energy_gev": 1000., "relative_tolerance": 1e-7, "scale_tolerance": 1e-10,
            "madloop_parameters": {"MLStabThres": 1e-9, "COLLIERRequiredAccuracy": 1e-12},
            "cases": cases}


def check_expected(expected, signature):
    if set(expected) != IDENTITIES or any(not isinstance(v, str) or not v for v in expected.values()):
        raise ValueError("All final core source/runtime identities are required")
    if expected["literal_baseline_revision"] != bench.BASELINE_REVISION:
        raise ValueError("Wrong literal baseline revision")
    policy.verified(signature, "reducer-runtime-signature-v2")
    if signature["identity_sha256"] != expected["candidate_runtime_identity_sha256"]:
        raise ValueError("Stress runtime descriptor does not belong to final core")
    return signature


def check_installed(expected, signature, prefix=None):
    check_expected(expected, signature)
    prefix = policy.path(prefix or ROOT / "HEPTools/ninjars")
    metadata = {name: supplement.digest(prefix / "lib" / name)
                for name in policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}}
    precision.match_runtime_archive(signature, supplement.digest(prefix / "lib/libninjars.a"),
                                    metadata, expected["candidate_runtime_identity_sha256"])


def serialize(events):
    return str(len(events))+"\n"+"".join(" ".join(f"{x:.17e}" for x in p)+"\n"
                                          for event in events for p in event)


def parse_points(path, nexternal=6):
    lines = Path(path).read_text().splitlines()
    count = int(lines[0])
    values = [list(map(float, line.split())) for line in lines[1:]]
    if count < 1 or len(values) != count*nexternal or any(len(p) != 4 for p in values):
        raise ValueError("Incomplete stress momentum stream")
    return [values[i*nexternal:(i+1)*nexternal] for i in range(count)]


def rotated(events):
    result = copy.deepcopy(events[:4])
    if len(result) != 4:
        raise ValueError("Four fixed rotation inputs are required")
    # Same rational rotations as the existing physical-point control utility.
    for event in result:
        for p in event:
            x, y, z = p[1:]
            x, y = (3*x-4*y)/5, (4*x+3*y)/5
            y, z = (5*y-12*z)/13, (12*y+5*z)/13
            p[1:] = [x, y, z]
    return result


def scale_events(events):
    if len(events) < 2:
        raise ValueError("Two fixed scale-control inputs are required")
    return [copy.deepcopy(event) for event in events[:2] for _ in range(4)]


def scale_schedule():
    return [1., 2., .5, 1.]*2


def scale_worker_source(text):
    """Only the three existing scale assignments plus a schedule reader differ."""
    statement = "MU_R=SQRT(ABS(TOTAL(0)**2-SUM(TOTAL(1:3)**2)))"
    if text.count(statement) != 3 or text.count("      REAL*8 ZERO\n") != 1:
        raise ValueError("Frozen worker scale/declaration structure changed")
    result = text.replace("      REAL*8 ZERO\n", "      REAL*8 ZERO,STRESS_SCALE\n")
    for index in ("1", "1+MOD(K-1,NPOINTS)", "K"):
        result = result.replace(statement, statement+"*STRESS_SCALE("+index+")", 1)
        # The replacement contains the original statement: mask it until all
        # three original occurrences have been consumed in source order.
        result = result.replace(statement+"*STRESS_SCALE("+index+")", "@STRESS_SCALE_"+index+"@", 1)
    for index in ("1", "1+MOD(K-1,NPOINTS)", "K"):
        result = result.replace("@STRESS_SCALE_"+index+"@", statement+"*STRESS_SCALE("+index+")", 1)
    result += """
C     Stress-driver input schedule only; production numerical code is unchanged.
      REAL*8 FUNCTION STRESS_SCALE(INDEX)
      IMPLICIT NONE
      INTEGER INDEX,I,IOS
      REAL*8 FACTORS(8)
      LOGICAL FIRST
      DATA FIRST/.TRUE./
      SAVE FIRST,FACTORS
      IF(FIRST) THEN
         OPEN(77,FILE='stress_scales.dat',STATUS='OLD')
         DO I=1,8
            READ(77,*,IOSTAT=IOS) FACTORS(I)
            IF(IOS.NE.0) STOP 81
         ENDDO
         CLOSE(77)
         FIRST=.FALSE.
      ENDIF
      IF(INDEX.LT.1.OR.INDEX.GT.8) STOP 82
      STRESS_SCALE=FACTORS(INDEX)
      END
"""
    return result


def verify_worker_source(text, frozen, origin, compiled, driver_sha256, scale=False):
    """Reject numerical-source edits hidden behind a driver-only adapter."""
    prefix = re.search(r"CALL\s+(\w*)GET_ANSWER_DIMENSION", text, re.I)
    born = re.search(r"PARAMETER\(NSQSO_BORN=(\d+)\)", text, re.I)
    if prefix is None or born is None:
        raise ValueError("Stress worker dimensions/prefix are missing")
    expected = frozen.replace("@PREFIX@", prefix.group(1)).replace("@NEXTERNAL@", "6").replace("@NBORN@", born.group(1))
    if text != expected:
        raise ValueError("Stress worker differs from the frozen generic worker")
    final = scale_worker_source(text) if scale else text
    checksum = hashlib.sha256(final.encode()).hexdigest()
    if checksum != driver_sha256:
        raise ValueError("Stress compiled driver differs from the reviewed source")
    changed = {name for name in origin["files"].keys() | compiled["files"].keys()
               if origin["files"].get(name) != compiled["files"].get(name)}
    drivers = {name for name in compiled["files"] if name.startswith("SubProcesses/P") and name.endswith("/check_sa.f")}
    if len(drivers) != 1 or changed != drivers:
        raise ValueError("Stress worker changes numerical sources outside its single driver")
    driver = next(iter(drivers))
    if compiled["files"][driver]["sha256"] != checksum:
        raise ValueError("Stress driver hash differs from its compiled source manifest")


@contextmanager
def bounded_generation(work, report):
    original = bench.run
    def run(command, cwd, log, **kwargs):
        kwargs.setdefault("timeout", 21600.)
        return original(command, cwd, log, **kwargs)
    with supplement.harness(work, report):
        bench.run = run
        try:
            yield
        finally:
            bench.run = original


def prepare(specification):
    """Actual CLI export/build and first point binding; never loop evaluation."""
    spec = read(specification)
    expected, signature = spec["expected"], spec["runtime_signature"]
    check_installed(expected, signature)
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    if report.exists() or work.exists():
        raise ValueError("Stress preparation namespace exists")
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *bench.GENERATOR_PATHS,
                    "vendor/ninjars"], cwd=ROOT, check=True)
    if bench.content_digest(bench.generator_tree_state()) != expected["candidate_generator_identity_sha256"]:
        raise ValueError("Candidate generator differs from final core")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    for name in ("sources", "points", "source-proofs"):
        (report / name).mkdir()
    shutil.copy2(Path(__file__), report / "sources/ninjars_stress_v2.py")
    shutil.copy2(specification, report / "input-spec.json")
    status = {"status": "incomplete", "cases": {}, "amplitude_evaluations": 0}
    bench.dump(report / "preparation-status.json", status)
    try:
        with bounded_generation(work, report):
            for case in configuration("ordinary")["cases"]:
                bench.generate("candidate", case)
                output = bench.output_dir("candidate", case)
                physical = checked(spec["physical_cards"][case["model"]])
                if (output / "Cards/param_card.dat").read_bytes() != physical.read_bytes():
                    raise ValueError("Generated stress physical card differs from final core")
                masses = bench.build("candidate", case)
                ordinary = bench.make_points(case, masses, configuration("ordinary"))
                events = parse_points(ordinary)
                streams = {"ordinary": events, "rotation": rotated(events), "scale": scale_events(events)}
                for variant, values in streams.items():
                    cid = case["id"] + ("-"+variant if variant != "ordinary" else "")
                    path = report / "points" / (cid+".dat")
                    if variant != "ordinary":
                        path.write_text(serialize(values))
                    metadata = supplement.point_metadata(path, masses, 4)
                    bench.dump(path.with_suffix(".physical.json"), metadata)
                for stage in ("generate", "compile", "masses"):
                    source_log = work / "logs" / f"{stage}-candidate-{case['id']}.log"
                    target_log = report / "logs" / source_log.name
                    target_log.parent.mkdir(exist_ok=True)
                    shutil.copy2(source_log, target_log)
                proof = policy.seal({"schema": "stress-generation-source-v1", "case": case,
                    "generation": read(output / "generation.json"), "compilation": read(output / "compilation.json"),
                    "compiled_manifest": bench.generated_source_manifest(output),
                    "physical_card": supplement.bound(report / "inputs/candidate" / case["id"] / "param_card.dat"),
                    "cli_card": supplement.bound(report / "commands" / ("candidate-"+case["id"]+".mg5")),
                    "cli_log": supplement.bound(report / "logs" / ("generate-candidate-"+case["id"]+".log"))})
                bench.dump(report / "source-proofs" / (case["id"]+".json"), proof)
                status["cases"][case["id"]] = {"output": policy.relative(output), "masses": masses,
                    "source_proof": supplement.bound(report / "source-proofs" / (case["id"]+".json"))}
                bench.dump(report / "preparation-status.json", status)
        check_installed(expected, signature)
        if bench.content_digest(bench.generator_tree_state()) != expected["candidate_generator_identity_sha256"]:
            raise ValueError("Candidate changed during stress preparation")
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(report / "preparation-status.json", status)
        raise
    for variant in VARIANTS:
        bench.dump(report / (variant+".json"), configuration(variant))
    (report / "points/stress_scales.dat").write_text("".join(f"{x:.17e}\n" for x in scale_schedule()))
    status["status"] = "complete"
    bench.dump(report / "preparation-status.json", status)
    record = policy.seal({"schema": "stress-preparation-v1", "expected": expected,
                         "runtime_signature": signature, "status": supplement.bound(report / "preparation-status.json"),
                         "report": policy.relative(report), "work": policy.relative(work),
                         "configurations": {v: supplement.bound(report / (v+".json")) for v in VARIANTS},
                         "source_proofs": {cid: item["source_proof"] for cid, item in status["cases"].items()},
                         "point_files": {case["id"]: supplement.bound(report / "points" / (case["id"]+".dat"))
                                         for v in VARIANTS for case in configuration(v)["cases"]},
                         "ordinary_point_metadata": {case["id"]: supplement.bound(report / "points" / (case["id"]+".json")) for case in configuration("ordinary")["cases"]},
                         "scale_schedule": supplement.bound(report / "points/stress_scales.dat")})
    bench.dump(report / "preparation.json", record)
    return record


def validate_preparation(filename, expected, signature, inputs=None):
    record = policy.verified(read(filename), "stress-preparation-v1")
    check_expected(expected, signature)
    if record["expected"] != expected or record["runtime_signature"] != signature:
        raise ValueError("Stress preparation belongs to another final core cohort")
    status = read(checked(record["status"], inputs))
    if status["status"] != "complete" or status["amplitude_evaluations"] != 0:
        raise ValueError("Stress input generation is incomplete or already evaluated")
    for variant in VARIANTS:
        if read(checked(record["configurations"][variant], inputs)) != configuration(variant):
            raise ValueError("Stress process/point inventory or numerical policy changed")
    if checked(record["scale_schedule"], inputs).read_text() != "".join(f"{x:.17e}\n" for x in scale_schedule()):
        raise ValueError("Fixed scale-change schedule changed")
    for index, case in enumerate(configuration("ordinary")["cases"]):
        cid = case["id"]
        proof = policy.verified(read(checked(record["source_proofs"][cid], inputs)), "stress-generation-source-v1")
        identity = proof["generation"]["source_identity"]
        if (proof["case"] != case or identity["tree_state_sha256"] != expected["candidate_generator_identity_sha256"] or
                identity.get("dirty_patch_sha256") is not None or identity["tree_state"]["untracked"] or
                bench.content_digest(identity["tree_state"]) != identity["tree_state_sha256"]):
            raise ValueError("Stress CLI source is not the final clean generic candidate")
        card = checked(proof["physical_card"], inputs)
        if supplement.digest(card) != proof["generation"]["parameter_card_sha256"]:
            raise ValueError("Stress CLI physical card changed")
        command, log = checked(proof["cli_card"], inputs).read_text(), checked(proof["cli_log"], inputs).read_text()
        if f"generate {case['process']}\n" not in command or "Error detected in sub-command" in log or "interrupted in sub-command" in log:
            raise ValueError("Actual stress CLI export is missing or failed")
        ordinary = checked(record["point_files"][cid], inputs)
        events = parse_points(ordinary)
        if len(events) != 20:
            raise ValueError("Both complete twenty-point ordinary streams are required")
        masses = status["cases"][cid]["masses"]
        metadata = read(checked(record["ordinary_point_metadata"][cid], inputs))
        if (metadata["seed"] != configuration("ordinary")["seed"]+index or
                metadata["points"] != 20 or metadata["energy_gev"] != 1000. or metadata["masses"] != masses or
                metadata["sha256"] != supplement.digest(ordinary)):
            raise ValueError("Stress ordinary points have changed seed/masses/energy/bytes")
        supplement.point_metadata(ordinary, masses, 4)
        for variant, values in (("rotation", rotated(events)), ("scale", scale_events(events))):
            path = checked(record["point_files"][cid+"-"+variant], inputs)
            if path.read_text() != serialize(values):
                raise ValueError("Derived stress points differ from fixed transformation")
            supplement.point_metadata(path, masses, 4)
    return record


@contextmanager
def scale_adapter(schedule, report):
    original = workers.prepare
    def prepare_worker(source, case, backend, config, work, output_report, clean=False):
        job, directory, metadata = original(source, case, backend, config, work, output_report, clean=clean)
        text = (directory / "check_sa.f").read_text()
        modified = scale_worker_source(text)
        (directory / "check_sa.f").write_text(modified)
        shutil.copy2(schedule, directory / "stress_scales.dat")
        (directory / "check_sa.o").unlink(missing_ok=True)
        (directory / "check").unlink(missing_ok=True)
        key = f"{case['id']}-{source}-{backend}"
        retained = output_report / "builds" / key
        (retained / "stress-scale-before.f").write_text(text)
        (retained / "stress-scale-after.f").write_text(modified)
        (retained / "stress-scale.patch").write_text("".join(difflib.unified_diff(text.splitlines(True), modified.splitlines(True), fromfile="frozen-worker", tofile="scale-input-worker")))
        shutil.copy2(schedule, retained / "stress_scales.dat")
        seconds = bench.run(["make", "-j4", "check"], directory, output_report / "logs" / ("compile-scale-"+key+".log"), timeout=21600.)
        target = job / "outputs" / source / case["id"]
        manifest = bench.generated_source_manifest(target)
        metadata.update(driver_sha256=supplement.digest(directory / "check_sa.f"),
                        executable_sha256=supplement.digest(directory / "check"),
                        generated_sources_sha256=manifest["files_sha256"],
                        stress_scale_compile_seconds=seconds,
                        stress_scale_adapter={name: supplement.bound(retained / name) for name in ("stress-scale-before.f", "stress-scale-after.f", "stress-scale.patch", "stress_scales.dat")})
        bench.dump(retained / "build.json", metadata)
        bench.dump(retained / "compiled-sources.json", manifest)
        return job, directory, metadata
    workers.prepare = prepare_worker
    try:
        yield
    finally:
        workers.prepare = original


def collect(specification):
    """One complete variant/cohort per namespace, using the frozen workers."""
    spec = read(specification)
    expected, signature = spec["expected"], spec["runtime_signature"]
    preparation = validate_preparation(checked(spec["preparation"]), expected, signature)
    variant = spec["variant"]
    config = configuration(variant)
    if spec.get("optional_helper", False):
        if variant != "ordinary":
            raise ValueError("Optional helpers cover ordinary points only")
        sources, backends, roles = ("baseline",), (1, 6, 7), {"baseline": "original_helper_corrected"}
        roots, proofs = {"baseline": spec["helper_output_root"]}, {"baseline": spec["helper_source_proofs"]}
    else:
        check_installed(expected, signature)
        sources, backends, roles = ("candidate",), BACKENDS, {"candidate": "candidate_corrected"}
        roots = {"candidate": policy.relative(policy.path(preparation["work"]) / "outputs/candidate")}
        proofs = {"candidate": preparation["source_proofs"]}
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    if report.exists() or work.exists():
        raise ValueError("Stress collection namespace exists")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    for name in ("sources", "points"):
        (report / name).mkdir()
    for name in workers.SOURCE_NAMES:
        shutil.copy2(ROOT / "tools" / name, report / "sources" / name)
    shutil.copy2(Path(__file__), report / "sources/ninjars_stress_v2.py")
    shutil.copy2(specification, report / "input-spec.json")
    for case in config["cases"]:
        shutil.copy2(checked(preparation["point_files"][case["id"]]), report / "points" / (case["id"]+".dat"))
    descriptor = workers.protocol_descriptor()
    bench.dump(report / "protocol.json", {"schema": workers.SCHEME, "configuration": config,
        "protocol_descriptor": descriptor, "protocol_identity_sha256": bench.content_digest(descriptor),
        "cpu": spec["cpu"], "repetitions": 2, "sources": sources[0], "backends": list(backends),
        "fresh_parity": False, "acceptance_eligible": False, "performance_acceptance_gate": False,
        "stress_variant": variant, "role_map": roles, "input_adapter": supplement.bound(report / "sources/ninjars_stress_v2.py"),
        "input_spec": supplement.bound(report / "input-spec.json")})
    status = {"status": "incomplete", "requested_cases": [c["id"] for c in config["cases"]], "completed_cases": []}
    bench.dump(report / "run-status.json", status)
    args = SimpleNamespace(cpu=spec["cpu"], repetitions=2, minimum_seconds=1e-12,
                           timeout=86400., fresh_parity=False, clean_build=True)
    aliases = {c["id"]: c["id"][:-len("-"+variant)] if variant != "ordinary" else c["id"] for c in config["cases"]}
    sentinel = work / "input-root-sentinel"
    from contextlib import nullcontext
    adapter = scale_adapter(checked(preparation["scale_schedule"]), report) if variant == "scale" else nullcontext()
    try:
        with supplement.harness(sentinel, report), supplement.input_roots(roots, sentinel, aliases), adapter:
            for case in config["cases"]:
                workers.run_case(case, sources, backends, config, args, work / "workers", report, spec["run_id"])
                status["completed_cases"].append(case["id"])
                bench.dump(report / "run-status.json", status)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(report / "run-status.json", status)
        raise
    status["status"] = "complete"
    bench.dump(report / "run-status.json", status)
    workers.checked_series(report, minimum=2)
    return supplement.bind_collected(report, work, roles, config, {"output_roots": roots, "source_proofs": proofs, "case_aliases": aliases})


def validate_group(entry, expected, signature, preparation, inputs=None):
    binding = policy.verified(read(checked(entry["bindings"], inputs)), "supplement-source-bindings-v1")
    report = policy.path(binding["report"])
    protocol, records = workers.checked_series(report, minimum=2)
    variant = entry["variant"]
    if protocol["configuration"] != configuration(variant) or protocol["performance_acceptance_gate"]:
        raise ValueError("Stress variant/counts/numerical policy changed")
    helper = bool(entry.get("optional_helper", False))
    roles = {"baseline": "original_helper_corrected"} if helper else {"candidate": "candidate_corrected"}
    if binding["role_map"] != roles or protocol["role_map"] != roles or protocol["repetitions"] != 2:
        raise ValueError("Stress cohort roles or repetitions changed")
    if helper and variant != "ordinary":
        raise ValueError("Optional helper cannot replace a mandatory candidate control")
    required = {(c["id"], b) for c in configuration(variant)["cases"] for b in ((1, 6, 7) if helper else BACKENDS)}
    if len(binding["targets"]) != len(required) or {(t["case"], t["backend"]) for t in binding["targets"].values()} != required:
        raise ValueError("Stress cohort omits a required process/backend")
    for key, target in binding["targets"].items():
        if key != f"{target['case']}-{target['source']}-{target['backend']}" or target["role"] != roles.get(target["source"]):
            raise ValueError("Stress target identity/role is inconsistent")
        build = read(checked(target["build"], inputs))
        manifest = target["compiled_manifest"]
        if (bench.content_digest(manifest["files"]) != manifest["files_sha256"] or
                manifest["files_sha256"] != build["generated_sources_sha256"] or
                target["executable_sha256"] != build["executable_sha256"]):
            raise ValueError("Stress source/executable binding changed")
        actual_runtime = policy.reducer_signature(target)
        if actual_runtime != target["reducer_runtime"]:
            raise ValueError("Stress reduction runtime changed")
        for item in target["runtime_metadata"].values():
            checked(item, inputs)
        identity = target["generation_source_identity"]
        if not helper:
            if (actual_runtime != signature or identity["tree_state_sha256"] != expected["candidate_generator_identity_sha256"] or
                    identity.get("dirty_patch_sha256") is not None or identity["tree_state"]["untracked"] or
                    bench.content_digest(identity["tree_state"]) != identity["tree_state_sha256"]):
                raise ValueError("Stress target was relabelled from another candidate/runtime")
        source = read(checked(target["source_proof"], inputs))
        origin = read(checked(target["origin_manifest"], inputs))
        if (origin["files_sha256"] != build["original_generated_sources_sha256"] or
                bench.content_digest(origin["files"]) != origin["files_sha256"] or
                read(checked(target["origin_compilation"], inputs)) != build["origin_compilation"] or
                supplement.digest(checked(target["physical_card"], inputs)) != build["parameter_card_sha256"]):
            raise ValueError("Stress original generated input differs from worker build")
        retained = report / "builds" / key
        worker = retained / "worker.f"
        checked(supplement.bound(worker), inputs)
        frozen = report / "sources/ninjars_benchmark_worker.f"
        checked(supplement.bound(frozen), inputs)
        verify_worker_source(worker.read_text(), frozen.read_text(), origin, manifest,
                             build["driver_sha256"], variant == "scale")
        if helper:
            policy.verified(source, "stress-helper-correction-v1")
            policy.original_identity(identity)
            correction = policy.validate_correction(checked(source["correction"], inputs))
            proof = correction["cases"][source["source_control_case"]]
            policy.verify_delta(source["original_manifest"], origin,
                                {name: proof["fresh_after_manifest"]["files"].get(name) for name in policy.MODEL_FILES})
        else:
            policy.verified(source, "stress-generation-source-v1")
            base_id = target["case"][:-len("-"+variant)] if variant != "ordinary" else target["case"]
            expected_source = read(checked(preparation["source_proofs"][base_id], inputs))
            if (source["compiled_manifest"] != origin or source["generation"]["source_identity"] != identity or
                    source["compilation"] != build["origin_compilation"] or source != expected_source or
                    identity != build["generation_source_identity"]):
                raise ValueError("Stress target does not derive from its actual CLI export")
        if variant == "scale":
            files = build["stress_scale_adapter"]
            before, after = checked(files["stress-scale-before.f"], inputs).read_text(), checked(files["stress-scale-after.f"], inputs).read_text()
            if before != worker.read_text() or scale_worker_source(before) != after or supplement.digest(checked(files["stress-scale-after.f"], inputs)) != build["driver_sha256"]:
                raise ValueError("Scale driver changes more than the fixed MU_R input schedule")
            if checked(files["stress_scales.dat"], inputs).read_bytes() != checked(preparation["scale_schedule"], inputs).read_bytes():
                raise ValueError("Scale worker used a changed schedule")
            checked(files["stress-scale.patch"], inputs)
    for key, (filename, record) in records.items():
        checked(supplement.bound(filename), inputs)
        target = binding["targets"][f"{record['case']}-{record['source']}-{record['backend']}"]
        if (record["generation_source_identity"] != target["generation_source_identity"] or
                record["executable_sha256"] != target["executable_sha256"] or
                record["generated_sources_sha256"] != target["compiled_manifest"]["files_sha256"]):
            raise ValueError("Stress raw result differs from its numerical source/runtime binding")
        points = report / "points" / (record["case"]+".dat")
        if (supplement.digest(points) != record["points_sha256"] or
                checked(preparation["point_files"][record["case"]], inputs).read_bytes() != points.read_bytes()):
            raise ValueError("Stress measurement points differ from bound fixed inputs")
        checked(supplement.bound(points), inputs)
    return binding, protocol, records


def align_rows(rows, point_map):
    return [{**r, "point": point_map[r["point"]]} for r in rows if r["point"] in point_map]


def invariant_comparisons(ordinary, transformed, variant):
    groups, pairs = {}, []
    for backend in BACKENDS:
        if variant == "rotation":
            groups[f"ordinary/{backend}"] = align_rows(ordinary[backend], {p: p for p in range(1, 5)})
            groups[f"rotation/{backend}"] = transformed[backend]
            pairs.append((f"ordinary/{backend}", f"rotation/{backend}"))
        elif variant == "scale":
            groups[f"ordinary/{backend}"] = align_rows(ordinary[backend], {1: 1, 2: 2})
            groups[f"initial/{backend}"] = align_rows(transformed[backend], {1: 1, 5: 2})
            groups[f"restored/{backend}"] = align_rows(transformed[backend], {4: 1, 8: 2})
            pairs += [(f"ordinary/{backend}", f"initial/{backend}"), (f"initial/{backend}", f"restored/{backend}")]
        else:
            raise ValueError("Unknown invariant stress comparison")
    return supplement.compare_rows(groups, pairs)


def assemble(specification):
    """Create a new immutable dataset; absent or failed targets stay visible."""
    spec = read(specification)
    expected, signature = spec["expected"], spec["runtime_signature"]
    validate_preparation(checked(spec["preparation"]), expected, signature)
    destination = policy.path(spec["output"])
    if destination.exists():
        raise ValueError("Stress dataset exists; retain prior data and use a new snapshot")
    groups, failed = [], []
    for entry in spec["collections"]:
        variant = entry["variant"]
        if variant not in VARIANTS:
            raise ValueError("Unknown stress collection")
        directory = policy.path(entry["report"])
        base = {"variant": variant, "optional_helper": bool(entry.get("optional_helper", False))}
        if (directory / "source-bindings.json").is_file():
            groups.append({**base, "bindings": supplement.bound(directory / "source-bindings.json")})
        elif (directory / "run-status.json").is_file():
            failed.append({**base, "run_status": supplement.bound(directory / "run-status.json"),
                           "rejected": [supplement.bound(p) for p in sorted((directory / "rejected").glob("*.json"))]})
    record = policy.seal({"schema": SCHEMA, "expected": expected, "runtime_signature": signature,
                          "preparation": spec["preparation"], "groups": groups, "failed_groups": failed,
                          "performance_acceptance_gate": False})
    bench.dump(destination, record)
    return report_snapshot(destination, expected, runtime_signature=signature)


def failed_case_ids(failure):
    """Identify the actual failed case without labelling unattempted work."""
    candidates = {item["case"] for item in failure["rejected"] if item.get("case")}
    status = failure["status"]
    if not candidates and status["status"] == "failed":
        unfinished = [cid for cid in status.get("requested_cases", [])
                      if cid not in status.get("completed_cases", [])]
        if unfinished:
            candidates.add(unfinished[0])
    variant = failure["variant"]
    return {cid[:-len("-"+variant)] if variant != "ordinary" and cid.endswith("-"+variant) else cid
            for cid in candidates}


def report_snapshot(filename, expected, inputs=None, runtime_signature=None):
    record = policy.verified(read(filename), SCHEMA)
    checked(supplement.bound(filename), inputs)
    signature = record["runtime_signature"]
    check_expected(expected, signature)
    if record["expected"] != expected or (runtime_signature is not None and signature != runtime_signature):
        raise ValueError("Stress dataset belongs to another final core cohort")
    if record["performance_acceptance_gate"]:
        raise ValueError("Additional stress cases have no performance acceptance gate")
    prep = validate_preparation(checked(record["preparation"], inputs), expected, signature, inputs)
    import ninjars_report_v2 as report
    grouped, helpers = {}, {}
    for entry in record["groups"]:
        binding, protocol, rows = validate_group(entry, expected, signature, prep, inputs)
        table = helpers if entry.get("optional_helper", False) else grouped
        for (cid, source, backend, repetition), (_, value) in rows.items():
            key = (entry["variant"], cid, backend)
            table.setdefault(key, []).append(value)
    for table in (grouped, helpers):
        for runs in table.values():
            runs.sort(key=lambda r: r["repetition"])
            if [r["repetition"] for r in runs] != [0, 1]:
                raise ValueError("Duplicate/missing stress observations")
    pending = {"status": "pending", "comparisons": None, "max_tolerance_fraction": None, "failures": []}
    cases, gates = [], []
    for case in configuration("ordinary")["cases"]:
        cid = case["id"]
        result, row_tables = {**case, "points": 20}, {}
        for variant, count in VARIANTS.items():
            vid = cid+("-"+variant if variant != "ordinary" else "")
            runs = {b: grouped.get((variant, vid, b), []) for b in BACKENDS}
            complete = all(runs.values())
            if complete:
                for key in ("points_sha256", "parameter_card_sha256", "shared_numerical_settings", "answer_dimensions", "cpu"):
                    if len({bench.content_digest(r[key]) for values in runs.values() for r in values}) != 1:
                        raise ValueError("Stress candidate inputs/settings/CPU differ across backends")
                row_tables[variant] = {b: values[0]["rows"] for b, values in runs.items()}
                row_groups = {f"candidate_corrected/{b}": rows for b, rows in row_tables[variant].items()}
                agreement = report.compare(row_groups)
                stable = report.stability_snapshot(row_groups)
            else:
                agreement, stable = copy.deepcopy(pending), {"status": "pending"}
            result[variant] = {"points": count, "agreement": agreement, "stability": stable,
                "diagnostic_timings": {str(b): report.summary(runs[b], count, 2) for b in BACKENDS},
                "performance_acceptance_gate": False}
            gates.append(agreement["status"])
        for variant, label in (("rotation", "invariance"), ("scale", "restoration")):
            if "ordinary" in row_tables and variant in row_tables:
                for backend in BACKENDS:
                    left = grouped["ordinary", cid, backend][0]
                    right = grouped[variant, cid+"-"+variant, backend][0]
                    for key in ("parameter_card_sha256", "shared_numerical_settings", "answer_dimensions", "cpu"):
                        if left[key] != right[key]:
                            raise ValueError("Stress invariant comparison changed physical inputs/settings: "+key)
            value = invariant_comparisons(row_tables["ordinary"], row_tables[variant], variant) if "ordinary" in row_tables and variant in row_tables else copy.deepcopy(pending)
            result[variant][label] = value
            gates.append(value["status"])
        reference_runs = {b: helpers.get(("ordinary", cid, b), []) for b in (1, 6, 7)}
        optional = {"required": False, "status": "pending", "agreement": copy.deepcopy(pending), "failures": []}
        if all(reference_runs.values()) and "ordinary" in row_tables:
            all_rows = {f"candidate_corrected/{b}": rows for b, rows in row_tables["ordinary"].items()}
            for b, runs in reference_runs.items():
                reference = runs[0]
                candidate = grouped["ordinary", cid, 8][0]
                for key in ("points_sha256", "parameter_card_sha256", "shared_numerical_settings", "answer_dimensions"):
                    if reference[key] != candidate[key]:
                        raise ValueError("Optional helper inputs differ from candidate")
                all_rows[f"original_helper_corrected/{b}"] = reference["rows"]
            optional["agreement"] = report.compare(all_rows)
            optional["status"] = optional["agreement"]["status"]
        result["optional_reference"] = optional
        cases.append(result)
    failures = []
    for item in record.get("failed_groups", []):
        status = read(checked(item["run_status"], inputs))
        if status["status"] == "complete":
            raise ValueError("Complete stress collection lacks source bindings")
        failure = {"variant": item["variant"], "optional_helper": bool(item.get("optional_helper", False)), "status": status,
                   "rejected": [read(checked(p, inputs)) for p in item["rejected"]]}
        failure["case_ids"] = sorted(failed_case_ids(failure))
        failures.append(failure)
        if failure["optional_helper"]:
            for case in cases:
                if case["id"] in failure["case_ids"]:
                    optional = case["optional_reference"]
                    optional["failures"].append(failure)
                    optional["status"] = "failed" if status["status"] == "failed" else "pending"
        else:
            gates.append("failed" if status["status"] == "failed" else "pending")
    status = "failed" if "failed" in gates else "passed" if all(g == "passed" for g in gates) else "pending"
    return {"schema": SCHEMA, **expected, "dataset_identity_sha256": record["identity_sha256"],
            "status": status, "complete_campaign_passed": status == "passed", "cases": cases,
            "failed_groups": failures, "performance_acceptance_gate": False,
            "interpretation": "Fixed additional correctness/stability stress; diagnostic times only; all existing core/loop-induced gates unchanged"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "collect", "assemble", "validate"))
    parser.add_argument("specification", type=Path)
    parser.add_argument("--execute-numerical", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.specification)
    elif args.action == "collect":
        if not args.execute_numerical:
            parser.error("Numerical collection requires root's explicit serialized handoff flag")
        result = collect(args.specification)
    elif args.action == "assemble":
        result = assemble(args.specification)
    else:
        spec = read(args.specification)
        result = report_snapshot(policy.path(spec["dataset"]), spec["expected"], runtime_signature=spec["runtime_signature"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
