#!/usr/bin/env python3
"""Early CLI/build/point work only; the real core dataset is attached later."""
import json
from contextlib import contextmanager
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "madgraph/__init__.py").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_precision_v2 as precision
import ninjars_stress_v2 as stress
import ninjars_supplement_v2 as supplement
import ninjars_benchmark_workers as workers

SCHEMA = "stress-early-build-cohort-v1"
PATHS = (*bench.GENERATOR_PATHS, "vendor")


def read(path):
    return json.loads(Path(path).read_text())


def sources(numerical_revision):
    """Source equality is established by bytes and Git diff, never labels."""
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *PATHS], cwd=ROOT, check=True)
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "--", *PATHS], cwd=ROOT)
    # This retained pre-task installer diagnostic is not imported or compiled.
    # No other untracked generator/numerical input is accepted.
    additions = set(untracked.decode().splitlines())
    if additions - {"vendor/StdHEP/fail"}:
        raise ValueError("Uncommitted generator/numerical input is present")
    subprocess.run(["git", "diff", "--exit-code", numerical_revision, "HEAD", "--", *PATHS], cwd=ROOT, check=True)
    names = subprocess.check_output(["git", "ls-files", "-z", "--", *PATHS], cwd=ROOT).decode().split("\0")
    changes = subprocess.check_output(["git", "diff", "--name-only", "-z", numerical_revision, "HEAD"], cwd=ROOT).decode().split("\0")
    outside = [name for name in changes if name and not any(name == p or name.startswith(p+"/") for p in PATHS)]
    if any(not name.startswith(("docs/", "tests/", "references/")) for name in outside):
        raise ValueError("Unreviewed non-document/test source differs from numerical freeze")
    return {"production": {name: supplement.digest(ROOT / name) for name in names if name},
            "document_test_changes": {name: supplement.digest(ROOT / name) if (ROOT / name).is_file() else None for name in outside},
            "retained_untracked_diagnostics": {name: supplement.digest(ROOT / name) for name in additions}}


def runtime(spec):
    signature = spec["runtime_signature"]
    policy.verified(signature, "reducer-runtime-signature-v2")
    actual = supplement.digest(ROOT / "HEPTools/ninjars/lib/libninjars.a")
    if actual != spec["archive_sha256"]:
        raise ValueError("Main archive differs from the approved final publication")
    metadata = {name: supplement.digest(ROOT / "HEPTools/ninjars/lib" / name)
                for name in policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}}
    precision.match_runtime_archive(signature, actual, metadata, signature["identity_sha256"])
    publication = read(supplement.checked(spec["publication"]))
    if (publication["schema"] != "main-ninjars-publication-transition-v1" or publication["status"] != "published" or
            publication["installation_revision"] != spec["numerical_revision"] or
            publication["main_destination"] != "HEPTools/ninjars" or
            publication["main_artifact_sha256"]["lib/libninjars.a"] != actual):
        raise ValueError("Publication record describes another source/archive/prefix")
    preparation = read(supplement.checked(publication["preparation"]))
    installation = read(supplement.checked(preparation["copied_installation_record"]))
    if installation["revision"] != spec["numerical_revision"] or installation["files"] != publication["main_artifact_sha256"]:
        raise ValueError("Actual private installation and main publication differ")
    for name, checksum in publication["main_artifact_sha256"].items():
        if supplement.digest(ROOT / "HEPTools/ninjars" / name) != checksum:
            raise ValueError("Published numerical/interface asset changed: "+name)
    for name, checksum in installation["archived_bridge_source_sha256"].items():
        if supplement.digest(ROOT / "vendor/ninjars" / name) != checksum:
            raise ValueError("Published archive source differs from frozen bridge: "+name)
    return signature


@contextmanager
def owned_generation(work, report):
    """Use the existing build environment and reap our CLI/make process group."""
    original = bench.run
    def run(command, cwd, log, **kwargs):
        if kwargs:
            raise ValueError("Unexpected early-generation subprocess override")
        log.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        with log.open("w") as stream:
            process = subprocess.Popen(command, cwd=cwd, env=workers.environment(work),
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = process.wait(timeout=21600.)
            finally:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        if code:
            raise RuntimeError(f"{command[0]} failed ({code}); see {log}")
        return time.perf_counter()-start
    with supplement.harness(work, report):
        bench.run = run
        try:
            yield
        finally:
            bench.run = original


def built_runtime(output, manifest, report):
    metadata = {}
    assets = report / "runtime-assets"
    assets.mkdir(exist_ok=True)
    for name, source in policy.runtime_metadata_files(output).items():
        target = assets / supplement.digest(source)
        if not target.exists():
            shutil.copy2(source, target)
        metadata[name] = supplement.bound(target)
    signature = policy.reducer_signature({"backend": 8, "libraries": policy.library_hashes(output),
                                           "runtime_metadata": metadata, "compiled_manifest": manifest})
    return signature, metadata


def prepare(specification):
    spec = read(specification)
    if "core_dataset_identity_sha256" in spec:
        raise ValueError("Early source registration must not assert a future core dataset")
    signature = runtime(spec)
    inventory = sources(spec["numerical_revision"])
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != spec["generation_revision"]:
        raise ValueError("Actual generation HEAD differs from the recorded freeze")
    generator = bench.generator_tree_state()
    if bench.content_digest(generator) != spec["generator_identity_sha256"]:
        raise ValueError("Generator input bytes differ from the final frozen cohort")
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    if report.exists() or work.exists():
        raise ValueError("Early build namespace exists; retain the first attempt")
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    for name in ("sources", "points", "source-proofs"):
        (report / name).mkdir()
    shutil.copy2(Path(__file__), report / "sources/early_build.py")
    shutil.copy2(specification, report / "early-input-spec.json")
    registration = policy.seal({"schema": SCHEMA, "generation_revision": revision,
        "numerical_revision": spec["numerical_revision"], "numerical_sources_sha256": inventory,
        "generator_identity_sha256": spec["generator_identity_sha256"],
        "runtime_signature": signature, "archive_sha256": spec["archive_sha256"],
        "publication": spec["publication"], "physical_cards": spec["physical_cards"],
        "configuration": stress.configuration("ordinary"),
        "report": policy.relative(report), "work": policy.relative(work),
        "core_dataset_link": "not-yet-available; no core identity asserted",
        "loop_amplitude_evaluations": 0, "initialized_symbolica_workers": 0})
    bench.dump(report / "early-registration.json", registration)
    status = {"status": "incomplete", "cases": {}, "amplitude_evaluations": 0}
    bench.dump(report / "preparation-status.json", status)
    try:
        with owned_generation(work, report):
            for case in stress.configuration("ordinary")["cases"]:
                bench.generate("candidate", case)
                output = bench.output_dir("candidate", case)
                card = supplement.checked(spec["physical_cards"][case["model"]])
                if (output / "Cards/param_card.dat").read_bytes() != card.read_bytes():
                    raise ValueError("Early stress card differs from the fixed final physical input")
                masses = bench.build("candidate", case)
                compiled = bench.generated_source_manifest(output)
                actual_runtime, actual_metadata = built_runtime(output, compiled, report)
                if actual_runtime != signature:
                    raise ValueError("Built stress reduction/transitive libraries or link inputs differ from the full frozen runtime")
                points = bench.make_points(case, masses, stress.configuration("ordinary"))
                events = stress.parse_points(points)
                for variant, values in (("ordinary", events), ("rotation", stress.rotated(events)), ("scale", stress.scale_events(events))):
                    cid = case["id"]+("-"+variant if variant != "ordinary" else "")
                    path = report / "points" / (cid+".dat")
                    if variant != "ordinary":
                        path.write_text(stress.serialize(values))
                    bench.dump(path.with_suffix(".physical.json"), supplement.point_metadata(path, masses, 4))
                for stage in ("generate", "compile", "masses"):
                    source = work / "logs" / f"{stage}-candidate-{case['id']}.log"
                    target = report / "logs" / source.name
                    target.parent.mkdir(exist_ok=True)
                    shutil.copy2(source, target)
                proof = policy.seal({"schema": "stress-generation-source-v1", "case": case,
                    "generation": read(output / "generation.json"), "compilation": read(output / "compilation.json"),
                    "compiled_manifest": compiled, "reducer_runtime": actual_runtime,
                    "runtime_metadata": actual_metadata,
                    "physical_card": supplement.bound(report / "inputs/candidate" / case["id"] / "param_card.dat"),
                    "cli_card": supplement.bound(report / "commands" / ("candidate-"+case["id"]+".mg5")),
                    "cli_log": supplement.bound(report / "logs" / ("generate-candidate-"+case["id"]+".log"))})
                target = report / "source-proofs" / (case["id"]+".json")
                bench.dump(target, proof)
                status["cases"][case["id"]] = {"output": policy.relative(output), "masses": masses,
                                               "source_proof": supplement.bound(target)}
                bench.dump(report / "preparation-status.json", status)
                if sources(spec["numerical_revision"]) != inventory:
                    raise ValueError("Numerical source changed during early CLI/build work")
                if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != revision:
                    raise ValueError("Generation HEAD changed during the frozen build window")
        runtime(spec)
    except BaseException as error:
        status.update(status="failed", error_type=type(error).__name__, error=str(error))
        bench.dump(report / "preparation-status.json", status)
        raise
    for variant in stress.VARIANTS:
        bench.dump(report / (variant+".json"), stress.configuration(variant))
    (report / "points/stress_scales.dat").write_text("".join(f"{x:.17e}\n" for x in stress.scale_schedule()))
    status["status"] = "complete"
    bench.dump(report / "preparation-status.json", status)
    result = policy.seal({"schema": "stress-early-build-result-v1", "registration": supplement.bound(report / "early-registration.json"),
        "status": supplement.bound(report / "preparation-status.json"),
        "configurations": {v: supplement.bound(report / (v+".json")) for v in stress.VARIANTS},
        "source_proofs": {cid: item["source_proof"] for cid, item in status["cases"].items()},
        "point_files": {c["id"]: supplement.bound(report / "points" / (c["id"]+".dat"))
                        for v in stress.VARIANTS for c in stress.configuration(v)["cases"]},
        "ordinary_point_metadata": {c["id"]: supplement.bound(report / "points" / (c["id"]+".json"))
                                    for c in stress.configuration("ordinary")["cases"]},
        "scale_schedule": supplement.bound(report / "points/stress_scales.dat")})
    bench.dump(report / "early-result.json", result)
    return result


def attach_core(filename, expected, signature):
    """Add the actual later dataset link in a new manifest; reuse exact assets."""
    result = policy.verified(read(filename), "stress-early-build-result-v1")
    registration = policy.verified(read(supplement.checked(result["registration"])), SCHEMA)
    stress.check_expected(expected, signature)
    if (registration["runtime_signature"] != signature or
            registration["generator_identity_sha256"] != expected["candidate_generator_identity_sha256"]):
        raise ValueError("Early assets belong to another final numerical cohort")
    report = policy.path(registration["report"])
    destination = report / "preparation.json"
    if destination.exists():
        raise ValueError("Retain the existing real-core linkage")
    record = policy.seal({"schema": "stress-preparation-v1", "expected": expected,
        "runtime_signature": signature, "status": result["status"],
        "report": registration["report"], "work": registration["work"],
        **{key: result[key] for key in ("configurations", "source_proofs", "point_files", "ordinary_point_metadata", "scale_schedule")},
        "early_source_registration": result["registration"], "early_build_result": supplement.bound(filename)})
    # Reuse the validated reader on the immutable source/card/point records;
    # this operation performs neither generation nor reduction.
    temporary = report / "preparation-core-link-pending.json"
    if temporary.exists():
        raise ValueError("A previous linkage attempt remains retained")
    bench.dump(temporary, record)
    stress.validate_preparation(temporary, expected, signature)
    bench.dump(destination, record)
    return destination


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: early_build.py EARLY-SPEC.json (CLI/build/masses/points only)")
    print(json.dumps(prepare(Path(sys.argv[1])), indent=2))
