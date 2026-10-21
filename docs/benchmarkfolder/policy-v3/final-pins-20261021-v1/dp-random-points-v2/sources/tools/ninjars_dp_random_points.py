#!/usr/bin/env python3
"""Prepare reproducible random DP measurement points, without numerical calls.

The proposal construction follows ninjars_benchmark.make_points: MadGraph RAMBO
and FortranList for multiplicity >= 2, or a common random boost for 2->1.
Rejection cuts define this separate measurement sample, not the fixed correctness
streams. No generated executable, compiler, reducer or Rust context is used.
"""
import argparse
from collections import Counter
import copy
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
CONFIG_SCHEMA = "ninjars-dp-random-points-config-v1"
SCHEMA = "ninjars-dp-random-points-v1"
RELEASE_KEYS = ("numerical_revision", "generation_revision", "generator_identity_sha256",
                "runtime_identity_sha256", "archive_sha256")
CASE_ORDER = ("qcd-1", "ew-1", "qcd-2", "ew-2", "qcd-3", "ew-3", "qcd-4", "ew-4",
              "supplement-gg-h", "supplement-gg-hh", "stress-qcd-4-gg-ttgg", "stress-ew-4-uu-4l")
BASE = ROOT / "docs/benchmarkfolder/policy-v3/final-pins-20261021-v1"
DEFAULT_OUTPUT = BASE / "dp-random-points-v2"
SOURCE_ROLES = {"tools/ninjars_dp_random_points.py": "execution",
                "madgraph/__init__.py": "execution",
                "madgraph/various/__init__.py": "execution",
                "madgraph/various/rambo.py": "execution",
                "tools/ninjars_supplement_v2.py": "execution",
                "tools/ninjars_benchmark.py": "execution",
                "tools/ninjars_benchmark_policy_v2.py": "execution",
                "tools/ninjars_benchmark_references_v2.py": "execution",
                "tools/ninjars_benchmark_workers.py": "execution",
                "tools/ninjars_dp_stability.py": "readiness-only"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def path(value):
    candidate = Path(value)
    candidate = candidate.resolve() if candidate.is_absolute() else (ROOT / candidate).resolve()
    if not candidate.is_relative_to(ROOT):
        raise ValueError("Random point evidence must remain inside the repository")
    return candidate


def bound(value):
    filename = path(value)
    return {"path": str(filename.relative_to(ROOT)), "sha256": digest(filename), "bytes": filename.stat().st_size}


def checked(record, inputs=None):
    filename = path(record["path"])
    if digest(filename) != record["sha256"] or ("bytes" in record and filename.stat().st_size != record["bytes"]):
        raise ValueError("Changed random point evidence: " + str(filename))
    if inputs is not None:
        inputs[str(filename.relative_to(ROOT))] = record["sha256"]
    return filename


def dump(filename, value):
    Path(filename).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def validate_origins(config, cohort, inputs=None):
    """Match every measurement definition to its independently retained export."""
    from ninjars_benchmark_policy_v2 import verified
    root = path(cohort["generation_root"])
    core_path = root / "status.json"
    core = json.loads(checked(bound(core_path), inputs).read_text())
    for name, case in config["cases"].items():
        actual = name.replace("supplement-", "control-") if name.startswith("supplement-") else name
        if name.startswith("stress-"):
            final_path = root.parent / "stress-generation/final-result.json"
            generation = bound(final_path)
            final = verified(json.loads(checked(generation, inputs).read_text()), "final-pin-stress-generation-v1")
            if final["status"] != "complete" or any(final[key] != config["release"][key] for key in
                    ("generation_revision", "generator_identity_sha256", "runtime_identity_sha256", "archive_sha256")):
                raise ValueError("Random stress origin belongs to another/incomplete cohort")
            early = verified(json.loads(checked(final["early_build_result"], inputs).read_text()), "stress-early-build-result-v1")
            status = json.loads(checked(early["status"], inputs).read_text())
            observed = status["cases"][actual]
            if status["status"] != "complete" or observed["source_proof"] != early["source_proofs"][actual]:
                raise ValueError("Random stress source proof differs from retained export")
            proof = verified(json.loads(checked(observed["source_proof"], inputs).read_text()), "stress-generation-source-v1")
            if proof["reducer_runtime"]["identity_sha256"] != config["release"]["runtime_identity_sha256"]:
                raise ValueError("Random stress source proof runtime differs")
            definition = proof["case"]
            physical_card = proof["physical_card"]
            authority = {"kind": "stress-v3", "generation_case": actual, "generation": generation}
        else:
            status_path = root.parent / "controls-generation/status.json" if name.startswith("supplement-") else core_path
            generation = bound(status_path)
            status = core if status_path == core_path else json.loads(checked(generation, inputs).read_text())
            if (status.get("complete") is not True or status["git_revision"] != config["release"]["generation_revision"] or
                    status["generator_tree_state_sha256"] != config["release"]["generator_identity_sha256"]):
                raise ValueError("Random CLI origin belongs to another/incomplete cohort")
            observed = status["cases"][actual]
            if observed["status"] != "complete":
                raise ValueError("Random CLI process export is incomplete")
            definition = observed["case"]
            physical_card = bound(status_path.parent / "inputs/candidate" / actual / "param_card.dat")
            if physical_card["sha256"] != observed["physical_inputs"]["param_card"]["sha256"]:
                raise ValueError("Random retained mass card differs from CLI export")
            authority = None
            if name.startswith("supplement-"):
                measurement = bound(ROOT / "docs/benchmarkfolder/policy-v3/paper-timing-software-proof-v1/specification-held-v2.json")
                checked(measurement, inputs)
                authority = {"kind": "controls-v3", "generation_case": actual,
                             "generation": generation, "measurement": measurement}
        expected = {"definition": {**definition, "id": name}, "actual_generation_case": actual,
                    "origin_authority": authority, "generation": generation,
                    "masses": observed["masses"], "physical_card": physical_card}
        if any(case[key] != value for key, value in expected.items()):
            raise ValueError("Random measurement case differs from retained CLI authority: " + name)
        checked(physical_card, inputs)


def release_identity(value):
    if set(value) != set(RELEASE_KEYS) or any(not isinstance(value[key], str) or not value[key] for key in RELEASE_KEYS):
        raise ValueError("Random measurement requires exactly five nonempty release identifiers")
    return {key: value[key] for key in RELEASE_KEYS}


def case_cuts(definition):
    if definition["nfinal"] == 1:
        return {"pt_min_gev": None, "delta_r_min": None, "eta_definition": "pseudorapidity",
                "interpretation": "On-shell 2->1 random common boosts are descriptive, with no cuts"}
    return {"pt_min_gev": None if "sqrvirt" in definition["process"] else 50.0,
            "delta_r_min": 0.5, "eta_definition": "pseudorapidity",
            "interpretation": "Strict final-state cuts; loop-induced reference omits pT only"}


def cut_values(event):
    final = event[2:]
    pts = [math.hypot(p[1], p[2]) for p in final]
    if any(pt == 0 for pt in pts):
        return min(pts), 0.0
    eta = [math.asinh(p[3] / pt) for p, pt in zip(final, pts)]
    phi = [math.atan2(p[2], p[1]) for p in final]
    distances = [math.hypot(eta[i] - eta[j], math.atan2(math.sin(phi[i] - phi[j]), math.cos(phi[i] - phi[j])))
                 for i in range(len(final)) for j in range(i)]
    return min(pts), min(distances) if distances else None


def rejection(event, cuts):
    pt, dr = cut_values(event)
    if cuts["pt_min_gev"] is not None and pt <= cuts["pt_min_gev"]:
        return "pt"
    if cuts["delta_r_min"] is not None and (dr is None or dr <= cuts["delta_r_min"]):
        return "delta_r"
    return None


def proposal(masses, energy, rng):
    """Use the established RAMBO/boost construction with a private RNG stream."""
    nfinal = len(masses) - 2
    if nfinal == 1:
        mass = masses[2]
        if mass <= sum(masses[:2]):
            raise ValueError("2->1 final mass must exceed the incoming threshold")
        e1 = (mass * mass + masses[0] ** 2 - masses[1] ** 2) / (2 * mass)
        pz = math.sqrt(e1 * e1 - masses[0] ** 2)
        event = [[e1, 0., 0., pz], [mass - e1, 0., 0., -pz], [mass, 0., 0., 0.]]
        beta = [rng.uniform(-0.25, 0.25) for _ in range(3)]
        b2 = sum(b * b for b in beta)
        gamma = 1 / math.sqrt(1 - b2)
        if b2:
            return [[gamma * (p[0] + sum(b * x for b, x in zip(beta, p[1:])))] + [
                x + ((gamma - 1) * sum(b * y for b, y in zip(beta, p[1:])) / b2 + gamma * p[0]) * b
                for b, x in zip(beta, p[1:])] for p in event]
        return event
    if energy <= max(sum(masses[:2]), sum(masses[2:])):
        raise ValueError("RAMBO energy is below threshold")
    from madgraph.various import rambo
    e1 = (energy ** 2 + masses[0] ** 2 - masses[1] ** 2) / (2 * energy)
    pz = math.sqrt(e1 ** 2 - masses[0] ** 2)
    xm = rambo.FortranList(nfinal)
    for index, mass in enumerate(masses[2:], 1):
        xm[index] = mass
    # RAMBO's only random primitive is its module's random.uniform. Restore it
    # even on failure, without changing Python's ambient RNG state.
    previous = rambo.random
    try:
        rambo.random = rng
        momenta, _ = rambo.RAMBO(nfinal, energy, xm)
    finally:
        rambo.random = previous
    return [[e1, 0., 0., pz], [energy - e1, 0., 0., -pz]] + [
        [momenta[4, index], momenta[1, index], momenta[2, index], momenta[3, index]]
        for index in range(1, nfinal + 1)]


def stream(case, count, energy, max_proposals):
    masses = case["masses"]
    if (len(masses) != case["definition"]["nfinal"] + 2 or
            any(not math.isfinite(m) or m < 0 for m in masses) or
            not isinstance(count, int) or isinstance(count, bool) or count <= 0 or
            not math.isfinite(energy) or energy <= 0 or
            not isinstance(case["seed"], int) or isinstance(case["seed"], bool) or
            not isinstance(max_proposals, int) or max_proposals < count):
        raise ValueError("Invalid random phase-space dimensions/masses/count")
    if case["cuts"] != case_cuts(case["definition"]):
        raise ValueError("Random phase-space cuts differ from the declared process policy")
    rng = random.Random(case["seed"])
    events, rejected, proposals = [], Counter(), 0
    while len(events) < count:
        if proposals >= max_proposals:
            raise ValueError("Random point proposal budget exhausted")
        event = proposal(masses, energy, rng)
        scale = max(p[0] for p in event)
        total = [event[0][j] + event[1][j] for j in range(4)]
        invariant_s = total[0] ** 2 - sum(value * value for value in total[1:])
        expected_s = masses[2] ** 2 if len(masses) == 3 else energy ** 2
        if (scale <= 0 or any(p[0] <= 0 for p in event) or
                any(not math.isfinite(value) for p in event for value in p) or
                abs(invariant_s - expected_s) > 1e-10 * expected_s or
                max(abs(sum(p[j] for p in event[:2]) - sum(p[j] for p in event[2:])) for j in range(4)) > 1e-10 * scale or
                max(abs(p[0] ** 2 - sum(value * value for value in p[1:]) - mass * mass)
                    for p, mass in zip(event, masses)) > 1e-10 * scale * scale):
            raise ValueError("Nonphysical RAMBO/boost proposal")
        proposals += 1
        reason = rejection(event, case["cuts"])
        if reason:
            rejected[reason] += 1
        else:
            events.append(event)
    raw = (str(count) + "\n" + "".join(" ".join(f"{value:.17e}" for value in p) + "\n"
                                      for event in events for p in event)).encode()
    accepted_values = [cut_values(event) for event in events]
    return raw, {"proposals": proposals, "rejected": proposals - count,
                 "rejection_reasons": dict(sorted(rejected.items())),
                 "min_final_pt_gev": min(value[0] for value in accepted_values),
                 "min_pair_delta_r": min(value[1] for value in accepted_values) if len(masses) > 3 else None}


def prepare(output=DEFAULT_OUTPUT, release_path=BASE / "candidate-cohort.json"):
    """Resolve the twelve existing CLI authorities and freeze before generation."""
    output = path(output)
    if output.exists():
        raise ValueError("Random point destination exists; choose a new namespace")
    # Readiness only. It validates original generation/source authorities but
    # does not select the random measurement stream or run any generated code.
    readiness_digest = digest(ROOT / "tools/ninjars_dp_stability.py")
    dp = importlib.import_module("ninjars_dp_stability")
    release = json.loads(path(release_path).read_text())
    identity = release_identity({key: release[key] for key in RELEASE_KEYS})
    core_path = path(release["generation_root"]) / "status.json"
    controls_path = core_path.parent.parent / "controls-generation/status.json"
    stress_path = core_path.parent.parent / "stress-generation/final-result.json"
    measurement_path = ROOT / "docs/benchmarkfolder/policy-v3/paper-timing-software-proof-v1/specification-held-v2.json"
    definitions, readiness_paths = {}, set()
    for index, name in enumerate(CASE_ORDER):
        run = {"case": name}
        if name.startswith("supplement-"):
            run["origin_authority"] = {"kind": "controls-v3", "generation_case": name.replace("supplement-", "control-"),
                "generation": bound(controls_path), "measurement": bound(measurement_path)}
        elif name.startswith("stress-"):
            run["origin_authority"] = {"kind": "stress-v3", "generation_case": name, "generation": bound(stress_path)}
        resolved = dp.resolve_origin(run, release, readiness_paths)
        observed = resolved["case"]
        definition = {**observed["case"], "id": name}
        masses = observed.get("masses")
        if masses is None:
            raise ValueError("CLI generation authority lacks retained external masses: " + name)
        physical_card = resolved["retained"] / "param_card.dat"
        card = bound(physical_card)
        expected_card = observed.get("physical_inputs", {}).get("param_card")
        if expected_card and card["sha256"] != expected_card["sha256"]:
            raise ValueError("Retained physical mass card differs from CLI authority")
        definitions[name] = {"definition": definition, "actual_generation_case": resolved["generation_case"],
            "origin_authority": copy.deepcopy(run.get("origin_authority")), "generation": bound(core_path) if "origin_authority" not in run else run["origin_authority"]["generation"],
            "masses": masses, "physical_card": card, "seed": 20261021 + index,
            "cuts": case_cuts(definition)}
    output.mkdir(parents=True)
    snapshots = output / "sources"
    snapshots.mkdir()
    sources = {}
    if digest(ROOT / "tools/ninjars_dp_stability.py") != readiness_digest:
        raise ValueError("Readiness resolver changed during configuration preparation")
    for relative, role in SOURCE_ROLES.items():
        snapshot = snapshots / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / relative).read_bytes())
        sources[relative] = {"role": role, "snapshot": bound(snapshot)}
    configuration = {"schema": CONFIG_SCHEMA, "sample_kind": "independent-random-dp-measurement",
        "release": identity, "candidate_cohort": bound(release_path), "points_per_case": 100,
        "seed_base": 20261021, "energy_gev": 1000., "max_proposals_per_case": 1000000,
        "boost_component_interval": [-0.25, 0.25], "case_order": list(CASE_ORDER), "cases": definitions,
        "sources": sources, "readiness_inputs": {str(path(filename).relative_to(ROOT)): digest(filename)
                                                   for filename in sorted(readiness_paths)}, "amplitude_evaluations": 0,
        "initialized_symbolica_workers": 0}
    authority_inputs = {}
    validate_origins(configuration, release, authority_inputs)
    configuration["readiness_inputs"].update(authority_inputs)
    dump(output / "configuration.json", configuration)
    return output / "configuration.json"


def validate_configuration(config, inputs=None):
    if config.get("schema") != CONFIG_SCHEMA or config.get("sample_kind") != "independent-random-dp-measurement":
        raise ValueError("Unsupported random point configuration")
    release_identity(config["release"])
    if (config["case_order"] != list(CASE_ORDER) or set(config["cases"]) != set(CASE_ORDER) or
            config["points_per_case"] != 100 or config["seed_base"] != 20261021 or
            config["energy_gev"] != 1000. or config["max_proposals_per_case"] != 1000000 or
            config["boost_component_interval"] != [-0.25, 0.25]):
        raise ValueError("Random measurement preregistration differs")
    cohort = json.loads(checked(config["candidate_cohort"], inputs).read_text())
    if {key: cohort[key] for key in RELEASE_KEYS} != config["release"]:
        raise ValueError("Random point candidate cohort differs")
    if {relative: item["role"] for relative, item in config["sources"].items()} != SOURCE_ROLES:
        raise ValueError("Random point source inventory differs")
    for relative, item in config["sources"].items():
        snapshot = checked(item["snapshot"], inputs)
        if item["role"] == "execution" and digest(ROOT / relative) != digest(snapshot):
            raise ValueError("Random point execution source changed: " + relative)
    authority_inputs = {}
    validate_origins(config, cohort, authority_inputs)
    if not authority_inputs or any(config["readiness_inputs"].get(name) != value for name, value in authority_inputs.items()):
        raise ValueError("Random point readiness input closure is incomplete")
    for relative, expected in config["readiness_inputs"].items():
        checked({"path": relative, "sha256": expected}, inputs)
    for index, name in enumerate(config["case_order"]):
        case = config["cases"][name]
        if (case["definition"]["id"] != name or case["seed"] != config["seed_base"] + index or
                case["cuts"] != case_cuts(case["definition"])):
            raise ValueError("Random point per-case preregistration differs")
        checked(case["physical_card"], inputs)
        checked(case["generation"], inputs)
    if config["amplitude_evaluations"] != 0 or config["initialized_symbolica_workers"] != 0:
        raise ValueError("Point preparation must not initialize numerical workers")
    return config


def generate(configuration_path):
    configuration_path = path(configuration_path)
    config = validate_configuration(json.loads(configuration_path.read_text()))
    output = configuration_path.parent
    if (output / "points").exists() or (output / "manifest.json").exists():
        raise ValueError("Random points already exist; never replace a frozen measurement stream")
    (output / "points").mkdir()
    rows = {}
    from ninjars_supplement_v2 import point_metadata
    for name in config["case_order"]:
        case = config["cases"][name]
        raw, accounting = stream(case, config["points_per_case"], config["energy_gev"], config["max_proposals_per_case"])
        filename = output / "points" / (name + ".dat")
        filename.write_bytes(raw)
        physics = point_metadata(filename, case["masses"], case["definition"]["nfinal"])
        metadata = {**case, **accounting, "physical_validation": physics, "points_count": config["points_per_case"],
                    "sample_kind": config["sample_kind"], "energy_gev": config["energy_gev"]}
        metadata_path = filename.with_suffix(".json")
        dump(metadata_path, metadata)
        rows[name] = {**case, **accounting, "points_count": config["points_per_case"],
                      "points": bound(filename), "metadata": bound(metadata_path)}
    manifest = {"schema": SCHEMA, "sample_kind": config["sample_kind"], "configuration": bound(configuration_path),
                "release": config["release"], "cases": rows, "amplitude_evaluations": 0, "initialized_symbolica_workers": 0}
    dump(output / "manifest.json", manifest)
    return output / "manifest.json"


def validate_manifest(manifest_path, expected_release=None, inputs=None):
    """Recompute all point bytes, cuts, accounting and physical residuals."""
    manifest_path = path(manifest_path)
    if inputs is not None:
        inputs[str(manifest_path.relative_to(ROOT))] = digest(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Unsupported random point manifest")
    config = validate_configuration(json.loads(checked(manifest["configuration"], inputs).read_text()), inputs)
    release_identity(manifest["release"])
    if (manifest["release"] != config["release"] or
            expected_release is not None and manifest["release"] != release_identity(expected_release)):
        raise ValueError("Random point release differs")
    if set(manifest["cases"]) != set(config["cases"]) or manifest["sample_kind"] != config["sample_kind"]:
        raise ValueError("Random point case inventory/sample differs")
    from ninjars_supplement_v2 import point_metadata
    for name, case in config["cases"].items():
        row = manifest["cases"][name]
        filename = checked(row["points"], inputs)
        metadata = json.loads(checked(row["metadata"], inputs).read_text())
        raw, accounting = stream(case, config["points_per_case"], config["energy_gev"], config["max_proposals_per_case"])
        if raw != filename.read_bytes():
            raise ValueError("Random points differ from frozen deterministic seed/cuts: " + name)
        physics = point_metadata(filename, case["masses"], case["definition"]["nfinal"])
        expected_metadata = {**case, **accounting, "physical_validation": physics, "points_count": config["points_per_case"],
                             "sample_kind": config["sample_kind"], "energy_gev": config["energy_gev"]}
        expected_row = {**case, **accounting, "points_count": config["points_per_case"],
                        "points": row["points"], "metadata": row["metadata"]}
        if row != expected_row or metadata != expected_metadata:
            raise ValueError("Random point metadata/accounting differs: " + name)
    if manifest["amplitude_evaluations"] != 0 or manifest["initialized_symbolica_workers"] != 0:
        raise ValueError("Random point generation must not initialize numerical workers")
    return manifest, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    prepare_parser.add_argument("--release", type=Path, default=BASE / "candidate-cohort.json")
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("configuration", type=Path)
    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("manifest", type=Path)
    arguments = parser.parse_args()
    if arguments.command == "prepare":
        print(prepare(arguments.output, arguments.release))
    elif arguments.command == "generate":
        print(generate(arguments.configuration))
    else:
        manifest, config = validate_manifest(arguments.manifest)
        print(json.dumps({"status": "verified", "cases": len(manifest["cases"]), "points_per_case": config["points_per_case"]}))
