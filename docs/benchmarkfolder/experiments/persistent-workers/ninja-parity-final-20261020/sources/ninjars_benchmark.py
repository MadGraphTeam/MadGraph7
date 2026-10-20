#!/usr/bin/env python3
"""Generate and measure identical MadLoop inputs through the real CLI.

Run in the pinned development shell. All generated programs and logs are kept
under .benchmark-work; retained point streams and measurements live in the report.
"""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/benchmarkfolder"
WORK = ROOT / ".benchmark-work"
BASELINE = WORK / "original"
BASELINE_REVISION = "27147b3300f556a75661a886e92bdbe00558ee25"
GENERATOR_PATHS = ("madgraph", "aloha", "Template", "bin", "models", "input")
sys.path.insert(0, str(ROOT))


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def run(command, cwd, log, **kwargs):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               XDG_CONFIG_HOME=str(WORK / "config"))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(str(p) for p in (
        BASELINE / "HEPTools/ninja/lib", BASELINE / "HEPTools/collier",
        ROOT / "HEPTools/ninjars/lib")) + (
            os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    # Existing MadLoop makefiles assume /bin/bash, absent on NixOS.
    env["MAKEFLAGS"] = "SHELL=" + shutil.which("bash")
    locks = ROOT / ".ninjars-build/symbolica-locks"
    locks.mkdir(parents=True, exist_ok=True, mode=0o700)
    env.setdefault("SYMBOLICA_LOCK_DIR", str(locks))
    start = time.perf_counter()
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, **kwargs)
    elapsed = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"{command[0]} failed ({result.returncode}); see {log}")
    return elapsed


def output_dir(source, case):
    return WORK / "outputs" / source / case["id"]


def content_digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def retain_source_identity(identity, patch=b""):
    """Store content-addressed source evidence separately from disposable builds."""
    retained = REPORT / "inputs/source-identities"
    retained.mkdir(parents=True, exist_ok=True)
    identity = dict(identity)
    identity["dirty_patch_sha256"] = hashlib.sha256(patch).hexdigest() if patch else None
    if patch:
        path = retained / (identity["dirty_patch_sha256"] + ".patch")
        path.write_bytes(patch)
        identity["dirty_patch_path"] = str(path.relative_to(REPORT))
    identity["identity_sha256"] = content_digest(identity)
    dump(retained / (identity["identity_sha256"] + ".json"), identity)
    return identity


def generator_tree_state():
    """Fingerprint tracked and untracked generator inputs, excluding report data."""
    def paths(*options):
        return subprocess.check_output(
            ["git", "ls-files", "-z", *options, "--", *GENERATOR_PATHS],
            cwd=ROOT).decode().split("\0")
    tracked, untracked = {}, {}
    for collection, names in ((tracked, paths()),
                              (untracked, paths("--others", "--exclude-standard"))):
        for name in filter(None, names):
            path = ROOT / name
            collection[name] = (hashlib.sha256(path.read_bytes()).hexdigest()
                                if path.is_file() else None)
    return {"tracked": tracked, "untracked": untracked}


def capture_generation_identity(source):
    if source == "baseline":
        return retain_source_identity({
            "status": "recorded-before-generation",
            "revision": BASELINE_REVISION,
            "source_kind": "verified git archive",
            "verification": "provenance.json:baseline_source_integrity",
            "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
    state = generator_tree_state()
    patch = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", *GENERATOR_PATHS], cwd=ROOT)
    # Untracked generator files are not part of a Git patch; retain their bytes too.
    additions = REPORT / "inputs/source-identities/untracked"
    for name, digest in state["untracked"].items():
        if digest:
            additions.mkdir(parents=True, exist_ok=True)
            (additions / digest).write_bytes((ROOT / name).read_bytes())
    return retain_source_identity({
        "status": "recorded-before-generation",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": list(GENERATOR_PATHS),
        "tree_state": state,
        "tree_state_sha256": content_digest(state),
        "captured_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }, patch)


def generation_identity(source, case):
    """Never substitute measurement-time HEAD for an unrecorded generation."""
    path = output_dir(source, case) / "generation.json"
    record = json.loads(path.read_text())
    if "source_identity" not in record:
        identity = {
            "status": "not-recorded",
            "revision": None,
            "reason": "Generation predates source-identity recording; current HEAD is not evidence.",
        }
        if source == "baseline":
            identity.update(status="reconstructed-from-verified-archive",
                            revision=BASELINE_REVISION,
                            reason="The original archive was independently verified unchanged.",
                            verification="provenance.json:baseline_source_integrity")
        record["source_identity"] = retain_source_identity(identity)
        dump(path, record)
    return record["source_identity"]


def generated_source_manifest(output):
    """Hash generated source/include/build files, following individual file links."""
    suffixes = {".f", ".f90", ".for", ".inc", ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp"}
    files = {}
    for subtree in ("Source", "SubProcesses"):
        for path in sorted((output / subtree).rglob("*")):
            if not (path.suffix.lower() in suffixes or
                    path.name.lower().startswith("makefile") or
                    path.name in ("make_opts", "MadLoop_makefile_definitions")):
                continue
            if path.is_symlink() and not path.exists():
                raise RuntimeError(f"Broken generated-source link: {path}")
            if path.is_file():
                content = path.read_bytes()
                files[str(path.relative_to(output))] = {
                    "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
    return {"schema_version": 1, "files": files,
            "files_sha256": content_digest(files),
            "scope": "Generated Source/SubProcesses source, include and make files; "
                     "external library trees are covered by dependency provenance."}


def retain_build_inputs(source, case):
    """Keep the exact generated cards/build settings outside disposable outputs."""
    output = output_dir(source, case)
    retained = REPORT / "inputs" / source / case["id"]
    retained.mkdir(parents=True, exist_ok=True)
    for relative in ("Cards/param_card.dat", "generation.json", "compilation.json",
                     "compiled-sources.json", "generated-sources-observed.json"):
        path = output / relative
        if path.exists():
            shutil.copy2(path, retained / path.name)
    manifest = output / "compiled-sources.json"
    if manifest.exists():
        digest = json.loads(manifest.read_text())["files_sha256"]
        target = retained / "compiled-source-manifests" / (digest + ".json")
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(manifest, target)
    for path in (output / "compilation-history").glob("*.json"):
        target = retained / "compilation-history" / path.name
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(path, target)


def generate(source, case):
    tree = BASELINE if source == "baseline" else ROOT
    output = output_dir(source, case)
    if (output / "generation.json").exists():
        generation_identity(source, case)
        retain_build_inputs(source, case)
        return
    if (output / "SubProcesses").exists():
        raise RuntimeError(f"Incomplete previous generation: move {output} aside before retry")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = case["model"]
    if model != "loop_sm":
        model = str(BASELINE / "models" / model)
    lines = [
        "set automatic_html_opening False",
        "set auto_update 0",
        "set nb_core 4",
        "set loop_optimized_output True",
        "set output_dependencies external",
        f"set complex_mass_scheme {bool(case.get('complex_mass_scheme'))}",
        f"set ninja {BASELINE / 'HEPTools/ninja/lib'}",
        f"set collier {BASELINE / 'HEPTools/collier'}",
    ]
    if source != "baseline":
        lines.append(f"set ninjars {ROOT / 'HEPTools/ninjars/lib'}")
    lines += [f"import model {model}", f"generate {case['process']}",
              f"output standalone {output}", "quit"]
    card = REPORT / "commands" / f"{source}-{case['id']}.mg5"
    card.parent.mkdir(parents=True, exist_ok=True)
    card.write_text("\n".join(lines) + "\n")
    log = WORK / "logs" / f"generate-{source}-{case['id']}.log"
    source_identity = capture_generation_identity(source)
    seconds = run([sys.executable, str(tree / "bin/madgraph"), "--plain",
                   str(card)], ROOT, log)
    if (source == "candidate" and
            content_digest(generator_tree_state()) != source_identity["tree_state_sha256"]):
        raise RuntimeError("Generator inputs changed during export; regenerate this output")
    text = log.read_text()
    if ("interrupted in sub-command" in text or "Error detected in sub-command" in text
            or not (output / "Cards/param_card.dat").exists()):
        raise RuntimeError(f"CLI generation failed: {log}")
    from madgraph.models.check_param_card import ParamCard
    param_path = output / "Cards/param_card.dat"
    param = ParamCard(str(param_path))
    if not case.get("complex_mass_scheme"):
        for decay in param["decay"]:
            decay.value = 0.0
    param.write(str(param_path))
    dump(output / "generation.json", {"seconds": seconds,
         "parameter_card_sha256": hashlib.sha256(param_path.read_bytes()).hexdigest(),
         "source_identity": source_identity})
    retain_build_inputs(source, case)


def subprocess_dir(source, case):
    directories = sorted(output_dir(source, case).glob("SubProcesses/P*_*"))
    if len(directories) != 1:
        raise RuntimeError(f"Expected one subprocess for {case['id']}: {directories}")
    return directories[0]


def build(source, case):
    directory = subprocess_dir(source, case)
    output = output_dir(source, case)
    previous = output / "compilation.json"
    if previous.exists():
        history = output / "compilation-history"
        history.mkdir(exist_ok=True)
        digest = hashlib.sha256(previous.read_bytes()).hexdigest()[:16]
        shutil.copy2(previous, history / f"{digest}.json")
    preexisting_objects = len(list(directory.glob("*.o")))
    check = directory / "check_sa.f"
    original = directory / "check_sa.original"
    if not original.exists():
        shutil.copy2(check, original)
    text = original.read_text()
    prefix = re.search(r"CALL\s+(\w*)GET_ANSWER_DIMENSION", text, re.I).group(1)
    n = case["nfinal"] + 2
    born_include = directory / "nsqso_born.inc"
    nborn = (int(re.search(r"NSQSO_BORN\s*=\s*(\d+)", born_include.read_text(),
                           re.I).group(1)) if born_include.exists() else 0)
    driver = (ROOT / "tools/ninjars_benchmark_driver.f").read_text()
    check.write_text(driver.replace("@PREFIX@", prefix).replace("@NEXTERNAL@", str(n))
                     .replace("@NBORN@", str(nborn)))
    source_identity = generation_identity(source, case)
    source_manifest = generated_source_manifest(output)
    log = WORK / "logs" / f"compile-{source}-{case['id']}.log"
    seconds = run(["make", "-j4", "check"], directory, log)
    if generated_source_manifest(output) != source_manifest:
        raise RuntimeError("Generated sources changed while compiling; rebuild this output")
    dump(output / "compiled-sources.json", source_manifest)
    build_settings = {}
    for relative in ("Source/make_opts", "SubProcesses/MadLoop_makefile_definitions"):
        path = output_dir(source, case) / relative
        build_settings[relative] = path.read_text()
    dump(output / "compilation.json",
         {"seconds": seconds, "settings": build_settings,
          "preexisting_subprocess_objects": preexisting_objects,
          "build_kind": "incremental" if preexisting_objects else "clean-subprocess",
          "source_identity": source_identity,
          "generated_sources_sha256": source_manifest["files_sha256"],
          "source_manifest_recording": "recorded-before-and-verified-after-build",
          "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
          "driver_sha256": hashlib.sha256(check.read_bytes()).hexdigest(),
          "executable_sha256": hashlib.sha256((directory / "check").read_bytes()).hexdigest()})
    retain_build_inputs(source, case)
    run(["./check", "--masses"], directory,
        WORK / "logs" / f"masses-{source}-{case['id']}.log")
    return [float(x) for x in (directory / "masses.dat").read_text().split()]


def make_points(case, masses, config):
    """Use MadGraph's RAMBO, with fixed seeds and boosted physical 2->1 points."""
    from madgraph.various.rambo import RAMBO, FortranList
    target = REPORT / "points" / f"{case['id']}.dat"
    meta = target.with_suffix(".json")
    seed = config["seed"] + next(i for i, c in enumerate(config["cases"])
                                if c["id"] == case["id"])
    if len(masses) != case["nfinal"] + 2 or any(
            not math.isfinite(m) or m < 0 for m in masses):
        raise RuntimeError(f"Invalid external masses for {case['id']}: {masses}")
    if target.exists():
        saved = json.loads(meta.read_text())
        if (saved["masses"] != masses or saved["seed"] != seed or
                saved["points"] != config["points_per_case"] or
                saved.get("energy_gev") != config["energy_gev"] or
                saved["sha256"] != hashlib.sha256(target.read_bytes()).hexdigest()):
            raise RuntimeError(f"Point inputs changed for {case['id']}")
        return target
    random.seed(seed)
    events = []
    n = case["nfinal"]
    for _ in range(config["points_per_case"]):
        if n == 1:
            mass = masses[2]
            if mass <= masses[0] + masses[1]:
                raise RuntimeError("The 2->1 final mass must exceed the incoming threshold")
            energy1 = (mass * mass + masses[0]**2 - masses[1]**2) / (2 * mass)
            pz = math.sqrt(energy1**2 - masses[0]**2)
            point = [[energy1, 0., 0., pz],
                     [mass-energy1, 0., 0., -pz], [mass, 0., 0., 0.]]
            # A common boost preserves the on-shell final state and invariant s.
            beta = [random.uniform(-0.25, 0.25) for _ in range(3)]
            b2 = sum(x*x for x in beta)
            gamma = 1 / math.sqrt(1-b2)
            if b2:
                boosted = []
                for e, *p in point:
                    bp = sum(b*x for b, x in zip(beta, p))
                    boosted.append([gamma*(e+bp)] + [
                        x + ((gamma-1)*bp/b2 + gamma*e)*b
                        for b, x in zip(beta, p)])
                point = boosted
        else:
            energy = config["energy_gev"]
            if energy <= max(sum(masses[:2]), sum(masses[2:])):
                raise RuntimeError(f"Energy below threshold for {case['id']}")
            energy1 = (energy**2 + masses[0]**2 - masses[1]**2)/(2*energy)
            pz = math.sqrt(energy1**2-masses[0]**2)
            point = [[energy1, 0., 0., pz], [energy-energy1, 0., 0., -pz]]
            xm = FortranList(n)
            for j, mass in enumerate(masses[2:], 1):
                xm[j] = mass
            momenta, _ = RAMBO(n, energy, xm)
            point += [[momenta[4, j], momenta[1, j], momenta[2, j], momenta[3, j]]
                      for j in range(1, n+1)]
        scale = max(p[0] for p in point)
        residual = [sum(p[j] for p in point[:2]) -
                    sum(p[j] for p in point[2:]) for j in range(4)]
        shells = [p[0]**2 - sum(x*x for x in p[1:]) - m*m
                  for p, m in zip(point, masses)]
        if (not all(math.isfinite(x) for p in point for x in p) or
                max(abs(x) for x in residual) > 1e-10*scale or
                max(abs(x) for x in shells) > 1e-10*scale*scale):
            raise RuntimeError(f"Invalid generated phase-space point for {case['id']}")
        events.append(point)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(str(len(events)) + "\n" + "".join(
        " ".join(f"{x:.17e}" for x in p)+"\n" for event in events for p in event))
    dump(meta, {"masses": masses, "points": len(events), "seed": seed,
                "energy_gev": config["energy_gev"],
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    return target


def configure(source, case, backend, config=None):
    from madgraph.various.banner import MadLoopParam
    output = output_dir(source, case)
    card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
    card["MLReductionLib"] = str(backend)
    card["CTModeRun"] = -1
    card["CTModeInit"] = 1
    card["UseLoopFilter"] = False
    card["COLLIERComputeUVpoles"] = True
    card["COLLIERComputeIRpoles"] = True
    if config is None:
        config = json.loads((REPORT / "cases.json").read_text())
    parameters = dict(config.get("madloop_parameters", {}))
    parameters.update(case.get("madloop_parameters", {}))
    for name, value in parameters.items():
        card[name] = value
    tree = BASELINE if source == "baseline" else ROOT
    template = str(tree / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat")
    # Write both the user-facing card and the resource read by standalone code.
    card.write(str(output / "Cards/MadLoopParams.dat"), template=template)
    for path in (output / "SubProcesses").rglob("MadLoopParams.dat"):
        card.write(str(path), template=template)
    # Each invocation builds its own helicity filters from the same points.
    for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
        for path in (output / "SubProcesses").rglob(pattern):
            path.unlink()
    retain_build_inputs(source, case)
    shutil.copy2(output / "Cards/MadLoopParams.dat",
                 REPORT / "inputs" / source / case["id"] / f"MadLoopParams-{backend}.dat")
    # The backend selector and the new NinjaRS-specific option are the only
    # intentional differences. Compare every shared numerical setting.
    settings = {str(k).lower(): v for k, v in card.items()
                if str(k).lower() not in ("mlreductionlib", "useqpintegrandforninjars")}
    qp_key = {1: "UseQPIntegrandForCutTools", 6: "UseQPIntegrandForNinja",
              8: "UseQPIntegrandForNinjaRS"}.get(backend)
    if qp_key:
        settings["selected_backend_qp_integrand"] = card[qp_key]
    return settings


def validate_status(code, backend):
    h, t, u = code // 100, (code // 10) % 10, code % 10
    allowed = {1: {(2, 1), (3, 9)}, 6: {(2, 6), (3, 8)},
               7: {(2, 7)}, 8: {(2, 8), (3, 2)}}
    if t not in (1, 2) or (h, u) not in allowed[backend]:
        raise RuntimeError(f"Backend {backend} returned unexpected status {code}; "
                           "initialization, failure or a different backend was used")


def measure(source, case, backend, points, loops, repetition, cpu, config=None, persist=True):
    directory = subprocess_dir(source, case)
    if config is None:
        config = json.loads((REPORT / "cases.json").read_text())
    numerical_settings = configure(source, case, backend, config=config)
    identity = generation_identity(source, case)
    compilation_path = output_dir(source, case) / "compilation.json"
    compilation = json.loads(compilation_path.read_text())
    executable_hash = hashlib.sha256((directory / "check").read_bytes()).hexdigest()
    if executable_hash != compilation["executable_sha256"]:
        raise RuntimeError("Executable differs from retained compilation metadata")
    retain_build_inputs(source, case)
    param_bytes = (output_dir(source, case) / "Cards/param_card.dat").read_bytes()
    # Match LHA_READ.F's exact search order for SETPARA('param_card.dat').
    param_candidates = [directory / "param_card.dat"] + [
        directory.joinpath(*([".."] * level), "Cards/param_card.dat") for level in range(6)]
    actual_param = next((path for path in param_candidates if path.is_file()), None)
    if actual_param is None or actual_param.read_bytes() != param_bytes:
        raise RuntimeError("Subprocess reads a different parameter card")
    for name in ("timing.dat", "values.dat", "timed_status.dat"):
        (directory / name).unlink(missing_ok=True)
    point_count = int(points.read_text().splitlines()[0])
    if point_count != config["points_per_case"]:
        raise RuntimeError(f"Expected {config['points_per_case']} input points")
    log = WORK / "logs" / f"run-{source}-{case['id']}-{backend}-{repetition}.log"
    command = ["taskset", "-c", str(cpu), "./check", str(points), str(loops)]
    wall = run(command, directory, log)
    timing = [float(x) for x in (directory / "timing.dat").read_text().split()]
    if len(timing) != 2 or not all(math.isfinite(x) and x > 0 for x in timing):
        raise RuntimeError(f"Invalid timing output: {log}")
    rows = []
    for line in (directory / "values.dat").read_text().splitlines():
        fields = line.split()
        if len(fields) != 8:
            raise RuntimeError(f"Malformed numerical output: {line}")
        rows.append({"point": int(fields[0]), "order": int(fields[1]),
                     "status": int(fields[2]),
                     "coefficients": [float(x) for x in fields[3:7]],
                     "accuracy": float(fields[7])})
    if not rows or not all(math.isfinite(x) for row in rows
                          for x in row["coefficients"] + [row["accuracy"]]):
        raise RuntimeError(f"Missing/nonfinite output: {log}")
    dimensions = [int(x) for x in (directory / "dimensions.dat").read_text().split()]
    if len(dimensions) != 3:
        raise RuntimeError("Missing answer/split-order dimensions")
    answer_dimension, loop_orders, born_orders = dimensions
    if answer_dimension != max(loop_orders, born_orders) or born_orders > loop_orders:
        raise RuntimeError("Driver cannot omit Born-only split-order columns")
    orders = sorted({r["order"] for r in rows})
    expected = [(p, o) for p in range(1, point_count+1) for o in orders]
    if (orders != list(range(loop_orders + 1)) or
            [(r["point"], r["order"]) for r in rows] != expected):
        raise RuntimeError(f"Missing, duplicated or unordered numerical output: {log}")
    for row in rows:
        validate_status(row["status"], backend)
        if row["accuracy"] < 0:
            raise RuntimeError(f"Missing stability estimate: {log}")
    counts = {}
    for row in rows:
        if row["order"] == 0:
            status = str(row["status"])
            counts[status] = counts.get(status, 0) + 1
    timed_counts = {}
    for line in (directory / "timed_status.dat").read_text().splitlines():
        code, count = map(int, line.split())
        validate_status(code, backend)
        if count <= 0 or str(code) in timed_counts:
            raise RuntimeError(f"Invalid timed status histogram: {log}")
        timed_counts[str(code)] = count
    if sum(timed_counts.values()) != loops*point_count:
        raise RuntimeError(f"Timed evaluation count does not match inputs: {log}")
    result = {"source": source, "case": case["id"], "backend": backend,
              "repetition": repetition, "loops": loops, "cpu": cpu,
              "cold_seconds": timing[0], "seconds": timing[1],
              "seconds_per_point": timing[1]/(loops*point_count),
              "process_wall_seconds": wall, "status_counts": counts,
              "timed_status_counts": timed_counts,
              "points_sha256": hashlib.sha256(points.read_bytes()).hexdigest(),
              "parameter_card_sha256": hashlib.sha256(
                  (output_dir(source, case) / "Cards/param_card.dat").read_bytes()).hexdigest(),
              "executable_sha256": executable_hash,
              "madloop_card_sha256": hashlib.sha256(
                  (output_dir(source, case) / "Cards/MadLoopParams.dat").read_bytes()).hexdigest(),
              "shared_numerical_settings": numerical_settings,
              "answer_dimensions": {"allocated": answer_dimension,
                                    "loop_orders": loop_orders, "born_orders": born_orders},
              "git_revision": identity["revision"],
              "generation_source_identity": identity,
              "generated_sources_sha256": compilation.get("generated_sources_sha256"),
              "source_manifest_recording": compilation.get(
                  "source_manifest_recording", "not-recorded-at-build"),
              "compilation_metadata_sha256": hashlib.sha256(
                  compilation_path.read_bytes()).hexdigest(),
              "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "rows": rows}
    if persist:
        dump(measurement_path(result), result)
    return result


def measurement_path(result):
    return REPORT / "data" / (
        f"{result['source']}-{result['case']}-{result['backend']}-{result['repetition']}.json")


def quarantine_measurement(result, reason):
    """Preserve rejected/superseded evidence outside every accepted-data glob."""
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    name = measurement_path(result).stem
    target = REPORT / "rejected" / (
        f"{timestamp.strftime('%Y%m%dT%H%M%S%fZ')}-{name}.json")
    dump(target, {"reason": str(reason), "recorded_utc": timestamp.isoformat(),
                  "measurement": result})
    measurement_path(result).unlink(missing_ok=True)
    return target


def accept_repetition(result, config):
    try:
        if result["repetition"]:
            reference_path = measurement_path(dict(result, repetition=0))
            reference = json.loads(reference_path.read_text())
            validate_repetition(result, reference, config)
    except Exception as error:
        quarantine_measurement(result, error)
        raise
    dump(measurement_path(result), result)


def compare(case, config):
    results = {}
    hashes = set()
    for backend in (1, 6, 7, 8):
        path = REPORT / "data" / f"candidate-{case['id']}-{backend}-0.json"
        data = json.loads(path.read_text())
        results[backend] = data["rows"]
        hashes.add((data["points_sha256"], data["parameter_card_sha256"]))
    if len(hashes) != 1:
        raise RuntimeError("Numerical inputs differ across reduction backends")
    if len({len(rows) for rows in results.values()}) != 1:
        raise RuntimeError("Different numbers of numerical outputs across backends")
    first = results[1]
    failures = []
    point_scales = {}
    for rows in results.values():
        for row in rows:
            point_scales[row["point"]] = max(point_scales.get(row["point"], 0.),
                                             *(abs(x) for x in row["coefficients"]))
    for i, row in enumerate(first):
        values = [results[b][i] for b in results]
        if any((r["point"], r["order"]) != (row["point"], row["order"]) for r in values):
            raise RuntimeError("Mismatched point/order streams")
        scale = point_scales[row["point"]]
        for left in results:
            for right in results:
                if left >= right:
                    continue
                for k, (a, b) in enumerate(zip(results[left][i]["coefficients"],
                                               results[right][i]["coefficients"])):
                    tolerance = config["relative_tolerance"]*max(abs(a), abs(b))
                    tolerance += config["scale_tolerance"]*scale
                    if abs(a-b) > tolerance:
                        failures.append({"point": row["point"], "order": row["order"],
                            "coefficient": k, "backends": [left, right],
                            "values": [a, b], "tolerance": tolerance})
    dump(REPORT / "data" / f"agreement-{case['id']}.json",
         {"passed": not failures, "failures": failures})
    if failures:
        raise RuntimeError(f"{case['id']}: {len(failures)} numerical disagreements")


def compare_legacy(case, config):
    """Check optimized legacy reducers against the untouched original source."""
    failures, inputs = [], {}
    for backend in (1, 6, 7):
        records = {}
        for source in ("baseline", "candidate"):
            path = REPORT / "data" / f"{source}-{case['id']}-{backend}-0.json"
            records[source] = json.loads(path.read_text())
            inputs[f"{source}-{backend}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        left, right = records["baseline"], records["candidate"]
        for key in ("points_sha256", "parameter_card_sha256"):
            if left[key] != right[key]:
                raise RuntimeError(f"Different {key} in legacy regression")
        a, b = left["rows"], right["rows"]
        if [(r["point"], r["order"]) for r in a] != [(r["point"], r["order"]) for r in b]:
            raise RuntimeError("Mismatched legacy regression point/order streams")
        scales = {}
        for row in a+b:
            scales[row["point"]] = max(scales.get(row["point"], 0.),
                                       *(abs(x) for x in row["coefficients"]))
        for r, s in zip(a, b):
            for k, (x, y) in enumerate(zip(r["coefficients"], s["coefficients"])):
                tolerance = config["relative_tolerance"]*max(abs(x), abs(y))
                tolerance += config["scale_tolerance"]*scales[r["point"]]
                if not math.isfinite(x+y) or abs(x-y) > tolerance:
                    failures.append({"backend": backend, "point": r["point"],
                                     "order": r["order"], "coefficient": k,
                                     "values": [x, y], "tolerance": tolerance})
    dump(REPORT / "data" / f"legacy-agreement-{case['id']}.json",
         {"passed": not failures, "failures": failures, "input_sha256": inputs})
    if failures:
        raise RuntimeError(f"{case['id']}: {len(failures)} original-backend regressions")


def validate_repetition(result, reference, config):
    """Do not accept timings whose numerical outputs drift across repetitions."""
    for key in ("points_sha256", "parameter_card_sha256", "executable_sha256",
                "shared_numerical_settings", "answer_dimensions", "cpu"):
        if result[key] != reference[key]:
            raise RuntimeError(f"Repetition inputs changed ({key}); rerun the full series")
    left, right = reference["rows"], result["rows"]
    if [(r["point"], r["order"]) for r in left] != [
            (r["point"], r["order"]) for r in right]:
        raise RuntimeError("Split-order stream changed across repetitions")
    scales = {}
    for row in left + right:
        scales[row["point"]] = max(scales.get(row["point"], 0.),
                                    *(abs(x) for x in row["coefficients"]))
    for a, b in zip(left, right):
        for x, y in zip(a["coefficients"], b["coefficients"]):
            tolerance = (config["relative_tolerance"] * max(abs(x), abs(y)) +
                         config["scale_tolerance"] * scales[a["point"]])
            if not math.isfinite(x+y) or abs(x-y) > tolerance:
                raise RuntimeError("Numerical result changed across repetitions")


def runtime_statistics(baseline, candidate, limit, seed=410083):
    """Paired bootstrap of the ratio of medians, preserving run-order pairing."""
    if len(baseline) != len(candidate) or len(baseline) < 7:
        raise RuntimeError("At least seven paired repetitions are required")
    if any(not math.isfinite(x) or x <= 0 for x in baseline + candidate):
        raise RuntimeError("Invalid timing samples")
    rng = random.Random(seed)
    n = len(baseline)
    ratios = []
    for _ in range(10000):
        indices = rng.choices(range(n), k=n)
        ratios.append(statistics.median(candidate[i] for i in indices) /
                      statistics.median(baseline[i] for i in indices))
    ratios.sort()
    ci = [ratios[249], ratios[9749]]
    ratio = statistics.median(candidate) / statistics.median(baseline)
    status = "passed" if ci[1] <= limit else "failed" if ci[0] > limit else "unresolved"
    def describe(values):
        med = statistics.median(values)
        return {"samples": values, "median": med, "minimum": min(values),
                "maximum": max(values), "stdev": statistics.stdev(values),
                "median_absolute_deviation": statistics.median(abs(x-med) for x in values)}
    return {"passed": status == "passed", "status": status, "runtime_ratio": ratio,
            "ratio_confidence_interval_95": ci, "maximum_ratio": limit,
            "bootstrap": {"method": "paired percentile", "resamples": 10000, "seed": seed},
            "measurements": {"baseline": describe(baseline), "candidate": describe(candidate)}}


def compare_performance(case, config):
    """Require warmed repeated measurements against untouched original Ninja."""
    groups = {}
    hashes = set()
    cpus = set()
    pair_ids = {}
    settings = set()
    for source, backend in (("baseline", 6), ("candidate", 8)):
        values = []
        executable_hashes = set()
        pairs = []
        paths = [p for p in (REPORT / "data").glob(f"{source}-{case['id']}-{backend}-*.json")
                 if p.stem.rsplit("-", 1)[1].isdigit()]
        paths.sort(key=lambda p: int(p.stem.rsplit("-", 1)[1]))
        if [int(p.stem.rsplit("-", 1)[1]) for p in paths] != list(range(len(paths))):
            raise RuntimeError("Repetition indices must be contiguous")
        for path in paths:
            data = json.loads(path.read_text())
            if not data.get("paired_baseline_candidate"):
                raise RuntimeError("Performance gate requires --source both measurements")
            values.append(data["seconds_per_point"])
            hashes.add((data["points_sha256"], data["parameter_card_sha256"]))
            cpus.add(data["cpu"])
            executable_hashes.add(data["executable_sha256"])
            if "shared_numerical_settings" not in data:
                raise RuntimeError("Missing recorded numerical settings; rerun measurements")
            settings.add(json.dumps(data["shared_numerical_settings"], sort_keys=True))
            pairs.append(data["measurement_run_id"])
        if len(executable_hashes) != 1:
            raise RuntimeError("Different executables across repetitions; rerun the full measurement")
        pair_ids[source] = pairs
        groups[source] = values
    if len(hashes) != 1:
        raise RuntimeError("Performance inputs differ between original Ninja and NinjaRS")
    if len(cpus) != 1:
        raise RuntimeError("Performance measurements used different pinned CPUs")
    if len(settings) != 1:
        raise RuntimeError("Performance measurements used different numerical settings")
    if pair_ids["baseline"] != pair_ids["candidate"]:
        raise RuntimeError("Baseline and candidate repetitions were not collected together")
    result = runtime_statistics(groups["baseline"], groups["candidate"],
                                config["max_runtime_ratio"], config["seed"])
    dump(REPORT / "data" / f"performance-{case['id']}.json", result)
    if not result["passed"]:
        raise RuntimeError(f"{case['id']}: performance gate {result['status']}; "
                           f"95% ratio interval {result['ratio_confidence_interval_95']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("generate", "build", "measure", "compare",
                                          "performance"))
    parser.add_argument("--source", choices=("baseline", "candidate", "both"), default="candidate")
    parser.add_argument("--cases", default="all")
    parser.add_argument("--config", type=Path, default=REPORT / "cases.json")
    parser.add_argument("--backends")
    parser.add_argument("--cpu", type=int, default=min(os.sched_getaffinity(0)))
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--start-repetition", type=int, default=0)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    cases = [c for c in config["cases"]
             if args.cases == "all" or c["id"] in args.cases.split(",")]
    if not cases:
        parser.error("No selected cases")
    for case in cases:
        sources = ("baseline", "candidate") if args.source == "both" else (args.source,)
        print(args.action, args.source, case["id"], flush=True)
        if args.action == "generate":
            for source in sources:
                generate(source, case)
        elif args.action == "build":
            for source in sources:
                masses = build(source, case)
                make_points(case, masses, config)
        elif args.action == "measure":
            measurement_run_id = datetime.datetime.now(datetime.timezone.utc).isoformat()
            backends = [int(b) for b in (args.backends or
                        ("1,6,7" if args.source == "baseline" else "1,6,7,8")).split(",")]
            if any(b not in (1, 6, 7, 8) or (args.source == "baseline" and b == 8)
                   for b in backends):
                parser.error("Invalid backend for this source")
            targets = [(s, b) for s in sources for b in backends
                       if not (s == "baseline" and b == 8)]
            gate_pair = [("baseline", 6), ("candidate", 8)]
            if all(target in targets for target in gate_pair):
                targets = gate_pair + [target for target in targets if target not in gate_pair]
            loops = {}
            points = REPORT / "points" / f"{case['id']}.dat"
            repetitions = args.repetitions or config["repetitions"]
            for source, backend in targets:
                result = measure(source, case, backend, points, 1,
                                 "calibration", args.cpu, config=config)
                loops[source, backend] = max(1, math.ceil(1/max(result["seconds"], 1e-6)))
            for rep in range(args.start_repetition, args.start_repetition+repetitions):
                for source, backend in targets if rep % 2 == 0 else reversed(targets):
                    pending = {"source": source, "case": case["id"], "backend": backend,
                               "repetition": rep}
                    old_path = measurement_path(pending)
                    if old_path.exists():
                        quarantine_measurement(json.loads(old_path.read_text()),
                                               "Superseded by a new measurement attempt")
                    try:
                        result = measure(source, case, backend, points,
                                         loops[source, backend], rep, args.cpu,
                                         config=config, persist=False)
                    except Exception as error:
                        directory = subprocess_dir(source, case)
                        pending["raw_outputs"] = {
                            name: (directory / name).read_text()
                            for name in ("timing.dat", "values.dat", "timed_status.dat",
                                         "dimensions.dat") if (directory / name).exists()}
                        quarantine_measurement(pending, error)
                        raise
                    result["paired_baseline_candidate"] = args.source == "both"
                    result["measurement_run_id"] = measurement_run_id
                    accept_repetition(result, config)
                    print(source, backend, rep, result["seconds_per_point"], flush=True)
        elif args.action == "compare":
            compare(case, config)
            if args.source == "both":
                compare_legacy(case, config)
        else:
            compare_performance(case, config)


if __name__ == "__main__":
    main()
