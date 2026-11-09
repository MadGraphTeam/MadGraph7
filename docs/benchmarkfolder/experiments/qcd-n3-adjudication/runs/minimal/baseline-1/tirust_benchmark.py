#!/usr/bin/env python3
"""Frozen-input MadLoop CLI campaign; run after sourcing tools/tirust_env.sh.

Commands deliberately separate preparation, numerical checks and acceptance
timings. A failed child or missing observation is retained and never imputed.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs/benchmarkfolder"
WORK = ROOT / ".benchmark-work/tirust"
BASELINE_REVISION = "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f"
BASELINE = WORK / "baseline"
GENERATOR_PATHS = ("madgraph", "aloha", "Template", "bin", "models", "input", "vendor")
sys.path.insert(0, str(ROOT))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_config():
    config = json.loads((REPORT / "cases.json").read_text())
    if config["baseline_revision"] != BASELINE_REVISION:
        raise ValueError("Benchmark baseline revision is immutable")
    if config["points_per_case"] != 100 or config["repetitions"] < 7:
        raise ValueError("Acceptance requires 100 points and at least seven pairs")
    if len({case["id"] for case in config["cases"]}) != len(config["cases"]):
        raise ValueError("Duplicate benchmark case")
    return config


def environment():
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               RAYON_NUM_THREADS="1", XDG_CONFIG_HOME=str(WORK / "config"))
    libraries = [ROOT / "HEPTools/ninja/lib", ROOT / "HEPTools/collier",
                 ROOT / "HEPTools/ninjars/lib", ROOT / "HEPTools/tirust/lib"]
    env["LD_LIBRARY_PATH"] = os.pathsep.join(map(str, libraries)) + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env["MAKEFLAGS"] = "SHELL=" + (shutil.which("bash") or "/bin/bash")
    return env


def run(command, cwd, log, timeout=86400):
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    env = environment()
    # Legacy dependency makefiles use the inherited shell PWD rather than
    # make's CURDIR. Keep it consistent with the actual subprocess directory.
    env["PWD"] = str(Path(cwd).resolve())
    with log.open("wb") as stream:
        result = subprocess.run(command, cwd=cwd, env=env, stdout=stream,
                                stderr=subprocess.STDOUT, timeout=timeout)
    seconds = time.perf_counter() - start
    if result.returncode:
        raise RuntimeError(f"Command exited {result.returncode}; raw log: {log}")
    return seconds


def tree_files(directory):
    """Source inventory excluding import caches and downloaded archive scratch."""
    return {str(path.relative_to(directory)): digest(path)
            for path in sorted(directory.rglob("*")) if path.is_file()
            and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pkl", ".log")}


def source_identity(tree):
    if tree == BASELINE:
        record = json.loads((WORK / "baseline.json").read_text())
        # Check frozen tracked source bytes before every generation.
        changed = [name for name, checksum in record["files"].items()
                   if not (tree / name).is_file() or digest(tree / name) != checksum]
        if changed:
            raise RuntimeError(f"Frozen baseline changed: {changed[:8]}")
        return {"revision": BASELINE_REVISION, "files_sha256": content_digest(record["files"]),
                "archive_sha256": record["archive_sha256"], "dirty": False}
    names = subprocess.check_output(["git", "ls-files", "-z", "--", *GENERATOR_PATHS],
                                    cwd=ROOT).decode().split("\0")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard",
                                         "-z", "--", *GENERATOR_PATHS], cwd=ROOT).decode().split("\0")
    files = {name: digest(ROOT / name) if (ROOT / name).is_file() else None
             for name in sorted(set(names + untracked)) if name}
    patch = subprocess.check_output(["git", "diff", "--binary", "HEAD", "--", *GENERATOR_PATHS], cwd=ROOT)
    result = {"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                  text=True).strip(),
              "files_sha256": content_digest(files), "files": files,
              "dirty": bool(patch or any(untracked)),
              "dirty_patch_sha256": hashlib.sha256(patch).hexdigest()}
    identity = content_digest(result)
    target = REPORT / "inputs/source-identities" / identity
    dump(target.with_suffix(".json"), result)
    if patch:
        target.with_suffix(".patch").write_bytes(patch)
    for name in filter(None, untracked):
        if (ROOT / name).is_file():
            target.mkdir(exist_ok=True)
            (target / digest(ROOT / name)).write_bytes((ROOT / name).read_bytes())
    return result


def prepare_baseline():
    if BASELINE.exists():
        source_identity(BASELINE)
        prepare_baseline_dependencies()
        return
    WORK.mkdir(parents=True, exist_ok=True)
    archive = subprocess.check_output(["git", "archive", "--format=tar", BASELINE_REVISION], cwd=ROOT)
    stage = WORK / "baseline.stage"
    stage.mkdir(exist_ok=False)
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as handle:
            handle.extractall(stage, filter="data")
        files = tree_files(stage)
        stage.rename(BASELINE)
        record = {"revision": BASELINE_REVISION, "archive_sha256": hashlib.sha256(archive).hexdigest(),
                  "created_utc": utc_now(), "files": files,
                  "default_MadLoopParams": (BASELINE / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat").read_text()}
        dump(WORK / "baseline.json", record)
        dump(REPORT / "inputs/baseline.json", record)
        prepare_baseline_dependencies()
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def prepare_baseline_dependencies():
    """Share only ignored prebuilt CutTools products, never baseline sources."""
    source = ROOT / "vendor/CutTools/includects"
    target = BASELINE / "vendor/CutTools/includects"
    if source.is_dir() and not target.exists():
        target.symlink_to(source, target_is_directory=True)
    iregi = BASELINE / "vendor/IREGI/src"
    if shutil.which("gfortran") and not (iregi / "libiregi.a").exists():
        # The outer IREGI makefile assumes PWD is an MG installation root.
        # Build the frozen tree directly so a baseline export cannot compile
        # or clean the candidate's ignored dependency objects.
        run(["make", "-f", "makefile_ML5_lib"], iregi,
            WORK / "logs" / "prepare-baseline-iregi.log")


def acquire_model(source=None, url=None):
    """Freeze downloaded model bytes; later runs refuse silent model changes."""
    name = "loop_qcd_qed_sm_Gmu"
    destination = WORK / "models" / name
    pin = REPORT / "inputs" / (name + ".json")
    expected = json.loads(pin.read_text()) if pin.exists() else None
    if destination.exists():
        if expected is None:
            raise RuntimeError("Existing UFO has no retained pin; preserve it and retry in a clean model directory")
        if tree_files(destination) != expected["files"]:
            raise RuntimeError("Pinned electroweak UFO changed")
        return destination
    archive_hash = None
    database_url = "https://madgraph.mi.infn.it/models_db.dat"
    if source is not None:
        source = Path(source).resolve()
        if not (source / "particles.py").is_file():
            raise ValueError("--model-source must contain a UFO model")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns("*.pkl", "*.pyc", "__pycache__"))
        origin = str(source)
    else:
        if url is None:
            with urllib.request.urlopen(database_url, timeout=60) as response:
                database = response.read().decode()
            matches = [line.split()[1] for line in database.splitlines()
                       if len(line.split()) == 2 and line.split()[0] == name]
            if len(matches) != 1:
                raise RuntimeError("Model database has no unique electroweak model URL")
            url = matches[0]
            if url.startswith("http://"):
                url = "https://" + url[len("http://"):]
        with urllib.request.urlopen(url, timeout=120) as response:
            archive = response.read()
        archive_hash = hashlib.sha256(archive).hexdigest()
        origin = url
        stage = WORK / "models/download.stage"
        stage.mkdir(parents=True, exist_ok=False)
        try:
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:*") as handle:
                handle.extractall(stage, filter="data")
            candidates = [p.parent for p in stage.rglob("particles.py")]
            if len(candidates) != 1:
                raise RuntimeError("Downloaded archive must contain exactly one UFO")
            shutil.copytree(candidates[0], destination)
            (destination.parent / (archive_hash + ".tar.gz")).write_bytes(archive)
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    # Match the CLI importer conversion before sealing model bytes.
    from madgraph.models.import_ufo import convert_model_py2_to_py3
    convert_model_py2_to_py3(str(destination))
    record = {"name": name, "origin": origin, "database_url": database_url,
              "archive_sha256": archive_hash, "files": tree_files(destination), "created_utc": utc_now()}
    record["tree_sha256"] = content_digest(record["files"])
    if expected is not None and (record["files"] != expected["files"] or
                                record["archive_sha256"] != expected["archive_sha256"]):
        raise RuntimeError("Downloaded UFO differs from the retained model pin")
    if expected is not None:
        return destination
    dump(pin, record)
    return destination


def output_dir(source, case):
    return WORK / "outputs" / source / case["id"]


def subprocess_dir(source, case):
    directories = sorted(output_dir(source, case).glob("SubProcesses/P*_*"))
    if len(directories) != 1:
        raise RuntimeError(f"Expected one subprocess; found {directories}")
    return directories[0]


def cli_lines(tree, source, case, output):
    model = case["model"] if case["model"] == "loop_sm" else str(acquire_model())
    lines = ["set automatic_html_opening False", "set auto_update 0", "set nb_core 4",
             "set loop_optimized_output True", "set output_dependencies external",
             f"set complex_mass_scheme {bool(case.get('complex_mass_scheme', False))}"]
    for name, suffix in (("ninja", "ninja/lib"), ("collier", "collier"),
                         ("ninjars", "ninjars/lib"), ("tirust", "tirust/lib")):
        path = ROOT / "HEPTools" / suffix
        if path.is_dir() and (source == "candidate" or name != "tirust"):
            lines.append(f"set {name} {path}")
    # The frozen CLI reserves noborn for integration outputs; sqrvirt exposes
    # the same loop-induced matrix element through standalone MadLoop.
    process = case['process'].replace('[noborn=QCD]', '[sqrvirt=QCD]')
    lines.extend((f"import model {model}", f"generate {process}",
                  f"output standalone {output}", "quit"))
    return lines


def generate(source, case):
    tree = BASELINE if source == "baseline" else ROOT
    output = output_dir(source, case)
    if (output / "generation.json").exists():
        return json.loads((output / "generation.json").read_text())
    if output.exists():
        raise RuntimeError(f"Incomplete generation exists; retain and move aside before retry: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    identity = source_identity(tree)
    command_file = REPORT / "commands" / f"{source}-{case['id']}.mg5"
    command_file.parent.mkdir(parents=True, exist_ok=True)
    command_file.write_text("\n".join(cli_lines(tree, source, case, output)) + "\n")
    log = WORK / "logs" / f"generate-{source}-{case['id']}.log"
    seconds = run([sys.executable, str(tree / "bin/madgraph"), "--plain", str(command_file)], ROOT, log)
    if ("Error detected" in log.read_text() or "interrupted in sub-command" in log.read_text()
            or not (output / "Cards/param_card.dat").is_file()):
        raise RuntimeError(f"CLI generation failed: {log}")
    if source == "candidate" and source_identity(tree)["files_sha256"] != identity["files_sha256"]:
        raise RuntimeError("Generator source changed during export; retain and regenerate")
    from madgraph.models.check_param_card import ParamCard
    param_path = output / "Cards/param_card.dat"
    param = ParamCard(str(param_path))
    if not case.get("complex_mass_scheme", False):
        for decay in param["decay"]:
            decay.value = 0.0
    param.write(str(param_path))
    from madgraph.various.banner import MadLoopParam
    default_chain = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))["MLReductionLib"]
    record = {"case": case, "source": source, "seconds": seconds, "source_identity": identity,
              "created_utc": utc_now(), "command_sha256": digest(command_file),
              "default_chain": str(default_chain), "parameter_card_sha256": digest(param_path)}
    dump(output / "generation.json", record)
    dump(REPORT / "inputs" / source / case["id"] / "generation.json", record)
    shutil.copy2(param_path, REPORT / "inputs" / source / case["id"] / "param_card.dat")
    return record


def generated_sources(output):
    suffixes = {".f", ".f90", ".for", ".inc", ".c", ".cc", ".h"}
    return {str(p.relative_to(output)): digest(p)
            for subtree in ("Source", "SubProcesses") for p in sorted((output / subtree).rglob("*"))
            if p.is_file() and (p.suffix.lower() in suffixes or p.name.startswith("makefile")
                               or p.name in ("make_opts", "MadLoop_makefile_definitions"))}


def render_worker(template, parameters):
    return (template.replace("@PREFIX@", parameters["prefix"])
            .replace("@NEXTERNAL@", str(parameters["nexternal"]))
            .replace("@NBORN@", str(parameters["nborn"])))


def verify_worker_template(compilation, template_path):
    if digest(template_path) != compilation.get("worker_template_sha256"):
        raise RuntimeError("Worker template differs from compilation")
    rendered = render_worker(Path(template_path).read_text(), compilation["worker_render_parameters"])
    if hashlib.sha256(rendered.encode()).hexdigest() != compilation["driver_sha256"]:
        raise RuntimeError("Compiled driver differs from the retained worker template")
    drivers = [checksum for name, checksum in compilation["sources"].items()
               if re.fullmatch(r"SubProcesses/P[^/]+/check_sa\.f", name)]
    if drivers != [compilation["driver_sha256"]]:
        raise RuntimeError("Worker driver differs from its generated-source inventory")


def same_compiler_policy(left, right):
    from tirust_benchmark_compiler import validate_compiler_equality
    try:
        validate_compiler_equality(left or {}, right or {})
    except ValueError:
        return False
    return all(set(left["scopes"][scope]["recipes"]) == set(right["scopes"][scope]["recipes"])
               for scope in left["scopes"])


def build(source, case, jobs=4):
    from tirust_benchmark_compiler import capture_effective_flags
    output = output_dir(source, case)
    directory = subprocess_dir(source, case)
    compilation_path = output / "compilation.json"
    template_path = ROOT / "tools/tirust_benchmark_worker.f"
    previous = None
    if compilation_path.exists():
        previous = json.loads(compilation_path.read_text())
        if digest(directory / "check") != previous["executable_sha256"]:
            raise RuntimeError("Compiled executable changed")
        if generated_sources(output) != previous["sources"]:
            raise RuntimeError("Generated sources changed; regenerate before building")
        current_dependencies = dependency_inputs(directory)
        current_flags = capture_effective_flags(directory, env=environment())
        if (previous.get("link_inputs_before") == current_dependencies
                and previous.get("link_inputs_after") == current_dependencies
                and same_compiler_policy(previous.get("effective_compiler_flags"), current_flags)
                and previous.get("numerical_build_from_sources") is True
                and previous.get("worker_template_sha256") == digest(template_path)):
            verify_worker_template(previous, template_path)
            return previous["masses"]
        if (previous.get("effective_compiler_flags") is not None
                and not same_compiler_policy(previous["effective_compiler_flags"], current_flags)):
            raise RuntimeError("Effective compiler flags changed; regenerate all objects before building")
        # A legacy record or changed archive requires a new link. Preserve the
        # old executable and evidence rather than assigning new libraries to it.
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        history = output / "build-history" / stamp
        history.mkdir(parents=True)
        shutil.copy2(directory / "check", history / "check")
        shutil.copy2(compilation_path, history / "compilation.json")
        dump(REPORT / "inputs" / source / case["id"] / "build-history" / stamp / "compilation.json", previous)
        old_log = WORK / "logs" / f"compile-{source}-{case['id']}.log"
        if old_log.exists():
            shutil.copy2(old_log, history / "compile.log")
        (directory / "check").unlink()
    driver_path = directory / "check_sa.f"
    original = driver_path.read_text()
    prefix = re.search(r"CALL\s+(\w*)GET_ANSWER_DIMENSION", original, re.I).group(1)
    born = directory / "nsqso_born.inc"
    nborn = int(re.search(r"NSQSO_BORN\s*=\s*(\d+)", born.read_text(), re.I).group(1)) if born.exists() else 0
    render_parameters = {"prefix": prefix, "nexternal": case["nfinal"] + 2, "nborn": nborn}
    driver_path.write_text(render_worker(template_path.read_text(), render_parameters))
    before = generated_sources(output)
    flags_before = capture_effective_flags(directory, env=environment())
    attempt = WORK / "build-attempts" / source / case["id"] / datetime.datetime.now(
        datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dump(attempt / "compiler-before.json", flags_before)
    full_rebuild = not previous or previous.get("numerical_build_from_sources") is not True
    external_before = dependency_inputs(directory)
    # MODEL/DHELAS archives are outputs of this source build, not external
    # dependencies. All other archive and metadata bytes remain frozen.
    internal = {str((output / "lib" / name).resolve()) for name in ("libmodel.a", "libdhelas.a")}
    external_before["archives"] = {key: value for key, value in external_before["archives"].items()
                                   if key not in internal}
    seconds = 0.
    build_logs = {}
    if full_rebuild:
        for scope in ("MODEL", "DHELAS"):
            log = WORK / "logs" / f"compile-{source}-{case['id']}-{scope}.log"
            seconds += run(["make", "--always-make", f"-j{jobs}"], output / "Source" / scope, log)
            build_logs[scope] = digest(log)
    link_before = dependency_inputs(directory)
    external_after = {"metadata": link_before["metadata"],
                      "archives": {key: value for key, value in link_before["archives"].items() if key not in internal}}
    if external_before != external_after:
        raise RuntimeError("External numerical libraries changed during source build")
    command = ["make", *(["--always-make"] if full_rebuild else []), f"-j{jobs}", "check"]
    compile_log = WORK / "logs" / f"compile-{source}-{case['id']}.log"
    seconds += run(command, directory, compile_log)
    build_logs["process"] = digest(compile_log)
    if generated_sources(output) != before:
        raise RuntimeError("Sources changed during build")
    link_after = dependency_inputs(directory)
    if link_before != link_after:
        raise RuntimeError("Static dependencies or provenance changed during link")
    flags_after = capture_effective_flags(directory, env=environment())
    dump(attempt / "compiler-after.json", flags_after)
    if not same_compiler_policy(flags_before, flags_after):
        raise RuntimeError("Effective compiler flags changed during build")
    run(["./check", "--masses"], directory, WORK / "logs" / f"masses-{source}-{case['id']}.log")
    masses = [float(x) for x in (directory / "masses.dat").read_text().split()]
    if len(masses) != case["nfinal"] + 2:
        raise RuntimeError("Driver returned wrong external mass count")
    record = {"seconds": seconds, "masses": masses, "driver_sha256": digest(driver_path),
              "worker_template_sha256": digest(template_path), "worker_render_parameters": render_parameters,
              "executable_sha256": digest(directory / "check"), "sources": before,
              "settings": {str(p.relative_to(output)): p.read_text() for p in (
                  output / "Source/make_opts", output / "SubProcesses/MadLoop_makefile_definitions")},
              "link_inputs_before": link_before, "link_inputs_after": link_after,
              "effective_compiler_flags": flags_before, "created_utc": utc_now(),
              "numerical_build_from_sources": True, "full_numerical_rebuild_this_invocation": full_rebuild,
              "build_logs": build_logs, "process_build_command": command,
              "previous_compilation_sha256": content_digest(previous) if previous else None,
              "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]}
    dump(compilation_path, record)
    evidence = REPORT / "inputs" / source / case["id"]
    dump(evidence / "compilation.json", record)
    for scope in build_logs:
        log = (compile_log if scope == "process" else WORK / "logs" / f"compile-{source}-{case['id']}-{scope}.log")
        shutil.copy2(log, evidence / f"compilation-{scope}.log")
    return masses


def freeze_points(case, masses, config):
    from tirust_benchmark_points import generate_points, generate_stress_points, write_points
    path = REPORT / "points" / (case["id"] + ".dat")
    metadata_path = path.with_suffix(".json")
    seed = config["seed"] + next(i for i, item in enumerate(config["cases"]) if item["id"] == case["id"])
    inputs = {"masses": masses[2:], "incoming_masses": masses[:2], "seed": seed,
              "count": config["points_per_case"], "energy": config["energy_gev"]}
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        if metadata["inputs"] != inputs or metadata["sha256"] != digest(path):
            raise RuntimeError("Frozen phase-space inputs changed")
        return path
    points = generate_points(**inputs)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_points(path, points)
    dump(metadata_path, {"inputs": inputs, "sha256": digest(path), "points": points})
    stress = generate_stress_points(masses[2:], incoming_masses=masses[:2], energy=config["energy_gev"], seed=seed)
    if stress:
        stress_path = REPORT / "points" / (case["id"] + "-stress.dat")
        write_points(stress_path, stress)
        dump(stress_path.with_suffix(".json"), {"inputs": inputs, "sha256": digest(stress_path), "points": stress})
    return path


def parse_chain(value):
    chain = [int(x) for x in str(value).replace(",", "|").split("|") if x.strip()]
    if not chain or len(chain) != len(set(chain)) or any(x not in range(1, 10) for x in chain):
        raise ValueError("Invalid reduction chain")
    return chain


def configure(source, case, chain, config):
    from madgraph.various.banner import MadLoopParam
    output = output_dir(source, case)
    path = output / "Cards/MadLoopParams.dat"
    card = MadLoopParam(str(path))
    card["MLReductionLib"] = "|".join(map(str, chain))
    options = dict(config["madloop_parameters"])
    if source == "candidate":
        options.update(config["candidate_parameters"])
    for name, value in options.items():
        if name.lower() not in {str(key).lower() for key in card}:
            raise RuntimeError(f"Required parameter is unavailable: {name}")
        card[name] = value
    tree = BASELINE if source == "baseline" else ROOT
    template = str(tree / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat")
    card.write(str(path), template=template)
    for destination in (output / "SubProcesses").rglob("MadLoopParams.dat"):
        card.write(str(destination), template=template)
    for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
        for destination in (output / "SubProcesses").rglob(pattern):
            destination.unlink()
    return {str(key).lower(): value for key, value in card.items()}


def compare_rows(reference, candidate, tolerance=3e-6):
    """LoopMEComparator criterion, mapped to native ANS(0:3) column order.

ANS columns are Born, finite, simple pole and double pole. The comparator uses
abs(a-b)/abs(a+b), and permits poles small compared with the finite coefficient.
Opposite nonzero inputs have an undefined relative ratio and fail unless the
pole-size exception applies. Retain undefined ratios as null, never JSON inf.
    """
    if [(r["point"], r["order"]) for r in reference] != [(r["point"], r["order"]) for r in candidate]:
        raise RuntimeError("Numerical rows do not describe identical points/orders")
    failures, residuals, worst = [], [], 0.0
    target_passed = True
    for left, right in zip(reference, candidate):
        finite_scale = max(abs(left["coefficients"][1]), abs(right["coefficients"][1]), 1e-99)
        for column, (a, b) in enumerate(zip(left["coefficients"], right["coefficients"])):
            magnitude = max(abs(a), abs(b))
            symmetric = abs(a - b) / abs(a + b) if a + b else 0.0 if not magnitude else None
            raw_relative = abs(a - b) / magnitude if magnitude else 0.0
            pole_fraction = magnitude / finite_scale if column >= 2 else None
            def accepted(threshold):
                return ((symmetric is not None and symmetric < threshold)
                        or (pole_fraction is not None and pole_fraction < threshold))
            target_passed = target_passed and accepted(1e-8)
            residual = {"point": left["point"], "order": left["order"], "column": column,
                        "reference": a, "candidate": b, "relative_error": raw_relative,
                        "symmetric_relative_error": symmetric, "pole_to_finite": pole_fraction,
                        "passed": accepted(tolerance)}
            residuals.append(residual)
            worst = max(worst, raw_relative)
            if not residual["passed"]:
                failures.append(residual)
    return {"passed": not failures, "maximum_relative_error": worst,
            "criterion": "LoopMEComparator: abs(a-b)/abs(a+b)<tolerance, or pole magnitude/finite maximum<tolerance",
            "relative_error_metric": "abs(a-b)/max(abs(a),abs(b)); zero-zero is zero",
            "ordinary_target_tolerance": 1e-8, "ordinary_target_passed": target_passed,
            "tolerance": tolerance, "failures": failures, "residuals": residuals}


def dependency_inputs(directory):
    """Identify static link inputs and the metadata that selects their ABI."""
    library_root = directory.parents[1] / "lib"
    containers = [library_root, *(path for path in library_root.glob("*") if path.is_dir())]
    archives, metadata = {}, {}
    for container in containers:
        for path in sorted(container.glob("*")):
            if not path.is_file():
                continue
            if path.suffix == ".a":
                archives[str(path.resolve())] = digest(path)
            elif (path.name.endswith(("-config.json", "-native-libs.txt", "-link-before.txt",
                                      "-dependency-provenance.json", "-dependency-notices.tar.gz", "-NOTICES.txt"))):
                metadata[str(path.resolve())] = digest(path)
    # NinjaRS supplies a compiler-resolved libgcc.a before its Rust archive.
    # It lives outside output/lib and is numerically significant for QP.
    link_texts = [Path(name).read_text() for name in metadata
                  if name.endswith(("-link-before.txt", "-native-libs.txt", "-config.json"))]
    for path in (directory / "makefile", directory.parent / "MadLoop_makefile_definitions"):
        if path.is_file():
            link_texts.append(path.read_text())
    for content in link_texts:
        for name in re.findall(r'(/[^\s\"\'<>]+\.a)(?=[\s\"\'<>]|$)', content):
            if Path(name).is_file():
                archives[str(Path(name).resolve())] = digest(name)
    return {"archives": archives, "metadata": metadata}


def runtime_identity(directory, executable="check"):
    command = ["ldd", str(directory / executable)]
    text = subprocess.check_output(command, env=environment(), text=True, stderr=subprocess.STDOUT)
    paths = re.findall(r"(?:=>\s*)?(/[^\s]+)", text)
    return {"ldd": text, "libraries": {name: digest(name) for name in paths if Path(name).is_file()},
            **dependency_inputs(directory)}


def machine_identity(cpu):
    cpuinfo = Path("/proc/cpuinfo")
    model = None
    if cpuinfo.exists():
        model = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                      if line.startswith("model name")), None)
    governor = Path(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_governor")
    return {"uname": dict(platform.uname()._asdict()), "cpu_model": model,
            "selected_cpu": cpu, "allowed_affinity": sorted(os.sched_getaffinity(0)),
            "governor": governor.read_text().strip() if governor.exists() else None,
            "thread_environment": {key: environment()[key] for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS")}}


def host_load_snapshot():
    """Record host interference without inspecting unrelated command lines."""
    allowed = set(os.sched_getaffinity(0))
    counters = {}
    for line in Path("/proc/stat").read_text().splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"cpu\d+", fields[0]):
            cpu = int(fields[0][3:])
            if cpu in allowed:
                values = [int(value) for value in fields[1:]]
                # guest/guest_nice are already part of user/nice.
                counters[str(cpu)] = {"total_ticks": sum(values[:8]),
                    "idle_ticks": values[3] + values[4], "iowait_ticks": values[4]}
    load = Path("/proc/loadavg").read_text().split()
    runnable, total = map(int, load[3].split("/"))
    return {"utc": utc_now(), "monotonic_seconds": time.monotonic(), "cpus": counters,
            "load_averages": [float(value) for value in load[:3]],
            "runnable_tasks": runnable, "total_tasks": total}


def select_quiet_cpu():
    before = host_load_snapshot()
    time.sleep(.25)
    after = host_load_snapshot()
    def load(cpu):
        left, right = before["cpus"][str(cpu)], after["cpus"][str(cpu)]
        total = right["total_ticks"] - left["total_ticks"]
        idle = right["idle_ticks"] - left["idle_ticks"]
        return (1. - idle / total if total else 1., cpu)
    return min(os.sched_getaffinity(0), key=load)


def retain_observation(directory, destination):
    destination.mkdir(parents=True, exist_ok=False)
    for name in ("timing.dat", "values.dat", "timed_status.dat", "dimensions.dat"):
        path = directory / name
        if path.exists():
            shutil.copy2(path, destination / name)


def start_worker(stack, source, case, chain, config, cpu, report, timeout=3600., points_path=None):
    from tirust_benchmark_worker import Worker
    from tirust_benchmark_compiler import capture_effective_flags
    directory = subprocess_dir(source, case)
    compilation = json.loads((output_dir(source, case) / "compilation.json").read_text())
    if digest(directory / "check") != compilation["executable_sha256"]:
        raise RuntimeError("Executable no longer matches compilation")
    if generated_sources(output_dir(source, case)) != compilation["sources"]:
        raise RuntimeError("Generated source no longer matches compilation")
    if compilation.get("numerical_build_from_sources") is not True:
        raise RuntimeError("Numerical objects lack source-build provenance; rebuild the worker")
    verify_worker_template(compilation, ROOT / "tools/tirust_benchmark_worker.f")
    dependencies = dependency_inputs(directory)
    if (compilation.get("link_inputs_before") != dependencies
            or compilation.get("link_inputs_after") != dependencies):
        raise RuntimeError("Dependencies no longer match link-time evidence; rebuild the worker")
    if not same_compiler_policy(capture_effective_flags(directory, env=environment()),
                                compilation.get("effective_compiler_flags")):
        raise RuntimeError("Compiler flags no longer match compilation")
    generation = json.loads((output_dir(source, case) / "generation.json").read_text())
    if generation["case"] != case:
        raise RuntimeError("Generated process differs from the requested case")
    parameter_card = output_dir(source, case) / "Cards/param_card.dat"
    parameter_hash = digest(parameter_card)
    if parameter_hash != generation["parameter_card_sha256"]:
        raise RuntimeError("Physical parameter card differs from the frozen generation")
    # SETPARA's first LHA_open_file call tries a local card before Cards/ in
    # this directory and five parents. A stale local card must not shadow the
    # pinned canonical card recorded below.
    candidates = [directory / "param_card.dat"] + [
        directory.joinpath(*([".."] * level), "Cards/param_card.dat") for level in range(6)]
    actual_card = next((path for path in candidates if path.is_file()), None)
    if actual_card is None or digest(actual_card) != parameter_hash:
        raise RuntimeError("Fortran resolves a parameter card different from the frozen card")
    settings = configure(source, case, chain, config)
    points = Path(points_path).resolve() if points_path is not None else REPORT / "points" / (case["id"] + ".dat")
    metadata = json.loads(points.with_suffix(".json").read_text())
    if digest(points) != metadata["sha256"]:
        raise RuntimeError("Saved points changed")
    full_masses = metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"]
    if full_masses != compilation["masses"]:
        raise RuntimeError("Saved point masses differ from the compiled process/card")
    if metadata["inputs"]["count"] != config["points_per_case"]:
        raise RuntimeError("Saved point count differs from the campaign configuration")
    dump(report / (source + "-inputs.json"), {"settings": settings, "points_sha256": digest(points),
         "param_card_sha256": parameter_hash, "resolved_parameter_card": str(actual_card.resolve()),
         "runtime": runtime_identity(directory), "compilation": compilation,
         "generation": generation})
    command = ["taskset", "-c", str(cpu), "./check", str(points)]
    worker = Worker(command, directory, report / (source + "-output.txt"), env=environment(), timeout=timeout)
    stack.callback(worker.close)
    return worker, settings


def observe(worker, chain, loops, destination, count=100):
    from tirust_benchmark_worker import parse_outputs
    try:
        worker.run(loops)
    finally:
        retain_observation(worker.directory, destination)
    result = parse_outputs(destination, chain, count, loops)
    dump(destination / "observation.json", result)
    return result


def pair(case, baseline_chain, candidate_chain, config, cpu, label, attempt_name=None):
    from tirust_benchmark_worker import paired_statistics
    from tirust_benchmark_compiler import validate_compiler_equality
    if attempt_name is not None and not re.fullmatch(re.escape(label) + r"-attempt-(?:[2-9]|[1-9][0-9]+)", attempt_name):
        raise ValueError("Invalid append-only attempt name")
    report = REPORT / "runs" / case["id"] / (attempt_name or label)
    report.mkdir(parents=True, exist_ok=False)
    protocol_sources = (Path(__file__), ROOT / "tools/tirust_benchmark_worker.py",
                        ROOT / "tools/tirust_benchmark_worker.f", ROOT / "tools/tirust_benchmark_compiler.py")
    retained_protocol = report / "protocol-sources"
    retained_protocol.mkdir()
    for path in protocol_sources:
        shutil.copy2(path, retained_protocol / path.name)
    specification = {"created_utc": utc_now(), "case": case, "cpu": cpu,
                     "logical_label": label, "attempt_directory": report.name,
                     "machine": machine_identity(cpu),
                     "host_load_before": host_load_snapshot(),
                     "host_exclusivity": "Not assumed; allowed-CPU counters and host load retained for each batch",
                     "calibration_target_seconds": 2. * config["minimum_batch_seconds"],
                     "baseline_chain": baseline_chain, "candidate_chain": candidate_chain,
                     "config_sha256": digest(REPORT / "cases.json"), "status": "running",
                     "protocol_sources": {p.name: digest(retained_protocol / p.name)
                                          for p in protocol_sources}}
    dump(report / "specification.json", specification)
    try:
        generation = json.loads((output_dir("baseline", case) / "generation.json").read_text())
        if label == "tirust-first":
            expected = parse_chain(generation["default_chain"])
            if baseline_chain != expected or candidate_chain != [9, *expected]:
                raise RuntimeError("tirust-first requires the frozen default chain and TIRust-first rescue chain")
        elif label in {f"backend-{value}" for value in (1, 6, 7, 8, 9)}:
            backend = int(label.split("-")[-1])
            expected_baseline = [1] if backend == 9 else [backend]
            if baseline_chain != expected_baseline or candidate_chain != [backend]:
                raise RuntimeError("Reserved backend label differs from the requested reduction chains")
        with ExitStack() as stack:
            if {8, 9} & set(baseline_chain + candidate_chain):
                lock_path = ROOT / ".tirust-build/numerical-tests.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(lock_path.open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX)
            workers, settings = {}, {}
            for source, chain in (("baseline", baseline_chain), ("candidate", candidate_chain)):
                workers[source], settings[source] = start_worker(stack, source, case, chain, config, cpu, report)
            common = set(settings["baseline"]) & set(settings["candidate"]) - {"mlreductionlib"}
            if any(settings["baseline"][key] != settings["candidate"][key] for key in common):
                raise RuntimeError("Baseline/candidate numerical parameters differ")
            cards = [digest(output_dir(source, case) / "Cards/param_card.dat") for source in workers]
            if cards[0] != cards[1]:
                raise RuntimeError("Baseline/candidate physical parameter cards differ")
            inputs = {source: json.loads((report / (source + "-inputs.json")).read_text())
                      for source in workers}
            if inputs["baseline"]["points_sha256"] != inputs["candidate"]["points_sha256"]:
                raise RuntimeError("Baseline/candidate workers read different saved points")
            if inputs["baseline"]["compilation"]["compiler"] != inputs["candidate"]["compilation"]["compiler"]:
                raise RuntimeError("Baseline/candidate Fortran compilers differ")
            validate_compiler_equality(inputs["baseline"]["compilation"]["effective_compiler_flags"],
                                       inputs["candidate"]["compilation"]["effective_compiler_flags"])
            libraries = {source: {Path(path).name: checksum for path, checksum in
                                  inputs[source]["runtime"]["libraries"].items()}
                         for source in workers}
            shared_libraries = set(libraries["baseline"]) & set(libraries["candidate"])
            if any(libraries["baseline"][name] != libraries["candidate"][name]
                   for name in shared_libraries):
                raise RuntimeError("Baseline/candidate shared runtime library bytes differ")
            for kind in ("archives", "metadata"):
                left, right = inputs["baseline"]["runtime"][kind], inputs["candidate"]["runtime"][kind]
                if any(left[name] != right[name] for name in left.keys() & right.keys()):
                    raise RuntimeError("Baseline/candidate shared static dependency bytes differ")
            chains = {"baseline": baseline_chain, "candidate": candidate_chain}
            calibration, loops = {}, {}
            for source in workers:
                trial = observe(workers[source], chains[source], 1, report / "raw" / source / "calibrate-0")
                target_seconds = 2. * config["minimum_batch_seconds"]
                initial_loops = max(1, math.ceil(target_seconds / trial["seconds"]))
                # A single traversal can overestimate steady cost. Calibrate
                # once more over a full-duration batch. A factor-two target
                # tolerates substantial shared-host calibration speed drift;
                # every actual duration remains visible and must pass the gate.
                sustained = observe(workers[source], chains[source], initial_loops,
                                    report / "raw" / source / "calibrate-1")
                loops[source] = max(1, math.ceil(initial_loops * target_seconds
                                                / sustained["seconds"]))
                calibration[source] = sustained
            parity = compare_rows(calibration["baseline"]["rows"], calibration["candidate"]["rows"],
                                  config["relative_tolerance"])
            dump(report / "calibration-parity.json", parity)
            if not parity["passed"]:
                raise RuntimeError("Numerical disagreement during calibration")
            timings = {source: [] for source in workers}
            observations = {source: [] for source in workers}
            host_load = []
            all_parity = []
            for repetition in range(config["repetitions"]):
                order = ("baseline", "candidate") if repetition % 2 == 0 else ("candidate", "baseline")
                for source in order:
                    load_before = host_load_snapshot()
                    observation = observe(workers[source], chains[source], loops[source],
                        report / "raw" / source / f"repetition-{repetition}")
                    host_load.append({"source": source, "repetition": repetition,
                                      "before": load_before, "after": host_load_snapshot()})
                    timings[source].append(observation["seconds_per_point"])
                    observations[source].append(observation)
                all_parity.append(compare_rows(observations["baseline"][-1]["rows"],
                    observations["candidate"][-1]["rows"], config["relative_tolerance"]))
            stats = paired_statistics(timings["baseline"], timings["candidate"],
                threshold=config["performance_upper_ratio"], minimum_pairs=config["repetitions"])
            enough_time = all(obs["seconds"] >= .95 * config["minimum_batch_seconds"]
                              for series in observations.values() for obs in series)
            numerical_pass = all(value["passed"] for value in all_parity)
            summary = {"status": "complete", "label": label, "timings": timings, "statistics": stats,
                       "parity": all_parity, "numerical_pass": numerical_pass,
                       "minimum_batch_duration_met": enough_time,
                       "acceptance_eligible": enough_time and numerical_pass,
                       "passed": enough_time and numerical_pass and stats["passed"],
                       "host_load": host_load,
                       "raw_observations": observations}
            for source in workers:
                directory = subprocess_dir(source, case)
                current_runtime = runtime_identity(directory)
                if any(current_runtime[key] != inputs[source]["runtime"][key]
                       for key in ("libraries", "archives", "metadata")):
                    raise RuntimeError("Runtime dependencies changed during measurement")
                if digest(directory / "check") != inputs[source]["compilation"]["executable_sha256"]:
                    raise RuntimeError("Executable changed during measurement")
        # A failing STOP/BYE exchange is a failed measurement. Publish only
        # after ExitStack has successfully closed every persistent worker.
        dump(report / "summary.json", summary)
        return summary
    except BaseException as error:
        dump(report / "failure.json", {"status": "failed", "error": repr(error), "utc": utc_now()})
        raise


def aggregate_usage(observations):
    counts = {}
    for item in observations:
        for key, value in item["backend_precision_counts"].items():
            counts[key] = counts.get(key, 0) + value
    return counts


def selected_run_directory(base, label):
    selector = base / "selected-attempts.json"
    selections = json.loads(selector.read_text()) if selector.exists() else {}
    name = selections.get(label, {}).get("directory", label)
    if name != label and not re.fullmatch(re.escape(label) + r"-attempt-(?:[2-9]|[1-9][0-9]+)", name):
        raise ValueError("Invalid selected attempt directory")
    return base / name


def select_attempt(base, label, directory, reason):
    if not reason or selected_run_directory(base, label) == directory:
        raise ValueError("Retry selection requires a new append-only attempt and a reason")
    selector = base / "selected-attempts.json"
    selections = json.loads(selector.read_text()) if selector.exists() else {}
    event = {"label": label, "directory": directory.name, "reason": reason,
             "previous_directory": selected_run_directory(base, label).name, "utc": utc_now()}
    with (base / "attempt-selection-history.jsonl").open("a") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    selections[label] = event
    dump(selector, selections)


def validate_retained_result(directory, label, case, config):
    """Recompute acceptance from immutable raw files and bind every campaign input."""
    from tirust_benchmark_worker import parse_outputs, paired_statistics
    from tirust_benchmark_compiler import validate_compiler_equality
    if (directory / "failure.json").exists():
        raise ValueError("Run has a retained failure, including possible worker shutdown failure")
    result = json.loads((directory / "summary.json").read_text())
    spec = json.loads((directory / "specification.json").read_text())
    if spec["case"] != case or spec["config_sha256"] != digest(REPORT / "cases.json"):
        raise ValueError("Run case/configuration differs from this campaign")
    if result.get("status") != "complete" or result["label"] != label:
        raise ValueError("Run is incomplete or its label changed")
    for name, checksum in spec["protocol_sources"].items():
        if digest(directory / "protocol-sources" / name) != checksum:
            raise ValueError("Retained measurement protocol source changed")
    expected_protocol = {"tirust_benchmark.py", "tirust_benchmark_worker.py", "tirust_benchmark_worker.f",
                         "tirust_benchmark_compiler.py"}
    if set(spec["protocol_sources"]) != expected_protocol:
        raise ValueError("Incomplete measurement protocol source inventory")
    inputs = {source: json.loads((directory / (source + "-inputs.json")).read_text())
              for source in ("baseline", "candidate")}
    default = parse_chain(inputs["baseline"]["generation"]["default_chain"])
    expected_chains = {"tirust-first": (default, [9, *default]),
        **{f"backend-{i}": ([i] if i != 9 else [1], [i]) for i in config["backends"]}}
    if label not in expected_chains:
        raise ValueError("Unrecognized acceptance run label")
    if (spec["baseline_chain"], spec["candidate_chain"]) != expected_chains[label]:
        raise ValueError("Run label does not select the required reduction chain")
    point_path = REPORT / "points" / (case["id"] + ".dat")
    point_meta = json.loads(point_path.with_suffix(".json").read_text())
    if point_meta["sha256"] != digest(point_path) or point_meta["inputs"]["count"] != config["points_per_case"]:
        raise ValueError("Frozen point stream changed")
    anchors = {}
    actual_observations = {}
    for source in inputs:
        value = inputs[source]
        generation = value["generation"]
        if generation["case"] != case or generation["source"] != source:
            raise ValueError("Generation record belongs to a different source/process")
        if value["points_sha256"] != digest(point_path):
            raise ValueError("Measurement used a different point stream")
        if value["param_card_sha256"] != generation["parameter_card_sha256"]:
            raise ValueError("Physical card differs from the frozen generated card")
        if source == "baseline" and generation["source_identity"]["revision"] != BASELINE_REVISION:
            raise ValueError("Wrong baseline revision")
        settings = dict(value["settings"])
        if parse_chain(settings.pop("mlreductionlib")) != spec[source + "_chain"]:
            raise ValueError("Actual card backend differs from run specification")
        masses = point_meta["inputs"]["incoming_masses"] + point_meta["inputs"]["masses"]
        if masses != value["compilation"]["masses"]:
            raise ValueError("Point masses differ from the compiled physical model")
        if value["compilation"].get("numerical_build_from_sources") is not True:
            raise ValueError("Numerical objects lack source-build provenance")
        verify_worker_template(value["compilation"], directory / "protocol-sources/tirust_benchmark_worker.f")
        runtime_dependencies = {key: value["runtime"][key] for key in ("archives", "metadata")}
        if (value["compilation"].get("link_inputs_before") != runtime_dependencies
                or value["compilation"].get("link_inputs_after") != runtime_dependencies):
            raise ValueError("Runtime dependencies differ from retained link-time evidence")
        anchors[source] = {"point_sha256": value["points_sha256"],
            "param_card_sha256": value["param_card_sha256"], "settings": settings,
            "source_identity": generation["source_identity"],
            "executable_sha256": value["compilation"]["executable_sha256"],
            "compiler": value["compilation"]["compiler"], "build_settings": value["compilation"]["settings"],
            "effective_compiler_flags": value["compilation"]["effective_compiler_flags"],
            "runtime_libraries": value["runtime"]["libraries"],
            "runtime_archives": value["runtime"].get("archives", {}),
            "runtime_metadata": value["runtime"].get("metadata", {}), "cpu": spec["cpu"]}
        anchors[source]["protocol_sources"] = spec["protocol_sources"]
        saved = result["raw_observations"][source]
        if len(saved) != config["repetitions"]:
            raise ValueError("Incomplete repetition series")
        parsed = []
        for repetition, observation in enumerate(saved):
            calls = sum(observation["timed_status_counts"].values())
            loops, remainder = divmod(calls, config["points_per_case"])
            if remainder or loops < 1:
                raise ValueError("Invalid full-traversal evaluation count")
            actual = parse_outputs(directory / "raw" / source / f"repetition-{repetition}",
                spec[source + "_chain"], config["points_per_case"], loops)
            if actual != observation:
                raise ValueError("Numerical summary differs from retained raw observation")
            parsed.append(actual)
        actual_observations[source] = parsed
        if result["timings"][source] != [item["seconds_per_point"] for item in parsed]:
            raise ValueError("Timing series differs from raw measurements")
    if anchors["baseline"]["param_card_sha256"] != anchors["candidate"]["param_card_sha256"]:
        raise ValueError("Baseline/candidate physics cards differ")
    if anchors["baseline"]["compiler"] != anchors["candidate"]["compiler"]:
        raise ValueError("Baseline/candidate Fortran compilers differ")
    validate_compiler_equality(anchors["baseline"]["effective_compiler_flags"],
                               anchors["candidate"]["effective_compiler_flags"])
    common = set(anchors["baseline"]["settings"]) & set(anchors["candidate"]["settings"])
    if any(anchors["baseline"]["settings"][key] != anchors["candidate"]["settings"][key] for key in common):
        raise ValueError("Baseline/candidate shared numerical settings differ")
    parity = [compare_rows(left["rows"], right["rows"], config["relative_tolerance"])
              for left, right in zip(actual_observations["baseline"], actual_observations["candidate"])]
    statistics_result = paired_statistics(result["timings"]["baseline"], result["timings"]["candidate"],
        threshold=config["performance_upper_ratio"], minimum_pairs=config["repetitions"])
    enough_time = all(item["seconds"] >= .95 * config["minimum_batch_seconds"]
        for series in actual_observations.values() for item in series)
    numerical_pass = all(item["passed"] for item in parity)
    if (result["statistics"] != statistics_result or result["numerical_pass"] != numerical_pass
            or result["minimum_batch_duration_met"] != enough_time
            or result["acceptance_eligible"] != (enough_time and numerical_pass)
            or result["passed"] != (enough_time and numerical_pass and statistics_result["passed"])):
        raise ValueError("Run acceptance differs from recomputed raw evidence")
    return result, anchors


def summarize(config):
    """Report observed evidence only; pending/failure never becomes zero time."""
    cases = []
    candidate_revisions = set()
    candidate_source_hashes = set()
    for case in config["cases"]:
        base = REPORT / "runs" / case["id"]
        loaded = {}
        anchors = {}
        validation_errors = {}
        for label in ["tirust-first", *(f"backend-{i}" for i in config["backends"])]:
            selected = selected_run_directory(base, label)
            path = selected / "summary.json"
            if path.exists():
                try:
                    result, anchor = validate_retained_result(selected, label, case, config)
                    if anchors and anchor != next(iter(anchors.values())):
                        raise ValueError("Input/source/runtime/CPU identity differs between process run labels")
                    loaded[label], anchors[label] = result, anchor
                    identity = anchor["candidate"]["source_identity"]
                    candidate_revisions.add(identity["revision"])
                    candidate_source_hashes.add(identity["files_sha256"])
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    validation_errors[label] = str(error)
        reference = loaded.get("backend-1", {}).get("raw_observations", {}).get("candidate", [])
        correctness, timings = [], []
        for backend in config["backends"]:
            label = f"backend-{backend}"
            result = loaded.get(label)
            failed = (selected_run_directory(base, label) / "failure.json").exists() or label in validation_errors
            observations = result["raw_observations"]["candidate"] if result else []
            checks = [compare_rows(left["rows"], right["rows"], config["relative_tolerance"])
                      for left, right in zip(reference, observations)]
            complete = (len(checks) == config["repetitions"])
            correct = complete and all(check["passed"] for check in checks)
            usage = aggregate_usage(observations)
            correctness.append({"backend": backend,
                "status": "complete" if complete else "failed" if failed else "pending",
                "passed": correct if complete else None, "samples": len(checks),
                "max_relative_error": max((check["maximum_relative_error"] for check in checks), default=None),
                "ordinary_target_passed": all(check["ordinary_target_passed"] for check in checks) if complete else None,
                "failures": [failure for check in checks for failure in check["failures"]],
                "actual_backend_precision_counts": usage,
                "native_dp_only": all(item["native_dp_only"] for item in observations) if observations else None})
            seconds = [item["seconds_per_point"] for item in observations]
            median = statistics.median(seconds) if seconds else None
            mad = statistics.median(abs(x - median) for x in seconds) if seconds else None
            # Absolute per-backend timings are descriptive; paired ratio intervals
            # belong exclusively to performance below, not absolute time columns.
            timings.append({"backend": backend, "status": "complete" if result else "failed" if failed else "pending",
                "median_seconds_per_point": median, "mad_seconds_per_point": mad,
                "initialization_and_warmup_seconds": observations[0]["cold_seconds"] if observations else None,
                "initialization_scope": "first amplitude initialization and saved-stream warmup; excludes startup/SETPARA",
                "lower_95": None, "upper_95": None, "samples": len(seconds),
                "actual_backend_precision_counts": usage,
                "usage_scope": "final accepted amplitude result; internal reducer attempts are not counted",
                "native_dp_only": all(item["native_dp_only"] for item in observations) if observations else None})
        gate = loaded.get("tirust-first")
        performance = {"status": "pending", "complete": False, "pairs": 0,
            "baseline_chain": None, "candidate_chain": None,
            "baseline_median_seconds_per_point": None, "candidate_median_seconds_per_point": None,
            "geometric_mean_ratio": None, "lower_95": None, "upper_95": None, "passed": None,
            "actual_backend_precision_counts": {}}
        if gate:
            stats = gate["statistics"]
            gate_directory = selected_run_directory(base, "tirust-first")
            gate_specification = json.loads((gate_directory / "specification.json").read_text())
            performance.update(status="complete" if gate["acceptance_eligible"] else "timing-ineligible",
                complete=gate["acceptance_eligible"], pairs=stats["pairs"],
                acceptance_eligible=gate["acceptance_eligible"], minimum_batch_duration_met=gate["minimum_batch_duration_met"],
                selected_attempt=gate_directory.name,
                baseline_chain=gate_specification["baseline_chain"],
                candidate_chain=gate_specification["candidate_chain"],
                baseline_median_seconds_per_point=statistics.median(gate["timings"]["baseline"]),
                candidate_median_seconds_per_point=statistics.median(gate["timings"]["candidate"]),
                geometric_mean_ratio=stats["geometric_mean_ratio"], lower_95=stats["lower_95"],
                upper_95=stats["upper_95"], passed=gate["passed"],
                actual_backend_precision_counts=aggregate_usage(gate["raw_observations"]["candidate"]))
        elif (selected_run_directory(base, "tirust-first") / "failure.json").exists() or "tirust-first" in validation_errors:
            performance["status"] = "failed"
        regressions = []
        for backend in (1, 6, 7, 8):
            label = f"backend-{backend}"
            value = loaded.get(label)
            failed = (selected_run_directory(base, label) / "failure.json").exists() or label in validation_errors
            regressions.append({"backend": backend,
                "status": "complete" if value else "failed" if failed else "pending",
                "complete": value["acceptance_eligible"] if value else False,
                "passed": value["passed"] if value else None,
                "statistics": value["statistics"] if value else None})
        point_path = REPORT / "points" / (case["id"] + ".json")
        points = json.loads(point_path.read_text())["inputs"]["count"] if point_path.exists() else None
        complete = (performance["complete"] and all(item["status"] == "complete" for item in correctness)
                    and all(item["complete"] for item in regressions))
        passed = (complete and performance["passed"] and all(item["passed"] for item in correctness)
                  and all(item["passed"] for item in regressions))
        status = "passed" if passed else "failed" if (complete or performance["status"] in ("failed", "timing-ineligible")
                    or performance["passed"] is False
                    or any(item["status"] == "failed" or item["passed"] is False for item in correctness)
                    or any(item["status"] == "failed" or item["passed"] is False for item in regressions)) else "pending"
        attempts = []
        for path in sorted(base.glob("*/summary.json")):
            try:
                value = json.loads(path.read_text())
                value, _ = validate_retained_result(path.parent, value["label"], case, config)
                attempts.append({"directory": path.parent.name, "label": value["label"],
                    "acceptance_eligible": value["acceptance_eligible"], "passed": value["passed"],
                    "statistics": value["statistics"], "selected": path.parent == selected_run_directory(base, value["label"])})
            except (OSError, ValueError, KeyError, RuntimeError) as error:
                attempts.append({"directory": path.parent.name, "status": "invalid-evidence", "error": str(error)})
        cases.append({**case, "status": status, "points": points, "correctness": correctness,
                      "retained_attempts": attempts,
                      "timings": timings, "performance": performance, "complete": complete,
                      "existing_backend_regressions": regressions, "validation_errors": validation_errors})
    main = [case for case in cases if case["role"] == "main"]
    reduction_chains = {}
    for source in ("baseline", "candidate"):
        observed = {tuple(case["performance"][source + "_chain"]) for case in cases
                    if case["performance"][source + "_chain"] is not None}
        reduction_chains[source] = list(next(iter(observed))) if len(observed) == 1 else None
    candidate_revision = next(iter(candidate_revisions)) if len(candidate_revisions) == 1 else None
    source_date_epoch = int(subprocess.check_output(
        ["git", "show", "-s", "--format=%ct", candidate_revision or BASELINE_REVISION], cwd=ROOT, text=True))
    qualification_path = REPORT / "qualification.json"
    qualification = json.loads(qualification_path.read_text()) if qualification_path.exists() else []
    if not isinstance(qualification, list):
        raise ValueError("qualification.json must contain an array of observed qualification records")
    summary = {"schema": "madloop-tirust-summary-v1", "created_utc": utc_now(),
        "field_definitions": {"native_dp_only": "Every accepted amplitude result used the first selected backend in DP; does not identify internal retry or higher-precision attempts",
                              "actual_backend_precision_counts": "Final accepted backend and precision per amplitude evaluation; internal reducer attempts are not counted"},
        "candidate_revision": candidate_revision, "candidate_revisions": sorted(candidate_revisions),
        "candidate_source_hashes": sorted(candidate_source_hashes),
        "reduction_chains": reduction_chains,
        "source_date_epoch": source_date_epoch, "date": utc_now()[:10], "qualification": qualification,
        "baseline_revision": BASELINE_REVISION, "cases": cases,
        "gate": {"complete": all(case["complete"] for case in main),
                 "passed": all(case["status"] == "passed" for case in main) and len(candidate_source_hashes) == 1,
                 "consistent_candidate_source": len(candidate_source_hashes) == 1,
                 "validation_errors": (["Multiple candidate numerical source identities occur in the campaign"]
                                       if len(candidate_source_hashes) > 1 else []),
                 "pending_cases": [case["id"] for case in main if case["status"] == "pending"],
                 "failed_cases": [case["id"] for case in main if case["status"] == "failed"]}}
    dump(REPORT / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "model", "generate", "build", "pair", "campaign", "summarize"))
    parser.add_argument("--source", choices=("baseline", "candidate", "both"), default="both")
    parser.add_argument("--cases", nargs="*", default=[])
    parser.add_argument("--model-source", type=Path)
    parser.add_argument("--model-url")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--cpu", type=int, help="Allowed CPU; default samples per-core load and selects the quietest")
    parser.add_argument("--baseline-chain")
    parser.add_argument("--candidate-chain")
    parser.add_argument("--label", default="tirust-first")
    parser.add_argument("--attempt", type=int, default=1, help="Append-only attempt number; retries retain earlier evidence")
    parser.add_argument("--retry-reason", help="Required reason for selecting an appended retry")
    args = parser.parse_args()
    if Path.cwd() != ROOT:
        parser.error(f"Run from {ROOT}")
    if args.attempt < 1 or (args.attempt > 1 and (args.command != "pair" or not args.retry_reason)):
        parser.error("Retries require pair, --attempt >=2 and --retry-reason")
    if args.cpu is None:
        args.cpu = select_quiet_cpu()
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("--cpu is outside the allowed CPU affinity")
    config = load_config()
    if args.command == "summarize":
        print(json.dumps(summarize(config)["gate"]))
        return
    unknown = set(args.cases) - {case["id"] for case in config["cases"]}
    if unknown:
        parser.error(f"Unknown cases: {sorted(unknown)}")
    cases = [case for case in config["cases"] if not args.cases or case["id"] in args.cases]
    if args.command == "init":
        prepare_baseline()
        print(json.dumps({"baseline": str(BASELINE), "revision": BASELINE_REVISION}))
        return
    if args.command == "model":
        print(acquire_model(args.model_source, args.model_url))
        return
    sources = ("baseline", "candidate") if args.source == "both" else (args.source,)
    if args.command in ("generate", "build"):
        prepare_baseline()
        for case in cases:
            for source in sources:
                generate(source, case)
                if args.command == "build":
                    freeze_points(case, build(source, case, args.jobs), config)
                print(f"{args.command}: {source} {case['id']}", flush=True)
        return
    campaign_failures = []
    for case in cases:
        generation = json.loads((output_dir("baseline", case) / "generation.json").read_text())
        baseline_chain = parse_chain(args.baseline_chain or generation["default_chain"])
        candidate_chain = parse_chain(args.candidate_chain) if args.candidate_chain else [9, *baseline_chain]
        labels = ([(f"backend-{backend}", [backend] if backend != 9 else [1], [backend])
                   for backend in (1, 6, 7, 8, 9)] if args.command == "campaign" else [])
        labels.append((args.label, baseline_chain, candidate_chain))
        for label, left, right in labels:
            base = REPORT / "runs" / case["id"]
            attempt_name = f"{label}-attempt-{args.attempt}" if args.attempt > 1 else None
            directory = base / attempt_name if attempt_name else selected_run_directory(base, label)
            try:
                if args.command == "campaign" and directory.exists():
                    summary, _ = validate_retained_result(directory, label, case, config)
                    spec = json.loads((directory / "specification.json").read_text())
                    if spec["cpu"] != args.cpu:
                        raise RuntimeError("Resume must use the retained CPU affinity")
                else:
                    summary = pair(case, left, right, config, args.cpu, label, attempt_name=attempt_name)
                    if attempt_name:
                        select_attempt(base, label, directory, args.retry_reason)
                print(json.dumps({"case": case["id"], "label": label, "statistics": summary["statistics"],
                                  "acceptance_eligible": summary["acceptance_eligible"]}), flush=True)
            except Exception as error:
                if args.command != "campaign":
                    raise
                campaign_failures.append({"case": case["id"], "label": label, "error": str(error)})
                print(json.dumps(campaign_failures[-1]), flush=True)
            finally:
                summarize(config)
    if campaign_failures:
        raise RuntimeError(f"Campaign retains {len(campaign_failures)} failed/incomplete labels; see summary.json")


if __name__ == "__main__":
    main()
