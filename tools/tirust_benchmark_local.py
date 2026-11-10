#!/usr/bin/env python3
"""Portable, append-only local measurement of a frozen benchmark campaign.

This is a measurement replica, not an optimization campaign. Origin manifests,
qualified failures and numerical inputs remain immutable. Native artifacts built
on this machine are a distinct deployment; a local result never replaces a
historical result. Run from the checkout root, after installing all five reducers.

Example (none of these commands run implicitly):
  python3 -B tools/tirust_benchmark_local.py install --campaign split-dp-terminal-1 \
      --installation hs-install-1 --tirust-prefix .tirust-build/local-hs-1/tirust
  python3 -B tools/tirust_benchmark_local.py prepare --campaign split-dp-terminal-1 \
      --measurement hs-local-1 --tirust-prefix .tirust-build/local-hs-1/tirust --cpu 0 \
      --install-receipt docs/benchmarkfolder/local-installations/hs-install-1/receipt.json
  python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 init
  python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 model
  python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 build
  python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 numerical
  python3 -B tools/tirust_benchmark_local.py forward --measurement hs-local-1 campaign
  python3 -B tools/tirust_benchmark_local.py verify --measurement hs-local-1

The numerical stage requires all six lanes for every frozen case. Formal timing
uses the unchanged core seven-pair protocol. There is no provisional profile,
arbitrary label, chain override, statistical retry, or total wall-time promise.
Install explicitly builds native code and records the source-to-artifact link
missing from ordinary installation metadata. Preparation copies source/evidence
only; forward init/build/numerical/campaign
can execute compilers or amplitudes. The original work directory is never used.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
import tomllib

import tirust_benchmark as bench
import tirust_benchmark_campaign as campaign

SCHEMA = "madloop-tirust-local-measurement-v1"
PROTOCOLS = ("tirust_benchmark_local.py", "tirust_benchmark.py",
             "tirust_benchmark_campaign.py", "tirust_benchmark_worker.py",
             "tirust_benchmark_worker.f", "tirust_benchmark_compiler.py",
             "tirust_benchmark_numerical.py", "tirust_benchmark_warmup.py")
GLOBALS = ("LEGACY_REPORT", "LEGACY_WORK", "REPORT", "WORK", "BASELINE",
           "BASELINE_RECORD", "MODEL_ROOT", "TIRUST_PREFIX", "CAMPAIGN_ID",
           "CAMPAIGN_MANIFEST", "CAMPAIGN_MANIFEST_SHA256", "campaign_guard",
           "case_worker_timeout")


def read(path):
    return json.loads(Path(path).read_text())


def within(root, name):
    """No absolute names, parent traversal, or symlink aliases in evidence."""
    root, name = Path(root), Path(name)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Local evidence path must be relative and contained")
    result = root / name
    for path in (result, *result.parents):
        if path == root.parent:
            break
        if path.is_symlink():
            raise ValueError(f"Local evidence may not alias a symlink: {path}")
    if not result.resolve().is_relative_to(root.resolve()):
        raise ValueError("Local evidence escapes its registry")
    return result


def roots(measurement):
    campaign._id(measurement)
    registry = bench.ROOT / "docs/benchmarkfolder/local-measurements"
    work = bench.ROOT / ".benchmark-work/tirust/local-measurements"
    return within(registry, measurement), within(work, measurement)


def host_identity():
    # CPU choice and user-supplied labels must not mint another host identity.
    machine_id = Path("/etc/machine-id")
    return {"machine_id_sha256": bench.digest(machine_id) if machine_id.is_file() else None,
            "node": platform.node(), "machine": platform.machine(), "system": platform.system()}


def deployment(prefix):
    native = campaign._native(bench, prefix)
    module = Path(native["prefix"]) / "lib/mtirust.mod"
    if not module.is_file():
        raise ValueError("Pending prerequisite: installed lib/mtirust.mod")
    native["files"]["lib/mtirust.mod"] = bench.digest(module)
    dependencies = {}
    for relative in ("HEPTools/ninja/lib", "HEPTools/collier", "HEPTools/ninjars/lib",
                     "vendor/CutTools/includects"):
        folder = bench.ROOT / relative
        files = {p.relative_to(bench.ROOT).as_posix(): bench.digest(p)
                 for p in sorted(folder.rglob("*")) if p.is_file()
                 and (p.suffix in (".a", ".so", ".dylib") or ".so." in p.name
                      or p.name.endswith(("-config.json", "-native-libs.txt", "-link-before.txt")))}
        if not files:
            raise ValueError(f"Pending prerequisite: installed reducer files in {relative}")
        dependencies.update(files)
    compilers = {}
    for variable, default in (("FC", "gfortran"), ("CC", "cc"), ("CXX", "c++")):
        executable = shutil.which(os.environ.get(variable, default))
        if executable is None:
            raise ValueError(f"Pending prerequisite: compiler {variable}")
        compilers[variable] = {"path": str(Path(executable).resolve()), "sha256": bench.digest(executable)}
    flags = {key: os.environ.get(key) for key in ("FFLAGS", "FCFLAGS", "CFLAGS", "CXXFLAGS", "LDFLAGS")}
    return {"native": native, "dependencies": dependencies, "compilers": compilers, "flags": flags}


def verify_recipe(prefix, canonical):
    """Validate the staged Cargo resolution, interfaces, capabilities and pins.

    Metadata alone cannot prove which bridge Rust source produced a new DSO.
    Noncanonical artifact bytes additionally require an actual-CLI receipt.
    """
    prefix = Path(prefix)
    config = read(prefix / "lib/tirust-config.json")
    for name, expected in (("abi_version", 1), ("quad_precision", True),
                           ("finite_terms", True), ("poles_only", True)):
        if type(config.get(name)) is not type(expected) or config[name] != expected:
            raise ValueError(f"Local TIRust recipe lacks required {name}")
    sources = canonical["candidate"]["exporter_identity"]["files"]
    for installed, source in (("Cargo.toml", "vendor/tirust/Cargo.toml"),
                              ("include/mtirust.f90", "vendor/tirust/mtirust.f90"),
                              ("include/tirust.h", "vendor/tirust/tirust.h")):
        if bench.digest(prefix / installed) != sources[source]:
            raise ValueError(f"Installed {installed} differs from canonical bridge source")
    expected_lock = canonical["candidate"]["native"]["files"]["Cargo.lock"]
    if bench.digest(prefix / "Cargo.lock") != expected_lock:
        raise ValueError("Installed resolved Cargo.lock differs from canonical staged build")
    lock = tomllib.loads((prefix / "Cargo.lock").read_text())
    frozen_lock = tomllib.loads((bench.ROOT / "vendor/tirust/Cargo.lock").read_text())
    git = lambda packages: sorted(({'name': p['name'], 'version': p['version'], 'source': p['source']}
                                   for p in packages if p.get('source', '').startswith('git+')),
                                  key=lambda p: (p['name'], p['source']))
    expected_git = [p for p in git(frozen_lock["package"]) if p["name"] not in ("iregi-rust", "iregi-masters")]
    provenance = {"manifest_sha256": bench.digest(prefix / "Cargo.toml"),
                  "lock_sha256": expected_lock, "git_dependencies": git(lock["package"])}
    if config.get("build_provenance") != provenance or provenance["git_dependencies"] != expected_git:
        raise ValueError("Installed Cargo manifest/lock/Git provenance differs from frozen dependency pins")
    patch = read(bench.ROOT / "vendor/tirust/patches/manifest.json")
    upstream = config.get("local_sources", {}).get("tirust", {})
    if (upstream.get("revision") != patch["upstream_revision"]
            or upstream.get("patch_sha256") != patch["patch_sha256"]):
        raise ValueError("Installed TIRust revision/patch differs from frozen pins")
    for name in ("libmadloop_tirust.so", "libtirust.a", "mtirust.mod"):
        if config.get("artifact_sha256", {}).get(name) != bench.digest(prefix / "lib" / name):
            raise ValueError("Installed native/module bytes differ from installation smoke receipt")
    return provenance


def verify_install_receipt(path, canonical, native):
    """A receipt binds a fresh real CLI build, including bridge Rust bytes."""
    path = Path(path)
    receipt = read(path)
    source = canonical["candidate"]["exporter_identity"]["files_sha256"]
    if (receipt.get("schema") != "madloop-tirust-local-install-v1"
            or receipt.get("status") != "complete"
            or receipt.get("source_before", {}).get("files_sha256") != source
            or receipt.get("source_after", {}).get("files_sha256") != source
            or receipt.get("native") != native
            or bench.digest(path.parent / "commands.mg5") != receipt.get("command_file_sha256")
            or bench.digest(path.parent / "console.log") != receipt.get("console_sha256")):
        raise ValueError("Local native build lacks a matching actual-CLI source receipt")
    return {"path": str(path.resolve()), "sha256": bench.digest(path),
            "command_file_sha256": receipt["command_file_sha256"], "console_sha256": receipt["console_sha256"]}


def install(campaign_id, installation, prefix, jobs=4):
    """Explicit native action, never called by prepare/verify/forward."""
    campaign._id(installation)
    if isinstance(jobs, bool) or jobs < 1:
        raise ValueError("Installation jobs must be positive")
    canonical = read(bench.ROOT / "docs/benchmarkfolder/campaigns" / campaign._id(campaign_id) / "manifest.json")
    source = bench.source_identity(bench.ROOT, persist=False)
    if source["files_sha256"] != canonical["candidate"]["exporter_identity"]["files_sha256"]:
        raise ValueError("Install checkout differs from canonical bridge/exporter sources")
    prefix = Path(prefix).resolve()
    campaign._inside(bench.ROOT, prefix)
    if prefix.name != "tirust" or prefix.exists():
        raise ValueError("Source-receipted installation requires a fresh repository-contained prefix ending in /tirust")
    directory = within(bench.ROOT / "docs/benchmarkfolder/local-installations", installation)
    directory.mkdir(parents=True, exist_ok=False)
    command = directory / "commands.mg5"
    command.write_text("install tirust --prefix=" + shlex.quote(str(prefix.parent))
                       + f" --jobs={jobs}\nquit\n")
    value = {"schema": "madloop-tirust-local-install-v1", "status": "running", "source_before": source,
             "canonical_manifest_sha256": bench.digest(bench.ROOT / "docs/benchmarkfolder/campaigns" / campaign_id / "manifest.json"),
             "command": [sys.executable, "-B", "bin/mg5_aMC", str(command)],
             "command_file_sha256": bench.digest(command), "started_utc": bench.utc_now()}
    bench.dump(directory / "intent.json", value)
    try:
        with (directory / "console.log").open("w") as log:
            subprocess.run(value["command"], cwd=bench.ROOT, stdout=log, stderr=subprocess.STDOUT,
                           check=True, timeout=86400, env=bench.environment())
        if "TIRust successfully installed in" not in (directory / "console.log").read_text():
            raise ValueError("MG CLI did not report a successful fresh TIRust publication")
        value["source_after"] = bench.source_identity(bench.ROOT, persist=False)
        if value["source_after"]["files_sha256"] != source["files_sha256"]:
            raise ValueError("Canonical source bytes changed during local installation")
        value["recipe"] = verify_recipe(prefix, canonical)
        value["native"] = deployment(prefix)["native"]
        value["console_sha256"] = bench.digest(directory / "console.log")
        value["status"] = "complete"
        value["finished_utc"] = bench.utc_now()
        bench.dump(directory / "receipt.json", value)
    except BaseException as error:
        value.update(status="failed", error=repr(error), finished_utc=bench.utc_now())
        bench.dump(directory / "failure.json", value)
        raise
    return {"install_receipt": str(directory / "receipt.json"), "tirust_prefix": str(prefix)}


def deployment_key(value):
    # Paths relocate; actual native/compiler bytes and build flags do not.
    return {"native": {name: value["native"]["files"][name] for name in campaign.NATIVE_NAMES[:2]},
            "dependencies": {name: checksum for name, checksum in value["dependencies"].items()
                             if Path(name).suffix in (".a", ".so", ".dylib") or ".so." in Path(name).name},
            "compilers": {key: item["sha256"] for key, item in value["compilers"].items()},
            "flags": value["flags"]}


def origin_inventory(campaign_id):
    """Freeze ancestry using the existing failure collector, including late failures."""
    base = bench.ROOT / "docs/benchmarkfolder"
    origin = within(base, "campaigns/" + campaign._id(campaign_id))
    manifest = read(origin / "manifest.json")
    if manifest.get("id") != campaign_id or manifest.get("schema") != campaign.SCHEMA:
        raise ValueError("Invalid canonical campaign manifest")
    # The caller is outside the isolated adapter here.
    predecessor = campaign._predecessor(bench, campaign_id)
    found = campaign._verify_predecessor(bench, predecessor)
    names = set()

    def add(path):
        relative = Path(path).relative_to(base).as_posix()
        within(base, relative)
        names.add(relative)

    node = predecessor
    while node:
        report = within(base, node["report"])
        add(report / "cases.json")
        add(within(base, node["summary_snapshot"]) if node.get("summary_snapshot") else report / "summary.json")
        if node["manifest_sha256"] is not None:
            add(report / "manifest.json")
            retained = read(report / "manifest.json")
            for name in retained["shared_inputs"]["files"]:
                add(within(report, name))
            for name in retained["candidate"]["numerical_sources"]:
                add(within(report, retained["candidate"]["source_snapshot"] + "/" + name))
        for discovery in node.get("inherited_discoveries", []):
            add(within(base, discovery["path"]))
        node = node.get("ancestor")
    for record in found:
        for name in record["failure"]["files"]:
            add(within(base, record["report"] + "/" + name))
    physical = {}
    for path in sorted((origin / "experiments/numerical").glob("*/*/*-inputs.json")):
        record = read(path)
        case_id = record["generation"]["case"]["id"]
        checksum = record["param_card_sha256"]
        if case_id in physical and physical[case_id] != checksum:
            raise ValueError(f"Canonical physical cards disagree for {case_id}")
        physical[case_id] = checksum
        add(path)
    return {"campaign": campaign_id, "manifest": manifest,
            "manifest_sha256": bench.digest(origin / "manifest.json"),
            "predecessor": predecessor, "qualified_failures": found,
            "physical_cards": physical,
            "files": {name: bench.digest(base / name) for name in sorted(names)}}


def materialize_baseline(work, record):
    """Use the frozen record verbatim; core init would otherwise replace its date."""
    archive = subprocess.check_output(["git", "archive", "--format=tar", bench.BASELINE_REVISION], cwd=bench.ROOT)
    if hashlib.sha256(archive).hexdigest() != read(record)["archive_sha256"]:
        raise ValueError("Local git archive differs from frozen baseline")
    destination = work / "baseline"
    destination.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as handle:
        handle.extractall(destination, filter="data")
    if bench.tree_files(destination) != read(record)["files"]:
        raise ValueError("Materialized baseline source inventory differs")
    shutil.copy2(record, work / "baseline.json")


def prepare(measurement, campaign_id, prefix, cpu, install_receipt=None):
    report, work = roots(measurement)
    if cpu not in os.sched_getaffinity(0):
        raise ValueError("Selected CPU is outside the allowed affinity")
    origin = origin_inventory(campaign_id)
    canonical = origin["manifest"]
    source = bench.source_identity(bench.ROOT, persist=False)
    if source["files_sha256"] != canonical["candidate"]["exporter_identity"]["files_sha256"]:
        raise ValueError("Checkout does not match the canonical exporter source inventory")
    native = deployment(prefix)
    recipe = verify_recipe(native["native"]["prefix"], canonical)
    canonical_bytes = all(native["native"]["files"][name] == canonical["candidate"]["native"]["files"][name]
                          for name in campaign.NATIVE_NAMES[:2])
    receipt = (verify_install_receipt(install_receipt, canonical, native["native"])
               if install_receipt is not None else None)
    if not canonical_bytes and receipt is None:
        raise ValueError("New local library bytes need an actual-CLI source receipt: run this launcher's install command first")
    config = read(bench.ROOT / "docs/benchmarkfolder/campaigns" / campaign_id / "cases.json")
    if config["points_per_case"] != 100 or config["repetitions"] < 7:
        raise ValueError("Local formal profile requires frozen 100 points and at least seven pairs")
    if set(origin["physical_cards"]) != {case["id"] for case in config["cases"]}:
        raise ValueError("Pending prerequisite: canonical physical-card bindings for every case")
    identity = {"numerical_source_sha256": bench.content_digest({name: item["numerical_sha256"]
                    for name, item in canonical["candidate"]["numerical_sources"].items()}),
                "policy_sha256": canonical["shared_inputs"]["policy_sha256"],
                "host": host_identity(), "deployment": deployment_key(native)}
    key = bench.content_digest(identity)
    registry = report.parent
    registry.mkdir(parents=True, exist_ok=True)
    with (registry / ".registration.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        keys = registry / ".keys"
        keys.mkdir(exist_ok=True)
        key_path = keys / (key + ".json")
        if key_path.exists():
            raise ValueError("This source/policy/host/build already has a local measurement; labels cannot replace it")
        for existing in sorted(registry.glob("*/measurement.json")):
            if read(existing).get("registry_key") == key:
                raise ValueError("This source/policy/host/build already has a local measurement; labels cannot replace it")
        if report.exists() or work.exists():
            raise ValueError("Local measurement IDs are append-only")
        report.mkdir()
        work.mkdir(parents=True)
        try:
            for name, checksum in origin["files"].items():
                target = within(report / "origin", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(bench.ROOT / "docs/benchmarkfolder" / name, target)
                if bench.digest(target) != checksum:
                    raise ValueError("Canonical evidence changed during snapshot")
            for name in canonical["shared_inputs"]["files"]:
                target = within(report / "report", name)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(report / "origin/campaigns" / campaign_id / name, target)
            materialize_baseline(work, report / "report/inputs/baseline.json")
            protocols = {name: bench.digest(bench.ROOT / "tools" / name) for name in PROTOCOLS}
            value = {"schema": SCHEMA, "id": measurement, "created_utc": bench.utc_now(),
                     "profile": "formal-local", "registry_key": key, "registry_identity": identity,
                     "origin": origin, "local_deployment": native, "cpu": cpu, "protocols": protocols,
                     "native_recipe": recipe, "install_receipt": receipt,
                     "native_source_binding": "actual-CLI-source-receipt" if receipt else "byte-identical-canonical-libraries",
                     "source_identity": source, "overall_goal_qualified": False,
                     "historical_results_replaced": False,
                     "note": "Local deployment measurement; not an optimization or replacement of historical failures"}
            with (report / "measurement.json").open("x") as stream:
                json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            with key_path.open("x") as stream:
                json.dump({"id": measurement, "registry_key": key,
                           "manifest_sha256": bench.digest(report / "measurement.json")}, stream, sort_keys=True)
        except BaseException:
            # Failed source-only preparation has no amplitude/timing evidence.
            shutil.rmtree(report)
            shutil.rmtree(work)
            raise
    return verify(measurement)


def verify(measurement, live=True):
    report, work = roots(measurement)
    value = read(report / "measurement.json")
    if value.get("schema") != SCHEMA or value.get("id") != measurement:
        raise ValueError("Invalid local measurement record")
    if bench.content_digest(value["registry_identity"]) != value["registry_key"]:
        raise ValueError("Local registry identity changed")
    registry_record = read(report.parent / ".keys" / (value["registry_key"] + ".json"))
    if registry_record != {"id": measurement, "registry_key": value["registry_key"],
                           "manifest_sha256": bench.digest(report / "measurement.json")}:
        raise ValueError("Local measurement differs from its immutable registry seal")
    origin = value["origin"]
    for name, checksum in origin["files"].items():
        if bench.digest(within(report / "origin", name)) != checksum:
            raise ValueError("Frozen canonical evidence or historical failure changed")
    canonical = origin["manifest"]
    origin_manifest = report / "origin/campaigns" / origin["campaign"] / "manifest.json"
    if read(origin_manifest) != canonical or bench.digest(origin_manifest) != origin["manifest_sha256"]:
        raise ValueError("Canonical origin manifest changed")
    for name, checksum in canonical["shared_inputs"]["files"].items():
        if bench.digest(within(report / "report", name)) != checksum:
            raise ValueError("Local frozen physics inputs changed")
    if bench.digest(work / "baseline.json") != canonical["baseline"]["record_sha256"]:
        raise ValueError("Local baseline record changed")
    if live:
        if host_identity() != value["registry_identity"]["host"]:
            raise ValueError("Prepare a measurement on this host; copied executable sessions cannot resume elsewhere")
        if value["cpu"] not in os.sched_getaffinity(0):
            raise ValueError("Registered CPU is no longer in the allowed affinity")
        if bench.source_identity(bench.ROOT, persist=False)["files_sha256"] != value["source_identity"]["files_sha256"]:
            raise ValueError("Candidate exporter/source changed after local registration")
        if deployment(value["local_deployment"]["native"]["prefix"]) != value["local_deployment"]:
            raise ValueError("Registered native dependency/compiler deployment changed")
        if verify_recipe(value["local_deployment"]["native"]["prefix"], canonical) != value["native_recipe"]:
            raise ValueError("Installed recipe changed")
        if value.get("install_receipt"):
            receipt = value["install_receipt"]
            if verify_install_receipt(receipt["path"], canonical, value["local_deployment"]["native"]) != receipt:
                raise ValueError("Local installation source receipt changed")
        for name, checksum in value["protocols"].items():
            if bench.digest(bench.ROOT / "tools" / name) != checksum:
                raise ValueError("Local measurement protocol changed")
        # An updated canonical failure inventory cannot silently disappear from
        # an old local registration. Publication must wait for closed inputs.
        if origin_inventory(origin["campaign"])["files"] != origin["files"]:
            raise ValueError("Canonical evidence changed or gained failures after local registration")
    model = work / "models" / campaign.MODEL
    pending = []
    if not model.exists():
        pending.append("model: pinned electroweak UFO is not materialized")
    elif campaign._inventory(bench, model) != canonical["shared_inputs"]["model_tree"]:
        raise ValueError("Local electroweak UFO differs from frozen model")
    config = read(report / "report/cases.json")
    for case in config["cases"]:
        for source in ("baseline", "candidate"):
            if not (work / "outputs" / source / case["id"] / "compilation.json").exists():
                pending.append(f"build: {source}/{case['id']}")
        numerical = report / "report/experiments/numerical" / ("ordinary-" + case["id"])
        if not (numerical / "results.json").exists():
            pending.append(f"numerical: {case['id']}")
        else:
            rows = read(numerical / "results.json").get("results", [])
            expected = {("baseline", 1), *(("candidate", b) for b in (1, 6, 7, 8, 9))}
            if (len(rows) != 6 or {(r.get("source"), r.get("backend")) for r in rows} != expected
                    or any(r.get("status") != "complete" or not (r.get("comparison") or {}).get("passed") for r in rows)
                    or any(numerical.glob("*/failure.json"))):
                pending.append(f"numerical: {case['id']} has retained failed/incomplete lanes")
    return {"measurement": measurement, "manifest_sha256": bench.digest(report / "measurement.json"),
            "pending_prerequisites": pending, "historical_qualified_failures": len(origin["qualified_failures"]),
            "verification_scope": "registry/source/input integrity; full numerical raw replay is mandatory before campaign",
            "overall_goal_qualified": False}


@contextmanager
def isolated(measurement):
    """Verified adapter; core physics/timing code is unchanged and fully isolated."""
    verify(measurement)
    report, work = roots(measurement)
    value = read(report / "measurement.json")
    checksum = bench.digest(report / "measurement.json")
    saved = {key: getattr(bench, key) for key in GLOBALS}

    def guard(stage, inputs=None):
        # Restore only canonical roots while verifying the origin registry.
        local = {key: getattr(bench, key) for key in ("LEGACY_REPORT", "LEGACY_WORK")}
        try:
            bench.LEGACY_REPORT = bench.ROOT / "docs/benchmarkfolder"
            bench.LEGACY_WORK = bench.ROOT / ".benchmark-work/tirust"
            verify(measurement)
        finally:
            for key, item in local.items():
                setattr(bench, key, item)
        if bench.digest(report / "measurement.json") != checksum:
            raise ValueError("Local measurement manifest changed during operation")
        if inputs is not None:
            generation = inputs.get("generation", inputs)
            if "parameter_card_sha256" in generation:
                expected = value["origin"]["physical_cards"][generation["case"]["id"]]
                if generation["parameter_card_sha256"] != expected:
                    raise ValueError("Local physical card differs from canonical campaign")
            if generation.get("source") == "candidate" and generation["source_identity"]["files_sha256"] != value["source_identity"]["files_sha256"]:
                raise ValueError("Recorded local candidate source differs from registration")
        return {"local_measurement_sha256": checksum}

    try:
        bench.LEGACY_REPORT = bench.REPORT = report / "report"
        bench.LEGACY_WORK = bench.WORK = work
        bench.BASELINE, bench.BASELINE_RECORD = work / "baseline", work / "baseline.json"
        bench.MODEL_ROOT = work / "models"
        bench.TIRUST_PREFIX = Path(value["local_deployment"]["native"]["prefix"])
        bench.CAMPAIGN_ID = bench.CAMPAIGN_MANIFEST = bench.CAMPAIGN_MANIFEST_SHA256 = None
        bench.campaign_guard = guard
        timeouts = value["origin"]["manifest"][campaign.TIMEOUT_FIELD]
        bench.case_worker_timeout = lambda case: float(timeouts[case["id"]])
        yield value
    finally:
        for key, item in saved.items():
            setattr(bench, key, item)


def verify_numerical(cpu):
    """Replay all final warmup streams; headline JSON alone is insufficient."""
    from tirust_benchmark_warmup import parse_warmup
    config = bench.load_config()
    verified = {}
    for case in config["cases"]:
        directory = bench.REPORT / "experiments/numerical" / ("ordinary-" + case["id"])
        specification = read(directory / "specification.json")
        expected_protocols = {"tirust_benchmark_numerical.py", "tirust_benchmark.py",
                              "tirust_benchmark_worker.py", "tirust_benchmark_worker.f",
                              "tirust_benchmark_compiler.py", "tirust_benchmark_warmup.py"}
        if (specification["case"] != case
                or specification["configuration_sha256"] != bench.digest(bench.REPORT / "cases.json")
                or specification.get("measurement_phase") != "steady-warmup"
                or specification.get("performance_eligible") is not False
                or specification["machine"]["selected_cpu"] != cpu
                or set(specification["protocols"]) != expected_protocols):
            raise ValueError("Numerical prerequisite uses different cases/configuration")
        for name, checksum in specification["protocols"].items():
            if bench.digest(directory / name) != checksum or bench.digest(bench.ROOT / "tools" / name) != checksum:
                raise ValueError("Numerical prerequisite protocol changed")
        results, reference = [], None
        for source, backend in [("baseline", 1), *(("candidate", b) for b in (1, 6, 7, 8, 9))]:
            lane = directory / f"{source}-{backend}"
            if (lane / "failure.json").exists():
                raise ValueError("Numerical prerequisite retains a failed lane")
            item = read(lane / "result.json")
            raw = parse_warmup(lane / "raw", [backend], 100, read(lane / "raw/ready.json"))
            inputs = read(lane / (source + "-inputs.json"))
            after = read(lane / "inputs-after.json")
            bench.campaign_guard("numerical-prerequisite", inputs=inputs)
            if (item.get("status") != "complete" or item["observation"] != raw
                    or item.get("source") != source or item.get("backend") != backend
                    or item.get("performance_eligible") is not False
                    or after.get("executable_sha256") != inputs["compilation"]["executable_sha256"]
                    or after.get("points_sha256") != inputs["points_sha256"]
                    or after.get("numerical_cards") != inputs["numerical_cards"]
                    or after.get("runtime") != inputs["runtime"]):
                raise ValueError("Numerical prerequisite raw/input/postcheck binding differs")
            if (inputs["generation"]["source"] != source or inputs["generation"]["case"] != case
                    or inputs["param_card_sha256"] != inputs["generation"]["parameter_card_sha256"]
                    or inputs.get("worker_timeout_seconds") != bench.case_worker_timeout(case)
                    or inputs["compilation"].get("numerical_build_from_sources") is not True):
                raise ValueError("Numerical prerequisite generation/build binding differs")
            expected_settings = dict(config["madloop_parameters"])
            if source == "candidate":
                expected_settings.update(config["candidate_parameters"])
            if (bench.parse_chain(inputs["settings"]["mlreductionlib"]) != [backend]
                    or any(inputs["settings"].get(name.lower()) != option
                           for name, option in expected_settings.items())):
                raise ValueError("Numerical prerequisite uses different stability/pole/backend settings")
            if inputs["points_sha256"] != bench.digest(bench.REPORT / "points" / (case["id"] + ".dat")):
                raise ValueError("Numerical prerequisite uses different physical points")
            output = bench.output_dir(source, case)
            compilation = inputs["compilation"]
            if (read(output / "compilation.json") != compilation
                    or read(output / "generation.json") != inputs["generation"]
                    or bench.generated_sources(output) != compilation["sources"]
                    or bench.digest(bench.subprocess_dir(source, case) / "check") != compilation["executable_sha256"]):
                raise ValueError("Numerical prerequisite no longer describes current compiled sources/executable")
            bench.verify_worker_template(compilation, directory / "tirust_benchmark_worker.f")
            for section in ("libraries", "archives", "metadata"):
                if any(bench.digest(path) != checksum for path, checksum in inputs["runtime"][section].items()):
                    raise ValueError("Numerical prerequisite dependency bytes changed")
            if reference is None:
                reference = raw
            comparison = bench.compare_rows(reference["rows"], raw["rows"], config["relative_tolerance"])
            if item["comparison"] != comparison or not comparison["passed"]:
                raise ValueError("Numerical prerequisite fails frozen regression tolerance")
            results.append(item)
        if read(directory / "results.json")["results"] != results:
            raise ValueError("Numerical prerequisite lane summary differs")
        verified[case["id"]] = {p.relative_to(directory).as_posix(): bench.digest(p)
            for p in sorted(directory.rglob("*")) if p.is_file() and "__pycache__" not in p.parts}
    return verified


def forward(measurement, stage, jobs=4):
    if stage not in ("init", "model", "build", "numerical", "campaign", "summarize"):
        raise ValueError("Unsupported local forward stage")
    report, work = roots(measurement)
    old_argv = sys.argv
    with isolated(measurement) as value:
        with (report / ".execution.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            journal = report / "commands"
            journal.mkdir(exist_ok=True)
            number = len(list(journal.glob("*.json")))
            event = journal / f"{number:04d}-{stage}.json"
            record = {"stage": stage, "started_utc": bench.utc_now(),
                      "local_measurement_sha256": bench.digest(report / "measurement.json"), "status": "running"}
            with event.open("x") as stream:
                json.dump(record, stream, sort_keys=True)
            try:
                if stage in ("init", "model", "build") and (bench.REPORT / "runs").exists():
                    raise ValueError("Formal measurements exist; preparation cannot mutate their deployment")
                if stage == "numerical":
                    import tirust_benchmark_numerical as numerical
                    for case in bench.load_config()["cases"]:
                        directory = bench.REPORT / "experiments/numerical" / ("ordinary-" + case["id"])
                        if directory.exists():
                            # Do not append another attempt or skip an incomplete lane.
                            if not (directory / "results.json").exists():
                                raise ValueError("Incomplete numerical case is retained; no automatic retry")
                            continue
                        sys.argv = ["tirust_benchmark_numerical.py", "--case", case["id"],
                                    "--label", "ordinary-" + case["id"], "--warmup-only", "--cpu", str(value["cpu"])]
                        numerical.main()
                    proof = verify_numerical(value["cpu"])
                    bench.dump(report / "numerical-prerequisites.json", proof)
                else:
                    if stage == "campaign":
                        proof = verify_numerical(value["cpu"])
                        seal = report / "numerical-prerequisites.json"
                        if not seal.exists() or read(seal) != proof:
                            raise ValueError("Complete numerical prerequisite seal is missing or changed")
                    sys.argv = ["tirust_benchmark.py", stage, "--cpu", str(value["cpu"]), "--jobs", str(jobs)]
                    if stage == "model":
                        pin = read(bench.REPORT / "inputs" / (campaign.MODEL + ".json"))
                        if not str(pin["origin"]).startswith("https://"):
                            raise ValueError("Pinned model origin is not an HTTPS archive; materialize its exact tree in the local models directory")
                        sys.argv += ["--model-url", pin["origin"]]
                    bench.main()
                bench.campaign_guard("forward-after")
                record["status"] = "complete"
            except BaseException as error:
                record.update(status="failed", error=repr(error))
                raise
            finally:
                record["finished_utc"] = bench.utc_now()
                bench.dump(event, record)
                sys.argv = old_argv
    return verify(measurement)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    installer = commands.add_parser("install", help="Explicitly run MG CLI in a fresh prefix and retain bridge source receipt (native build)")
    installer.add_argument("--campaign", required=True)
    installer.add_argument("--installation", required=True)
    installer.add_argument("--tirust-prefix", type=Path, required=True)
    installer.add_argument("--jobs", type=int, default=4)
    prep = commands.add_parser("prepare", help="Freeze a local registry entry and source/evidence copies; no compilation")
    prep.add_argument("--campaign", required=True)
    prep.add_argument("--measurement", required=True)
    prep.add_argument("--tirust-prefix", type=Path, required=True)
    prep.add_argument("--cpu", type=int, required=True)
    prep.add_argument("--install-receipt", type=Path, help="Required for locally rebuilt native bytes; produced by install")
    run = commands.add_parser("forward", help="Explicitly execute one isolated core workflow stage")
    run.add_argument("--measurement", required=True)
    run.add_argument("stage", choices=("init", "model", "build", "numerical", "campaign", "summarize"))
    run.add_argument("--jobs", type=int, default=4)
    check = commands.add_parser("verify", help="Verify source/input/ancestry seals and list pending prerequisites")
    check.add_argument("--measurement", required=True)
    args = parser.parse_args()
    if Path.cwd() != bench.ROOT:
        parser.error(f"Run from {bench.ROOT}")
    if args.command == "install":
        result = install(args.campaign, args.installation, args.tirust_prefix, args.jobs)
    elif args.command == "prepare":
        result = prepare(args.measurement, args.campaign, args.tirust_prefix, args.cpu, args.install_receipt)
    elif args.command == "forward":
        result = forward(args.measurement, args.stage, args.jobs)
    else:
        result = verify(args.measurement)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
