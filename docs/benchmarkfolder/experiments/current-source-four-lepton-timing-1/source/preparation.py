"""Pure file admission and private export copies; no compiler/loader queries.

Importing this module does nothing. The coordinating controller admits prepare
explicitly. Real worker startup retains the standard compiler/runtime checks.
"""
import hashlib
import json
from pathlib import Path
import shutil
import stat

CASES = ("qcd_n4_uux_epem_mupmum",)
BACKENDS = (1, 6, 7, 8, 9)
SCHEMA = "madloop-current-source-four-lepton-private-preparation-v1"
ORIGIN = ".tirust-build/current-source-four-lepton-preparation-1"
MANIFEST = "docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/manifest.json"
BORROWED = "docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/experiments/numerical/full100-warmup-qcd_n1_uux_z-1/candidate-9"
SOURCE_FILES_SHA = "afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6"
SOURCE_REVISION = "0fcd2cd873d61d134837e1d9e87afb13a47824fd"
NUMERICAL_IDENTITY = "3c13ad88b825d9d19b1030b62b317c0f31759e002d8dfdb9050142301a7cae24"
LIBRARY_SHA = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
PINS = {
    MANIFEST: "4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076",
    ORIGIN + "/status.json": "706dfd8d09517ebefc836814cad6ead5572dd95d6d203b7c9c7b8e36789f41ca",
    ORIGIN + "/prepared.json": "1e53ca535492a65f002222c6fcbe0b2ea6dbee731ec70323e3754581974e9c82",
    ORIGIN + "/case-plan.json": "818bdee2aec6dc00a39e049526ace8ee396eb86a04d6f18e9b302d0811a742a3",
    ORIGIN + "/report/cases.json": "fda7852415d58c0f5d5cf4a31ef4746b5c742d767182cb5dda40643e89fd5a64",
    ORIGIN + "/work/outputs/candidate/qcd_n4_uux_epem_mupmum/generation.json": "f24e0016e301498e3e4fdf74a3f6a2d1ad929fbfc80c1e817c41b8b1c5eb279c",
    ORIGIN + "/work/outputs/candidate/qcd_n4_uux_epem_mupmum/compilation.json": "09c60658c5407177358c41fb886d5371e8fda2870c191d69508d7eedb7e0c0a5",
    ORIGIN + "/report/points/qcd_n4_uux_epem_mupmum.dat": "a5c2c8e47b17fa2ba495eff47e869d41cc04a5c68973b32df00fc2a781455f39",
    ORIGIN + "/report/points/qcd_n4_uux_epem_mupmum.json": "b59bc647e5e7aaebf3e28ac98d28b9840413508f27ef3e8f24d1c2e06727941f",
    BORROWED + "/candidate-inputs.json": "9ae42c28fbe0fb74ec8f426f3b709ee9d3b22208646b20215d27436ca5574649",
    BORROWED + "/inputs-after.json": "eaa76a366611072355204aacb3549efd14d503a16c1987a7ec1e2c39c5a42c89",
}
SCRATCH = {"values.dat", "timing.dat", "timed_status.dat", "dimensions.dat", "warmup.dat",
           "worker-protocol.fifo"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def inside(root, name):
    path = Path(root) / name
    require(not Path(name).is_absolute() and ".." not in Path(name).parts
            and path.resolve().is_relative_to(Path(root).resolve()), "Input escapes its declared root")
    return path


def objects(bench, output):
    return {path.relative_to(output).as_posix(): bench.digest(path)
            for scope in ("Source", "SubProcesses") for path in sorted((output / scope).rglob("*"))
            if path.is_file() and path.suffix.lower() in (".o", ".mod", ".a")
            and "build-history" not in path.parts and "worker-raw" not in path.parts}


def mutable_paths_are_local(output):
    output = Path(output).resolve()
    paths = [output / "Cards", *(output / "Cards").rglob("*")]
    for path in (output / "SubProcesses").rglob("*"):
        if (path.name == "MadLoop5_resources" or "MadLoop5_resources" in path.parts
                or path.name == "MadLoopParams.dat" or path.name in SCRATCH
                or path.name.endswith(("HelFilter.dat", "LoopFilter.dat"))):
            paths.append(path)
    for path in paths:
        require(path.resolve().is_relative_to(output), "Mutable card/resource escapes the private export: " + str(path))


def subprocess_directory(output):
    directories = sorted(Path(output).glob("SubProcesses/P*_*"))
    require(len(directories) == 1 and directories[0].is_dir(), "Expected one original/private subprocess")
    require(not (directories[0] / "worker-protocol.fifo").exists(), "A worker is active in this export")
    return directories[0]


def runtime_files(bench, runtime):
    """Hash each recorded path independently; duplicate DSO basenames are valid."""
    result = {}
    for section in ("libraries", "archives", "metadata"):
        require(isinstance(runtime.get(section), dict), "Incomplete historical runtime closure")
        result[section] = {}
        for name, checksum in sorted(runtime[section].items()):
            path = Path(name)
            require(path.is_absolute() and path.is_file() and bench.digest(path) == checksum,
                    "Recorded runtime file changed: " + name)
            result[section][name] = {"sha256": checksum, "bytes": path.stat().st_size,
                                     "resolved_path": str(path.resolve())}
    return result


def shared_bindings(bench):
    """Join the closed private build to its unchanged registered source bytes."""
    require(bench.campaign_binding() is None and bench.CAMPAIGN_ID is None
            and bench.CAMPAIGN_MANIFEST is None, "Four-lepton case must remain outside registered campaigns")
    origin = inside(bench.ROOT, ORIGIN)
    require(Path(bench.REPORT).resolve() == (origin / "report").resolve()
            and Path(bench.WORK).resolve() == (origin / "work").resolve(), "Wrong closed four-lepton namespace")
    for name, checksum in PINS.items():
        require(bench.digest(inside(bench.ROOT, name)) == checksum, "Closed preparation input changed: " + name)
    manifest_path = inside(bench.ROOT, MANIFEST)
    manifest = read(manifest_path)
    closed_plan = read(origin / "case-plan.json")
    configurations = {path.relative_to(bench.ROOT).as_posix(): bench.digest(path)
                      for path in sorted((Path(bench.ROOT) / "input").glob("*configuration*.txt")) if path.is_file()}
    require(configurations == closed_plan["configuration_files"], "Production configuration set/bytes changed")
    native = manifest["candidate"]["native"]
    prefix = inside(bench.ROOT, closed_plan["tirust_prefix"])
    require(str(prefix) == native["prefix"] and Path(bench.TIRUST_PREFIX).resolve() == prefix.resolve()
            and native["files"]["lib/libmadloop_tirust.so"] == LIBRARY_SHA, "Canonical TIRust prefix/provider differs")
    for name, checksum in native["files"].items():
        require(bench.digest(inside(prefix, name)) == checksum, "Canonical TIRust file changed: " + name)
    source = manifest["candidate"]["exporter_identity"]
    require(source["revision"] == SOURCE_REVISION and source["files_sha256"] == SOURCE_FILES_SHA
            and manifest["candidate"]["numerical_identity_sha256"] == NUMERICAL_IDENTITY,
            "Registered source/numerical identity differs")
    for name, checksum in source["files"].items():
        path = inside(bench.ROOT, name)
        actual = bench.digest(path) if path.is_file() else None
        require(actual == checksum, "Registered current source changed: " + name)
    canonical = hashlib.sha256(json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(canonical == source["files_sha256"], "Registered source inventory digest differs")
    closed = read(origin / "status.json")
    require(closed["status"] == "prepared" and closed["amplitude_evaluations"] == 0
            and [row["id"] for row in closed["stages"]] == ["generate", "build"]
            and all(row["status"] == "passed" and row["returncode"] == 0
                    and row["cleanup"]["survivors"] == [] for row in closed["stages"]),
            "Original preparation is not cleanly closed")
    model_files = {name: checksum for name, checksum in source["files"].items()
                   if name.startswith("models/loop_sm/")}
    require(model_files, "loop_sm source inventory missing")
    template = Path(bench.ROOT) / "tools/tirust_benchmark_worker.f"
    return {"campaign_binding": None, "manifest_sha256": bench.digest(manifest_path),
            "source_identity": source, "model_root": str((Path(bench.ROOT) / "models/loop_sm").resolve()),
            "model_files": model_files, "worker_template": str(template.resolve()),
            "worker_template_sha256": bench.digest(template),
            "closed_preparation_inputs": dict(PINS), "numerical_identity_sha256": NUMERICAL_IDENTITY}


def borrowed_runtime(bench, compilation):
    """Expected provider set only; this is not a four-lepton runtime observation."""
    directory = inside(bench.ROOT, BORROWED)
    inputs, after = (read(directory / name) for name in ("candidate-inputs.json", "inputs-after.json"))
    runtime = inputs["runtime"]
    require(inputs["generation"]["source_identity"]["files_sha256"] == SOURCE_FILES_SHA
            and inputs["compilation"]["compiler"] == compilation["compiler"],
            "Borrowed provider source/toolchain differs")
    require(all(after["runtime"][key] == runtime[key] for key in ("libraries", "archives", "metadata"))
            and after["executable_sha256"] == inputs["compilation"]["executable_sha256"]
            and after["points_sha256"] == inputs["points_sha256"]
            and after["numerical_cards"] == inputs["numerical_cards"], "Borrowed runtime postcheck differs")
    require(LIBRARY_SHA in runtime["libraries"].values(), "Borrowed runtime lacks expected eb3 TIRust provider")
    # Static archives and metadata belong to the new original, never the donor.
    expected = {"libraries": dict(runtime["libraries"]), "ldd": runtime["ldd"],
                **compilation["link_inputs_before"]}
    return expected, {name: {"path": str((directory / name).resolve()), "sha256": bench.digest(directory / name)}
                      for name in ("candidate-inputs.json", "inputs-after.json")}


def collect_originals(bench):
    """File-only inventory; actual PT_INTERP and maps checks belong to RUN."""
    shared = shared_bindings(bench)
    config = bench.load_config()
    require(len(config["cases"]) == 1 and config["cases"][0]["id"] == CASES[0]
            and config["cases"][0]["model"] == "loop_sm" and config["cases"][0]["nfinal"] == 4
            and config["cases"][0]["process"] == "u u~ > e+ e- mu+ mu- [virt=QCD]"
            and config["seed"] == 20261109 and config["points_per_case"] == 100 and config["energy_gev"] == 1000.,
            "The pinned single four-lepton case/policy differs")
    case = config["cases"][0]
    case_id = case["id"]
    output = Path(bench.output_dir("candidate", case)).resolve()
    require(output == (inside(bench.ROOT, ORIGIN) / "work/outputs/candidate" / case_id).resolve(), "Wrong original export")
    directory = subprocess_directory(output)
    generation, compilation = read(output / "generation.json"), read(output / "compilation.json")
    require(generation["source"] == "candidate" and generation["case"] == case
            and all(item.get("campaign_binding") is None for item in (generation, compilation))
            and not generation.get("reused_from"), "Original generation/build belongs to another case")
    require(all(generation["source_identity"][key] == shared["source_identity"][key]
                for key in ("files", "files_sha256")), "Original exporter differs from current registered source")
    require(compilation.get("numerical_build_from_sources") is True
            and compilation.get("full_numerical_rebuild_this_invocation") is True
            and compilation.get("previous_compilation_sha256") is None
            and set(compilation["build_logs"]) == {"MODEL", "DHELAS", "process"}, "Original full source build lacks provenance")
    bench.verify_worker_template(compilation, Path(shared["worker_template"]))
    require(bench.generated_sources(output) == compilation["sources"]
            and bench.digest(directory / "check") == compilation["executable_sha256"], "Original source/executable changed")
    card = output / "Cards/param_card.dat"
    resolved_card = bench.resolved_parameter_card(directory)
    require(resolved_card is not None and bench.digest(card) == generation["parameter_card_sha256"]
            == bench.digest(resolved_card), "Original physical card changed or is shadowed")
    points = Path(bench.REPORT) / "points" / (case_id + ".dat")
    metadata = read(points.with_suffix(".json"))
    require(bench.digest(points) == metadata["sha256"]
            and metadata["inputs"] == {"count": 100, "seed": 20261109, "energy": 1000.,
                "incoming_masses": compilation["masses"][:2], "masses": compilation["masses"][2:]},
            "Original saved points/masses differ")
    dependencies = bench.dependency_inputs(directory)
    require(all(compilation[key] == dependencies for key in ("link_inputs_before", "link_inputs_after")),
            "Original link dependencies changed")
    runtime, evidence = borrowed_runtime(bench, compilation)
    original = {"case": case, "original_output": str(output), "subprocess": directory.name,
        "generation_sha256": bench.digest(output / "generation.json"),
        "compilation_sha256": bench.digest(output / "compilation.json"),
        "executable_sha256": compilation["executable_sha256"], "generated_sources": compilation["sources"],
        "numerical_objects": objects(bench, output), "physical_card_sha256": bench.digest(card),
        "resolved_parameter_card": str(resolved_card.resolve()), "numerical_cards": bench.numerical_card_inputs(output),
        "points_path": str(points.resolve()), "points_sha256": metadata["sha256"],
        "points_metadata_sha256": bench.digest(points.with_suffix(".json")), "shared": shared,
        "runtime": runtime, "runtime_files": runtime_files(bench, runtime),
        "runtime_binding_method": "Expected library provider set borrowed from closed same-source/compiler n1 lane; own four-lepton static archives/metadata; no loader/compiler query",
        "runtime_verified_for_original": False,
        "runtime_verification_required": "Actual four-lepton PT_INTERP --list and live maps must match before timing",
        "historical_ldd_sha256": hashlib.sha256(runtime["ldd"].encode()).hexdigest(),
        "borrowed_runtime_evidence": evidence}
    return {case_id: original}


def verify_clone(bench, output, prepared):
    """Check immutable clone bytes; private numerical cards may be configured."""
    output = Path(output).resolve()
    require(str(output) == prepared["private_output"], "Wrong private output selected")
    directory = subprocess_directory(output)
    require(directory.name == prepared["subprocess"], "Private subprocess changed")
    for name in ("generation", "compilation"):
        require(bench.digest(output / (name + ".json")) == prepared[name + "_sha256"], "Private provenance changed")
    require(bench.digest(directory / "check") == prepared["executable_sha256"]
            and bench.generated_sources(output) == prepared["generated_sources"]
            and objects(bench, output) == prepared["numerical_objects"], "Private source/executable/objects changed")
    actual_card = bench.resolved_parameter_card(directory)
    require(actual_card is not None and bench.digest(actual_card) == prepared["physical_card_sha256"]
            == bench.digest(output / "Cards/param_card.dat"), "Private physical card changed")
    mutable_paths_are_local(output)
    require(set(bench.numerical_card_inputs(output)) == set(prepared["numerical_cards"]), "Private numerical-card locations changed")
    require((output / "lib").is_symlink()
            and (output / "lib").resolve() == (Path(prepared["original_output"]) / "lib").resolve(), "Private library alias changed")
    require(runtime_files(bench, prepared["runtime"]) == prepared["runtime_files"], "Prepared runtime path/bytes changed")
    dependencies = bench.dependency_inputs(directory)
    require(all(dependencies[key] == prepared["runtime"][key] for key in ("archives", "metadata")), "Private link inputs changed")
    return {"status": "verified", "native_queries_performed": False}


def prepare(bench, work, record_dir):
    """Append-once copy of five exports; call only after controller admission."""
    work, record_dir = Path(work).resolve(), Path(record_dir).resolve()
    require(not work.exists() and not (record_dir / "preparation.json").exists()
            and not (record_dir / "preparation-failure.json").exists(), "Preparation destination already exists")
    originals = collect_originals(bench)
    work.mkdir(parents=True, exist_ok=False)
    record_dir.mkdir(parents=True, exist_ok=True)
    try:
        clones = {}
        for case_id in CASES:
            original = Path(originals[case_id]["original_output"])
            for backend in BACKENDS:
                output = work / str(backend) / "outputs/candidate" / case_id
                def ignored(directory, names):
                    return [name for name in names if name in ("build-history", "worker-raw")
                            or (Path(directory) == original and name == "lib")
                            or stat.S_ISFIFO((Path(directory) / name).lstat().st_mode)]
                shutil.copytree(original, output, symlinks=True, ignore=ignored)
                # Shared library bytes are treated as immutable. Never chmod the
                # symlink target, which would mutate the original installation.
                (output / "lib").symlink_to(original / "lib", target_is_directory=True)
                prepared = {**originals[case_id], "private_output": str(output), "backend": backend,
                            "compiled_or_relinked": False, "native_queries_performed": False}
                verify_clone(bench, output, prepared)
                require(bench.numerical_card_inputs(output) == prepared["numerical_cards"], "Initial private numerical card bytes differ")
                clones[case_id + "/" + str(backend)] = prepared
        require(collect_originals(bench) == originals, "Original input changed during private preparation")
        result = {"schema": SCHEMA, "status": "prepared", "campaign_binding": bench.campaign_binding(),
                  "originals": originals, "clones": clones, "clone_count": len(clones),
                  "compiled_or_relinked": False, "native_queries_performed": False,
                  "scratch_or_filter_files_removed": False, "performance_acceptance_eligible": False}
        exclusive(record_dir / "preparation.json", result)
        return result
    except BaseException as error:
        exclusive(record_dir / "preparation-failure.json", {"status": "failed", "error": repr(error),
            "native_queries_performed": False, "previous_attempt_replaced": False})
        raise
