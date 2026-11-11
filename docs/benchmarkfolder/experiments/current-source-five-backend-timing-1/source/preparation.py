"""Pure file admission and private export copies; no compiler/loader queries.

Importing this module does nothing. The coordinating controller admits prepare
explicitly. Real worker startup retains the standard compiler/runtime checks.
"""
import hashlib
import json
from pathlib import Path
import shutil
import stat

CASES = ("qcd_n1_uux_z", "qcd_n2_gg_ttx", "ew_n2_epem_mupmum")
BACKENDS = (1, 6, 7, 8, 9)
SCHEMA = "madloop-current-source-five-backend-private-preparation-v1"
MODEL = "loop_qcd_qed_sm_Gmu"
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
    manifest_path = Path(bench.REPORT) / "manifest.json"
    manifest = read(manifest_path)
    binding = bench.campaign_binding()
    require(binding is not None and binding["manifest_sha256"] == bench.digest(manifest_path)
            and manifest == bench.CAMPAIGN_MANIFEST, "Current campaign manifest differs")
    source = manifest["candidate"]["exporter_identity"]
    for name, checksum in source["files"].items():
        path = inside(bench.ROOT, name)
        # Match source_identity's recorded absence for tracked deleted files.
        # A newly present file must never pass an expected-null inventory slot.
        actual = bench.digest(path) if path.is_file() else None
        require(actual == checksum, "Registered current source changed: " + name)
    canonical = hashlib.sha256(json.dumps(source["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(canonical == source["files_sha256"], "Registered source inventory digest differs")
    model = Path(bench.MODEL_ROOT) / MODEL
    actual = {}
    for path in sorted(model.rglob("*")):
        require(not path.is_symlink(), "Pinned model contains a symlink")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix not in (".pyc", ".pkl", ".log"):
            actual[path.relative_to(model).as_posix()] = bench.digest(path)
    require(actual == manifest["shared_inputs"]["model_tree"], "Pinned model bytes changed")
    template = Path(bench.ROOT) / "tools/tirust_benchmark_worker.f"
    return {"campaign_binding": binding, "manifest_sha256": bench.digest(manifest_path),
            "source_identity": source, "model_root": str(model.resolve()), "model_files": actual,
            "worker_template": str(template.resolve()), "worker_template_sha256": bench.digest(template)}


def collect_originals(bench):
    """Read-only inventory; call again after timing to guard original cards too."""
    shared = shared_bindings(bench)
    config = bench.load_config()
    selected = {case["id"]: case for case in config["cases"] if case["id"] in CASES}
    require(set(selected) == set(CASES), "The three registered timing cases are missing")
    originals = {}
    for case_id in CASES:
        case = selected[case_id]
        output = Path(bench.output_dir("candidate", case)).resolve()
        directory = subprocess_directory(output)
        generation, compilation = read(output / "generation.json"), read(output / "compilation.json")
        require(generation["source"] == "candidate" and generation["case"] == case
                and all(item.get("campaign_binding") == shared["campaign_binding"] for item in (generation, compilation)),
                "Original generation/build belongs to another campaign/case")
        require(all(generation["source_identity"][key] == shared["source_identity"][key]
                    for key in ("files", "files_sha256")), "Original exporter differs from current registered source")
        require(compilation.get("numerical_build_from_sources") is True, "Original numerical build lacks provenance")
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
                and metadata["inputs"]["count"] == config["points_per_case"]
                and metadata["inputs"]["incoming_masses"] + metadata["inputs"]["masses"] == compilation["masses"],
                "Original saved points/masses differ")
        for path in (points, points.with_suffix(".json")):
            relative = path.relative_to(bench.REPORT).as_posix()
            require(bench.digest(path) == bench.CAMPAIGN_MANIFEST["shared_inputs"]["files"][relative],
                    "Saved points differ from registered manifest")
        closed = Path(bench.REPORT) / "experiments/numerical" / ("full100-warmup-" + case_id + "-1") / "candidate-9"
        inputs, after, result = (read(closed / name) for name in ("candidate-inputs.json", "inputs-after.json", "result.json"))
        require(result.get("status") == "complete" and result.get("source") == "candidate" and result.get("backend") == 9,
                "Closed runtime observation is unavailable")
        require(inputs["generation"] == generation and inputs["compilation"] == compilation
                and inputs["campaign_binding"] == shared["campaign_binding"]
                and inputs["param_card_sha256"] == bench.digest(card)
                and inputs["points_sha256"] == metadata["sha256"]
                and inputs["points_metadata_sha256"] == bench.digest(points.with_suffix(".json")),
                "Closed runtime observation does not bind this original build/input")
        runtime = inputs["runtime"]
        require(all(after["runtime"][key] == runtime[key] for key in ("libraries", "archives", "metadata"))
                and after["executable_sha256"] == compilation["executable_sha256"]
                and after["points_sha256"] == metadata["sha256"]
                and after["numerical_cards"] == inputs["numerical_cards"], "Closed runtime postcheck differs")
        dependencies = bench.dependency_inputs(directory)
        require(all(compilation[key] == dependencies for key in ("link_inputs_before", "link_inputs_after"))
                and all(runtime[key] == dependencies[key] for key in ("archives", "metadata")), "Original link dependencies changed")
        originals[case_id] = {"case": case, "original_output": str(output), "subprocess": directory.name,
            "generation_sha256": bench.digest(output / "generation.json"),
            "compilation_sha256": bench.digest(output / "compilation.json"),
            "executable_sha256": compilation["executable_sha256"], "generated_sources": compilation["sources"],
            "numerical_objects": objects(bench, output), "physical_card_sha256": bench.digest(card),
            "resolved_parameter_card": str(resolved_card.resolve()), "numerical_cards": bench.numerical_card_inputs(output),
            "points_path": str(points.resolve()), "points_sha256": metadata["sha256"],
            "points_metadata_sha256": bench.digest(points.with_suffix(".json")), "shared": shared,
            "runtime": runtime, "runtime_files": runtime_files(bench, runtime),
            "runtime_binding_method": "Current file hashes joined to closed candidate-9 runtime; no loader/compiler query",
            "historical_ldd_sha256": hashlib.sha256(runtime["ldd"].encode()).hexdigest(),
            "closed_runtime_evidence": {name: {"path": str((closed / name).resolve()), "sha256": bench.digest(closed / name)}
                for name in ("candidate-inputs.json", "inputs-after.json", "result.json")}}
    return originals


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
    """Append-once copy of 15 exports; call only after controller admission."""
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
