#!/usr/bin/env python3
"""Hash original inputs and prepare one point; never compile or run a worker.

Manual invocation writes only this proposal's fresh preparation/LABEL directory.
The output is a source/build-input receipt, not authorization or native evidence.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import struct
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAMPAIGN = ROOT / "docs/benchmarkfolder/campaigns/split-dp-terminal-1"
CASE = "qcd_n4_uux_zzgg"
SUBPROCESS = "P0_uux_zzgg"
GENERATED = ROOT / ".benchmark-work/tirust/campaigns/split-dp-terminal-1/outputs/candidate" / CASE
INPUTS = CAMPAIGN / "inputs/candidate" / CASE
SPECIFICATION = CAMPAIGN / "numerical-qualification-2/specification.json"
SPECIFICATION_SHA = "fe2e9a491573d33aa417b3ce548fa6b66cf52721f56d51644782f60711bfc02f"
MANIFEST_SHA = "1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31"
COMPILATION_SHA = "196c05615db72da5ce92902eb313ec8a42e5e3f1440fd482a94254ee859780f0"
GENERATION_SHA = "3a3ddde735e0e8c26494ba831ac3c7360c4745a048540d6088f96cc0e8959775"
EXECUTABLE_SHA = "36605c1543452982ad6b9a0cec741ce99aa057c9e1973ddb119d315881e12e0b"
POINTS_SHA = "d2f5bd557aff1eb2eea56237447d46c91271bd0b5c063cbd8eb6060167cd0f8e"
POINTS_JSON_SHA = "4362b5144dd63d63275676f84071e460f048723f47c8d9d4e8a1c770183b7d63"
ONE_POINT_SHA = "c811e4bb36b5f33525157c40ee5487858e1e2d7841c87101b0aab930530a6c3c"
DSO_SHA = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
RETAINED_RUNTIME = CAMPAIGN / f"experiments/numerical/full100-warmup-{CASE}-1/candidate-8/candidate-inputs.json"
RUNTIME_RECORD_SHA = "f4f04cb6fe973231e7b2bf43ec66af9c68a192fbed7b6cd32ce485f04794afcb"
OBJECT_RECORD = ROOT / "docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/build-plan.json"
OBJECT_RECORD_SHA = "69333b95bd0eff677948ac9c75d311989e3bc74d4df8f915304456e087334047"
TOOLCHAIN_RECORD = ROOT / "docs/benchmarkfolder/experiments/dp-rescue-cause-observer/controller/build-plan.json"
TOOLCHAIN_RECORD_SHA = "20137ff143ca65980df1e53d487294d2677a886c5de7eb0d573eb827994b9b7c"
CARD_SHA = "5300bd6d7f3c4553c3402ab7a044e8fbca3c646f451a5f586215933bfa113b85"
PARAM_SHA = "5a535b7044777bd87c7088dc14bf9605d165ebae843dbcbde91a02f545c05433"
IDENT_SHA = "a39f898e29da6781c365a7946f64d0c9fe1b2ba096137295b638aee0c3dc2493"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def within(path, root):
    return path == root or root in path.parents


def relative(root, name):
    require(isinstance(name, str), "Relative filename must be text")
    part = PurePosixPath(name)
    require(not part.is_absolute() and part.parts and ".." not in part.parts
            and name == part.as_posix(), "Unsafe relative filename")
    return root.joinpath(*part.parts)


def controlled(path):
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Unsafe absolute input path")
    resolved = path.resolve(strict=True)
    roots = (ROOT, Path("/nix/store"))
    require(any(within(path, root) for root in roots)
            and any(within(resolved, root) for root in roots), "Input escaped repository/Nix roots")
    require(resolved.is_file(), "Input is not a regular file")
    return path


def pin(path, expected=None):
    path = controlled(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    value = digest.hexdigest()
    require(expected is None or value == expected, "Changed frozen input: " + str(path))
    return {"path": str(path), "resolved_path": str(path.resolve()), "sha256": value,
            "size_bytes": path.stat().st_size,
            "symlink": str(path.readlink()) if path.is_symlink() else None}


def read(path, expected=None):
    record = pin(path, expected)
    return json.loads(Path(path).read_text()), record


def absolute_pins(mapping):
    return {name: pin(Path(name), digest) for name, digest in sorted(mapping.items())}


def interpreter(path):
    """Read PT_INTERP from ELF bytes without executing readelf or a loader."""
    data = controlled(path).read_bytes()
    require(data[:6] == b"\x7fELF\x02\x01", "Expected little-endian ELF64")
    offset = struct.unpack_from("<Q", data, 32)[0]
    size, count = struct.unpack_from("<HH", data, 54)
    require(size >= 56 and count <= 4096 and offset + size * count <= len(data), "Malformed ELF program headers")
    names = []
    for index in range(count):
        header = offset + index * size
        if struct.unpack_from("<I", data, header)[0] == 3:
            start = struct.unpack_from("<Q", data, header + 8)[0]
            length = struct.unpack_from("<Q", data, header + 32)[0]
            require(1 < length <= 4096 and start + length <= len(data), "Malformed PT_INTERP")
            value = data[start:start + length]
            require(value[-1:] == b"\0" and b"\0" not in value[:-1], "Malformed interpreter filename")
            names.append(Path(value[:-1].decode("ascii")))
    require(len(names) == 1, "Expected exactly one ELF interpreter")
    return pin(names[0])


def card_values(path):
    values, key = {}, None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("#"):
            key = line[1:].strip()
        elif line and not line.startswith("!") and key is not None:
            values[key] = line
            key = None
    return values


def prepare():
    require(Path.cwd().resolve() == ROOT, "Run from the MadGraph repository root")
    specification, spec_pin = read(SPECIFICATION, SPECIFICATION_SHA)
    manifest, manifest_pin = read(CAMPAIGN / "manifest.json", MANIFEST_SHA)
    compilation, comp_pin = read(INPUTS / "compilation.json", COMPILATION_SHA)
    generation, gen_pin = read(INPUTS / "generation.json", GENERATION_SHA)
    compiled = specification["compiled_inputs"]["candidate/" + CASE]
    require(compiled == {"compilation_sha256": COMPILATION_SHA, "generation_sha256": GENERATION_SHA,
                        "executable_sha256": EXECUTABLE_SHA, "points_sha256": POINTS_SHA,
                        "points_metadata_sha256": POINTS_JSON_SHA}, "Registered compiled-input identity differs")
    require(generation["campaign_binding"]["manifest_sha256"] == MANIFEST_SHA
            and compilation["campaign_binding"]["manifest_sha256"] == MANIFEST_SHA, "Campaign binding differs")
    sources = {name: pin(relative(GENERATED, name), digest) for name, digest in compilation["sources"].items()}
    objects = {p.relative_to(GENERATED).as_posix(): pin(p) for p in sorted(GENERATED.rglob("*.o"))}
    modules = {p.relative_to(GENERATED).as_posix(): pin(p) for p in sorted(GENERATED.rglob("*.mod"))}
    require(objects and modules, "Original numerical objects/modules absent")
    object_record, object_pin = read(OBJECT_RECORD, OBJECT_RECORD_SHA)
    frozen_objects = {}
    for name in object_record["models"]["qcd"]["reuse_objects"]:
        path = Path(name)
        require(within(path, GENERATED) and path.suffix == ".o", "Unrelated frozen object")
        record_name = path.relative_to(ROOT).as_posix()
        frozen_objects[path.relative_to(GENERATED).as_posix()] = pin(path, object_record["dependencies"][record_name])
    require(len(frozen_objects) == 24, "Unexpected existing frozen numerical-object set")
    executable = pin(GENERATED / "SubProcesses" / SUBPROCESS / "check", EXECUTABLE_SHA)
    original_interpreter = interpreter(Path(executable["path"]))
    require(compilation["link_inputs_before"] == compilation["link_inputs_after"], "Original link-input records differ")
    link_inputs = {kind: absolute_pins(values) for kind, values in compilation["link_inputs_after"].items()}
    build_logs = {kind: pin(INPUTS / f"compilation-{kind}.log", digest)
                  for kind, digest in compilation["build_logs"].items()}
    lines = Path(build_logs["process"]["path"]).read_text().splitlines()
    links = [shlex.split(line) for line in lines if " -o check " in line]
    require(len(links) == 1, "Original link command is absent or ambiguous")
    compile_lines = compilation["effective_compiler_flags"]["scopes"]["process"]["expanded_make_output"].splitlines()
    recipes = {}
    tools = {}
    for scope in compilation["effective_compiler_flags"]["scopes"].values():
        for recipe in scope["recipes"].values():
            for tool in recipe["compiler"]:
                tools[tool["path"]] = pin(Path(tool["path"]), tool["sha256"])
    for name in ("check_sa.f", "loop_matrix.f", "TIRUST_interface.f", "TIR_interface.f"):
        commands = [shlex.split(line) for line in compile_lines if " -c " + name + " " in line]
        require(len(commands) == 1, "Original source compile command is absent or ambiguous: " + name)
        recipes[name] = commands[0]
    toolchain, toolchain_pin = read(TOOLCHAIN_RECORD, TOOLCHAIN_RECORD_SHA)
    toolchain_links = {name: pin(Path(value["path"]), value["sha256"])
                       for name, value in toolchain["toolchain_link_inputs"].items()}
    fortran_prefix = Path(recipes["check_sa.f"][0]).parent.parent
    compiler_support = {name: pin(Path(value["path"]), value["sha256"])
                        for name, value in toolchain["compiler_support_files"].items()
                        if within(Path(name), fortran_prefix)}
    require(compiler_support, "Frozen Fortran wrapper support files absent")
    include_dirs = set()
    for command in recipes.values():
        for index, argument in enumerate(command):
            if argument == "-I":
                require(index + 1 < len(command), "Missing compiler include directory")
                value = command[index + 1]
            elif argument.startswith("-I"):
                value = argument[2:]
            else:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = GENERATED / "SubProcesses" / SUBPROCESS / path
            path = path.resolve(strict=True)
            require(path.is_dir() and (within(path, ROOT) or within(path, Path("/nix/store"))),
                    "Compiler include directory escaped declared roots")
            include_dirs.add(path)
    external_modules = {str(path): pin(path) for directory in sorted(include_dirs)
                        for path in sorted(directory.glob("*.mod")) if not within(path, GENERATED)}
    require({"mtirust.mod", "mninja.mod", "mninjars.mod", "collier.mod"}
            <= {Path(path).name for path in external_modules}, "Required external Fortran module missing")
    native = manifest["candidate"]["native"]
    native_prefix = Path(native["prefix"])
    native_files = {name: pin(relative(native_prefix, name), digest) for name, digest in native["files"].items()}
    for path in sorted((native_prefix / "lib").glob("*.mod")):
        native_files[path.relative_to(native_prefix).as_posix()] = pin(path)
    require(native_files["lib/libmadloop_tirust.so"]["sha256"] == DSO_SHA, "Wrong installed native DSO")
    runtime, runtime_pin = read(RETAINED_RUNTIME, RUNTIME_RECORD_SHA)
    libraries = absolute_pins(runtime["runtime"]["libraries"])
    require(libraries[original_interpreter["path"]]["sha256"] == original_interpreter["sha256"],
            "Actual ELF interpreter is not in the retained closure")
    expected_runtime = {name: record for name, record in libraries.items()
                        if Path(name).name != "ld-linux-x86-64.so.2" or name == original_interpreter["path"]}
    cards = {"Cards/MadLoopParams.dat": pin(GENERATED / "Cards/MadLoopParams.dat", CARD_SHA),
             "Cards/param_card.dat": pin(GENERATED / "Cards/param_card.dat", PARAM_SHA),
             "Cards/ident_card.dat": pin(GENERATED / "Cards/ident_card.dat", IDENT_SHA)}
    resources = {name: pin(GENERATED / "SubProcesses/MadLoop5_resources" / name)
                 for name in ("ML5_0_ColorDenomFactors.dat", "ML5_0_ColorNumFactors.dat", "ML5_0_HelConfigs.dat",
                              "MadLoopParams.dat", "param_card.dat", "ident_card.dat")}
    for name in ("MadLoopParams.dat", "param_card.dat", "ident_card.dat"):
        require(resources[name]["sha256"] == cards["Cards/" + name]["sha256"], "Resource/card identity differs")
    point_path = CAMPAIGN / "points" / (CASE + ".dat")
    point_pin = pin(point_path, POINTS_SHA)
    point_json_path = CAMPAIGN / "points" / (CASE + ".json")
    point_json_pin = pin(point_json_path, POINTS_JSON_SHA)
    # Do not parse or reformat momentum decimals. The exact six original rows
    # are transported unchanged, with only the declared point-count header new.
    rows = point_path.read_bytes().splitlines(keepends=True)
    require(len(rows) == 601 and rows[0] == b"100\n" and all(len(row.split()) == 4 for row in rows[1:]),
            "Unexpected registered point stream")
    point_bytes = b"1\n" + b"".join(rows[1:7])
    require(hashlib.sha256(point_bytes).hexdigest() == ONE_POINT_SHA, "Selected point bytes differ")
    point_json = json.loads(point_json_path.read_text(), parse_float=str)
    require(len(point_json["points"]) == 100 and point_json["points"][0]["id"] == "ordinary-0000",
            "Registered point membership differs")
    private = {p.relative_to(HERE).as_posix(): pin(p) for p in sorted(HERE.rglob("*"))
               if p.is_file() and not p.is_symlink() and p.name != "index.json"
               and not p.name.endswith((".tmp", ".pyc"))
               and not {"preparation", "__pycache__", ".pytest_cache"}.intersection(p.relative_to(HERE).parts)}
    for name in ("prepare.py", "build.py", "transform.py", "driver.f.in", "observer.f90"):
        require(name in private, "Required private proposal source absent: " + name)
    spec = importlib.util.spec_from_file_location("occupancy_source_transform", HERE / "transform.py")
    transform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transform)
    adapter_name = "SubProcesses/" + SUBPROCESS + "/TIRUST_interface.f"
    original = Path(sources[adapter_name]["path"]).read_bytes()
    observed = transform.transform(original.decode("ascii")).encode("ascii")
    rendered = {"original/TIRUST_interface.f": original, "observed/TIRUST_interface.f": observed,
                "common/driver.f": (HERE / "driver.f.in").read_bytes(),
                "common/observer.f90": (HERE / "observer.f90").read_bytes(),
                "observed/TIRUST_interface.patch": "".join(difflib.unified_diff(
                    original.decode("ascii").splitlines(keepends=True), observed.decode("ascii").splitlines(keepends=True),
                    fromfile="original/TIRUST_interface.f", tofile="observed/TIRUST_interface.f")).encode("ascii")}
    return {"schema": "qcd-n4-cache-occupancy-preparation-v1", "source_only": True,
            "case": CASE, "subprocess": SUBPROCESS, "generated_root": str(GENERATED),
            "compiled_inputs": compiled, "specification": spec_pin, "campaign_manifest": manifest_pin,
            "compilation": comp_pin, "generation": gen_pin, "executable": executable,
            "sources": sources, "objects": objects, "modules": modules,
            "existing_frozen_objects": frozen_objects, "existing_object_record": object_pin,
            "link_inputs": link_inputs, "build_logs": build_logs, "compile_argv": recipes,
            "link_argv": links[0], "effective_compiler_flags": compilation["effective_compiler_flags"],
            "historical_process_build_command": compilation["process_build_command"], "tools": tools,
            "toolchain_record": toolchain_pin, "toolchain_link_inputs": toolchain_links,
            "compiler_support_files": compiler_support,
            "include_directories": [str(path) for path in sorted(include_dirs)], "external_modules": external_modules,
            "installed_native": native_files, "interpreter": original_interpreter,
            "retained_runtime": {"record": runtime_pin, "libraries": libraries,
                                 "ldd_text_historical_only": runtime["runtime"]["ldd"],
                                 "expected_elf_closure": expected_runtime, "loaded_runtime_proven": False},
            "cards": cards, "initial_resources": resources, "numerical_settings": card_values(GENERATED / "Cards/MadLoopParams.dat"),
            "point": {"parent_dat": point_pin, "parent_metadata": point_json_pin, "index_zero_based": 0,
                      "id": "ordinary-0000", "source_lines_one_based": [2, 7], "dat_sha256": ONE_POINT_SHA},
            "protocol": {"public_calls": [{"sequence": 0, "point": 1, "phase": 0},
                                          {"sequence": 1, "point": 1, "phase": 1}],
                         "input_words": 25, "input_words_after": 25, "answer_words": 8, "accuracy_words": 2,
                         "arms": ["original", "disabled", "enabled"], "serial": True,
                         "warmup_retry_count": 0, "additional_points": 0, "timing_loops": 0,
                         "all_public_raw_words_and_entry_exit_ieee_states_must_match": True,
                         "internal_amplitude_call_count_is_not_public_call_count": True},
            "build_proposal": {"cpu": 12, "jobs": 1, "seconds_cap": 180,
                               "rust_build": False, "numerical_library_changes": False,
                               "original_numerical_inputs_unchanged": True,
                               "execution_requires_separate_root_instruction": True,
                               "execution_shared_lock": str(ROOT / ".tirust-build/numerical-tests.lock"),
                               "disabled_enabled_same_executable": True},
            "private_sources": private, "native_executed": False,
            "runtime_provenance_validated": False, "performance_eligible": False}, point_bytes, rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    require(re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,40}", args.label), "Unsafe preparation label")
    snapshot, point_bytes, rendered = prepare()
    destination = HERE / "preparation" / args.label
    require(destination.resolve() == destination and not destination.exists(), "Preparation path already exists or aliases another path")
    destination.mkdir(parents=True, exist_ok=False)
    with (destination / "points.dat").open("xb") as stream:
        stream.write(point_bytes)
    snapshot["point"]["dat"] = pin(destination / "points.dat", ONE_POINT_SHA)
    snapshot["rendered"] = {}
    for name, contents in rendered.items():
        path = relative(destination, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(contents)
        snapshot["rendered"][name] = pin(path, hashlib.sha256(contents).hexdigest())
    with (destination / "snapshot.json").open("x") as stream:
        json.dump(snapshot, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"snapshot": pin(destination / "snapshot.json"), "source_only": True}, sort_keys=True))


if __name__ == "__main__":
    main()
