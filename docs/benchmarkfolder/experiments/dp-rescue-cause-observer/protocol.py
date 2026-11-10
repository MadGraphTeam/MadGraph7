#!/usr/bin/env python3
"""Prepare private DP-rescue provenance; never compile, link, or launch workers.

This module uses only file reads until write_snapshot() is explicitly called.
JSON real literals remain strings; target selection uses integer status fields.
Preparation is not permission to perform the later native parity experiment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
CAMPAIGN_ID = "split-dp-terminal-1"
CAMPAIGN = ROOT / "docs/benchmarkfolder/campaigns" / CAMPAIGN_ID
GENERATED = ROOT / ".benchmark-work/tirust/campaigns" / CAMPAIGN_ID / "outputs/candidate"
MANIFEST_SHA256 = "1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31"
DSO_SHA256 = "eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9"
CASES = {
    "qcd_n3_uux_zgg": {
        "subprocess": "P0_uux_zgg",
        "observation_sha256": "4f6dc97b0ff4feaaf38f3e45a847cd4aae183a658dfe74f435d498f5b4427483",
        "rescued_indices": (16, 18, 21, 32, 39, 52, 54, 55, 65, 94, 97, 98, 99),
    },
    "ew_n3_epem_ttxa": {
        "subprocess": "P0_epem_ttxa",
        "observation_sha256": "3dd2905e9c245b28af634ec0e6bec253293c9a18f821811b0ca1023fad4d1bec",
        "rescued_indices": (3, 7, 10, 17, 18, 21, 28, 29, 30, 34, 42, 44, 58, 62, 65, 77, 83, 87),
    },
}
SOURCE_SUFFIXES = {".f", ".f90", ".for", ".inc", ".c", ".cc", ".h"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def reject_constant(value):
        raise ValueError("Non-JSON number: " + value)
    return json.loads(Path(path).read_text(), parse_float=str, parse_constant=reject_constant)


def within(path, directory):
    return path == directory or directory in path.parents


def safe_relative(base, name):
    """Recorded relative names may not select another tree with '..'."""
    require(isinstance(name, str), "Non-string inventory path")
    relative = PurePosixPath(name)
    require(not relative.is_absolute() and name == relative.as_posix()
            and relative.parts and all(part not in (".", "..") for part in relative.parts),
            "Unsafe inventory path: " + name)
    return base.joinpath(*relative.parts)


def file_pin(path, expected=None):
    """Hash bytes and preserve lexical/symlink identity in controlled roots."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, "Unsafe absolute path")
    allowed = (ROOT, Path("/nix/store"))
    resolved = path.resolve(strict=True)
    require(any(within(path, base) for base in allowed)
            and any(within(resolved, base) for base in allowed),
            "Recorded path is outside declared source/runtime roots: " + str(path))
    require(resolved.is_file(), "Expected regular file: " + str(path))
    actual = sha256(path)
    if expected is not None:
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None, "Invalid expected digest")
        require(actual == expected, "Pinned bytes changed: " + str(path))
    return {"path": str(path), "resolved_path": str(resolved),
            "sha256": actual, "size_bytes": resolved.stat().st_size,
            "symlink": str(path.readlink()) if path.is_symlink() else None}


def pin_inventory(base, inventory):
    require(isinstance(inventory, dict) and inventory, "Empty or malformed inventory")
    return {name: file_pin(safe_relative(base, name), checksum)
            for name, checksum in sorted(inventory.items())}


def pin_absolute_inventory(inventory):
    require(isinstance(inventory, dict), "Malformed absolute inventory")
    return {name: file_pin(Path(name), checksum) for name, checksum in sorted(inventory.items())}


def warmup_records(text):
    """Retain real tokens verbatim; parse only call/point/order/status integers."""
    records = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        words = line.split()
        require(len(words) == 9, "Malformed warmup record")
        require(all(re.fullmatch(r"-?[0-9]+", word) for word in words[:4]),
                "Malformed integer warmup header")
        call, point, order, status = map(int, words[:4])
        records.append({"call": call, "point": point, "order": order,
                        "status": status, "real_tokens": words[4:]})
    return records


def point_membership(case_id, metadata, observation, warmup_text):
    """Validate all 100 points; rescue membership is a diagnostic label only."""
    require(case_id in CASES, "Unknown case")
    points = metadata["points"]
    require(len(points) == 100, "Expected the unchanged full-100 point stream")
    require(all(p["kind"] == "ordinary" and p["id"] == f"ordinary-{i:04d}"
                for i, p in enumerate(points)), "Ordinary point metadata/order changed")
    require(observation["measurement_phase"] == "steady-warmup", "Wrong attribution phase")
    require(observation["answer_dimensions"] == {"allocated": 1, "born_orders": 1, "loop_orders": 1},
            "Answer dimensions changed")
    require(observation["warmup_first_call"] == 1 and observation["warmup_last_call"] == 100,
            "Historical warmup interval changed")
    require(observation["warmup_call_point_order"] == [{"call": i, "point": i} for i in range(1, 101)],
            "Historical warmup order changed")
    require(observation["raw_outputs"]["warmup.dat"] == warmup_text,
            "Retained raw warmup differs from observation")
    raw = warmup_records(warmup_text)
    require([(r["call"], r["point"], r["order"]) for r in raw]
            == [(call, max(1, call), order) for call in range(101) for order in (0, 1)],
            "Full bootstrap-plus-100 warmup stream changed")
    rows = observation["rows"]
    require([(r["point"], r["order"]) for r in rows]
            == [(point, order) for point in range(1, 101) for order in (0, 1)],
            "Missing, duplicated, or reordered answer rows")
    require(all(row["status"] == record["status"] for row, record in zip(rows, raw[2:])),
            "JSON and raw warmup statuses disagree")
    rescued = tuple(r["point"] for r in rows if r["order"] == 0 and r["status"] == 323)
    require(rescued == CASES[case_id]["rescued_indices"], "Historical rescue membership changed")
    counts = {"219": 100 - len(rescued), "323": len(rescued)}
    require(observation["status_counts"] == counts, "Historical status counts changed")
    require(all(r["status"] in (219, 323) for r in rows), "Unexpected status")
    require(all(rows[i]["status"] == rows[i + 1]["status"] for i in range(0, 200, 2)),
            "Answer orders disagree on point status")
    return {"point_count": 100, "status_counts": counts,
            "selection_policy": "classify all points; never prune the input or warmup",
            "bootstrap": {"call": 0, "point": 1},
            "warmup_call_point_order": observation["warmup_call_point_order"],
            "points": [{"point_index": i, "point_id": p["id"],
                        "historical_rescued": i in rescued} for i, p in enumerate(points, 1)],
            "rescued_indices": list(rescued),
            "rescued_ids": [points[i - 1]["id"] for i in rescued]}


def prepare_case(case_id, manifest):
    case = CASES[case_id]
    generated = GENERATED / case_id
    subprocess_dir = generated / "SubProcesses" / case["subprocess"]
    evidence = CAMPAIGN / "experiments/numerical" / f"full100-warmup-{case_id}-1"
    retained = evidence / "candidate-9"
    inputs = read_json(retained / "candidate-inputs.json")
    compilation = inputs["compilation"]
    binding = {"id": CAMPAIGN_ID, "manifest_sha256": MANIFEST_SHA256}
    require(inputs["campaign_binding"] == binding and compilation["campaign_binding"] == binding,
            "Candidate inputs belong to a different campaign")
    points_dat = CAMPAIGN / "points" / f"{case_id}.dat"
    points_json = points_dat.with_suffix(".json")
    require(inputs["points_path"] == str(points_dat), "Point path changed")
    require(points_dat.read_text().splitlines()[0].strip() == "100", "Point stream is not full-100")
    points_pins = {"dat": file_pin(points_dat, inputs["points_sha256"]),
                   "metadata": file_pin(points_json, inputs["points_metadata_sha256"])}
    for suffix in ("dat", "json"):
        name = f"points/{case_id}.{suffix}"
        require(manifest["shared_inputs"]["files"][name]
                == inputs["points_sha256" if suffix == "dat" else "points_metadata_sha256"],
                "Point pin does not match campaign")
    observation_path = retained / "raw/observation.json"
    file_pin(observation_path, case["observation_sha256"])
    observation = read_json(observation_path)
    targets = point_membership(case_id, read_json(points_json), observation,
                               (retained / "raw/warmup.dat").read_text())
    sources = pin_inventory(generated, compilation["sources"])
    actual_source_names = {
        p.relative_to(generated).as_posix()
        for tree in ("Source", "SubProcesses") for p in (generated / tree).rglob("*")
        if p.is_file() and (p.suffix.lower() in SOURCE_SUFFIXES or p.name.startswith("makefile")
                           or p.name in ("make_opts", "MadLoop_makefile_definitions"))}
    require(set(sources) == actual_source_names, "Generated source/include inventory changed")
    focus = {name: file_pin(subprocess_dir / name)
             for name in ("loop_matrix.f", "TIR_interface.f", "TIRUST_interface.f", "check_sa.f")}
    require(focus["check_sa.f"]["sha256"] == compilation["driver_sha256"], "Driver bytes changed")
    executable = file_pin(subprocess_dir / "check", compilation["executable_sha256"])
    current_compilation = read_json(generated / "compilation.json")
    require(current_compilation == compilation, "Current and retained compilation records differ")
    require(read_json(generated / "generation.json") == inputs["generation"],
            "Current and retained generation records differ")
    require(compilation["link_inputs_before"] == compilation["link_inputs_after"],
            "Recorded link inputs changed during original compilation")
    cards = pin_inventory(generated, inputs["numerical_cards"])
    cards["Cards/param_card.dat"] = file_pin(generated / "Cards/param_card.dat", inputs["param_card_sha256"])
    objects = {p.relative_to(generated).as_posix(): file_pin(p)
               for tree in ("Source", "SubProcesses", "lib") for p in sorted((generated / tree).rglob("*"))
               if p.is_file() and p.suffix.lower() in (".o", ".a", ".mod")}
    require(objects, "Missing existing numerical objects")
    resources = {p.relative_to(generated).as_posix(): file_pin(p)
                 for p in sorted((generated / "SubProcesses/MadLoop5_resources").rglob("*")) if p.is_file()}
    runtime = {kind: pin_absolute_inventory(inputs["runtime"][kind])
               for kind in ("archives", "libraries", "metadata")}
    link_inputs = {kind: pin_absolute_inventory(compilation["link_inputs_after"][kind])
                   for kind in ("archives", "metadata")}
    evidence_names = ("candidate-inputs.json", "inputs-after.json", "result.json", "candidate-output.txt",
                      "raw/observation.json", "raw/warmup.dat", "raw/dimensions.dat", "raw/ready.json")
    evidence_pins = {name: file_pin(retained / name) for name in evidence_names}
    protocols = {name: file_pin(safe_relative(evidence, name), checksum)
                 for name, checksum in read_json(evidence / "specification.json")["protocols"].items()}
    protocols["worker_template"] = file_pin(evidence / "tirust_benchmark_worker.f",
                                           compilation["worker_template_sha256"])
    return {"targets": targets, "points": points_pins, "generated_root": str(generated),
            "focus_sources": focus, "all_sources_and_includes": sources,
            "existing_numerical_objects": objects, "existing_executable": executable,
            "cards": cards, "initial_resource_inventory": resources,
            "retained_evidence": evidence_pins, "retained_protocols": protocols,
            "specification": file_pin(evidence / "specification.json"),
            "campaign_manifest_copy": file_pin(evidence / "campaign-manifest.json", MANIFEST_SHA256),
            "generation": file_pin(generated / "generation.json"),
            "compilation": file_pin(generated / "compilation.json"),
            "effective_compiler_flags": compilation["effective_compiler_flags"],
            "historical_process_build_command": compilation["process_build_command"],
            "build_settings": compilation["settings"], "numerical_settings": inputs["settings"],
            "link_inputs": link_inputs, "runtime": runtime,
            "recorded_ldd_text_only": inputs["runtime"]["ldd"]}


def prepare_snapshot():
    require(Path.cwd().resolve() == ROOT, "Run only from the MadGraph repository root")
    manifest_pin = file_pin(CAMPAIGN / "manifest.json", MANIFEST_SHA256)
    manifest = read_json(CAMPAIGN / "manifest.json")
    require(manifest["id"] == CAMPAIGN_ID, "Unexpected campaign")
    native = manifest["candidate"]["native"]
    prefix = ROOT / ".tirust-build/qp-transport-install/mg/HEPTools/tirust"
    require(native["prefix"] == str(prefix), "Native prefix changed")
    require(native["files"]["lib/libmadloop_tirust.so"] == DSO_SHA256, "Native DSO identity changed")
    installed = pin_inventory(prefix, native["files"])
    for name in ("lib/mtirust.mod", "install.log", "lib/tirust-dependency-provenance.json",
                 "lib/tirust-dependency-notices.tar.gz"):
        installed[name] = file_pin(prefix / name)
    saved_numerical_sources = pin_inventory(safe_relative(CAMPAIGN, manifest["candidate"]["source_snapshot"]),
        {name: entry["sha256"] for name, entry in manifest["candidate"]["numerical_sources"].items()})
    private_sources = {p.relative_to(HERE).as_posix(): file_pin(p)
                       for p in sorted(HERE.rglob("*"))
                       if p.is_file() and p.relative_to(HERE).parts[0] != "preparation"
                       and p.relative_to(HERE).as_posix() != "index.json"
                       and "__pycache__" not in p.relative_to(HERE).parts}
    return {"schema": "dp-rescue-cause-preparation-v1", "execution_authorized": False,
            "build_authorized": False, "performance_eligible": False,
            "manifest": manifest_pin, "installed_native": installed,
            "saved_numerical_sources": saved_numerical_sources, "private_sources": private_sources,
            "cases": {case_id: prepare_case(case_id, manifest) for case_id in CASES},
            "limitations": ["Metadata-only snapshot; source bytes must be copied and reverified before a later build.",
                            "Current resource inventory does not authorize reusing warm filters in a fresh process.",
                            "Recorded runtime dependencies are not proof of actual later loaded mappings.",
                            "No compiler, linker, worker, parser test, or parity run is performed."]}


def write_snapshot(label, snapshot):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", label) is not None, "Unsafe label")
    parent = HERE / "preparation"
    require(not parent.is_symlink(), "Preparation directory must not be a symlink")
    parent.mkdir(exist_ok=True)
    require(parent.resolve() == parent, "Preparation directory escaped private namespace")
    destination = parent / label
    destination.mkdir(exist_ok=False)
    output = destination / "snapshot.json"
    with output.open("x") as stream:
        json.dump(snapshot, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="New private preparation label; existing labels are refused")
    arguments = parser.parse_args()
    output = write_snapshot(arguments.label, prepare_snapshot())
    print(output)


if __name__ == "__main__":
    main()
