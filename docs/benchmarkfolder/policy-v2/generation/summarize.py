#!/usr/bin/env python3
"""Index the retained fresh exports without rebuilding or evaluating them."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "madgraph/__init__.py").is_file())


def load(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    status = load(HERE / "status.json")
    if not all(status.get(key) is True for key in
               ("complete", "preserved_existing_inputs", "dependencies_unchanged")):
        raise RuntimeError("Generation, input preservation or dependency check failed")
    if sha256(HERE / "regenerate.py") != status["script_sha256"]:
        raise RuntimeError("Generation script differs from the executed source")
    protected = load(HERE / "protected-inputs-before.json")
    cases = {}
    for case_id, record in status["cases"].items():
        if record["status"] != "complete":
            raise RuntimeError(f"Incomplete case: {case_id}")
        before = protected["original_sources"][case_id]
        after_path = HERE / "inputs/candidate" / case_id / "compiled-sources.json"
        after = load(after_path)
        if after["files_sha256"] != record["compilation"]["generated_sources_sha256"]:
            raise RuntimeError(f"Compiled manifest mismatch: {case_id}")
        changed = sorted(name for name in before["files"].keys() | after["files"].keys()
                         if before["files"].get(name) != after["files"].get(name))
        dimensions = list(map(int, record["dimensions"].split()))
        cases[case_id] = {
            "status": record["status"],
            "process": record["case"]["process"],
            "model": record["case"]["model"],
            "output": record["output"],
            "generation_seconds": record["generation"]["seconds"],
            "compilation_seconds": record["compilation"]["seconds"],
            "build_kind": record["compilation"]["build_kind"],
            "preexisting_subprocess_objects":
                record["compilation"]["preexisting_subprocess_objects"],
            "generation_revision": record["generation"]["source_identity"]["revision"],
            "generation_tree_state_sha256":
                record["generation"]["source_identity"]["tree_state_sha256"],
            "executable_sha256": record["compilation"]["executable_sha256"],
            "compiled_sources_sha256": after["files_sha256"],
            "compiled_manifest": str(after_path.relative_to(ROOT)),
            "parameter_card_sha256": record["physical_inputs"]["param_card"]["sha256"],
            "points_sha256": record["physical_inputs"]["points"]["sha256"],
            "masses": record["masses"],
            "answer_dimensions": dict(zip(("allocated", "loop_orders", "born_orders"),
                                         dimensions)),
            "model_delta": record["model_delta"],
            "changed_model_files": record["changed_model_files"],
            "other_generated_source_changes":
                [name for name in changed if not name.startswith("Source/MODEL/")],
            "loop_amplitude_evaluations": record["loop_amplitude_evaluations"],
            "initialized_symbolica_workers": record["initialized_symbolica_workers"],
        }
    result = {
        "schema_version": 1,
        "complete": True,
        "core_case_count": len(load(HERE / "cases.json")["cases"]),
        "control_case_count": len(load(HERE / "controls.json")["cases"]),
        "source_revision": status["git_revision"],
        "generator_tree_state_sha256": status["generator_tree_state_sha256"],
        "preserved_existing_inputs": True,
        "dependencies_unchanged": True,
        "status": {"path": str((HERE / "status.json").relative_to(ROOT)),
                   "sha256": sha256(HERE / "status.json")},
        "generation_script_sha256": status["script_sha256"],
        "summary_script_sha256": sha256(Path(__file__)),
        "cases": cases,
        "timing_scope": "Concurrent CLI generation and clean subprocess builds; "
                        "Source-library compilation is part of generation. "
                        "No amplitude/runtime measurements.",
    }
    (HERE / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Indexed {len(cases)} complete exports; original inputs unchanged.")


if __name__ == "__main__":
    main()
