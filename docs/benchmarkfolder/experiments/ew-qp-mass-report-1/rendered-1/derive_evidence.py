#!/usr/bin/env python3
"""Read only two published EW capsules; derive report data without native work."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import types

PACKETS = [
    {"label": "Ordinary EW exact-control qualification",
     "path": "experiments/fortran-qp-mass-ew-qualification-1", "index_name": "index.json",
     "index_sha256": "48a8b22cdb652cb173ebe2714aab7cf8efee07ec5793b1a30ceedfabdec6b04a"},
    {"label": "EW threshold refusal in both arms",
     "path": "experiments/fortran-qp-mass-ew-threshold-qualification-1", "index_name": "index.json",
     "index_sha256": "df6f4e453f5787eb31d5bfd42d80030af61ce5cd36a1f9965e5e2a919337df15"},
]
STYLE_SHA = "68e21877f61f92de75525e8d088b7de3fc00d8223b3aef393babc6015b59255e"
PRODUCTION = "0fcd2cd873d61d134837e1d9e87afb13a47824fd"


def require(value, message):
    if not value:
        raise ValueError(message)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def identity(raw):
    return {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def read_json(raw):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "Duplicate JSON key")
            out[key] = value
        return out
    def bad(value):
        raise ValueError("Nonfinite JSON: " + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)


def contained(root, name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and name and str(path) == name and not path.is_absolute()
            and ".." not in path.parts and "\\" not in name, "Unsafe capsule path")
    result = root
    for part in path.parts:
        result /= part
        require(not result.is_symlink(), "Symlink capsule member")
    return result


def replay(root, selection):
    folder = contained(root, selection["path"])
    index_raw = (folder / "index.json").read_bytes()
    require(identity(index_raw)["sha256"] == selection["index_sha256"], "Scientific capsule changed")
    index = read_json(index_raw)
    # Authenticate executable Python before importing this already reviewed,
    # file-only capsule replayer. Its own guards authenticate/replay all members.
    raw = contained(folder, "replay.py").read_bytes()
    require(identity(raw) == index["files"]["replay.py"], "Capsule replay source changed")
    module = types.ModuleType("EW_report_" + selection["index_sha256"][:8])
    module.__file__ = str(folder / "replay.py")
    exec(compile(raw, module.__file__, "exec"), module.__dict__)
    facts = module.verify(folder, selection["index_sha256"])
    require(facts["verified"] is True and all(facts[k] == v for k, v in index["claims"].items()),
            "Published facts disagree with actual contained replay")
    return facts


def assemble(ordinary, threshold):
    require(ordinary["ordinary_physical_points"] == ordinary["accepted_DP219_points_each"] == 100
            and ordinary["ordinary_workers"] == 2 and ordinary["accepted_QP_points_each"] == 0
            and ordinary["all_observed_words_status_history_IEEE_software_equal"] is True
            and ordinary["changed_QP_mass_path_observed"] is False
            and ordinary["pair_regression_tolerance"] == 3e-6 and ordinary["pair_target_tolerance"] == 1e-8
            and ordinary["independent_absolute_accuracy_qualified"] is False,
            "Ordinary facts exceed the DP exact-control scope")
    require(threshold["threshold_refused420_points_each"] == 1
            and threshold["accepted_control_DP219_points_each"] == 1
            and threshold["accepted_QP_points_each"] == 0
            and threshold["failed_buffer_word_differences_diagnostic_only"] == 4
            and threshold["threshold_numerical_comparison"] is None
            and threshold["stored_filters_learned"] is False
            and threshold["mass_fix_resolved_EW_stress"] is False
            and threshold["first_pre_READY_failure_preserved"] is True,
            "Threshold refusal or full-sum initialization scope changed")
    for facts in (ordinary, threshold):
        require(facts["performance_acceptance"] is False and facts["new_native_executions"] == 0,
                "Unexpected performance or new-evaluation claim")
    require(ordinary["production_promotion"] is False and threshold["source_promoted"] is False,
            "Unexpected source promotion")
    data = {
        "schema": "madloop-tirust-ew-qualification-report-data-v1",
        "process": "e+ e- > t t~ a [virt=QED]", "model": "loop_qcd_qed_sm_Gmu",
        "backend": 9, "production_revision": PRODUCTION,
        "ordinary": ordinary, "threshold": threshold,
        "interpretation": {
            "LoopPoleMode": 0, "persistent_filter_learning_qualified": False,
            "warmup_policy": "Full sums; stored filter state restored after each call",
            "threshold_status_each": 420, "threshold_reported_accuracy_each": 1,
            "threshold_accepted_coefficient_comparison": None,
            "ordinary_pair_difference": 0,
            "duplicate_overall_and_sole_order_rows_are_independent_points": False,
            "ordinary_unobserved_internal_QP_attempts_excluded": False,
            "production_mass_patch_promoted": False, "independent_physical_truth": False,
            "performance_acceptance": False, "new_native_physics_evaluations": 0,
        },
        "evidence": PACKETS,
    }
    selection = {
        "schema": "madloop-tirust-ew-qualification-report-selection-v1",
        "production_revision": PRODUCTION, "packets": PACKETS, "style_sha256": STYLE_SHA,
        "production_source_promotion": False, "source_promoted": False,
        "performance_acceptance": False, "new_native_physics_evaluations": 0,
        "persistent_filter_learning_qualified": False,
        "threshold_accepted_coefficient_comparison": None,
    }
    return data, selection


def derive(root):
    return assemble(*(replay(Path(root), item) for item in PACKETS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path, help="Published benchmark root")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = dict(zip(("ew-qualification-data.json", "ew-qualification-selection.json"), derive(args.root)))
    local = Path(__file__).resolve().parent
    for name, value in outputs.items():
        raw = encoded(value)
        if args.check:
            require((local / name).read_bytes() == raw, "Derived report input differs: " + name)
        else:
            with (local / name).open("xb") as stream:
                stream.write(raw)
    print(encoded({"verified": True, "inputs": {n: identity(encoded(v)) for n, v in outputs.items()},
                   "capsules": PACKETS, "new_native_physics_evaluations": 0}).decode(), end="")


if __name__ == "__main__":
    main()
