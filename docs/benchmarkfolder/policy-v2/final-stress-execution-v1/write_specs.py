#!/usr/bin/env python3
"""Write explicit final-cohort stress specs; never run CLI/build/evaluations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "madgraph/__init__.py").is_file())
sys.path.insert(0, str(ROOT / "tools"))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_supplement_v2 as supplement
import ninjars_stress_v2 as stress
import early_build

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    if path.exists():
        raise ValueError("Retain prior spec; choose a new namespace: "+str(path))
    bench.dump(path, data)


def prepare(args):
    expected = read(args.expected)
    signature = read(args.runtime_signature)
    stress.check_expected(expected, signature)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    early = None
    if args.early_result:
        result = policy.verified(read(args.early_result), "stress-early-build-result-v1")
        early = policy.verified(read(supplement.checked(result["registration"])), early_build.SCHEMA)
    generation_revision = early["generation_revision"] if early else revision
    if generation_revision != args.commit:
        raise ValueError("Explicit final committed source does not match HEAD")
    if bench.content_digest(bench.generator_tree_state()) != expected["candidate_generator_identity_sha256"]:
        raise ValueError("Final generator tree differs from the core identity")
    installation = read(args.installation)
    numerical_revision = args.numerical_revision or installation["revision"]
    early_build.sources(numerical_revision)
    if installation["revision"] != numerical_revision or installation["files"]["lib/libninjars.a"] != args.archive_sha256:
        raise ValueError("Actual-CLI final installation uses another revision/archive")
    stress.check_installed(expected, signature)
    if supplement.digest(ROOT / "HEPTools/ninjars/lib/libninjars.a") != args.archive_sha256:
        raise ValueError("Main generated-output dependency is not the final installed archive")
    spec = {"expected": expected, "runtime_signature": signature,
            "report": policy.relative(args.report), "work": policy.relative(args.work),
            "physical_cards": {"loop_sm": supplement.bound(args.loop_sm_card),
                               "loop_qcd_qed_sm_Gmu": supplement.bound(args.ew_card)},
            "final_source": {"revision": generation_revision, "numerical_revision": numerical_revision,
                             "installation": supplement.bound(args.installation),
                             "installed_archive_sha256": args.archive_sha256},
            "purpose": "Approved two-process stress generation/build/masses only; numerical collection separate"}
    if early:
        if (spec["report"] != early["report"] or spec["work"] != early["work"] or
                spec["physical_cards"] != early["physical_cards"]):
            raise ValueError("Early source/card namespace differs from final reader inputs")
        attached = early_build.attach_core(args.early_result, expected, signature)
        spec["reuse_verified_early_preparation"] = supplement.bound(attached)
    write(HERE / "prepare-spec.json", spec)
    return {"prepared_spec": supplement.bound(HERE / "prepare-spec.json"),
            "cli_or_compiler_or_engine_executions": 0}


def early(args):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if revision != args.generation_revision:
        raise ValueError("Actual HEAD differs from the requested generation freeze")
    early_build.sources(args.numerical_revision)
    signature = read(args.runtime_signature)
    spec = {"generation_revision": revision, "numerical_revision": args.numerical_revision,
            "generator_identity_sha256": bench.content_digest(bench.generator_tree_state()),
            "archive_sha256": args.archive_sha256, "runtime_signature": signature,
            "publication": supplement.bound(args.publication),
            "physical_cards": {"loop_sm": supplement.bound(args.loop_sm_card),
                               "loop_qcd_qed_sm_Gmu": supplement.bound(args.ew_card)},
            "report": policy.relative(args.report), "work": policy.relative(args.work)}
    early_build.runtime(spec)
    write(HERE / "early-spec.json", spec)
    return {"early_spec": supplement.bound(HERE / "early-spec.json"),
            "core_dataset_identity_asserted": False, "cli_or_compiler_or_engine_executions": 0}


def bind(args):
    initial = read(HERE / "prepare-spec.json")
    expected, signature = initial["expected"], initial["runtime_signature"]
    report, work = policy.path(initial["report"]), policy.path(initial["work"])
    prepared = report / "preparation.json"
    stress.validate_preparation(prepared, expected, signature)
    results, collections = {}, []
    for variant in stress.VARIANTS:
        run_report, run_work = report / "runs" / variant, work / "runs" / variant
        spec = {"expected": expected, "runtime_signature": signature,
                "preparation": supplement.bound(prepared), "variant": variant,
                "cpu": args.cpu, "report": policy.relative(run_report), "work": policy.relative(run_work),
                "run_id": "final-stress-"+variant+"-v1", "optional_helper": False}
        path = HERE / ("collect-"+variant+".json")
        write(path, spec)
        results[variant] = supplement.bound(path)
        collections.append({"variant": variant, "optional_helper": False, "report": policy.relative(run_report)})
    # Helpers are separately optional. Their actual same-process CLI exports,
    # exact MODEL postimages and private rebuild proof must be reviewed first.
    helper = {"expected": expected, "runtime_signature": signature,
              "preparation": supplement.bound(prepared), "variant": "ordinary", "optional_helper": True,
              "cpu": args.cpu, "report": policy.relative(report / "runs/helpers"),
              "work": policy.relative(work / "runs/helpers"), "run_id": "final-stress-helpers-v1",
              "helper_output_root": "__REVIEWED_PRIVATE_HELPER_OUTPUT_ROOT__",
              "helper_source_proofs": {case["id"]: {"path": "__REVIEWED_SAME_PROCESS_SOURCE_PROOF__",
                  "sha256": "__BOUND_AFTER_HELPER_REBUILD__"} for case in stress.configuration("ordinary")["cases"]}}
    write(HERE / "optional-helper-spec.template.json", helper)
    assembly = {"expected": expected, "runtime_signature": signature,
                "preparation": supplement.bound(prepared), "collections": collections,
                "output": policy.relative(report / "dataset-v1.json")}
    write(HERE / "assemble-spec.json", assembly)
    write(HERE / "validate-spec.json", {"expected": expected, "runtime_signature": signature,
          "dataset": assembly["output"]})
    results["assembly"] = supplement.bound(HERE / "assemble-spec.json")
    return {"collection_specs": results, "cli_or_compiler_or_engine_executions": 0,
            "optional_helper_status": "template only; requires separate actual export/postimage/rebuild proof"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    first = commands.add_parser("prepare")
    first.add_argument("--commit", required=True)
    first.add_argument("--numerical-revision")
    first.add_argument("--early-result", type=Path)
    first.add_argument("--archive-sha256", required=True)
    first.add_argument("--expected", required=True, type=Path)
    first.add_argument("--runtime-signature", required=True, type=Path)
    first.add_argument("--installation", required=True, type=Path)
    first.add_argument("--loop-sm-card", required=True, type=Path)
    first.add_argument("--ew-card", required=True, type=Path)
    first.add_argument("--report", required=True, type=Path)
    first.add_argument("--work", required=True, type=Path)
    initial = commands.add_parser("early")
    initial.add_argument("--generation-revision", required=True)
    initial.add_argument("--numerical-revision", required=True)
    initial.add_argument("--archive-sha256", required=True)
    initial.add_argument("--runtime-signature", required=True, type=Path)
    initial.add_argument("--publication", required=True, type=Path)
    initial.add_argument("--loop-sm-card", required=True, type=Path)
    initial.add_argument("--ew-card", required=True, type=Path)
    initial.add_argument("--report", required=True, type=Path)
    initial.add_argument("--work", required=True, type=Path)
    second = commands.add_parser("bind")
    second.add_argument("--cpu", required=True, type=int)
    args = parser.parse_args()
    result = early(args) if args.action == "early" else prepare(args) if args.action == "prepare" else bind(args)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
