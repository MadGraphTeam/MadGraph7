"""Compile the measured report and record its exact document inputs.

Numerical measurements are never launched by this script. Run the benchmark
summarizer first; incomplete evidence is rendered as incomplete, not fabricated.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
BACKENDS = {"1", "6", "7", "8", "9"}
CAMPAIGN_SCHEMA = "madloop-tirust-campaign-v1"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def campaign_id(value):
    if (not isinstance(value, str) or value == "legacy" or
            not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", value)):
        raise ValueError("Invalid campaign ID")
    return value


def inside(root, name):
    """Require a relative, non-aliased report input, including its parents."""
    root = Path(root).resolve()
    if (not isinstance(name, str) or not name or "\\" in name or
            Path(name).is_absolute() or ".." in Path(name).parts):
        raise ValueError("Unsafe report input path: " + str(name))
    path = root
    for part in Path(name).parts:
        path /= part
        if path.is_symlink():
            raise ValueError("Report input may not be a symlink: " + str(path))
    if not path.resolve().is_relative_to(root):
        raise ValueError("Report input escapes its root")
    return path


class ReportInputs:
    def __init__(self, root):
        self.root, self.files = Path(root).resolve(), {}

    def add(self, name, expected=None):
        path = inside(self.root, name)
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError("Report input hash mismatch: " + name)
        self.files[name] = {"sha256": digest, "bytes": len(data)}
        return data

    def json(self, name, expected=None):
        def reject(value):
            raise ValueError("Nonfinite JSON value in " + name + ": " + value)
        return json.loads(self.add(name, expected), parse_constant=reject)


def joined(directory, name):
    # Validate each part separately so an absolute child cannot discard its root.
    for value in (directory, name):
        if (not isinstance(value, str) or not value or "\\" in value or
                Path(value).is_absolute() or ".." in Path(value).parts):
            raise ValueError("Unsafe report input path: " + str(value))
    return (Path(directory) / name).as_posix()


def predecessor_history(inputs, manifest, summary, report):
    """Validate frozen ancestry and discoveries; return compact display rows."""
    nodes, failures = {}, {}

    def retain(record):
        identifier = record["predecessor_id"]
        if identifier not in nodes:
            raise ValueError("Discovery refers to an unknown ancestor")
        node, snapshot = nodes[identifier]
        if (record["report"] != node["report"] or
                record["policy_sha256"] != node["policy_sha256"]):
            raise ValueError("Discovery has a different ancestor policy or path")
        failure = record["failure"]
        directory = failure["directory"]
        inside(inputs.root, joined(node["report"], directory))
        if (Path(directory).parts[:1] != ("runs",) or
                len(Path(directory).parts) != 3 or Path(directory).parts[1] != failure["case"]):
            raise ValueError("Invalid qualified failure directory")
        key = (identifier, directory)
        if key in failures:
            if failures[key]["record"] != record:
                raise ValueError("Conflicting ancestor failure records")
            return
        files = failure["files"]
        for name, checksum in files.items():
            if not Path(name).is_relative_to(Path(directory)):
                raise ValueError("Failure input lies outside its retained directory")
            inputs.add(joined(node["report"], name), checksum)
        required = [joined(directory, name) for name in ("summary.json", "candidate-inputs.json")]
        if any(name not in files for name in required):
            raise ValueError("Qualified failure lacks summary or source identity")
        evidence = inputs.json(joined(node["report"], required[0]), files[required[0]])
        candidate = inputs.json(joined(node["report"], required[1]), files[required[1]])
        if (evidence.get("acceptance_eligible") is not True or evidence.get("passed") is not False
                or evidence.get("label") != failure["label"]):
            raise ValueError("Retained predecessor result is not a qualified failure")
        case = next((row for row in snapshot["cases"] if row["id"] == failure["case"]), None)
        if case is None:
            raise ValueError("Ancestor summary omits the failed process")
        identity = candidate["generation"]["source_identity"]
        statistics = evidence.get("statistics", {})
        for name in ("geometric_mean_ratio", "lower_95", "upper_95"):
            value = statistics.get(name)
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("Invalid predecessor timing statistic")
        failures[key] = {"record": record, "display": {
            "predecessor_id": identifier, "id": case["id"], "process": case["process"],
            "label": failure["label"], "directory": joined(node["report"], directory),
            "revision": identity["revision"], "source_sha256": identity["files_sha256"],
            "geometric_mean_ratio": statistics.get("geometric_mean_ratio"),
            "lower_95": statistics.get("lower_95"), "upper_95": statistics.get("upper_95"),
            "passed": False, "acceptance_eligible": True}}

    def visit(node):
        identifier = node["id"]
        expected_report = "." if identifier == "legacy" else "campaigns/" + campaign_id(identifier)
        if identifier in nodes or node["report"] != expected_report:
            raise ValueError("Cyclic, duplicate or aliased predecessor")
        snapshot_name = node.get("summary_snapshot") or joined(node["report"], "summary.json")
        snapshot = validate_summary(inputs.json(snapshot_name, node["summary_sha256"]))
        if identifier == "legacy":
            if node["manifest_sha256"] is not None or node.get("ancestor") is not None:
                raise ValueError("Invalid legacy predecessor ancestry")
        else:
            previous = inputs.json(joined(node["report"], "manifest.json"), node["manifest_sha256"])
            if (previous.get("schema") != CAMPAIGN_SCHEMA or previous.get("id") != identifier or
                    previous["candidate"]["numerical_identity_sha256"] != node["numerical_identity_sha256"] or
                    previous["predecessor"] != node.get("ancestor") or
                    snapshot.get("campaign_binding") != {"id": identifier, "manifest_sha256": node["manifest_sha256"]} or
                    snapshot.get("campaign_numerical_identity_sha256") != node["numerical_identity_sha256"]):
                raise ValueError("Predecessor manifest, snapshot or ancestor binding differs")
        nodes[identifier] = (node, snapshot)
        if node.get("ancestor"):
            visit(node["ancestor"])
        for failure in node["qualified_failures"]:
            retain({"predecessor_id": identifier, "report": node["report"],
                    "policy_sha256": node["policy_sha256"], "failure": failure})
        for discovery in node.get("inherited_discoveries", []):
            record = inputs.json(discovery["path"], discovery["sha256"])
            if record != discovery["record"]:
                raise ValueError("Inherited predecessor discovery changed")
            retain(record)

    visit(manifest["predecessor"])
    discoveries = inside(inputs.root, joined(report, "inputs/predecessor/discoveries"))
    for path in sorted(discoveries.glob("*.json")):
        record = inputs.json(path.relative_to(inputs.root).as_posix())
        if path.stem != content_digest(record):
            raise ValueError("Predecessor discovery filename/hash differs")
        retain(record)
    references = [{"predecessor_id": item["record"]["predecessor_id"],
                   "case": item["record"]["failure"]["case"],
                   "label": item["record"]["failure"]["label"],
                   "directory": item["record"]["failure"]["directory"],
                   "evidence_sha256": content_digest(item["record"]["failure"]["files"])}
                  for item in failures.values()]
    expected = {name: manifest["predecessor"][name] for name in ("id", "manifest_sha256", "summary_sha256")}
    actual = dict(summary.get("campaign_predecessor", {}))
    actual_references = actual.pop("qualified_failures", None)
    if (actual != expected or actual_references is None or
            sorted(map(content_digest, actual_references)) != sorted(map(content_digest, references))):
        raise ValueError("Campaign summary omits or changes retained predecessor failures; summarize again")
    return [failures[key]["display"] for key in sorted(failures)]


def prepare_report(root=ROOT, campaign=None, campaign_report=None, supplement=None):
    helper_path = Path(__file__).with_name("intermediate_report.py")
    if campaign is None and (Path(root) / "intermediate-status.json").exists() and not helper_path.is_file():
        raise ValueError("Intermediate status exists but its validator is missing")
    inputs = ReportInputs(root)
    report = "." if campaign is None else "campaigns/" + campaign_id(campaign)
    summary_name, cases_name = joined(report, "summary.json"), joined(report, "cases.json")
    summary = inputs.json(summary_name)
    cases = inputs.json(cases_name)
    configured = {row["id"]: row for row in cases["cases"]}
    if (len(configured) != len(cases["cases"]) or set(configured) != {row["id"] for row in summary["cases"]} or
            cases["baseline_revision"] != summary["baseline_revision"]):
        raise ValueError("Report cases differ from their configuration")
    for row in summary["cases"]:
        if any(row.get(key) != value for key, value in configured[row["id"]].items()):
            raise ValueError("Report process differs from its configuration: " + row["id"])
    context = {"campaign_id": campaign, "summary_path": summary_name, "cases_path": cases_name,
               "predecessors": [], "numerical_identity_sha256": None, "ratio_figure": None}
    manifest_name = ""
    if campaign is not None:
        manifest_name = joined(report, "manifest.json")
        manifest = inputs.json(manifest_name)
        if (manifest.get("schema") != CAMPAIGN_SCHEMA or manifest.get("id") != campaign or
                summary.get("campaign_binding") != {"id": campaign, "manifest_sha256": inputs.files[manifest_name]["sha256"]} or
                summary.get("campaign_numerical_identity_sha256") != manifest["candidate"]["numerical_identity_sha256"] or
                manifest["baseline"]["revision"] != summary["baseline_revision"]):
            raise ValueError("Selected campaign summary/manifest binding differs")
        shared = manifest["shared_inputs"]
        if shared["policy_sha256"] != content_digest(cases) or "cases.json" not in shared["files"]:
            raise ValueError("Campaign policy differs from the sealed configuration")
        for name, checksum in shared["files"].items():
            inputs.add(joined(report, name), checksum)
        identity = manifest["candidate"]["exporter_identity"]
        observed_revisions = summary.get("candidate_revisions", [])
        observed_revision = summary.get("candidate_revision")
        if (any(value != identity["files_sha256"] for value in summary.get("candidate_source_hashes", []))
                or len(set(summary.get("candidate_source_hashes", []))) > 1
                or len(set(observed_revisions)) != len(observed_revisions)
                or (observed_revision is not None and observed_revisions != [observed_revision])
                or (observed_revision is None and len(observed_revisions) == 1)):
            raise ValueError("Summary candidate source differs from the registered candidate")
        # Documentation commits can change the observed HEAD without changing
        # a single exporter byte. Keep raw observations unchanged and label the
        # registered anchor separately; never invent an observed Git revision.
        context["registered_exporter_revision"] = identity["revision"]
        context["observed_candidate_revisions"] = summary.get("candidate_revisions", [])
        validate_summary(summary, registered_identity=identity)
        context["numerical_identity_sha256"] = manifest["candidate"]["numerical_identity_sha256"]
        context["predecessors"] = predecessor_history(inputs, manifest, summary, report)
    elif summary.get("campaign_binding") is not None:
        raise ValueError("A bound campaign summary requires --campaign")
    else:
        validate_summary(summary)
    for name in ("build_report.py", "benchmark.typ", "style.typ", "poles-timing.typ", "implementation-start.json",
                 "style-provenance.json", "experiments/numerical/completed-pilots-index.json",
                 "experiments/poles-timing/historical-release-paired/analysis-summary.json"):
        inputs.add(name)
    if summary.get("ratio_figure"):
        context["ratio_figure"] = joined(report, summary["ratio_figure"])
        inputs.add(context["ratio_figure"])
    # The supplement binds later implementation and historical diagnostic
    # evidence independently of the selected acceptance campaign. It never
    # replaces or upgrades that campaign's numerical or performance gate.
    if campaign is None and helper_path.exists():
        spec = importlib.util.spec_from_file_location("intermediate_report", helper_path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        supplement = helper.validate(inputs, summary)
        if supplement is not None:
            context["intermediate_path"] = "intermediate-status.json"
    if campaign is not None:
        dossier_path = campaign_report or joined(report, "report-data.json")
        if campaign_report is not None or inside(inputs.root, dossier_path).exists():
            if not Path(dossier_path).is_relative_to(Path(report)):
                raise ValueError("Campaign report dossier must belong to the selected campaign")
            dossier_helper = Path(__file__).with_name("campaign_report.py")
            if not dossier_helper.is_file():
                raise ValueError("Campaign report dossier exists but its validator is missing")
            spec = importlib.util.spec_from_file_location("campaign_report", dossier_helper)
            helper = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(helper)
            dossier = helper.validate(inputs, dossier_path, manifest, summary)
            context["campaign_report_path"] = dossier_path
            context["delivery_complete"] = dossier["delivery_complete"]
            inputs.add("campaign-report.typ")
    context_name = joined(report, "report-context.json")
    if supplement is not None:
        if campaign is None or "campaign_report_path" not in context:
            raise ValueError("A source supplement requires an explicitly validated historical campaign dossier")
        spec = importlib.util.spec_from_file_location("current_source_supplement",
            Path(__file__).with_name("current_source_supplement.py"))
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
        context["source_supplement"], context_name = helper.validate(
            inputs, supplement, campaign, context["campaign_report_path"])
        inputs.add("current_source_supplement.py")
        inputs.add("current-source-supplement.typ")
    return summary, inputs, context, {"summary": summary_name, "manifest": manifest_name,
                                     "context": context_name}


def validate_summary(data, registered_identity=None):
    """Refuse a passing headline when declared numerical evidence is absent."""
    cases = data["cases"]
    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Duplicate process rows in report")
    if data["baseline_revision"] != "9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f":
        raise ValueError("Report baseline differs from the approved frozen revision")
    if data["gate"]["passed"]:
        main = [row for row in cases if row["role"] == "main"]
        if (not data["gate"]["complete"] or len(main) != 8 or
                not data["gate"].get("consistent_candidate_source") or
                (not data.get("candidate_revision") and not (
                    registered_identity is not None and data.get("campaign_binding") is not None
                    and data.get("candidate_source_hashes") == [registered_identity["files_sha256"]]
                    and bool(data.get("candidate_revisions")))) or data["gate"].get("failed_cases") or
                data["gate"].get("validation_errors")):
            raise ValueError("Passing report requires all eight main cases")
        for row in main:
            checks = row["correctness"]
            timing = row["performance"]
            if (row["points"] is None or row["points"] < 100 or len(checks) != 5 or
                    row.get("authoritative_qualified_failures") or row.get("validation_errors") or
                    row.get("status", "passed") != "passed" or
                    {str(check["backend"]) for check in checks} != BACKENDS or
                    not all(check["passed"] is True and check["samples"] >= 7
                            for check in checks) or
                    timing["passed"] is not True or not timing["complete"] or
                    timing["pairs"] < 7 or
                    timing["upper_95"] is None or
                    not math.isfinite(timing["upper_95"]) or timing["upper_95"] > 1.05):
                raise ValueError("Unqualified process in passing report: " + row["id"])
            regressions = row.get("existing_backend_regressions", [])
            if (len(regressions) != 4 or
                    {str(check["backend"]) for check in regressions} != BACKENDS - {"9"} or
                    not all(check["complete"] and check["passed"] is True
                            for check in regressions)):
                raise ValueError("Missing existing-backend regression evidence: " + row["id"])
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark.pdf")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--font-path", type=Path, action="append", default=[])
    parser.add_argument("--campaign", help="Registered campaign ID; omit for the original campaign")
    parser.add_argument("--campaign-report", help="Explicit frozen docsroot-relative dossier for that campaign")
    parser.add_argument("--supplement", help="Explicit frozen source-progress selection; does not alter selected campaign gates")
    args = parser.parse_args()
    if args.campaign_report and not args.campaign:
        parser.error("--campaign-report requires --campaign")
    summary, inventory, context, typst_inputs = prepare_report(ROOT, args.campaign, args.campaign_report, args.supplement)
    context_path = inside(ROOT, typst_inputs["context"])
    context_bytes = (json.dumps(context, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    # A custom PDF must not overwrite the canonical PDF's reproducibility ledger.
    ledger = (ROOT / "report-build.json" if args.output.resolve() == (ROOT / "benchmark.pdf").resolve()
              else args.output.with_suffix(".build.json"))
    if args.verify_only:
        if context_path.read_bytes() != context_bytes:
            raise ValueError("Report context differs from validated campaign evidence")
        inventory.add(typst_inputs["context"])
        recorded = json.loads(ledger.read_text())
        if (inventory.files != recorded["inputs"] or
                recorded.get("campaign") != args.campaign or recorded.get("typst_inputs") != typst_inputs):
            raise ValueError("Report inputs differ from the recorded PDF build")
        if sha256(args.output) != recorded["pdf_sha256"]:
            raise ValueError("PDF differs from its recorded build")
        print("Report inputs and PDF hashes verified")
        return
    typst = shutil.which("typst")
    if not typst:
        raise RuntimeError("Source tools/tirust_env.sh or activate the Nix shell")
    version = subprocess.check_output([typst, "--version"], text=True).strip()
    if not version.startswith("typst 0.15.1 "):
        raise RuntimeError("Reproduction requires Typst 0.15.1; found " + version)
    epoch = int(summary.get("source_date_epoch", 1794182400))
    command = [typst, "compile", "--root", str(ROOT),
               "--creation-timestamp", str(epoch)]
    for key, value in typst_inputs.items():
        command.extend(["--input", key + "=" + value])
    for directory in args.font_path:
        command.extend(["--font-path", str(directory.resolve())])
    environment = dict(os.environ, SOURCE_DATE_EPOCH=str(epoch), RAYON_NUM_THREADS="1")
    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_bytes(context_bytes)
    inventory.add(typst_inputs["context"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".benchmark-", suffix=".pdf", dir=args.output.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        command.extend([str(ROOT / "benchmark.typ"), str(temporary.resolve())])
        subprocess.run(command, env=environment, check=True)
        for name, identity in inventory.files.items():
            if sha256(inside(ROOT, name)) != identity["sha256"]:
                raise ValueError("Report input changed during compilation: " + name)
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {"typst": version, "source_date_epoch": epoch,
                "inputs": inventory.files, "pdf_sha256": sha256(args.output),
                "campaign": args.campaign, "typst_inputs": typst_inputs,
                "font_paths": [str(path.resolve()) for path in args.font_path],
                "font_search_environment": environment.get("TYPST_FONT_PATHS", ""),
                "numerical_campaign_rerun": False}
    ledger.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"pdf": str(args.output), "sha256": manifest["pdf_sha256"],
                      "gate": summary["gate"]}, indent=2))


if __name__ == "__main__":
    main()
