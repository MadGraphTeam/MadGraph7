"""Validate the frozen intermediate supplement without running physics code."""
from pathlib import Path

SCHEMA = "madloop-tirust-intermediate-v1"
REQUIRED = {"workspace", "terminal", "installer", "late-ew3", "late-qcd4",
            "late-ew4", "ew-threshold", "cms", "born-prototype", "cohorts",
            "n3-completeness", "adapter-o3", "adapter-scratch", "adapter-sequence",
            "common-allocation", "finite-scan", "terminal-bulk"}
STATUSES = {"pass", "partial", "fail", "pending", "unpromoted", "rejected"}


def at(value, pointer):
    for part in pointer:
        value = value[part]
    return value


def validate(inputs, summary, required=False):
    """Hash every declared dependency and verify machine-readable claim anchors.

    The supplement describes completed implementation checks and historical
    diagnostics. It cannot turn these into final-campaign acceptance. Existing
    report fixtures without a supplement remain supported.
    """
    if not (inputs.root / "intermediate-status.json").exists():
        if required:
            raise ValueError("Intermediate status supplement is missing")
        return None
    data = inputs.json("intermediate-status.json")
    if (data.get("schema") != SCHEMA or data.get("kind") != "intermediate"
            or data.get("final_campaign_status") != "pending"
            or data.get("final_campaign_accepted") is not False
            or data.get("new_measurements_performed") is not False):
        raise ValueError("Intermediate checks cannot claim final campaign acceptance")
    if summary.get("gate", {}).get("passed") is True:
        raise ValueError("Refresh the pending supplement before reporting final campaign acceptance")
    for name, identity in data["files"].items():
        raw = inputs.add(name, identity["sha256"])
        if len(raw) != identity["bytes"]:
            raise ValueError("Intermediate evidence byte count differs: " + name)
    for name in ("intermediate_report.py", "collect_intermediate.py", "intermediate.typ"):
        inputs.add(name)

    def frozen_json(name):
        if name not in data["files"]:
            raise ValueError("Unbound intermediate JSON: " + name)
        return inputs.json(name, data["files"][name]["sha256"])

    if data["qualification"] != frozen_json("qualification.json"):
        raise ValueError("Displayed qualification differs from frozen report data")
    claims = [claim for section in data["sections"] for claim in section["claims"]]
    ids = [claim["id"] for claim in claims]
    if len(ids) != len(set(ids)) or not REQUIRED.issubset(ids):
        raise ValueError("Intermediate supplement omits or duplicates a required result")
    for claim in claims:
        if (claim["status"] not in STATUSES or not claim["scope"]
                or claim.get("performance_acceptance_eligible") is not False
                or not claim["evidence"]):
            raise ValueError("Missing scope or inflated acceptance: " + claim["id"])
        for name in claim["evidence"]:
            if name not in data["files"]:
                raise ValueError("Unbound claim evidence: " + name)
        for assertion in claim.get("assertions", []):
            name = assertion["path"]
            if name not in claim["evidence"]:
                raise ValueError("Claim assertion lacks a declared source")
            try:
                observed = at(frozen_json(name), assertion["pointer"])
            except (KeyError, IndexError, TypeError) as error:
                raise ValueError("Claim source field is missing: " + claim["id"]) from error
            if observed != assertion["expected"]:
                raise ValueError("Claim differs from its raw source: " + claim["id"])
    failure = data["authoritative_failure"]
    retained = inputs.json(failure["path"], data["files"][failure["path"]]["sha256"])
    if (retained.get("acceptance_eligible") is not True or retained.get("passed") is not False
            or retained["statistics"] != failure["statistics"]):
        raise ValueError("Authoritative historical failure was changed or hidden")
    if summary.get("campaign_binding") is None:
        case = next(row for row in summary["cases"] if row["id"] == failure["case"])
        if (failure["case"] not in summary["gate"].get("failed_cases", [])
                or case["performance"]["passed"] is not False):
            raise ValueError("Legacy summary omits the qualified historical failure")
    if data["installer_identity"]["path"] not in data["files"]:
        raise ValueError("Installer identity has no frozen evidence")
    installer = frozen_json(data["installer_identity"]["path"])
    if (installer["native_artifacts"]["libmadloop_tirust.so"]["sha256"]
            != data["installer_identity"]["library_sha256"]
            or installer["source_revision"] != data["installer_identity"]["source_revision"]):
        raise ValueError("Displayed installer identity differs from actual publication")
    for row in data["diagnostic_timings"]:
        if row["performance_acceptance_eligible"] is not False:
            raise ValueError("Diagnostic timing promoted to acceptance")
        record = inputs.json(row["path"], data["files"][row["path"]]["sha256"])
        statistics = at(record, row["pointer"])
        if statistics != row["statistics"] or record.get("performance_eligible") is not False:
            raise ValueError("Diagnostic ratio differs from retained evidence")
    return data
