"""Freeze and revalidate report-only campaign evidence; never launch physics.

Numerical warmup, paired acceptance and traced cohorts are separate datasets.
The immutable dossier records the exact files visible at collection time; later
measurements require a new dossier, not changes to an earlier report snapshot.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCHEMA = "madloop-tirust-campaign-report-v1"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def document_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Evidence:
    def __init__(self, root, frozen=None):
        self.root = Path(root).resolve()
        self.files = {}
        self.frozen = frozen

    def path(self, name):
        name = str(name)
        relative = Path(name)
        if (not name or relative.is_absolute() or ".." in relative.parts or "\\" in name):
            raise ValueError("Unsafe report evidence path: " + name)
        path = self.root
        for part in relative.parts:
            path /= part
            if path.is_symlink():
                raise ValueError("Report evidence may not contain symlinks")
        return path

    def exists(self, name):
        self.path(name)
        if self.frozen is not None:
            prefix = str(name).rstrip("/") + "/"
            return str(name) in self.frozen or any(p.startswith(prefix) for p in self.frozen)
        return self.path(name).exists()

    def add(self, name, expected=None):
        name = str(name)
        raw = self.path(name).read_bytes()
        identity = {"sha256": digest(raw), "bytes": len(raw)}
        if expected is not None and identity["sha256"] != expected:
            raise ValueError("Evidence hash differs: " + name)
        if self.frozen is not None and self.frozen.get(name) != identity:
            raise ValueError("Frozen report evidence differs or is unbound: " + name)
        self.files[name] = identity
        return raw

    def json(self, name, expected=None):
        def invalid(value):
            raise ValueError("Nonfinite JSON in " + str(name) + ": " + value)
        return json.loads(self.add(name, expected), parse_constant=invalid)

    def tree(self, name):
        path = self.path(name)
        if self.frozen is None:
            paths = sorted(path.rglob("*")) if path.is_dir() else [path]
            names = [p.relative_to(self.root).as_posix() for p in paths
                     if "__pycache__" not in p.parts and (p.is_file() or p.is_symlink())]
        else:
            prefix = str(name).rstrip("/") + "/"
            names = sorted(p for p in self.frozen if p == str(name) or p.startswith(prefix))
        for item in names:
            self.add(item)
        return names


def validate_fortran_proof_identity(proof, manifest):
    """A historical proof only qualifies the exact registered production code."""
    production = {name: checksum for name, checksum in proof["source_and_test_sha256"].items()
                  if not name.startswith("tests/")}
    for invocation in proof["protocol_invocations"]:
        for name, checksum in invocation["source_sha256"].items():
            if name in production and production[name] != checksum:
                raise ValueError("Fortran proof has inconsistent production source: " + name)
            production[name] = checksum
    if not production or "madgraph/iolibs/helas_call_writers.py" not in production:
        raise ValueError("Fortran proof is missing its production source closure")
    candidate = manifest["candidate"]
    for name, checksum in production.items():
        inventories = [group for group in (candidate["exporter_identity"]["files"],
                       candidate.get("numerical_sources", {})) if name in group]
        values = [group[name]["sha256"] if isinstance(group[name], dict) else group[name]
                  for group in inventories]
        if not values or any(value != checksum for value in values):
            raise ValueError("Fortran proof differs from registered production source: " + name)
    return production


def implementation(evidence, manifest):
    """Only completed frozen implementation checks; not amplitude acceptance."""
    prefix = "qualification/tirust-install/qp-transport-cache/"
    terminal = "experiments/dp-higher-rank-production/"
    evidence.tree(prefix)
    evidence.tree(terminal)
    installed = evidence.json(prefix + "check-installation.json")
    parity = evidence.json(prefix + "success-parity/results.json")
    proof = evidence.json(terminal + "validation-summary.json")
    proof_sources = validate_fortran_proof_identity(proof, manifest)
    native = manifest["candidate"]["native"]["files"]
    for name, identity in installed["native_artifacts"].items():
        key = "lib/" + name
        if key in native and identity["sha256"] != native[key]:
            raise ValueError("Implementation dossier and registered native package differ")
    archive = installed["native_artifacts"]["libtirust.a"]["sha256"]
    library = installed["native_artifacts"]["libmadloop_tirust.so"]["sha256"]
    checks = {
        "installer_available": installed.get("available") is True,
        "smoke": installed.get("serialized_installer_smoke_passed") is True,
        "native_identity_preserved": installed.get("immutable_installation_hashes_unchanged") is True,
        "public_fortran_parity": parity.get("passed") is True and parity.get("exact_frame_bytes_equal") is True,
        "higher_rank_fortran_proof": proof.get("status") == "passed",
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
        "installer_source_revision": installed["source_revision"], "library_sha256": library,
        "fortran_archive_sha256": archive, "public_fortran_calls_per_installation": parity["calls_per_installation"],
        "raw_dp_calls_per_library": installed["raw_dp_calls_per_library"],
        "raw_dp_event_records_identical": installed["raw_dp_event_records_identical"],
        "raw_dp_successes_per_library": installed["raw_dp_successes_per_library"],
        "raw_dp_expected_failures_per_library": installed["raw_dp_expected_failures_per_library"],
        "higher_rank_summary": proof,
        "higher_rank_registered_sources": proof_sources,
        "scope": "Installed ABI, transport/cache rejection and Fortran arithmetic/source qualification; no new amplitude or performance claim",
        "evidence": [prefix + "README.md", prefix + "check-installation.json", prefix + "archive-tests/results.json",
                     prefix + "archive-tests/real-boundary-analysis.json", terminal + "README.md", terminal + "validation-summary.json"]}


def statistics_qualification(evidence, manifest):
    return document_module("campaign_report_qualification").collect_statistics(evidence, manifest)


def stress_limitations(evidence, manifest):
    return document_module("campaign_report_qualification").collect_stress(evidence, manifest)


def stress_acceptance(rows):
    """Only the established physical stress requirements control this gate."""
    required_cases = {"EW threshold mass transport", "QCD soft/collinear prototype",
                      "Default stability estimator", "Complex-mass extremes"}
    if any(type(row.get("required_for_stress_acceptance")) is not bool for row in rows):
        return "open"
    if any(row["required_for_stress_acceptance"] != (row.get("title") in required_cases) for row in rows):
        return "open"
    required = [row for row in rows if row["required_for_stress_acceptance"]]
    if len(required) != len(required_cases) or {row["title"] for row in required} != required_cases:
        return "open"
    return "passed" if all(row["status"] == "passed" for row in required) else "open"


def collect(evidence, campaign_id):
    if not campaign_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in campaign_id):
        raise ValueError("Invalid campaign identifier")
    directory = "campaigns/" + campaign_id
    manifest = evidence.json(directory + "/manifest.json")
    binding = {"id": campaign_id, "manifest_sha256": evidence.files[directory + "/manifest.json"]["sha256"]}
    if manifest.get("id") != campaign_id or manifest.get("schema") != "madloop-tirust-campaign-v1":
        raise ValueError("Unexpected registered campaign manifest")
    config = evidence.json(directory + "/cases.json", manifest["shared_inputs"]["files"]["cases.json"])
    for name, sha in manifest["shared_inputs"]["files"].items():
        evidence.add(directory + "/" + name, sha)
    if len(config["cases"]) != 11 or len({c["id"] for c in config["cases"]}) != 11:
        raise ValueError("Report requires all eleven configured physical cases")
    numerical = document_module("campaign_report_numerical").collect_numerical(evidence, directory, manifest, config)
    paired = document_module("campaign_report_pairs").collect_pairs(evidence, directory, manifest, config)
    cohorts = document_module("campaign_report_cohorts").collect_cohorts(evidence, directory, manifest, config)
    qualified = implementation(evidence, manifest)
    statistics = statistics_qualification(evidence, manifest)
    stress = stress_limitations(evidence, manifest)
    lifecycle = document_module("campaign_report_lifecycle").collect_lifecycle(evidence, manifest)
    criteria = {
        "all_eleven_five_backend_numerical": numerical["status"],
        "eight_main_runtime_gates": paired["main_status"],
        "existing_backend_regressions": paired["regression_status"],
        "paired_campaign_overall": paired["status"],
        "native_dp_and_rescued_cohorts": cohorts["status"],
        "installed_abi_and_fortran_checks": qualified["status"],
        "subprocess_event_cache_lifecycle": lifecycle["status"],
        "statistics_roundtrip": statistics["status"],
        "stress_accuracy": stress_acceptance(stress),
    }
    return {"schema": SCHEMA, "campaign_binding": binding,
        "registered_exporter_revision": manifest["candidate"]["exporter_identity"]["revision"],
        "registered_source_sha256": manifest["candidate"]["exporter_identity"]["files_sha256"],
        "numerical_identity_sha256": manifest["candidate"]["numerical_identity_sha256"],
        "numerical": numerical, "paired": paired, "cohorts": cohorts,
        "implementation": qualified, "subprocess_event_cache_lifecycle": lifecycle,
        "statistics": statistics,
        "stress": stress, "criteria": criteria,
        "delivery_complete": all(value == "passed" for value in criteria.values()),
        "new_numerical_execution": False}


def validate(inputs, path, manifest, summary):
    dossier = inputs.json(path)
    if dossier.get("schema") != SCHEMA or dossier.get("new_numerical_execution") is not False:
        raise ValueError("Unknown or execution-bearing report dossier")
    frozen = dossier["files"]
    evidence = Evidence(inputs.root, frozen)
    for name, identity in frozen.items():
        inputs.add(name, identity["sha256"])
        evidence.add(name)
    actual = collect(evidence, manifest["id"])
    if {k: v for k, v in dossier.items() if k != "files"} != actual:
        raise ValueError("Campaign report differs from independently reparsed frozen evidence")
    if actual["campaign_binding"] != summary.get("campaign_binding"):
        raise ValueError("Report dossier and selected summary belong to different campaigns")
    for name in ("campaign_report.py", "campaign_report_numerical.py", "campaign_report_pairs.py",
                 "campaign_report_cohorts.py", "campaign_report_lifecycle.py", "campaign_report_qualification.py"):
        inputs.add(name)
    return dossier


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", required=True, help="New docsroot-relative immutable dossier path")
    args = parser.parse_args()
    evidence = Evidence(ROOT)
    result = collect(evidence, args.campaign)
    result["files"] = dict(sorted(evidence.files.items()))
    output = evidence.path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"path": str(output), "delivery_complete": result["delivery_complete"], "criteria": result["criteria"]}))


if __name__ == "__main__":
    main()
