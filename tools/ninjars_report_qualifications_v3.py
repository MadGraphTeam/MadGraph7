"""Read the concrete final-release proof authorities without running physics.

These adapters accept the named retained capsules only. They call the capsules'
read-only verifiers and bind their raw payloads to the measured generator/runtime;
an arbitrary JSON success flag is never a qualification authority.
"""
from contextlib import redirect_stdout
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_precision_v2 as precision

ROOT = bench.ROOT
RELEASE = ROOT / "docs/benchmarkfolder/policy-v3/final-pins-20261021-v1"
INSTALLATION = RELEASE / "install-export/installation.json"
REGRESSIONS = RELEASE / "regressions-update-v1/summary.json"
NUMERICAL = RELEASE / "numerical-controls-v1/manifest-v2.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def retain(filename, inputs, checksum=None, size=None):
    filename = policy.path(filename)
    raw = filename.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    require(checksum is None or checksum == actual, "Changed qualification input: " + policy.relative(filename))
    require(size is None or size == len(raw), "Changed qualification input size")
    inputs[policy.relative(filename)] = actual
    return raw


def read(filename, inputs):
    return json.loads(retain(filename, inputs))


def capsule(directory, inputs, callback):
    """Use a concrete owner verifier, then retain the exact checked payloads."""
    require(__debug__, "Qualification readers require Python assertions enabled")
    manifest_path = directory / "manifest.json"
    manifest_raw = retain(manifest_path, inputs)
    manifest = json.loads(manifest_raw)
    payloads = manifest.get("files", manifest.get("payloads"))
    require(isinstance(payloads, dict) and "verify.py" in payloads, "Missing concrete qualification verifier")
    verifier = directory / "verify.py"
    source = retain(verifier, inputs, payloads["verify.py"]["sha256"], payloads["verify.py"]["bytes"])
    module = ModuleType("ninjars_concrete_qualification")
    module.__file__ = str(verifier)
    previous_bytecode = sys.dont_write_bytecode
    output = io.StringIO()
    try:
        sys.dont_write_bytecode = True
        exec(compile(source, str(verifier), "exec"), module.__dict__)
        with redirect_stdout(output):
            result = callback(module)
    finally:
        sys.dont_write_bytecode = previous_bytecode
    if result is None:
        result = json.loads(output.getvalue())
    for name, record in payloads.items():
        filename = (directory / name).resolve()
        require(filename.is_relative_to(directory), "Qualification payload leaves its capsule")
        retain(filename, inputs, record["sha256"], record["bytes"])
    retain(manifest_path, inputs, hashlib.sha256(manifest_raw).hexdigest())
    return result


def match_generator(recorded, candidate):
    # The generated Fortran identity has this exact scope. The Rust bridge is
    # qualified separately through the installed archive/source precision proof.
    prefixes = bench.GENERATOR_PATHS
    production = {name: value for name, value in recorded.items()
                  if any(name == p or name.startswith(p + "/") for p in prefixes)}
    require(production and all(candidate.get(name) == value for name, value in production.items()),
            "Qualification source differs from measured candidate generator")
    return len(production)


def installation_snapshot(filename, runtime_signature, candidate_generator_files, inputs):
    require(policy.path(filename) == INSTALLATION, "Use the concrete final installation/export authority")
    directory = INSTALLATION.parent
    verified = capsule(directory, inputs, lambda module: module.verify(live=False))
    require(verified.get("passed") is True and verified.get("engine_executions") == 0,
            "Installation/export verifier did not complete read-only verification")
    installation = read(INSTALLATION, inputs)
    metadata = {name: installation["files"]["lib/" + name]
                for name in policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}}
    precision.match_runtime_archive(runtime_signature, installation["files"]["lib/libninjars.a"], metadata,
                                    runtime_signature["identity_sha256"])
    generators = read(directory / "archived-generator-sources.json", inputs)
    matched = match_generator(generators, candidate_generator_files)
    runtime = read(directory / "runtime-results.json", inputs)
    return {"status": "passed", "installation": policy.bound_file(INSTALLATION),
            "manifest": policy.bound_file(directory / "manifest.json"),
            "candidate_runtime_identity_sha256": runtime_signature["identity_sha256"],
            "archive_sha256": installation["files"]["lib/libninjars.a"],
            "bridge_source_sha256": installation["archived_bridge_source_sha256"],
            "matched_generator_files": matched, "standalone": runtime["standalone"],
            "nlo_poles_tested": runtime["pole_checks_tested"],
            "nlo_poles_passed": runtime["pole_checks_passed"],
            "nlo_pole_tolerance": runtime["pole_test_tolerance"],
            "soft_collinear_checks": runtime["soft_collinear_checks"],
            "scope": "Actual pinned CLI update and installed smoke; relocated internal standalone DP/QP/stability and FKS/NLO checks. NLO pole smoke tolerance is separate from core pairwise tolerance; no shower validation."}


def regression_snapshot(filename, candidate_generator_files, inputs):
    require(policy.path(filename) == REGRESSIONS, "Use the concrete final software authority")
    directory = REGRESSIONS.parent
    verified = capsule(directory, inputs, lambda module: module.main())
    require(verified == {"verified": True, "fresh_passed": 47, "reused_passed": 20,
                         "reused_existing_skip": 1, "actual_reduction_engine_evaluations": 0},
            "Unexpected concrete software verification result")
    summary = read(REGRESSIONS, inputs)
    sources = json.loads(gzip.decompress(retain(directory / "fresh-source-hashes.json.gz", inputs)))
    recorded = dict(sources["repository_files"])
    correction = read(directory / "cfg-test-equivalence.json", inputs)
    # This single retained cfg(test)-only correction is checked by the owner
    # verifier. It does not replace the actual tested source or claim a rerun.
    require(correction["path"] == "vendor/ninjars/src/upstream_master_tests.rs",
            "Unexpected software source-equivalence exception")
    recorded[correction["path"]] = correction["current_sha256"]
    matched = match_generator(recorded, candidate_generator_files)
    for name, checksum in recorded.items():
        retain(ROOT / name, inputs, checksum)
    prior = RELEASE / "regressions"
    manifest = read(prior / "manifest.json", inputs)
    for name, record in manifest["payloads"].items():
        path = (prior / name).resolve()
        require(path.is_relative_to(prior), "Historical software payload leaves its capsule")
        retain(path, inputs, record["sha256"], record["bytes"])
    return {"status": "passed", "qualification": "passed-with-explicit-existing-python3-bypass",
            "tests_run": summary["fresh_tests_run"], "skipped": summary["fresh_tests_skipped"],
            "fresh_tests_passed": summary["fresh_tests_passed"],
            "reused_tests_observed": summary["reused_tests_observed"],
            "reused_tests_passed": summary["reused_tests_passed"],
            "reused_tests_skipped": summary["reused_tests_skipped"],
            "coverage_cases": summary["coverage_cases"], "matched_generator_files": matched,
            "suites": [{"name": "Affected installation/CLI/Fortran-boundary tests", "scope": "fresh", "tests_run": 47, "skipped": 0},
                       {"name": "Exporter and FKS tests", "scope": "retained observation with verified source equivalence; not rerun", "tests_run": 21, "skipped": 1}],
            "evidence": policy.bound_file(REGRESSIONS),
            "scope": "47 fresh software tests and 21 retained exporter/FKS observations (20 passed, one existing GoSam Python3 bypass); zero actual reduction-engine evaluations. Historical polynomial/HELAS counts are not added."}


def numerical_snapshot(filename, installed_precision, runtime_signature, inputs):
    """Keep new-source convergence separate from historical bit identity."""
    require(policy.path(filename) == NUMERICAL, "Use the concrete final numerical authority")
    manifest_raw = retain(NUMERICAL, inputs)
    manifest = json.loads(manifest_raw)
    require(installed_precision is not None and policy.path(installed_precision) ==
            policy.path(manifest["installed_precision_portable"]["path"]),
            "Final numerical controls require their actual installed64 result")
    import ninjars_final_numerical as numerical
    source = Path(numerical.__file__).resolve()
    retain(source, inputs, manifest["verifier_sha256"])
    installation = read(INSTALLATION, inputs)
    metadata = {name: installation["files"]["lib/" + name]
                for name in policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}}
    precision.match_runtime_archive(runtime_signature, installation["files"]["lib/libninjars.a"], metadata,
                                    runtime_signature["identity_sha256"])
    result = numerical.verify(NUMERICAL, installation=installation, inputs=inputs)
    retain(NUMERICAL, inputs, hashlib.sha256(manifest_raw).hexdigest())
    require(result.get("status") == "passed" and result.get("records_each_precision") == 1392 and
            result.get("final_source_convergence_passed") is True and result.get("performance_claim") is False,
            "Final numerical verifier did not validate the complete source-bound convergence stream")
    return {**result, "manifest": policy.bound_file(NUMERICAL),
            "candidate_runtime_identity_sha256": runtime_signature["identity_sha256"],
            "scope": f"All 1,392 original inputs converge at 256/384 for complete Laurent, CC and same-basis R1. The {result['historical_binary128_changed_records']} historical binary128 changes and original exit 101 remain failed bit-identity observations; direct-provider failed controls remain explicit. Actual installed64 and full MadLoop acceptance are separate."}
