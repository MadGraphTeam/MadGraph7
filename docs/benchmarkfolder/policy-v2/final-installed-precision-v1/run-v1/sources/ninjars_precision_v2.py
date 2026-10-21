#!/usr/bin/env python3
"""Recompute final installed-archive fixture/precision evidence from raw calls."""
import argparse
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_supplement_v2 as supplement

ROOT = bench.ROOT
SCHEMA = "installed-ninjars-precision-v1"
MANDATORY_TESTS = (
    "tadpole_rebinds_coefficients_masses_and_scale",
    "invalid_input_preserves_outputs_and_reports_error",
    "version_and_malformed_outputs_are_transactional",
    "every_short_buffer_is_rejected_before_access_or_output",
    "mp_transport_preserves_binary128_information",
    "complete_mp_tadpole_retains_input_precision",
    "mp_bubbles_cover_massless_orderings_and_context_rebinding",
    "complex_mass_pentagon_converges_at_binary128_output_precision",
    "mp_transport_preserves_binary128_finite_dyadics_and_sign",
    "mp_transport_preserves_complex_branch_lips",
)


def read(path):
    return json.loads(Path(path).read_text())


def checked(record, inputs=None):
    return supplement.checked(record, inputs)


def match_runtime_archive(signature, archive_sha256, metadata, runtime_identity=None):
    """A supplied identity cannot relabel a stale archive or interface."""
    policy.verified(signature, "reducer-runtime-signature-v2")
    if runtime_identity is not None and signature["identity_sha256"] != runtime_identity:
        raise ValueError("Precision proof belongs to another measured runtime")
    archives = {value for name, value in signature["libraries"].items() if Path(name).name == "libninjars.a"}
    if archives != {archive_sha256}:
        raise ValueError("Actual installed precision archive differs from measured NinjaRS")
    required = policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}
    for name in required:
        recorded = {value for path, value in signature["metadata"].items() if Path(path).name == name}
        if name not in metadata or recorded != {metadata[name]}:
            raise ValueError("Actual precision interface/link metadata differs from measured NinjaRS: "+name)
    return signature


def unit_evidence(proof_path, installation_path, inputs=None):
    proof = read(proof_path)
    install = read(installation_path)
    for name in ("stdout", "stderr", "build_record"):
        checked(proof[name], inputs)
    if proof["exit_code"] != 0:
        raise ValueError("Final Rust suite did not exit successfully")
    build = read(checked(proof["build_record"], inputs))
    sources = proof["bridge_source_sha256"]
    if (sources != install["archived_bridge_source_sha256"] or
            build["bridge_source_sha256"] != sources or
            build["executable"] != proof["executable"]):
        raise ValueError("Private Rust precision tests and installed archive have different source")
    # The source/executable hashes were captured by the build controller while
    # the artifact existed; retain the build record after cleanup.  If the
    # executable still exists, check it again without requiring shipping ELF.
    executable = policy.path(proof["executable"]["path"])
    if executable.exists():
        checked(proof["executable"], inputs)
    stdout = checked(proof["stdout"], inputs).read_text()
    totals = re.findall(r"test result: ok\. (\d+) passed; 0 failed;", stdout)
    if not totals or max(map(int, totals)) < 18:
        raise ValueError("Final Rust suite lacks a complete successful unit result")
    for name in MANDATORY_TESTS:
        pattern = r"test [^\n ]*"+re.escape(name)+r" \.\.\. (?:(?!\ntest ).)*?\bok(?:\n|$)"
        if not re.search(pattern, stdout, re.S):
            raise ValueError("Required final transport/lifecycle test did not pass: "+name)
    lines = [line.split("NINJARS_CONVERGENCE ", 1)[1]
             for line in stdout.splitlines() if "NINJARS_CONVERGENCE " in line]
    if len(lines) != 1:
        raise ValueError("Exactly one raw final-source convergence record is required")
    convergence = json.loads(lines[0])
    if (convergence["production_bits"] != 256 or convergence["reference_bits"] != 384 or
            convergence["convergence_tolerance"] != "1e-31"):
        raise ValueError("Final convergence precision/threshold changed")
    if not verify_convergence(convergence["result_256"], convergence["result_384"])["passed"]:
        raise ValueError("Final-source 256/384 convergence failed")
    return proof, install, convergence, int(max(map(int, totals)))


def verify_convergence(values, reference):
    if len(values) != 8 or len(reference) != 8:
        raise ValueError("Incomplete 256/384 output fields")
    with localcontext() as ctx:
        ctx.prec = 180
        left, right = list(map(Decimal, values)), list(map(Decimal, reference))
        if any(not x.is_finite() for x in left+right):
            raise ValueError("Nonfinite convergence output")
        # R1 is a basis-dependent diagnostic.  A large rational/CC
        # cancellation must never loosen the complete Laurent comparison.
        scale = max(map(abs, right[:6]))
        if not scale:
            raise ValueError("Convergence fixture has zero normalization")
        differences = [abs(x-y) for x, y in zip(left, right)]
        errors = [x/scale for x in differences]
        return {"passed": all(x <= Decimal("1e-31") for x in errors),
                "maximum_scaled_error": str(max(errors[:6])), "tolerance": "1e-31",
                "normalization_scope": "largest complete finite/single/double Laurent component; excludes R1",
                "rational_maximum_scaled_difference": str(max(errors[6:])),
                "rational_comparison": "same generic basis at256/384bits, checked separately with the stricter physical Laurent scale",
                "fields": len(left), "field_errors": [str(x) for x in errors]}


def fixture_requests(data):
    provenance = read(data / "provenance.json")
    for item in provenance["files"]:
        if supplement.digest(data / item["file"]) != item["sha256"]:
            raise ValueError("Changed retained tensor fixture: "+item["file"])
    requests = []
    for case_path in sorted((data / "cases").glob("*.json")):
        case, reference = read(case_path), read(data / "ninja" / case_path.name)
        references = {r["id"]: r for r in reference["lanes"]}
        for lane in case["lanes"]:
            for mode in (0, 1, 2):
                flat = lambda key: [x for row in lane[key] for x in row]
                request = {"record": len(requests), "case": case["name"], "lane": lane["id"],
                           "mode": mode, "n": case["n"], "rank": case["rank"],
                           "coefficients": flat("tensor"), "s_matrix": flat("s_matrix"),
                           "momenta": flat("shifts"), "masses_squared": flat("masses_squared"),
                           "scale_squared": [lane["mu_squared"]], "reference": references[lane["id"]]}
                if mode == 2:
                    for key in ("coefficients", "s_matrix", "momenta", "masses_squared", "scale_squared"):
                        request[key] = [format(float(x), ".39e") for x in request[key]]
                requests.append(request)
    if len(requests) != 63 or len({r["case"] for r in requests}) != 7:
        raise ValueError("Final fixture inventory must contain all seven cases/21 lanes/three modes")
    return requests


def encode(requests):
    result = []
    for index, row in enumerate(requests):
        if row["record"] != index:
            raise ValueError("Disordered precision input")
        result.append(f"{row['mode']} {row['n']} {row['rank']} {len(row['coefficients'])//2}")
        for key in ("coefficients", "s_matrix", "momenta", "masses_squared", "scale_squared"):
            for value in row[key]:
                result.append(value if isinstance(value, str) else format(float(value), ".17e"))
    return "\n".join(result)+"\n"


def prepare(specification):
    spec = read(specification)
    report, work = policy.path(spec["report"]), policy.path(spec["work"])
    if report.exists() or work.exists():
        raise ValueError("Precision evidence destination already exists")
    installation_path = checked(spec["installation"])
    proof_path = checked(spec["unit_proof"])
    proof, install, convergence, count = unit_evidence(proof_path, installation_path)
    installed = policy.path(install["installed"])
    if (install["exit_code"] != 0 or install.get("reference_source_overrides", True) or
            install["abi_version"] != 3 or
            supplement.digest(installed / "lib/libninjars.a") != install["files"]["lib/libninjars.a"]):
        raise ValueError("Installed archive differs from tested CLI installation")
    if supplement.digest(installed / "include/ninjars.h") != proof["bridge_source_sha256"]["ninjars.h"]:
        raise ValueError("Installed ABI header differs from final tested source")
    signature = spec["candidate_runtime_signature"]
    metadata_hashes = {name: supplement.digest(installed / "lib" / name)
                       for name in policy.RUNTIME_METADATA_NAMES | {"mninjars.mod"}}
    match_runtime_archive(signature, install["files"]["lib/libninjars.a"], metadata_hashes,
                          spec["candidate_runtime_identity_sha256"])
    requests = fixture_requests(ROOT / "vendor/ninjars/tests/data")
    requests.append({"record": len(requests), "mode": 2, "case": "beyond-dp-complex-pentagon-convergence",
                     "lane": convergence["lane"], **convergence["input"],
                     "reference_256": convergence["result_256"], "reference_384": convergence["result_384"]})
    report.mkdir(parents=True)
    work.mkdir(parents=True)
    for name in ("sources", "inputs"):
        (report / name).mkdir()
    for name in ("ninjars_precision_v2.py", "ninjars_precision_driver.c", "ninjars_supplement_v2.py"):
        shutil.copy2(ROOT / "tools" / name, report / "sources" / name)
    bench.dump(report / "requests.json", requests)
    (report / "requests.txt").write_text(encode(requests))
    shutil.copy2(specification, report / "input-spec.json")
    for name, source in (("unit-proof.json", proof_path), ("installation.json", installation_path)):
        shutil.copy2(source, report / "inputs" / name)
    installation_cli = {}
    for name, key in (("install-cli.log", "cli_log_sha256"), ("install.mg5", "cli_card_sha256")):
        source = installation_path.parent / name
        if supplement.digest(source) != install[key]:
            raise ValueError("Installed precision archive lacks its exact actual-CLI proof")
        destination = report / "inputs" / name
        shutil.copy2(source, destination)
        installation_cli[key] = supplement.bound(destination)
    runtime_metadata = {}
    for name in sorted(metadata_hashes):
        source = installed / "lib" / name
        destination = report / "inputs" / name
        shutil.copy2(source, destination)
        runtime_metadata[name] = supplement.bound(destination)
    compiler = shutil.which("cc")
    command = [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror",
               "-I"+str(installed / "include"), str(report / "sources/ninjars_precision_driver.c"),
               "-o", str(work / "precision-driver")]
    command += shlex.split((installed / "lib/ninjars-link-before.txt").read_text())
    command += [str(installed / "lib/libninjars.a")]
    command += shlex.split((installed / "lib/ninjars-native-libs.txt").read_text())
    seconds = bench.run(command, ROOT, report / "build.log")
    record = policy.seal({"schema": SCHEMA, "status": "prepared-not-executed",
                          "installation": supplement.bound(report / "inputs/installation.json"),
                          "unit_proof": supplement.bound(report / "inputs/unit-proof.json"),
                          "requests": supplement.bound(report / "requests.json"),
                          "request_stream": supplement.bound(report / "requests.txt"),
                          "archive_sha256": install["files"]["lib/libninjars.a"],
                          "candidate_runtime_identity_sha256": spec["candidate_runtime_identity_sha256"],
                          "candidate_runtime_signature": signature,
                          "runtime_metadata": runtime_metadata,
                          "bridge_source_sha256": proof["bridge_source_sha256"],
                          "installation_cli": installation_cli,
                          "unit_passed_tests": count, "executable": supplement.bound(work / "precision-driver"),
                          "compile_command": command, "compile_seconds": seconds,
                          "sources": {name: supplement.bound(report / "sources" / name) for name in ("ninjars_precision_v2.py", "ninjars_precision_driver.c", "ninjars_supplement_v2.py")}})
    bench.dump(report / "prepared.json", record)
    return record


def compare_requests(requests, outputs):
    if len(outputs) != len(requests):
        raise ValueError("Incomplete installed precision output")
    checks, failures, worst = [], [], 0.
    for expected, actual in zip(requests, outputs):
        if actual["record"] != expected["record"] or actual["mode"] != expected["mode"]:
            raise ValueError("Installed precision output reordered")
        record = {"record": expected["record"], "case": expected["case"], "lane": expected["lane"],
                  "mode": expected["mode"], "status": actual["status"]}
        if actual["status"] != 0 or len(actual["output"]) != 8:
            failures.append({**record, "reason": "Failed public-ABI reduction", "raw": actual})
            checks.append(record)
            continue
        if "reference_384" in expected:
            record.update(verify_convergence(actual["output"], expected["reference_384"]))
            record["same_final_source_256_fields"] = actual["output"] == expected["reference_256"]
            if not record["passed"] or not record["same_final_source_256_fields"]:
                failures.append(record)
        else:
            values = [float(x) if x is not None else math.nan for x in actual["output"]]
            errors, rational = [], None
            for i, key in enumerate(("eps0", "eps_m1", "eps_m2", "rational")):
                x = complex(*values[2*i:2*i+2])
                y = complex(*expected["reference"][key])
                error = abs(x-y)/max(1., abs(y))
                if not math.isfinite(error):
                    failures.append({**record, "component": key, "reason": "Nonfinite output"})
                if key == "rational" and expected["mode"] == 2:
                    rational = error
                else:
                    errors.append(error)
                    if error > 1e-9:
                        failures.append({**record, "component": key, "scaled_error": error, "tolerance": 1e-9})
            maximum = max(errors)
            worst = max(worst, maximum)
            record.update(maximum_scaled_error=maximum, tolerance=1e-9,
                          rational_compared=expected["mode"] != 2,
                          basis_dependent_rational_scaled_difference=rational)
        checks.append(record)
    return {"status": "failed" if failures else "passed", "checks": checks,
            "failures": failures, "maximum_fixture_scaled_error": worst,
            "fixture_cases": 7, "fixture_lanes": 21, "fixture_modes": ["native-dp", "compensated-dp", "genuine256"],
            "working_bits": 256, "convergence_bits": 384,
            "cross_basis_oracle": "Full finite/single/double Laurent coefficients; MP rational split reported separately"}


def execution_summary(requests, stdout, exit_code):
    """A crashed/incomplete campaign remains a published failed observation."""
    outputs, parser_error = [], None
    for line in stdout.splitlines():
        if line.startswith("NINJARS_INSTALLED "):
            try:
                outputs.append(json.loads(line[len("NINJARS_INSTALLED "):]))
            except (ValueError, TypeError) as error:
                parser_error = str(error)
                break
    try:
        summary = compare_requests(requests[:len(outputs)], outputs) if outputs and len(outputs) <= len(requests) else {
            "status": "failed", "checks": [], "failures": [], "maximum_fixture_scaled_error": None,
            "fixture_cases": 7, "fixture_lanes": 21, "working_bits": 256, "convergence_bits": 384}
    except (ValueError, TypeError, KeyError) as error:
        summary = {"status": "failed", "checks": [], "failures": [{"reason": str(error)}],
                   "working_bits": 256, "convergence_bits": 384}
    if len(outputs) != len(requests) or parser_error is not None or exit_code != 0:
        summary["status"] = "failed"
        summary["failures"].append({"reason": "Incomplete, malformed or failed installed public-ABI execution",
                                    "expected_calls": len(requests), "observed_calls": len(outputs),
                                    "exit_code": exit_code, "parser_error": parser_error})
    return summary


def validate(filename, runtime_identity=None, inputs=None, runtime_signature=None):
    result = policy.verified(read(filename), SCHEMA)
    if result["status"] != "executed":
        raise ValueError("Precision campaign was not executed")
    prepared = policy.verified(read(checked(result["prepared"], inputs)), SCHEMA)
    if policy.path(prepared["executable"]["path"]).exists():
        checked(prepared["executable"], inputs)
    if runtime_identity is not None and prepared["candidate_runtime_identity_sha256"] != runtime_identity:
        raise ValueError("Precision controls use another installed runtime")
    if runtime_signature is not None and prepared["candidate_runtime_signature"] != runtime_signature:
        raise ValueError("Precision runtime descriptor differs from final measured runtime")
    proof, install, convergence, count = unit_evidence(checked(prepared["unit_proof"], inputs), checked(prepared["installation"], inputs), inputs)
    if prepared["archive_sha256"] != install["files"]["lib/libninjars.a"]:
        raise ValueError("Precision archive changed after preparation")
    metadata_hashes = {name: record["sha256"] for name, record in prepared["runtime_metadata"].items()}
    for record in prepared["runtime_metadata"].values():
        checked(record, inputs)
    match_runtime_archive(prepared["candidate_runtime_signature"], prepared["archive_sha256"],
                          metadata_hashes, runtime_identity)
    for key, record in prepared["installation_cli"].items():
        if record["sha256"] != install[key]:
            raise ValueError("Precision installation CLI identity changed")
        checked(record, inputs)
    for item in prepared["sources"].values():
        checked(item, inputs)
    requests = read(checked(prepared["requests"], inputs))
    expected = fixture_requests(ROOT / "vendor/ninjars/tests/data")
    expected.append({"record": len(expected), "mode": 2, "case": "beyond-dp-complex-pentagon-convergence",
                     "lane": convergence["lane"], **convergence["input"],
                     "reference_256": convergence["result_256"], "reference_384": convergence["result_384"]})
    if requests != expected:
        raise ValueError("Precision inputs/references differ from the actual fixtures/final Rust oracle")
    if checked(prepared["request_stream"], inputs).read_text() != encode(requests):
        raise ValueError("Precision ASCII request stream differs from exact inputs")
    stdout = checked(result["stdout"], inputs).read_text()
    checked(result["stderr"], inputs)
    libraries = read(checked(result["loaded_libraries"], inputs))
    stderr = checked(result["stderr"], inputs).read_text()
    paths = sorted({line[len("NINJARS_LOADED "):] for line in stderr.splitlines()
                    if line.startswith("NINJARS_LOADED ")})
    if paths != sorted(libraries):
        raise ValueError("Actual loaded-library inventory differs from driver runtime report")
    if result["exit_code"] == 0 and not any(name.startswith("/") and "libc.so" in Path(name).name for name in paths):
        raise ValueError("Actual precision process lacks its resolved system-library inventory")
    for name, record in libraries.items():
        if name.startswith("/") and Path(name).exists() and supplement.digest(name) != record["sha256"]:
            raise ValueError("Precision driver shared library changed: "+name)
    summary = execution_summary(requests, stdout, result["exit_code"])
    if summary != result["summary"]:
        raise ValueError("Precision conclusion differs from raw installed results")
    raw_checks = summary.pop("checks")
    fixture_ok = len(raw_checks) >= 63 and not any(item.get("record", -1) < 63 for item in summary["failures"])
    convergence_ok = len(raw_checks) == 64 and not any(item.get("record", 63) == 63 for item in summary["failures"])
    summary.update(checks=[
                       {"name": "Complete source-bound bridge Rust suite, including transport/domain/reset/lifecycle", "status": "passed", "tests": count},
                       {"name": "Actual installed native/compensated DP and256-bit fixture replay", "status": "passed" if fixture_ok else "failed", "cases": 7, "lanes": 21, "calls": 63},
                       {"name": "Actual installed beyond-DP complex pentagon versus final-source256/384-bit results", "status": "passed" if convergence_ok else "failed", "calls": 1}],
                   raw_call_checks=raw_checks,
                   final_source_unit_tests=count, archive_sha256=prepared["archive_sha256"],
                   candidate_runtime_identity_sha256=prepared["candidate_runtime_identity_sha256"],
                   source_bound_private_384_oracle=True, actual_installed_public_256_replay=True,
                   identity_sha256=result["identity_sha256"])
    return summary


def execute(prepared_path, cpu):
    prepared_path = policy.path(prepared_path)
    prepared = policy.verified(read(prepared_path), SCHEMA)
    report = prepared_path.parent
    if (report / "result.json").exists():
        raise ValueError("Precision execution already retained")
    executable = checked(prepared["executable"])
    request_stream = checked(prepared["request_stream"])
    exit_code = None
    with request_stream.open() as stdin, (report / "stdout.log").open("x") as stdout, (report / "stderr.log").open("x") as stderr:
        try:
            process = subprocess.run(["taskset", "-c", str(cpu), str(executable)], cwd=ROOT,
                                     stdin=stdin, stdout=stdout, stderr=stderr, timeout=86400.)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            stderr.write("Installed precision execution reached its declared86400-second timeout.\n")
    raw = (report / "stdout.log").read_text()
    loaded = {}
    for line in (report / "stderr.log").read_text().splitlines():
        if not line.startswith("NINJARS_LOADED "):
            continue
        name = line[len("NINJARS_LOADED "):]
        path = Path(name)
        if name.startswith("/"):
            loaded[name] = {"sha256": supplement.digest(path), "bytes": path.stat().st_size}
        elif name == "linux-vdso.so.1":
            loaded[name] = {"virtual_kernel_object": True}
        else:
            raise ValueError("Actual loaded precision library has an unresolved path: "+name)
    bench.dump(report / "loaded-libraries.json", loaded)
    summary = execution_summary(read(checked(prepared["requests"])), raw, exit_code)
    result = policy.seal({"schema": SCHEMA, "status": "executed", "prepared": supplement.bound(prepared_path),
                          "cpu": cpu, "exit_code": exit_code,
                          "stdout": supplement.bound(report / "stdout.log"), "stderr": supplement.bound(report / "stderr.log"),
                          "loaded_libraries": supplement.bound(report / "loaded-libraries.json"),
                          "summary": summary})
    bench.dump(report / "result.json", result)
    return validate(report / "result.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "execute", "validate"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--cpu", type=int, default=2)
    parser.add_argument("--execute-numerical", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        result = prepare(args.path)
    elif args.action == "execute":
        if not args.execute_numerical:
            parser.error("Execution requires the explicit serialized-MP handoff flag")
        result = execute(args.path, args.cpu)
    else:
        result = validate(args.path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
