"""Read the shared final-rust-abi3-v1 proof without running its executable."""
import hashlib
import json
from pathlib import Path
import re

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(path):
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Evidence path is outside the working repository")
    return {"path": str(path.relative_to(ROOT)), "sha256": digest(path),
            "bytes": path.stat().st_size}


def checked(record):
    path = (ROOT / record["path"]).resolve()
    if not path.is_relative_to(ROOT) or not path.is_file():
        raise ValueError("Missing repository-local evidence: " + str(path))
    if digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError("Changed final Rust evidence: " + str(path))
    return path


def source_inventory(directory):
    return {str(p.relative_to(directory)): digest(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def successful_run(run, executable):
    if run["exit_code"] != 0 or run["executable"] != executable:
        raise ValueError("Final Rust runs must succeed with the same test executable")
    stdout = checked(run["stdout"]).read_text()
    checked(run["stderr"])
    totals = re.findall(
        r"(?m)^test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored; "
        r"\d+ measured; (\d+) filtered out;", stdout)
    if len(totals) != 1:
        raise ValueError("One successful libtest result is required for each named run")
    observed = dict(zip(("passed", "failed", "ignored", "filtered_out"), map(int, totals[0])))
    if observed != run["tests"] or observed["passed"] < 1 or observed["failed"]:
        raise ValueError("Named Rust test totals differ from retained libtest output")
    names = run["observed_test_names"]
    actual = re.findall(r"(?m)^test (\S+) \.\.\. ", stdout)
    if (not names or len(names) != len(set(names)) or set(names) != set(actual)
            or len(actual) != observed["passed"] + observed["ignored"]):
        raise ValueError("Named Rust tests differ from the retained test output")
    passed, ignored = run["passed_test_names"], run["ignored_test_names"]
    if (len(passed) != observed["passed"] or len(ignored) != observed["ignored"]
            or set(passed) & set(ignored) or set(passed) | set(ignored) != set(names)):
        raise ValueError("Named successful/ignored tests do not partition the observed tests")
    headers = list(re.finditer(r"(?m)^test (\S+) \.\.\. ", stdout))
    actual_passed, actual_ignored = [], []
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else stdout.index("test result: ok.")
        tail = stdout[header.end():end]
        if tail.startswith("ignored"):
            actual_ignored.append(header.group(1))
        elif re.search(r"\bok\s*$", tail):
            actual_passed.append(header.group(1))
        else:
            raise ValueError("Missing actual successful Rust status: " + header.group(1))
    if passed != actual_passed or ignored != actual_ignored:
        raise ValueError("Passed/ignored names differ from actual per-test statuses")
    return {"tests": observed, "observed_test_names": names}


def load(proof_path, expected_manifest_sha256, require_executable=False):
    proof_path = proof_path.resolve()
    binding(proof_path)
    manifest_path = proof_path.parent / "proof-manifest.json"
    if not re.fullmatch(r"[0-9a-f]{64}", expected_manifest_sha256):
        raise ValueError("An explicit full final Rust proof-manifest SHA256 is required")
    if digest(manifest_path) != expected_manifest_sha256:
        raise ValueError("Final Rust proof manifest differs from the selected freeze")
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "final-rust-abi3-proof-manifest-v1":
        raise ValueError("Unsupported final Rust proof manifest")
    actual = {str(p.relative_to(proof_path.parent))
              for p in proof_path.parent.rglob("*")
              if p.is_file() and p != manifest_path and "__pycache__" not in p.parts}
    if actual != set(manifest["files"]):
        raise ValueError("Final Rust proof retained-file inventory changed")
    for name, record in manifest["files"].items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid final Rust retained-file path")
        path = proof_path.parent / relative
        if path.is_symlink() or digest(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("Changed final Rust retained file: " + name)
    proof = json.loads(proof_path.read_text())
    if proof["schema"] != "final-rust-abi3-v1" or proof["exit_code"] != 0:
        raise ValueError("Final one-executable Rust suite did not succeed")
    if not re.fullmatch(r"[0-9a-f]{40}", proof["revision"]):
        raise ValueError("Final Rust proof lacks a full committed revision")
    if manifest["revision"] != proof["revision"] or manifest["count"] != len(manifest["files"]):
        raise ValueError("Final Rust manifest revision or file count changed")
    build = json.loads(checked(proof["build_record"]).read_text())
    for name in ("revision", "bridge_source_sha256", "executable"):
        if build[name] != proof[name]:
            raise ValueError("Final Rust build/proof differs: " + name)
    sources = proof["bridge_source_sha256"]
    if not sources or not all(re.fullmatch(r"[0-9a-f]{64}", v) for v in sources.values()):
        raise ValueError("Final Rust proof lacks complete source checksums")
    for name in ("Cargo.toml", "Cargo.lock", "Makefile", "src/lib.rs", "mninjars.f90", "ninjars.h"):
        if name not in sources:
            raise ValueError("Final Rust source map omits " + name)
    executable_path = ROOT / proof["executable"]["path"]
    if require_executable or executable_path.exists():
        checked(proof["executable"])
    runs = proof["named_runs"]
    if set(runs) != {"ordinary", "fixtures", "replay", "convergence"}:
        raise ValueError("Final Rust proof lacks a required named run")
    for run in runs.values():
        if run["revision"] != proof["revision"] or run["bridge_source_sha256"] != sources:
            raise ValueError("Named final Rust run differs from the tested source freeze")
    summary = {name: successful_run(run, proof["executable"])
               for name, run in runs.items()}
    if runs["ordinary"]["tests"]["passed"] < 18:
        raise ValueError("Final ordinary Rust suite does not cover the existing unit_evidence baseline")
    if proof["stdout"] != runs["ordinary"]["stdout"] or proof["stderr"] != runs["ordinary"]["stderr"]:
        raise ValueError("The unit_evidence-compatible top-level logs must be the ordinary run")
    for name in ("fixtures", "replay", "convergence"):
        if runs[name]["summary"].get("passed") is not True:
            raise ValueError("Final Rust named validation did not pass: " + name)
        summary[name]["summary"] = runs[name]["summary"]
    if runs["fixtures"]["summary"] != {"fixtures": 7, "lanes": 21, "calls": 63,
                                        "failed_calls": 0, "passed": True}:
        raise ValueError("Final Rust fixture inventory changed")
    for name in ("replay", "convergence"):
        checked(runs[name]["capture"])
    if runs["replay"]["capture"] != runs["convergence"]["capture"]:
        raise ValueError("Replay and convergence must use the same final ignored-run capture")
    if (runs["replay"]["stdout"] != runs["convergence"]["stdout"] or
            runs["replay"]["stderr"] != runs["convergence"]["stderr"]):
        raise ValueError("Replay and convergence must share one final ignored run")
    replay, convergence = runs["replay"]["summary"], runs["convergence"]["summary"]
    if (replay["records"] != 1392 or replay["fields"] != 11136 or
            replay["historical_binary128_failures"] != 0):
        raise ValueError("Final full-stream binary128 replay did not pass")
    required_convergence = {"records": 1392, "fields_per_precision": 11136,
        "production_bits": 256, "reference_bits": 384, "tolerance": "1e-31",
        "failed_records": 0, "raw_comparisons_recomputed": True}
    if any(convergence[key] != value for key, value in required_convergence.items()):
        raise ValueError("Final full-stream 256/384 convergence did not pass")
    # Preserve branch-lip differences as a separate review condition. Passing
    # an install/linkage check never declares them semantically equivalent.
    summary["requires_review"] = convergence["sign_changed_records"] != 0
    summary.update(revision=proof["revision"], executable=proof["executable"],
                   manifest_sha256=expected_manifest_sha256)
    return proof, summary
