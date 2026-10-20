#!/usr/bin/env python3
"""Read-only analysis of retained profiles and the frozen sparse updater."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
HERE = Path(__file__).resolve().parent
PROFILE = ROOT / ".benchmark-work/experiments/mp-qcd3-point44/profile.perf"
OLD = ROOT / ".benchmark-work/experiments/mp-qcd3-point44/qcd3"
NEW = ROOT / ".benchmark-work/policy-v2/generation/outputs/candidate/qcd-3"
PERF = "/nix/store/7ccpnz8xkn5qsyw8nkz998vjmb6jpl49-perf-linux-6.19.6/bin/perf"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    old_poly = next(OLD.glob("SubProcesses/P*/polynomial.f"))
    new_poly = next(NEW.glob("SubProcesses/P*/polynomial.f"))
    command = [PERF, "script", "--no-inline", "-i", str(PROFILE)]
    process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, text=True)
    owners, leaves, memory = Counter(), defaultdict(Counter), Counter()
    samples = qp_samples = 0

    def consume(lines):
        nonlocal samples, qp_samples
        if not lines:
            return
        samples += 1
        stack = "\n".join(lines)
        if not any(name in stack for name in
                   ("mp_compute_loop_coefs_", "mp_coef_construction_")):
            return
        qp_samples += 1
        names = [re.sub(r"^\s*[0-9a-f]+ ", "", line).split("+")[0]
                 for line in lines[1:]]
        owner = next((name for name in names if re.match(
            r"(mp_ml5_.*(?:update_wl|create_loop_coefs|merge_wl)|mp_\w+|"
            r"ml5_\w+_mp_coef_construction)", name)), "unresolved")
        owners[owner] += 1
        leaves[owner][names[0] if names else "unresolved"] += 1
        if any(re.search(r"(malloc|realloc|calloc|free|memcpy|memmove|memset)",
                         name, re.I) for name in names):
            memory[owner] += 1

    block = []
    for line in process.stdout:
        if line.strip():
            block.append(line.rstrip())
        else:
            consume(block)
            block = []
    consume(block)
    if process.wait():
        raise RuntimeError("Retained perf decoding failed")
    old_source, new_source = old_poly.read_text(), new_poly.read_text()
    assembly = subprocess.check_output(
        ["objdump", "-dr", str(new_poly.with_suffix(".o"))], text=True, cwd=ROOT)
    calls, symbol = defaultdict(Counter), None
    for line in assembly.splitlines():
        match = re.match(r"^[0-9a-f]+ <([^>]+)>:", line)
        if match:
            symbol = match[1]
        relocation = re.search(r"R_X86_64_PLT32\s+(\S+?)(?:-0x4)?$", line)
        if symbol and symbol.startswith("mp_") and relocation:
            calls[symbol][relocation[1]] += 1
    report = {
        "scope": "Read-only old profile attribution and final-v2 generated source/object inspection. "
                 "The old profile predates sparse QP and cannot measure its remaining cost.",
        "profile_command": command,
        "profile_sha256": digest(PROFILE),
        "old_polynomial_sha256": digest(old_poly),
        "new_polynomial_sha256": digest(new_poly),
        "new_polynomial_object_sha256": digest(new_poly.with_suffix(".o")),
        "old_has_sparse_guard": "WL_NEAREST" in old_source,
        "new_has_sparse_guard": "WL_NEAREST" in new_source,
        "samples": samples, "fortran_qp_samples": qp_samples,
        "owners": [{"name": name, "samples": count,
                    "percent_all_samples": 100 * count / samples,
                    "percent_fortran_qp_samples": 100 * count / qp_samples,
                    "top_leaves": leaves[name].most_common(8),
                    "memory_callchain_samples": memory[name]}
                   for name, count in owners.most_common()],
        "new_object_static_call_sites": dict(calls),
        "new_source_contains_allocate": bool(re.search(r"\bALLOCATE\s*\(", new_source, re.I)),
        "new_source_contains_allocatable": bool(re.search(r"\bALLOCATABLE\b", new_source, re.I)),
    }
    (HERE / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                     ("samples", "fortran_qp_samples", "old_has_sparse_guard",
                      "new_source_contains_allocate", "new_source_contains_allocatable")}))
    for item in report["owners"][:12]:
        print(item["name"], item["samples"], round(item["percent_fortran_qp_samples"], 2))


if __name__ == "__main__":
    main()
