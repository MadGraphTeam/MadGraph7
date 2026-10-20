#!/usr/bin/env python3
"""Reproduce the GNU Fortran/Rust compiler-runtime isolation check (no MP calls)."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / ".benchmark-work/experiments/quad-runtime-verification"
LIB = ROOT / "HEPTools/ninjars/lib"


def main():
    BUILD.mkdir(parents=True, exist_ok=True)
    native = shlex.split((LIB / "ninjars-native-libs.txt").read_text())
    before = shlex.split((LIB / "ninjars-link-before.txt").read_text())
    outputs = {}
    commands = []
    symbols = {}
    for source in ("quad-rounding", "quad-helpers"):
        for variant in ("baseline", "rust-first", "gcc-first"):
            target = BUILD / f"{source}-{variant}"
            command = ["gfortran", "-O3", "-frounding-math", str(HERE / f"{source}.f90")]
            if variant == "baseline":
                command += [str(HERE / "abi-stub.f90")]
            else:
                if variant == "gcc-first":
                    command += before
                command += [str(LIB / "libninjars.a")] + native
            command += ["-o", str(target)]
            commands.append(command)
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True)
            run = subprocess.run([str(target)], cwd=ROOT, check=True,
                                 capture_output=True, text=True)
            outputs[f"{source}-{variant}"] = run.stdout
            (HERE / f"{source}-{variant}.txt").write_text(run.stdout)
            nm = subprocess.check_output(["nm", str(target)], text=True)
            symbols[f"{source}-{variant}"] = [
                line for line in nm.splitlines()
                if any(name in line for name in (
                    "__addtf3", "__subtf3", "__multf3", "__divtf3",
                    "__trunctfdf2", "__extenddftf2", "__floatditf"))]
        assert outputs[f"{source}-gcc-first"] == outputs[f"{source}-baseline"]
        assert outputs[f"{source}-rust-first"] != outputs[f"{source}-baseline"]
    result = {
        "passed": True,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "target": subprocess.check_output(["gfortran", "-dumpmachine"], text=True).strip(),
        "commands": commands,
        "installed_archive_sha256": hashlib.sha256((LIB / "libninjars.a").read_bytes()).hexdigest(),
        "prefix_flags": before, "suffix_flags": native, "symbols": symbols,
        "coverage": "Four IEEE rounding modes; quad arithmetic/comparisons/conversions, "
                    "quad complex multiplication/division, tiny/huge inputs, exception flags.",
        "interpretation": "Prefix restores GNU baseline output exactly. Raw archive link "
                          "changes directed rounding. This is correctness evidence, not a speed claim."}
    (HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
    print("GNU baseline parity and Rust-first directed-rounding regression confirmed.")


if __name__ == "__main__":
    main()
