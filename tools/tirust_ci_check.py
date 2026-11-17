#!/usr/bin/env python3
"""End-to-end TIRust check for CI on Linux and macOS.

  tools/tirust_ci_check.py --work DIR (--prefix DIR | --tirust LIBDIR) [--jobs N]

1. With --prefix, runs ``install tirust --prefix=DIR`` through MadGraph. The
   installer builds the bridge and runs its DP, QP and poles-only smoke test.
2. Exports ``u u~ > z [virt=QCD]`` as standalone output against that
   installation, builds the check driver, and runs it with MLReductionLib=9
   (TIRust) and then 1 (CutTools).
3. Requires MadLoop return code 213 from TIRust, and agreement of the Born,
   finite part and both poles with CutTools within 1e-9 relative.

The script never saves MadGraph options; it passes them with --no_save.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROCESS = "u u~ > z [virt=QCD]"
TOLERANCE = 1e-9
RESULTS = {"born": "Matrix element born", "finite": "Matrix element finite",
           "single pole": "Matrix element 1eps", "double pole": "Matrix element 2eps"}


def madgraph(lines: list[str], log: Path) -> str:
    script = log.with_suffix(".mg5")
    script.write_text("\n".join(lines + ["quit"]) + "\n")
    result = subprocess.run([sys.executable, str(ROOT / "bin/madgraph"), str(script)],
                            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, stdin=subprocess.DEVNULL)
    log.write_text(result.stdout)
    if result.returncode:
        sys.exit(f"MadGraph failed ({result.returncode}); see {log}")
    return result.stdout


def set_reduction(card: Path, value: str) -> None:
    lines = card.read_text().splitlines(keepends=True)
    index = next(i for i, line in enumerate(lines) if line.startswith("#MLReductionLib"))
    lines[index + 1] = value + "\n"
    card.write_text("".join(lines))


def run_check(directory: Path) -> dict:
    output = subprocess.run(["./check"], cwd=directory, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, timeout=600).stdout
    values = {}
    code = re.search(r"MadLoop return code\s*=\s*(\d+)", output)
    values["return code"] = int(code.group(1)) if code else None
    for key, label in RESULTS.items():
        match = re.search(re.escape(label) + r"\s*=\s*(\S+)", output)
        values[key] = float(match.group(1).replace("D", "E")) if match else None
    if None in values.values():
        print(output)
        sys.exit(f"Could not read the check output in {directory}")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prefix", type=Path, help="install TIRust into this prefix first")
    source.add_argument("--tirust", type=Path, help="use this installed TIRust lib directory")
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)

    if args.prefix:
        prefix = args.prefix.resolve()
        output = madgraph([f"install tirust --prefix={prefix} --jobs={args.jobs}"],
                          work / "install.log")
        if "TIRust successfully installed" not in output:
            sys.exit(f"install tirust did not succeed; see {work / 'install.log'}")
        libdir = prefix / "tirust/lib"
    else:
        libdir = args.tirust.resolve()
    print(f"TIRust: {libdir}")

    export = work / "uuz"
    # 'set ninja None' skips the first-loop-output question about installing
    # other reducers, which a fresh checkout would otherwise ask.
    madgraph([f"set tirust {libdir} --no_save", "set ninja None --no_save",
              "import model loop_sm", f"generate {PROCESS}",
              f"output standalone {export} -f"], work / "output.log")
    process = export / "SubProcesses/P0_uux_z"
    make = ["make", "check"]
    if not Path("/bin/bash").exists():
        make.append("SHELL=" + shutil.which("bash"))
    with open(work / "build.log", "w") as log:
        if subprocess.run(make, cwd=process, stdout=log, stderr=subprocess.STDOUT).returncode:
            sys.exit(f"Building the check driver failed; see {work / 'build.log'}")

    card = (process / "MadLoop5_resources/MadLoopParams.dat").resolve()
    results = {}
    for name, value in (("TIRust", "9"), ("CutTools", "1")):
        set_reduction(card, value)
        results[name] = run_check(process)
        print(f"{name:9s}", results[name])
    failures = []
    if results["TIRust"]["return code"] != 213:
        failures.append(f"TIRust return code {results['TIRust']['return code']}, expected 213")
    for key in RESULTS:
        tirust, reference = results["TIRust"][key], results["CutTools"][key]
        error = abs(tirust - reference) / max(abs(reference), 1e-300)
        print(f"{key:12s} relative difference {error:.2e}")
        if error > TOLERANCE:
            failures.append(f"{key}: TIRust {tirust!r}, CutTools {reference!r}")
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        return 1
    print("TIRust agrees with CutTools.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
