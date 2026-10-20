#!/usr/bin/env python3
"""Validate final-source internal exports after private CLI installation.

Run from the repository root in the pinned development shell. Export the two
processes with the adjacent MG5 cards first. --build never invokes NinjaRS;
--run uses the shared Symbolica slot and must be scheduled with other MP jobs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import prepare

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
REPORT = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/policy-v2/install-export-abi3-zero-sign-v1"
SA = WORK / "standalone-relocated"
NLO = WORK / "nlo-relocated"
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import MadLoopParam


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(name, value):
    (REPORT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def hide_installation():
    original = WORK / "source/HEPTools/ninjars"
    hidden = original.with_name("ninjars-unavailable")
    if original.exists():
        assert not hidden.exists(), "Existing hidden installation"
        original.rename(hidden)
    assert not original.exists() and hidden.is_dir()
    installed = json.loads((REPORT / "installation.json").read_text())
    for name, checksum in installed["files"].items():
        assert digest(hidden / name) == checksum, name
    interfaces = {}
    for output in (SA, NLO):
        found = []
        for path in output.rglob("mninjars.f90"):
            assert path.is_file() and not path.is_symlink(), path
            found.append({"path": str(path.relative_to(ROOT)), "sha256": digest(path)})
        assert found, output
        interfaces[output.name] = found
    dump("installation-hidden.json", {
        "original_installation_path_absent": True,
        "retained_at": str(hidden.relative_to(ROOT)), "copied_interfaces": interfaces,
    })


def environment(external=False):
    result = os.environ.copy()
    for key in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        result.pop(key, None)
    result.update(
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        XDG_CONFIG_HOME=str(WORK / "config"),
        MAKEFLAGS="SHELL=" + shutil.which("bash"),
        SYMBOLICA_LOCK_DIR=str(ROOT / ".ninjars-build/symbolica-locks"),
    )
    # Both outputs copy their dependencies; no legacy shared-library path is used.
    return result


def command(args, cwd, log, external=False):
    with (REPORT / log).open("w") as stream:
        result = subprocess.run(args, cwd=cwd, env=environment(external), stdout=stream,
                                stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{args!r} failed; see {log}")
    return (REPORT / log).read_text()


def configure(output, mode=-1):
    if output == NLO:
        run_card = output / "Cards/run_card.dat"
        text, count = re.subn(r"(?m)^\s*\S+\s*=\s*pdlabel\b",
                              " nn23lo1 = pdlabel", run_card.read_text(), count=1)
        assert count == 1
        run_card.write_text(text)
    card = MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
    for key, value in {
        "MLReductionLib": "8",
        "CTModeRun": mode,
        "MLStabThres": 1e-9,
        "COLLIERRequiredAccuracy": 1e-12,
    }.items():
        card[key] = value
    paths = [output / "Cards/MadLoopParams.dat"]
    paths += list((output / "SubProcesses").rglob("MadLoopParams.dat"))
    for path in paths:
        card.write(str(path), template=str(
            ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    for name in ("MadLoopParams.dat", "param_card.dat", "run_card.dat"):
        path = output / "Cards" / name
        if path.exists():
            shutil.copy2(path, REPORT / (output.name + "-" + name))


def symbols(executable, label):
    raw = subprocess.check_output(["nm", "-D", str(executable)], cwd=ROOT,
                                  env=environment(), text=True)
    (WORK / (label + "-nm-full.txt")).write_text(raw)
    selected = [line for line in raw.splitlines()
                if re.search(r"\b__(?:[a-z0-9]*tf[a-z0-9]*)(?:@|$)", line)]
    (REPORT / (label + "-nm.txt")).write_text("\n".join(selected) + "\n")
    helpers = [line for line in raw.splitlines()
               if re.search(r"__(?:add|sub|mul|div)tf3(?:@|$)", line)]
    assert helpers and all(re.search(r"\bU __\w+@GCC_", line) for line in helpers), helpers
    linked = command(["ldd", str(executable)], ROOT, label + "-ldd.txt",
                     external=label.startswith("nlo-"))
    assert "not found" not in linked
    assert "libgcc_s.so" in linked
    return {"executable_sha256": digest(executable),
            "symbol_command": ["nm", "-D"],
            "full_nm_stdout_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "binary128_helpers": helpers, "libgcc_s_resolved": True}


def source_hashes():
    result = {}
    for output in (SA, NLO):
        files = {}
        for path in sorted(output.rglob("*")):
            if path.is_file() and (path.suffix.lower() in (".f", ".f90", ".inc")
                                   or path.name in ("make_opts", "makefile",
                                                    "MadLoop_makefile_definitions")):
                files[str(path.relative_to(output))] = digest(path)
        result[output.name] = files
    return result


def build():
    if (WORK / "nlo").exists():
        assert not NLO.exists(), "Both original and relocated NLO outputs exist"
        (WORK / "nlo").rename(NLO)
    assert NLO.is_dir() and not (WORK / "nlo").exists()
    original = WORK / "standalone"
    if original.exists():
        assert not SA.exists(), "Both original and relocated outputs exist"
        original.rename(SA)
    assert SA.is_dir() and not original.exists()
    for output in (SA, NLO):
        configure(output)
    copied = {}
    for name in ("libninjars.a", "mninjars.mod", "ninjars.h", "mninjars.f90",
                 "ninjars-config.json", "ninjars-link-before.txt",
                 "ninjars-native-libs.txt"):
        path = SA / "lib" / name
        assert path.is_file() and not path.is_symlink(), path
        copied[name] = {"sha256": digest(path), "bytes": path.stat().st_size,
                        "regular_file": True}
    for name in ("ninjars-config.json", "ninjars-link-before.txt",
                 "ninjars-native-libs.txt"):
        shutil.copy2(SA / "lib" / name, REPORT / name)
    definitions = SA / "SubProcesses/MadLoop_makefile_definitions"
    link = definitions.read_text()
    assert "HEPTools/ninjars" not in link
    assert link.index("-lgcc_s") < link.index("-lninjars")
    shutil.copy2(definitions, REPORT / "standalone-MadLoop_makefile_definitions")
    shutil.copy2(NLO / "Source/make_opts", REPORT / "nlo-make_opts")
    subprocess_dir = SA / "SubProcesses/P0_uux_z"
    source = subprocess_dir / "check_sa.f"
    force = (
        "      CALL GET_COMMAND_ARGUMENT(1,BUFF(1))\n"
        "      IF (TRIM(BUFF(1)).EQ.'stable') THEN\n"
        "        CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)\n"
        "      ENDIF")
    old = "C       CALL ML5_0_FORCE_STABILITY_CHECK(.TRUE.)"
    if not (REPORT / "check_sa-generated.f").exists():
        shutil.copy2(source, REPORT / "check_sa-generated.f")
    original_driver = (REPORT / "check_sa-generated.f").read_text()
    assert old in original_driver
    assert "PARAMETER (NPSPOINTS = 4)" in original_driver
    source.write_text(original_driver.replace(old, force).replace(
        "PARAMETER (NPSPOINTS = 4)", "PARAMETER (NPSPOINTS = 20)"))
    assert force in source.read_text()
    shutil.copy2(source, REPORT / "check_sa-stability.f")
    command(["make", "-C", str(subprocess_dir), "-j1",
             "SHELL=" + shutil.which("bash"), "check"], ROOT,
            "standalone-stability-build.log")
    metadata = {
        "original_output_removed": not original.exists(),
        "original_nlo_output_removed": not (WORK / "nlo").exists(),
        "nlo_dependencies": "internal",
        "original_output": str(original.relative_to(ROOT)),
        "relocated_output": str(SA.relative_to(ROOT)),
        "copied_files": copied,
        "link_flags_reference_original_ninjars": False,
        "installed_config": json.loads((SA / "lib/ninjars-config.json").read_text()),
        "standalone_symbols": symbols(subprocess_dir / "check", "standalone"),
        "driver_change": "Use 20 calls to finish helicity/filter initialization. Argument 'stable' enables the documented force-stability call during initialization; normal DP/QP calls keep the original driver policy.",
        "environment_libraries_removed": ["LD_LIBRARY_PATH", "LIBRARY_PATH"],
    }
    dump("relocation.json", metadata)
    dump("generated-source-hashes.json", source_hashes())
    print("Relocated standalone built; GNU binary128 helpers verified.")


def parse_check(text):
    result = {}
    for name, label in (("born", "Born"), ("finite", "Finite"),
                        ("single_pole", "1EPS"), ("double_pole", "2EPS")):
        # Read the raw totals, not the normalized result.dat FIN/1EPS/2EPS.
        matches = re.findall(
            rf"(?im)^\s*Matrix element {label}\s*=\s*([+-]?[0-9.]+(?:[dDeE][+-]?\d+)?)", text)
        if not matches:
            raise RuntimeError(f"Missing {label} in standalone output")
        result[name] = float(matches[-1].replace("D", "E").replace("d", "e"))
    result["return_code"] = int(re.findall(r"MadLoop return code\s*=\s*(\d+)", text)[-1])
    return result


def run():
    subprocess_dir = SA / "SubProcesses/P0_uux_z"
    measurements = {}
    for label, mode, expected in (("dp", 1, 110), ("qp", 4, 120), ("stable", -1, 218)):
        configure(SA, mode)
        shutil.copy2(SA / "Cards/MadLoopParams.dat", REPORT / f"standalone-{label}-MadLoopParams.dat")
        for pattern in ("*HelFilter.dat", "*LoopFilter.dat"):
            for path in (SA / "SubProcesses").rglob(pattern):
                path.unlink()
        args = [str(subprocess_dir / "check")]
        if label == "stable":
            args.append("stable")
        text = command(args, subprocess_dir,
                       f"standalone-{label}.log")
        shutil.copy2(subprocess_dir / "result.dat", REPORT / f"standalone-{label}-result.dat")
        values = parse_check(text)
        assert values["return_code"] == expected, (label, values)
        measurements[label] = values
        dump("standalone-results.json", measurements)
    reference = json.loads((ROOT / "docs/benchmarkfolder/data/installer-validation.json").read_text())
    reference = reference["tests"]["external_abi2_qcd1"]["comparison_backends"]["1"]
    mapping = {"born": "born", "finite": "finite", "single_pole": "1eps", "double_pole": "2eps"}
    for values in measurements.values():
        for key, old_key in mapping.items():
            assert abs(values[key] - reference[old_key]) <= 1e-11 * max(1, abs(reference[old_key]))
    configure(NLO)
    # Force the installed source tree unavailable during internal-export execution.
    installed = WORK / "source/HEPTools/ninjars"
    assert not installed.exists(), "Hide the private installation before running exports"
    nlo_log = command([sys.executable, str(NLO / "bin/aMCatNLO"),
                       str(REPORT / "nlo-compile.mg5")], ROOT, "nlo-compile.log",
                      external=True)
    collect(nlo_log, measurements)
    print("Standalone DP/QP/stability calls and complete NLO compile finished.")


def collect(nlo_log=None, measurements=None):
    if nlo_log is None:
        nlo_log = (REPORT / "nlo-compile.log").read_text()
    if measurements is None:
        measurements = json.loads((REPORT / "standalone-results.json").read_text())
    assert "Poles successfully cancel for 20 points over 20" in nlo_log
    assert "Result for test_ME:" in nlo_log and "Passed." in nlo_log
    nlo_subprocess = NLO / "SubProcesses/P0_uux_z"
    soft_collinear = (nlo_subprocess / "test_ME.log").read_text()
    assert "Fraction of failures: 0.00" in soft_collinear
    assert "FAILED" not in soft_collinear
    assert re.search(r">\s*MLReductionLib\s*=\s*8\b",
                     (nlo_subprocess / "check_poles.log").read_text())
    shutil.copy2(nlo_subprocess / "test_ME.log", REPORT / "nlo-test_ME.log")
    shutil.copy2(nlo_subprocess / "check_poles.log", REPORT / "nlo-check_poles.log")
    nlo_symbols = {
        name: symbols(nlo_subprocess / name, "nlo-" + name)
        for name in ("check_poles", "madevent_mintFO")
    }
    for name in ("MadLoopParams.dat", "param_card.dat", "run_card.dat"):
        shutil.copy2(NLO / "Cards" / name, REPORT / ("nlo-after-compile-" + name))
    dump("compiled-source-hashes.json", source_hashes())
    dump("runtime-results.json", {
        "standalone": measurements,
        "standalone_comparison": "All four coefficients agree with retained CutTools reference at 1e-11 relative or absolute.",
        "nlo_symbols": nlo_symbols,
        "nlo_cli_success": True,
        "pole_checks_passed": 20,
        "pole_checks_tested": 20,
        "pole_test_tolerance": 1e-5,
        "soft_collinear_checks": "Passed, zero failures",
        "scope": "Linux/GNU Fortran 15.3.0; no shower validation.",
    })
    print("Collected successful numerical and linkage evidence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--hide-installation", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args()
    if not (args.build or args.hide_installation or args.run or args.collect):
        parser.error("Select --build, --hide-installation, --run, and/or --collect")
    if args.build:
        build()
    if args.hide_installation:
        hide_installation()
    if args.run:
        run()
    if args.collect:
        collect()
