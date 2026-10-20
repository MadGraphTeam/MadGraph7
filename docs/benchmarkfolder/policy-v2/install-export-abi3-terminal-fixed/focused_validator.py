#!/usr/bin/env python3
"""Finish the focused ABI3 controls after successful archived SA/NLO checks."""
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
import runtime_validator as runtime

ROOT, REPORT, WORK, SOURCE = prepare.ROOT, prepare.REPORT, prepare.WORK, prepare.SOURCE


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, value):
    (REPORT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run(command, log, cwd=ROOT):
    with (REPORT / log).open("w") as stream:
        result = subprocess.run(command, cwd=cwd, env=runtime.environment(),
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}); retained {log}")
    return command


def require_success():
    result = json.loads((REPORT / "runtime-results.json").read_text())
    assert result["pole_checks_passed"] == result["pole_checks_tested"] == 20
    assert result["nlo_cli_success"] is True
    assert result["standalone"]["qp"]["return_code"] == 120


def units():
    command = [sys.executable, "-m", "unittest",
               "tests.unit_tests.loop.test_ninjars_fortran",
               "tests.unit_tests.various.test_ninjars_install",
               "tests.unit_tests.various.test_ninjars_statistics"]
    run(command, "focused-unit-tests.log")
    text = (REPORT / "focused-unit-tests.log").read_text()
    assert "Ran 16 tests" in text and re.search(r"(?m)^OK$", text)
    prior_dir = ROOT / "docs/benchmarkfolder/experiments/t3-dp-retry-port"
    prior = json.loads((prior_dir / "validation.json").read_text())
    # The generator-only fix leaves the complete tested bridge unchanged.
    identities = {}
    for category in ("source_sha256", "additional_source_sha256"):
        for name, expected in prior[category].items():
            if name.startswith("vendor/ninjars/"):
                assert sha(ROOT / name) == expected, name
                identities[name] = expected
    save("focused-unit-results.json", {
        "passed": True, "tests": 16, "command": command,
        "log_sha256": sha(REPORT / "focused-unit-tests.log"),
        "transport_and_guard_suite": {
            "evidence": str(prior_dir.relative_to(ROOT)),
            "validation_sha256": sha(prior_dir / "validation.json"),
            "tests_previously_passed": 16,
            "current_bridge_sources_match_tested_sources": identities,
            "note": "No redundant Rust execution. Actual installed MP smoke and relocated QP run are fresh.",
        },
    })


def absent(resume=False):
    output = WORK / "absent-ninjars"
    command = [sys.executable, str(SOURCE / "bin/madgraph"), "--plain",
               str(REPORT / "absent-install.mg5")]
    if resume:
        assert (REPORT / "absent-failed-ninja6/failure.json").is_file()
        assert output.is_dir()
    else:
        assert not output.exists(), "Refusing to overwrite absent-library evidence"
        run(command, "absent-export.log")
    directory = output / "SubProcesses/P0_uux_z"
    assert directory.is_dir()
    forbidden_files = [
        path for path in output.rglob("*")
        if path.name in ("mninjars.mod", "mninjars.f90", "libninjars.a")]
    assert not forbidden_files, forbidden_files
    definitions = output / "SubProcesses/MadLoop_makefile_definitions"
    assert "-lninjars" not in definitions.read_text()
    bad = re.compile(r"(?im)^\s*(?:USE\s+MNINJARS\b|CALL\s+\w*NINJARS_\w+\b)")
    checked = {}
    for path in directory.glob("*.f"):
        source = path.read_text()
        assert not bad.search(source), path
        checked[path.name] = sha(path)
    matrix = (directory / "loop_matrix.f").read_text()
    availability = re.search(r"DATA\s+LOOPLIBS_AVAILABLE/(.*?)/", matrix, re.I | re.S)
    assert availability, "Missing generated library availability table"
    slots = re.findall(r"\.(TRUE|FALSE)\.", availability.group(1), re.I)
    assert len(slots) == 8 and slots[7].upper() == "FALSE", slots
    # Internal dependencies omit optional original Ninja; use bundled CutTools.
    # Twenty calls complete initialization; force DP only.
    driver = directory / "check_sa.f"
    if not resume:
        shutil.copy2(driver, REPORT / "absent-check_sa-generated.f")
    checked["check_sa.f"] = sha(REPORT / "absent-check_sa-generated.f")
    text = (REPORT / "absent-check_sa-generated.f").read_text()
    assert "PARAMETER (NPSPOINTS = 4)" in text
    driver.write_text(text.replace("PARAMETER (NPSPOINTS = 4)",
                                  "PARAMETER (NPSPOINTS = 20)"))
    shutil.copy2(driver, REPORT / "absent-check_sa-twenty.f")
    card = runtime.MadLoopParam(str(output / "Cards/MadLoopParams.dat"))
    card["MLReductionLib"] = "1"
    card["CTModeRun"] = 1
    card["MLStabThres"] = 1e-9
    for path in [output / "Cards/MadLoopParams.dat",
                 *(output / "SubProcesses").rglob("MadLoopParams.dat")]:
        card.write(str(path), template=str(
            ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    shutil.copy2(output / "Cards/MadLoopParams.dat", REPORT / "absent-MadLoopParams.dat")
    build_command = ["make", "-C", str(directory), "-j2",
                     "SHELL=" + shutil.which("bash"), "check"]
    run(build_command, "absent-build.log")
    run([str(directory / "check")], "absent-runtime.log", directory)
    values = runtime.parse_check((REPORT / "absent-runtime.log").read_text())
    assert values["return_code"] == 110, values
    expected = json.loads((REPORT / "standalone-results.json").read_text())["dp"]
    for key in ("born", "finite", "single_pole", "double_pole"):
        assert abs(values[key] - expected[key]) <= 1e-11 * max(1, abs(expected[key]))
    symbols = subprocess.check_output(["nm", str(directory / "check")],
                                      cwd=ROOT, env=runtime.environment(), text=True)
    assert not re.search(r"\bninjars_(?:context|evaluate|abi_version|last_error)", symbols)
    shutil.copy2(directory / "result.dat", REPORT / "absent-result.dat")
    shutil.copy2(definitions, REPORT / "absent-MadLoop_makefile_definitions")
    # A Fortran STOP may return zero: validate the actual error and absence of a result.
    card["MLReductionLib"] = "8"
    paths = [output / "Cards/MadLoopParams.dat",
             *(output / "SubProcesses").rglob("MadLoopParams.dat")]
    for path in paths:
        card.write(str(path), template=str(
            ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    shutil.copy2(output / "Cards/MadLoopParams.dat",
                 REPORT / "absent-exclusive8-MadLoopParams.dat")
    with (REPORT / "absent-exclusive8.log").open("w") as stream:
        rejected = subprocess.run([str(directory / "check")], cwd=directory,
                                  env=runtime.environment(), stdout=stream,
                                  stderr=subprocess.STDOUT)
    rejection = (REPORT / "absent-exclusive8.log").read_text()
    assert "No available loop reduction lib is provided." in rejection
    assert "Make sure MLReductionLib is correct." in rejection
    assert re.search(r"MLReductionLib\s*=\s*8\b", rejection)
    assert not re.search(r"Matrix element Born\s*=", rejection, re.I)
    card["MLReductionLib"] = "1"
    for path in paths:
        card.write(str(path), template=str(
            ROOT / "Template/loop_material/StandAlone/Cards/MadLoopParams.dat"))
    save("absent-results.json", {
        "passed": True, "generation_command": command, "build_command": build_command,
        "runtime_command": [str(directory / "check")], "values": values,
        "ninjars_files_absent": True, "ninjars_link_flag_absent": True,
        "module_and_tensor_calls_absent": True, "availability_slot_8_false": True,
        "bridge_symbols_absent": True, "executable_sha256": sha(directory / "check"),
        "generated_sources_before_driver_adjustment": checked,
        "driver_adjustment": "20 calls instead of 4, bundled CutTools forced DP. No production source changed.",
        "exclusive8": {"rejected": True, "exit_code": rejected.returncode,
                       "actionable_error": "No available loop reduction lib is provided. Make sure MLReductionLib is correct.",
                       "no_numerical_stdout": True,
                       "result_file_scope": "The prior successful result.dat was retained; only current-process stdout was checked for rejection."},
        "retained_setup_rejection": "absent-failed-ninja6/failure.json" if resume else None,
    })


def reference_manifest():
    """Hash precisely the source-tree domain copied by the installer."""
    result = {}
    excluded = {".git", "target", "references", "__pycache__", ".direnv"}
    for name in ("ninjars", "oneloopmaster"):
        source = ROOT.parent / "references" / name
        files = {}
        for directory, dirs, names in os.walk(source):
            dirs[:] = sorted(d for d in dirs if d not in excluded)
            assert not any((Path(directory) / name).is_symlink() for name in dirs), \
                "Copied reference domain contains a symlink directory; extend manifest traversal before proceeding"
            for filename in sorted(names):
                if filename in excluded:
                    continue
                path = Path(directory) / filename
                files[str(path.relative_to(source))] = sha(path)
        env = os.environ.copy()
        env["GIT_OPTIONAL_LOCKS"] = "0"
        revision = subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], env=env, text=True).strip()
        result[name] = {"path": str(source), "revision": revision, "files": files}
    return result


def local_source():
    prefix = WORK / "local-source-HEPTools"
    assert not prefix.exists(), "Refusing to overwrite local-source evidence"
    before = reference_manifest()
    save("local-source-reference-before.json", before)
    command = [sys.executable, str(SOURCE / "bin/madgraph"), "--plain",
               str(REPORT / "local-source-install.mg5")]
    try:
        run(command, "local-source-cli.log")
    finally:
        after = reference_manifest()
        save("local-source-reference-after.json", after)
    assert before == after, "Read-only reference source contents changed"
    installed = prefix / "ninjars"
    config = json.loads((installed / "lib/ninjars-config.json").read_text())
    assert config["abi_version"] == 3 and config["quad_precision"] is True
    assert config["link_before"] == ["-lgcc_s"]
    for name, identity in before.items():
        assert config["local_sources"][name]["revision"] == identity["revision"]
    log = (installed / "install.log").read_text()
    assert "NinjaRS ABI3 native/compensated DP and genuine binary128-input rescue passed" in log
    retained = REPORT / "local-source-installation"
    retained.mkdir()
    for name in ("install.log", "Cargo.lock", "Cargo.toml"):
        shutil.copy2(installed / name, retained / name)
    shutil.copy2(installed / "lib/ninjars-config.json", retained / "ninjars-config.json")
    for name in ("mninjars.f90", "ninjars.h"):
        assert sha(installed / "include" / name) == sha(ROOT / "vendor/ninjars" / name)
    save("local-source-results.json", {
        "passed": True, "command": command, "installed": str(installed.relative_to(ROOT)),
        "abi_version": 3, "actual_native_compensated_native_and_mp_smoke_passed": True,
        "reference_source_domain_unchanged": True,
        "reference_manifest_domain": "Installer copied domain, excluding .git, target, references, __pycache__, .direnv.",
        "reference_file_counts": {name: len(data["files"]) for name, data in before.items()},
        "metadata": config, "archive_sha256": sha(installed / "lib/libninjars.a"),
        "retained_files": {p.name: sha(p) for p in sorted(retained.iterdir())},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("units", "absent", "absent-resume", "local-source"))
    action = parser.parse_args().action
    require_success()
    {"units": units, "absent": absent, "absent-resume": lambda: absent(resume=True),
     "local-source": local_source}[action]()
