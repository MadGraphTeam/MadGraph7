#!/usr/bin/env python3
"""Real CLI installation in a private archived candidate checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
REPORT = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/policy-v2/install-export-abi3-terminal"
SOURCE = WORK / "source"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", default="HEAD")
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", args.revision+"^{commit}"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "diff", "--exit-code", revision, "--",
                    "madgraph", "aloha", "Template", "vendor/ninjars"], cwd=ROOT, check=True)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--",
         "madgraph", "aloha", "Template", "vendor/ninjars"], cwd=ROOT, text=True)
    if untracked.strip():
        raise RuntimeError("Commit numerical source additions before archived validation")
    if WORK.exists():
        raise RuntimeError("Use a fresh namespace; installation evidence is immutable")
    WORK.mkdir(parents=True)
    SOURCE.mkdir()
    archive = WORK / "candidate.tar"
    with archive.open("wb") as out:
        subprocess.run(["git", "archive", revision], cwd=ROOT, stdout=out, check=True)
    with tarfile.open(archive) as source:
        source.extractall(SOURCE, filter="tar")
    # The prefix is inside this checkout so install does not write a user-level
    # configuration shared with the production or original benchmark checkout.
    prefix = SOURCE / "HEPTools"
    commands = [
        "set automatic_html_opening False", "set auto_update 0",
        "set nb_core 4", "set fortran_compiler gfortran",
        f"install ninjars --prefix={prefix} --jobs=4",
        "quit",
    ]
    card = REPORT / "install.mg5"
    card.write_text("\n".join(commands) + "\n")
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(ROOT / ".ninjars-build/cargo")
    env["CARGO_BUILD_BUILD_DIR"] = str(ROOT / ".ninjars-build/cargo/build")
    env["TMPDIR"] = str(ROOT / ".ninjars-build/tmp")
    env["SYMBOLICA_LOCK_DIR"] = str(ROOT / ".ninjars-build/symbolica-locks")
    env["MAKEFLAGS"] = "SHELL=" + subprocess.check_output(
        ["bash", "-c", "command -v bash"], text=True).strip()
    command = [sys.executable, str(SOURCE / "bin/madgraph"), "--plain", str(card)]
    start = time.perf_counter()
    with (REPORT / "install-cli.log").open("w") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
    installed = prefix / "ninjars"
    required = ("lib/libninjars.a", "lib/mninjars.mod", "include/mninjars.f90",
                "include/ninjars.h", "lib/ninjars-config.json",
                "lib/ninjars-link-before.txt", "lib/ninjars-native-libs.txt",
                "Cargo.lock", "Cargo.toml", "install.log")
    result = {
        "revision": revision, "archive_sha256": digest(archive),
        "command": command, "exit_code": completed.returncode,
        "seconds": time.perf_counter()-start,
        "cli_card_sha256": digest(card), "cli_log_sha256": digest(REPORT / "install-cli.log"),
        "installed": str(installed.relative_to(ROOT)),
        "files": {name: digest(installed / name) for name in required
                  if (installed / name).is_file()},
        "reference_source_overrides": False,
    }
    config_file = installed / "lib/ninjars-config.json"
    config = json.loads(config_file.read_text()) if config_file.is_file() else {}
    smoke_passed = (installed / "install.log").is_file() and (
        "NinjaRS ABI3 native/compensated DP and genuine binary128-input rescue passed"
        in (installed / "install.log").read_text())
    result["abi_version"] = config.get("abi_version")
    result["compiled_smoke_passed"] = smoke_passed
    result["passed"] = (completed.returncode == 0 and len(result["files"]) == len(required)
                        and config.get("abi_version") == 3
                        and config.get("quad_precision") is True and smoke_passed
                        and result["files"]["include/mninjars.f90"] ==
                        digest(ROOT / "vendor/ninjars/mninjars.f90"))
    (REPORT / "installation.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["passed"]:
        raise RuntimeError("Actual CLI installation did not publish every required artifact")
    for name in ("install.log", "Cargo.lock", "Cargo.toml"):
        (REPORT / name).write_bytes((installed / name).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
