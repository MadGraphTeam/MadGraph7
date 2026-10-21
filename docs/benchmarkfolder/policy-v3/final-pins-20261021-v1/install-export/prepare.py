#!/usr/bin/env python3
"""Real CLI installation in a private archived candidate checkout."""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
import tarfile
import time

sys.dont_write_bytecode = True
import tomllib

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "madgraph/__init__.py").is_file())
REPORT = Path(__file__).resolve().parent
WORK = ROOT / ".benchmark-work/policy-v3/final-pins-20261021-v1/install-export"
SOURCE = WORK / "source"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def generator_sources(directory, names=None):
    """Bind the actual archived CLI generator, separately from the bridge."""
    if names is not None:
        return {name: digest(directory / name) for name in names}
    result = {}
    for name in ("madgraph", "aloha", "Template", "bin", "models", "input"):
        for path in sorted((directory / name).rglob("*")):
            if path.is_file():
                result[str(path.relative_to(directory))] = digest(path)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True,
                        help="Explicit final committed MadGraph revision")
    parser.add_argument("--bridge-freeze", type=Path, required=True,
                        help="Final release bridge source-freeze JSON inside the repository")
    parser.add_argument("--bridge-freeze-sha256", required=True,
                        help="Expected SHA256 of that complete bridge source-freeze record")
    parser.add_argument("--previous-installation", type=Path, required=True,
                        help="Validated old repository-local NinjaRS prefix to copy and update")
    parser.add_argument("--previous-archive-sha256", required=True,
                        help="Expected archive SHA256 of that old validated installation")
    parser.add_argument("--cpu", type=int, required=True)
    args = parser.parse_args()
    if subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip() != "madloop_ninjars":
        raise RuntimeError("Final validation must use branch madloop_ninjars")
    if args.cpu not in os.sched_getaffinity(0):
        raise RuntimeError("Requested CPU is outside the allowed affinity")
    os.sched_setaffinity(0, {args.cpu})
    revision = subprocess.check_output(
        ["git", "rev-parse", args.revision+"^{commit}"], cwd=ROOT, text=True).strip()
    scope = ["madgraph", "aloha", "Template", "bin", "models", "input",
             "vendor/ninjars", "tests/unit_tests"]
    subprocess.run(["git", "diff", "--exit-code", revision, "--"] + scope,
                   cwd=ROOT, check=True)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--",
         *scope], cwd=ROOT, text=True)
    if untracked.strip():
        raise RuntimeError("Commit numerical source additions before archived validation")
    freeze_path = args.bridge_freeze.expanduser().resolve()
    if not freeze_path.is_relative_to(ROOT) or digest(freeze_path) != args.bridge_freeze_sha256:
        raise RuntimeError("Final bridge source freeze is missing or changed")
    freeze = json.loads(freeze_path.read_text())
    tested_sources = freeze["files"]
    if not tested_sources or not all(len(v) == 64 for v in tested_sources.values()):
        raise RuntimeError("Final bridge source freeze lacks complete SHA256 values")
    if WORK.exists():
        raise RuntimeError("Use a fresh namespace; installation evidence is immutable")
    WORK.mkdir(parents=True)
    SOURCE.mkdir()
    archive = WORK / "candidate.tar"
    with archive.open("wb") as out:
        subprocess.run(["git", "archive", revision], cwd=ROOT, stdout=out, check=True)
    with tarfile.open(archive) as source:
        source.extractall(SOURCE, filter="tar")
    archived_sources = {
        str(p.relative_to(SOURCE / "vendor/ninjars")): digest(p)
        for p in sorted((SOURCE / "vendor/ninjars").rglob("*")) if p.is_file()}
    if archived_sources != tested_sources:
        raise RuntimeError("Final archived bridge differs from the complete tested source map")
    makefile = (SOURCE / "vendor/ninjars/Makefile").read_text()
    rust_command = "$(CARGO) rustc --release --locked --lib -- --print native-static-libs"
    if rust_command not in makefile:
        raise RuntimeError("Final installer Cargo invocation differs from the validated release command")
    source_identity = generator_sources(SOURCE)
    write_json(REPORT / "archived-generator-sources.json", source_identity)
    retained_freeze = REPORT / "final-bridge-source-freeze.json"
    with retained_freeze.open("xb") as out:
        out.write(freeze_path.read_bytes())
    freeze_binding = {"path": str(retained_freeze.relative_to(ROOT)),
                      "sha256": digest(retained_freeze), "bytes": retained_freeze.stat().st_size,
                      "original_source_path": str(freeze_path.relative_to(ROOT))}
    write_json(REPORT / "bridge-freeze-binding.json", freeze_binding)
    # The prefix is inside this checkout so install does not write a user-level
    # configuration shared with the production or original benchmark checkout.
    prefix = SOURCE / "HEPTools"
    previous_source = args.previous_installation.expanduser().resolve()
    if (not previous_source.is_relative_to(ROOT) or
            digest(previous_source / "lib/libninjars.a") != args.previous_archive_sha256):
        raise RuntimeError("The selected old validated installation is missing or changed")
    if any(p.is_symlink() for p in previous_source.rglob("*")):
        raise RuntimeError("Previous installation must contain regular files")
    previous_image = {str(p.relative_to(previous_source)): digest(p)
                      for p in sorted(previous_source.rglob("*")) if p.is_file()}
    retained_previous = WORK / "previous-installation"
    shutil.copytree(previous_source, retained_previous)
    prefix.mkdir(parents=True, exist_ok=True)
    shutil.copytree(retained_previous, prefix / "ninjars")
    aliases = prefix / "lib"
    aliases.mkdir()
    for name in ("libninjars.a", "mninjars.mod", "ninjars-config.json",
                 "ninjars-native-libs.txt", "ninjars-link-before.txt"):
        (aliases / name).symlink_to("../ninjars/lib/" + name)
    for source_name, retained_name in (("Cargo.lock", "previous-Cargo.lock"),
            ("Cargo.toml", "previous-Cargo.toml"), ("install.log", "previous-install.log"),
            ("lib/ninjars-config.json", "previous-ninjars-config.json")):
        shutil.copy2(retained_previous / source_name, REPORT / retained_name)
    previous_record = {
        "schema": "final-pins-actual-installed-update-input-v1",
        "source": str(previous_source.relative_to(ROOT)),
        "retained_at": str(retained_previous.relative_to(ROOT)),
        "copied_to": str((prefix / "ninjars").relative_to(ROOT)),
        "files": previous_image,
        "owned_aliases": {p.name: os.readlink(p) for p in sorted(aliases.iterdir())},
    }
    write_json(REPORT / "previous-installation-image.json", previous_record)
    commands = [
        "set automatic_html_opening False", "set auto_update 0",
        "set nb_core 1", "set fortran_compiler gfortran",
        f"install ninjars --update --prefix={prefix} --jobs=1",
        "quit",
    ]
    card = REPORT / "install.mg5"
    with card.open("x") as stream:
        stream.write("\n".join(commands) + "\n")
    env = os.environ.copy()
    candidate = ROOT / ".ninjars-build/final-dependency-candidate-20261021"
    env["CARGO_HOME"] = str(candidate / "cargo-home")
    env["CARGO_TARGET_DIR"] = str(candidate / "target")
    env["CARGO_BUILD_BUILD_DIR"] = str(candidate / "build")
    env["TMPDIR"] = str(candidate / "tmp")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["SYMBOLICA_LOCK_DIR"] = str(ROOT / ".ninjars-build/symbolica-locks")
    env["MAKEFLAGS"] = "SHELL=" + subprocess.check_output(
        ["bash", "-c", "command -v bash"], text=True).strip()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               RAYON_NUM_THREADS="1", MKL_NUM_THREADS="1")
    command = [sys.executable, str(SOURCE / "bin/madgraph"), "--plain", str(card)]
    start = time.perf_counter()
    with (REPORT / "install-cli.log").open("x") as log:
        completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT)
    installed = prefix / "ninjars"
    required = ("lib/libninjars.a", "lib/mninjars.mod", "include/mninjars.f90",
                "include/ninjars.h", "lib/ninjars-config.json",
                "lib/ninjars-link-before.txt", "lib/ninjars-native-libs.txt",
                "Cargo.lock", "Cargo.toml", "install.log")
    result = {
        "revision": revision, "archive_sha256": digest(archive),
        "archived_bridge_source_sha256": tested_sources,
        "archived_generator_sources_sha256": digest(REPORT / "archived-generator-sources.json"),
        "bridge_freeze_binding_sha256": digest(REPORT / "bridge-freeze-binding.json"),
        "installer_rust_command": rust_command,
        "archived_installer_sha256": digest(SOURCE / "madgraph/various/ninjars_install.py"),
        "execution_cpu": args.cpu, "cargo_jobs": 1, "purpose": "Final locked dependency installation qualification; not a performance sample",
        "command": command, "exit_code": completed.returncode,
        "seconds": time.perf_counter()-start,
        "cli_card_sha256": digest(card), "cli_log_sha256": digest(REPORT / "install-cli.log"),
        "installed": str(installed.relative_to(ROOT)),
        "files": {name: digest(installed / name) for name in required
                  if (installed / name).is_file()},
        "reference_source_overrides": False,
        "actual_cli_update": True,
        "previous_installation_image_sha256": digest(REPORT / "previous-installation-image.json"),
        "previous_archive_sha256": args.previous_archive_sha256,
        "previous_installation_preserved": previous_image == {
            str(p.relative_to(retained_previous)): digest(p)
            for p in sorted(retained_previous.rglob("*")) if p.is_file()},
        "previous_source_unchanged": previous_image == {
            str(p.relative_to(previous_source)): digest(p)
            for p in sorted(previous_source.rglob("*")) if p.is_file()},
        "installed_file_sha256": {
            str(p.relative_to(installed)): digest(p)
            for p in sorted(installed.rglob("*")) if p.is_file()},
        "installed_symlinks": [str(p.relative_to(installed))
                               for p in sorted(installed.rglob("*")) if p.is_symlink()],
    }
    config_file = installed / "lib/ninjars-config.json"
    config = json.loads(config_file.read_text()) if config_file.is_file() else {}
    smoke_passed = (installed / "install.log").is_file() and (
        "NinjaRS ABI3 native/compensated DP and genuine binary128-input rescue passed"
        in (installed / "install.log").read_text())
    locked = tomllib.loads((SOURCE / "vendor/ninjars/Cargo.lock").read_text())
    git_dependencies = sorted(
        [{"name": p["name"], "version": p["version"], "source": p["source"]}
         for p in locked["package"] if p.get("source", "").startswith("git+")],
        key=lambda item: (item["name"], item["source"]))
    expected_provenance = {
        "manifest_sha256": tested_sources["Cargo.toml"],
        "lock_sha256": tested_sources["Cargo.lock"], "git_dependencies": git_dependencies}
    result["build_provenance"] = config.get("build_provenance")
    result["expected_build_provenance"] = expected_provenance
    result["local_sources"] = config.get("local_sources")
    result["abi_version"] = config.get("abi_version")
    result["compiled_smoke_passed"] = smoke_passed
    result["passed"] = (completed.returncode == 0 and len(result["files"]) == len(required)
                        and config.get("abi_version") == 3
                        and config.get("quad_precision") is True and smoke_passed
                        and config.get("local_sources") == {}
                        and config.get("build_provenance") == expected_provenance
                        and result["installed_file_sha256"] == result["files"]
                        and not result["installed_symlinks"]
                        and result["files"].get("lib/libninjars.a") != args.previous_archive_sha256
                        and result["previous_installation_preserved"]
                        and result["previous_source_unchanged"]
                        and result["files"]["include/mninjars.f90"] ==
                        tested_sources["mninjars.f90"])
    write_json(REPORT / "installation.json", result)
    if not result["passed"]:
        raise RuntimeError("Actual CLI installation did not publish every required artifact")
    for name in ("install.log", "Cargo.lock", "Cargo.toml"):
        (REPORT / name).write_bytes((installed / name).read_bytes())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
