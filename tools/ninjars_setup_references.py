#!/usr/bin/env python3
"""Reproduce the archived MadLoop baseline and its reference dependencies.

Requires the repository's pinned development shell and network access. Sources,
temporary build directories and logs remain under .benchmark-work.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".benchmark-work"
BASELINE = WORK / "original"
REVISION = "27147b3300f556a75661a886e92bdbe00558ee25"
SOURCES = {
    "installer": ("https://madgraph.mi.infn.it/Downloads/HEPToolsInstaller/"
                  "HEPToolsInstaller_V168.tar.gz"),
    "ninja": ("https://github.com/peraro/ninja/releases/download/v1.2.1/"
              "ninja-latest.tar.gz"),
    "oneloop": "https://madgraph.mi.infn.it/Downloads/OneLOop-3.6.tgz",
    "collier": "https://collier.hepforge.org/downloads/collier-1.2.9.tar.gz",
}


def download(name, url):
    path = WORK / "downloads" / (name + ".tar.gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = path.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-only", action="store_true")
    options = parser.parse_args()
    WORK.mkdir(exist_ok=True)
    (WORK / "tmp").mkdir(exist_ok=True)
    (WORK / "logs").mkdir(exist_ok=True)
    if not (BASELINE / "bin/madgraph").exists():
        BASELINE.mkdir(parents=True, exist_ok=True)
        archive = WORK / "baseline.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", REVISION], cwd=ROOT, stdout=stream, check=True)
        with tarfile.open(archive) as source:
            source.extractall(BASELINE, filter="data")
        archive.unlink()
    paths = {name: download(name, url) for name, url in SOURCES.items()}
    inventory = {name: {"url": SOURCES[name],
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for name, path in paths.items()}
    manifest = ROOT / "docs/benchmarkfolder/reference-downloads.json"
    if manifest.exists():
        recorded = json.loads(manifest.read_text())
        for name, data in inventory.items():
            if name in recorded and data["sha256"] != recorded[name]["sha256"]:
                raise RuntimeError(f"Reference download checksum mismatch: {name}")
    else:
        manifest.write_text(json.dumps(inventory, indent=2) + "\n")
    if options.download_only:
        return
    heptools = BASELINE / "HEPTools"
    installer = heptools / "HEPToolsInstallers"
    if not installer.exists():
        heptools.mkdir(exist_ok=True)
        with tarfile.open(paths["installer"]) as source:
            source.extractall(heptools, filter="data")
    # Only host compatibility changes: no numerical library source modifications.
    for path in installer.glob("*.sh"):
        text = path.read_text()
        if text.startswith("#!/bin/bash"):
            path.write_text(text.replace("#!/bin/bash", "#!/usr/bin/env bash", 1))
    environment = os.environ.copy()
    environment["TMPDIR"] = str(WORK / "tmp")
    environment["XDG_CONFIG_HOME"] = str(WORK / "config")
    environment["MAKEFLAGS"] = "SHELL=" + shutil.which("bash")
    expected = {"ninja": heptools / "ninja/lib/libninja.a",
                "collier": heptools / "collier/libcollier.a"}
    for tool in ("ninja", "collier"):
        if expected[tool].exists():
            continue
        command = [sys.executable, str(installer / "HEPToolInstaller.py"), tool,
                   "--prefix=" + str(heptools), "--fortran_compiler=gfortran",
                   "--cpp_compiler=g++", "--mg5_path=" + str(BASELINE),
                   "--keep_source",
                   "--ninja_tarball=" + str(paths["ninja"]),
                   "--oneloop_tarball=" + str(paths["oneloop"]),
                   "--collier_tarball=" + str(paths["collier"])]
        with (WORK / "logs" / f"install-{tool}.log").open("w") as stream:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=stream,
                           stderr=subprocess.STDOUT, check=True)
        if not expected[tool].is_file():
            raise RuntimeError(f"{tool} installer did not produce {expected[tool]}")
    if not (BASELINE / "models/loop_qcd_qed_sm_Gmu/particles.py").exists():
        card = WORK / "import-ew-model.mg5"
        card.write_text("set auto_update 0\nimport model loop_qcd_qed_sm_Gmu\nquit\n")
        with (WORK / "logs/model-import.log").open("w") as stream:
            subprocess.run([sys.executable, str(BASELINE / "bin/madgraph"),
                            "--plain", str(card)], cwd=ROOT, env=environment,
                           stdout=stream, stderr=subprocess.STDOUT, check=True)
        if not (BASELINE / "models/loop_qcd_qed_sm_Gmu/particles.py").exists():
            raise RuntimeError("EW model import failed; inspect model-import.log")


if __name__ == "__main__":
    main()
