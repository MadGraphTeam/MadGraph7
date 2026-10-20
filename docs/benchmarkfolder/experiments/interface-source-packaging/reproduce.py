#!/usr/bin/env python3
"""Exercise interface-source packaging using an already validated archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
from madgraph.loop.loop_exporters import LoopProcessOptimizedExporterFortranSA

parser = argparse.ArgumentParser()
parser.add_argument("--work", type=Path,
                    default=ROOT / ".benchmark-work/experiments/interface-source-packaging")
args = parser.parse_args()
work = args.work.resolve()
if not work.is_relative_to(ROOT) or work.exists():
    parser.error("Choose a new work directory inside this repository")
build = work / "build"
build.mkdir(parents=True)
installed = ROOT / "HEPTools/ninjars"
for name in ("Makefile", "mninjars.f90", "ninjars.h"):
    shutil.copy2(ROOT / "vendor/ninjars" / name, build / name)
for name in ("libninjars.a", "mninjars.mod", "ninjars-native-libs.txt",
             "ninjars-link-before.txt"):
    shutil.copy2(installed / "lib" / name, build / name)

# -o explicitly reuses the validated archive, exercising only the changed
# installation recipe. Compiler/runtime numerical validation is separate.
command = ["make", "-o", "libninjars.a", "install", f"PREFIX={work / 'installed'}"]
with (HERE / "package-install.log").open("w") as log:
    subprocess.run(command, cwd=build, stdout=log, stderr=subprocess.STDOUT, check=True)
lib = work / "installed/lib"
shutil.copy2(installed / "lib/ninjars-config.json", lib)
exporter = object.__new__(LoopProcessOptimizedExporterFortranSA)
exporter.tir_available_dict = {"ninjars": True}
exporter.dependencies = "internal"
copied = work / "relocated/lib"
exporter.link_TIR(str(copied), str(lib), "libninjars.a", "ninjars")
files = ("libninjars.a", "mninjars.mod", "ninjars-config.json",
         "ninjars-native-libs.txt", "ninjars-link-before.txt",
         "ninjars.h", "mninjars.f90")
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
hashes = {}
for name in files:
    target = copied / name
    assert target.is_file() and not target.is_symlink(), name
    hashes[name] = digest(target)
assert hashes["mninjars.f90"] == digest(ROOT / "vendor/ninjars/mninjars.f90")
assert hashes["libninjars.a"] == digest(installed / "lib/libninjars.a")

compile_dir = work / "recompile"
compile_dir.mkdir()
compile_command = ["gfortran", "-O3", "-fPIC", "-c", str(copied / "mninjars.f90")]
with (HERE / "module-recompile.log").open("w") as log:
    subprocess.run(compile_command, cwd=compile_dir, stdout=log,
                   stderr=subprocess.STDOUT, check=True)
assert (compile_dir / "mninjars.mod").is_file()
result = {
    "passed": True,
    "scope": "Installation copy recipe, internal dependency copy and module-source recompilation",
    "numerical_archive_rebuilt": False,
    "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
    "commands": [command, compile_command],
    "copied_regular_files_sha256": hashes,
    "recompiled_module_sha256": digest(compile_dir / "mninjars.mod"),
}
(HERE / "verification.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
