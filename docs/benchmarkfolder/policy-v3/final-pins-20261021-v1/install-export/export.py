#!/usr/bin/env python3
"""Generate final-source standalone and NLO outputs with internal dependencies."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import prepare

ROOT, WORK, REPORT, SOURCE = prepare.ROOT, prepare.WORK, prepare.REPORT, prepare.SOURCE


def main():
    installed = SOURCE / "HEPTools/ninjars"
    metadata = json.loads((REPORT / "installation.json").read_text())
    if not metadata["passed"]:
        raise RuntimeError("CLI installation has not passed")
    for name, checksum in metadata["files"].items():
        if prepare.digest(installed / name) != checksum:
            raise RuntimeError(f"Installation changed: {name}")
    source_identity = json.loads((REPORT / "archived-generator-sources.json").read_text())
    if (prepare.digest(REPORT / "archived-generator-sources.json") !=
            metadata["archived_generator_sources_sha256"] or
            prepare.generator_sources(SOURCE, source_identity) != source_identity):
        raise RuntimeError("Archived CLI generator differs from its final-source binding")
    source_names = ("madgraph/iolibs/export_v4.py",
                    "madgraph/iolibs/ufo_expression_parsers.py",
                    "madgraph/various/q_polynomial.py",
                    "madgraph/loop/loop_exporters.py",
                    "madgraph/iolibs/export_fks.py",
                    "madgraph/iolibs/helas_call_writers.py",
                    "madgraph/iolibs/template_files/loop_optimized/NINJARS_interface.inc",
                    "madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc",
                    "madgraph/iolibs/template_files/loop_optimized/mp_helas_calls_split.inc")
    identities = {name: prepare.digest(SOURCE / name) for name in source_names}
    # Generate from the immutable archive. Later documentation or worktree
    # changes cannot relabel its actual committed generation provenance.
    env = os.environ.copy()
    env["TMPDIR"] = str(ROOT / ".ninjars-build/tmp")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["MAKEFLAGS"] = "SHELL=" + subprocess.check_output(
        ["bash", "-c", "command -v bash"], text=True).strip()
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               RAYON_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = {"source_revision": metadata["revision"], "source_sha256": identities,
              "archived_generator_sources_sha256": metadata["archived_generator_sources_sha256"],
              "installation_sha256": prepare.digest(REPORT / "installation.json"), "exports": {}}
    for name, process, output_type in (
            ("standalone", "u u~ > z [virt=QCD]", "standalone_fortran "),
            ("nlo", "u u~ > z [QCD]", "")):
        output = WORK / name
        if output.exists():
            raise RuntimeError("Refusing to overwrite an existing generated output")
        lines = ["set automatic_html_opening False", "set auto_update 0",
                 "set nb_core 1", "set ninja None", "set collier None",
                 f"set ninjars {installed / 'lib'}", "set output_dependencies internal",
                 "import model loop_sm", f"generate {process}",
                 f"output {output_type}{output}", "quit"]
        card = REPORT / f"{name}.mg5"
        with card.open("x") as stream:
            stream.write("\n".join(lines) + "\n")
        command = [sys.executable, str(SOURCE / "bin/madgraph"), "--plain", str(card)]
        start = time.perf_counter()
        with (REPORT / f"{name}-export.log").open("x") as log:
            completed = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
        result["exports"][name] = {"command": command, "exit_code": completed.returncode,
                                   "seconds": time.perf_counter()-start,
                                   "card_sha256": prepare.digest(card),
                                   "log_sha256": prepare.digest(REPORT / f"{name}-export.log")}
        (REPORT / "exports.json").write_text(json.dumps(result, indent=2) + "\n")
        if completed.returncode or not (output / "SubProcesses/P0_uux_z").is_dir():
            raise RuntimeError(f"{name} export failed")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
