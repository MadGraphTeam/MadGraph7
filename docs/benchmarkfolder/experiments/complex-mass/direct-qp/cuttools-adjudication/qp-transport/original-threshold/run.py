#!/usr/bin/env python3
"""Isolate CutTools QP momentum transport without changing production code.

The new CTSXCUT_QPINPUT symbol preserves QP propagator momenta. Its private
object is linked before the unchanged CutTools archive. Masses, scale,
numerators, coefficients, physical cards and points retain their provenance.
These fixed-QP diagnostics do not certify stability or runtime performance.
"""
import argparse
import difflib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time

REPORT = Path(__file__).resolve().parent
PARENT = REPORT.parent
DIRECT = PARENT.parent
ROOT = REPORT.parents[6]
PROCESS = Path("SubProcesses/P0_epem_mupmum")
ORIGIN = ROOT / ".tirust-build/complex-stress-cuttools-baseline"
WORK = ROOT / ".tirust-build/complex-stress-cuttools-qp-transport"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def inventory(work):
    return {str(p.relative_to(work)): digest(p)
            for p in sorted(list(work.rglob("*.o")) + list(work.rglob("*.a")))}


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Unexpected source anchor: " + old)
    return text.replace(old, new)


def patch_interface(text):
    prefix, rest = text.split("      SUBROUTINE ML5_0_INITCT()\n", 1)
    for old, new in (
        ("      REAL*8 PL(0:3,NLOOPLINE)", "      REAL*16 PL(0:3,NLOOPLINE)"),
        ("      REAL*8 PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)",
         "      REAL*16 PCT(0:3,0:NLOOPLINE-1),ABSPCT(0:3)"),
        ("      REAL*8 REF_P", "      REAL*16 REF_P"),
        ("      CALL CTSXCUT(CTMODE", "      CALL CTSXCUT_QPINPUT(CTMODE"),
        ("C     INITIALIZE CUTTOOLS IF NEEDED", "      IF (CTMODE.LT.4.OR.CTMODE.GT.6)\n"
         "     $ STOP 'QP transport diagnostic requires CTMode 4, 5 or 6'\n"
         "C     INITIALIZE CUTTOOLS IF NEEDED"),
    ):
        prefix = replace_once(prefix, old, new)
    old = "CALL ML5_0_CTLOOP(NLOOPLINE,PL,M2L,RANK,LOOPRES(1"
    if rest.count(old) != 4:
        raise ValueError("Unexpected generated CutTools callers")
    rest = rest.replace(old, "CALL ML5_0_CTLOOP(NLOOPLINE,MP_PL,M2L,RANK,LOOPRES(1")
    return prefix + "      SUBROUTINE ML5_0_INITCT()\n" + rest


def patch_cuttools(text):
    text = replace_once(text, "  subroutine ctsxcut(imode,", "  subroutine ctsxcut_qpinput(imode,")
    text = replace_once(text, "  end subroutine ctsxcut\n", "  end subroutine ctsxcut_qpinput\n")
    text = replace_once(text,
        "  include 'cts_dpr.h'\n   , intent(in), dimension(0:3,0:(number_propagators-1)) :: p",
        "  include 'cts_mpr.h'\n   , intent(in), dimension(0:3,0:(number_propagators-1)) :: p")
    old = "      do k= 0,3; mp_dn(j)%p(k)= dn(j)%p(k); enddo"
    if text.count(old) != 2:
        raise ValueError("Unexpected direct/reverse QP transport")
    text = text.replace(old, "      do k= 0,3; mp_dn(j)%p(k)= p(k,j); enddo", 1)
    text = text.replace(old, "      do k= 0,3; mp_dn(j)%p(k)= p(k,number_propagators-1-j); enddo", 1)
    return text


def save_pair(destination, stem, original, modified):
    (destination / (stem + ".original.f")).write_text(original)
    (destination / (stem + ".instrumented.f")).write_text(modified)
    (destination / (stem + ".patch")).write_text("".join(difflib.unified_diff(
        original.splitlines(True), modified.splitlines(True),
        fromfile=stem + ".original", tofile=stem + ".instrumented")))


def build():
    destination = REPORT / "build"
    destination.mkdir(exist_ok=False)
    shutil.copy2(__file__, destination / "run.py")
    if WORK.exists():
        raise ValueError("Refusing to overwrite isolated build")
    parent_record_path = PARENT / "build-baseline/build.json"
    parent_record = json.loads(parent_record_path.read_text())
    if inventory(ORIGIN) != parent_record["objects_archives_after"]:
        raise ValueError("Origin no longer matches its audited cutoff build")
    for filename, field in (("check", "executable_sha256"),
                            ("check_sa.f", "driver_sha256"),
                            ("CT_interface.f", "instrumented_interface_sha256")):
        if digest(ORIGIN / PROCESS / filename) != parent_record[field]:
            raise ValueError("Origin bound executable/source changed")
    if digest(ORIGIN / "Cards/param_card.dat") != parent_record["physical_card_sha256"]:
        raise ValueError("Origin physical card changed")
    shutil.copytree(ORIGIN, WORK, symlinks=True)
    proc = WORK / PROCESS
    before = inventory(WORK)
    interface = (proc / "CT_interface.f").read_text()
    patched_interface = patch_interface(interface)
    (proc / "CT_interface.f").write_text(patched_interface)
    save_pair(destination, "CT_interface", interface, patched_interface)
    original_source = ROOT / "vendor/CutTools/src/cts/cts_cutroutines.f90"
    original = original_source.read_text()
    modified = patch_cuttools(original)
    private = proc / "qpinput"
    private.mkdir()
    for header in original_source.parent.glob("*.h"):
        shutil.copy2(header, private / header.name)
    (private / "cts_cutroutines_qpinput.f90").write_text(modified)
    save_pair(destination, "cts_cutroutines", original, modified)
    overlay = "include makefile\nLINKLIBS := cts_cutroutines_qpinput.o $(LINKLIBS)\n"
    (proc / "qp_transport.mk").write_text(overlay)
    (destination / "qp_transport.mk").write_text(overlay)
    module = ROOT / "vendor/CutTools/includects"
    flags = ["-w", "-fPIC", "-ffixed-line-length-132", "-I../../Source/MODEL",
             "-I../../Source/DHELAS", "-I" + str(module)]
    definitions = (WORK / "SubProcesses/MadLoop_makefile_definitions").read_text()
    include = re.findall(r"(?m)^LOOP_INCLUDE = (.*)$", definitions)
    if len(include) != 1 or "$" in include[0]:
        raise ValueError("Unexpected generated include flags")
    flags += shlex.split(include[0])
    commands = [
        [shutil.which("gfortran"), *flags, "-c", "CT_interface.f", "-o", "CT_interface.o"],
        [shutil.which("gfortran"), "-w", "-fPIC", "-I" + str(module), "-c",
         "cts_cutroutines_qpinput.f90", "-o", "../cts_cutroutines_qpinput.o"],
        ["make", "-f", "qp_transport.mk", "SHELL=" + shutil.which("bash"), "-j1", "check"],
    ]
    with (destination / "build.log").open("w") as log:
        for command, cwd in zip(commands, (proc, private, proc)):
            subprocess.run(command, cwd=cwd, env=dict(os.environ, PWD=str(cwd)),
                           stdout=log, stderr=subprocess.STDOUT, check=True)
    after = inventory(WORK)
    changed = [p for p in before if before[p] != after.get(p)]
    added = sorted(after.keys() - before.keys())
    if changed != [str(PROCESS / "CT_interface.o")] or added != [str(PROCESS / "cts_cutroutines_qpinput.o")]:
        raise ValueError("Unexpected numerical object/archive changes")
    if inventory(ORIGIN) != before or digest(original_source) != hashlib.sha256(original.encode()).hexdigest():
        raise ValueError("Origin or vendor source changed")
    shutil.copy2(__file__, destination / "run.py")
    dump(destination / "build.json", {
        "origin": str(ORIGIN.relative_to(ROOT)), "work": str(WORK.relative_to(ROOT)),
        "parent_build_manifest_sha256": digest(parent_record_path), "commands": commands,
        "command_working_directories": [str(p.relative_to(ROOT)) for p in (proc, private, proc)],
        "objects_archives_before": before, "objects_archives_after": after,
        "changed_objects": changed, "added_objects": added,
        "private_symbol": "ctsxcut_qpinput_", "production_sources_changed": False,
        "vendor_source_sha256": digest(original_source),
        "interface_source_sha256": digest(proc / "CT_interface.f"),
        "private_source_sha256": digest(private / "cts_cutroutines_qpinput.f90"),
        "private_headers": {p.name: digest(p) for p in sorted(private.glob("*.h"))},
        "make_overlay_sha256": digest(proc / "qp_transport.mk"),
        "patches": {p.name: digest(p) for p in sorted(destination.glob("*.patch"))},
        "headers_modules": {p.name: digest(p) for p in sorted(module.iterdir())
                            if p.suffix in (".h", ".mod") and p.is_file()},
        "driver_sha256": digest(proc / "check_sa.f"),
        "executable_sha256": digest(proc / "check"),
        "physical_card_sha256": digest(WORK / "Cards/param_card.dat"),
        "runner_sha256": digest(Path(__file__)),
    })


def run(args):
    destination = REPORT / args.label
    destination.mkdir(exist_ok=False)
    proc = WORK / PROCESS
    build_file = REPORT / "build/build.json"
    record = json.loads(build_file.read_text())
    if inventory(WORK) != record["objects_archives_after"]:
        raise ValueError("Bound numerical objects/archives changed")
    for filename, field in (("check", "executable_sha256"), ("check_sa.f", "driver_sha256"),
                            ("CT_interface.f", "interface_source_sha256"),
                            ("qpinput/cts_cutroutines_qpinput.f90", "private_source_sha256")):
        if digest(proc / filename) != record[field]:
            raise ValueError("Bound source or executable changed")
    if digest(WORK / "Cards/param_card.dat") != record["physical_card_sha256"]:
        raise ValueError("Bound physical card changed")
    # Use exact frozen point-20 bytes from the completed cutoff experiment.
    point = PARENT / "baseline-threshold-1e-10/PS.input"
    parent_status = json.loads((point.parent / "status.json").read_text())
    if digest(point) != parent_status["point_input_sha256"]:
        raise ValueError("Frozen point-20 input changed")
    card = (point.parent / "MadLoopParams.dat").read_text()
    if hashlib.sha256(card.encode()).hexdigest() != parent_status["numerical_card_sha256"]:
        raise ValueError("Parent numerical card changed")
    for name in ("CTModeRun", "CTModeInit"):
        card, count = re.subn(r"(?mi)(^#" + name + r"\s*\n)[^\n]*", lambda m: m[1] + str(args.mode), card)
        if count != 1:
            raise ValueError("Unexpected numerical card layout")
    (destination / "PS.input").write_bytes(point.read_bytes())
    (destination / "MadLoopParams.dat").write_text(card)
    (destination / "cuttools_onshell.input").write_text("%.17e\n" % args.threshold)
    shutil.copy2(__file__, destination / "run.py")
    for filename in ("PS.input", "cuttools_onshell.input"):
        shutil.copy2(destination / filename, proc / filename)
    (proc / "MadLoopParams.dat").unlink(missing_ok=True)
    (WORK / "SubProcesses/MadLoop5_resources/MadLoopParams.dat").write_text(card)
    (proc / "result.dat").unlink(missing_ok=True)
    status = {"source": "baseline-with-isolated-QP-input-transport", "backend": 1,
              "point_index": 20, "parent_points_sha256": parent_status["parent_points_sha256"],
              "mode": args.mode, "cuttools_scalar_threshold": args.threshold,
              "stability_certified": False, "performance_eligible": False,
              "build_manifest_sha256": digest(build_file),
              "physical_card_sha256": record["physical_card_sha256"],
              "numerical_card_sha256": digest(destination / "MadLoopParams.dat"),
              "point_input_sha256": digest(destination / "PS.input"),
              "threshold_input_sha256": digest(destination / "cuttools_onshell.input"),
              "runner_sha256": digest(Path(__file__))}
    started = time.monotonic()
    with (destination / "output.log").open("w") as log:
        try:
            result = subprocess.run([str(proc / "check")], cwd=proc, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=args.timeout,
                                    env=dict(os.environ, GFORTRAN_UNBUFFERED_ALL="y"))
            status.update(returncode=result.returncode, timed_out=False)
        except subprocess.TimeoutExpired:
            status.update(timed_out=True, timeout_seconds=args.timeout)
    status["elapsed_seconds_diagnostic_only"] = time.monotonic() - started
    if (proc / "result.dat").exists():
        shutil.copy2(proc / "result.dat", destination / "result.dat")
        status["result_sha256"] = digest(destination / "result.dat")
    status["output_sha256"] = digest(destination / "output.log")
    if inventory(WORK) != record["objects_archives_after"]:
        raise ValueError("Numerical objects/archives changed during observation")
    dump(destination / "status.json", status)
    print(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("build", "run"))
    parser.add_argument("--label")
    parser.add_argument("--threshold", type=float, default=1e-10)
    parser.add_argument("--mode", type=int, choices=(4, 5, 6), default=4)
    parser.add_argument("--timeout", type=float, default=30.)
    args = parser.parse_args()
    if Path.cwd().resolve() != ROOT:
        parser.error("Run from the repository root")
    if args.phase == "build":
        build()
    else:
        if not args.label or not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
            parser.error("Run needs a safe, previously unused label")
        if not math.isfinite(args.threshold) or not 0 < args.threshold < 1:
            parser.error("Invalid scalar threshold")
        run(args)


if __name__ == "__main__":
    main()
