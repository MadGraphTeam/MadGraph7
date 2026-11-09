#!/usr/bin/env python3
"""Separate numerical CMS qualification; no performance acceptance claims.

Source tools/tirust_env.sh, then invoke prepare/build/points/run/summarize.
The standard benchmark protocol and exact regression criterion are reused, but
all command files, model/card/point pins, failures and observations are retained
under experiments/complex-mass. Generated executables remain ignored.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from decimal import Decimal, localcontext
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

import tirust_benchmark as campaign
from tirust_benchmark_points import generate_points, validate_points, write_points

MAIN_REPORT = campaign.REPORT
REPORT = MAIN_REPORT / "experiments/complex-mass"
CASE = {"id": "cms_ew_n2_epem_mupmum", "model": "loop_qcd_qed_sm_Gmu",
        "process": "e+ e- > mu+ mu- [virt=QED]", "nfinal": 2,
        "role": "complex-mass-stress", "complex_mass_scheme": True}


def gram_diagnostic(point):
    """Three-vector Gram determinant from round-trippable input decimals.

    All four external vectors are dependent in 2->2 scattering. The incoming
    pair and first outgoing vector are independent away from the beam, and
    expose the forward small-Gram limit without a structurally zero 4x4 test.
    Decimal arithmetic avoids diagnosing a determinant via cancelled DP sums.
    """
    with localcontext() as context:
        context.prec = 80
        vectors = [[Decimal("%.17e" % value) for value in vector]
                   for vector in point["momenta"][:3]]
        matrix = [[2*(left[0]*right[0]-sum(a*b for a,b in zip(left[1:],right[1:])))
                   for right in vectors] for left in vectors]
        a,b,c = matrix
        determinant = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
        scale = max(abs(value) for row in matrix for value in row)
        return {"vectors": [1,2,3], "definition": "G_ij=2*p_i.p_j, metric (+---)",
                "decimal_precision": 80, "determinant_gev6": str(determinant),
                "normalized_determinant": str(determinant/scale**3)}


def setup():
    REPORT.mkdir(parents=True, exist_ok=True)
    pin = MAIN_REPORT / "inputs/loop_qcd_qed_sm_Gmu.json"
    destination = REPORT / "inputs" / pin.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.read_bytes() != pin.read_bytes():
        raise ValueError("Pinned EW model changed")
    if not destination.exists():
        shutil.copy2(pin, destination)
    campaign.REPORT = REPORT
    config = json.loads((MAIN_REPORT / "cases.json").read_text())
    config["cases"] = [CASE]
    return config


def two_body(identifier, kind, masses, energy, angle):
    """Physical 2->2 momenta, with a controllably small transverse component."""
    def pair(m1, m2):
        if energy <= m1 + m2:
            raise ValueError("Energy must exceed two-body threshold")
        e1 = (energy * energy + m1 * m1 - m2 * m2) / (2 * energy)
        e2 = energy - e1
        momentum = math.sqrt((energy-m1-m2)*(energy+m1+m2)*
                             (energy-m1+m2)*(energy+m1-m2))/(2*energy)
        return e1, e2, momentum
    a, b, incoming = pair(*masses[:2])
    c, d, outgoing = pair(*masses[2:])
    transverse, longitudinal = outgoing*math.sin(angle), outgoing*math.cos(angle)
    return {"id": identifier, "kind": kind, "sqrt_s": energy,
            "masses": masses[2:], "incoming_masses": masses[:2],
            "momenta": [[a, 0., 0., incoming], [b, 0., 0., -incoming],
                        [c, transverse, 0., longitudinal],
                        [d, -transverse, 0., -longitudinal]],
            "angle_rad": angle}


def freeze_points(source):
    from madgraph.models.check_param_card import ParamCard
    output = campaign.output_dir(source, CASE)
    masses = json.loads((output / "compilation.json").read_text())["masses"]
    card = ParamCard(str(output / "Cards/param_card.dat"))
    mz, mw = card["mass"].get((23,)).value, card["mass"].get((24,)).value
    wz, ww = card["decay"].get((23,)).value, card["decay"].get((24,)).value
    if min(wz, ww) <= 0:
        raise ValueError("Complex-mass qualification requires nonzero W and Z widths")
    seed = 20261109
    points = generate_points(masses[2:], count=12, energy=1000., seed=seed,
                             incoming_masses=masses[:2])
    for offset in (-1., 0., 1.):
        points.append(two_body(f"z-resonance-{offset:+g}", "z-resonance", masses,
                               mz + offset*wz, 0.8))
    for offset in (-1e-6, 0., 1e-6):
        points.append(two_body(f"ww-threshold-{offset:+g}", "internal-ww-threshold",
                               masses, 2*mw*(1+offset), 0.8))
    for angle in (1e-3, 1e-6, 1e-8):
        points.append(two_body(f"forward-{angle:g}", "small-gram", masses, 1000., angle))
    validate_points(points, masses[2:], incoming_masses=masses[:2])
    for point in points:
        point["gram"] = gram_diagnostic(point)
    path = REPORT / "points" / (CASE["id"] + ".dat")
    metadata_path = path.with_suffix(".json")
    inputs = {"masses": masses[2:], "incoming_masses": masses[:2], "count": len(points),
              "seed": seed, "energy": 1000., "wz": wz, "ww": ww, "mz": mz, "mw": mw}
    if path.exists():
        metadata = json.loads(metadata_path.read_text())
        if (metadata["inputs"] != inputs or metadata["points"] != points
                or metadata["sha256"] != campaign.digest(path)):
            raise ValueError("Frozen CMS points changed")
        return path, metadata
    path.parent.mkdir(parents=True, exist_ok=True)
    write_points(path, points)
    metadata = {"inputs": inputs, "points": points, "sha256": campaign.digest(path),
                "scope": "Numerical stress fixtures, not main-campaign acceptance points"}
    campaign.dump(metadata_path, metadata)
    return path, metadata


def prove_complex_model(source):
    """Execute generated SETPARA and inspect its actual complex mass common block."""
    output = campaign.output_dir(source, CASE)
    directory = campaign.subprocess_dir(source, CASE)
    probe = output / "cms_probe.f"
    probe.write_text("""      PROGRAM CMS_PROBE
      IMPLICIT NONE
      INCLUDE 'coupl.inc'
      INCLUDE 'input.inc'
      CALL SETPARA('param_card.dat')
      OPEN(73,FILE='cms_masses.dat',STATUS='REPLACE')
      WRITE(73,'(ES26.17E3)') MDL_CMSPARAM
      WRITE(73,'(4ES26.17E3)') MDL_MW,MDL_WW,CMASS_MDL_MW**2
      WRITE(73,'(4ES26.17E3)') MDL_MZ,MDL_WZ,CMASS_MDL_MZ**2
      CLOSE(73)
      END
""")
    executable = output / "cms_probe"
    campaign.run(["gfortran", "-o", str(executable), "-I" + str(output / "Source/MODEL"),
                  str(probe), str(output / "lib/libmodel.a")], campaign.ROOT,
                 campaign.WORK / "logs" / f"cms-probe-compile-{source}.log")
    campaign.run([str(executable)], directory,
                 campaign.WORK / "logs" / f"cms-probe-{source}.log")
    lines = (directory / "cms_masses.dat").read_text().splitlines()
    if float(lines[0]) != 1.:
        raise ValueError("Generated runtime model did not activate CMS")
    masses = {}
    for name, line in zip(("W", "Z"), lines[1:]):
        mass, width, real, imaginary = map(float, line.split())
        error = abs(complex(real, imaginary)-complex(mass*mass, -mass*width))/mass**2
        if width <= 0. or imaginary >= 0. or error > 1e-13:
            raise ValueError("Runtime complex mass differs from M²-iMΓ")
        masses[name] = {"mass": mass, "width": width, "squared_mass": [real, imaginary],
                        "relative_reconstruction_error": error}
    destination = REPORT / "inputs" / source / CASE["id"]
    shutil.copy2(probe, destination / probe.name)
    shutil.copy2(directory / "cms_masses.dat", destination / "cms_masses.dat")
    evidence = {"CMSParam": 1., "masses": masses, "probe_sha256": campaign.digest(probe),
                "parameter_card_sha256": campaign.digest(output / "Cards/param_card.dat"),
                "model_library_sha256": campaign.digest(output / "lib/libmodel.a"),
                "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0]}
    campaign.dump(destination / "complex-model-proof.json", evidence)
    return evidence


def observe(source, backend, precision, config, cpu, timeout=300., tag=""):
    _, points = freeze_points(source)
    config["points_per_case"] = points["inputs"]["count"]
    if precision == "qp":
        config["madloop_parameters"]["CTModeRun"] = 4
        config["madloop_parameters"]["CTModeInit"] = 4
    if not re.fullmatch(r"[A-Za-z0-9_-]*", tag):
        raise ValueError("Run tag must contain only letters, digits, underscore or hyphen")
    label = f"backend-{backend}-{precision}" + ("-" + tag if tag else "")
    report = REPORT / "runs" / source / label
    report.mkdir(parents=True, exist_ok=False)
    proof_path = REPORT / "inputs" / source / CASE["id"] / "complex-model-proof.json"
    proof = json.loads(proof_path.read_text())
    if (proof["CMSParam"] != 1. or proof["parameter_card_sha256"] !=
            campaign.digest(campaign.output_dir(source, CASE) / "Cards/param_card.dat")):
        raise ValueError("Runtime complex-model proof differs from the selected physical card")
    shutil.copy2(Path(__file__), report / Path(__file__).name)
    record = {"case": CASE, "backend": backend, "source": source, "precision_request": precision,
              "created_utc": campaign.utc_now(), "scope": "Numerical only; timing is not qualified",
              "points_sha256": points["sha256"], "complex_model": proof, "timeout_seconds": timeout}
    campaign.dump(report / "specification.json", record)
    try:
        with ExitStack() as stack:
            # Symbolica-backed reducers share a bounded concurrent-run allowance.
            # Serialize numerical qualification with the other native audits.
            if backend in (8, 9):
                lock = stack.enter_context((campaign.ROOT / ".tirust-build/numerical-tests.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX)
            worker, settings = campaign.start_worker(stack, source, CASE, [backend], config, cpu, report,
                                                    timeout=timeout)
            result = campaign.observe(worker, [backend], 1, report / "raw", count=config["points_per_case"])
            record.update(status="complete", settings=settings, observation=result)
        # ExitStack's successful STOP/BYE is part of qualification.
        campaign.dump(report / "summary.json", record)
        return record
    except BaseException as error:
        campaign.dump(report / "failure.json", {**record, "status": "failed", "error": repr(error)})
        raise


def summarize():
    from tirust_benchmark_worker import parse_outputs
    complete, failed, comparisons = {}, {}, []
    for report in sorted((REPORT / "runs").glob("*/*")):
        label = "/".join(report.parts[-2:])
        failure = report / "failure.json"
        summary = report / "summary.json"
        if failure.exists():
            failed[label] = json.loads(failure.read_text())
        elif summary.exists():
            value = json.loads(summary.read_text())
            metadata = json.loads((REPORT / "points" / (CASE["id"] + ".json")).read_text())
            point_file = REPORT / "points" / (CASE["id"] + ".dat")
            inputs = json.loads((report / (value["source"] + "-inputs.json")).read_text())
            if (value["points_sha256"] != metadata["sha256"]
                    or campaign.digest(point_file) != metadata["sha256"]
                    or value["case"] != CASE or inputs["generation"]["case"] != CASE
                    or inputs["points_sha256"] != metadata["sha256"]
                    or inputs["param_card_sha256"] != inputs["generation"]["parameter_card_sha256"]):
                raise ValueError("CMS observation input identity changed")
            actual = parse_outputs(report / "raw", [value["backend"]], metadata["inputs"]["count"], 1)
            if actual != value["observation"]:
                raise ValueError("CMS summary differs from retained raw observations")
            value["param_card_sha256"] = inputs["param_card_sha256"]
            complete[label] = value
    for left_name, left in complete.items():
        for right_name, right in complete.items():
            if left_name >= right_name:
                continue
            if left["points_sha256"] != right["points_sha256"]:
                raise ValueError("Refusing comparison of different points")
            if left["param_card_sha256"] != right["param_card_sha256"]:
                raise ValueError("Refusing comparison of different physical parameter cards")
            comparison = campaign.compare_rows(left["observation"]["rows"], right["observation"]["rows"])
            comparisons.append({"reference": left_name, "candidate": right_name, **comparison})
    result = {"case": CASE, "scope": "Complex-mass numerical qualification only; no runtime claim",
              "created_utc": campaign.utc_now(), "complete": sorted(complete), "failures": failed,
              "comparisons": comparisons,
              "five_backend_complete": all(any(value["backend"] == backend for value in complete.values())
                                           for backend in (1, 6, 7, 8, 9)),
              "numerical_agreement": None if not comparisons else all(x["passed"] for x in comparisons)}
    campaign.dump(REPORT / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("generate", "build", "points", "run", "summarize"))
    parser.add_argument("--source", choices=("baseline", "candidate"), default="baseline")
    parser.add_argument("--backend", type=int, choices=(1, 6, 7, 8, 9))
    parser.add_argument("--precision", choices=("auto", "qp"), default="auto")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=300.)
    parser.add_argument("--tag", default="")
    parser.add_argument("--cpu", type=int, default=min(__import__("os").sched_getaffinity(0)))
    args = parser.parse_args()
    config = setup()
    if args.phase == "generate":
        result = campaign.generate(args.source, CASE)
        log = campaign.WORK / "logs" / f"generate-{args.source}-{CASE['id']}.log"
        shutil.copy2(log, REPORT / "commands" / log.name)
    elif args.phase == "build":
        result = {"masses": campaign.build(args.source, CASE, jobs=args.jobs),
                  "complex_model": prove_complex_model(args.source)}
    elif args.phase == "points":
        result = str(freeze_points(args.source)[0])
    elif args.phase == "run":
        if args.backend is None:
            parser.error("run requires --backend")
        if args.source == "baseline" and args.backend == 9:
            parser.error("Baseline has no TIRust backend")
        result = observe(args.source, args.backend, args.precision, config, args.cpu, args.timeout, args.tag)
    else:
        result = summarize()
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
