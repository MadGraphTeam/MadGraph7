"""Reference values of the Sudakov approximation of the EW corrections of
MadGraph7 (ewsudakov_wrapper.f, ewsudakov_functions.f, compute_ewsudakov of
fks_singular.f).

For every process below, this script generates an 'output ewsudakovsa'
directory with MadGraph7, builds dump_ewsudakov.f in its P* directories
(linked like ewsudakov_sa) with dump_ewsudakov_calls.f, the calls to the
matrix elements of the Sudakov approximation which it writes from the
fks_configurations.json of the directory, and writes the dumped values to
madspace/tests/test_data/ewsudakov/<name>_<subprocess>.json. The madspace
tests read only the JSON files, so they need no Fortran.

The processes need the models loop_qcd_qed_sm_Gmu_forSudakov and
loop_qcd_qed_sm_forSudakov of the UFO model database, which MadGraph
downloads, or which --models takes from a directory.

Run from anywhere with a gfortran in the PATH:

    python madspace/tests/fks_reference/make_ewsudakov_reference.py
        [--points N] [--models DIR] [--keep DIR]

The JSON layout, with Fortran conventions kept (1-based helicities,
interferences and positions, momenta as (E, px, py, pz) in the partonic
c.m. frame):

    process, subprocess, model, ewsudakov_functions_sha (git blob hash of
    ewsudakov_functions.f), fks_configurations (the fks_configurations.json
    of the P* directory, with the ewsudakov block, see get_ewsudakov_info of
    export_fks.py), params (mw, mz, mh, mt, mb, gal (the QED coupling e of
    the logarithms, gal(1)), g (the strong coupling), aewm1, ee, gf,
    scale (the renormalization scale of GET_QCD_LO2)), lo (has_lo1,
    has_lo2, lo1_pos, lo2_pos of ewsudakov_haslo.inc), modes (the mode m of
    every entry below: sud_mod, s_to_rij and rij_ge_mw, m = 4 sud_mod +
    2 s_to_rij + rij_ge_mw)
    points[]: momenta, born (sborn),
        helicities[]: base (0 the Born, k the Goldstone amplitude k), diag
            (the base amplitude at the helicity at LO1), lo2 (at LO2),
            derivatives (of diag with respect to the six parameters of
            numder_*.f), all times comp_idfac,
        interferences[]: [helicity, interference, re, im] (the interference
            at LO1 times comp_idfac, for the interferences of the base
            amplitude of the helicity),
        coefficients[]: [mode, helicity, lsc, ssc, xxc] (get_lsc_diag,
            get_ssc_n_diag, get_xxc_diag for the PDG codes of the base
            amplitude, each as re, im),
        qcd_coefficients[]: [mode, helicity, re, im] (GET_QCD_LO2 at LO2),
        interference_coefficients[]: [mode, helicity, interference, re, im]
            (get_lsc_nondiag + get_xxc_nondiag for cew, get_ssc_n_nondiag_1
            for iz1, get_ssc_n_nondiag_2 for iz2, get_ssc_c for ipm2),
        lo1[]: [mode, helicity, lsc, ssc, xxc, par] (amp_split_ewsud_* of
            sudakov_wrapper at LO1 with ewsud_helselect = helicity, 0 for
            the sum, each as re, im),
        lo2[]: [mode, helicity, qcd, parqcd] (the same at LO2),
        weight[]: per mode, the weight of compute_ewsudakov without
            f_b / g^qcd_power: 2 Re(lsc + ssc + xxc + par) at LO1 and
            2 Re(qcd + parqcd) at LO2 (both at the NLO order LO1 + 2 QED).
    The rows of the lists without a nonzero value are left out.
"""

import argparse
import glob
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
MG_ROOT = os.path.realpath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(HERE, "..", "test_data", "ewsudakov")

GMU = "loop_qcd_qed_sm_Gmu_forSudakov"
ALPHA_MZ = "loop_qcd_qed_sm_forSudakov"
# name: (model, process, subprocesses (None keeps all)). Together they have
# every kind of amplitude of the Sudakov approximation (Goldstone bosons,
# cew, iz1, iz2, ipm2), both schemes of the parameter renormalisation and a
# LO2.
PROCESSES = {
    "uux_wpwm": (GMU, "u u~ > w+ w- [LOonly=QCD] --ewsudakov", None),
    "uux_zz": (ALPHA_MZ, "u u~ > z z [LOonly=QCD] --ewsudakov", None),
    "uux_zh": (GMU, "u u~ > z h [LOonly=QCD] --ewsudakov", None),
    "epem_ttx": (ALPHA_MZ, "e+ e- > t t~ [LOonly=QCD] --ewsudakov", None),
    # a LO2 (the QCD part), with a gluon
    "uux_ddxg_lo2": (GMU, "u u~ > d d~ g [LOonly=QCD QED] --ewsudakov", None),
    "udx_wpa": (GMU, "u d~ > w+ a [LOonly=QCD] --ewsudakov", None),
    "epem_ttxa": (ALPHA_MZ, "e+ e- > t t~ a [LOonly=QCD] --ewsudakov", None),
    # iz1, and identical Higgs bosons
    "epem_zhh": (GMU, "e+ e- > z h h [LOonly=QCD] --ewsudakov", None),
}
# the strong coupling and the renormalization scale of the dump
G = math.sqrt(4 * math.pi * 0.118)
SCALE = 500.0
TYPES = {"cew": 1, "iz1": 2, "iz2": 3, "ipm2": 4}
MODES = [(sm, s2r, rge) for sm in (0, 1) for s2r in (0, 1) for rge in (0, 1)]


def run(cmd, cwd, log, stdin=None):
    with open(log, "w") as f:
        subprocess.run(
            cmd, cwd=cwd, stdin=stdin, stdout=f, stderr=subprocess.STDOUT, check=True
        )


def generate(name, model, process, work, models):
    out = os.path.join(work, name)
    if not os.path.isdir(os.path.join(out, "SubProcesses")):
        script = os.path.join(work, f"{name}.mg")
        model_path = os.path.join(models, model) if models else model
        with open(script, "w") as f:
            f.write("set automatic_html_opening False\n")
            f.write(f"import model {model_path}\n")
            f.write(f"generate {process}\n")
            f.write(f"output ewsudakovsa {out} -f\n")
        run(
            [sys.executable, os.path.join(MG_ROOT, "bin", "madgraph"), script],
            work,
            os.path.join(work, f"{name}_mg.log"),
        )
    # Source needs run_card.inc, which a launch would write
    sys.path.insert(0, MG_ROOT)
    import madgraph.various.banner as banner

    card = banner.RunCardNLO(os.path.join(out, "Cards", "run_card.dat"))
    card.write_include_file(os.path.join(out, "Source"))
    run(
        ["make", "-j8"],
        os.path.join(out, "Source"),
        os.path.join(work, f"{name}_src.log"),
    )
    return out


def write_calls(pdir, info):
    """dump_ewsudakov_calls.f: the calls to the base amplitudes, to their
    parameter renormalisation and to the interferences, by number."""
    ngold = len(info["goldstone"])
    lines = [
        "      subroutine dump_base_me(base, p, hels, ihel)",
        "      implicit none",
        "      include 'nexternal.inc'",
        "      integer base, hels(nexternal-1), ihel",
        "      double precision p(0:3, nexternal-1), ans",
        "      if (base.eq.0) call sborn_onehel(p, hels, ihel, ans)",
    ]
    for k in range(1, ngold + 1):
        lines.append(f"      if (base.eq.{k}) call ewsdk_gold_me_{k}(p, hels, ans)")
    lines += [
        "      end",
        "",
        "      subroutine dump_par_ren(base, p, hels, ihel, invariants)",
        "      implicit none",
        "      include 'nexternal.inc'",
        "      integer base, hels(nexternal-1), ihel",
        "      double precision p(0:3, nexternal-1)",
        "      double precision invariants(nexternal-1, nexternal-1)",
        "      if (base.eq.0) call par_ren_sborn_onehel(p, hels, ihel,",
        "     $ invariants)",
    ]
    for k in range(1, ngold + 1):
        lines += [
            f"      if (base.eq.{k}) call par_ren_ewsdk_gold_me_{k}(p, hels,",
            "     $ ihel, invariants)",
        ]
    lines += [
        "      end",
        "",
        "      subroutine dump_interference(t, p, hels)",
        "      implicit none",
        "      include 'nexternal.inc'",
        "      integer t, hels(nexternal-1)",
        "      double precision p(0:3, nexternal-1)",
        "      double complex ans",
    ]
    for t in range(1, len(info["interference"]) + 1):
        lines.append(f"      if (t.eq.{t}) call ewsdk_me_{t}(p, hels, ans)")
    lines += [
        "      end",
        "",
        "      double precision function dump_idfac(base)",
        "      implicit none",
        "      integer base",
        "      double precision compensate_identical_factor",
        "      external compensate_identical_factor",
        "      dump_idfac = 1d0",
        "      if (base.gt.0) dump_idfac = compensate_identical_factor(base)",
        "      end",
    ]
    with open(os.path.join(pdir, "dump_ewsudakov_calls.f"), "w") as f:
        f.write("\n".join(lines) + "\n")


def two_body(p, m1, m2, rng):
    """Two momenta of masses m1, m2 from the decay of p, isotropic in its
    rest frame."""
    m = math.sqrt(max(p[0] ** 2 - p[1] ** 2 - p[2] ** 2 - p[3] ** 2, 0.0))
    lam = (m * m - (m1 + m2) ** 2) * (m * m - (m1 - m2) ** 2)
    k = math.sqrt(max(lam, 0.0)) / (2 * m)
    cos = rng.uniform(-1, 1)
    sin = math.sqrt(1 - cos * cos)
    phi = rng.uniform(0, 2 * math.pi)
    q = [k * sin * math.cos(phi), k * sin * math.sin(phi), k * cos]
    out = []
    for sign, mass in ((1, m1), (-1, m2)):
        e = math.sqrt(k * k + mass * mass)
        v = [sign * x for x in q]
        # boost from the rest frame of p
        b = [p[i] / p[0] for i in (1, 2, 3)]
        b2 = sum(x * x for x in b)
        gamma = p[0] / m
        bv = sum(b[i] * v[i] for i in range(3))
        factor = (gamma - 1) * bv / b2 if b2 > 0 else 0.0
        out.append(
            [gamma * (e + bv)]
            + [v[i] + factor * b[i] + gamma * b[i] * e for i in range(3)]
        )
    return out


def born_point(sqrt_s, masses, rng, cos_theta=None):
    """Born momenta at sqrt_s in the partonic c.m. frame, the incoming along
    z, the final state from sequential two-body decays (the first particle
    with cos_theta along z where given)."""
    m1, m2 = masses[:2]
    s = sqrt_s * sqrt_s
    k = math.sqrt((s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2)) / (2 * sqrt_s)
    e1 = (s + m1 * m1 - m2 * m2) / (2 * sqrt_s)
    p = [[e1, 0.0, 0.0, k], [sqrt_s - e1, 0.0, 0.0, -k]]
    rest = [sqrt_s, 0.0, 0.0, 0.0]
    final = masses[2:]
    for i in range(len(final) - 1):
        others = final[i + 1 :]
        m_rest = math.sqrt(rest[0] ** 2 - sum(x * x for x in rest[1:]))
        low = sum(others)
        high = m_rest - final[i]
        if len(others) == 1:
            m_others = others[0]
        else:
            m_others = low + (high - low) * rng.uniform(0.05, 0.95)
        a, b = two_body(rest, final[i], m_others, rng)
        if i == 0 and cos_theta is not None:
            k = math.sqrt(sum(x * x for x in a[1:]))
            sin = math.sqrt(1 - cos_theta**2)
            a = [a[0], k * sin, 0.0, k * cos_theta]
            b = [rest[0] - a[0], -a[1], -a[2], -a[3]]
        p.append(a)
        rest = b
    p.append(rest)
    return p


def masses_of(pdgs, params):
    mass = {6: params["mt"], 23: params["mz"], 24: params["mw"], 25: params["mh"]}
    return [mass.get(abs(pdg), 0.0) for pdg in pdgs]


def dump(pdir, info, points, log):
    write_calls(pdir, info)
    for f in ["dump_ewsudakov.f", "makefile_ewsudakov"]:
        shutil.copy(os.path.join(HERE, f), pdir)
    run(["make", "-f", "makefile_ewsudakov", "dump_ewsudakov"], pdir, log)
    lines = [f"{len(points)}", f"{G!r} {SCALE!r}", f"{len(info['goldstone'])}"]
    for gold in info["goldstone"]:
        lines.append(" ".join(str(x) for x in [len(gold["legs"])] + gold["legs"]))
    born_pdgs = info["born_pdgs"]
    lines.append(" ".join(str(x) for x in born_pdgs))
    for gold in info["goldstone"]:
        lines.append(" ".join(str(x) for x in gold["pdgs"]))
    lines.append(f"{len(info['interference'])}")
    for amp in info["interference"]:
        fields = [TYPES[amp["type"]], amp["base"]]
        for j in range(2):
            if j < len(amp["legs"]):
                fields += [amp["legs"][j], amp["old_pdgs"][j], amp["new_pdgs"][j]]
            else:
                fields += [0, 0, 0]
        lines.append(" ".join(str(x) for x in fields))
    for p in points:
        lines += [" ".join(repr(x) for x in momentum) for momentum in p]
    result = subprocess.run(
        ["./dump_ewsudakov"],
        cwd=pdir,
        input="\n".join(lines) + "\n",
        capture_output=True,
        text=True,
        check=True,
    )
    # the Fortran stops with exit code 0, and reports some failures only
    if "error" in result.stdout.lower() or "strange" in result.stdout.lower():
        raise RuntimeError(f"dump_ewsudakov in {pdir} failed:\n{result.stdout[-2000:]}")
    return result.stdout


def floats(fields):
    return [float(v) for v in fields]


def parse(text):
    data = {"points": []}
    point = None
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        key, rest = fields[0], fields[1:]
        if key == "PARAMS":
            mw, mz, mh, mt, gal, g = floats(rest)
            data["params"] = dict(mw=mw, mz=mz, mh=mh, mt=mt, gal=gal, g=g)
        elif key == "PARAMS2":
            aewm1, ee, scale, gf = floats(rest)
            data["params"].update(aewm1=aewm1, ee=ee, scale=scale, gf=gf)
        elif key == "MB":
            data["params"]["mb"] = float(rest[0])
        elif key == "LO":
            data["lo"] = dict(
                has_lo1=rest[0] == "T",
                has_lo2=rest[1] == "T",
                lo1_pos=int(rest[2]),
                lo2_pos=int(rest[3]),
            )
        elif key == "POINT":
            point = {
                "momenta": [],
                "helicities": [],
                "interferences": [],
                "coefficients": [],
                "qcd_coefficients": [],
                "interference_coefficients": [],
                "lo1": [],
                "lo2": [],
            }
            data["points"].append(point)
        elif key == "P":
            point["momenta"].append(floats(rest[1:]))
        elif key == "BORN":
            point["born"] = float(rest[0])
        elif key == "HEL":
            values = floats(rest[2:])
            point["helicities"].append(
                dict(
                    base=int(rest[1]),
                    diag=values[0],
                    lo2=values[1],
                    derivatives=values[2:],
                )
            )
        elif key == "INT":
            point["interferences"].append(
                [int(rest[0]), int(rest[1])] + floats(rest[2:])
            )
        elif key == "COEF":
            point["coefficients"].append(
                [int(rest[0]), int(rest[1])] + floats(rest[2:])
            )
        elif key == "QCOEF":
            point["qcd_coefficients"].append(
                [int(rest[0]), int(rest[1])] + floats(rest[2:])
            )
        elif key == "ICOEF":
            point["interference_coefficients"].append(
                [int(x) for x in rest[:3]] + floats(rest[3:])
            )
        elif key == "WLO1":
            point["lo1"].append([int(rest[0]), int(rest[1])] + floats(rest[2:]))
        elif key == "WLO2":
            point["lo2"].append([int(rest[0]), int(rest[1])] + floats(rest[2:]))
    for point in data["points"]:
        # the rows without a nonzero value are left out: the tests take
        # them as zero
        for key, start in (
            ("interferences", 2),
            ("coefficients", 2),
            ("qcd_coefficients", 2),
            ("interference_coefficients", 3),
            ("lo1", 2),
            ("lo2", 2),
        ):
            point[key] = [row for row in point[key] if any(row[start:])]
        weight = [0.0] * len(MODES)
        for row in point["lo1"]:
            if row[1] == 0:
                weight[row[0]] += 2 * sum(row[2::2])
        for row in point["lo2"]:
            if row[1] == 0:
                weight[row[0]] += 2 * sum(row[2::2])
        point["weight"] = weight
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=4, help="points per process")
    parser.add_argument("--models", help="directory with the models of the processes")
    parser.add_argument("--keep", help="keep the generated outputs in this directory")
    parser.add_argument(
        "--only", nargs="*", help="regenerate only these processes (names of PROCESSES)"
    )
    args = parser.parse_args()

    sha = subprocess.run(
        ["git", "hash-object", "Template/NLO/SubProcesses/ewsudakov_functions.f"],
        cwd=MG_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    os.makedirs(OUT_DIR, exist_ok=True)
    work = args.keep or tempfile.mkdtemp(prefix="ewsudakov_reference_")
    os.makedirs(work, exist_ok=True)
    rng = random.Random(20261106)
    for name, (model, process, keep) in PROCESSES.items():
        if args.only and name not in args.only:
            continue
        out = generate(name, model, process, work, args.models)
        for pdir in sorted(glob.glob(os.path.join(out, "SubProcesses", "P*"))):
            sub = os.path.basename(pdir)
            if keep is not None and sub not in keep:
                continue
            with open(os.path.join(pdir, "fks_configurations.json")) as f:
                configurations = json.load(f)
            info = dict(
                configurations["ewsudakov"], born_pdgs=configurations["born_pdgs"]
            )
            # the masses for the points, from a first dump without points
            first = parse(dump(pdir, info, [], os.path.join(work, f"{name}_{sub}.log")))
            masses = masses_of(configurations["born_pdgs"], first["params"])
            points = [born_point(3000.0, masses, rng) for _ in range(args.points)]
            # a forward one (small |t|, below MW^2 where rij_ge_mw matters)
            # and one at a lower energy
            points.append(born_point(3000.0, masses, rng, cos_theta=0.99999))
            points.append(born_point(max(500.0, 1.5 * sum(masses)), masses, rng))
            data = parse(
                dump(pdir, info, points, os.path.join(work, f"{name}_{sub}.log"))
            )
            data.update(
                process=process,
                subprocess=sub,
                model=model,
                ewsudakov_functions_sha=sha,
                fks_configurations=configurations,
                modes=[list(m) for m in MODES],
            )
            path = os.path.join(OUT_DIR, f"{name}_{sub}.json")
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.write("\n")
            print(f"{path}: {len(data['points'])} points")
    if not args.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
