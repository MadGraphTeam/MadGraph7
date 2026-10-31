"""Reference values of the FKS phase space of MadGraph7's genps_fks.f.

For every process below, this script generates an
'output standalone_fortran --fks --limits' directory with MadGraph7, builds
fks_kinematics_dump.f in its P* directories (linked like test_soft_col_limits)
and writes the dumped kinematics to madspace/tests/test_data/fks/<name>.json.
The madspace FKS tests read only the JSON files, so they need no Fortran.

Run from anywhere with a gfortran in the PATH:

    python madspace/tests/fks_reference/make_fks_reference.py [--points N]

The JSON layout, with Fortran conventions kept (1-based particle indices,
counter-event index k = -2..2 stored at position k + 2, momenta as
(E, px, py, pz) in the partonic c.m. frame):

    process, subprocess, genps_fks_sha (git blob hash of genps_fks.f),
    nexternal, nincoming, ndim, lpp, ebeam
    configs[]: nfks, i_fks, j_fks, pdg, masses, points[]
    points[]:  x (unit-cube vector, radiation in the last three entries),
               isolsign, jac, jac_cnt (full weight of generate_momenta),
               p_born, p_ev, p_cnt, xi_ev, xi_hat_ev, y_ev, xiimax_ev,
               xi_cnt, xi_hat_cnt, xiimax_cnt, xbjrk_ev, xbjrk_cnt, shat_ev,
               shat_cnt, tau_ev, ycm_ev, tau_cnt, ycm_cnt, kperp,
               shat_born, tau_born, ycm_born, xbjrk_born, tau_lower_bound,
               and for prefix in (rad, cut): prefix_jac, prefix_jac_cnt,
               prefix_p_ev, prefix_p_cnt, prefix_xi_ev, prefix_xi_hat_ev,
               prefix_y_ev, prefix_xiimax_ev, prefix_xi_cnt, prefix_xi_hat_cnt,
               prefix_xiimax_cnt, prefix_xbjrk_ev, prefix_xbjrk_cnt,
               prefix_shat_ev, prefix_shat_cnt
    The rad_ and cut_ entries come from rerunning generate_FKS_kinematics on
    the Born of the point with unit Born weights (the jac entries are then
    the radiation Jacobian times compute_flux), cut_ with and rad_ without
    the lower bound tau_lower_bound (on tau = s_hat / s_lab) that the cuts
    put on the partonic energy of the real emission. The bound restricts xi
    and y for an initial-state emitter. rad_ is the FKS map itself, cut_
    reproduces the production values.
    Entries of counter-events that genps_fks.f did not generate keep the
    values it leaves behind (jac_cnt <= 0 marks them).
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
MG_ROOT = os.path.realpath(os.path.join(HERE, "..", "..", ".."))
OUT_DIR = os.path.join(HERE, "..", "test_data", "fks")

# name -> MadGraph process. Together they cover massless and massive
# final-state emitters and initial-state emitters, with and without PDFs,
# and massless final-state emitters (gluon and quark) in a hadronic process
# with spectators of different kinds.
# The subprocess list drops beam-mirrored copies (None keeps all).
PROCESSES = {
    "epem_uuxg": ("e+ e- > u u~ g [QCD]", None),
    "epem_ttx": ("e+ e- > t t~ [QCD]", None),
    "pp_epem": ("p p > e+ e- [QCD]", ["P0_uux_emep"]),
    "pp_ttx": ("p p > t t~ [QCD]", ["P0_gg_ttx", "P0_uux_ttx"]),
    "pp_epemj": ("p p > e+ e- j [QCD]", ["P0_uux_emepg", "P0_ug_uemep"]),
}


def run(cmd, cwd, log, stdin=None):
    with open(log, "w") as f:
        subprocess.run(
            cmd, cwd=cwd, stdin=stdin, stdout=f, stderr=subprocess.STDOUT, check=True
        )


def generate(name, process, work):
    out = os.path.join(work, name)
    if not os.path.isdir(os.path.join(out, "SubProcesses")):
        script = os.path.join(work, f"{name}.mg")
        with open(script, "w") as f:
            f.write("set automatic_html_opening False\n")
            f.write(f"generate {process}\n")
            f.write(f"output standalone_fortran {out} --fks --limits\n")
        run(
            ["python3", os.path.join(MG_ROOT, "bin", "madgraph"), script],
            work,
            os.path.join(work, f"{name}_mg.log"),
        )
    run(
        ["make", "-j8"],
        os.path.join(out, "Source"),
        os.path.join(work, f"{name}_src.log"),
    )
    return out


def dump(pdir, points, log):
    for f in ["fks_kinematics_dump.f", "makefile_dump"]:
        shutil.copy(os.path.join(HERE, f), pdir)
    run(["make", "-j8", "-f", "makefile_dump", "fks_kinematics_dump"], pdir, log)
    result = subprocess.run(
        ["./fks_kinematics_dump"],
        cwd=pdir,
        input=f"{points}\n1\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def floats(fields):
    return [float(v) for v in fields]


def momenta(values):
    return [values[i : i + 4] for i in range(0, len(values), 4)]


RERUN_KEYS = ("JAC", "PEV", "PCNT", "XI", "XBJ", "SHAT")


def parse_rerun(point, prefix, key, rest):
    """The entries of the reruns of generate_FKS_kinematics, rad_ without and
    cut_ with the lower bound tau_lower_bound."""
    if key == "JAC":
        v = floats(rest)
        point[f"{prefix}_jac"], point[f"{prefix}_jac_cnt"] = v[0], v[1:]
    elif key == "PEV":
        point[f"{prefix}_p_ev"] = momenta(floats(rest))
    elif key == "PCNT":
        point.setdefault(f"{prefix}_p_cnt", []).append(momenta(floats(rest[1:])))
    elif key == "XI":
        v = floats(rest)
        point.update(
            {
                f"{prefix}_xi_ev": v[0],
                f"{prefix}_xi_hat_ev": v[1],
                f"{prefix}_y_ev": v[2],
                f"{prefix}_xiimax_ev": v[3],
                f"{prefix}_xi_cnt": v[4:9],
                f"{prefix}_xi_hat_cnt": v[9:14],
                f"{prefix}_xiimax_cnt": v[14:19],
            }
        )
    elif key == "XBJ":
        v = floats(rest)
        point[f"{prefix}_xbjrk_ev"] = v[:2]
        point[f"{prefix}_xbjrk_cnt"] = [v[2 + 2 * k : 4 + 2 * k] for k in range(5)]
    elif key == "SHAT":
        v = floats(rest)
        point[f"{prefix}_shat_ev"], point[f"{prefix}_shat_cnt"] = v[0], v[1:6]


def parse(text):
    data = {"configs": []}
    conf = point = None
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        key, rest = fields[0], fields[1:]
        if key == "HEADER":
            nexternal, nincoming, _, ndim, lpp1, lpp2, _ = map(int, rest)
            data.update(nexternal=nexternal, nincoming=nincoming, ndim=ndim)
            data["lpp"] = [lpp1, lpp2]
        elif key == "EBEAM":
            data["ebeam"] = floats(rest)
        elif key == "CONF":
            ints = list(map(int, rest))
            conf = {"nfks": ints[0], "i_fks": ints[1], "j_fks": ints[2]}
            conf["pdg"] = ints[3:]
            conf["points"] = []
            data["configs"].append(conf)
        elif key == "MASS":
            conf["masses"] = floats(rest)
        elif key == "POINT":
            point = {"isolsign": int(rest[2])}
            conf["points"].append(point)
        elif key == "X":
            point["x"] = floats(rest)
        elif key == "JAC":
            v = floats(rest)
            point["jac"], point["jac_cnt"] = v[0], v[1:]
        elif key == "PBORN":
            point["p_born"] = momenta(floats(rest))
        elif key == "PEV":
            point["p_ev"] = momenta(floats(rest))
        elif key == "PCNT":
            point.setdefault("p_cnt", []).append(momenta(floats(rest[1:])))
        elif key == "XI":
            v = floats(rest)
            point.update(xi_ev=v[0], xi_hat_ev=v[1], y_ev=v[2], xiimax_ev=v[3])
            point.update(xi_cnt=v[4:9], xi_hat_cnt=v[9:14], xiimax_cnt=v[14:19])
        elif key == "XBJ":
            v = floats(rest)
            point["xbjrk_ev"] = v[:2]
            point["xbjrk_cnt"] = [v[2 + 2 * k : 4 + 2 * k] for k in range(5)]
        elif key == "SHAT":
            v = floats(rest)
            point.update(shat_ev=v[0], shat_cnt=v[1:6], tau_ev=v[6], ycm_ev=v[7])
            point.update(tau_cnt=v[8:13], ycm_cnt=v[13:18])
        elif key == "KPERP":
            point["kperp"] = floats(rest)
        elif key == "BORNX":
            v = floats(rest)
            point.update(shat_born=v[0], tau_born=v[1], ycm_born=v[2])
            point["xbjrk_born"] = v[3:5]
        elif key == "TAULB":
            point["tau_lower_bound"] = float(rest[0])
        elif key[:3] in ("RAD", "CUT") and key[3:] in RERUN_KEYS:
            parse_rerun(point, key[:3].lower(), key[3:], rest)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=8, help="points per FKS config")
    parser.add_argument("--keep", help="keep the generated outputs in this directory")
    args = parser.parse_args()

    # identifies the Fortran the data come from without changing with every
    # commit of the repository
    genps_fks_sha = subprocess.run(
        ["git", "hash-object", "Template/NLO/SubProcesses/genps_fks.f"],
        cwd=MG_ROOT,
        capture_output=True,
        text=True,
    ).stdout.strip()
    os.makedirs(OUT_DIR, exist_ok=True)
    work = args.keep or tempfile.mkdtemp(prefix="fks_reference_")
    os.makedirs(work, exist_ok=True)
    for name, (process, keep) in PROCESSES.items():
        out = generate(name, process, work)
        for pdir in sorted(glob.glob(os.path.join(out, "SubProcesses", "P*"))):
            sub = os.path.basename(pdir)
            if keep is not None and sub not in keep:
                continue
            data = parse(dump(pdir, args.points, os.path.join(work, f"{sub}.log")))
            data.update(process=process, subprocess=sub, genps_fks_sha=genps_fks_sha)
            path = os.path.join(OUT_DIR, f"{name}_{sub}.json")
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.write("\n")
            npts = sum(len(c["points"]) for c in data["configs"])
            print(f"{path}: {len(data['configs'])} FKS configs, {npts} points")
    if not args.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
