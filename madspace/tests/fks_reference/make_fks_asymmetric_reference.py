"""Reference values of the asymmetric FKS emission off an incoming parton
(FKSISRMapping 2) of the MCcntRefactor_Sfun branch of MadGraph7.

For the processes of PROCESSES below, this script generates the
'output standalone_fortran --fks --limits' directories with a checkout of that
branch (--mg-root), builds fks_kinematics_dump_modules.f in their P*
directories, the dump of make_fks_reference.py ported to the fks_phase_space
modules of the branch, and runs it with FKSISRMapping = 2. It keeps the FKS
configurations with an incoming emitter and the kinematics of their points,
those entries of make_fks_reference.py that genps_fks.f sets, and writes them
to madspace/tests/test_data/fks_asymmetric/<name>_<subprocess>.json, with the
fks_configurations of test_data/fks (the exporter of the branch does not write
them, the configurations are the same) and

    initial_map ("asymmetric"), mg_branch and mg_commit (the commit of the
    checkout), jac_sampling_factors (true: the Jacobians of the branch include
    the factors 1 - stiny and 1 - ctiny of the sampling for an incoming
    emitter, which genps_fks.f of main leaves out, but not without the
    projection of the event of lepton PDFs)

The points are those of make_fks_reference.py at the same random numbers, so
their Borns and counter-events are those of test_data/fks. The branch also
generates the soft counter-events of a Born below the lower bound
tau_lower_bound on the partonic energy, which genps_fks.f of main does not
(jac_cnt = -34 there). They are kept.

Run from anywhere with a gfortran in the PATH:

    python madspace/tests/fks_reference/make_fks_asymmetric_reference.py
        --mg-root PATH [--keep DIR]
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(HERE, "..", "test_data", "fks_asymmetric")
sys.path.insert(0, HERE)
import make_fks_reference as mfr  # noqa: E402

# names of make_fks_reference.PROCESSES, with the subprocesses to keep: a
# massless and a massive final state of a 2 -> 2 Born, a 2 -> 3 Born with
# quark and gluon emitters, photon emitters, and lepton PDFs, whose incoming
# emitters radiate without the projection of the event
PROCESSES = {
    "pp_epem": ["P0_uux_emep"],
    "pp_ttx": ["P0_gg_ttx", "P0_uux_ttx"],
    "pp_epemj": ["P0_uux_emepg", "P0_ug_uemep"],
    "pp_epem_qed": ["P0_aa_emep"],
    "epem_mumu_qed": ["P0_epem_mummup"],
    "epem_uuxg_mix": ["P0_epem_uuxg"],
}
# FKSISRMapping of the branch
ISR_MAPPING = 2
# the entries of a point that are not kinematics
NOT_KINEMATICS = {
    "s_ij_ev",
    "s_ij_cnt",
    "eik_pairs",
    "eik_soft",
    "coll_cnt",
    "prefactors",
    "fks_symmetry_factor",
    "fks_symmetry_factor_deg",
    "sreal",
    "sreal_orders",
    "born",
    "born_orders",
    "extra_born_orders",
    "color_links",
    "color_links_orders",
    "collinear_remnants",
    "collinear_remnants_orders",
    "scheme_remnants_orders",
    "soft_virtual",
    "soft_virtual_orders",
}
TOP_LEVEL = ["nexternal", "nincoming", "ndim", "lpp", "ebeam"]


def dump(pdir, points, massive, limit, log, lpp):
    shutil.copy(
        os.path.join(HERE, "fks_kinematics_dump_modules.f"),
        os.path.join(pdir, "fks_kinematics_dump.f"),
    )
    shutil.copy(
        os.path.join(HERE, "makefile_dump_modules"), os.path.join(pdir, "makefile_dump")
    )
    mfr.run(["make", "-j8", "-f", "makefile_dump", "fks_kinematics_dump"], pdir, log)
    result = subprocess.run(
        ["./fks_kinematics_dump"],
        cwd=pdir,
        input=f"{points}\n1\n{massive}\n{limit}\n{lpp}\n{ISR_MAPPING}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def kinematics(point):
    return {k: v for k, v in point.items() if k not in NOT_KINEMATICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mg-root", required=True, help="checkout of the MCcntRefactor_Sfun branch"
    )
    parser.add_argument("--points", type=int, default=4, help="points per FKS config")
    parser.add_argument("--massive", type=int, default=8)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--keep", help="keep the generated outputs in this directory")
    args = parser.parse_args()

    mfr.MG_ROOT = os.path.realpath(args.mg_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mfr.MG_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    os.makedirs(OUT_DIR, exist_ok=True)
    work = args.keep or tempfile.mkdtemp(prefix="fks_asymmetric_reference_")
    os.makedirs(work, exist_ok=True)
    for name, keep in PROCESSES.items():
        process = mfr.PROCESSES[name][0]
        out = mfr.generate(
            name, process, work, mfr.RUN_CARD.get(name), mfr.MODELS.get(name)
        )
        if name in mfr.FKS_POWERS:
            mfr.set_fks_powers(out, mfr.FKS_POWERS[name])
        for pdir in sorted(glob.glob(os.path.join(out, "SubProcesses", "P*"))):
            sub = os.path.basename(pdir)
            if sub not in keep:
                continue
            log = os.path.join(work, f"{name}_{sub}_asymmetric.log")
            lpp = mfr.LEPTON_PDFS.get(name, 0)
            parsed = mfr.parse(
                dump(pdir, args.points, args.massive, args.limit, log, lpp)
            )
            data = {key: parsed[key] for key in TOP_LEVEL}
            data.update(
                process=process,
                subprocess=sub,
                initial_map="asymmetric",
                mg_branch="MCcntRefactor_Sfun",
                mg_commit=commit,
                jac_sampling_factors=True,
            )
            # the exporter of the branch does not write fks_configurations.json:
            # that of the reference data of main, for the same configurations
            with open(os.path.join(mfr.OUT_DIR, f"{name}_{sub}.json")) as f:
                main = json.load(f)
            data["fks_configurations"] = main["fks_configurations"]
            configs = []
            for conf, conf_main in zip(parsed["configs"], main["configs"], strict=True):
                for key in ["nfks", "i_fks", "j_fks", "pdg", "masses"]:
                    if conf[key] != conf_main[key]:
                        raise ValueError(f"{name} {sub}: {key} differs from main")
                if conf["j_fks"] > parsed["nincoming"]:
                    continue
                conf = dict(conf)
                conf["points"] = [kinematics(p) for p in conf["points"]]
                conf["limit_points"] = [
                    kinematics(p) for p in conf.get("limit_points", [])
                ]
                configs.append(conf)
            data["configs"] = configs
            path = os.path.join(OUT_DIR, f"{name}_{sub}.json")
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.write("\n")
            npts = sum(len(c["points"]) + len(c["limit_points"]) for c in configs)
            print(f"{path}: {len(configs)} FKS configs, {npts} points")
    if not args.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
