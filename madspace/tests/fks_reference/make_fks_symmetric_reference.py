"""Reference values of the symmetric FKS emission off an incoming parton
(FKSISRMapping 1) of MadGraph7, beside its default, the asymmetric map of
make_fks_reference.py.

For the processes of PROCESSES below, this script generates the
'output standalone_fortran --fks --limits' directories with MadGraph7, builds
fks_kinematics_dump.f in their P* directories and runs it with FKSISRMapping
= 1. It keeps the FKS configurations with an incoming emitter and the
kinematics of their points, those entries of make_fks_reference.py that the
phase space sets, and writes them to
madspace/tests/test_data/fks_symmetric/<name>_<subprocess>.json, with

    initial_map ("symmetric"), mg_commit (the commit of MadGraph7) and
    jac_sampling_factors (true, see make_fks_reference.py)

The points are those of make_fks_reference.py at the same random numbers, so
their Borns and counter-events are those of test_data/fks.

Run from anywhere with a gfortran in the PATH:

    python madspace/tests/fks_reference/make_fks_symmetric_reference.py
        [--keep DIR]
"""

import argparse
import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(HERE, "..", "test_data", "fks_symmetric")
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
# FKSISRMapping of the symmetric map
ISR_MAPPING = 1
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


def kinematics(point):
    return {k: v for k, v in point.items() if k not in NOT_KINEMATICS}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=4, help="points per FKS config")
    parser.add_argument("--massive", type=int, default=8)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--keep", help="keep the generated outputs in this directory")
    args = parser.parse_args()

    commit = mfr.commit(mfr.MG_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    work = args.keep or tempfile.mkdtemp(prefix="fks_symmetric_reference_")
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
            log = os.path.join(work, f"{name}_{sub}_symmetric.log")
            lpp = mfr.LEPTON_PDFS.get(name, 0)
            parsed = mfr.parse(
                mfr.dump(
                    pdir,
                    args.points,
                    args.massive,
                    args.limit,
                    log,
                    lpp,
                    isr=ISR_MAPPING,
                )
            )
            data = {key: parsed[key] for key in TOP_LEVEL}
            data.update(
                process=process,
                subprocess=sub,
                initial_map="symmetric",
                mg_commit=commit,
                jac_sampling_factors=True,
            )
            with open(os.path.join(pdir, "fks_configurations.json")) as f:
                data["fks_configurations"] = json.load(f)
            # the configurations and Borns of the points are those of the
            # default map
            with open(os.path.join(mfr.OUT_DIR, f"{name}_{sub}.json")) as f:
                main = json.load(f)
            configs = []
            for conf, conf_main in zip(parsed["configs"], main["configs"], strict=True):
                for key in ["nfks", "i_fks", "j_fks", "pdg", "masses"]:
                    if conf[key] != conf_main[key]:
                        raise ValueError(
                            f"{name} {sub}: {key} differs from test_data/fks"
                        )
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
