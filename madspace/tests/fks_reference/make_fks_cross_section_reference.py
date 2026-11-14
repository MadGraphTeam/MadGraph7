"""Reference cross sections of MG5aMC at fixed order, LO and NLO, for the
cross sections of the MG7 NLO driver (madgraph/mg7/fks_driver.py) with the
FKS integrand of madspace.

For every setup below, this script runs a fixed-order integration of an
aMC@NLO output of MadGraph7 (the outputs of make_fks_integrand_reference.py,
one per process, and those of OUTPUTS, generated here where they are
missing) to the accuracy of ACCURACY, at LO and at NLO, and collects into
madspace/tests/test_data/fks_cross_section/<name>.json:

    process, output (the name of the output), mg_commit, run_card (the values
    of RUN_CARD, those of the setup, and those the integrand depends on, as
    the run leaves them), param_card, n_f, n_l, and per order (lo, nlo) the
    cross section and its error in pb (a width in GeV for a decay) and per
    subprocess (the P* directory) the sum of its channels with their errors
    added in quadrature, from the last res_*.txt of the run (summary.txt
    rounds the total to four digits).

Run from anywhere with a gfortran in the PATH:

    python madspace/tests/fks_reference/make_fks_cross_section_reference.py
        --keep DIR [--only NAME] [--accuracy ACC]
"""

import argparse
import glob
import json
import os
import re

import make_fks_integrand_reference as mfir

HERE = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(HERE, "..", "test_data", "fks_cross_section")

# name -> (output of make_fks_integrand_reference.py or of OUTPUTS, run_card
# values). The fixed scale of e+ e- > j j without cuts gives sigma_NLO =
# sigma_LO (1 + alpha_s / pi).
FIXED = dict(
    fixed_ren_scale=True,
    fixed_fac_scale=True,
    mur_ref_fixed=1000.0,
    muf_ref_fixed=1000.0,
    ptj=0,
)
SETUPS = {
    "epem_jj_fixed": ("epem_jj", FIXED),
    "epem_jj": ("epem_jj", {}),
    "epem_jj_antikt": ("epem_jj", mfir.PROCESSES["epem_jj_antikt"][1]),
    "pp_epem": ("pp_epem", mfir.PROCESSES["pp_epem"][1]),
    "pp_ttx": ("pp_ttx", mfir.PROCESSES["pp_ttx"][1]),
    "t_bw": ("t_bw", {}),
    "epem_jjj": ("epem_jjj", {}),
    "pp_epemj": ("pp_epemj", {}),
    "t_bw_mb0": ("t_bw_mb0", {}),
}
# outputs of the cross sections alone, without an integrand reference: name
# -> (process, run_card values, model). p p > e+ e- j has a jet that radiates
# against the incoming partons and incoming partons that radiate against it,
# t > b w+ with a massless b the collinear limits of a decay.
OUTPUTS = {
    "pp_epemj": (
        "p p > e+ e- j [QCD]",
        dict(mfir.HADRONIC, mll_sf=30, ptj=30, jetalgo=-1, jetradius=0.4),
        None,
    ),
    "t_bw_mb0": ("t > b w+ QED=1 [QCD]", {}, "loop_sm-no_b_mass"),
}
# the relative accuracy of the runs (req_acc_FO), per setup where it differs
ACCURACY = 2e-4
ACCURACIES = {"pp_ttx": 1e-3, "epem_jjj": 1e-3, "pp_epemj": 2e-3}
# the values of every run, the defaults of the run_card where a setup of
# the same output changes them
RUN_CARD = {
    "iseed": 4711,
    "reweight_scale": False,
    "reweight_PDF": False,
    "fixed_ren_scale": False,
    "fixed_fac_scale": False,
    "mur_ref_fixed": 91.118,
    "muf_ref_fixed": 91.118,
    "jetalgo": 1,
    "jetradius": 0.7,
    "ptj": 10,
    "etaj": -1,
}


def launch(out, values, order, accuracy, work, name):
    run_name = f"{name}_{order.lower()}"
    script = os.path.join(work, f"{run_name}_launch.mg")
    with open(script, "w") as f:
        f.write(f"launch {out} -n {run_name}\nfixed_order=ON\norder={order}\ndone\n")
        for key, value in {**RUN_CARD, **values, "req_acc_FO": accuracy}.items():
            f.write(f"set {key} {value}\n")
        f.write("done\n")
    env = {k: v for k, v in os.environ.items() if k != "MADFKS_INTEGRAND_DUMP"}
    log = os.path.join(work, f"{run_name}_launch.log")
    summary = os.path.join(out, "Events", run_name, "summary.txt")
    mfir.mfr.madgraph(script, work, log, env=env, expect=summary)
    return summary


def parse_summary(path):
    """The total cross section and that of each subprocess, with errors."""
    with open(path) as f:
        text = f.read()
    # a decay has its width there
    total = re.search(r"(?:cross section|decay width):\s+(\S+)\s+\+-\s+(\S+)", text)
    if total is None:
        raise ValueError(f"no total cross section in {path}")
    results = glob.glob(os.path.join(os.path.dirname(path), "res_*.txt"))
    last = max(results, key=lambda p: int(re.search(r"res_(\d+)", p).group(1)))
    with open(last) as f:
        text = f.read()
    block = text.split("Cross section per integration channel:")[1]
    block = block.split("ABS cross section")[0]
    subprocesses = {}
    for line in block.splitlines():
        fields = line.split()
        if len(fields) >= 4 and fields[0].startswith("P"):
            value, error = float(fields[2]), float(fields[3])
            entry = subprocesses.setdefault(fields[0], [0.0, 0.0])
            entry[0] += value
            entry[1] += error**2
    sigma = sum(v for v, _ in subprocesses.values())
    error = sum(e for _, e in subprocesses.values()) ** 0.5
    if abs(sigma - float(total.group(1))) > 1e-3 * abs(sigma) + float(total.group(2)):
        raise ValueError(f"the channels of {last} do not add up to the total")
    return dict(
        sigma=sigma,
        error=error,
        subprocesses={
            name: dict(sigma=v, error=e**0.5) for name, (v, e) in subprocesses.items()
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", required=True, help="directory of the outputs")
    parser.add_argument("--only", help="only this setup")
    parser.add_argument("--accuracy", type=float, help="req_acc_FO of every run")
    parser.add_argument(
        "--reuse", action="store_true", help="collect the runs already there"
    )
    args = parser.parse_args()

    commit = mfir.mfr.commit(mfir.MG_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    work = args.keep
    os.makedirs(work, exist_ok=True)
    names = [args.only] if args.only else list(SETUPS)
    for name in names:
        output, values = SETUPS[name]
        process, base, model = OUTPUTS.get(output) or (*mfir.PROCESSES[output], None)
        out = mfir.generate(
            output, process, work, os.path.join(work, f"{output}_gen.log"), model
        )
        mfir.sum_helicities(out)
        accuracy = args.accuracy or ACCURACIES.get(name, ACCURACY)
        values = {**base, **values}
        results = {}
        for order in ("LO", "NLO"):
            summary = os.path.join(out, "Events", f"{name}_{order.lower()}")
            summary = os.path.join(summary, "summary.txt")
            if not (args.reuse and os.path.isfile(summary)):
                summary = launch(out, values, order, accuracy, work, name)
            results[order.lower()] = parse_summary(summary)
        run_card, param_card = mfir.read_cards(out)
        data = dict(
            process=process,
            output=output,
            mg_commit=commit,
            accuracy=accuracy,
            run_card=run_card,
            param_card=param_card,
            **mfir.light_flavors(out),
            **results,
        )
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, indent=1)
            f.write("\n")
        nlo = results["nlo"]
        print(f"{path}: NLO {nlo['sigma']} +- {nlo['error']}")


if __name__ == "__main__":
    main()
