"""Reference cross sections of MG5aMC at fixed order, LO and NLO, for the
cross sections of the MG7 NLO driver (madgraph/mg7/nlo/fks.py) with the
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
    rounds the total to four digits), and with reweight_scale at NLO the
    cross section at each pair of factors of mu_R and mu_F
    (scale_variations, mu_r, mu_f and sigma, from the total rate of the
    histograms of the run).

    A process with --ewsudakov has at NLO also the Sudakov approximation of
    the EW corrections (ew_sudakov, per sud_mod its sigma and error), from
    the histograms of analysis_HwU_ewsudakov.f, which the run takes as its
    FO analysis. Its NLO cross section is that of MadFKS, which sums them
    with the NLO corrections.

The outputs of BUILT have no cross section here. The script only compiles
their executables with the cards of a fixed-order run at NLO, as the tests of
the provider need them.

Run from anywhere with a gfortran in the PATH:

    python madspace/tests/fks_reference/make_fks_cross_section_reference.py
        --keep DIR [--only NAME] [--accuracy ACC] [--models DIR]

The processes with --ewsudakov need the model loop_qcd_qed_sm_Gmu_forSudakov
of the UFO model database, which MadGraph downloads, or which --models takes
from a directory.
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys

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
# the lepton densities of MadFKS for e+ on the first beam and e- on the second
DRESSED = dict(lpp1=-3, lpp2=3, pdlabel="isronlyll")
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
    # dressed electrons and positrons, ISR alone at the Z pole (at the fixed
    # scale of mur_ref_fixed) and at 500 GeV, and with the beamstrahlung of
    # the ILC
    "epem_jj_isr": (
        "epem_jj",
        dict(DRESSED, ebeam1=45.6, ebeam2=45.6, dynamical_scale_choice=-2),
    ),
    "epem_ttx_isr": ("epem_ttx", DRESSED),
    "epem_ttx_ilc": ("epem_ttx", dict(DRESSED, pdlabel="ilc500ll")),
    # the resonance-local recoil of the u u~ of a Z
    "epem_epemuux": ("epem_epemuux", {}),
    # isolated photons with QCD corrections, and the QED corrections of a
    # lepton pair with the recombination of photons
    "pp_aa": ("pp_aa", {}),
    "pp_epem_qed": ("pp_epem_qed", {}),
    # tagged photons with QED corrections, whose couplings are in the
    # alpha(0) scheme and the others in the G_mu scheme
    "pp_aa_tagged": ("pp_aa_tagged", {}),
    # elastic photons from protons (epa_proton of PhotonFlux.f)
    "aa_ttx": ("aa_ttx", {}),
    # the scale variations of the reweighting of MadFKS
    "pp_epem_rw": ("pp_epem", dict(mfir.PROCESSES["pp_epem"][1], reweight_scale=True)),
    # a Born of one final-state particle, a W at rest in the partonic frame
    "pp_wp": ("pp_wp", {}),
    # a gluon of a decay that splits into g g and q q~ against the top, with
    # a massive and a massless b
    "t_bwg": ("t_bwg", {}),
    "t_bwg_mb0": ("t_bwg_mb0", {}),
    # the Sudakov approximation of the EW corrections at a 3 TeV collider
    "epem_ttx_sdk": ("epem_ttx_sdk", {}),
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
    "epem_ttx": ("e+ e- > t t~ [QCD]", dict(ebeam1=250, ebeam2=250), None),
    "epem_epemuux": (
        "e+ e- > e+ e- u u~ [QCD]",
        dict(ptj=20, ptl=20, mll_sf=30),
        None,
    ),
    "pp_aa": ("p p > a a [QCD]", dict(mfir.HADRONIC, ptgmin=20), None),
    "pp_epem_qed": (
        "p p > e+ e- [QED]",
        dict(mfir.HADRONIC, mll_sf=30, ptl=10),
        None,
    ),
    # mixed QCD and EW corrections with an extra counterterm (a u > u a, for
    # a quark pair from a gluon and from a photon), in BUILT
    "gu_ua_mix": (
        "g u > u a QED^2<=4 QCD^2<=2 [QCD QED]",
        dict(mfir.HADRONIC, ptj=20, jetalgo=-1, jetradius=0.4),
        "loop_qcd_qed_sm",
    ),
    # loop_qcd_qed_sm_Gmu of the model database of MadGraph, which import
    # model downloads
    "pp_aa_tagged": (
        "p p > !a! !a! [QED]",
        dict(mfir.HADRONIC, ptgmin=20),
        "loop_qcd_qed_sm_Gmu-a0",
    ),
    "aa_ttx": ("a a > t t~ [QCD]", dict(mfir.HADRONIC, lpp1=2, lpp2=2), None),
    "pp_wp": ("p p > w+ [QCD]", dict(mfir.HADRONIC), None),
    # the jet of the gluon with its cuts in the rest frame of the top
    "t_bwg": ("t > b w+ g QED=1 [QCD]", dict(ptj=10, jetalgo=-1, jetradius=0.4), None),
    "t_bwg_mb0": (
        "t > b w+ g QED=1 [QCD]",
        dict(ptj=10, jetalgo=-1, jetradius=0.4),
        "loop_sm-no_b_mass",
    ),
    # the forSudakov models of the model database of MadGraph
    "epem_ttx_sdk": (
        "e+ e- > t t~ [QCD] --ewsudakov",
        dict(ebeam1=1500, ebeam2=1500),
        "loop_qcd_qed_sm_Gmu_forSudakov",
    ),
}
# the FO analysis of the outputs with --ewsudakov, for the totals of the
# Sudakov approximation per sud_mod
EWSUDAKOV_ANALYSIS = "analysis_HwU_ewsudakov"
# the outputs in the complex-mass scheme: the EW corrections of a process
# with an s-channel Z are gauge invariant only there, as MadGraph warns
# (arXiv:1804.10017), and close to the Z the single poles of MadLoop and
# MadFKS differ otherwise, which stops the run
COMPLEX_MASS = {"pp_epem_qed"}
# the outputs of OUTPUTS without a cross section, whose executables the
# tests of the provider take. The untagged photon of g u > u a splits into
# lepton pairs at NLO, which no setting of MadFKS turns back into a photon or
# a jet, so that the real emission with a collinear pair and its
# counter-event see different cuts and the cross section diverges. With the
# recombination of photons, cuts.f stops on two photons to recombine
# instead. Its limit tests fail for the sum of all orders while every order
# passes them (the orders of the collinear d d~ of g u > u d d~ cancel
# there), which stops compile FO after the executables.
BUILT = {"gu_ua_mix"}
# the relative accuracy of the runs (req_acc_FO), per setup where it differs
ACCURACY = 2e-4
ACCURACIES = {
    "pp_ttx": 1e-3,
    "epem_jjj": 1e-3,
    "pp_epemj": 2e-3,
    "epem_epemuux": 1e-3,
    "pp_aa": 1e-3,
    "pp_epem_qed": 1e-3,
    "pp_aa_tagged": 1e-3,
    # the Sudakov approximation on every point makes MadFKS slow, and it is
    # -30% of the LO (sud_mod 0) at 3 TeV
    "epem_ttx_sdk": 1e-3,
    "t_bwg": 1e-3,
    "t_bwg_mb0": 1e-3,
}
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
    env = {k: v for k, v in os.environ.items() if k != "MADFKS_INTEGRAND_DUMP"}
    log = os.path.join(work, f"{run_name}_launch.log")
    summary = os.path.join(out, "Events", run_name, "summary.txt")
    with open(script, "w") as f:
        f.write(f"launch {out} -n {run_name}\n")
        f.write(f"fixed_order=ON\norder={order}\ndone\n")
        for key, value in {**RUN_CARD, **values, "req_acc_FO": accuracy}.items():
            f.write(f"set {key} {value}\n")
        f.write("done\n")
    mfir.mfr.madgraph(script, work, log, env=env, expect=summary)
    return summary


def build(out, values, work, name):
    """Compiles the executables of the output out with the cards of a
    fixed-order run at NLO (compile FO of aMC@NLO) without running them. The
    tests of MadFKS that follow may fail."""
    script = os.path.join(work, f"{name}_build.mg")
    with open(script, "w") as f:
        f.write("compile FO\n")
        for key, value in {**RUN_CARD, **values}.items():
            f.write(f"set {key} {value}\n")
        f.write("done\n")
    executables = glob.glob(os.path.join(out, "SubProcesses", "P*", "madevent_mintFO"))
    for stale in executables:
        os.remove(stale)
    log = os.path.join(work, f"{name}_build.log")
    with open(log, "w") as f:
        command = [sys.executable, os.path.join(out, "bin", "aMCatNLO"), script]
        subprocess.run(
            command, cwd=work, stdout=f, stderr=subprocess.STDOUT, check=True
        )
    for directory in glob.glob(os.path.join(out, "SubProcesses", "P*")):
        if not os.path.isfile(os.path.join(directory, "madevent_mintFO")):
            raise RuntimeError(f"{script} built no madevent_mintFO in {directory}")


def install_ewsudakov_analysis(out):
    """analysis_HwU_ewsudakov.f in the FixedOrderAnalysis of the output out
    and in its FO_analyse_card.dat, as the analysis of its runs."""
    target = os.path.join(out, "FixedOrderAnalysis", f"{EWSUDAKOV_ANALYSIS}.f")
    with open(os.path.join(HERE, f"{EWSUDAKOV_ANALYSIS}.f")) as f:
        source = f.read()
    with open(target, "w") as f:
        f.write(source)
    card = os.path.join(out, "Cards", "FO_analyse_card.dat")
    with open(card) as f:
        text = f.read()
    text = re.sub(
        r"^FO_ANALYSE *=.*$",
        f"FO_ANALYSE = {EWSUDAKOV_ANALYSIS}.o",
        text,
        count=1,
        flags=re.M,
    )
    with open(card, "w") as f:
        f.write(text)


def parse_histogram_total(path, title):
    """The value and the error of the first bin of the histogram title of
    the HwU histograms at path (of one bin of width one, its integral)."""
    with open(path) as f:
        lines = f.read().splitlines()
    names = [n.strip() for n in lines[0].lstrip("#&").split("&")]
    for k, line in enumerate(lines):
        if line.startswith("<histogram>") and f'"{title}' in line:
            values = [float(v) for v in lines[k + 1].split()]
            return dict(
                zip(
                    ("sigma", "error"),
                    (values[names.index("central value")], values[names.index("dy")]),
                )
            )
    raise ValueError(f"no histogram {title} in {path}")


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


def parse_scale_variations(path):
    """The cross section at each pair of factors of mu_R and mu_F of the
    first dynamical scale choice, the first bin of the total rate of the
    HwU histograms of the run at path, whose columns are the weights."""
    with open(path) as f:
        lines = f.read().splitlines()
    # the header "##& xmin & xmax & central value & ..." names the columns
    names = [n.strip() for n in lines[0].lstrip("#&").split("&")]
    rate = next(k for k, line in enumerate(lines) if "total rate" in line)
    values = [float(v) for v in lines[rate + 1].split()]
    if len(values) != len(names):
        raise ValueError(f"the columns of {path} do not match its header")
    variations, choice = [], None
    for name, value in zip(names, values):
        found = re.search(r"dyn=\s*(-?\d+)\s+muR=\s*(\S+)\s+muF=\s*(\S+)", name)
        if found is None:
            continue
        dyn, ren, fact = int(found[1]), float(found[2]), float(found[3])
        choice = dyn if choice is None else choice
        if dyn == choice:
            variations.append(dict(mu_r=ren, mu_f=fact, sigma=value))
    if not variations:
        raise ValueError(f"no scale variations in {path}")
    return variations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep", required=True, help="directory of the outputs")
    parser.add_argument("--only", help="only this setup")
    parser.add_argument("--accuracy", type=float, help="req_acc_FO of every run")
    parser.add_argument(
        "--reuse", action="store_true", help="collect the runs already there"
    )
    parser.add_argument("--models", help="directory with the models of the outputs")
    args = parser.parse_args()

    def model_path(model):
        if model and args.models and os.path.isdir(os.path.join(args.models, model)):
            return os.path.join(args.models, model)
        return model

    commit = mfir.mfr.commit(mfir.MG_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    work = args.keep
    os.makedirs(work, exist_ok=True)
    names = [args.only] if args.only else [*SETUPS, *BUILT]
    for name in names:
        if name in BUILT:
            process, values, model = OUTPUTS[name]
            out = mfir.generate(
                name,
                process,
                work,
                os.path.join(work, f"{name}_gen.log"),
                model_path(model),
            )
            mfir.sum_helicities(out)
            build(out, values, work, name)
            print(f"{out}: built")
            continue
        output, values = SETUPS[name]
        process, base, model = OUTPUTS.get(output) or (*mfir.PROCESSES[output], None)
        out = mfir.generate(
            output,
            process,
            work,
            os.path.join(work, f"{output}_gen.log"),
            model_path(model),
            complex_mass=output in COMPLEX_MASS,
        )
        mfir.sum_helicities(out)
        ewsudakov = "--ewsudakov" in process
        if ewsudakov:
            install_ewsudakov_analysis(out)
        accuracy = args.accuracy or ACCURACIES.get(name, ACCURACY)
        values = {**base, **values}
        results = {}
        for order in ("LO", "NLO"):
            summary = os.path.join(out, "Events", f"{name}_{order.lower()}")
            summary = os.path.join(summary, "summary.txt")
            if not (args.reuse and os.path.isfile(summary)):
                summary = launch(out, values, order, accuracy, work, name)
            results[order.lower()] = parse_summary(summary)
            histograms = os.path.join(os.path.dirname(summary), "MADatNLO.HwU")
            if order == "NLO" and values.get("reweight_scale"):
                results["nlo"]["scale_variations"] = parse_scale_variations(histograms)
            if order == "NLO" and ewsudakov:
                results["nlo"]["ew_sudakov"] = [
                    dict(
                        parse_histogram_total(histograms, f"EW Sudakov sud_mod {mode}"),
                        sud_mod=mode,
                    )
                    for mode in (0, 1)
                ]
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
