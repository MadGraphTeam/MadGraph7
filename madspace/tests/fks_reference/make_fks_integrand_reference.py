"""Reference values of the fixed-order NLO integrand of MadFKS (sigint of
driver_mintFO.f), point by point.

For every process below, this script generates an aMC@NLO output with
MadGraph7, appends fks_integrand_dump.f to SubProcesses/driver_mintFO.f and
calls it from sigint right after include_PDF_and_alphas. A short
fixed-order run (one iteration of grid setup and one of integration, with
MADFKS_INTEGRAND_DUMP points per channel) then writes the weights of the
first points of every channel as MadFKS sums them, and this script collects
them into madspace/tests/test_data/fks_integrand/<name>.json:

    process, mg_commit, run_card (the values of RUN_CARD below and those the
    integrand depends on), param_card (masses, widths, couplings of the
    run), n_f, n_l (light quarks and massless leptons of the model),
    channels[]: subprocess, channel (the G directory), fks_configurations
    (fks_configurations.json of the subprocess), points[]:
        call (the number of the call of sigint), ichan, iconfig, nfks_picked
        (the FKS configuration of the n+1-body part), iran_picked, nfks_born
        (that of the n-body part), real_type (1 for the hard-real channel),
        ndim, sum (all configurations summed instead of picked), vegas_wgt,
        vol (the weight of the pick of the configuration), xx, x (the random
        numbers before and after update_vegas_x), born (valid, xjac, xpswgt,
        shat, tau, ycm, xbjrk, stot, weight of the phase-space point, and
        tau_born_lower_bound, tau_lower_bound_resonance, tau_lower_bound),
        resonance (resonance_recoil, mass2, momentum of the resonance of the
        resonance-local recoil of the event),
        p_born, event and counter_events[k] for k = 0, 1, 2 (valid,
        jacobian, xi, y, xi_hat, xi_max, xbjrk), radiation (the three random
        numbers of the radiation), prefactors (f_b, f_nb, f_r, f_s, f_c,
        f_dc, f_sc, f_dsc, f_pdf_scheme as sigint leaves them, with the
        multichannel factor where include_multichannel_enhance ran),
        cuts (where the Born is valid: passcuts of sigint at the soft
        counter-event, at the collinear one, that of the soft one with event
        projection, and at the event), symmetry_factors (fkssymmetryfactor,
        its Born and Deg versions),
        xi_norm (xinorm_ev, xinorm_cnt[k], xiimax_ev, xiimax_cnt[k] for
        k = 0, 1, 2), contributions[]:
            index, type (add_wgt: 1 real, 2 Born, 3 integrated counterterms,
            4 soft, 5 collinear, 6 soft-collinear, 14 virtual, 15 average
            virtual of the virtual trick, 20 + sud_mod EW Sudakov), nfks,
            qcd_power, h_event, niproc, orders_tag, amp_pos, wgt
            (the coefficients of 1, log(mu_R^2/Q^2), log(mu_F^2/Q^2)), bjx,
            scales2 (Q^2, mu_R^2, mu_F^2), g_strong, y_bst, cpower, weight
            (wgts(1), PDFs and couplings included, in pb), wgt_me_tree,
            momenta, pdg, parton_iproc (the luminosity per subprocess times
            the weight without PDFs), parton_pdg (the PDG codes of each
            subprocess), parton_pdg_uborn (those of its underlying Born,
            set_pdg_codes)
        g (the coupling at the end of the call), blocks (the matrix-element
        blocks of the point, where the Born is valid, each with the coupling
        it took): nfks, i_fks, j_fks (the picked configuration), g_real,
        real, real_orders (at the event, per split-order position), k_perp
        (xij_kperp of the event), azimuthal_phase (real and imaginary
        part of xij_aor, the phase of azimuthal_orders that sborncol_isr
        and sborncol_fsr take at y = 1), g_born, born, born_orders,
        azimuthal_orders (real and imaginary part, at the position of the
        counterterm, sborn), channel (config, diagram, symmetry_factor,
        enhance and amp2 of the Born diagrams of the multichannel factor
        of include_multichannel_enhance), nbody (nfks, i_fks, j_fks of the
        configuration of the n-body part, links[]: m, n (legs of the real
        process), g, value, orders of sborn_sf for the pairs sbornsoft
        takes, channel there, virtual where bornsoftvirtual evaluated it:
        qes2 (the Ellis-Sexton scale squared as sigint leaves it, that of
        its last kinematics, the virtual is at that of the n-body part,
        scales2 of its weight lines), fraction (virtual_fraction of the
        channel), orders (the finite part of MadLoop per split-order
        position at the Born, summed over helicities, with the coupling of
        the n-body contributions))
    orders[] (per channel): position, qcd_pos, qed_pos, orders of the
    split-order positions (amp_split_pos_to_orders)

Run from anywhere with a gfortran in the PATH:

    python madspace/tests/fks_reference/make_fks_integrand_reference.py
        [--points N] [--keep DIR] [--only NAME]
"""

import argparse
import glob
import json
import os
import re
import shutil
import tempfile

import make_fks_reference as mfr

HERE = os.path.dirname(os.path.realpath(__file__))
MG_ROOT = mfr.MG_ROOT
OUT_DIR = os.path.join(HERE, "..", "test_data", "fks_integrand")

# name -> (process, run_card values). The validation processes of the FKS
# integrand: final-state emitters without PDFs, initial-state emitters with
# PDFs, and both with massive final states.
HADRONIC = {
    "lpp1": 1,
    "lpp2": 1,
    "ebeam1": 6500,
    "ebeam2": 6500,
    "pdlabel": "nn23nlo",
}
PROCESSES = {
    "epem_jj": ("e+ e- > j j [QCD]", {}),
    "pp_epem": ("p p > e+ e- [QCD]", dict(HADRONIC, mll_sf=30)),
    "pp_ttx": ("p p > t t~ [QCD]", HADRONIC),
    # the cuts of cuts.f: jets with the kT and anti-kT algorithms, leptons
    "epem_jj_kt": (
        "e+ e- > j j [QCD]",
        dict(ptj=150, etaj=0.8, jetalgo=1, jetradius=1.0),
    ),
    "epem_jj_antikt": (
        "e+ e- > j j [QCD]",
        dict(ptj=150, etaj=0.8, jetalgo=-1, jetradius=0.4),
    ),
    "pp_epem_cuts": (
        "p p > e+ e- [QCD]",
        dict(HADRONIC, mll_sf=30, ptl=20, etal=2.5, drll=0.4),
    ),
    # mu_R = 2 H_T/2 and mu_F = H_T/4, with the Ellis-Sexton scale at mu_R
    "pp_epem_scales": (
        "p p > e+ e- [QCD]",
        dict(HADRONIC, mll_sf=30, mur_over_ref=2.0, muf_over_ref=0.5),
    ),
    # a decay: a massive incoming particle at rest, a final-state emitter
    "t_bw": ("t > b w+ QED=1 [QCD]", {}),
    # a final-state gluon in the Born: its splittings into g g and q q~
    "epem_jjj": ("e+ e- > j j j [QCD]", dict(ptj=50, jetalgo=-1, jetradius=0.4)),
}
# more points where the jet cuts fail rarely at the event alone
POINTS = {"epem_jj_kt": 100, "epem_jj_antikt": 100}
# a short run: the points come from its first iteration
RUN_CARD = {
    "req_acc_FO": -1,
    "npoints_FO_grid": 100,
    "niters_FO_grid": 1,
    "npoints_FO": 100,
    "niters_FO": 1,
    "iseed": 4711,
    "reweight_scale": False,
    "reweight_PDF": False,
}
DUMP_CALL = (
    "      call fks_integrand_dump(xx,x,vegas_wgt,nFKS_picked,iran_picked\n"
    "     $     ,vol,nFKS_born,sum,real_channel_type,phase_space\n"
    "     $     ,passcuts_nbody,passcuts_coll,passcuts_n1body)\n"
)


def generate(name, process, work, log, model=None):
    out = os.path.join(work, name)
    if not os.path.isdir(os.path.join(out, "SubProcesses")):
        script = os.path.join(work, f"{name}_gen.mg")
        with open(script, "w") as f:
            f.write("set automatic_html_opening False\n")
            if model:
                f.write(f"import model {model}\n")
            f.write(f"generate {process}\n")
            f.write(f"output {out}\n")
        mfr.madgraph(script, work, log, expect=os.path.join(out, "SubProcesses"))
    return out


def patch_driver(out):
    """sigint calls fks_integrand_dump after include_PDF_and_alphas. A kept
    output gets the current call and fks_integrand_dump.f, and loses its
    executables, so that a failed compilation stops the run instead of
    running the old ones."""
    path = os.path.join(out, "SubProcesses", "driver_mintFO.f")
    with open(path) as f:
        source = f.read()
    with open(os.path.join(HERE, "fks_integrand_dump.f")) as f:
        dump = f.read()
    head = dump.splitlines()[0]
    if head in source:
        source = source[: source.index(head)].rstrip("\n") + "\n"
    source = re.sub(r"      call fks_integrand_dump\(.*\n(     \$.*\n)*", "", source)
    anchor = "      call include_PDF_and_alphas\n"
    if source.count(anchor) != 1:
        raise ValueError("sigint: include_PDF_and_alphas not found once")
    source = source.replace(anchor, anchor + DUMP_CALL)
    with open(path, "w") as f:
        f.write(source + "\n" + dump)
    for stale in glob.glob(os.path.join(out, "SubProcesses", "P*", "madevent_mintFO")):
        os.remove(stale)


def sum_helicities(out):
    """MadLoop sums the helicities of the virtual explicitly
    (NHelForMCoverHels = -1 in FKS_params.dat) instead of sampling one, so
    that the finite part it gives is that of the point."""
    path = os.path.join(out, "Cards", "FKS_params.dat")
    with open(path) as f:
        lines = f.read().split("\n")
    k = lines.index("#NHelForMCoverHels")
    lines[k + 1] = "-1"
    with open(path, "w") as f:
        f.write("\n".join(lines))


def launch(out, values, points, work, name):
    run_name = f"dump_{len(os.listdir(os.path.join(out, 'Events')))}"
    script = os.path.join(work, f"{name}_launch.mg")
    with open(script, "w") as f:
        f.write(f"launch {out} -n {run_name}\nfixed_order=ON\ndone\n")
        for key, value in {**RUN_CARD, **values}.items():
            f.write(f"set {key} {value}\n")
        f.write("done\n")
    env = dict(os.environ, MADFKS_INTEGRAND_DUMP=str(points))
    log = os.path.join(work, f"{name}_launch.log")
    summary = os.path.join(out, "Events", run_name, "summary.txt")
    mfr.madgraph(script, work, log, env=env, expect=summary)


def floats(fields):
    return [float(v) for v in fields]


def configuration(fields):
    v = floats(fields[1:])
    return dict(
        valid=bool(int(fields[0])),
        jacobian=v[0],
        xi=v[1],
        y=v[2],
        xi_hat=v[3],
        xi_max=v[4],
        xbjrk=v[5:7],
    )


def parse(text, nexternal):
    points, point, contribution, orders = [], None, None, []
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        key, rest = fields[0], fields[1:]
        if key == "ORDERS":
            v = [int(f) for f in rest]
            orders.append(dict(position=v[0], qcd_pos=v[1], qed_pos=v[2], orders=v[3:]))
        elif key == "G":
            point["g"] = float(rest[0])
        elif key == "FKS":
            v = [int(f) for f in rest]
            point["blocks"] = dict(nfks=v[0], i_fks=v[1], j_fks=v[2])
        elif key == "REAL":
            v = floats(rest)
            point["blocks"].update(g_real=v[0], real=v[1], real_orders=v[2:])
        elif key == "KPERP":
            v = floats(rest)
            point["blocks"].update(k_perp=v[:4], azimuthal_phase=v[4:6])
        elif key == "BOUNDS":
            v = floats(rest)
            point["born"].update(
                tau_born_lower_bound=v[0],
                tau_lower_bound_resonance=v[1],
                tau_lower_bound=v[2],
            )
        elif key == "RESONANCE":
            v = floats(rest[1:])
            point["resonance"] = dict(
                resonance_recoil=bool(int(rest[0])), mass2=v[0], momentum=v[1:5]
            )
        elif key == "BORN_BLOCK":
            g, *v = floats(rest)
            n = len(orders)
            point["blocks"].update(
                g_born=g,
                born=v[0],
                born_orders=v[1 : 1 + n],
                azimuthal_orders=[v[1 + n : 1 + 2 * n], v[1 + 2 * n : 1 + 3 * n]],
            )
        elif key == "FKSB":
            v = [int(f) for f in rest]
            point["blocks"]["nbody"] = dict(nfks=v[0], i_fks=v[1], j_fks=v[2], links=[])
        elif key == "LINK":
            g, *v = floats(rest[2:])
            point["blocks"]["nbody"]["links"].append(
                dict(m=int(rest[0]), n=int(rest[1]), g=g, value=v[0], orders=v[1:])
            )
        elif key in ("CHANNEL", "CHANNELB"):
            channel = dict(
                config=int(rest[0]),
                diagram=int(rest[1]),
                symmetry_factor=float(rest[2]),
                enhance=float(rest[3]),
            )
            if key == "CHANNEL":
                point["blocks"]["channel"] = channel
            else:
                point["blocks"]["nbody"]["channel"] = channel
        elif key == "AMP2":
            # the channel of the last CHANNEL or CHANNELB line
            nbody = point["blocks"]["nbody"] if "nbody" in point["blocks"] else None
            target = nbody if nbody and "channel" in nbody else point["blocks"]
            target["channel"]["amp2"] = floats(rest)
        elif key == "VIRTUAL":
            qes2, fraction, *v = floats(rest)
            point["blocks"]["nbody"]["virtual"] = dict(
                qes2=qes2, fraction=fraction, orders=v
            )
        elif key == "FACTORS":
            v = floats(rest)
            point["prefactors"] = dict(
                f_b=v[0],
                f_nb=v[1],
                f_r=v[2],
                f_s=v[3],
                f_c=v[4],
                f_dc=v[5],
                f_sc=v[6],
                f_dsc=v[7:11],
                f_pdf_scheme=v[11:14],
            )
        elif key == "CUTS":
            nbody, collinear, event = (bool(int(v)) for v in rest)
            point["cuts"] = dict(soft=nbody, collinear=collinear, event=event)
        elif key == "SYMMETRY":
            point["symmetry_factors"] = floats(rest)
        elif key == "XINORM":
            v = floats(rest)
            point["xi_norm"] = dict(
                xinorm_ev=v[0], xinorm_cnt=v[1:4], xiimax_ev=v[4], xiimax_cnt=v[5:8]
            )
        elif key == "POINT":
            v = [int(f) for f in rest]
            point = dict(
                call=v[0],
                ichan=v[1],
                iconfig=v[2],
                nfks_picked=v[3],
                iran_picked=v[4],
                nfks_born=v[5],
                real_type=v[6],
                ndim=v[7],
                sum=bool(v[8]),
                counter_events=[],
                contributions=[],
            )
        elif key == "WEIGHTS":
            point["vegas_wgt"], point["vol"] = floats(rest)
        elif key == "XX":
            point["xx"] = floats(rest)
        elif key == "X":
            point["x"] = floats(rest)
        elif key == "BORN":
            v = floats(rest[1:])
            point["born"] = dict(
                valid=bool(int(rest[0])),
                xjac=v[0],
                xpswgt=v[1],
                shat=v[2],
                tau=v[3],
                ycm=v[4],
                xbjrk=v[5:7],
                stot=v[7],
                weight=v[8],
            )
        elif key == "BORNP":
            v = floats(rest)
            point["p_born"] = [v[k : k + 4] for k in range(0, len(v), 4)]
        elif key == "EVENT":
            point["event"] = configuration(rest)
        elif key == "CNT":
            point["counter_events"].append(configuration(rest[1:]))
        elif key == "RAD":
            point["radiation"] = floats(rest)
        elif key == "CINT":
            v = [int(f) for f in rest]
            contribution = dict(
                index=v[0],
                type=v[1],
                nfks=v[2],
                qcd_power=v[3],
                h_event=bool(v[4]),
                niproc=v[5],
                orders_tag=v[6],
                amp_pos=v[7],
            )
            point["contributions"].append(contribution)
        elif key == "CWGT":
            v = floats(rest)
            contribution.update(
                wgt=v[0:3],
                bjx=v[3:5],
                scales2=v[5:8],
                g_strong=v[8],
                y_bst=v[9],
                cpower=v[10],
                weight=v[11],
                wgt_me_tree=v[12:14],
            )
        elif key == "CMOM":
            v = floats(rest)
            contribution["momenta"] = [v[k : k + 4] for k in range(0, 4 * nexternal, 4)]
        elif key == "CPDG":
            contribution["pdg"] = [int(f) for f in rest]
        elif key == "CLUM":
            contribution["parton_iproc"] = floats(rest)
        elif key in ("CFLV", "CUBN"):
            codes = [int(f) for f in rest]
            size = nexternal if key == "CFLV" else nexternal - 1
            name = "parton_pdg" if key == "CFLV" else "parton_pdg_uborn"
            contribution[name] = [
                codes[k : k + size] for k in range(0, len(codes), size)
            ]
        elif key == "END":
            points.append(point)
            point = None
    return orders, points


def read_cards(out):
    """The values of the run_card and param_card the integrand depends on,
    with maxjetflavor of run_card.inc where the run_card hides it."""
    run_card = {}
    with open(os.path.join(out, "Cards", "run_card.dat")) as f:
        for line in f:
            line = line.split("!")[0]
            if "=" in line and not line.lstrip().startswith("#"):
                value, key = (s.strip() for s in line.split("=", 1))
                run_card[key] = value
    with open(os.path.join(out, "Source", "run_card.inc")) as f:
        for line in f:
            if line.split("=")[0].strip().upper() == "MAXJETFLAVOR":
                run_card.setdefault("maxjetflavor", line.split("=")[1].strip())
    blocks, block = {}, None
    with open(os.path.join(out, "Cards", "param_card.dat")) as f:
        for line in f:
            line = line.split("#")[0].strip()
            if not line:
                continue
            fields = line.split()
            if fields[0].lower() == "block":
                block = fields[1].lower()
                blocks[block] = {}
            elif fields[0].lower() == "decay":
                blocks.setdefault("decay", {})[fields[1]] = float(fields[2])
                block = None
            elif block:
                blocks[block][" ".join(fields[:-1])] = float(fields[-1])
    return run_card, blocks


def light_flavors(out):
    """The numbers of light quarks and massless leptons of the model
    (NF and NL of coupl.inc), which the subtraction takes."""
    values = {}
    with open(os.path.join(out, "Source", "coupl.inc")) as f:
        for line in f:
            for key in ("NF", "NL"):
                head = f"PARAMETER({key}="
                if head in line.replace(" ", ""):
                    text = line.replace(" ", "").split(head)[1].split(")")[0]
                    values[key.lower()] = float(text.lower().replace("d", "e"))
    return dict(n_f=values["nf"], n_l=values["nl"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--points", type=int, default=40, help="points per channel (see POINTS)"
    )
    parser.add_argument("--keep", help="keep the generated outputs in this directory")
    parser.add_argument("--only", help="only this process")
    args = parser.parse_args()

    commit = mfr.commit(MG_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    work = args.keep or tempfile.mkdtemp(prefix="fks_integrand_reference_")
    os.makedirs(work, exist_ok=True)
    names = [args.only] if args.only else list(PROCESSES)
    for name in names:
        process, values = PROCESSES[name]
        out = generate(name, process, work, os.path.join(work, f"{name}_gen.log"))
        patch_driver(out)
        sum_helicities(out)
        launch(out, values, POINTS.get(name, args.points), work, name)
        run_card, param_card = read_cards(out)
        channels = []
        sub_root = os.path.join(out, "SubProcesses")
        for sub in sorted(d for d in os.listdir(sub_root) if d.startswith("P")):
            pdir = os.path.join(sub_root, sub)
            with open(os.path.join(pdir, "fks_configurations.json")) as f:
                fks_configurations = json.load(f)
            nexternal = len(fks_configurations["born_pdgs"]) + 1
            for channel in sorted(os.listdir(pdir)):
                path = os.path.join(pdir, channel, "fks_integrand_dump.txt")
                if not os.path.isfile(path):
                    continue
                with open(path) as f:
                    orders, points = parse(f.read(), nexternal)
                channels.append(
                    dict(
                        subprocess=sub,
                        channel=channel,
                        fks_configurations=fks_configurations,
                        orders=orders,
                        points=points,
                    )
                )
        data = dict(
            process=process,
            mg_commit=commit,
            run_card=run_card,
            param_card=param_card,
            **light_flavors(out),
            channels=channels,
        )
        path = os.path.join(OUT_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
            f.write("\n")
        npts = sum(len(c["points"]) for c in channels)
        print(f"{path}: {len(channels)} channels, {npts} points")
    if not args.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
