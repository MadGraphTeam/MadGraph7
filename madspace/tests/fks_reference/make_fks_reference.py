"""Reference values of the FKS phase space of MadGraph7's genps_fks.f and of
the subtraction of fks_singular.f.

For every process below, this script generates an
'output standalone_fortran --fks --limits' directory with MadGraph7, builds
fks_kinematics_dump.f in its P* directories (linked like test_soft_col_limits)
and writes the dumped kinematics to madspace/tests/test_data/fks/<name>.json.
At the Born of every point it adds the Born and its color-correlated and
spin-correlated versions from pyAmpliCol (pip install pyamplicol), in the
conventions that the madspace FKS limits take. The madspace FKS tests read
only the JSON files, so they need neither Fortran nor pyAmpliCol.

Run from anywhere with a gfortran in the PATH and pyAmpliCol importable:

    python madspace/tests/fks_reference/make_fks_reference.py [--points N]
        [--massive M] [--limit L]

Each FKS configuration gets N points at uniform random x, and one with a
massive final-state emitter M more with y_ij < 0 and xi close to its upper
end, which cover the second solution for the momentum of the emitter
(isolsign = -1). Each configuration also gets L points close to the soft
limit, every second one also close to the collinear one, where sreal and
fks_Sij switch to the limits. They are stored apart, as limit_points, with
the entries of points.

The JSON layout, with Fortran conventions kept (1-based particle indices,
counter-event index k = -2..2 stored at position k + 2, momenta as
(E, px, py, pz) in the partonic c.m. frame):

    process, subprocess, genps_fks_sha (git blob hash of genps_fks.f),
    nexternal, nincoming, ndim, lpp, ebeam, g (the strong coupling of the
    Fortran), born_blocks (source, alpha_s and top width of the Born blocks),
    subtraction (xicut, xi_s_cut, delta_s, delta_o, delta_i of fks_powers.inc,
    set for some processes away from their defaults, see FKS_POWERS),
    pdf_scheme (pdfscheme of the run_card, 0 MSbar or 1 DIS, see RUN_CARD),
    n_f (the number of light flavors), n_l (the number of massless
    leptons), scales (mu_f2 of the two beams and
    qes2, the Ellis-Sexton scale, set by the dump for the collinear
    remnants), soft_virtual_scales (mu_f2 of the two beams, mu_r2 and qes2,
    set by the dump for bornsoftvirtual), split_orders (amp_split_size,
    amp_split_size_born, qcd_pos, qed_pos of orders.inc, 1-based, -1 if
    absent, and amp_split: the powers of each split order at every position
    of the amp_split arrays), gal (the QED coupling e of the Fortran),
    fks_configurations (the fks_configurations.json the FKS exporter writes
    into the P* directory, with its own conventions, see madspace/fks.py)
    configs[]: nfks, i_fks, j_fks, pdg, masses, particle_type,
               particle_charge, fks_j_from_i (the common blocks of the
               configuration after fks_inc_chooser), iden_comp, points[],
               limit_points[] (the same for the points close to the limits)
    points[]:  x (unit-cube vector, radiation in the last three entries),
               isolsign, jac, jac_cnt (full weight of generate_momenta),
               p_born, p_ev, p_cnt, xi_ev, xi_hat_ev, y_ev, xiimax_ev,
               xi_cnt, xi_hat_cnt, xiimax_cnt, xbjrk_ev, xbjrk_cnt, shat_ev,
               shat_cnt, tau_ev, ycm_ev, tau_cnt, ycm_cnt, kperp,
               p_i_hat_ev, p_i_hat_cnt (p_i_fks_ev, p_i_fks_cnt: i per unit
               xi), s_ij_ev, s_ij_cnt (fks_Sij at the event and at the
               counter-events k = 0, 1, 2 as compute_soft_counter_term and
               its siblings call it, -1 where the counter-event is not
               generated or, for a massive emitter, has no collinear limit),
               eik_pairs, eik_soft (eikonal_reduced at the soft
               counter-event for the pairs (m, n) of sbornsoft, 1-based,
               in its order; empty where it is not generated), coll_cnt
               (at the collinear and soft-collinear counter-events: z, t and
               AP_reduced, Qterms_reduced_timelike/spacelike as
               sborncol_fsr/isr call them, QCD over g^2 and QED over
               gal(1)^2; z = -1 where not generated or j is massive),
               prefactors (f_r, f_s, f_c, f_dc, f_sc, f_dsc of
               compute_prefactors_n1body with unit vegas weight),
               fks_symmetry_factor, fks_symmetry_factor_deg, xinorm_ev,
               xinorm_cnt, no_counter_events (where the event is generated),
               sreal (sreal at the event and at the soft, collinear and
               soft-collinear counter-events as compute_real_emission and
               compute_*_counter_term call it, -1 where not generated or, for
               the collinear ones, j is massive), born (the QCD part of the
               Born of sborn at p_born) and color_links (sborn_sf for the
               pairs of sbornsoft, with g^2, empty where i is a quark), both
               where the event is generated, born_blocks (where the event is
               generated: born, color_correlated and, for a gluon or photon
               emitter, spin_correlated of pyAmpliCol at p_born, see
               born_blocks),
               collinear_remnants (sreal_deg at the collinear and
               soft-collinear counter-events of a massless incoming
               emitter: wgtdegrem_xi, wgtdegrem_lxi, wgtdegrem_muF and the
               PDF-scheme parts p, l, d, zero elsewhere), prefactors also
               has f_pdf_scheme (f_pdfsch_d, f_pdfsch_p, f_pdfsch_l),
               soft_virtual (bornsoftvirtual at the soft counter-event
               without the virtual, where it is generated and i has colour
               or charge links: bsv_wgt, born_wgt and the sums over the
               split orders of wgtnstmp, wgtwnstmpmur, wgtwnstmpmuf),
               and per position of amp_split: sreal_orders (amp_split after
               the four calls of sreal, zero where not generated),
               born_orders (amp_split_cnt for a QCD and for a QED
               splitting), extra_born_orders (the same for the Born of the
               extra counterterm, zero without one), color_links_orders (amp_split_soft for each pair
               of color_links, which are the charge-linked Borns where i is
               a photon), collinear_remnants_orders (the six pieces of
               collinear_remnants per counter-event),
               scheme_remnants_orders (the parts p, l and d of the PDF
               scheme of sreal_deg per counter-event for each of the
               schemes 1 to 7 of pdfscheme, with xiAI = 1 and RAI = 31 and
               35 GeV^-1 of the nuclei of UPC) and
               soft_virtual_orders (amp_split after bornsoftvirtual,
               wgtnstmp, wgtwnstmpmur, wgtwnstmpmuf),
               with lepton PDFs (LEPTON_PDFS) where an incoming emitter
               radiates without the projection of the event: noevpr
               (tau_born_lower_bound, x and 1 - x of the luminosity, and
               born_rec: s_rec, the weight of generate_momenta_born and the
               Born momenta in their rest frame for the event and the soft,
               collinear and soft-collinear counter-events), p_born_coll and
               collinear_born_blocks (the Born blocks at p_born_coll),
               shat_born, tau_born, ycm_born, xbjrk_born, tau_lower_bound,
               and for prefix in (rad, cut): prefix_jac, prefix_jac_cnt,
               prefix_p_ev, prefix_p_cnt, prefix_xi_ev, prefix_xi_hat_ev,
               prefix_y_ev, prefix_xiimax_ev, prefix_xi_cnt, prefix_xi_hat_cnt,
               prefix_xiimax_cnt, prefix_xbjrk_ev, prefix_xbjrk_cnt,
               prefix_shat_ev, prefix_shat_cnt, prefix_p_i_hat_ev,
               prefix_p_i_hat_cnt (p_i_fks_ev, p_i_fks_cnt: i per unit xi)
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
import math
import os
import re
import shutil
import subprocess
import sys
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
    # the parameters of the subtraction (fks_powers.inc) away from their
    # defaults, under which some prefactors vanish
    "pp_epemj_params": ("p p > e+ e- j [QCD]", ["P0_ug_uemep"]),
    # PDFs in the DIS scheme, for the collinear remnants of initial-state
    # emitters of every kind
    "pp_epemj_dis": ("p p > e+ e- j [QCD]", ["P0_ug_uemep"]),
    "pp_ttx_dis": ("p p > t t~ [QCD]", ["P0_gg_ttx"]),
    # QED corrections: photon emission off quarks and leptons, initial
    # photons splitting into quarks and quarks into photons
    "pp_epem_qed": ("p p > e+ e- [QED]", ["P0_uux_emep", "P0_aa_emep"]),
    # the same in the DIS scheme, for the QED parts of its kernels
    "pp_epem_qed_dis": ("p p > e+ e- [QED]", ["P0_uux_emep", "P0_aa_emep"]),
    # a final-state photon that splits into quarks and leptons
    "pp_epema_qed": ("p p > e+ e- a [QED]", ["P0_uux_emepa"]),
    # photon emission off massive charged particles
    "aa_ttx_qed": ("a a > t t~ [QED]", ["P0_aa_ttx"]),
    # mixed QCD and QED corrections: a Born of three orders, and extra
    # counterterms for a quark pair from a gluon or a photon, off a final-state
    # gluon and off an incoming quark (a leptonic Born of these needs lepton
    # PDFs, MadFKS has no initial-state emitter at fixed s_hat)
    "uux_uux_mix": ("u u~ > u u~ QCD^2<=4 QED^2<=4 [QCD QED]", None),
    "pp_aj_mix": (
        "p p > a j QED^2<=4 QCD^2<=2 [QCD QED]",
        ["P0_gu_ua", "P0_uux_ag"],
    ),
    # lepton PDFs (LEPTON_PDFS): photon emission off the incoming leptons
    # without the projection of the event, beside final-state emitters
    "epem_uuxg_mix": ("e+ e- > u u~ g QED^2<=6 QCD^2<=2 [QCD QED]", None),
    "epem_mumu_qed": ("e+ e- > mu+ mu- [QED]", None),
}
# the model of the processes that need another one than MadGraph picks
MODELS = {
    "pp_epem_qed": "loop_qcd_qed_sm",
    "pp_epem_qed_dis": "loop_qcd_qed_sm",
    "pp_epema_qed": "loop_qcd_qed_sm",
    "aa_ttx_qed": "loop_qcd_qed_sm",
    "uux_uux_mix": "loop_qcd_qed_sm",
    "pp_aj_mix": "loop_qcd_qed_sm",
    "epem_uuxg_mix": "loop_qcd_qed_sm",
    "epem_mumu_qed": "loop_qcd_qed_sm",
}
# the processes with lepton PDFs, lpp of both beams: the Source of an NLO
# output takes no run_card with them, so the dump sets it
LEPTON_PDFS = {"epem_uuxg_mix": 3, "epem_mumu_qed": 3}
FKS_POWERS = {
    "pp_epemj_params": dict(xicut=0.3, xiScut=0.25, deltaO=0.8, deltaI=0.7, deltaS=0.6),
}
# parameters of the run_card set away from their defaults
RUN_CARD = {
    "pp_epemj_dis": dict(pdfscheme=1),
    "pp_ttx_dis": dict(pdfscheme=1),
    "pp_epem_qed_dis": dict(pdfscheme=1),
}


def run(cmd, cwd, log, stdin=None):
    with open(log, "w") as f:
        subprocess.run(
            cmd, cwd=cwd, stdin=stdin, stdout=f, stderr=subprocess.STDOUT, check=True
        )


def generate(name, process, work, run_card=None, model=None):
    out = os.path.join(work, name)
    if not os.path.isdir(os.path.join(out, "SubProcesses")):
        script = os.path.join(work, f"{name}.mg")
        with open(script, "w") as f:
            f.write("set automatic_html_opening False\n")
            if model:
                f.write(f"import model {model}\n")
            f.write(f"generate {process}\n")
            f.write(f"output standalone_fortran {out} --fks --limits\n")
        run(
            ["python3", os.path.join(MG_ROOT, "bin", "madgraph"), script],
            work,
            os.path.join(work, f"{name}_mg.log"),
        )
    if run_card:
        # Source turns the run_card into run_card.inc
        set_run_card(out, run_card)
    run(
        ["make", "-j8"],
        os.path.join(out, "Source"),
        os.path.join(work, f"{name}_src.log"),
    )
    return out


def set_run_card(out, values):
    """Sets parameters of Cards/run_card.dat, before Source is built."""
    path = os.path.join(out, "Cards", "run_card.dat")
    with open(path) as f:
        text = f.read()
    for key, value in values.items():
        text, count = re.subn(
            rf"^\s*\S+(\s*=\s*{key}\b)", rf" {value}\1", text, flags=re.MULTILINE
        )
        if count != 1:
            raise ValueError(f"{key} not found in {path}")
    with open(path, "w") as f:
        f.write(text)


def set_fks_powers(out, values):
    """Sets parameters of SubProcesses/fks_powers.inc, before anything is
    built."""
    path = os.path.join(out, "SubProcesses", "fks_powers.inc")
    with open(path) as f:
        text = f.read()
    for key, value in values.items():
        text, count = re.subn(
            rf"parameter \({key}=[^)]*\)", f"parameter ({key}={value!r}d0)", text
        )
        if count != 1:
            raise ValueError(f"{key} not found in {path}")
    with open(path, "w") as f:
        f.write(text)


def dump(pdir, points, massive, limit, log, lpp=0):
    for f in ["fks_kinematics_dump.f", "makefile_dump"]:
        shutil.copy(os.path.join(HERE, f), pdir)
    run(["make", "-j8", "-f", "makefile_dump", "fks_kinematics_dump"], pdir, log)
    result = subprocess.run(
        ["./fks_kinematics_dump"],
        cwd=pdir,
        input=f"{points}\n1\n{massive}\n{limit}\n{lpp}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def floats(fields):
    return [float(v) for v in fields]


def momenta(values):
    return [values[i : i + 4] for i in range(0, len(values), 4)]


RERUN_KEYS = ("JAC", "PEV", "PCNT", "XI", "XBJ", "SHAT", "PIHAT")


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
    elif key == "PIHAT":
        v = momenta(floats(rest))
        point[f"{prefix}_p_i_hat_ev"], point[f"{prefix}_p_i_hat_cnt"] = v[0], v[1:6]


def per_order(values, data, count=None):
    """Values the Fortran writes as arrays over the positions of amp_split,
    one after the other (count of them where it is given), as lists."""
    size = data["split_orders"]["amp_split_size"]
    if len(values) % size or (count is not None and len(values) != count * size):
        raise ValueError(f"{len(values)} values for arrays of {size}")
    return [values[k : k + size] for k in range(0, len(values), size)]


def parse(text):
    data = {"configs": []}
    conf = point = None
    limit = False
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
        elif key == "SCHEME":
            data["pdf_scheme"], data["n_f"] = int(rest[0]), float(rest[1])
            data["n_l"] = float(rest[2])
        elif key == "ORDERS":
            size, born_size, qcd_pos, qed_pos, _ = map(int, rest)
            data["split_orders"] = dict(
                amp_split_size=size,
                amp_split_size_born=born_size,
                qcd_pos=qcd_pos,
                qed_pos=qed_pos,
                amp_split=[],
            )
        elif key == "AMPSPLIT":
            data["split_orders"]["amp_split"].append(list(map(int, rest[1:])))
        elif key == "COUPL":
            data["gal"] = floats(rest)[0]
        elif key == "CONF":
            # the extra points for massive emitters come in a second pass
            ints = list(map(int, rest))
            known = [c for c in data["configs"] if c["nfks"] == ints[0]]
            if known:
                conf = known[0]
            else:
                conf = {"nfks": ints[0], "i_fks": ints[1], "j_fks": ints[2]}
                conf["pdg"] = ints[3:]
                conf["points"] = []
                data["configs"].append(conf)
        elif key == "MASS":
            conf["masses"] = floats(rest)
        elif key == "PTYPE":
            conf["particle_type"] = list(map(int, rest))
        elif key == "PCHARGE":
            conf["particle_charge"] = floats(rest)
        elif key == "JFROMI":
            ints = list(map(int, rest))
            js = conf.setdefault("fks_j_from_i", [])
            if len(js) < ints[0]:
                js.append(ints[1:])
        elif key == "PASS":
            # the third pass has the points close to the limits
            limit = int(rest[0]) == 3
        elif key == "POINT":
            point = {"isolsign": int(rest[2])}
            conf.setdefault("limit_points" if limit else "points", []).append(point)
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
        elif key == "PIHAT":
            v = momenta(floats(rest))
            point["p_i_hat_ev"], point["p_i_hat_cnt"] = v[0], v[1:6]
        elif key == "SIJ":
            v = floats(rest)
            point["s_ij_ev"], point["s_ij_cnt"] = v[0], v[1:4]
        elif key == "EIKPAIRS":
            v = [int(f) for f in rest]
            point["eik_pairs"] = [v[1 + 2 * k : 3 + 2 * k] for k in range(v[0])]
        elif key == "EIK":
            point["eik_soft"] = floats(rest)
        elif key == "COLL":
            v = floats(rest)
            point["coll_cnt"] = [v[0:6], v[6:12]]
        elif key == "PREF":
            v = floats(rest)
            point["prefactors"] = dict(
                zip(["f_r", "f_s", "f_c", "f_dc", "f_sc"], v[0:5]), f_dsc=v[5:9]
            )
            point.update(
                fks_symmetry_factor=v[9],
                fks_symmetry_factor_deg=v[10],
                xinorm_ev=v[11],
                xinorm_cnt=v[12:17],
                no_counter_events=bool(v[23]),
            )
            point["prefactors"]["f_pdf_scheme"] = v[24:27]
            data["subtraction"] = dict(
                zip(
                    ["xicut", "xi_s_cut", "delta", "delta_s", "delta_o", "delta_i"],
                    v[17:23],
                )
            )
            data["subtraction"].pop("delta")
        elif key == "SREAL":
            point["sreal"] = floats(rest)
        elif key == "SREALO":
            point["sreal_orders"] = per_order(floats(rest), data, 4)
        elif key == "BORNO":
            point["born_orders"] = per_order(floats(rest), data, 2)
        elif key == "BORNXO":
            point["extra_born_orders"] = per_order(floats(rest), data, 2)
        elif key == "BLINKO":
            point["color_links_orders"] = per_order(floats(rest), data)
        elif key == "DEGO":
            v = per_order(floats(rest), data, 12)
            point["collinear_remnants_orders"] = [v[0:6], v[6:12]]
        elif key == "NOEVPR":
            v = floats(rest)
            point["noevpr"] = dict(
                tau_born_lower_bound=v[1], x=v[2:4], one_minus_x=v[4:6]
            )
        elif key == "PBORNCOLL":
            point["p_born_coll"] = momenta(floats(rest))
        elif key == "BORNREC":
            v = floats(rest[1:])
            point["noevpr"].setdefault("born_rec", []).append(
                dict(s_rec=v[0], weight=v[1], p_born=momenta(v[2:]))
            )
        elif key == "PSCHO":
            v = per_order(floats(rest), data, 42)
            point["scheme_remnants_orders"] = [
                [v[6 * scheme + 3 * k : 6 * scheme + 3 * k + 3] for k in range(2)]
                for scheme in range(7)
            ]
        elif key == "BSVO":
            point["soft_virtual_orders"] = per_order(floats(rest), data, 4)
        elif key == "BORN":
            g, iden_comp, _, born = floats(rest)
            if data.setdefault("g", g) != g:
                raise ValueError("g changes from point to point")
            if conf.setdefault("iden_comp", iden_comp) != iden_comp:
                raise ValueError("iden_comp changes from point to point")
            point["born"] = born
        elif key == "BLINK":
            point["color_links"] = floats(rest)
        elif key == "DEG":
            v = floats(rest)
            point["collinear_remnants"] = [v[0:6], v[6:12]]
            scales = dict(mu_f2=v[12:14], qes2=v[14])
            if data.setdefault("scales", scales) != scales:
                raise ValueError("the scales change from point to point")
        elif key == "BSV":
            v = floats(rest)
            point["soft_virtual"] = v[0:5]
            scales = dict(mu_f2=v[5:7], mu_r2=v[7], qes2=v[8])
            if data.setdefault("soft_virtual_scales", scales) != scales:
                raise ValueError("the scales change from point to point")
        elif key == "BORNX":
            v = floats(rest)
            point.update(shat_born=v[0], tau_born=v[1], ycm_born=v[2])
            point["xbjrk_born"] = v[3:5]
        elif key == "TAULB":
            point["tau_lower_bound"] = float(rest[0])
        elif key[:3] in ("RAD", "CUT") and key[3:] in RERUN_KEYS:
            parse_rerun(point, key[:3].lower(), key[3:], rest)
    return data


# PDG code -> particle of the built-in Standard Model of pyAmpliCol
PARTICLES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t", 11: "e-", 13: "mu-"}
PARTICLES.update({15: "ta-", 21: "g", 22: "a", 23: "z", 25: "h"})


def particle(pdg):
    if pdg in (21, 22, 23, 25):
        return PARTICLES[pdg]
    name = PARTICLES[abs(pdg)]
    if pdg > 0:
        return name
    return name + "~" if abs(pdg) <= 6 else name[:-1] + "+"


def colored(pdg):
    return abs(pdg) <= 6 or pdg == 21


# the massless vector bosons, whose splittings have azimuthal correlations
VECTORS = (21, 22)


def born_runtime(born_pdgs, nincoming, path, model=None, color=True):
    """The pyAmpliCol runtime of a Born, with the color correlators of all
    pairs of colored particles (where color is set) and the spin correlators
    of every gluon and photon, in the built-in Standard Model or the given
    model source, with the coupling orders of the lowest order."""
    import pyamplicol as pa
    from pyamplicol.config import ColorConfig, RunConfig

    legs = [k + 1 for k, pdg in enumerate(born_pdgs) if colored(pdg)]
    if not color:
        legs = []
    if not os.path.isdir(path):
        process = " ".join(particle(pdg) for pdg in born_pdgs[:nincoming])
        process += " > " + " ".join(particle(pdg) for pdg in born_pdgs[nincoming:])
        correlators = pa.CorrelatorConfig(
            color_correlations=tuple(
                pa.ColorCorrelator.dipole(f"T{a}_{b}", a, b)
                for k, a in enumerate(legs)
                for b in legs[k:]
            ),
            spin_correlations=tuple(
                (k + 1,) for k, pdg in enumerate(born_pdgs) if pdg in VECTORS
            ),
        )
        # pyAmpliCol 1.0.1 has no full color for amplitudes without a QCD
        # vertex, whose color factor is exact at its default accuracy where
        # they have one quark line
        quarks = sum(1 for pdg in born_pdgs if abs(pdg) <= 6)
        color = ColorConfig(accuracy="full")
        if not any(pdg == 21 for pdg in born_pdgs) and not legs and quarks <= 2:
            color = ColorConfig()
        config = RunConfig(action="generate", color=color)
        pa.Generator(config).generate(
            process,
            path,
            model=model or pa.ModelSource.built_in_sm(),
            correlators=correlators,
        )
    runtime = pa.Runtime.load(path)
    if tuple(runtime.external_permutation) != tuple(range(len(born_pdgs))):
        raise ValueError(f"pyAmpliCol reorders the particles of {born_pdgs}")
    return runtime, legs


def sbornsoft_pairs(conf):
    """The pairs (m, n) of sbornsoft for the configuration of reference data,
    1-based in the real process, in the order of the color links of the
    dump."""
    i_fks = conf["i_fks"]
    partners = conf["fks_j_from_i"][i_fks - 1]
    pairs = []
    for k, m in enumerate(partners):
        for n in partners[: k + 1]:
            if (m != n or conf["masses"][m - 1] != 0.0) and i_fks not in (m, n):
                pairs.append((m, n))
    return pairs


def fortran_orders(data, conf, point):
    """The Born and the color-correlated Borns per Born order at a point of
    reference data, from the Fortran: the Born of amp_split_cnt of the
    splitting type with which all Born orders are kept, the color-correlated
    ones from the color-linked Borns of sborn_sf per order (C_m B from color
    conservation for a massless m) where i is a gluon, zero elsewhere. For a
    Born of several orders: pyAmpliCol 1.0.1 drops the amplitudes without a
    QCD vertex where it has color correlators or full color (u u~ > u u~ at
    QCD^2 <= 4 and QED^2 <= 4 gives the order alpha_s^2 alone), and
    misses part of their interference where it has neither."""
    orders = data["fks_configurations"]["orders"]
    names = orders["split_orders"]
    amp_split = data["split_orders"]["amp_split"]
    n = len(data["fks_configurations"]["born_pdgs"])

    def position(order, name):
        order = list(order)
        order[names.index(name)] += 2
        if any(p > m for p, m in zip(order, orders["max_nlo"])):
            return None
        return amp_split.index(order)

    for k, name in enumerate(["QCD", "QED"]):
        if name not in names:
            continue
        positions = [position(o, name) for o in orders["born"]]
        if None not in positions:
            break
    else:
        raise ValueError("no splitting type keeps all Born orders")
    blocks = [
        dict(
            born=point["born_orders"][k][p],
            color_correlated=[[0.0] * n for _ in range(n)],
        )
        for p in positions
    ]
    i_fks = conf["i_fks"]
    if conf["pdg"][i_fks - 1] == 21:
        g2 = data["g"] ** 2
        qcd = [position(o, "QCD") for o in orders["born"]]
        for (m, l), links in zip(sbornsoft_pairs(conf), point["color_links_orders"]):
            mb, lb = m - (m > i_fks), l - (l > i_fks)
            for block, p in zip(blocks, qcd):
                value = links[p] / g2 * (2 if m == l else 1)
                block["color_correlated"][mb - 1][lb - 1] = value
                block["color_correlated"][lb - 1][mb - 1] = value
        for block in blocks:
            matrix = block["color_correlated"]
            for m in range(n):
                if matrix[m][m] == 0.0 and any(matrix[m]):
                    matrix[m][m] = -math.fsum(matrix[m][l] for l in range(n) if l != m)
    return blocks


def param_card(out):
    """The values of the blocks of the param_card of an output, as
    {(block, code): value}."""
    values, block = {}, None
    with open(os.path.join(out, "Cards", "param_card.dat")) as f:
        for line in f:
            fields = line.split("#")[0].split()
            if not fields:
                continue
            if fields[0].lower() == "block":
                block = fields[1].lower()
            elif fields[0].lower() == "decay":
                values[("decay", int(fields[1]))] = float(fields[2])
            elif block is not None and len(fields) == 2:
                values[(block, int(fields[0]))] = float(fields[1])
    return values


def ufo_sm_parameters(out):
    """The external parameters of the sm UFO of MadGraph that give the tree
    level of loop_qcd_qed_sm at the param_card of an output: the same
    alpha(M_Z), M_Z, masses and widths, with G_F from its M_W (an input of
    loop_qcd_qed_sm, derived from G_F in sm)."""
    card = param_card(out)
    m_z, m_w = card[("mass", 23)], card[("mass", 24)]
    alpha = 1 / card[("sminputs", 1)]
    g_f = (
        math.pi
        * alpha
        * m_z**2
        / (math.sqrt(2) * (m_z**4 / 4 - (m_w**2 - m_z**2 / 2) ** 2))
    )
    return {
        "aEWM1": card[("sminputs", 1)],
        "Gf": g_f,
        "MZ": m_z,
        "MT": card[("mass", 6)],
        "MH": card[("mass", 25)],
        "WT": card[("decay", 6)],
        "WZ": card[("decay", 23)],
        "WW": card[("decay", 24)],
    }


def ufo_model(work):
    """The sm UFO of MadGraph as a pyAmpliCol model source, compiled once into
    the work directory (pyAmpliCol does not load the counterterm parameters
    of loop_qcd_qed_sm). In this process: Symbolica refuses a second process
    under a restricted license."""
    import pyamplicol as pa
    from pyamplicol.config import EvaluatorConfig

    pack = os.path.join(work, "sm.pyamplicol-model")
    if not os.path.exists(pack):
        source = pa.ModelSource.from_path(os.path.join(MG_ROOT, "models", "sm"))
        source.compile(prepared_output=pack, evaluator=EvaluatorConfig())
    return pa.ModelSource.from_path(pack)


def born_blocks(data, path, model=None, out=None, work=None):
    """The Born at p_born of every point where the event is generated, from
    pyAmpliCol: the Born B, the color-correlated Borns <T_m . T_n> B of every
    pair of colored particles (C_m B for m = n, all-outgoing color charges,
    zero for colorless particles) and, where the emitter of the configuration
    is a gluon or a photon in the Born, the spin-correlated tensor S with
    B(e) = e^mu e^nu S_mu,nu (contravariant components of e, summed with unit
    weights) the Born with the real vector e in place of the polarization of
    the emitter. A Born with several orders has them from the Fortran per
    order (orders, see fortran_orders), and their sums. A configuration with
    an extra
    counterterm has the Born of the other mother (extra_counterterms of the
    exporter) and its spin-correlated tensor per order of that Born (extra).
    The parameters are those of the param_card of MadGraph (the defaults of
    the built-in Standard Model of pyAmpliCol) at the alpha_s of the Fortran,
    but the top width: MadGraph sets it to zero in t-channel propagators
    (zerowidth_tchannel), the only ones of the top in these Borns. For
    loop_qcd_qed_sm (MODELS), the sm UFO of MadGraph at the parameters of the
    output (ufo_sm_parameters), since the built-in Standard Model of
    pyAmpliCol 1.0.1 is off by a factor 5.27 for every external photon."""
    import pyamplicol as pa

    configurations = data["fks_configurations"]
    nincoming = configurations["nincoming"]
    orders = configurations["orders"]
    alpha_s = data["g"] ** 2 / (4 * math.pi)
    if model == "loop_qcd_qed_sm":
        source_model = ufo_model(work)
        parameters = dict(ufo_sm_parameters(out), aS=alpha_s, WT=0.0)
        source = "sm UFO of MadGraph at the parameters of loop_qcd_qed_sm"
    else:
        source_model = None
        parameters = {
            "normalization.alpha_s_me_check": alpha_s,
            "particle.6.width": 0.0,
        }
        source = "built-in Standard Model"
    data["born_blocks"] = dict(
        source=f"pyAmpliCol {pa.package_version()}, {source}",
        alpha_s=alpha_s,
        top_width=0.0,
    )

    def runtime_of(pdgs, where, color):
        # the runtime of the Born and its colored particles
        runtime, legs = born_runtime(pdgs, nincoming, where, source_model, color)
        runtime.set_model_parameters(parameters)
        return runtime, legs

    def evaluate(pdgs, runtime, legs, p_born, born_j):
        # the blocks at p_born
        n = len(pdgs)
        vector = pdgs[born_j - 1] in VECTORS
        requests = {"born": pa.CorrelatedRequest("born", {})}
        for k, a in enumerate(legs):
            for b in legs[k:]:
                requests[f"T{a}_{b}"] = pa.CorrelatedRequest(f"T{a}_{b}", {})
        if vector:
            # the Born at e = u_mu + u_nu, with the unit vectors u_mu
            for mu in range(4):
                for nu in range(mu, 4):
                    e = [0.0] * 4
                    e[mu] = e[nu] = 1.0
                    requests[f"S{mu}{nu}"] = pa.CorrelatedRequest(
                        "born", {born_j: tuple(e)}
                    )
        values = runtime.evaluate_correlated_many(
            [p_born], requests, arithmetic="double-double"
        )
        value = {k: float(v[0].real) for k, v in values.items()}
        blocks = dict(born=value["born"])
        # zero for a Born without colored particles
        correlated = [[0.0] * n for _ in range(n)]
        for k, a in enumerate(legs):
            for b in legs[k:]:
                correlated[a - 1][b - 1] = value[f"T{a}_{b}"]
                correlated[b - 1][a - 1] = value[f"T{a}_{b}"]
        blocks["color_correlated"] = correlated
        if vector:
            tensor = [[0.0] * 4 for _ in range(4)]
            for mu in range(4):
                for nu in range(4):
                    diagonal = value[f"S{mu}{mu}"] + value[f"S{nu}{nu}"]
                    both = value[f"S{min(mu, nu)}{max(mu, nu)}"]
                    tensor[mu][nu] = both if mu == nu else 0.5 * (both - diagonal)
            blocks["spin_correlated"] = tensor
        return blocks

    def add(values):
        # the sum of numbers or of nested lists of them
        if isinstance(values[0], list):
            return [add(list(column)) for column in zip(*values)]
        return math.fsum(values)

    born_pdgs = configurations["born_pdgs"]
    several = len(orders["born"]) > 1
    born = None if several else runtime_of(born_pdgs, path, True)
    extras = []
    for k, (extra, extra_orders) in enumerate(
        zip(configurations.get("extra_counterterms", []), orders.get("extra_born", []))
    ):
        if len(extra_orders) > 1:
            raise NotImplementedError("an extra counterterm of several orders")
        extras.append(runtime_of(extra["born_pdgs"], f"{path}_extra{k}", False))
    if several:
        data["born_blocks"]["source"] = (
            "MadFKS per Born order: the Born of amp_split_cnt and the "
            "color-linked Borns of sborn_sf, see fortran_orders"
        )
    for conf, configuration in zip(data["configs"], configurations["configurations"]):
        i_fks, j_fks = conf["i_fks"], conf["j_fks"]
        born_j = j_fks if j_fks < i_fks else j_fks - 1
        extra = configuration["extra_cnt_index"]
        for point in conf["points"] + conf.get("limit_points", []):
            if "born" not in point:
                continue
            if several:
                if any(pdg in VECTORS for pdg in born_pdgs):
                    raise NotImplementedError(
                        "spin-correlated Borns of a Born of several orders"
                    )
                per_order = fortran_orders(data, conf, point)
                blocks = {key: add([b[key] for b in per_order]) for key in per_order[0]}
                blocks["orders"] = per_order
            else:
                blocks = evaluate(born_pdgs, *born, point["p_born"], born_j)
            if extra >= 0:
                pdgs = configurations["extra_counterterms"][extra]["born_pdgs"]
                blocks["extra"] = [
                    evaluate(pdgs, *extras[extra], point["p_born"], born_j)
                ]
            point["born_blocks"] = blocks
            if "p_born_coll" in point:
                if several or extra >= 0:
                    raise NotImplementedError(
                        "the Born of the collinear counter-event of a Born of "
                        "several orders or with an extra counterterm"
                    )
                point["collinear_born_blocks"] = evaluate(
                    born_pdgs, *born, point["p_born_coll"], born_j
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=4, help="points per FKS config")
    parser.add_argument(
        "--massive",
        type=int,
        default=8,
        help="extra points per FKS config with a massive final-state emitter",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="points per FKS config close to the soft (and collinear) limit",
    )
    parser.add_argument("--keep", help="keep the generated outputs in this directory")
    parser.add_argument(
        "--only", nargs="*", help="regenerate only these processes (names of PROCESSES)"
    )
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
        if args.only and name not in args.only:
            continue
        out = generate(name, process, work, RUN_CARD.get(name), MODELS.get(name))
        if name in FKS_POWERS:
            set_fks_powers(out, FKS_POWERS[name])
        for pdir in sorted(glob.glob(os.path.join(out, "SubProcesses", "P*"))):
            sub = os.path.basename(pdir)
            if keep is not None and sub not in keep:
                continue
            log = os.path.join(work, f"{name}_{sub}.log")
            data = parse(
                dump(
                    pdir,
                    args.points,
                    args.massive,
                    args.limit,
                    log,
                    LEPTON_PDFS.get(name, 0),
                )
            )
            data.update(process=process, subprocess=sub, genps_fks_sha=genps_fks_sha)
            # the FKS configurations as the exporter writes them for madspace
            with open(os.path.join(pdir, "fks_configurations.json")) as f:
                data["fks_configurations"] = json.load(f)
            pyamplicol_dir = os.path.join(work, f"{name}_{sub}_pyamplicol")
            born_blocks(data, pyamplicol_dir, MODELS.get(name), out, work)
            path = os.path.join(OUT_DIR, f"{name}_{sub}.json")
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.write("\n")
            npts = sum(
                len(c["points"]) + len(c.get("limit_points", []))
                for c in data["configs"]
            )
            print(f"{path}: {len(data['configs'])} FKS configs, {npts} points")
    if not args.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
