"""Reference values of the resonance-local recoil of an FKS emission off a
final-state parton, of the MCcntRefactor_Sfun branch of MadGraph7.

For the processes of PROCESSES below, this script generates the
'output standalone_fortran --fks --limits' directories with a checkout of that
branch (--mg-root), builds fks_kinematics_dump_modules.f in their P*
directories (the dump of make_fks_reference.py ported to the fks_phase_space
modules of the branch) and runs it for some Born configurations (iconfig),
the channels of the integration. The branch installs the resonance-local
recoil by default (select_fks_recoil, with the members of the resonance from
set_granny). The script keeps the entries of make_fks_reference.py, the
subtraction with the scales of the resonance too, with the Born blocks from the
Fortran (fortran_born_blocks), and writes one file per channel to
madspace/tests/test_data/fks_resonance/<name>_<subprocess>_c<iconfig>.json,
with the fks_configurations of the same process from MadGraph7 of this
repository (--main-root; the exporter of the branch does not write them) and

    iconfig, born_forest (per propagator of the Born configuration its index,
    its two daughters, sprop and tprid), born_props (their masses and
    widths), cuts (ptj, ptgmin, ptl, mll, mll_sf, maxjetflavor, gamma_is_j
    and pdg_cuts of the run_card, with maxjetflavor at least the number of
    light flavors of the model, as setcuts.f sets it), mg_branch, mg_commit,
    jac_sampling_factors (true, see make_fks_asymmetric_reference.py)
    configs[]: those of make_fks_reference.py, and
        granny: igranny, iaunt, granny_is_res and granny_chain_real_final of
            set_granny (1-based, as booleans over the real process)
        resonance: resonance_recoil, initial_recoil_leg and
            resonance_members of select_fks_recoil (1-based, as booleans)
    points[]: those of make_fks_reference.py, and resonance_mass2,
        resonance_momentum (the momentum K of the resonance), xij_aor
        (the azimuthal phase of the collinear limit, real and imaginary part)
        and resonance_scales (soft, collinear, angular and mismatch_log of
        resonance_subtraction_scales at the soft counter-event, where the
        event is generated, 1, 1, 1 and 0 without a resonance). coll_cnt
        has z and t of sborncol_fsr, from the energies in the rest frame of
        the resonance and t = z M_K^2 / 4.

Run from anywhere with a gfortran in the PATH:

    python madspace/tests/fks_reference/make_fks_resonance_reference.py
        --mg-root PATH [--keep DIR] [--list NAME]

--list NAME prints the decisions of set_granny for every Born configuration
of the first subprocess of NAME, to choose the channels.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.realpath(__file__))
OUT_DIR = os.path.join(HERE, "..", "test_data", "fks_resonance")
sys.path.insert(0, HERE)
import make_fks_reference as mfr  # noqa: E402

# name: the process, its run_card values, the subprocess, its channels and
# the numbers of points where they differ from the arguments.
# e+ e- > e+ e- u u~: the u u~ pair from a Z (resonant) or a photon, in
# s-channel diagrams and next to a t-channel; with ptj = 50 the Z cannot be on
# shell with two jets; e+ e- > e+ e- b b~: massive b quarks from a Z or a
# Higgs boson (resonant) or a photon. The xi map of a massive emitter of the
# branch differs from that of main for y < 0 (reported on PR 397), so the b
# quarks get more points at uniform random x, about half of them at y >= 0,
# and no extra points (all at y < 0).
PROCESSES = {
    "epem_epemuux": (
        "e+ e- > e+ e- u u~ [QCD]",
        {},
        "P0_epem_emepuux",
        [9, 11, 28, 43, 45],
        {},
    ),
    "epem_epemuux_ptj50": (
        "e+ e- > e+ e- u u~ [QCD]",
        {"ptj": 50.0},
        "P0_epem_emepuux",
        [11],
        {},
    ),
    "epem_epembbx": (
        "e+ e- > e+ e- b b~ [QCD]",
        {},
        "P0_epem_bbxemep",
        [1, 3, 9, 26],
        {"points": 16, "massive": 0},
    ),
}
# FKSISRMapping of the branch (final-state emitters do not depend on it)
ISR_MAPPING = 1


def dump(pdir, iconfig, points, massive, limit, log):
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
        input=f"{points}\n{iconfig}\n{massive}\n{limit}\n0\n{ISR_MAPPING}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def parse_resonance(text, data):
    """The entries of the resonance (NCONFIGS, FOREST, PROPS, GRANNY,
    RESONANCE, RESK, AOR, SCALES), added to the configurations and points that
    make_fks_reference.parse made of the same text."""
    configs = {conf["nfks"]: conf for conf in data["configs"]}
    forest, props = [], []
    conf, limit, counts = None, False, {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        key, rest = fields[0], fields[1:]
        if key == "NCONFIGS":
            data["iconfig"] = int(rest[1])
        elif key == "FOREST":
            forest.append([int(v) for v in rest])
        elif key == "PROPS":
            v = mfr.floats(rest)
            props = [v[k : k + 2] for k in range(0, len(v), 2)]
        elif key == "CONF":
            conf = configs[int(rest[0])]
        elif key == "PASS":
            limit = int(rest[0]) == 3
        elif key == "POINT":
            k = (conf["nfks"], limit)
            counts[k] = counts.get(k, 0) + 1
        elif key == "GRANNY":
            v = [int(f) for f in rest]
            conf["granny"] = dict(
                igranny=v[0],
                iaunt=v[1],
                granny_is_res=bool(v[2]),
                granny_chain_real_final=[bool(f) for f in v[3:]],
            )
        elif key == "RESONANCE":
            v = [int(f) for f in rest]
            conf["resonance"] = dict(
                resonance_recoil=bool(v[0]),
                initial_recoil_leg=v[1],
                resonance_members=[bool(f) for f in v[2:]],
            )
        elif key in ("RESK", "AOR", "SCALES"):
            points = conf["limit_points" if limit else "points"]
            point = points[counts[conf["nfks"], limit] - 1]
            v = mfr.floats(rest)
            if key == "RESK":
                point["resonance_mass2"] = v[0]
                point["resonance_momentum"] = v[1:5]
            elif key == "AOR":
                point["xij_aor"] = v
            else:
                point["resonance_scales"] = dict(
                    soft=v[0], collinear=v[1], angular=v[2], mismatch_log=v[3]
                )
    data["born_forest"] = forest
    data["born_props"] = props


def fortran_born_blocks(data):
    """The Born blocks of every point where the event is generated, from the
    Fortran of the branch (make_fks_reference.fortran_orders): the Born of
    amp_split_cnt and the color-correlated Borns from the color-linked Borns
    of sborn_sf, summed over the Born orders. The emitters of PROCESSES are
    quarks, so there are no spin-correlated Borns. pyAmpliCol is not used
    since MadGraph drops the widths of t-channel propagators
    (zerowidth_tchannel), here the Z of the t-channel diagrams."""
    data["born_blocks"] = dict(
        source="MadFKS of the branch per Born order: the Born of amp_split_cnt "
        "and the color-linked Borns of sborn_sf, see fortran_orders",
        alpha_s=data["g"] ** 2 / (4 * math.pi),
        top_width=0.0,
    )
    for conf in data["configs"]:
        for point in conf["points"] + conf["limit_points"]:
            if "born" not in point:
                continue
            per_order = mfr.fortran_orders(data, conf, point)
            blocks = dict(
                born=math.fsum(b["born"] for b in per_order),
                color_correlated=[
                    [math.fsum(column) for column in zip(*rows)]
                    for rows in zip(*(b["color_correlated"] for b in per_order))
                ],
            )
            if len(per_order) > 1:
                blocks["orders"] = per_order
            point["born_blocks"] = blocks


def run_card_cuts(out):
    """The cuts of the run_card that set_tau_min reads, from Source/run_card.inc,
    and maxjetflavor at least NF of the model, as setcuts.f sets it."""
    with open(os.path.join(out, "Source", "run_card.inc")) as f:
        text = f.read()

    def value(name, index=None):
        key = re.escape(name if index is None else f"{name}({index})")
        match = re.search(rf"^\s*{key}\s*=\s*(\S+)", text, re.MULTILINE | re.IGNORECASE)
        if match is None:
            raise ValueError(f"{name} not found in run_card.inc")
        raw = match.group(1)
        if raw.upper() in (".TRUE.", ".FALSE."):
            return raw.upper() == ".TRUE."
        return float(raw.replace("D", "E").replace("d", "e"))

    with open(os.path.join(out, "Source", "MODEL", "coupl.inc")) as f:
        nf = float(
            re.search(r"PARAMETER\s*\(\s*NF\s*=\s*(\S+?)\)", f.read(), re.IGNORECASE)
            .group(1)
            .replace("D", "E")
        )
    pdg_cuts = []
    if value("PDG_CUT", 1) != 0:
        for k in range(1, int(value("PDG_CUT", 0)) + 1):
            pdg_cuts.append(
                dict(
                    pdg=int(value("PDG_CUT", k)),
                    ptmin=value("PTMIN4PDG", k),
                    mxxmin=value("MXXMIN4PDG", k),
                    part_antipart=value("MXXPART_ANTIPART", k),
                )
            )
    return dict(
        ptj=value("PTJ"),
        ptgmin=value("PTGMIN"),
        ptl=value("PTL"),
        mll=value("MLL"),
        mll_sf=value("MLL_SF"),
        maxjetflavor=max(int(value("MAXJETFLAVOR")), int(nf)),
        gamma_is_j=value("GAMMA_IS_J"),
        pdg_cuts=pdg_cuts,
    )


def generate(mg_root, name, process, work, run_card):
    mfr.MG_ROOT = mg_root
    return mfr.generate(name, process, work, run_card or None)


def list_channels(pdir, work):
    """The decisions of set_granny for every Born configuration."""
    text = dump(pdir, 1, 1, 0, 0, os.path.join(work, "list.log"))
    count = int(re.search(r"^NCONFIGS (\d+)", text, re.MULTILINE).group(1))
    for iconfig in range(1, count + 1):
        text = dump(pdir, iconfig, 1, 0, 0, os.path.join(work, "list.log"))
        data = mfr.parse(text)
        parse_resonance(text, data)
        grannies = [
            (c["nfks"], c["granny"]["igranny"], c["granny"]["granny_is_res"])
            for c in data["configs"]
        ]
        print(iconfig, data["born_forest"], grannies)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mg-root", required=True, help="checkout of the MCcntRefactor_Sfun branch"
    )
    parser.add_argument(
        "--main-root",
        default=os.path.realpath(os.path.join(HERE, "..", "..", "..")),
        help="MadGraph7 with the FKS exporter that writes fks_configurations.json",
    )
    parser.add_argument("--points", type=int, default=4, help="points per FKS config")
    parser.add_argument("--massive", type=int, default=8)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--keep", help="keep the generated outputs in this directory")
    parser.add_argument("--list", help="list the channels of this process")
    args = parser.parse_args()

    mg_root = os.path.realpath(args.mg_root)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=mg_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    os.makedirs(OUT_DIR, exist_ok=True)
    work = args.keep or tempfile.mkdtemp(prefix="fks_resonance_reference_")
    for sub in ("branch", "main"):
        os.makedirs(os.path.join(work, sub), exist_ok=True)
    names = [args.list] if args.list else list(PROCESSES)
    main_outputs = {}
    for name in names:
        process, run_card, sub, iconfigs, counts = PROCESSES[name]
        points = counts.get("points", args.points)
        massive = counts.get("massive", args.massive)
        out = generate(mg_root, name, process, os.path.join(work, "branch"), run_card)
        pdir = os.path.join(out, "SubProcesses", sub)
        if args.list:
            list_channels(pdir, work)
            continue
        # the configurations do not depend on the run_card
        if process not in main_outputs:
            main_outputs[process] = generate(
                os.path.realpath(args.main_root),
                name,
                process,
                os.path.join(work, "main"),
                None,
            )
        main_pdir = os.path.join(main_outputs[process], "SubProcesses", sub)
        with open(os.path.join(main_pdir, "fks_configurations.json")) as f:
            fks_configurations = json.load(f)
        cuts = run_card_cuts(out)
        for iconfig in iconfigs:
            log = os.path.join(work, f"{name}_{sub}_c{iconfig}.log")
            text = dump(pdir, iconfig, points, massive, args.limit, log)
            parsed = mfr.parse(text)
            parse_resonance(text, parsed)
            # everything make_fks_reference.parse reads, the subtraction too
            data = {key: value for key, value in parsed.items() if key != "configs"}
            data.update(
                process=process,
                subprocess=sub,
                iconfig=parsed["iconfig"],
                born_forest=parsed["born_forest"],
                born_props=parsed["born_props"],
                cuts=cuts,
                mg_branch="MCcntRefactor_Sfun",
                mg_commit=commit,
                jac_sampling_factors=True,
                fks_configurations=fks_configurations,
            )
            entries = fks_configurations["configurations"]
            configs = []
            for conf, entry in zip(parsed["configs"], entries, strict=True):
                if [conf["i_fks"], conf["j_fks"]] != [entry["i_fks"], entry["j_fks"]]:
                    raise ValueError(f"{name} {sub}: the configurations differ")
                conf = dict(conf)
                conf.setdefault("limit_points", [])
                configs.append(conf)
            data["configs"] = configs
            fortran_born_blocks(data)
            path = os.path.join(OUT_DIR, f"{name}_{sub}_c{iconfig}.json")
            with open(path, "w") as f:
                json.dump(data, f, separators=(",", ":"))
                f.write("\n")
            npts = sum(len(c["points"]) + len(c["limit_points"]) for c in configs)
            resonant = [
                c["nfks"] for c in configs if c["resonance"]["resonance_recoil"]
            ]
            print(
                f"{path}: {len(configs)} FKS configs, {npts} points, resonant {resonant}"
            )
    if not args.keep:
        shutil.rmtree(work)


if __name__ == "__main__":
    main()
