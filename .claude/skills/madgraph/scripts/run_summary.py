#!/usr/bin/env python3
"""Summarise a finished MadGraph run: cross section, MC error, events, seed,
scale/PDF bands and output files.

    run_summary.py <process_dir> [run_name]
    run_summary.py --combine <process_dir> <run_name> <run_name> ...

Without a run name the most recent run in <process_dir>/Events is used.
--combine merges independent runs of the same setup (different seeds) into
one cross section with an honest error.
Handles mg7 output (Events/<run>/info.json), madevent output
(Events/<run>/<run>_tag_*_banner.txt plus the LHE <init> block) and aMC@NLO
output (summary.txt, banner, and the scale/PDF weights of NLO events).
Exits 1 when the run is missing, unfinished or empty.
Standard library only: any python3 runs it.
"""

import glob
import gzip
import json
import os
import re
import sys


def fail(msg):
    print("ERROR: " + msg)
    sys.exit(1)


def pick_run(proc_dir, run_name):
    events = os.path.join(proc_dir, "Events")
    if not os.path.isdir(events):
        fail("no Events directory in %s: nothing was launched yet" % proc_dir)
    if run_name:
        path = os.path.join(events, run_name)
        if not os.path.isdir(path):
            fail("no run %s in %s" % (run_name, events))
        return path
    runs = [p for p in glob.glob(os.path.join(events, "*")) if os.path.isdir(p)]
    if not runs:
        fail("no run in %s" % events)
    return max(runs, key=os.path.getmtime)


def run_card_text(proc_dir, run_dir):
    """The run_card.toml this run used, from the <MG7RunCard> block of its
    LHE header (header.lhe for npy output, events.lhe.gz otherwise). Falls
    back to the current Cards/run_card.toml, which a later launch may have
    changed: the second return value says which one it is."""
    for name in ("header.lhe", "events.lhe.gz", "unweighted_events.lhe.gz"):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        opener = gzip.open if name.endswith(".gz") else open
        block, inside = [], False
        with opener(path, "rt") as fsock:
            for line in fsock:
                if "<MG7RunCard>" in line:
                    inside = True
                elif "</MG7RunCard>" in line or "</header>" in line:
                    break
                elif inside:
                    block.append(line)
        if block:
            return "".join(block), True
    try:
        with open(os.path.join(proc_dir, "Cards", "run_card.toml")) as fsock:
            return fsock.read(), False
    except OSError:
        return "", False


def run_card_value(card, section, key):
    """Value of [section] key in a run_card.toml text (simple parser)."""
    current = None
    for line in card.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("["):
            current = line.strip("[]")
        elif current == section and "=" in line:
            name, value = (x.strip() for x in line.split("=", 1))
            if name == key:
                return value.strip('"')
    return None


def run_card_section(card, section):
    """The 'key = value' lines of one [section] of a run_card.toml text."""
    lines, current = [], None
    for line in card.splitlines():
        line = line.split("#", 1)[0].strip()
        if line.startswith("["):
            current = line.strip("[]")
        elif current == section and "=" in line:
            lines.append(" ".join(line.split()))
    return lines


# the particle families the cuts are written for
FAMILIES = {
    "jet": {"g", "u", "d", "s", "c", "u~", "d~", "s~", "c~"},
    "bottom": {"b", "b~"},
    "lepton": {"e+", "e-", "mu+", "mu-", "ta+", "ta-"},
    "missing": {"ve", "vm", "vt", "ve~", "vm~", "vt~"},
    "photon": {"a"},
}
UNSTABLE = {"t": "top", "w": "W", "z": "Z", "h": "Higgs"}
# PDG codes, as written in the 'define' lines of a 5-flavour proc card
PDG_NAMES = {1: "d", 2: "u", 3: "s", 4: "c", 5: "b", 6: "t", 11: "e-",
             12: "ve", 13: "mu-", 14: "vm", 15: "ta-", 16: "vt", 21: "g",
             22: "a", 23: "z", 24: "w+", 25: "h"}
KNOWN = set().union(*FAMILIES.values()) | {"t", "t~", "w+", "w-", "z", "h"}


def final_state(proc_dir):
    """Names of the final-state particles of the generated processes (decay
    chains resolved, multiparticles expanded with the 'define' lines of
    Cards/proc_card_mg5.dat), or None when the card cannot be read."""
    try:
        with open(os.path.join(proc_dir, "Cards", "proc_card_mg5.dat")) as fsock:
            lines = fsock.read().splitlines()
    except OSError:
        return None
    defines, processes = {}, []
    for line in lines:
        line = line.split("#", 1)[0].strip()
        words = line.split()
        if not words:
            continue
        if words[0] == "define" and "=" in line:
            name, rest = line[len("define"):].split("=", 1)
            defines[name.strip()] = rest.split()
        elif words[0] == "generate":
            processes.append(line.split(None, 1)[1])
        elif words[:2] == ["add", "process"]:
            processes.append(line.split(None, 2)[2])
    names = set()
    for proc in processes:
        proc = re.sub(r"\[.*?\]|--\S+|@\S*", " ", proc).replace("(", ",")
        proc = proc.replace(")", ",")
        final = []
        for index, chunk in enumerate(c for c in proc.split(",") if ">" in c):
            mother, products = chunk.split(">", 1)[0], chunk.rsplit(">", 1)[1]
            products = re.split(r"\$|/", products)[0].split()
            products = [w for w in products if not re.search(r"[=<^]", w)]
            if index and mother.split() and mother.split()[0] in final:
                final.remove(mother.split()[0])
            final.extend(products)
        for name in final:
            names.update(expand(name, defines))
    return names


def expand(name, defines, depth=0):
    """A particle or multiparticle name as particle names: nested defines
    ('define j = p') and PDG codes ('define p = 21 2 ... 5') resolved."""
    if name in defines and depth < 10:
        out = set()
        for member in defines[name]:
            out |= expand(member, defines, depth + 1)
        return out
    try:
        code = int(name)
    except ValueError:
        return {name}
    base = PDG_NAMES.get(abs(code), name)
    return {base if code > 0 or base == name else
            base.replace("-", "+") if base.endswith("-") else
            base.replace("+", "-") if base.endswith("+") else base + "~"}


def families_of(names):
    """Cut families (jet, lepton, ...) plus the bare names of the other
    final-state particles (t, w, z, ...): what a cut group can match."""
    found = set()
    for name in names:
        for family, members in FAMILIES.items():
            if name in members:
                found.add(family)
                break
        else:
            found.add(name.rstrip("+-~"))
    return found


def pct(part, whole):
    return 100.0 * part / whole if whole else float("nan")


def summarise_mg7(proc_dir, run_dir):
    with open(os.path.join(run_dir, "info.json")) as fsock:
        info = json.load(fsock)
    proc = info.get("process", {})
    status = info.get("status")
    print("format      : mg7")
    print("status      : %s" % status)
    if status != "done" or not proc.get("done"):
        print("WARNING: the run did not finish (status %r)" % status)
    xsec, err = proc.get("mean"), proc.get("error")
    if xsec is None:
        fail("no cross section in info.json")
    print("cross section: %.6g +- %.3g pb  (MC rel. error %.2f%%)"
          % (xsec, err, pct(err, abs(xsec))))
    print("events      : %d unweighted" % proc.get("count_unweighted", 0))
    print("seed        : %s" % info.get("seed"))
    card, from_run = run_card_text(proc_dir, run_dir)
    if not from_run:
        print("WARNING: settings below are from the CURRENT Cards/run_card.toml, "
              "which may differ from what this run used")
    value = lambda key, section="beam": run_card_value(card, section, key)
    if str(value("leptonic")).lower() == "true":
        print("collider    : lepton, sqrt(s) = %s GeV" % value("e_cm"))
    else:
        print("collider    : hadron, sqrt(s) = %s GeV, PDF %s"
              % (value("e_cm"), value("pdf")))
    if str(value("fixed_ren_scale")).lower() == "true":
        print("scales      : fixed, mu_R = %s GeV, mu_F = %s / %s GeV"
              % (value("ren_scale"), value("fact_scale1"), value("fact_scale2")))
    else:
        factor = value("scale_factor")
        print("scales      : dynamical, %s%s"
              % (value("dynamical_scale_choice"),
                 "" if factor in (None, "1.0", "1") else " x %s" % factor))
    print_mg7_cuts(proc_dir, run_card_section(card, "cuts"))
    syst = info.get("systematics") or {}
    scale = syst.get("scale")
    if scale and xsec:
        mur, muf = value("mur", "systematics"), value("muf", "systematics")
        together = str(value("together", "systematics")).lower() == "true"
        print("scale band  : +%.1f%% -%.1f%%  (%.4g .. %.4g pb; envelope of "
              "central + %d variations, mu_R in %s, mu_F in %s, %s)"
              % (pct(scale["max"] - xsec, xsec), pct(xsec - scale["min"], xsec),
                 scale["min"], scale["max"], len(scale.get("weight_ids", [])),
                 mur, muf, "varied together and independently" if together
                 else "one at a time"))
    for pdf in syst.get("pdf", []):
        method = {"replicas": "standard deviation over the replicas",
                  "hessian": "Hessian eigenvector sum in quadrature"}.get(
                      pdf.get("error_type"), pdf.get("error_type"))
        print("PDF band    : +%.2f%% -%.2f%%  (%s, %s)"
              % (pct(pdf["uncertainty_up"], xsec), pct(pdf["uncertainty_down"], xsec),
                 pdf.get("pdf_set"), method))
    if not scale and not syst.get("pdf"):
        print("bands       : none (systematics off, or no events generated)")


def print_mg7_cuts(proc_dir, cuts):
    """Cuts whose particle groups all occur in the final state (the others
    cannot act and must not be reported as part of the setup)."""
    names = final_state(proc_dir)
    if names is None:
        print("cuts        : %s  (process unknown: some may not act)"
              % ("; ".join(cuts) or "none"))
        return
    present = families_of(names)
    unknown = sorted(n for n in names if n not in KNOWN)
    active, inactive = [], []
    for cut in cuts:
        if is_off(cut.split("=", 1)[1]):
            continue
        key = cut.split("=", 1)[0].strip().rsplit(".", 1)[0]
        groups = [g for g in key.split("-")[:-1] if g != "sum"]
        (active if set(groups) <= present else inactive).append(cut)
    print("cuts        : %s" % ("; ".join(active) or "none"))
    print_inactive([c.split("=")[0].strip() for c in inactive], unknown)
    print_stable(present)


def print_inactive(keys, unknown):
    if not keys:
        return
    if unknown:
        print("              possibly not acting (but the final state has "
              "particles this script cannot classify: %s): %s"
              % (" ".join(unknown), ", ".join(keys)))
    else:
        print("              not acting (no such particle in the final state): %s"
              % ", ".join(keys))


def is_off(value):
    """A cut set to 0 / -1 (a minimum of 0, or 'no cut') does nothing."""
    try:
        return float(value) in (0.0, -1.0)
    except ValueError:
        return False


def print_stable(present):
    stable = [UNSTABLE[n] for n in sorted(present) if n in UNSTABLE]
    if stable:
        print("note        : %s left undecayed (stable, no branching ratio)"
              % ", ".join(stable))


def lhe_init_error(run_dir):
    """Sum of XSECUP and XERRUP (quadrature) from the LHE <init> block."""
    files = sorted(glob.glob(os.path.join(run_dir, "*.lhe.gz"))
                   + glob.glob(os.path.join(run_dir, "*.lhe")))
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as fsock:
            lines = []
            for line in fsock:
                if "<init>" in line:
                    lines = []
                    continue
                if "</init>" in line:
                    break
                lines.append(line.split())
                if len(lines) > 1000:
                    break
        if len(lines) < 2:
            continue
        nproc = int(lines[0][-1])
        xsec = sum(float(row[0]) for row in lines[1:1 + nproc])
        err = sum(float(row[1]) ** 2 for row in lines[1:1 + nproc]) ** 0.5
        return xsec, err
    return None, None


def nlo_event_weights(path):
    """Cross sections of the scale and PDF variations of an aMC@NLO LHE file
    (event_norm = average): mean of each <wgt>, the nominal weight, and the
    fraction of negative events. Only the 'tag 0' scale group is the total;
    the other tags split it by coupling order."""
    groups, current = [], None
    sums, nominal, negative, count = {}, 0.0, 0, 0
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fsock:
        in_event = first = False
        for line in fsock:
            if "<weightgroup" in line:
                name = re.search(r"name=['\"]([^'\"]*)", line).group(1)
                current = (" ".join(name.split()), [])
                groups.append(current)
            elif "<weight " in line and current is not None:
                current[1].append(re.search(r"id=['\"]([^'\"]+)", line).group(1))
            elif line.startswith("<event"):
                in_event = first = True
                count += 1
            elif in_event and first:
                weight = float(line.split()[2])
                nominal += weight
                negative += weight < 0
                first = False
            elif in_event and "<wgt" in line:
                match = re.search(r"<wgt id=['\"]([^'\"]+)['\"]>\s*(\S+)", line)
                if match:
                    sums[match.group(1)] = sums.get(match.group(1), 0.0) \
                        + float(match.group(2))
            elif line.startswith("</event"):
                in_event = False
    if not count:
        return None
    means = {key: value / count for key, value in sums.items()}
    return {"count": count, "nominal": nominal / count,
            "negative": negative / count, "groups": groups, "means": means}


def summarise_amcatnlo(run_dir, banner):
    """Fixed-order and event runs of an aMC@NLO ([QCD]) output."""
    with open(banner) as fsock:
        text = fsock.read()

    def grab(key):
        match = re.search(r"^\s*(\S+)\s*=\s*%s\b" % key, text, re.M | re.I)
        return match.group(1).strip() if match else None

    events = sorted(glob.glob(os.path.join(run_dir, "events*.lhe.gz")))
    print("format      : aMC@NLO, %s" % ("NLO events (MC@NLO)" if events
                                         else "fixed order"))
    summary = os.path.join(run_dir, "summary.txt")
    if not os.path.exists(summary):
        fail("no summary.txt in %s: the run did not finish" % run_dir)
    with open(summary) as fsock:
        stext = fsock.read()
    xsec, err = amcatnlo_xsec(run_dir)
    print("cross section: %.6g +- %.3g pb  (MC rel. error %.2f%%; quote this "
          "one)" % (xsec, err, pct(err, abs(xsec))))
    match = re.search(r"Number of events generated:\s*(\d+)", stext)
    if match and not events:
        print("events      : %s" % match.group(1))
    blocks = re.findall(r"^\s*(\S.*?):\s*\n\s*\S+ pb\s+\+([\d.]+)% -([\d.]+)%",
                        stext, re.M)
    for header, up, down in blocks:
        if header.lower().startswith("dynamical_scale_choice"):
            print("scale band  : +%s%% -%s%%  (%s; percentages computed from the "
                  "histograms, apply them to the cross section above)"
                  % (up, down, header))
        else:
            print("PDF band    : +%s%% -%s%%  (%s)" % (up, down, header))
    print("seed        : %s  (the card is reset to 0 after a run; this is the "
          "one used)" % grab("iseed"))
    proc_dir = os.path.dirname(os.path.dirname(run_dir.rstrip(os.sep)))
    name = pdf_name(proc_dir, grab("lhaid"))
    print("collider    : lpp %s/%s, ebeam %s + %s GeV, PDF lhaid %s%s"
          % (grab("lpp1"), grab("lpp2"), grab("ebeam1"), grab("ebeam2"),
             grab("lhaid"), " (%s)" % name if name else ""))
    if str(grab("fixed_ren_scale")).lower() in ("true", "t", ".true."):
        print("scales      : fixed, mu_R = %s, mu_F = %s GeV"
              % (grab("muR_ref_fixed"), grab("muF_ref_fixed")))
    else:
        print("scales      : dynamical_scale_choice %s (-1 = default = H_T/2, "
              "half the sum of the final-state transverse masses), x muR %s, "
              "muF %s"
              % (grab("dynamical_scale_choice"), grab("muR_over_ref"),
                 grab("muF_over_ref")))
    families = {"jet": ("ptj", "etaj", "jetalgo", "jetradius"),
                "lepton": ("ptl", "etal", "drll", "drll_sf", "mll", "mll_sf"),
                "photon": ("ptgmin", "etagamma", "R0gamma")}
    names = final_state(os.path.dirname(os.path.dirname(run_dir.rstrip(os.sep))))
    present = families_of(names) if names is not None else set(families)
    unknown = sorted(n for n in names or () if n not in KNOWN)
    active, inactive = [], []
    for family, keys in families.items():
        for key in keys:
            value = grab(key)
            if value is not None and not (is_off(value) and key not in
                                          ("jetalgo", "jetradius")):
                (active if family in present else inactive).append(
                    "%s = %s" % (key, value))
    print("cuts        : %s" % ("; ".join(active) or "none"))
    if "jet" in present and any(c.startswith("ptj") for c in active):
        print("              (jet cuts: every Born-level jet must pass, jets "
              "clustered with jetalgo (1 kT, -1 anti-kT) / jetradius)")
    print_inactive([c.split("=")[0].strip() for c in inactive], unknown)
    if names is not None:
        print_stable(present)
    print("precision   : req_acc_FO %s, req_acc %s"
          % (grab("req_acc_FO"), grab("req_acc")))
    for path in events:
        info = nlo_event_weights(path)
        if not info:
            continue
        xsec = info["nominal"]
        print("events      : %d in %s, %.1f%% with negative weight"
              % (info["count"], os.path.basename(path), 100 * info["negative"]))
        for name, ids in info["groups"]:
            values = [info["means"][i] for i in ids if i in info["means"]]
            if not values:
                continue
            if name.startswith("scale_variation") and name.split()[1] == "0":
                print("scale band  : +%.1f%% -%.1f%%  (envelope of %d muR x muF "
                      "points, from the event weights)"
                      % (pct(max(values) - xsec, xsec),
                         pct(xsec - min(values), xsec), len(values)))
            elif name.startswith("PDF_variation") and len(values) > 2:
                members = values[1:]
                mean = sum(members) / len(members)
                std = (sum((v - mean) ** 2 for v in members)
                       / (len(members) - 1)) ** 0.5
                print("PDF members : %s, %d members, standard deviation "
                      "%.2f%% (right for replica sets such as NNPDF; Hessian "
                      "sets need their own prescription)"
                      % (name.split(None, 1)[-1], len(members), pct(std, xsec)))


MADEVENT_SCALES = {"-1": "default: CKKW back-clustering of the event",
                   "0": "user-defined", "1": "sum of E_T",
                   "2": "sum of transverse masses (H_T)",
                   "3": "H_T/2", "4": "sqrt(s-hat)"}
# madevent cut names, by the particle families they need
MADEVENT_CUTS = [
    (("jet",), ("ptj", "etaj", "drjj", "mmjj", "xptj", "ptjmax", "mmjjmax")),
    (("bottom",), ("ptb", "etab", "drbb", "mmbb", "xptb", "ptbmax")),
    (("lepton",), ("ptl", "etal", "drll", "mmll", "xptl", "ptlmax", "mmllmax",
                   "ptllmin", "ptllmax")),
    (("missing",), ("misset", "missetmax")),
    (("photon",), ("pta", "etaa", "draa", "mmaa", "xpta", "ptamax", "ptgmin")),
    (("bottom", "jet"), ("drbj",)), (("photon", "jet"), ("draj",)),
    (("jet", "lepton"), ("drjl",)), (("photon", "bottom"), ("drab",)),
    (("bottom", "lepton"), ("drbl",)), (("photon", "lepton"), ("dral",)),
]


def lhapdf_configs(proc_dir):
    """Candidate lhapdf-config executables, most specific first: the process
    directory, this installation, the user's MadGraph7 config, then PATH."""
    mg_root = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                           "..", "..", "..", ".."))
    files = [os.path.join(proc_dir, "Cards", card) for card in
             ("me5_configuration.txt", "amcatnlo_configuration.txt")]
    files += [os.path.join(mg_root, "input", "mg7_configuration.txt"),
              os.path.expanduser("~/.mg7/mg7_configuration.txt")]
    for path in files:
        try:
            with open(path) as fsock:
                for line in fsock:
                    line = line.split("#", 1)[0].strip()
                    if "=" in line and line.split("=")[0].strip() in (
                            "lhapdf", "lhapdf_py3"):
                        value = line.split("=", 1)[1].split()
                        if value:
                            yield value[0]
        except OSError:
            continue
    yield "lhapdf-config"


def reweight_summary(run_dir, quoted_xsec=None):
    """New cross sections of the reweighting weights (<weightgroup
    name='mg_reweighting'>) found in the run's LHE files: mean of each new
    weight, ratio to the mean original weight, spread of the per-event
    ratios and the effective number of events."""
    for name in ("unweighted_events.lhe.gz", "events.lhe.gz"):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        ids, in_group, header = [], False, True
        sums, sum_w0, count = {}, 0.0, 0
        stats = {}
        in_event = first = False
        current = {}
        with gzip.open(path, "rt") as fsock:
            for line in fsock:
                if header:
                    if "<weightgroup" in line and "mg_reweighting" in line:
                        in_group = True
                    elif in_group and "<weight id=" in line:
                        ids.append(re.search(r"id=['\"]([^'\"]+)", line).group(1))
                    elif "</weightgroup>" in line:
                        in_group = False
                    elif "</header>" in line or "<init>" in line:
                        header = False
                        if not ids:
                            break
                    continue
                if line.startswith("<event"):
                    in_event = first = True
                    current = {}
                elif in_event and first:
                    w0 = float(line.split()[2])
                    first = False
                elif in_event and "<wgt id=" in line:
                    match = re.search(r"<wgt id=['\"]([^'\"]+)['\"]>\s*(\S+)", line)
                    if match and match.group(1) in ids:
                        current[match.group(1)] = float(match.group(2))
                elif line.startswith("</event"):
                    in_event = False
                    count += 1
                    sum_w0 += w0
                    for key, value in current.items():
                        st = stats.setdefault(key, [0.0, 0.0, float("inf"), -float("inf"), 0])
                        st[0] += value
                        st[1] += value * value
                        if w0:
                            ratio = value / w0
                            st[2] = min(st[2], ratio)
                            st[3] = max(st[3], ratio)
                        st[4] += 1
        if not ids or not count:
            continue
        mean0 = sum_w0 / count
        print("reweighting : %s (%d events)" % (os.path.join(os.path.basename(run_dir), name), count))
        for key in ids:
            if key not in stats:
                print("  %-20s missing from the events" % key)
                continue
            total, total2, rmin, rmax, n = stats[key]
            neff = total * total / total2 if total2 else 0
            ratio = total / count / mean0 if mean0 else float("nan")
            line = "  %-20s sigma = %.6g pb (ratio to original %.5f" % (key, total / count, ratio)
            if quoted_xsec:
                line += ", x quoted sigma = %.6g pb" % (ratio * quoted_xsec)
            print(line + "); per-event new/old in [%.3g, %.3g], N_eff = %.0f" % (rmin, rmax, neff))
        print("              quote the ratio times the run's cross section; check the "
              "validity conditions (reference/reweight.md)")
        return


def pdf_name(proc_dir, lhaid):
    """LHAPDF set name of an lhaid, from the pdfsets.index of the LHAPDF
    MadGraph uses for this process directory (None if not found)."""
    import subprocess
    for config in lhapdf_configs(proc_dir):
        try:
            datadir = subprocess.run([config, "--datadir"], capture_output=True,
                                     text=True, timeout=20).stdout.strip()
            with open(os.path.join(datadir, "pdfsets.index")) as index:
                for row in index:
                    words = row.split()
                    if words and words[0] == str(lhaid).strip():
                        return words[1]
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def print_madspin_card(run_dir):
    """Spin mode, seed and decays of a MadSpin output, from the <madspin>
    block of its LHE header, plus the number of events."""
    path = os.path.join(run_dir, "unweighted_events.lhe.gz")
    if not os.path.exists(path):
        return
    header, count = [], 0
    with gzip.open(path, "rt") as fsock:
        for line in fsock:
            if count == 0 and "<event" not in line:
                header.append(line)
            if line.startswith("<event"):
                count += 1
    match = re.search(r"<madspin>(.*?)</madspin>", "".join(header), re.S)
    print("events      : %d" % count)
    if not match:
        return
    options, decays = {}, []
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        words = line.split()
        if len(words) >= 3 and words[0] == "set":
            options[words[1].lower()] = " ".join(words[2:])
        elif words[:1] == ["decay"]:
            decays.append(line[len("decay"):].strip())
    print("MadSpin     : spinmode %s, seed %s, BW_cut %s, decay_generator %s"
          % (options.get("spinmode", "madspin (default)"),
             options.get("seed", "not set (random)"),
             options.get("bw_cut", "from the run card"),
             options.get("decay_generator", "mg7 (default)")))
    for decay in decays:
        print("  decay     : %s" % decay)
    if options.get("decay_generator", "mg7") == "mg7":
        print("              decays are NOT reproducible with decay_generator mg7, "
              "even with a seed (use decay_generator madevent)")


def madevent_numbers(run_dir, banner):
    """Cross section, MC error and event count of a madevent run (the last
    block wins: a re-showered run's banner repeats the earlier ones)."""
    with open(banner) as fsock:
        text = fsock.read()

    def grab(pattern):
        found = re.findall(pattern, text, re.M)
        return found[-1].strip() if found else None

    xsec = grab(r"^#\s*Integrated weight \(pb\)\s*:\s*(\S+)")
    nevt = grab(r"Number of Events\s*:\s*(\S+)")
    init_xsec, err = lhe_init_error(run_dir)
    if xsec is None:
        # a MadSpin run_XX_decayed_N banner has no summary: the LHE <init>
        # block carries the cross section, branching ratios included
        xsec = init_xsec
    return (float(xsec) if xsec is not None else None), err, nevt, grab


def summarise_madevent(proc_dir, run_dir, banner):
    decayed = "_decayed_" in os.path.basename(run_dir)
    print("format      : %s" % ("MadSpin output (LHE)" if decayed else "madevent"))
    xsec, err, nevt, grab = madevent_numbers(run_dir, banner)
    if xsec is None:
        fail("no cross section in %s: did the run finish?" % banner)
    if decayed:
        print("note        : MadSpin-decayed sample, cross section includes "
              "the branching ratios (and, in the density spin modes, the "
              "Breit-Wigner truncation at BW_cut)")
        print_madspin_card(run_dir)
    card = lambda key: grab(r"^\s*(\S+)\s*=\s*%s\b" % key)
    mlm = card("ickkw") == "1"
    if err is not None:
        print("cross section: %.6g +- %.3g pb  (MC rel. error %.2f%%%s)"
              % (xsec, err, pct(err, abs(xsec)),
                 "; parton level, before the MLM veto" if mlm else ""))
    else:
        print("cross section: %.6g pb  (MC error not found)" % xsec)
    if nevt is not None:
        print("events      : %s%s" % (nevt, " (parton level, before the MLM veto)"
                                      if mlm else ""))
    if card("iseed") is not None:
        print("seed        : %s" % card("iseed"))
    if card("lpp1") is None:
        return
    beams = "%s + %s GeV" % (card("ebeam1"), card("ebeam2"))
    if card("lpp1") == "0" and card("lpp2") == "0":
        print("collider    : lepton, %s" % beams)
    else:
        lhaid = card("lhaid")
        name = pdf_name(proc_dir, lhaid) if card("pdlabel") == "lhapdf" else None
        print("collider    : lpp %s/%s, %s, PDF %s"
              % (card("lpp1"), card("lpp2"), beams,
                 "%s (%s)" % (lhaid, name) if name else
                 lhaid if card("pdlabel") == "lhapdf" else card("pdlabel")))
    if str(card("fixed_ren_scale")).lower() in ("true", "t"):
        print("scales      : fixed, mu_R = %s GeV, mu_F = %s / %s GeV"
              % (card("scale"), card("dsqrt_q2fact1"), card("dsqrt_q2fact2")))
    else:
        choice = card("dynamical_scale_choice")
        factor = card("scalefact")
        print("scales      : dynamical_scale_choice %s (%s)%s"
              % (choice, MADEVENT_SCALES.get(choice, "see the run card"),
                 "" if factor in (None, "1.0", "1") else " x %s" % factor))
    names = final_state(proc_dir)
    present = families_of(names) if names is not None else None
    unknown = sorted(n for n in names or () if n not in KNOWN)
    active, inactive = [], []
    for families, keys in MADEVENT_CUTS:
        for key in keys:
            value = card(key)
            if value is None or is_off(value):
                continue
            acts = present is None or set(families) <= present
            (active if acts else inactive).append("%s = %s" % (key, value))
    print("cuts        : %s" % ("; ".join(active) or "none"))
    print_inactive([c.split("=")[0].strip() for c in inactive], unknown)
    if present is not None:
        print_stable(present)
    syst = os.path.join(run_dir, "parton_systematics.log")
    if os.path.exists(syst):
        with open(syst) as fsock:
            stext = fsock.read()
        for label, key in (("scale band", "scale variation"),
                           ("emission sc.", "emission scale variation"),
                           ("PDF band", "PDF variation")):
            match = re.search(r"^#\s*" + key + r":\s*\+([\d.]+)% -([\d.]+)%",
                              stext, re.M)
            if match:
                print("%-12s: +%s%% -%s%%  (parton_systematics.log%s)"
                      % (label, match.group(1), match.group(2),
                         "; before the MLM veto, quote as relative uncertainties"
                         if mlm else ""))
    if mlm:
        summarise_mlm(run_dir, xsec, err, nevt, card)


def pythia8_setting(run_dir, tag, key):
    try:
        with open(os.path.join(run_dir, "%s_pythia8.cmd" % tag)) as fsock:
            for line in fsock:
                if line.split("!")[0].strip().lower().startswith(key.lower()):
                    value = line.split("=", 1)[1].split("!")[0].strip()
                    try:
                        return "%g" % float(value)
                    except ValueError:
                        return value
    except OSError:
        pass
    return None


def mlm_results(run_dir, xsec, err, nevt):
    """One entry per shower (tag) of an MLM run: qCut, matched cross section
    with its error, fraction and number of events kept by the veto."""
    banners = glob.glob(os.path.join(run_dir, "*_tag_*_banner.txt"))
    banners.sort(key=lambda b: int(re.search(r"_tag_(\d+)_banner", b).group(1)))
    results = []
    for banner in banners:
        tag = "tag_" + re.search(r"_tag_(\d+)_banner", banner).group(1)
        with open(banner) as fsock:
            found = re.findall(r"Matched Integrated weight \(pb\)\s*:\s*(\S+)",
                               fsock.read())
        if not found:
            continue
        matched = float(found[-1])
        kept = matched / xsec if xsec else float("nan")
        rel = (err / xsec) ** 2 if err else 0.0
        if nevt and 0 < kept < 1:
            rel += (1 - kept) / (kept * float(nevt))
        results.append({
            "tag": tag, "matched": matched, "error": matched * rel ** 0.5,
            "kept": kept,
            "events": int(round(kept * float(nevt))) if nevt else None,
            "qcut": pythia8_setting(run_dir, tag, "JetMatching:qCut"),
            "njetmax": pythia8_setting(run_dir, tag, "JetMatching:nJetMax")})
    return results


def summarise_mlm(run_dir, xsec, err, nevt, card):
    """MLM-merged madevent run: the cross section after the Pythia8 veto,
    for every shower (tag) of the run, i.e. a merging-scale scan."""
    print("MLM         : ickkw 1, xqcut %s GeV, maxjetflavor %s, auto_ptj_mjj %s"
          % (card("xqcut"), card("maxjetflavor"), card("auto_ptj_mjj")))
    results = mlm_results(run_dir, xsec, err, nevt)
    for res in results:
        print("matched     : %.6g +- %.3g pb  (%s: qCut %s, nJetMax %s; %.1f%% "
              "kept by the veto = %s merged events in %s_pythia8_events.hepmc.gz)"
              % (res["matched"], res["error"], res["tag"], res["qcut"],
                 res["njetmax"], 100 * res["kept"], res["events"], res["tag"]))
    if results:
        print("              quote the matched (merged) cross section, not the "
              "parton-level one; identify the showers by their qCut, not by "
              "their tag number")
    else:
        print("matched     : none (no Pythia8 shower yet: the parton-level "
              "number above is not the merged cross section)")


def amcatnlo_xsec(run_dir):
    """Total cross section and MC error. The last res_<n>.txt has them at full
    precision (second line after 'Total:'); summary.txt only has 4 digits."""
    res = sorted(glob.glob(os.path.join(run_dir, "res_*.txt")),
                 key=lambda f: int(re.findall(r"res_(\d+)", f)[0]))
    if res:
        with open(res[-1]) as fsock:
            match = re.search(r"Total:\s*\n\s*\S+\s*\+-\s*\S+.*\n\s*(\S+)\s*\+-\s*(\S+)",
                              fsock.read())
        if match:
            return float(match.group(1)), float(match.group(2))
    with open(os.path.join(run_dir, "summary.txt")) as fsock:
        match = re.search(r"Total cross section:\s*(\S+)\s*\+-\s*(\S+)\s*pb",
                          fsock.read())
    if not match:
        fail("no total cross section in %s/summary.txt" % run_dir)
    return float(match.group(1)), float(match.group(2))


def run_xsec(proc_dir, run_dir):
    """{label: (cross section, MC error)} in pb for one run. One entry,
    except for an MLM run: one matched cross section per qCut."""
    if os.path.exists(os.path.join(run_dir, "info.json")):
        with open(os.path.join(run_dir, "info.json")) as fsock:
            proc = json.load(fsock).get("process", {})
        return {"": (proc["mean"], proc["error"])}
    if os.path.exists(os.path.join(run_dir, "summary.txt")):
        return {"": amcatnlo_xsec(run_dir)}
    banners = glob.glob(os.path.join(run_dir, "*_banner.txt"))
    if banners:
        xsec, err, nevt, grab = madevent_numbers(run_dir, sorted(banners)[-1])
        if grab(r"^\s*(\S+)\s*=\s*ickkw\b") == "1":
            results = mlm_results(run_dir, xsec, err, nevt)
            if not results:
                fail("%s is an MLM run without a shower: no matched cross "
                     "section to combine" % run_dir)
            return dict(("qCut %s" % r["qcut"], (r["matched"], r["error"]))
                        for r in results)
        if xsec is not None:
            return {"": (xsec, err)}
    fail("no cross section found in %s" % run_dir)


def combine_values(values):
    weights = [1.0 / err ** 2 for _, err in values]
    mean = sum(w * x for w, (x, _) in zip(weights, values)) / sum(weights)
    error = sum(weights) ** -0.5
    chi2 = sum(((x - mean) / e) ** 2 for x, e in values) / (len(values) - 1)
    plain = sum(x for x, _ in values) / len(values)
    spread = (sum((x - plain) ** 2 for x, _ in values) / (len(values) - 1)) ** 0.5
    return mean, error * max(1.0, chi2 ** 0.5), chi2, plain, spread


def combine(proc_dir, run_names):
    """Weighted mean of independent runs (different seeds), with the error
    scaled by sqrt(chi2/ndf) when the runs scatter more than their quoted
    errors say (PDG prescription), plus the plain mean for comparison. MLM
    runs are combined per qCut, using the matched cross sections."""
    groups = {}
    for name in run_names:
        for label, value in run_xsec(proc_dir, pick_run(proc_dir, name)).items():
            groups.setdefault(label, []).append(value)
            print("%-12s: %.6g +- %.3g pb%s"
                  % (name, value[0], value[1], "  (%s)" % label if label else ""))
    for label, values in sorted(groups.items()):
        if len(values) < 2:
            print("%s: only one run, nothing to combine" % (label or "result"))
            continue
        mean, error, chi2, plain, spread = combine_values(values)
        prefix = "%s: " % label if label else ""
        print("%scombined : %.6g +- %.3g pb  (weighted mean of %d runs; "
              "chi2/ndf = %.2f%s; quote this one)"
              % (prefix, mean, error, len(values), chi2,
                 ", error scaled by %.2f" % chi2 ** 0.5 if chi2 > 1 else ""))
        print("%splain mean: %.6g +- %.3g pb  (standard deviation of the runs "
              "%.3g)" % (prefix, plain, spread / len(values) ** 0.5, spread))
        if chi2 > 10:
            print("WARNING: the runs disagree far beyond their errors: check that "
                  "they differ only in their seed (energy, PDF, cuts, scales)")


def main():
    if len(sys.argv) < 2:
        fail(__doc__.strip())
    if sys.argv[1] == "--combine":
        if len(sys.argv) < 5:
            fail(__doc__.strip())
        combine(sys.argv[2], sys.argv[3:])
        return
    proc_dir = sys.argv[1]
    run_dir = pick_run(proc_dir, sys.argv[2] if len(sys.argv) > 2 else None)
    print("run         : %s" % run_dir)
    banners = sorted(glob.glob(os.path.join(run_dir, "*_banner.txt")),
                     key=lambda b: int((re.findall(r"_tag_(\d+)_", b) or [0])[0]))
    if os.path.exists(os.path.join(run_dir, "info.json")):
        summarise_mg7(proc_dir, run_dir)
    elif banners and os.path.exists(os.path.join(proc_dir, "Cards",
                                                  "FKS_params.dat")):
        summarise_amcatnlo(run_dir, banners[-1])
    elif banners:
        summarise_madevent(proc_dir, run_dir, banners[-1])
    else:
        fail("neither info.json nor a banner in %s: the run did not finish"
             % run_dir)
    try:
        quoted = run_xsec(proc_dir, run_dir).get("")
    except SystemExit:
        quoted = None
    reweight_summary(run_dir, quoted[0] if quoted else None)
    files = sorted(os.listdir(run_dir))
    events = [f for f in files if f.startswith(("events", "unweighted_events"))
              or f.endswith((".lhe.gz", ".hepmc.gz", ".hepmc"))]
    print("event files : %s" % (", ".join(events) or "none"))
    plots = os.path.join(run_dir, "plots")
    if os.path.isdir(plots):
        print("plots       : %s (%d files)" % (plots, len(os.listdir(plots))))
    for extra in ("MADatLO.HwU", "MADatNLO.HwU", "MADatNLO.pdf",
                  "events.weights.json"):
        if extra in files:
            print("also        : %s" % os.path.join(run_dir, extra))
    for decayed in sorted(glob.glob(run_dir.rstrip(os.sep) + "_decayed_*")):
        print("decayed     : %s (MadSpin; summarise it with its run name)"
              % decayed)


if __name__ == "__main__":
    main()
