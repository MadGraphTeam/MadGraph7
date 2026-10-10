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
    print("collider    : lpp %s/%s, ebeam %s + %s GeV, PDF lhaid %s"
          % (grab("lpp1"), grab("lpp2"), grab("ebeam1"), grab("ebeam2"),
             grab("lhaid")))
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


def summarise_madevent(run_dir, banner):
    with open(banner) as fsock:
        text = fsock.read()
    print("format      : %s" % ("MadSpin output (LHE)"
                                  if "_decayed_" in os.path.basename(run_dir)
                                  else "madevent"))

    def grab(pattern):
        match = re.search(pattern, text, re.M)
        return match.group(1).strip() if match else None

    xsec = grab(r"Integrated weight \(pb\)\s*:\s*(\S+)")
    nevt = grab(r"Number of Events\s*:\s*(\S+)")
    init_xsec, err = lhe_init_error(run_dir)
    if xsec is None:
        # a MadSpin run_XX_decayed_N banner has no summary: the LHE <init>
        # block carries the cross section, branching ratios included
        xsec = init_xsec
    if xsec is None:
        fail("no cross section in %s: did the run finish?" % banner)
    xsec = float(xsec)
    if "_decayed_" in os.path.basename(run_dir):
        print("note        : MadSpin-decayed sample, cross section includes "
              "the branching ratios")
    if err is not None:
        print("cross section: %.6g +- %.3g pb  (MC rel. error %.2f%%)"
              % (xsec, err, pct(err, abs(xsec))))
    else:
        print("cross section: %.6g pb  (MC error not found)" % xsec)
    seed = grab(r"^\s*(\d+)\s*=\s*iseed")
    if nevt is not None:
        print("events      : %s" % nevt)
    if seed is not None:
        print("seed        : %s" % seed)
    for key in ("lpp1", "lpp2", "ebeam1", "ebeam2", "pdlabel", "lhaid",
                "dynamical_scale_choice", "fixed_ren_scale", "scale"):
        value = grab(r"^\s*(\S+)\s*=\s*%s\b" % key)
        if value is not None:
            print("%-12s: %s" % (key, value))


def amcatnlo_xsec(run_dir):
    with open(os.path.join(run_dir, "summary.txt")) as fsock:
        match = re.search(r"Total cross section:\s*(\S+)\s*\+-\s*(\S+)\s*pb",
                          fsock.read())
    if not match:
        fail("no total cross section in %s/summary.txt" % run_dir)
    return float(match.group(1)), float(match.group(2))


def run_xsec(proc_dir, run_dir):
    """(cross section, MC error) in pb of one run, whatever its format."""
    if os.path.exists(os.path.join(run_dir, "info.json")):
        with open(os.path.join(run_dir, "info.json")) as fsock:
            proc = json.load(fsock).get("process", {})
        return proc["mean"], proc["error"]
    if os.path.exists(os.path.join(run_dir, "summary.txt")):
        return amcatnlo_xsec(run_dir)
    xsec, err = lhe_init_error(run_dir)
    if xsec is None:
        fail("no cross section found in %s" % run_dir)
    return xsec, err


def combine(proc_dir, run_names):
    """Weighted mean of independent runs (different seeds), with the error
    scaled by sqrt(chi2/ndf) when the runs scatter more than their quoted
    errors say (PDG prescription), plus the plain mean for comparison."""
    values = []
    for name in run_names:
        xsec, err = run_xsec(proc_dir, pick_run(proc_dir, name))
        values.append((xsec, err))
        print("%-12s: %.6g +- %.3g pb" % (name, xsec, err))
    if len(values) < 2:
        fail("--combine needs at least two runs")
    weights = [1.0 / err ** 2 for _, err in values]
    mean = sum(w * x for w, (x, _) in zip(weights, values)) / sum(weights)
    error = sum(weights) ** -0.5
    chi2 = sum(((x - mean) / e) ** 2 for x, e in values) / (len(values) - 1)
    scale = max(1.0, chi2 ** 0.5)
    print("combined    : %.6g +- %.3g pb  (weighted mean; chi2/ndf = %.2f%s; "
          "quote this one)" % (mean, error * scale, chi2,
                               ", error scaled by %.2f" % scale if scale > 1
                               else ""))
    plain = sum(x for x, _ in values) / len(values)
    spread = (sum((x - plain) ** 2 for x, _ in values) / (len(values) - 1)) ** 0.5
    print("plain mean  : %.6g +- %.3g pb  (standard deviation of the runs %.3g)"
          % (plain, spread / len(values) ** 0.5, spread))


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
    banners = sorted(glob.glob(os.path.join(run_dir, "*_banner.txt")))
    if os.path.exists(os.path.join(run_dir, "info.json")):
        summarise_mg7(proc_dir, run_dir)
    elif banners and os.path.exists(os.path.join(proc_dir, "Cards",
                                                  "FKS_params.dat")):
        summarise_amcatnlo(run_dir, banners[-1])
    elif banners:
        summarise_madevent(run_dir, banners[-1])
    else:
        fail("neither info.json nor a banner in %s: the run did not finish"
             % run_dir)
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
