#!/usr/bin/env python3
"""Summarise a finished MadGraph run: cross section, MC error, events, seed,
scale/PDF bands and output files.

    run_summary.py <process_dir> [run_name]

Without a run name the most recent run in <process_dir>/Events is used.
Handles mg7 output (Events/<run>/info.json) and madevent output
(Events/<run>/<run>_tag_*_banner.txt plus the LHE <init> block).
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
    cuts = run_card_section(card, "cuts")
    print("cuts        : %s" % ("; ".join(cuts) if cuts else "none"))
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


def main():
    if len(sys.argv) < 2:
        fail(__doc__.strip())
    proc_dir = sys.argv[1]
    run_dir = pick_run(proc_dir, sys.argv[2] if len(sys.argv) > 2 else None)
    print("run         : %s" % run_dir)
    banners = sorted(glob.glob(os.path.join(run_dir, "*_banner.txt")))
    if os.path.exists(os.path.join(run_dir, "info.json")):
        summarise_mg7(proc_dir, run_dir)
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
    for extra in ("MADatLO.HwU", "events.weights.json"):
        if extra in files:
            print("also        : %s" % os.path.join(run_dir, extra))
    for decayed in sorted(glob.glob(run_dir.rstrip(os.sep) + "_decayed_*")):
        print("decayed     : %s (MadSpin; summarise it with its run name)"
              % decayed)


if __name__ == "__main__":
    main()
