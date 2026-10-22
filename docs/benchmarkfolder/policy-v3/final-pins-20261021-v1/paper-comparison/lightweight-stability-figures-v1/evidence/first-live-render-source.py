"""Private actual-data figures; execute only in the root's released CPU window.

The existing concrete DP reader supplies numerical authority. This file changes
presentation only and never calls a reducer, compiler, or generated executable.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import textwrap

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "bin/madgraph").is_file())
sys.path.insert(0, str(ROOT))
from tools import ninjars_paper_report_v3 as paper

BASE = ROOT / ".benchmark-work/paper-dp-stability-20261021-v1/lightweight-final-eight-qualified-v2"
PINS = {"manifest.json": "de92bfae90b1068ba0fcb22034163d0b97ec9e4ab416ad17146bfa605e87f1a2",
        "summary.json": "c46368e08f039d39d4acb6369f067407785315cf5a591099db97f1e9adb6b12e",
        "cross-backend-discrepancy.json": "392f3ff2360df961aa1cb2170b08ca2aea446e1db5dc55c27b710bd76b13d94d"}
ARITHMETIC_AUDIT = ROOT / ".benchmark-work/final-release-audit-20261021-v1/light-eight-arithmetic-audit-v1.json"
ARITHMETIC_AUDIT_SHA256 = "99e0f508d32c2c668fe0a54a9c2e4cf3361e8915197ad54782137ef4c027baa7"
COHORT = {"numerical_revision": "856e76e669618ec8d0f799b9c8e6f86819dcd986",
          "generation_revision": "856e76e669618ec8d0f799b9c8e6f86819dcd986",
          "generator_identity_sha256": "77b16df0251af4104a2c541bab913c11f4e67496a290495ce1a1c5fc526dfe3c",
          "runtime_identity_sha256": "a7577a9a33292ffb5d675ccf1b9c3667db3ab3613d77ccd270f1efc9e9802008",
          "archive_sha256": "b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c"}
PAIRS = {"on-shell": ("qcd-1", "ew-1"), "two-particle": ("qcd-2", "ew-2"),
         "three-particle": ("qcd-3", "ew-3"), "loop-induced": ("supplement-gg-h", "supplement-gg-hh")}
LABELS = {"qcd-1": r"$u\bar{u}\to Z$ (QCD)", "ew-1": r"$u\bar{d}\to W^+$ (EW)",
          "qcd-2": r"$gg\to t\bar{t}$ (QCD)", "ew-2": r"$e^+e^-\to\mu^+\mu^-$ (EW)",
          "qcd-3": r"$u\bar{u}\to Zgg$ (QCD)", "ew-3": r"$e^+e^-\to t\bar{t}\gamma$ (EW)",
          "supplement-gg-h": r"$gg\to H$ (loop induced)", "supplement-gg-hh": r"$gg\to HH$ (loop induced)"}
COLORS = {"6": "#2a78d6", "8": "#1baf7a"}


def write(filename, value):
    filename.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def bound(filename):
    raw = filename.read_bytes()
    return {"path": str(filename.relative_to(ROOT)), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def validate():
    for name, digest in PINS.items():
        paper.require(bound(BASE/name)["sha256"] == digest, "Actual input changed: "+name)
    paper.require(bound(ARITHMETIC_AUDIT)["sha256"] == ARITHMETIC_AUDIT_SHA256,
                  "Independent arithmetic audit changed")
    raw = json.loads((BASE/"manifest.json").read_text())
    paper.require(raw["cohort"] == COHORT, "Unexpected final study cohort")
    snapshot = paper.build_snapshot(stability=BASE/"manifest.json", expected=COHORT)
    dp = snapshot["stability"]
    paper.require(dp == json.loads((BASE/"summary.json").read_text()), "Fresh authority replay differs from retained actual summary")
    paper.require(dp["complete"] is True and dp["physical_qualified"] is True and
                  dp["observations_complete"] is True and dp["cohort_validated"] is True and
                  dp["requested_tools"] == [6, 8], "No complete qualified actual two-tool observations")
    paper.require(set(dp["cases"]) == set(LABELS), "The lightweight study must contain exactly the declared eight cases")
    for cid, case in dp["cases"].items():
        paper.require(case["N"] == 100 and set(case["tools"]) == {"6", "8"}, "Unexpected case point/tool scope: "+cid)
        for backend, tool in case["tools"].items():
            paper.require(tool["status"] == "complete" and tool["physical_qualified"] is True and
                          tool["initialization"]["safe"] is True, "Unsafe or incomplete physical observations: "+cid+"/"+backend)
    return snapshot


def draw(prepared, pair, estimator, output, pyplot):
    fig = pyplot.figure(figsize=(8.4, 5.8), facecolor="white")
    outer = fig.add_gridspec(1, 2, wspace=.29)
    for index, cid in enumerate(pair):
        case = prepared[cid]
        inner = outer[index].subgridspec(3, 1, height_ratios=(3, 1, 1.2), hspace=.18)
        top = fig.add_subplot(inner[0]); ratio = fig.add_subplot(inner[1], sharex=top)
        foot = fig.add_subplot(inner[2]); foot.axis("off")
        counts = []
        for backend, label in (("6", "Ninja"), ("8", "NinjaRS")):
            tool = case["tools"][backend]
            paper.require(tool["tail"] is not None and tool["ratio"] is not None,
                          "Do not render a missing observed tool")
            observed = tool["tail"]
            rows, contrasts = observed["rows"], tool["ratio"]
            x = [row["threshold"] for row in rows]
            top.step(x, [row["fraction"] if row["k"] else math.nan for row in rows],
                     where="post", color=COLORS[backend], label=label, linewidth=1.5)
            zero = next((row for row in rows if row["k"] == 0), None)
            if zero:
                top.scatter([zero["threshold"]], [zero["zero_event_upper95"]], marker="v",
                            facecolors="none", edgecolors=COLORS[backend], s=36, zorder=5)
            ratio.step(x, [row["ratio"] if row["ratio"] is not None else math.nan for row in contrasts],
                       where="post", color=COLORS[backend], linewidth=1.3)
            counts.append(f"{label}: N={observed['N']}; {observed['failure_or_unbounded']} failed/unbounded")
        top.set(xscale="log", yscale="log", xlim=(case["declared_thresholds"][0], case["declared_thresholds"][-1]),
                ylim=(.005, 1.15), ylabel="fraction with spread > Δ")
        top.tick_params(labelbottom=False)
        top.set_title(LABELS[cid], fontsize=12, pad=7)
        top.axhline(1/case["N"], color="#939ca5", linestyle=":", linewidth=.8)
        top.legend(loc="upper right", fontsize=8, frameon=False)
        ratio.set(xscale="log", xlabel="relative spread threshold Δ", ylabel="NRS / Ninja")
        ratio.set_ylim(bottom=0)
        ratio.axhline(1., color="#939ca5", linestyle="--", linewidth=.7)
        for axis in (top, ratio):
            axis.grid(True, color="#d7dfe7", linewidth=.45)
            axis.spines[["top", "right"]].set_visible(False)
        caption = case["sampling_caption"] or "Sampling metadata not supplied"
        wrapped = "\n".join(textwrap.fill(line, width=64) for line in caption.splitlines())
        foot.text(0, .95, wrapped+"\n"+"\n".join(counts), va="top", fontsize=7.3, color="#334d69")
    label = {"combined": "direction + Lorentz transformation", "direction": "loop-direction reversal",
             "lorentz": "Lorentz transformation"}[estimator]
    fig.suptitle("Unrecovered native double precision · "+label, fontsize=13, color="#1f3b63", y=.99)
    fig.text(.5, .025, "Steps use all N=100 observations; resolution 1%. Hollow triangles: zero events, one-sided 95% upper reference 2.95%.\n"
             "Ninja-zero tail ratios are undefined/unbounded gaps; no epsilon or smoothing. Estimates are self-consistency, not error to an oracle.",
             ha="center", fontsize=7.2, color="#334d69")
    fig.subplots_adjust(top=.89, bottom=.12, left=.09, right=.98)
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None, "Creator": "MadLoop NinjaRS actual lightweight DP study"})
    if estimator == "combined":
        fig.savefig(output.with_suffix(".png"), dpi=180)
    pyplot.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); output = paper.path(args.output)
    paper.require(not output.exists(), "Preserve existing actual-data figure attempts")
    snapshot = validate()  # All raw/source/runtime checks precede output creation.
    output.mkdir(parents=True)
    write(output/"validated-snapshot.json", snapshot)
    for name in PINS:
        shutil.copyfile(BASE/name, output/("input-"+name))
    shutil.copyfile(ARITHMETIC_AUDIT, output/"independent-arithmetic-audit.json")
    os.environ["MPLCONFIGDIR"] = str(ROOT/".ninjars-build/paper-report-mpl")
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot, font_manager
    font = Path(font_manager.findfont("Libertinus Serif"))
    dp = snapshot["stability"]
    settings = {"font.family": "Libertinus Serif", "font.size": 8.5, "text.color": "#1f3b63",
                "axes.labelcolor": "#1f3b63", "axes.titlecolor": "#1f3b63", "mathtext.fontset": "stix",
                "svg.fonttype": "none", "svg.hashsalt": "actual-lightweight-dp-v1"}
    with pyplot.rc_context(settings):
        for estimator in ("combined", "direction", "lorentz"):
            prepared = paper.figure_data(dp["cases"], dp["thresholds"], estimator, requested_tools=[6, 8])
            write(output/("figure-data-"+estimator+".json"), prepared)
            for group, pair in PAIRS.items():
                draw(prepared, pair, estimator, output/(estimator+"-"+group), pyplot)
    counts = {cid: {"N": case["N"], "sample_kind": case["sample_kind"],
                    "helicity_selection": case["helicity_selection"],
                    "center_of_mass_energy_gev": case["center_of_mass_energy_gev"],
                    "tools": {backend: {"classifications": {estimator: tool["contrasts"][estimator]["classification_counts"]
                        for estimator in ("combined", "direction", "lorentz")},
                        "maximum_combined_spread": max((row["delta"] for row in tool["scores"]["combined"] if row["delta"] is not None), default=None),
                        "origin_card_policy": copy.deepcopy(tool["qualification"].get("origin_card_policy"))}
                        for backend, tool in case["tools"].items()}}
              for cid, case in dp["cases"].items()}
    write(output/"counts.json", counts)
    sources = {str(Path(paper.__file__).relative_to(ROOT)): hashlib.sha256(Path(paper.__file__).read_bytes()).hexdigest(),
               str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    sources.update(snapshot["inputs_sha256"])
    for name in PINS:
        sources[str((BASE/name).relative_to(ROOT))] = PINS[name]
    sources[str(ARITHMETIC_AUDIT.relative_to(ROOT))] = ARITHMETIC_AUDIT_SHA256
    write(output/"inputs-sha256.json", dict(sorted(sources.items())))
    receipt = {"schema": "actual-lightweight-dp-figure-render-v1", "software_fixture": False,
        "physical_observations_qualified": True, "independent_arithmetic_audit": bound(ARITHMETIC_AUDIT),
        "independent_raw_audit": "pending external receipt",
        "publication": "private pending root review and independent raw custody audit", "cohort": COHORT,
        "cases": list(LABELS), "requested_tools": [6, 8], "points_per_case_per_tool": 100,
        "empirical_resolution": .01, "zero_event_upper95": -math.expm1(math.log(.05)/100),
        "native_evaluations": 0, "numerical_oracle_claim": False, "timing_or_fullsum_rescue_claim": False,
        "python": platform.python_version(), "matplotlib": matplotlib.__version__,
        "font": {"path": str(font), "sha256": hashlib.sha256(font.read_bytes()).hexdigest()},
        "artifacts": [bound(filename) for filename in sorted(output.iterdir()) if filename.is_file()]}
    write(output/"render-receipt.json", receipt)
    print(json.dumps({"output": str(output), "figures": len(PAIRS)*3, "cases": len(dp["cases"]),
                      "physical_observations_qualified": True, "publication": receipt["publication"]}))


if __name__ == "__main__":
    main()
