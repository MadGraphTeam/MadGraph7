"""Recreate the final timing vector figure from copied, validated summaries."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".benchmark-work/final-report-plot-cache-v1"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "timing.svg")
    args = parser.parse_args()
    for font in (HERE / "fonts").glob("*.otf"):
        font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family": "Libertinus Serif", "font.size": 10,
        "svg.hashsalt": "madloop-final-20261022", "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": 10, "ytick.labelsize": 9.5})
    data = json.loads((HERE / "report-data.json").read_text())
    rows = [row for page in data["stock_cli"]["pages"] for row in page["rows"]]
    fig, (ax, ratio) = plt.subplots(2, 1, figsize=(7.0, 7.1), sharex=True,
                                   gridspec_kw={"height_ratios": [1.4, 1], "hspace": .16})
    navy, green = "#1f3b63", "#3a944a"
    for i, row in enumerate(rows, 1):
        for quantity, marker, shift in (("reduction_estimate", "o", -.11), ("total", "s", .11)):
            q = row["quantities"][quantity]
            for backend, color, offset, filled in (("6", navy, -.035, False), ("8", green, .035, True)):
                values = q["backend"][backend]
                y = values["median_seconds_per_point"] * 1000
                lo, hi = [v * 1000 for v in values["confidence_interval95"]]
                ax.errorbar(i + shift + offset, y, yerr=[[max(0, y-lo)], [max(0, hi-y)]],
                    marker=marker, markersize=5, linestyle="none", color=color,
                    markerfacecolor=color if filled else "white", capsize=2, linewidth=.9,
                    label=("Ninja" if backend == "6" else "NinjaRS") +
                          (" · reduction estimate" if quantity == "reduction_estimate" else " · total") if i == 1 else None)
            r = q["ninjars_over_ninja"]
            if r["ratio"] is not None:
                lo, hi = r["confidence_interval95"]; y = r["ratio"]
                ratio.errorbar(i + shift, y, yerr=[[max(0, y-lo)], [max(0, hi-y)]],
                    marker=marker, markersize=5, linestyle="none", color=navy if marker == "o" else green,
                    capsize=2, linewidth=.9)
        if row["quantities"]["total"]["ninjars_over_ninja"]["ratio"] is None:
            ratio.text(i, .014, "NA", ha="center", va="bottom", fontsize=9, color="#707070")
    ax.set_yscale("symlog", linthresh=.001)
    ax.set_ylabel("Wall time per point [ms]")
    ax.grid(axis="y", color="#e2e8ef", linewidth=.6)
    ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
    ratio.set_yscale("log"); ratio.set_ylim(.004, 5)
    ratio.axhline(1, color="#666666", linestyle="--", linewidth=.8)
    ratio.set_ylabel("NinjaRS / Ninja")
    ratio.set_xlabel("Process index (mapping below)")
    ratio.set_xticks(range(1, 13)); ratio.set_xlim(.5, 12.5)
    ratio.grid(axis="y", color="#e2e8ef", linewidth=.6)
    ratio.plot([], [], "o", color=navy, label="Estimated reduction")
    ratio.plot([], [], "s", color=green, label="Total")
    ratio.legend(loc="lower right", frameon=False, fontsize=9)
    fig.subplots_adjust(left=.12, right=.985, top=.97, bottom=.09)
    fig.savefig(args.output, metadata={"Date": None, "Creator": "Matplotlib final report source"})
    plt.close(fig)


if __name__ == "__main__":
    main()
