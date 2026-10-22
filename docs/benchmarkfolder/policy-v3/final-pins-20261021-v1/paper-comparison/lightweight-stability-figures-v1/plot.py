"""Offline vector reproduction of the materialized actual N100 DP study.

This is a rendering recipe, not a source/runtime qualification authority.
No native MadLoop/Rust archive, compiler or generated process is used.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import textwrap

HERE=Path(__file__).resolve().parent
INSPECTION_PNG=False
def require(condition,message):
    if not condition: raise ValueError(message)
PAIRS={'on-shell': ('qcd-1', 'ew-1'), 'two-particle': ('qcd-2', 'ew-2'), 'three-particle': ('qcd-3', 'ew-3'), 'loop-induced': ('supplement-gg-h', 'supplement-gg-hh')}
LABELS={'qcd-1': '$u\\bar{u}\\to Z$ (QCD)', 'ew-1': '$u\\bar{d}\\to W^+$ (EW)', 'qcd-2': '$gg\\to t\\bar{t}$ (QCD)', 'ew-2': '$e^+e^-\\to\\mu^+\\mu^-$ (EW)', 'qcd-3': '$u\\bar{u}\\to Zgg$ (QCD)', 'ew-3': '$e^+e^-\\to t\\bar{t}\\gamma$ (EW)', 'supplement-gg-h': '$gg\\to H$ (loop induced)', 'supplement-gg-hh': '$gg\\to HH$ (loop induced)'}
COLORS={'6': '#2a78d6', '8': '#1baf7a'}

def draw(prepared, pair, estimator, output, pyplot):
    fig = pyplot.figure(figsize=(7.0, 7.2), facecolor="white")
    outer = fig.add_gridspec(1, 2, wspace=.29)
    for index, cid in enumerate(pair):
        case = prepared[cid]
        inner = outer[index].subgridspec(3, 1, height_ratios=(3, 1, 2.5), hspace=.40)
        top = fig.add_subplot(inner[0]); ratio = fig.add_subplot(inner[1], sharex=top)
        foot = fig.add_subplot(inner[2]); foot.axis("off")
        counts = []
        for backend, label in (("6", "Ninja"), ("8", "NinjaRS")):
            tool = case["tools"][backend]
            require(tool["tail"] is not None and tool["ratio"] is not None,
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
        top.legend(loc="upper right", fontsize=9.5, frameon=False)
        ratio.set(xscale="log", xlabel="relative spread threshold Δ", ylabel="NRS / Ninja")
        ratio.set_ylim(bottom=0)
        from matplotlib.ticker import LogLocator
        ratio.xaxis.set_major_locator(LogLocator(base=10, numticks=6))
        ratio.axhline(1., color="#939ca5", linestyle="--", linewidth=.7)
        for axis in (top, ratio):
            axis.grid(True, color="#d7dfe7", linewidth=.45)
            axis.spines[["top", "right"]].set_visible(False)
        caption = case["sampling_caption"] or "Sampling metadata not supplied"
        wrapped = "\n".join(textwrap.fill(line, width=43) for line in caption.splitlines())
        foot.text(0, .95, wrapped+"\n"+"\n".join(counts), va="top", fontsize=9.5, color="#334d69")
    label = {"combined": "direction + Lorentz transformation", "direction": "loop-direction reversal",
             "lorentz": "Lorentz transformation"}[estimator]
    fig.suptitle("Unrecovered native double precision · "+label, fontsize=13, color="#1f3b63", y=.99)
    fig.text(.5, .025, "All N=100 observations; resolution 1%. Hollow triangles: zero events, one-sided 95% upper reference 2.95%.\n"
             "Ninja-zero tail ratios are undefined/unbounded gaps; no epsilon or smoothing.\n"
             "Self-consistency estimates do not measure error against a finite-part oracle.",
             ha="center", fontsize=8.5, color="#334d69")
    fig.subplots_adjust(top=.89, bottom=.12, left=.09, right=.98)
    fig.savefig(output.with_suffix(".svg"), metadata={"Date": None, "Creator": "MadLoop NinjaRS actual lightweight DP study"})
    if INSPECTION_PNG and estimator == "combined":
        fig.savefig(output.with_suffix(".png"), dpi=180)
    pyplot.close(fig)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); output=args.output.resolve()
    require(not output.exists(),'Preserve previous vector reproductions')
    manifest=json.loads((HERE/'manifest.json').read_text())
    for name,record in manifest['files'].items():
        filename=(HERE/name).resolve()
        require(filename.is_relative_to(HERE),'Invalid capsule path')
        raw=filename.read_bytes()
        require(hashlib.sha256(raw).hexdigest()==record['sha256'] and len(raw)==record['bytes'],'Changed figure capsule input: '+name)
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot,font_manager
    require(matplotlib.__version__==manifest['rendering']['matplotlib'],'Use the recorded Matplotlib version for byte-identical vectors')
    font=Path(font_manager.findfont('Libertinus Serif'))
    require(hashlib.sha256(font.read_bytes()).hexdigest()==manifest['rendering']['font_sha256'],'Use the recorded Libertinus Serif font for byte-identical vectors')
    output.mkdir(parents=True)
    with pyplot.rc_context({'font.family':'Libertinus Serif','font.size':10.3,'text.color':'#1f3b63',
            'axes.labelcolor':'#1f3b63','axes.titlecolor':'#1f3b63','mathtext.fontset':'stix',
            'svg.fonttype':'none','svg.hashsalt':'actual-lightweight-dp-v1'}):
        for estimator in ('combined','direction','lorentz'):
            prepared=json.loads((HERE/'data'/('figure-data-'+estimator+'.json')).read_text())
            require(set(prepared)==set(LABELS),'Unexpected materialized case selection')
            for case in prepared.values():
                require(case['N']==100 and case['requested_tools']==[6,8] and set(case['tools'])=={'6','8'},'Unexpected materialized point/tool scope')
            for group,pair in PAIRS.items(): draw(prepared,pair,estimator,output/(estimator+'-'+group),pyplot)
    expected={name:record['sha256'] for name,record in manifest['files'].items() if name.startswith('assets/')}
    for name,sha256 in expected.items():
        require(hashlib.sha256((output/Path(name).name).read_bytes()).hexdigest()==sha256,'Reproduced vector bytes differ: '+name)
    print(json.dumps({'vectors':len(expected),'byte_identical':True,'native_artifacts_opened':0,'new_scientific_qualification':False}))

if __name__=='__main__': main()
