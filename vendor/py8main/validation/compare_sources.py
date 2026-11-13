"""Beamstrahlung validation across event generators, MG7 paper style.

The same physics, generated more than one way, overlaid with a ratio to
the first sample given:

  * MadEvent with `apply_flavor_grouping False` -- the reference.
  * MG7 / madspace, which samples the split from the same CIRCE grid
    through `BeamstrahlungSampler`.
  * MadEvent with grouping on. Since 28489295f this produces the same
    events as the reference; on an older checkout it does not, because
    `dressed_lep` was false there and auto_dsig never summed the four
    dressed components.

Three observables, all read straight from the LHE so no shower is needed:

  1  the sampled beamstrahlung spectrum, 1 - x_BS, one entry per beam
  2  the beam energy the collision lost to it, 1 - s'/s = 1 - x_BS1 x_BS2
  3  the energy at the hard vertex, sqrt(shat)/sqrt(s), after beamstrahlung
     AND ISR -- the one the component bug distorts

Panels 1 and 2 drop the non-radiating entries, which are a delta at zero
and would otherwise pile into the first bin and dominate it; the fraction
that radiates is printed instead, and is itself one of the numbers being
compared. Everything is weighted by the cross section, so the ratio panel
carries a normalisation difference rather than hiding it in a shape.

Usage:
    python3 compare_sources.py  out.pdf  "label=events.lhe.gz" ...

The first file given is the reference for the ratio panels.
"""
import gzip
import sys

import matplotlib as mpl
mpl.use('pgf')                       # usetex alone drops the math minus in pdf
mpl.rcParams.update({'pgf.texsystem': 'pdflatex'})
mpl.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.size": 14})
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

lw = 1.2
allcolors = list(mcolors.TABLEAU_COLORS.values())
allcolors[1] = 'black'
allcolors[0] = 'blue'
allcolors[3] = 'red'
font_legend = font_manager.FontProperties(size=9)

E_CM = 240.0
# Below this a beam counts as not having radiated. It is set by the *file*,
# not by the physics: MadEvent writes the tag from Fortran as E15.7, i.e. 7
# significant digits, so any 1-x_BS below ~5e-8 is stored as exactly 1 --
# there is a hard edge at that value, 0 MadEvent beams against 74 mg7 ones in
# 1e-8..5e-8. mg7 writes "%15.7e", which is 8 significant digits, and keeps
# them. Counting at 1e-9 therefore compares mg7's resolution against
# MadEvent's truncation and shows a spurious 8% excess; at 1e-7 the two agree
# to 1.6 sigma. Nothing physical lives here either way: 5e-8 of a 120 GeV
# beam is 6 micro-eV.
RADIATES = 1e-7


def update_legend(ax, ncol, pos=None, border=False, font=font_legend,
                  handlelength=2., columnspacing=1.4):
    handles, labels = ax.get_legend_handles_labels()
    new_handles = []
    for h in handles:
        if 'fill' in dir(h):
            ls, lc = h.get_linestyle(), h.get_edgecolor()
            new_handles.append(mlines.Line2D([], [], c=lc, linestyle=ls))
        else:
            new_handles.append(h)
    ax.legend(handles=new_handles, labels=labels, prop=font, frameon=border,
              ncol=ncol, loc='upper left', bbox_to_anchor=pos,
              handlelength=handlelength, columnspacing=columnspacing)


def read_lhe(path):
    """Per event: weight, the two x_BS, and sqrt(shat).

    sqrt(shat) comes from the final state rather than from the incoming
    momenta on purpose: both codes boost the event before writing it and
    they do not pick the same frame, so the beam energies in the file are
    not comparable between them. The invariant mass of the final state is.
    """
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as fsock:
        text = fsock.read()
    weight, y1, y2, shat = [], [], [], []
    for block in text.split('<event>')[1:]:
        body = block.split('</event>')[0]
        w = None
        tot = np.zeros(4)
        ybs = (1.0, 1.0)
        for line in body.split('\n'):
            line = line.strip()
            if line.startswith('<beamstrahlung>'):
                field = line.replace('<beamstrahlung>', ' ') \
                            .replace('</beamstrahlung>', ' ').split()
                ybs = (float(field[0]), float(field[1]))
                continue
            if line.startswith('<') or not line:
                continue
            field = line.split()
            if w is None and len(field) == 6:
                w = float(field[2])
            elif len(field) >= 13 and field[1] == '1':
                tot += np.array([float(field[9]), float(field[6]),
                                 float(field[7]), float(field[8])])
        if w is None:
            continue
        weight.append(w)
        y1.append(ybs[0])
        y2.append(ybs[1])
        shat.append(max(tot[0] ** 2 - tot[1:].dot(tot[1:]), 0.0))
    weight = np.array(weight)
    # MadGraph writes the TOTAL cross section as XWGTUP on every unweighted
    # event, not sigma/N -- dividing here makes sum(w) the cross section
    if len(weight):
        weight = weight / len(weight)
    return dict(w=weight, y1=np.array(y1), y2=np.array(y2),
                shat=np.sqrt(np.array(shat)))


def binned(values, weights, bins, per_decade):
    """Sum of weights and its statistical error, per unit of the x axis."""
    inside = (values >= bins[0]) & (values < bins[-1])
    values, weights = values[inside], weights[inside]
    width = np.diff(np.log10(bins)) if per_decade else np.diff(bins)
    total, _ = np.histogram(values, bins=bins, weights=weights)
    sq, _ = np.histogram(values, bins=bins, weights=weights ** 2)
    return total / width, np.sqrt(sq) / width


def draw(ax, rax, bins, series, ref_name, xlabel, ylabel, logx,
         headroom=30.):
    centre = 0.5 * (bins[1:] + bins[:-1])
    if logx:
        centre = np.sqrt(bins[1:] * bins[:-1])
    ref = None
    peak = 0.0
    chi2 = {}
    # a later series that lands exactly on an earlier one would be invisible
    # drawn solid, which is precisely the result worth seeing
    styles = ['solid', 'solid', 'dashed', 'dotted']
    for i, (name, values, weights, col) in enumerate(series):
        y, e = binned(values, weights, bins, logx)
        ax.step(bins, np.concatenate([y[:1], y]), where='pre', linewidth=lw,
                color=col, linestyle=styles[i % len(styles)], label=name)
        peak = max(peak, y.max())
        if name == ref_name:
            ref = (y, e)
    # nudge each series sideways so overlapping error bars stay readable
    n_other = sum(1 for nm, _, _, _ in series if nm != ref_name)
    drawn = 0
    for name, values, weights, col in series:
        y, e = binned(values, weights, bins, logx)
        good = (ref[0] != 0) & np.isfinite(y)
        if name == ref_name:
            # the reference's own statistical error, as a band rather than
            # folded into every other series' bars
            rel = np.zeros(len(y))
            rel[good] = ref[1][good] / np.abs(ref[0][good])
            rax.fill_between(bins, np.concatenate([(1 - rel)[:1], 1 - rel]),
                             np.concatenate([(1 + rel)[:1], 1 + rel]),
                             step='pre', color='grey', alpha=0.3, linewidth=0)
            continue
        ratio = np.full(len(y), np.nan)
        err = np.full(len(y), np.nan)
        ratio[good] = y[good] / ref[0][good]
        err[good] = e[good] / np.abs(ref[0][good])
        shift = (drawn - 0.5 * (n_other - 1)) * 0.22
        drawn += 1
        at = (centre * (bins[1] / bins[0]) ** shift if logx
              else centre + shift * np.diff(bins))
        rax.errorbar(at, ratio, yerr=err, fmt='o', markersize=2.5,
                     linewidth=0.8, color=col)
        # the reference's error counts here even though it is drawn as a
        # band rather than on the points
        sigma = np.sqrt(e[good] ** 2 + ref[1][good] ** 2)
        live = sigma > 0
        pulls = (y[good][live] - ref[0][good][live]) / sigma[live]
        chi2[name] = (float((pulls ** 2).sum()), int(live.sum()))
    rax.axhline(1.0, color='grey', linewidth=0.8, linestyle='dashed')
    if logx:
        ax.set_xscale('log')
    ax.set_xlim(bins[0], bins[-1])
    ax.set_yscale('log')
    ax.set_ylim(peak * 2e-5, peak * headroom)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelbottom=False)
    rax.set_xlabel(xlabel)
    rax.set_ylabel('ratio', fontsize=11)
    rax.set_ylim(0.55, 1.45)
    rax.set_yticks([0.75, 1.0, 1.25])
    return chi2


def main():
    out = sys.argv[1]
    series_files = [arg.partition('=')[::2] for arg in sys.argv[2:]]
    ref_name = series_files[0][0]

    data = {}
    for label, path in series_files:
        data[label] = read_lhe(path)
        d = data[label]
        rad = np.concatenate([d['y1'], d['y2']])
        on = rad < 1 - RADIATES
        print('%-32s %6d events  sigma = %.4f pb  %.2f%% of beams radiate  '
              '<1-x_BS | radiating> = %.3e'
              % (label, len(d['w']), d['w'].sum(), 100.0 * on.mean(),
                 (1 - rad[on]).mean()))

    colours = [allcolors[1], allcolors[0], allcolors[3], allcolors[2]]
    fig = plt.figure()
    fig.set_size_inches(7 * 0.75, 9.4)
    outer = gridspec.GridSpec(3, 1, hspace=0.42)
    axes = []
    for row in range(3):
        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row], height_ratios=[3, 1.15], hspace=0.0)
        top = fig.add_subplot(inner[0])
        axes += [top, fig.add_subplot(inner[1], sharex=top)]
    fig.suptitle("$e^-e^+ \\to \\mu^-\\mu^+$ at the FCC-ee, 240 GeV"
                 "\\quad(\\texttt{fcce240ll})", fontsize=14, y=0.945)

    def pack(getter, dup=False):
        out = []
        for i, (label, _) in enumerate(series_files):
            d = data[label]
            w = d['w']
            if dup:
                w = np.concatenate([w, w])
            out.append((label, getter(d), w, colours[i % len(colours)]))
        return out

    # ---------------------------------------------------------- panel 1
    bins = np.logspace(-7, -1, 31)
    report = {}
    report['$1-x_{BS}$'] = draw(
        axes[0], axes[1], bins,
        pack(lambda d: 1.0 - np.concatenate([d['y1'], d['y2']]), dup=True),
        ref_name, '$1-x_{\\rm BS}$  (per beam, radiating only)',
        '$d\\sigma/d\\log_{10}(1-x_{\\rm BS})$ [pb]', logx=True,
        headroom=3e3)
    update_legend(axes[0], 1, pos=(0.02, 0.99))
    lines = []
    for label, _ in series_files:
        d = data[label]
        rad = np.concatenate([d['y1'], d['y2']])
        lines.append('%.2f\\%%' % (100.0 * (rad < 1 - RADIATES).mean()))
    axes[0].text(0.97, 0.06, 'radiating beams: ' + ', '.join(lines),
                 transform=axes[0].transAxes, ha='right', fontsize=9)

    # ---------------------------------------------------------- panel 2
    report["$1-s'/s$"] = draw(
         axes[2], axes[3], bins,
         pack(lambda d: 1.0 - d['y1'] * d['y2']),
         ref_name, "$1-s'/s$  (per event, radiating only)",
         "$d\\sigma/d\\log_{10}(1-s'/s)$ [pb]", logx=True)

    # ---------------------------------------------------------- panel 3
    bins = np.linspace(0.0, 1.0, 41)
    report['$\\sqrt{shat}/\\sqrt{s}$'] = draw(
         axes[4], axes[5], bins,
         pack(lambda d: d['shat'] / E_CM),
         ref_name, '$\\sqrt{\\hat s}/\\sqrt{s}$',
         '$d\\sigma/d(\\sqrt{\\hat s}/\\sqrt{s})$ [pb]', logx=False)

    print()
    print('%-22s %-32s %s' % ('observable', 'series', 'chi2/ndf vs reference'))
    for obs, per_series in report.items():
        for name, (value, ndf) in per_series.items():
            print('%-22s %-32s %6.1f / %-3d = %.2f'
                  % (obs, name, value, ndf, value / max(ndf, 1)))

    plt.savefig(out, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
