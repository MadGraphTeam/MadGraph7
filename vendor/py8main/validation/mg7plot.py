"""Beamstrahlung validation figure, MG7 paper style.

Conventions taken from MG7_paper/plots/lepton/plots.py: usetex serif at 14,
figure 7*0.75 wide (the horizontal size is fixed), each axes expanded by
10% with set_position, suptitle naming collider and energy, frameless
line-legends through update_legend, and no tight_layout.

  1  forward-photon energy the shower builds, beams nominal vs degraded
  2  the beamstrahlung spectra behind it
  3  the collision energy left after beamstrahlung
"""
import math
import os

import matplotlib as mpl
mpl.use('pgf')                       # usetex alone drops the math minus in pdf
mpl.rcParams.update({'pgf.texsystem': 'pdflatex'})
mpl.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.size": 14})
mpl.rcParams.update({'lines.markersize': 8})
import matplotlib.colors as mcolors
import matplotlib.font_manager as font_manager
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

S = os.path.dirname(os.path.abspath(__file__)) + '/'

lw = 1.2
allcolors = list(mcolors.TABLEAU_COLORS.values())
allcolors[1] = 'black'
allcolors[0] = 'blue'
allcolors[3] = 'red'

font_legend = font_manager.FontProperties(size=9)


def update_legend(ax, ncol, pos=None, border=False, font=font_legend,
                  handlelength=2., columnspacing=2.):
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


def load(name):
    d = np.loadtxt(S + name)
    return dict(y1=d[:, 0], y2=d[:, 1], egam=d[:, 7])


s240_p, s240_d = load('fcc_plain.dat'), load('fcc_degr.dat')
s365_p, s365_d = load('f365_plain.dat'), load('f365_degr.dat')
ilc_p, ilc_d = load('ilc_plain.dat'), load('ilc_degr.dat')

SETS = ((s240_p, s240_d, allcolors[0], 'FCC-ee 240 GeV'),
        (s365_p, s365_d, allcolors[3], 'FCC-ee 365 GeV'),
        (ilc_p, ilc_d, allcolors[1], 'ILC 500 GeV'))

fig, axes = plt.subplots(nrows=3, ncols=1)
fig.set_size_inches(7 * 0.75, 7)
fig.suptitle("\n\n" + "    $e^-e^+ \\to \\mu^-\\mu^+$  at the FCC-ee "
             "and the ILC ", fontsize=16, y=1.02)

# the paper's scripts also stretch the height by 1.1, but they share a
# single x axis; here each panel carries its own, so only the width is
# widened and the panels are given room for their labels
fig.subplots_adjust(hspace=0.55)
for ax in axes:
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width * 1.1, pos.height])

# ---------------------------------------------------------------- panel 1
ax = axes[0]
for pl, dg, col, lab in SETS:
    for d, ls, tag in ((pl, 'dashed', 'nominal beams'),
                       (dg, 'solid', 'degraded beams')):
        sig = dg['y1'] * dg['y2'] < 1 - 1e-9
        v = np.sort(np.maximum(d['egam'][sig], 1e-3))
        ax.step(v, np.arange(1, len(v) + 1) / len(v), where='post',
                linewidth=lw, color=col, linestyle=ls,
                label=lab if ls == 'solid' else None)
ax.set_xscale('log')
ax.set_xlim(1e-2, 400)
ax.set_ylim(0, 1.5)
ax.set_xlabel('$E_\\gamma^{\\rm fwd}$ [GeV]')
ax.set_ylabel('fraction of events')
h, l = ax.get_legend_handles_labels()
h += [mlines.Line2D([], [], c='grey', linestyle='solid'),
      mlines.Line2D([], [], c='grey', linestyle='dashed')]
l += ['degraded beams', 'nominal beams']
ax.legend(handles=h, labels=l, prop=font_legend, frameon=False, ncol=2,
          loc='upper left', bbox_to_anchor=(0.0, 1.03), handlelength=2.,
          columnspacing=2.)

# ---------------------------------------------------------------- panel 2
ax = axes[1]
bins = np.logspace(-7, 0, 43)
dlog = np.log10(bins[1] / bins[0])
for pl, dg, col, lab in SETS:
    yy = np.concatenate([dg['y1'], dg['y2']])
    r = 1.0 - yy[yy < 1 - 1e-9]
    ax.hist(r, bins=bins, histtype='step', linewidth=lw, color=col,
            weights=np.full(len(r), 1.0 / len(yy) / dlog), label=lab)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(1e-7, 1.0)
ax.set_ylim(1e-4, 0.5)
ax.set_xlabel('$1-x_{\\rm BS}$')
ax.set_ylabel('$1/N \\, dN/d\\log_{10}$')
update_legend(ax, 3, pos=(0.02, 0.98))

# ---------------------------------------------------------------- panel 3
ax = axes[2]
bins = np.linspace(0.5, 1.0, 51)
for pl, dg, col, lab in SETS:
    r = np.sqrt(dg['y1'] * dg['y2'])
    ax.hist(np.clip(r, bins[0], bins[-1]), bins=bins, histtype='step',
            linewidth=lw, color=col,
            weights=np.full(len(r), 1.0 / len(r)), label=lab)
ax.set_yscale('log')
ax.set_xlim(0.5, 1.0)
ax.set_ylim(1e-5, 4.0)
ax.set_xlabel("$\\sqrt{s'}/\\sqrt{s}$")
ax.set_ylabel('fraction of events')
update_legend(ax, 2, pos=(0.02, 0.98))

plt.savefig(S + 'beamstrahlung_fccee.pdf', bbox_inches='tight')
print('wrote', S + 'beamstrahlung_fccee.pdf')

for pl, dg, col, lab in SETS:
    sig = dg['y1'] * dg['y2'] < 1 - 1e-9
    yy = np.concatenate([dg['y1'], dg['y2']])
    r = np.sqrt(dg['y1'] * dg['y2'])
    print('%s: %.1f%% of beams radiate, <dE>/E=%.5f%%, '
          '<sqrt(s\')/sqrt(s)>=%.6f, frac below 0.999 = %.2f%%'
          % (lab, 100.0 * (yy < 1 - 1e-9).mean(), 100.0 * np.mean(1 - yy),
             r.mean(), 100.0 * (r < 0.999).mean()))
