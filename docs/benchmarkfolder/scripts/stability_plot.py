#!/usr/bin/env python3
"""Stability profiles for benchmark.typ, after appendix C of arXiv:1604.01363:
the fraction of phase-space points whose double-precision accuracy estimate
is at least Delta, against Delta, one panel per setup.

    python3 scripts/stability_plot.py data/raw/stability figures/stability.svg

Writes the figure and data/stability.json (the summary table). Points
MadLoop evaluated without a stability test (return code 1xx) are dropped.
The colours are the first four slots of the validated categorical palette
(dataviz reference palette, light mode, checked on white); the reference
process of the paper is a dashed neutral line."""
import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ORDER = ['etacg', 'jpsig', 'jpsig08', 'etac8g', 'chic0g', 'ttg']
LABEL = {
    'etacg': r'$gg\to\eta_c\,g$',
    'jpsig': r'$gg\to J/\psi\,g$',
    'jpsig08': r'$gg\to J/\psi\,g$, TIRust 08edee3',
    'etac8g': r'$gg\to c\bar c[{}^1S_0^{(8)}]\,g$',
    'chic0g': r'$u\bar u\to\chi_{c0}\,g$ (P wave)',
    'ttg': r'$gg\to t\bar t\,g$ (reference, NinjaRS chain)',
}
TYPST = {
    'etacg': r'g g -> eta_c thin g',
    'jpsig': r'g g -> J\/psi thin g',
    'jpsig08': r'g g -> J\/psi thin g',
    'etac8g': r'g g -> c overline(c) thin [""^1 S_0^((8))] thin g',
    'chic0g': r'u overline(u) -> chi_(c 0) thin g',
    'ttg': r'g g -> t overline(t) thin g',
}
COLOR = {'etacg': '#2a78d6', 'jpsig': '#eb6834', 'jpsig08': '#eb6834', 'etac8g': '#1baf7a',
         'chic0g': '#eda100', 'ttg': '#8a8985'}
STYLE = {'ttg': '--', 'jpsig08': ':'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SETUP_TITLE = {'A': r'$\sqrt{s}=1$ TeV, $p_T>50$ GeV', 'B': r'$\sqrt{s}=20$ GeV, $p_T>2$ GeV'}
THRESHOLD = 1e-3


def load(folder):
    runs = {}
    for f in sorted(glob.glob(os.path.join(folder, '*.json'))):
        d = json.load(open(f))
        m = d['meta']
        if 'points' not in d:
            d['points'] = [dict(accuracy=a, ret_code=c, time=t) for a, c, t in
                           zip(d['accuracy'], d['ret_code'], d['time'])]
        tested = [p for p in d['points'] if p['ret_code'] // 100 != 1]
        runs[(m['case'], m['setup'])] = (m, tested)
    return runs


def accuracies(points):
    # a failed reduction is reported with accuracy 1; a missing estimate (-1)
    # cannot occur for tested points
    return np.array([max(p['accuracy'], 1e-17) for p in points])


LIBNAMES = {'9': 'TIRust', '8': 'NinjaRS', '6': 'Ninja', '7': 'COLLIER', '1': 'CutTools', '3': 'IREGI'}


def reducer(meta):
    if meta['libs'] == '9':
        return 'TIRust ' + ('(pinned)' if meta['tirust'] == 'pinned' else meta['tirust'].split(' (')[0])
    return '|'.join(LIBNAMES.get(l, l) for l in meta['libs'].split('|'))


def summary(runs):
    rows = []
    for (case, setup), (m, pts) in sorted(runs.items(), key=lambda kv: (kv[0][1], ORDER.index(kv[0][0]))):
        a = accuracies(pts)
        rows.append(dict(case=case, setup=setup, label=TYPST[case], process=m['process'],
                         reducer=reducer(m),
                         libs=m['libs'], energy=m['energy'], ptmin=m['ptmin'], npoints=len(pts),
                         frac_1e3=float(np.mean(a >= 1e-3)), frac_1e6=float(np.mean(a >= 1e-6)),
                         frac_1e9=float(np.mean(a >= 1e-9)), median=float(np.median(a)),
                         ms=1e3 * statistics.median(p['time'] for p in pts)))
    return rows


def plot(runs, out):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2,
                         'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'mathtext.fontset': 'dejavusans'})
    setups = sorted(set(s for _, s in runs))
    fig, axes = plt.subplots(1, len(setups), figsize=(7.2, 3.3), sharey=True)
    if len(setups) == 1:
        axes = [axes]
    grid = np.logspace(-16, 0.5, 600)
    for ax, setup in zip(axes, setups):
        ax.set_facecolor('#ffffff')
        ax.axvline(THRESHOLD, color='#b7b5ae', lw=1.0, zorder=1)
        for case in ORDER:
            if (case, setup) not in runs:
                continue
            a = accuracies(runs[(case, setup)][1])
            frac = np.array([np.mean(a >= d) for d in grid])
            frac = np.where(frac > 0, frac, np.nan)
            ax.step(grid, frac, where='post', color=COLOR[case], lw=1.4,
                    ls=STYLE.get(case, '-'), label=LABEL[case], zorder=3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1e-16, 2)
        ax.set_ylim(1e-4, 1.4)
        ax.set_xlabel(r'target accuracy $\Delta$')
        ax.set_title(SETUP_TITLE.get(setup, setup), fontsize=8, color=INK)
        ax.grid(True, which='major', color=GRID, lw=0.5)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_xticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3, 1])
    axes[0].set_ylabel(r'fraction of points with accuracy $\geq\Delta$')
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=7.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    fig.savefig(out)


if __name__ == '__main__':
    folder, out = sys.argv[1], sys.argv[2]
    runs = load(folder)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plot(runs, out)
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out))), 'data', 'stability.json')
    json.dump(summary(runs), open(data, 'w'), indent=1)
    print('wrote', out, data)
