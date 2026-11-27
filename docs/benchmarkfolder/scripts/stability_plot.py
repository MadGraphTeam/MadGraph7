#!/usr/bin/env python3
"""Stability profiles for benchmark.typ, after appendix C of arXiv:1604.01363:
the fraction of phase-space points whose double-precision accuracy estimate
is at least Delta, against Delta. One row per quarkonium family (charmonium,
bottomonium), one column per setup.

    python3 scripts/stability_plot.py data/raw/stability figures/stability.svg

Writes the figure, a second one for the P-wave colour octets next to it
(figures/stability_octets.svg) and data/stability.json (the summary table
of both). Points MadLoop evaluated without a stability test (return code
1xx) are dropped. The colours are the first five slots of the validated
categorical palette (dataviz reference palette, light mode, checked on
white); the octet figure starts the slot order again, one slot per state. A
bottomonium process takes the colour of its charmonium analogue; a run with
TIRust 5025e12 with the bridge's binary64 gate and relation tiers (PR #56; case suffix 56)
takes the colour of its process, dotted; a P-wave run with the 80-bit switch
(#PWaveExtendedPrecision; TIRust d17085a with the bridge v2; suffix ep) dash-dotted;
the reference process of the paper is a dashed neutral line in every row."""
import glob
import json
import os
import statistics
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# family -> processes in plotting order; the reference ttg is drawn in every row
FAMILIES = [
    ('charmonium', 'charmonium: loop_sm_onia-c_mass ($m_c = 1.55$ GeV, $n_l = 3$)',
     ['etacg', 'jpsig', 'etac8g', 'chic0g', 'ggchic0g',
      'etacg56', 'jpsig56', 'etac8g56', 'chic0g56', 'ggchic0g56',
      'chic0gep', 'ggchic0gep']),
    ('bottomonium', 'bottomonium: loop_sm_onia-no_tau_mass ($m_b = 4.7$ GeV, $n_l = 4$)',
     ['etabg', 'upsg', 'etab8g', 'chib0g', 'ggchib0g',
      'etabg56', 'upsg56', 'etab8g56', 'chib0g56', 'ggchib0g56',
      'chib0gep', 'ggchib0gep']),
]
# the P-wave colour octets, in a figure of their own
OCTET_FAMILIES = [
    ('charmonium octets', r'charmonium P-wave octets: $e^+e^-\to c\bar c[n^{(8)}]\,g$, loop_sm_onia-c_mass',
     ['cc3p08g', 'cc3p18g', 'cc3p28g', 'cc1p18g', 'cc3p08g56', 'cc3p18g56', 'cc3p28g56',
      'cc1p18g56', 'cc3p08gep', 'cc3p18gep', 'cc3p28gep',
      'cc1p18gep']),
    ('bottomonium octets', r'bottomonium P-wave octets: $e^+e^-\to b\bar b[n^{(8)}]\,g$, loop_sm_onia-no_tau_mass',
     ['bb3p08g', 'bb3p18g', 'bb3p28g', 'bb1p18g', 'bb3p08g56', 'bb3p18g56', 'bb3p28g56',
      'bb1p18g56', 'bb3p08gep', 'bb3p18gep', 'bb3p28gep',
      'bb1p18gep']),
]
# TIRust da7902b with the bridge patch v16 (suffix 16; with the 80-bit switch 16ep) against the
# builds of the base cases, in figures of their own
V16_S = ['etacg', 'jpsig', 'etac8g', 'chic0g', 'ggchic0g']
V16_B = ['etabg', 'upsg', 'etab8g', 'chib0g', 'ggchib0g']
V16_PW = ('chic0g', 'ggchic0g', 'chib0g', 'ggchib0g')
V16_FAMILIES = [
    ('charmonium v16', 'charmonium: loop_sm_onia-c_mass ($m_c = 1.55$ GeV, $n_l = 3$)',
     V16_S + [c + '16' for c in V16_S] + [c + '16ep' for c in V16_S if c in V16_PW]),
    ('bottomonium v16', 'bottomonium: loop_sm_onia-no_tau_mass ($m_b = 4.7$ GeV, $n_l = 4$)',
     V16_B + [c + '16' for c in V16_B] + [c + '16ep' for c in V16_B if c in V16_PW]),
]
V16_OCT = {q: [q * 2 + k + 'g' for k in ('3p08', '3p18', '3p28', '1p18')] for q in 'cb'}
V16_OCTET_FAMILIES = [
    ('charmonium octets v16', OCTET_FAMILIES[0][1],
     V16_OCT['c'] + [c + '16' for c in V16_OCT['c']] + [c + '16ep' for c in V16_OCT['c']]),
    ('bottomonium octets v16', OCTET_FAMILIES[1][1],
     V16_OCT['b'] + [c + '16' for c in V16_OCT['b']] + [c + '16ep' for c in V16_OCT['b']]),
]
# (file suffix, families, legend columns); colour = process, line style = TIRust build
FIGURES = [('', FAMILIES, 3), ('_octets', OCTET_FAMILIES, 4), ('_v16', V16_FAMILIES, 3),
           ('_v16_octets', V16_OCTET_FAMILIES, 4)]
ALL_FAMILIES = FAMILIES + OCTET_FAMILIES + V16_FAMILIES + V16_OCTET_FAMILIES
REFERENCE = 'ttg'
ORDER = [c for _, _, cases in ALL_FAMILIES for c in cases] + [REFERENCE]
LABEL = {
    'etacg': r'$gg\to\eta_c\,g$',
    'jpsig': r'$gg\to J/\psi\,g$',
    'jpsig08': r'$gg\to J/\psi\,g$, TIRust 08edee3',
    'etac8g': r'$gg\to c\bar c[{}^1S_0^{(8)}]\,g$',
    'chic0g': r'$u\bar u\to\chi_{c0}\,g$ (P wave)',
    'ggchic0g': r'$gg\to\chi_{c0}\,g$ (P wave, TIRust 370b632)',
    'etabg': r'$gg\to\eta_b\,g$',
    'upsg': r'$gg\to\Upsilon\,g$',
    'upsg08': r'$gg\to\Upsilon\,g$, TIRust 08edee3',
    'etab8g': r'$gg\to b\bar b[{}^1S_0^{(8)}]\,g$',
    'chib0g': r'$u\bar u\to\chi_{b0}\,g$ (P wave)',
    'ggchib0g': r'$gg\to\chi_{b0}\,g$ (P wave, TIRust 370b632)',
    'ttg': r'$gg\to t\bar t\,g$ (reference, NinjaRS chain)',
}
for q in 'cb':
    for key, state in (('3p08', '{}^3P_0^{(8)}'), ('3p18', '{}^3P_1^{(8)}'), ('3p28', '{}^3P_2^{(8)}'),
                       ('1p18', '{}^1P_1^{(8)}')):
        # the row title names the process
        LABEL[q * 2 + key + 'g'] = r'$%s\bar %s[%s]$' % (q, q, state)
TYPST = {
    'etacg': r'g g -> eta_c thin g',
    'jpsig': r'g g -> J\/psi thin g',
    'jpsig08': r'g g -> J\/psi thin g',
    'etac8g': r'g g -> c overline(c) thin [""^1 S_0^((8))] thin g',
    'chic0g': r'u overline(u) -> chi_(c 0) thin g',
    'ggchic0g': r'g g -> chi_(c 0) thin g',
    'etabg': r'g g -> eta_b thin g',
    'upsg': r'g g -> Upsilon thin g',
    'upsg08': r'g g -> Upsilon thin g',
    'etab8g': r'g g -> b overline(b) thin [""^1 S_0^((8))] thin g',
    'chib0g': r'u overline(u) -> chi_(b 0) thin g',
    'ggchib0g': r'g g -> chi_(b 0) thin g',
    'ttg': r'g g -> t overline(t) thin g',
}
for q in 'cb':
    for key, state in (('3p08', '^3 P_0'), ('3p18', '^3 P_1'), ('3p28', '^3 P_2'), ('1p18', '^1 P_1')):
        TYPST[q * 2 + key + 'g'] = r'e^+ e^- -> %s overline(%s) thin [""%s^((8))] thin g' % (q, q, state)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#8a8985'
COLOR = {'etacg': BLUE, 'jpsig': ORANGE, 'jpsig08': ORANGE, 'etac8g': AQUA, 'chic0g': YELLOW, 'ggchic0g': MAGENTA,
         'etabg': BLUE, 'upsg': ORANGE, 'upsg08': ORANGE, 'etab8g': AQUA, 'chib0g': YELLOW, 'ggchib0g': MAGENTA,
         'ttg': GREY}
for q in 'cb':
    COLOR.update({q * 2 + '3p08g': BLUE, q * 2 + '3p18g': ORANGE, q * 2 + '3p28g': AQUA,
                  q * 2 + '1p18g': YELLOW})
# line style = TIRust build: the builds of the base cases (pinned for the S waves, 48958d1 and
# 370b632 for the P waves) solid, 08edee3 with the gate patch 0006 (suffix 08, no longer drawn)
# dash-dotted, 5025e12, the head of PR #56, with the bridge patch for the binary64 gate 0006 and
# the relation tiers (suffix 56), dotted
CANDIDATE = '5025e12'
BUILD_STYLE = {'': '-', '08': '-.', '56': ':', 'ep': '-.', '16': ':', '16ep': '-.'}
BUILD_KEY = {'08': 'TIRust 08edee3 + gate 0006', '56': 'TIRust %s + gate (PR #56)' % CANDIDATE,
             'ep': '80-bit switch, TIRust d17085a + bridge v2',
             '16': 'TIRust da7902b + bridge v16 (PR #71)', '16ep': '80-bit switch, v16'}
STYLE = {'ttg': '--'}


def build_of(case):
    # no base case ends in 08, 16, 56 or ep (they end in g)
    if case.endswith('16ep'):
        return '16ep'
    return case[-2:] if case[-2:] in ('08', '16', '56', 'ep') else ''


def base_of(case):
    return case[:-len(build_of(case))] if build_of(case) else case


for c in [c for _, _, cases in ALL_FAMILIES for c in cases if build_of(c) in ('16', '16ep')]:
    base = base_of(c)
    LABEL[c] = LABEL[base].replace(', TIRust 370b632', '') + (', v16, 80-bit' if build_of(c) == '16ep' else ', v16')
    TYPST[c], COLOR[c] = TYPST[base], COLOR[base]
for c in [c for _, _, cases in FAMILIES + OCTET_FAMILIES for c in cases if build_of(c) == 'ep']:
    base = c[:-2]
    LABEL[c] = LABEL[base].replace(', TIRust 370b632', '') + ', 80-bit'
    TYPST[c], COLOR[c] = TYPST[base], COLOR[base]
for c in [c for _, _, cases in FAMILIES + OCTET_FAMILIES for c in cases if build_of(c) == '56']:
    base = c[:-2]
    LABEL[c] = LABEL[base].replace(', TIRust 370b632', '').replace(')', ', TIRust %s)' % CANDIDATE) \
        if base in ('chic0g', 'ggchic0g', 'chib0g', 'ggchib0g') else LABEL[base] + ', TIRust ' + CANDIDATE
    TYPST[c], COLOR[c] = TYPST[base], COLOR[base]
for c in list(LABEL):
    STYLE.setdefault(c, BUILD_STYLE[build_of(c)])
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SETUP_TITLE = {'A': r'$\sqrt{s}=1$ TeV, $p_T>50$ GeV', 'B': r'$\sqrt{s}=20$ GeV, $p_T>2$ GeV'}
THRESHOLD = 1e-3
LIBNAMES = {'9': 'TIRust', '8': 'NinjaRS', '6': 'Ninja', '7': 'COLLIER', '1': 'CutTools', '3': 'IREGI'}


def family_of(case):
    for name, _, cases in ALL_FAMILIES:
        if case in cases:
            return name
    return 'reference'


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
    # a failed reduction is reported with accuracy 1
    return np.array([max(p['accuracy'], 1e-17) for p in points])


def reducer(meta):
    if meta['libs'] == '9':
        return 'TIRust ' + ('(pinned)' if meta['tirust'] == 'pinned' else meta['tirust'].split(' (')[0])
    return '|'.join(LIBNAMES.get(l, l) for l in meta['libs'].split('|'))


def summary(runs):
    fam_order = [f for f, _, _ in ALL_FAMILIES] + ['reference']
    rows = []
    for (case, setup), (m, pts) in sorted(
            runs.items(), key=lambda kv: (fam_order.index(family_of(kv[0][0])), kv[0][1],
                                          ORDER.index(kv[0][0]))):
        a = accuracies(pts)
        rows.append(dict(case=case, setup=setup, family=family_of(case), label=TYPST[case],
                         process=m['process'], model=m['model'], reducer=reducer(m),
                         libs=m['libs'], energy=m['energy'], ptmin=m['ptmin'], npoints=len(pts),
                         frac_1e3=float(np.mean(a >= 1e-3)), frac_1e6=float(np.mean(a >= 1e-6)),
                         frac_1e9=float(np.mean(a >= 1e-9)), median=float(np.median(a)),
                         ms=1e3 * statistics.median(p['time'] for p in pts)))
    return rows


def plot(runs, out, family_list, ncol=3):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': INK2,
                         'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                         'mathtext.fontset': 'dejavusans'})
    setups = sorted(set(s for _, s in runs))
    families = [f for f in family_list if any((c, s) in runs for c in f[2] for s in setups)]
    fig = plt.figure(figsize=(7.2, 3.3 * len(families)))
    subfigs = fig.subfigures(len(families), 1, squeeze=False)[:, 0]
    grid = np.logspace(-16, 0.5, 600)
    for sub, (family, title, cases) in zip(subfigs, families):
        axes = sub.subplots(1, len(setups), sharey=True, squeeze=False)[0]
        for ax, setup in zip(axes, setups):
            ax.set_facecolor('#ffffff')
            ax.axvline(THRESHOLD, color='#b7b5ae', lw=1.0, zorder=1)
            for case in cases + [REFERENCE]:
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
        axes[0].set_ylabel(r'fraction with accuracy $\geq\Delta$')
        handles, labels = [], []
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in labels:
                    handles.append(h)
                    labels.append(l)
        sub.suptitle(title, fontsize=8.5, color=INK, x=0.06, ha='left')
        # the legend: one entry per process (its colour, its label without the build), the
        # reference, then one key per line style; laid out row by row, ncol per row
        from matplotlib.lines import Line2D
        base = [c for c in cases if not build_of(c)]
        procs = [Line2D([], [], color=COLOR[c], lw=1.4, label=LABEL[c].replace(', TIRust 370b632', ''))
                 for c in base]
        ref = [h for h, l in zip(handles, labels) if l == LABEL[REFERENCE]]
        builds = sorted(set(reducer(runs[(c, s)][0]).replace('TIRust ', '').replace('(pinned)', 'pinned')
                            for c in base for s in setups if (c, s) in runs),
                        key=lambda b: (b != 'pinned', b))
        keys = [Line2D([], [], color=INK2, lw=1.4, ls='-', label='TIRust ' + ', '.join(builds))]
        keys += [Line2D([], [], color=INK2, lw=1.4, ls=BUILD_STYLE[b], label=BUILD_KEY[b])
                 for b in ('08', '56', 'ep', '16', '16ep') if any(build_of(c) == b for c in cases)]
        blank = Line2D([], [], alpha=0, label=' ')
        cells = procs + ref
        cells += [blank] * (-len(cells) % ncol) + keys
        cells += [blank] * (-len(cells) % ncol)
        nrows = len(cells) // ncol
        handles = [cells[r * ncol + col] for col in range(ncol) for r in range(nrows)]
        labels = [h.get_label() for h in handles]
        sub.legend(handles, labels, loc='lower center', ncol=ncol, frameon=False, fontsize=7.5,
                   bbox_to_anchor=(0.5, 0.0))
        sub.subplots_adjust(left=0.09, right=0.98, top=0.84, bottom=0.33, wspace=0.08)
    fig.savefig(out)


if __name__ == '__main__':
    folder, out = sys.argv[1], sys.argv[2]
    runs = load(folder)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    for suffix, family_list, ncol in FIGURES:
        stem, ext = os.path.splitext(out)
        plot(runs, stem + suffix + ext, family_list, ncol)
    data = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(out))), 'data', 'stability.json')
    json.dump(summary(runs), open(data, 'w'), indent=1)
    print('wrote', ', '.join(os.path.splitext(out)[0] + f[0] + os.path.splitext(out)[1] for f in FIGURES), data)
