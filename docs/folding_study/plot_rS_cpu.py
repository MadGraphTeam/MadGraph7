"""r_S versus CPU per event, with iso-lines of CPU per effective event.
usage: plot_rS_cpu.py <scratch fold dir> <out dir> [process keys, comma separated] [output name]"""
import sys, gzip, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

S, OUT = [a for a in sys.argv if not a.startswith('--')][1:3]
FOLDCOL = {'1,1,1': '#2a78d6', '2,2,1': '#eb6834', '4,4,1': '#1baf7a', '8,4,1': '#4a3aa7'}
METHOD = {  # marker, filled
    'lattice': ('o', True), 'cubic': ('s', True), 'SumSets': ('D', True),
    'cubic+SumSets': ('D', True), 'cubic+SumSets+RealOnce': ('^', False),
    'lattice+RealOnce': ('v', False), 'lattice+SumSets': ('h', True)}
PROCS = {
    'T1_tt_v': ('p p > t t~ [QCD]', 20000, [
        ('q21A', '1,1,1', 'lattice'), ('q21J', '1,1,1', 'SumSets'),
        ('r21B', '2,2,1', 'lattice'), ('l21L', '2,2,1', 'lattice+SumSets'), ('q21I', '2,2,1', 'cubic'), ('q21K', '2,2,1', 'cubic+SumSets'),
        ('t21G', '2,2,1', 'cubic+SumSets+RealOnce'),
        ('l21M', '4,4,1', 'lattice+SumSets'), ('q21D', '4,4,1', 'cubic'), ('q21E', '4,4,1', 'cubic+SumSets'), ('s21F', '4,4,1', 'cubic+SumSets+RealOnce'),
        ('r21C', '8,4,1', 'lattice'), ('s21H', '8,4,1', 'lattice+RealOnce')], [5]),
    'T2_eej_v': ('p p > e+ e- j [QCD]', 10000, [
        ('q21A', '1,1,1', 'lattice'), ('q21J', '1,1,1', 'SumSets'),
        ('r21B', '2,2,1', 'lattice'), ('l21L', '2,2,1', 'lattice+SumSets'), ('q21I', '2,2,1', 'cubic'), ('q21K', '2,2,1', 'cubic+SumSets'),
        ('x21G', '2,2,1', 'cubic+SumSets+RealOnce'),
        ('l21M', '4,4,1', 'lattice+SumSets'), ('q21D', '4,4,1', 'cubic'), ('q21E', '4,4,1', 'cubic+SumSets'),
        ('r21C', '8,4,1', 'lattice')], [7]),
    'T3_ee4b_v': ('e+ e- > b b~ b b~ [QCD]', 10000, [
        ('s21A', '1,1,1', 'lattice'), ('s21J', '1,1,1', 'SumSets'),
        ('s21B', '2,2,1', 'lattice'), ('l21L', '2,2,1', 'lattice+SumSets'), ('s21I', '2,2,1', 'cubic'), ('s21K', '2,2,1', 'cubic+SumSets'),
        ('l21M', '4,4,1', 'lattice+SumSets'), ('s21D', '4,4,1', 'cubic'), ('s21E', '4,4,1', 'cubic+SumSets'),
        ('s21C', '8,4,1', 'lattice')], [7]),
    'T5_zjj': ('p p > z j j [QCD]  (nic6, 32 cores per run)', 10000, [
        ('z21A', '1,1,1', 'lattice'), ('z21J', '1,1,1', 'SumSets'),
        ('z21B', '2,2,1', 'lattice'), ('z21L', '2,2,1', 'lattice+SumSets'), ('z21I', '2,2,1', 'cubic'), ('z21K', '2,2,1', 'cubic+SumSets'),
        ('z21M', '4,4,1', 'lattice+SumSets'), ('z21D', '4,4,1', 'cubic'), ('z21E', '4,4,1', 'cubic+SumSets'),
        ('z21C', '8,4,1', 'lattice')], [6]),
}
ARGS = [a for a in sys.argv if not a.startswith('--')]
KEYS = ARGS[3].split(',') if len(ARGS) > 3 else ['T1_tt_v', 'T2_eej_v']
NAME = ARGS[4] if len(ARGS) > 4 else 'rS_vs_cpu'
PROCS = {k: PROCS[k] for k in KEYS}
if '--with-realonce' not in sys.argv:
    PROCS = {k: (t, n, [v for v in vs if 'RealOnce' not in v[2]], h) for k, (t, n, vs, h) in PROCS.items()}


def events(D, tag, hsizes):
    nS = nSn = nH = nHn = 0
    with gzip.open(os.path.join(D, 'Events', tag, 'events.lhe.gz'), 'rt') as f:
        inev = False
        for l in f:
            if l.startswith('<event'):
                inev = True; continue
            if inev:
                v = l.split(); n = int(v[0]); neg = float(v[2]) < 0
                if n in hsizes:
                    nH += 1; nHn += neg
                else:
                    nS += 1; nSn += neg
                inev = False
    return nS, nSn, nH, nHn


def cpu(D, tag):
    c = 0.
    for f in os.listdir(os.path.join(D, 'vlogs', tag)):
        if re.search(r'log_MINT[12]\.txt$', f):
            for l in open(os.path.join(D, 'vlogs', tag, f), errors='replace'):
                if 'Time spent in Total' in l:
                    c += float(l.split()[-1])
    return c


plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#888888', 'axes.labelcolor': '#333333',
                     'xtick.color': '#555555', 'ytick.color': '#555555'})
fig, axes = plt.subplots(len(PROCS), 1, figsize=(12.5, 5.25 * len(PROCS)), squeeze=False)
axes = axes[:, 0]
for ax, (pdir, (title, nev, variants, hsizes)) in zip(axes, PROCS.items()):
    D = os.path.join(S, pdir)
    pts = []
    for tag, fold, meth in variants:
        if not os.path.exists(os.path.join(D, 'Events', tag, 'events.lhe.gz')):
            print('  (missing %s, skipped)' % tag)
            continue
        nS, nSn, nH, nHn = events(D, tag, hsizes)
        N = nS + nH
        r = (nSn + nHn) / N
        pts.append(dict(tag=tag, fold=fold, meth=meth, x=cpu(D, tag) / N, rS=nSn / nS, rH=nHn / nH,
                        fS=nS / N, r=r, ce=cpu(D, tag) / (N * (1 - 2 * r) ** 2), xs_tag=tag))
    # iso-lines: CPU/N_eff = x / (1-2r)^2, r = fS*rS + (1-fS)*rH with the average fS, rH
    fS = np.mean([p['fS'] for p in pts]); rH = np.mean([p['rH'] for p in pts])
    xs = np.array([p['x'] for p in pts]); ys = np.array([p['rS'] for p in pts])
    xlo, xhi = xs.min() / 2.5, xs.max() * 2.5
    ylo, yhi = 0, max(ys) * 1.15
    yy = np.linspace(ylo, yhi, 200)
    rr = fS * yy + (1 - fS) * rH
    cmin, cmax = min(p['ce'] for p in pts), max(p['ce'] for p in pts)
    levels = np.geomspace(cmin / 2, cmax * 2, 9)
    for c in levels:
        ax.plot(c * (1 - 2 * rr) ** 2, yy, color='#bbbbbb', lw=0.9, zorder=1)
        # label on the line itself, at the height where it is drawn
        yl = yhi * 0.80
        rl = fS * yl + (1 - fS) * rH
        xl = c * (1 - 2 * rl) ** 2
        if xlo < xl < xhi:
            ax.text(xl, yl, ('%.3g' % (c * 1000) + ' ms') if c < 1 else '%.3g s' % c,
                    fontsize=7.5, color='#888888', ha='center', va='top', rotation=90,
                    bbox=dict(fc='white', ec='none', pad=0.5))
    for p in pts:
        m, filled = METHOD[p['meth']]
        col = FOLDCOL[p['fold']]
        ax.scatter(p['x'], p['rS'], marker=m, s=70, zorder=3,
                   facecolor=col if filled else 'white', edgecolor=col if not filled else 'white',
                   linewidth=1.8 if not filled else 1.2,
                   label='%s %s  (r_H=%.3f, CPU/N_eff=%s)' % (
                       p['fold'], p['meth'], p['rH'],
                       ('%.3g ms' % (p['ce'] * 1000)) if p['ce'] < 1 else '%.3g s' % p['ce']))
    ax.set_xscale('log')
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel('CPU per event [s]  (MINT1+MINT2 CPU / number of events)')
    ax.set_ylabel('r_S  (negative fraction of S events)')
    ax.set_title(title, color='#222222', fontsize=11)
    ax.grid(False)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=8.5, loc='upper left', bbox_to_anchor=(1.02, 1.0), ncol=1, frameon=False,
              title='folding, method  (r_H, CPU/N_eff)', title_fontsize=8.5)
    ax.text(0.99, 0.99, 'grey: constant CPU/N_eff, N_eff = N(1-2r)$^2$,\n'
            'r = f$_S$ r$_S$ + (1-f$_S$) r$_H$, f$_S$=%.2f, r$_H$=%.3f (averages)' % (fS, rH),
            transform=ax.transAxes, fontsize=7.5, color='#777777', va='top', ha='right',
            bbox=dict(fc='white', ec='none', alpha=0.9))
    md = open(os.path.join(OUT, NAME + '_%s.md' % pdir), 'w')
    md.write('| folding | method | r | r_S | r_H | CPU/event | CPU/N_eff |\n|---|---|---|---|---|---|---|\n')
    fmt = lambda c: ('%.3g ms' % (c * 1000)) if c < 1 else '%.3g s' % c
    for p in pts:
        md.write('| %s | %s | %.3f | %.3f | %.3f | %s | %s |\n' % (p['fold'], p['meth'], p['r'], p['rS'], p['rH'], fmt(p['x']), fmt(p['ce'])))
    md.close()
    print(title)
    for p in pts:
        print('  %-6s %-6s %-24s x=%.4g rS=%.4f rH=%.4f r=%.4f CPU/Neff=%.4g' % (
            p['tag'], p['fold'], p['meth'], p['x'], p['rS'], p['rH'], p['r'], p['ce']))
fig.suptitle('MC@NLO folding variants: S-event negative fraction vs CPU (seed 21%s)' % ('; open markers: FoldRealOnce' if any('RealOnce' in v[2] for P in PROCS.values() for v in P[2]) else ''),
             fontsize=11, color='#222222')
fig.tight_layout(rect=(0, 0, 1, 0.97 if len(PROCS) > 1 else 0.94))
fig.savefig(os.path.join(OUT, NAME + '.png'), dpi=150)
fig.savefig(os.path.join(OUT, NAME + '.pdf'))
