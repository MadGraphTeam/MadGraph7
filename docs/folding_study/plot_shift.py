"""Shift of the (CPU per event, r_S) points between two code versions run side by side.

usage: plot_shift.py <old process dir> <old tag prefix> <new process dir> <new tag prefix> <out dir> <name> <title> [H-event particle count, default 7]

The two versions of each variant ran concurrently on the same machine
(same number of cores), so that their CPU times see the same load.
"""
import sys, gzip, os, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OLD, PO, NEW, PN, OUT, NAME, TITLE = sys.argv[1:8]
HSIZES = [int(sys.argv[8]) if len(sys.argv) > 8 else 7]
FOLDCOL = {'1,1,1': '#2a78d6', '2,2,1': '#eb6834', '4,4,1': '#1baf7a', '8,4,1': '#4a3aa7'}
METHOD = {'lattice': 'o', 'cubic': 's', 'SumSets': 'D', 'cubic+SumSets': 'D', 'lattice+SumSets': 'h'}
VARIANTS = [('A', '1,1,1', 'lattice'), ('J', '1,1,1', 'SumSets'),
            ('B', '2,2,1', 'lattice'), ('L', '2,2,1', 'lattice+SumSets'), ('I', '2,2,1', 'cubic'),
            ('K', '2,2,1', 'cubic+SumSets'),
            ('M', '4,4,1', 'lattice+SumSets'), ('D', '4,4,1', 'cubic'), ('E', '4,4,1', 'cubic+SumSets'),
            ('C', '8,4,1', 'lattice')]


def point(D, tag):
    ev = os.path.join(D, 'Events', tag, 'events.lhe.gz')
    if not os.path.exists(ev):
        return None
    nS = nSn = nH = nHn = 0
    with gzip.open(ev, 'rt') as f:
        inev = False
        for l in f:
            if l.startswith('<event'):
                inev = True; continue
            if inev:
                v = l.split(); neg = float(v[2]) < 0
                if int(v[0]) in HSIZES:
                    nH += 1; nHn += neg
                else:
                    nS += 1; nSn += neg
                inev = False
    c = 0.
    d = os.path.join(D, 'vlogs', tag)
    for f in os.listdir(d):
        if re.search(r'log_MINT[12]\.txt$', f):
            for l in open(os.path.join(d, f), errors='replace'):
                if 'Time spent in Total' in l:
                    c += float(l.split()[-1])
    N = nS + nH
    r = (nSn + nHn) / N
    return dict(x=c / N, rS=nSn / nS, rH=nHn / nH, r=r, fS=nS / N, ce=c / (N * (1 - 2 * r) ** 2))


fmt = lambda c: ('%.3g ms' % (c * 1000)) if c < 1 else '%.3g s' % c
rows = []
for T, fold, meth in VARIANTS:
    o, n = point(OLD, PO + T), point(NEW, PN + T)
    if o and n:
        rows.append((T, fold, meth, o, n))

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#888888', 'axes.labelcolor': '#333333',
                     'xtick.color': '#555555', 'ytick.color': '#555555'})


def isolines(ax, pts, xlo, xhi, yhi):
    fS = np.mean([p['fS'] for p in pts]); rH = np.mean([p['rH'] for p in pts])
    yy = np.linspace(0, yhi, 200)
    rr = fS * yy + (1 - fS) * rH
    ces = [p['ce'] for p in pts]
    for c in np.geomspace(min(ces) / 2, max(ces) * 2, 9):
        ax.plot(c * (1 - 2 * rr) ** 2, yy, color='#bbbbbb', lw=0.9, zorder=1)
        yl = yhi * 0.80
        xl = c * (1 - 2 * (fS * yl + (1 - fS) * rH)) ** 2
        if xlo < xl < xhi:
            ax.text(xl, yl, fmt(c), fontsize=7.5, color='#888888', ha='center', va='top', rotation=90,
                    bbox=dict(fc='white', ec='none', pad=0.5))
    return fS, rH


def frame(ax, fS, rH):
    ax.set_xscale('log')
    ax.set_xlabel('CPU per event [s]  (MINT1+MINT2 CPU / number of events)')
    ax.set_ylabel('r_S  (negative fraction of S events)')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.text(0.99, 0.99, 'grey: constant CPU/N_eff, N_eff = N(1-2r)$^2$,\n'
            'r = f$_S$ r$_S$ + (1-f$_S$) r$_H$, f$_S$=%.2f, r$_H$=%.3f (averages)' % (fS, rH),
            transform=ax.transAxes, fontsize=7.5, color='#777777', va='top', ha='right',
            bbox=dict(fc='white', ec='none', alpha=0.9))


allp = [o for *_, o, n in rows] + [n for *_, o, n in rows]
xs = np.array([p['x'] for p in allp]); ys = np.array([p['rS'] for p in allp])
xlo, xhi, yhi = xs.min() / 2.5, xs.max() * 2.5, ys.max() * 1.15

# 1) shift plot
fig, ax = plt.subplots(figsize=(12.5, 5.6))
fS, rH = isolines(ax, allp, xlo, xhi, yhi)
for T, fold, meth, o, n in rows:
    col = FOLDCOL[fold]; m = METHOD[meth]
    ax.scatter(o['x'], o['rS'], marker=m, s=60, facecolor='white', edgecolor=col, linewidth=1.5, zorder=3)
    ax.scatter(n['x'], n['rS'], marker=m, s=70, facecolor=col, edgecolor='white', linewidth=1.2, zorder=4,
               label='%s %s  (CPU/N_eff %s -> %s, x%.2f)' % (fold, meth, fmt(o['ce']), fmt(n['ce']),
                                                            o['ce'] / n['ce']))
    ax.annotate('', xy=(n['x'], n['rS']), xytext=(o['x'], o['rS']),
                arrowprops=dict(arrowstyle='->', color=col, lw=1.1, shrinkA=5, shrinkB=5), zorder=2)
ax.set_xlim(xlo, xhi); ax.set_ylim(0, yhi)
frame(ax, fS, rH)
ax.set_title(TITLE + ': before (open) and after (filled) merging #305', color='#222222', fontsize=11)
ax.legend(fontsize=8.5, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False,
          title='folding, method  (CPU/N_eff before -> after, gain)', title_fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, NAME + '_shift.png'), dpi=150)
fig.savefig(os.path.join(OUT, NAME + '_shift.pdf'))

# 2) updated plot (new code only)
fig, ax = plt.subplots(figsize=(12.5, 5.6))
newp = [n for *_, o, n in rows]
fS, rH = isolines(ax, newp, xlo, xhi, yhi)
for T, fold, meth, o, n in rows:
    col = FOLDCOL[fold]; m = METHOD[meth]
    ax.scatter(n['x'], n['rS'], marker=m, s=70, facecolor=col, edgecolor='white', linewidth=1.2, zorder=3,
               label='%s %s  (r_H=%.3f, CPU/N_eff=%s)' % (fold, meth, n['rH'], fmt(n['ce'])))
ax.set_xlim(xlo, xhi); ax.set_ylim(0, yhi)
frame(ax, fS, rH)
ax.set_title(TITLE + ' (with #305: gscale + Born kept across folds)', color='#222222', fontsize=11)
ax.legend(fontsize=8.5, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False,
          title='folding, method  (r_H, CPU/N_eff)', title_fontsize=8.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, NAME + '.png'), dpi=150)
fig.savefig(os.path.join(OUT, NAME + '.pdf'))

with open(os.path.join(OUT, NAME + '.md'), 'w') as md:
    md.write('| folding | method | r_S | r | CPU/event before | after | CPU/N_eff before | after | gain |\n'
             '|---|---|---|---|---|---|---|---|---|\n')
    for T, fold, meth, o, n in rows:
        md.write('| %s | %s | %.3f | %.3f | %s | %s | %s | %s | x%.2f |\n' % (
            fold, meth, n['rS'], n['r'], fmt(o['x']), fmt(n['x']), fmt(o['ce']), fmt(n['ce']), o['ce'] / n['ce']))
for T, fold, meth, o, n in rows:
    print('%s %-6s %-16s rS %.4f/%.4f r %.4f/%.4f CPU/Neff %s -> %s (x%.2f)' % (
        T, fold, meth, o['rS'], n['rS'], o['r'], n['r'], fmt(o['ce']), fmt(n['ce']), o['ce'] / n['ce']))
