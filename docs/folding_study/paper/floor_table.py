"""Negative fractions of the 8,4,1 cubic+FoldSumSets runs, compared with 2,2,1 and 4,4,1.

usage: floor_table.py <scratch fold dir>   (writes floor_rows.txt and prints the rows)
"""
import sys, gzip, os, re, math

S = sys.argv[1]
# process: (directory of the 8,4,1 run, H-event particle count, earlier 2,2,1 and 4,4,1 cubic+sum runs)
PROCS = [
    ('$pp\\to t\\bar t$', 'F_tt', 5, [('T1_tt_v', 'q21K'), ('T1_tt_v', 'q21E')]),
    ('$pp\\to e^+e^-j$', 'F_eej', 7, [('T2_eej_v', 'q21K'), ('T2_eej_v', 'q21E')]),
    ('$e^+e^-\\to b\\bar b b\\bar b$', 'F_ee4b', 7, [('T3_ee4b_g', 'h21K'), ('T3_ee4b_g', 'h21E')]),
    ('$pp\\to Zjj$', 'F_zjj', 6, [('T5_zjj_new', 'n21K'), ('T5_zjj_new', 'n21E')]),
]


def counts(D, tag, hsize):
    ev = os.path.join(S, D, 'Events', tag, 'events.lhe.gz')
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
                if int(v[0]) == hsize:
                    nH += 1; nHn += neg
                else:
                    nS += 1; nSn += neg
                inev = False
    return nS, nSn, nH, nHn


def nvirt(D, tag):
    d = os.path.join(S, D, 'vlogs', tag)
    v = 0
    for f in os.listdir(d):
        if f.endswith('log_MINT2.txt'):
            for l in open(os.path.join(d, f), errors='replace'):
                if 'events generated, virt' in l:
                    v += int(l.split()[-1])
    return v


def fmt(nS, nSn, nH, nHn, nv):
    rS = nSn / nS; rH = nHn / nH; r = (nSn + nHn) / (nS + nH)
    e = math.sqrt(rS * (1 - rS) / nS)
    # virt events are S events with the sign of V-A: about half of them negative
    return rS, e, rH, r, 0.5 * nv / nS


rows = []
for name, d8, hs, prev in PROCS:
    vals = []
    for D, tag in prev + [(d8, 'f21C')]:
        c = counts(D, tag, hs)
        vals.append(fmt(*c, nvirt(D, tag)) if c else None)
    rows.append((name, vals))
    print(name, ['%.4f+-%.4f (rH %.3f, r %.3f, virt %.4f)' % v if v else 'missing' for v in vals])
with open('floor_rows.txt', 'w') as out:
    for name, vals in rows:
        cells = ['$%.3f\\pm%.3f$ (%.3f)' % (v[0], v[1], v[4]) if v else '--' for v in vals]
        rh = vals[-1][2] if vals[-1] else float('nan')
        rr = ['%.3f' % v[3] if v else '--' for v in vals]
        out.write('%s & %s & %s & %s & %.3f & %s & %s & %s \\\\\n' % (name, cells[0], cells[1], cells[2], rh, rr[0], rr[1], rr[2]))
