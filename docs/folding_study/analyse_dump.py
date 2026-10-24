#!/usr/bin/env python3
"""Analyse fold_dump.out files written with FoldDump=1.

usage: analyse_dump.py fold_dump.dat chanA:dumpA1.out,dumpA2.out chanB:dumpB1.out ...

Each line of a dump: point, group, ns, nimages, snb(1:ns), scn(1:ns), habs, hsum
for one rule (line of fold_dump.dat) and one random shift (replica).
S = snb + scn is the S-event weight per Born flavour configuration.
Channels (P dirs) are averaged per point and then summed.
"""
import sys
import numpy as np

NAMES = {0: 'lattice', 1: 'tent', 2: 'cubic', 3: 'quintic', 4: 'sin2', 5: 'Simpson(det)'}


def read_cfg(fn):
    return [tuple(int(a) for a in l.split()[:6]) for l in open(fn) if len(l.split()) >= 6]


def cfg_label(c):
    tr, lat, nx, ny, nphi, nrep = c
    if tr == 5:
        n = (nx + 1) * (ny + 1) * nphi
        return '%s %dx%d' % (NAMES[tr], nx, ny), n
    if lat > 0:
        return '%s rank1 N=%d%s' % (NAMES[tr], lat, ' x%d' % nphi if nphi > 1 else ''), lat * nphi
    return '%s %dx%dx%d' % (NAMES[tr], nx, ny, nphi), nx * ny * nphi


def read_dump(fns, ngroups):
    """returns list of points; each point: (S[ng,ns], Snb[ng,ns], H[ng,2])"""
    pts = []
    for fn in fns:
        cur = {}
        for line in open(fn):
            v = line.split()
            p, g, ns = int(v[0]), int(v[1]) - 1, int(v[2])
            w = np.array([float(a) for a in v[4:]])
            if p not in cur:
                cur[p] = [np.zeros((ngroups, ns)), np.zeros((ngroups, ns)), np.zeros((ngroups, 2)), 0, ns]
            e = cur[p]
            if ns != e[4]:
                # rare: a group with a different number of Born flavours (no S)
                ns2 = min(ns, e[4])
            e[1][g, :ns] = w[:ns]
            e[0][g, :ns] = w[:ns] + w[ns:2 * ns]
            e[2][g] = w[2 * ns:2 * ns + 2]
            e[3] += 1
        for p in sorted(cur):
            e = cur[p]
            if e[3] == ngroups:
                pts.append((e[0], e[1], e[2]))
    return pts


def stats(pts, gsel, rsel, nb_exact=False):
    npt = len(pts)
    acc = dict(absS=0., negS=0., absH=0., negH=0., sumS=0., refsum=0., err2=0., ref2=0., var=0.)
    rel, relw = [], []
    for S, Snb, H in pts:
        h = H[gsel]
        acc['absH'] += h[:, 0].mean()
        acc['negH'] += 0.5 * (h[:, 0] - h[:, 1]).mean()
        if S.shape[1] == 0:
            continue
        ref = S[rsel].mean(axis=0)
        est = S[gsel].copy()
        if nb_exact:
            est = est - Snb[gsel] + Snb[rsel].mean(axis=0)[None, :]
        acc['absS'] += np.abs(est).sum(axis=1).mean()
        acc['negS'] += (np.abs(est) * (est < 0)).sum(axis=1).mean()
        acc['sumS'] += est.sum(axis=1).mean()
        acc['refsum'] += ref.sum()
        d = est - ref[None, :]
        acc['err2'] += (d ** 2).sum(axis=1).mean()
        acc['ref2'] += (ref ** 2).sum()
        if est.shape[0] > 1:
            acc['var'] += est.var(axis=0, ddof=1).sum()
        for j in range(S.shape[1]):
            if ref[j] != 0:
                rel.append(np.abs(d[:, j]).mean() / abs(ref[j]))
                relw.append(abs(ref[j]))
    for k in acc:
        acc[k] /= npt
    acc['rel'] = np.array(rel)
    acc['relw'] = np.array(relw)
    return acc


def main():
    cfg = read_cfg(sys.argv[1])
    groups = np.array([ic for ic, c in enumerate(cfg) for r in range(c[5])])
    chans = []
    for a in sys.argv[2:]:
        name, fns = a.split(':')
        pts = read_dump(fns.split(','), len(groups))
        chans.append((name, pts))
        print('channel %s: %d Born points' % (name, len(pts)))
    iref = 0
    print('reference rule (per Born point): %s, %d nodes' % cfg_label(cfg[iref]))
    hdr = ('%-26s %5s | %7s %7s %6s | %7s %7s %6s | %7s %7s %7s %8s' % (
        'rule', 'nodes', 'r_S', 'r_tot', 'Neff', 'r_S*', 'r_tot*', 'Neff*',
        'rmsErr', 'medRel', 'wRel', 'bias'))
    print(hdr)
    print('  (* = n-body weight replaced by its exact radiation integral)')
    for ic, c in enumerate(cfg):
        gsel = np.where(groups == ic)[0]
        rsel = np.where(groups == iref)[0]
        tot = {}
        tot2 = {}
        rel, relw = [], []
        for name, pts in chans:
            a = stats(pts, gsel, rsel)
            b = stats(pts, gsel, rsel, nb_exact=True)
            for k in ('absS', 'negS', 'absH', 'negH', 'sumS', 'refsum', 'err2', 'ref2', 'var'):
                tot[k] = tot.get(k, 0) + a[k]
                tot2[k] = tot2.get(k, 0) + b[k]
            rel.append(a['rel'])
            relw.append(a['relw'])
        rel = np.concatenate(rel)
        relw = np.concatenate(relw)
        lab, n = cfg_label(c)

        def rr(t):
            rS = t['negS'] / t['absS']
            rt = (t['negS'] + t['negH']) / (t['absS'] + t['absH'])
            return rS, rt, (1 - 2 * rt) ** 2
        print('%-26s %5d | %7.4f %7.4f %6.3f | %7.4f %7.4f %6.3f | %7.4f %7.4f %7.4f %+8.4f' % (
            (lab, n) + rr(tot) + rr(tot2) + (np.sqrt(tot['err2'] / tot['ref2']), np.median(rel),
                                             (rel * relw).sum() / relw.sum(),
                                             tot['sumS'] / tot['refsum'] - 1)))


if __name__ == '__main__':
    main()
