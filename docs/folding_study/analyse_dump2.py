#!/usr/bin/env python3
"""Analyse FoldDump=2 dumps (replica r of each rule uses FKS set (r-1)%nsets+1).

usage: analyse_dump2.py fold_dump2.dat chanA:dA1.out,dA2.out chanB:...

dump line: point group ns nimages iset nsets vol1 snb(1:ns) scn(1:ns) habs hsum

S_ck = snb + scn for rule c and FKS set k already contains the 1/p_k
(n+1-body) and 1/(N p_k) (n-body) factors of the MC over sets.
Schemes for the S events:
  MC     : today: one set per point, picked with p_k
  MCnb   : MC over sets, but the n-body weight does not depend on the set
           (alpha_k = p_k instead of 1/N)
  SUM    : the S contributions of all sets are summed (cost x nsets for the
           S part of each image)
"""
import sys
import numpy as np

NAMES = {0: 'lattice', 1: 'tent', 2: 'cubic', 3: 'quintic', 4: 'sin2', 5: 'Simpson(det)'}


def read_cfg(fn):
    return [tuple(int(a) for a in l.split()[:6]) for l in open(fn) if len(l.split()) >= 6]


def cfg_label(c):
    tr, lat, nx, ny, nphi, nrep = c
    if tr == 5:
        return '%s %dx%d' % (NAMES[tr], nx, ny), (nx + 1) * (ny + 1) * nphi
    if lat > 0:
        return '%s rank1 N=%d' % (NAMES[tr], lat), lat * nphi
    return '%s %dx%dx%d' % (NAMES[tr], nx, ny, nphi), nx * ny * nphi


def read_dump(fns, ngroups):
    pts = []
    for fn in fns:
        cur = {}
        for line in open(fn):
            v = line.split()
            p, g, ns, nimg, iset, nsets = (int(a) for a in v[:6])
            w = np.array([float(a) for a in v[6:]])
            vol1 = w[0]
            w = w[1:]
            e = cur.setdefault(p, {'g': {}, 'nsets': nsets})
            e['g'][g - 1] = (iset, vol1, ns, w[:ns].copy(), w[ns:2 * ns].copy(), w[2 * ns:2 * ns + 2].copy())
        for p in sorted(cur):
            if len(cur[p]['g']) == ngroups:
                pts.append(cur[p])
    return pts


def main():
    cfg = read_cfg(sys.argv[1])
    groups = [(ic, r) for ic, c in enumerate(cfg) for r in range(c[5])]
    chans = []
    for a in sys.argv[2:]:
        name, fns = a.split(':')
        pts = read_dump(fns.split(','), len(groups))
        chans.append((name, pts))
        print('channel %s: %d Born points, nsets=%s' % (name, len(pts), pts[0]['nsets'] if pts else '?'))
    print()
    print('Precision of the rule at fixed FKS set, relative to the cubic 16x16 reference')
    print('  dS/S   : S weight of the set (n-body + integrated n+1-body S terms)')
    print('  dI/I   : n+1-body S terms alone (the radiation integral of the counterterms)')
    print('  f<1%   : fraction of (point,set,flavour) with |dS/S|<1%  (|S|-weighted)')
    print('%-22s %5s | %7s %7s %6s | %7s %7s | %8s' % ('rule', 'nodes', 'med dS/S', 'w dS/S', 'f<1%',
                                                       'med dI/I', 'w dI/I', 'bias(S)'))
    res = {}
    for ic, c in enumerate(cfg):
        lab, n = cfg_label(c)
        dS, wS, dI, sb, sr = [], [], [], 0., 0.
        acc = {s: dict(abs=0., neg=0., absH=0., negH=0., nb_abs=0., nb_neg=0., nbx_abs=0., nbx_neg=0.)
               for s in ('MC', 'MCnb', 'SUM', 'SUMx')}
        npt_tot = 0
        for name, pts in chans:
            for e in pts:
                npt_tot += 1
                G = e['g']
                nsets = e['nsets']
                # reference per set
                ref = {}
                refnb = {}
                for gi, (jc, r) in enumerate(groups):
                    if jc != 0:
                        continue
                    iset, vol1, ns, snb, scn, H = G[gi]
                    ref.setdefault(iset, []).append((snb + scn, scn, snb, vol1))
                # this rule, per set
                mine = {}
                Hk = {}
                for gi, (jc, r) in enumerate(groups):
                    if jc != ic:
                        continue
                    iset, vol1, ns, snb, scn, H = G[gi]
                    mine.setdefault(iset, []).append((snb + scn, scn, snb, vol1))
                    Hk.setdefault(iset, []).append((H, vol1))
                absH = sum(np.mean([h[0][0] for h in v]) * v[0][1] for v in Hk.values())
                negH = sum(np.mean([0.5 * (h[0][0] - h[0][1]) for h in v]) * v[0][1] for v in Hk.values())
                for s in acc:
                    acc[s]['absH'] += absH
                    acc[s]['negH'] += negH
                # fixed-set precision
                for k, lst in mine.items():
                    if k not in ref:
                        continue
                    lens = [len(a[0]) for a in ref[k]]
                    if min(lens) != max(lens) or lens[0] == 0:
                        continue
                    rS = np.mean([a[0] for a in ref[k]], axis=0)
                    rI = np.mean([a[1] for a in ref[k]], axis=0)
                    for S_, I_, nb_, v_ in lst:
                        if len(S_) != len(rS):
                            continue
                        for j in range(len(rS)):
                            if rS[j] == 0:
                                continue
                            dS.append(abs(S_[j] - rS[j]) / abs(rS[j]))
                            wS.append(abs(rS[j]) * v_)
                            if rI[j] != 0:
                                dI.append(abs(I_[j] - rI[j]) / abs(rI[j]))
                        sb += (S_.sum()) * v_
                        sr += rS.sum() * v_
                # negative weights per scheme
                ks = sorted(mine)
                ns_all = set(len(a[0]) for k in ks for a in mine[k]) | set(len(a[0]) for k in ref for a in ref[k])
                if len(ks) != nsets or len(ref) != nsets or len(ns_all) != 1 or 0 in ns_all:
                    # the H events are counted above; points where some set
                    # has no S contribution are left out of the S statistics
                    continue
                # MC over sets: E = sum_k p_k f(S_k) (average over replicas of set k)
                summ = 0.
                summx = 0.
                for k in ks:
                    Sk = np.array([a[0] for a in mine[k]])
                    nbk = np.array([a[2] for a in mine[k]])
                    pk = mine[k][0][3]
                    acc['MC']['abs'] += pk * np.abs(Sk).sum(axis=1).mean()
                    acc['MC']['neg'] += pk * (np.abs(Sk) * (Sk < 0)).sum(axis=1).mean()
                    Snb = Sk - nbk + nbk * nsets * pk
                    acc['MCnb']['abs'] += pk * np.abs(Snb).sum(axis=1).mean()
                    acc['MCnb']['neg'] += pk * (np.abs(Snb) * (Snb < 0)).sum(axis=1).mean()
                    summ = summ + pk * Sk.mean(axis=0)
                    # with exact n-body (reference n-body of the set, rescaled)
                    rnb = np.mean([a[2] for a in ref[k]], axis=0)
                    summx = summx + pk * (Sk - nbk + rnb[None, :]).mean(axis=0)
                acc['SUM']['abs'] += np.abs(summ).sum()
                acc['SUM']['neg'] += (np.abs(summ) * (summ < 0)).sum()
                acc['SUMx']['abs'] += np.abs(summx).sum()
                acc['SUMx']['neg'] += (np.abs(summx) * (summx < 0)).sum()
        dS, wS, dI = np.array(dS), np.array(wS), np.array(dI)
        print('%-22s %5d | %7.4f %7.4f %6.3f | %7.4f %7.4f | %+8.4f' % (
            lab, n, np.median(dS), (dS * wS).sum() / wS.sum(), (wS * (dS < 0.01)).sum() / wS.sum(),
            np.median(dI), np.mean(np.minimum(dI, 10)), sb / sr - 1))
        res[ic] = acc
    print()
    print('Negative weights (r_S: S events only; r: all events incl. H; Neff=(1-2r)^2)')
    print('  MC   = MC over FKS sets (today), MCnb = MC over sets with set-independent n-body,')
    print('  SUM  = S terms summed over the FKS sets, SUMx = SUM with exact n-body integral')
    print('%-22s %5s | %s' % ('rule', 'nodes', ' | '.join('%-4s r_S     r   Neff' % s for s in ('MC', 'MCnb', 'SUM', 'SUMx'))))
    for ic, c in enumerate(cfg):
        lab, n = cfg_label(c)
        out = []
        for s in ('MC', 'MCnb', 'SUM', 'SUMx'):
            a = res[ic][s]
            rS = a['neg'] / a['abs']
            r = (a['neg'] + a['negH']) / (a['abs'] + a['absH'])
            out.append('     %6.4f %6.4f %5.3f' % (rS, r, (1 - 2 * r) ** 2))
        print('%-22s %5d |%s' % (lab, n, ' |'.join(out)))


if __name__ == '__main__':
    main()
