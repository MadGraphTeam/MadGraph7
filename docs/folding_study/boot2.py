import sys, numpy as np
sys.argv=[sys.argv[0]]+sys.argv[1:]
from analyse_dump2 import read_cfg, cfg_label, read_dump
cfg = read_cfg(sys.argv[1])
groups = [(ic, r) for ic, c in enumerate(cfg) for r in range(c[5])]
pts = []
for a in sys.argv[2:]:
    name, fns = a.split(':')
    pts += read_dump(fns.split(','), len(groups))
rng = np.random.default_rng(1)
B = 300
print('%-22s %5s  %-22s %-22s' % ('rule', 'nodes', 'r_S MC (boot err)', 'r_S SUM (boot err)'))
for ic, c in enumerate(cfg):
    negMC, absMC, negSU, absSU = [], [], [], []
    for e in pts:
        G = e['g']; nsets = e['nsets']
        mine = {}
        for gi, (jc, r) in enumerate(groups):
            if jc != ic: continue
            iset, vol1, ns, snb, scn, H = G[gi]
            mine.setdefault(iset, []).append((snb + scn, vol1))
        lens = set(len(a[0]) for v in mine.values() for a in v)
        if len(mine) != nsets or len(lens) != 1 or 0 in lens:
            negMC.append(0); absMC.append(0); negSU.append(0); absSU.append(0); continue
        nm = am = 0.; summ = 0.
        for k, v in mine.items():
            Sk = np.array([a[0] for a in v]); pk = v[0][1]
            am += pk * np.abs(Sk).sum(axis=1).mean()
            nm += pk * (np.abs(Sk) * (Sk < 0)).sum(axis=1).mean()
            summ = summ + pk * Sk.mean(axis=0)
        negMC.append(nm); absMC.append(am)
        negSU.append((np.abs(summ) * (summ < 0)).sum()); absSU.append(np.abs(summ).sum())
    negMC, absMC, negSU, absSU = map(np.array, (negMC, absMC, negSU, absSU))
    n = len(negMC)
    bm, bs = [], []
    for b in range(B):
        i = rng.integers(0, n, n)
        bm.append(negMC[i].sum() / absMC[i].sum()); bs.append(negSU[i].sum() / absSU[i].sum())
    lab, nn = cfg_label(c)
    print('%-22s %5d  %.4f +- %.4f        %.4f +- %.4f' % (lab, nn, negMC.sum() / absMC.sum(), np.std(bm), negSU.sum() / absSU.sum(), np.std(bs)))
