"""Compare MadLoop's onia virtuals with the MG5 2.5.5 V0 references.
compare.py <case> [mu]: points = the reference's PS*.input (deduplicated)
and random points; the reference in the decoupling scheme (1), direct
evaluation (calcoption 3) and its default; MadLoop through ml_side.py."""
import glob, json, math, os, subprocess, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import tirust_benchmark_points as points_lib
SP = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad'
R = '/shared/external/madloop_onia/references/NLO_Onia_Nov2026'
CF, CA = 4. / 3., 3.
# case: reference V0, build dir, m_Q, ONIUM_FACTOR, Casimir sum, default calcoption, energies
# (gg_jpsig: the user's Jpsi_double_cs_NEW, with a direct evaluation, at the points of
# gg_etacg's PS*.input too; the 2.5.5_Onia V0 has only its grid)
CASES = {
    'gg_etacg': ('2.5.5_Onia1S01/pp2etacj_Grid_FKS_MultiChan/SubProcesses/P0_gg_ccxg/V0_gg_ccxg',
                 'etac/gg', 1.55, 1.16 / 18 / (3 * 1.55), 3 * CA, 3, (5., 11., 40.)),
    'gu_etacu': ('2.5.5_Onia1S01/pp2etacj_Grid_FKS_MultiChan/SubProcesses/P0_gu_ccxu/V0_gu_ccxu',
                 'etac/gu', 1.55, 1.16 / 18 / (3 * 1.55), CA + 2 * CF, 3, (5., 11., 40.)),
    'ug_etacu': ('2.5.5_Onia1S01/pp2etacj_Grid_FKS_MultiChan/SubProcesses/P0_ug_ccxu/V0_ug_ccxu',
                 'etac/ug', 1.55, 1.16 / 18 / (3 * 1.55), CA + 2 * CF, 3, (5., 11., 40.)),
    'uux_etacg': ('2.5.5_Onia1S01/pp2etacj_Grid_FKS_MultiChan/SubProcesses/P0_uux_ccxg/V0_uux_ccxg',
                  'etac/uux', 1.55, 1.16 / 18 / (3 * 1.55), CA + 2 * CF, 3, (5., 11., 40.)),
    'gg_jpsig': ('Jpsi_double_cs_NEW', 'jpsi_new/gg', 1.55, 1.16 / (18 * 3 * 1.55), 3 * CA, 1,
                 (5., 11., 40.)),
    'gg_etabg': ('2.5.5_bOnia1S01/pp2etabj_Grid_FKS_MultiChan/SubProcesses/P0_gg_bbxg/V0_gg_bbxg',
                 'etab/gg', 4.75, 9.28 / 18 / (3 * 4.75), 3 * CA, 1, (11., 20., 80.)),
    'gu_etabu': ('2.5.5_bOnia1S01/pp2etabj_Grid_FKS_MultiChan/SubProcesses/P0_gu_bbxu/V0_gu_bbxu',
                 'etab/gu', 4.75, 9.28 / 18 / (3 * 4.75), CA + 2 * CF, 1, (11., 20., 80.)),
    'ug_etabu': ('2.5.5_bOnia1S01/pp2etabj_Grid_FKS_MultiChan/SubProcesses/P0_ug_bbxu/V0_ug_bbxu',
                 'etab/ug', 4.75, 9.28 / 18 / (3 * 4.75), CA + 2 * CF, 1, (11., 20., 80.)),
    'uux_etabg': ('2.5.5_bOnia1S01/pp2etabj_Grid_FKS_MultiChan/SubProcesses/P0_uux_bbxg/V0_uux_bbxg',
                  'etab/uux', 4.75, 9.28 / 18 / (3 * 4.75), CA + 2 * CF, 1, (11., 20., 80.)),
    'gg_upsg': ('2.5.5_bOnia3S11/pp2Yj_Grid_FKS_MultiChan/SubProcesses/P0_gg_bbxg/V0_gg_bbxg',
                'ups/gg', 4.75, 9.28 / 18 / (3 * 4.75), 3 * CA, 1, (11., 20., 80.)),
}
# cases evaluated at the PS*.input points of another reference (same masses)
POINTS_FROM = {'gg_jpsig': CASES['gg_etacg'][0]}
G = math.sqrt(4 * math.pi * 0.118)
A = 0.118 / (2 * math.pi)

def ref_points(src):
    pts, seen = [], set()
    for f in sorted(glob.glob(os.path.join(R, src, 'PS*.input'))):
        rows = [[float(x) for x in l.split()] for l in open(f) if l.strip()]
        key = tuple(round(c, 10) for r in rows for c in r)
        if len(rows) == 5 and key not in seen:
            seen.add(key)
            pts.append((os.path.basename(f), rows))
    return pts

def random_points(mq, energies, n=3):
    out = []
    for e in energies:
        for p in points_lib.generate_points([2 * mq, 0.], count=n, energy=e, seed=20261121):
            k = p['momenta']
            half = [0.5 * c for c in k[2]]
            out.append(('rnd%g' % e, [k[0], k[1], half, list(half), k[3]]))
    return out

def run_ref(bdir, mq, of, scheme, opt, pts, mu):
    lines = ['%.17e %.17e %.17e %.17e %d %d' % (G, mu, mq, of, scheme, opt if opt is not None else 0),
             str(len(pts))]
    for _, rows in pts:
        lines += ['%.17e %.17e %.17e %.17e' % tuple(r) for r in rows]
    # the references read their grids from grid_dir, "./" or "../" (FKS layout)
    up = any('grid_dir="../"' in open(f).read() for f in glob.glob(os.path.join(bdir, '*.f90')))
    run = os.path.join(bdir, 'run') if up else bdir
    os.makedirs(run, exist_ok=True)
    proc = subprocess.run([os.path.join(bdir, 'refdrv')], cwd=run, input='\n'.join(lines) + '\n',
                          capture_output=True, text=True, timeout=3600)
    out = proc.stdout
    if out.count('\nV ') + out.startswith('V ') != len(pts):
        raise RuntimeError('reference failed in %s: %s' % (bdir, (out + proc.stderr)[-800:]))
    vals = [[float(x) for x in l.split()[1:]] for l in out.splitlines() if l.startswith('V ')]
    pols = [[float(x) for x in l.split()[1:]] for l in out.splitlines() if l.startswith('P ')]
    return vals, pols

def main(case, mu):
    src, b, mq, of, csum, dopt, energies = CASES[case]
    bdir = os.path.join(SP, 'swj', 'ref', b)
    pts = ref_points(POINTS_FROM.get(case, src)) + random_points(mq, energies)
    opts = [3] + ([dopt] if dopt not in (None, 3) else []) if dopt is not None else [None]
    ref, pol = {}, {}
    for o in opts:
        ref[o], pol[o] = run_ref(bdir, mq, of, 1, o, pts, mu)
    pf = os.path.join(SP, 'swj', 'pts_%s.json' % case)
    json.dump([r for _, r in pts], open(pf, 'w'))
    mf = os.path.join(SP, 'swj', 'ml_%s.json' % case)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ml_side.py'),
                    case, pf, str(mu), mf],
                   check=True, capture_output=True)
    ml = json.load(open(mf))
    rows = []
    for i, (tag, rows_) in enumerate(pts):
        s = (rows_[0][0] + rows_[1][0]) ** 2 - sum((rows_[0][j] + rows_[1][j]) ** 2 for j in (1, 2, 3))
        m = ml[i]
        entry = dict(tag=tag, sqrts=math.sqrt(s), ml=dict(born=m['born'], fin=m['finite_norm'],
                     c1=m['single_norm'], c2=m['double_norm'], code=m['ret_code'],
                     accuracy=m['accuracy']), ref={})
        for o in opts:
            v = ref[o][i]
            bn = -v[2] / csum
            entry['ref'][str(o)] = dict(born=bn / A, fin=v[0] / bn, c1=v[1] / bn, c2=v[2] / bn)
            if pol[o]:
                # transverse and longitudinal onium, each in units of its own Born
                for k, name in ((0, 'T'), (3, 'L')):
                    w = pol[o][i][k:k + 3]
                    b = -w[2] / csum
                    entry['ref'][str(o)][name] = dict(born=b / A, fin=w[0] / b, c1=w[1] / b, c2=w[2] / b)
        rows.append(entry)
    json.dump(rows, open(os.path.join(SP, 'swj', 'cmp_%s.json' % case), 'w'), indent=1)
    o = str(opts[0])
    print('%s mu=%g  (reference calcoption %s; rel. diff of fin, abs. diff of c1)' % (case, mu, o))
    for e in rows:
        r, m = e['ref'][o], e['ml']
        extra = ''
        for oo in opts[1:]:
            extra += ' | opt%s fin %.10f' % (oo, e['ref'][str(oo)]['fin'])
        print('%-9s rs %8.4f  ML fin %15.9f c1 %12.8f code %s  REF fin %15.9f c1 %12.8f  dfin %.1e dc1 %.1e born ratio %.6f%s' % (
            e['tag'], e['sqrts'], m['fin'], m['c1'], m['code'], r['fin'], r['c1'],
            abs(m['fin'] - r['fin']) / max(abs(r['fin']), 1e-300), abs(m['c1'] - r['c1']),
            m['born'] / r['born'], extra))

if __name__ == '__main__':
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 11.0)
