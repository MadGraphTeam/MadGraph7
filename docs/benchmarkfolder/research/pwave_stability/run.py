"""u u~ > chic0 g at setup A (1 TeV, pT > 50): DP (CTModeRun 1), QP
(CTModeRun 4), and DP with the QP TIRust arrays substituted (all, plain only,
powers only). Usage: run.py OUTDIR NPOINTS [ENERGY PTMIN]"""
import json, os, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/docs/benchmarkfolder/research/stability')
import onia_loop_check as olc
import stability as st
outdir, npts = sys.argv[1], int(sys.argv[2])
energy = float(sys.argv[3]) if len(sys.argv) > 3 else 1000.
ptmin = float(sys.argv[4]) if len(sys.argv) > 4 else 50.
W = os.path.dirname(os.path.abspath(outdir))
pdir = olc.subprocess_dir(outdir)
olc.build_driver(pdir)
masses = olc.driver_masses(pdir)
points = st.make_points(masses, npts, energy, ptmin)
IDX = [int(x) for x in os.environ.get('IDX', '').split(',') if x] or list(range(npts))
tag = os.path.basename(outdir.rstrip('/')) + '_%g' % energy

def read_dump(path):
    calls, cur = [], None
    for line in open(path):
        if line.startswith('C '):
            f = line.split()
            cur = dict(prec=f[1], idx=int(f[2]), kind=int(f[3]), n=int(f[4]), rank=int(f[5]),
                       nc=int(f[6]), status=int(f[7]), rows=[])
            calls.append(cur)
        else:
            cur['rows'].append(line.split())
    return calls

def run(mode, p, dump=None, subst=None):
    for k in ('ONIA_DBG_DUMP', 'ONIA_DBG_SUBST'):
        os.environ.pop(k, None)
    if dump:
        os.environ['ONIA_DBG_DUMP'] = dump
    if subst:
        os.environ['ONIA_DBG_SUBST'] = subst
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
    r = olc.run_points(pdir, [p], None, warmup=0, timeout=3600)[0]
    for k in ('ONIA_DBG_DUMP', 'ONIA_DBG_SUBST'):
        os.environ.pop(k, None)
    return r

def write_subst(path, qp, kinds, nmin=0):
    sel = [c for c in qp if c['kind'] in kinds and c['status'] == 0 and c['n'] >= nmin]
    with open(path, 'w') as f:
        f.write('%d\n' % len(sel))
        for c in sel:
            f.write('%d %d\n' % (c['idx'], c['nc']))
            for row in c['rows']:
                f.write(' '.join(row) + '\n')

res = []
for k, p in enumerate(points):
    if k not in IDX:
        continue
    base = os.path.join(W, '%s_p%d' % (tag, k))
    dp = run('1', p, dump=base + '_dp.dump')
    qp = run('4', p, dump=base + '_qp.dump')
    cdp, cqp = read_dump(base + '_dp.dump'), read_dump(base + '_qp.dump')
    sig = lambda c: (c['kind'], c['n'], c['rank'], c['nc'])
    aligned = len(cdp) == len(cqp) and all(sig(a) == sig(b) for a, b in zip(cdp, cqp))
    row = dict(point=k, dp=dp['finite_norm'], qp=qp['finite_norm'], born_dp=dp['born'], born_qp=qp['born'],
               c1_dp=dp['single_norm'], c1_qp=qp['single_norm'], code_dp=dp['ret_code'],
               ncalls=len(cdp), npow=sum(1 for c in cdp if c['kind'] == 1), aligned=aligned)
    if aligned:
        for name, kinds, nmin in (('sub_all', (0, 1), 0), ('sub_plain', (0,), 0), ('sub_pow', (1,), 0), ('sub_pow5', (1,), 5)):
            write_subst(base + '_%s.subst' % name, cqp, kinds, nmin)
            row[name] = run('1', p, subst=base + '_%s.subst' % name)['finite_norm']
    q = row['qp']
    print('p%d err dp %.1e | sub_all %.1e plain %.1e pow %.1e pow5 %.1e | born %.1e | calls %d (pow %d) aligned %s' % (
        k, abs(row['dp'] - q) / abs(q), abs(row.get('sub_all', 0) - q) / abs(q),
        abs(row.get('sub_plain', 0) - q) / abs(q), abs(row.get('sub_pow', 0) - q) / abs(q),
        abs(row.get('sub_pow5', 0) - q) / abs(q), abs(row['born_dp'] - row['born_qp']) / abs(row['born_qp']), row['ncalls'], row['npow'], aligned), flush=True)
    res.append(row)
json.dump(res, open(os.path.join(W, '%s_summary%s.json' % (tag, os.environ.get('IDX', '').replace(',', '_'))), 'w'), indent=1)
