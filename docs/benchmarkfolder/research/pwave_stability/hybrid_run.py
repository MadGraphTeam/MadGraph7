"""The hybrid test (hybrid.py) at the points of an estim.py run: MadLoop in
quadruple precision (CTModeRun 4) with TIRust's double-precision integrals,
for the calls with raised powers (ONIA_DBG_DPTIR=1) and for every call (=2),
against the plain quadruple precision and the double-precision primary pass
of estim.py.

    hybrid_run.py OUTDIR ESTIM.json NWORKERS"""
import concurrent.futures
import json
import os
import statistics
import sys

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc

outdir, est, nw = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3])
pdir = olc.subprocess_dir(outdir)
pts = est['points']


def run(level):
    os.environ['ONIA_DBG_DPTIR'] = str(level)
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '4', 'MLStabThres': '1.0d99'})
    chunks = [pts[i::nw] for i in range(nw)]
    with concurrent.futures.ThreadPoolExecutor(nw) as pool:
        res = list(pool.map(lambda c: olc.run_points(pdir, c, None, warmup=0, timeout=36000) if c else [], chunks))
    os.environ.pop('ONIA_DBG_DPTIR')
    return [res[i % nw][i // nw]['finite_norm'] for i in range(len(pts))]


out = dict(points=pts, qp=est['qp'], ct1=est['ct1'])
for level in (1, 2):
    out['dptir%d' % level] = run(level)
rel = lambda k: [abs(x - q) / abs(q) for x, q in zip(out[k], out['qp'])]
tag = os.path.join(os.path.dirname(os.path.abspath(outdir)), 'hybrid_%s.json' % os.path.basename(outdir.rstrip('/')))
json.dump(out, open(tag, 'w'))
for k, name in (('ct1', 'DP MadLoop, DP integrals'), ('dptir1', 'QP MadLoop, DP raised-power integrals'),
                ('dptir2', 'QP MadLoop, DP integrals (all)')):
    e = rel(k)
    print('%-40s largest %.1e  median %.1e' % (name, max(e), statistics.median(e)))
for i in range(len(pts)):
    print('p%-3d ' % i + '  '.join('%s %.1e' % (k, rel(k)[i]) for k in ('ct1', 'dptir1', 'dptir2')))
