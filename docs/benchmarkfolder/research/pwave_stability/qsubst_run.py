"""The converse of the hybrid test (qsubst.py) at the points of an
estim.py run: the double-precision primary pass with the raised-power
integrals at full quadruple precision, contracted in quadruple precision
(modes 1 to 3 of qsubst.py), against the plain quadruple precision.

    qsubst_run.py OUTDIR ESTIM.json NWORKERS"""
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
work = os.path.join(os.path.dirname(os.path.abspath(outdir)), 'qsubst_' + os.path.basename(outdir.rstrip('/')))
os.makedirs(work, exist_ok=True)


def one(args):
    i, env = args
    os.environ.update(env)
    return olc.run_points(pdir, [pts[i]], None, warmup=0, timeout=36000)[0]['finite_norm']


def run(mode, envs):
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
    with concurrent.futures.ProcessPoolExecutor(nw) as pool:
        return list(pool.map(one, [(i, envs(i)) for i in range(len(pts))]))


dump = lambda i: os.path.join(work, 'q%03d.txt' % i)
out = dict(points=pts, qp_ref=est['qp'], ct1=est['ct1'])
out['qp'] = run('4', lambda i: {'ONIA_DBG_QDUMP': dump(i)})
for m in (1, 2, 3):
    out['qmode%d' % m] = run('1', lambda i, m=m: {'ONIA_DBG_QSUBST': dump(i), 'ONIA_DBG_QMODE': str(m)})
json.dump(out, open(work + '.json', 'w'))
rel = lambda k: [abs(x - q) / abs(q) for x, q in zip(out[k], out['qp'])]
print('QP with dumps against estim QP: largest %.1e' % max(abs(x - q) / abs(q) for x, q in zip(out['qp'], out['qp_ref'])))
for k, name in (('ct1', 'DP, as production'),
                ('qmode1', 'DP, QP doubled-line integrals, QC in DP, QP sum'),
                ('qmode2', 'DP, QP doubled-line integrals, QC and sum in QP'),
                ('qmode3', 'DP, doubled-line integrals rounded, QP sum')):
    e = rel(k)
    print('%-50s largest %.1e  median %.1e' % (name, max(e), statistics.median(e)))
