"""Substitute the QP array of one powers call at a time, in DP, at point K of
setup A (dumps BASE_qp.dump of run.py, its summary JSON).

    onebyone.py OUTDIR BASE K SUMMARY.json"""
import json, os, sys, concurrent.futures, shutil
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onia_loop_check as olc
outdir, base, k, summary = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
W = os.path.dirname(os.path.abspath(outdir))
sys.argv = [sys.argv[0], outdir, str(k + 1)]
import importlib.util
spec = importlib.util.spec_from_file_location('st', '/shared/external/madloop_onia/madgraph7_madloop_onia/docs/benchmarkfolder/research/stability/stability.py')
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
pdir = olc.subprocess_dir(outdir)
masses = olc.driver_masses(pdir)
p = st.make_points(masses, k + 1, 1000., 50.)[k]
def read(path):
    calls, cur = [], None
    for line in open(path):
        if line.startswith('C '):
            f = line.split(); cur = dict(idx=int(f[2]), kind=int(f[3]), n=int(f[4]), rank=int(f[5]), nc=int(f[6]), st=int(f[7]), rows=[]); calls.append(cur)
        else:
            cur['rows'].append(line.split())
    return calls
qp = read(base + '_qp.dump')
summ = [r for r in json.load(open(summary)) if r['point'] == k][0]
olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '1', 'MLStabThres': '1.0d99'})
def one(c):
    path = base + '_one%d.subst' % c['idx']
    with open(path, 'w') as f:
        f.write('1\n%d %d\n' % (c['idx'], c['nc']))
        for row in c['rows']:
            f.write(' '.join(row) + '\n')
    env = dict(os.environ, ONIA_DBG_SUBST=path)
    import subprocess
    # run_points uses os.environ; run in a subprocess-safe way via a thread-local copy
    old = os.environ.get('ONIA_DBG_SUBST')
    return c, path
calls = [c for c in qp if c['kind'] == 1 and c['st'] == 0]
out = {}
for c in calls:
    c, path = one(c)
    os.environ['ONIA_DBG_SUBST'] = path
    r = olc.run_points(pdir, [p], None, warmup=0, timeout=600)[0]
    out[c['idx']] = r['finite_norm'] - summ['dp']
    os.environ.pop('ONIA_DBG_SUBST')
q = summ['qp']
tot = summ['dp'] - q
print('point %d: dp - qp = %.3e (rel %.1e); sum of single-call shifts %.3e' % (k, tot, tot / abs(q), sum(out.values())))
for idx, d in sorted(out.items(), key=lambda kv: -abs(kv[1]))[:12]:
    c = [x for x in calls if x['idx'] == idx][0]
    print('  call %3d N%d rank %d: shift %.3e (%.0f%% of the error)' % (idx, c['n'], c['rank'], d, -100 * d / tot))
json.dump(out, open(base + '_onebyone.json', 'w'))
