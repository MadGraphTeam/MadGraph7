"""DP check pass with and without TIRust's invariant shift (env ONIA_DBG_NOSHIFT),
against QP of estim.py, all points."""
import concurrent.futures, json, os, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
outdir, est = sys.argv[1], json.load(open(sys.argv[2]))
pdir = olc.subprocess_dir(outdir)
olc.build_driver(pdir)
olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '2', 'MLStabThres': '1.0d99'})
pts = est['points']
os.environ['ONIA_DBG_NOSHIFT'] = '1'
r = olc.run_points(pdir, pts, None, warmup=0, timeout=3600)
os.environ.pop('ONIA_DBG_NOSHIFT')
ns = [x['finite_norm'] for x in r]
rel = lambda a, i: abs(a - est['qp'][i]) / abs(est['qp'][i])
print('point   true(ct1)  ct2(shift)  ct2(no shift) |ct1-ct2| |ct1-ct2noshift|')
for i in range(len(pts)):
    print('p%-3d   %.1e    %.1e    %.1e       %.1e   %.1e' % (i, rel(est['ct1'][i], i), rel(est['ct2'][i], i), rel(ns[i], i),
          abs(est['ct1'][i] - est['ct2'][i]) / abs(est['qp'][i]), abs(est['ct1'][i] - ns[i]) / abs(est['qp'][i])))
json.dump(ns, open(os.path.join(os.path.dirname(outdir), 'ct2_noshift.json'), 'w'))
