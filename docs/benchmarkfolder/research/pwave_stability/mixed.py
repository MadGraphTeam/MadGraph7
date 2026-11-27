"""Mixed precision prototype: DP passes with the powers calls of at least
ONIA_DBG_MIXED lines evaluated by TIRust in QP. Errors against QP of
estim.py, the MadLoop estimate (CTModeRun -1) and time per point."""
import json, os, sys, statistics
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
outdir, est, nmin = sys.argv[1], json.load(open(sys.argv[2])), sys.argv[3]
pdir = olc.subprocess_dir(outdir)
olc.build_driver(pdir)
pts = est['points']
res = {}
os.environ['ONIA_DBG_MIXED'] = nmin
for name, mode, extra in (('ct1', '1', {}), ('ct2', '2', {}), ('prod', '-1', {'NRotations_DP': '1'})):
    olc.set_reduction(outdir, ['9'], extra=dict({'CTModeRun': mode, 'MLStabThres': '1.0d99'}, **extra))
    res[name] = olc.run_points(pdir, pts, None, warmup=0, timeout=7200)
os.environ.pop('ONIA_DBG_MIXED')
q = est['qp']
e1 = [abs(r['finite_norm'] - q[i]) / abs(q[i]) for i, r in enumerate(res['ct1'])]
e2 = [abs(r['finite_norm'] - q[i]) / abs(q[i]) for i, r in enumerate(res['ct2'])]
acc = [r['accuracy'] for r in res['prod']]
ep = [abs(r['finite_norm'] - q[i]) / abs(q[i]) for i, r in enumerate(res['prod'])]
for i in range(len(pts)):
    print('p%-3d ct1 %.1e ct2 %.1e | estimate %.1e true %.1e %s' % (i, e1[i], e2[i], acc[i], ep[i], 'UNDER' if acc[i] < ep[i] else ''))
print('max ct1 %.1e, max ct2 %.1e, max estimate %.1e, points above 1e-3 (estimate) %d/%d, median time %.3f s (ct1 %.3f s)' % (
    max(e1), max(e2), max(acc), sum(a >= 1e-3 for a in acc), len(acc),
    statistics.median(r['time'] for r in res['prod']), statistics.median(r['time'] for r in res['ct1'])))
