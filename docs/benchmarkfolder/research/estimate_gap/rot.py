"""rot.py NAME [NAME ...]: on the est4 outputs (ONIA_TIRK=0 after calib2.py), MadLoop's estimate
with NRotations_DP = 1 against the true error (cal_NAME.json), and the time per point."""
import json, os, statistics, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
W = os.path.dirname(os.path.abspath(__file__))
for name in sys.argv[1:]:
    out = W + '/out/' + name; pdir = olc.subprocess_dir(out)
    d = json.load(open(W + '/pts_%s.json' % name)); pts, mu = d['points'], d.get('mu')
    cal = json.load(open(W + '/cal_%s.json' % name))
    res = {}
    for nrot in ('0', '1'):
        olc.set_reduction(out, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2',
                                             'NRotations_DP': nrot, 'NRotations_QP': '0'})
        res[nrot] = olc.run_points(pdir, pts, mu, timeout=36000)
    olc.set_reduction(out, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
    miss0 = sum(c['err'] > r['accuracy'] for c, r in zip(cal, res['0']))
    miss1 = sum(c['err'] > r['accuracy'] for c, r in zip(cal, res['1']))
    worst1 = max(c['err'] / max(r['accuracy'], 1e-300) for c, r in zip(cal, res['1']))
    infl = statistics.median(r1['accuracy'] / max(r0['accuracy'], 1e-17) for r0, r1 in zip(res['0'], res['1']))
    t0 = statistics.median(r['time'] for r in res['0']); t1 = statistics.median(r['time'] for r in res['1'])
    above = sum(r['accuracy'] >= 1e-3 for r in res['1'])
    print('%-12s %3d pts: estimate below error without rotation %3d, with %3d (worst err/acc %.1f) | acc median x%.1f | >=1e-3 with rot %d | time x%.2f' % (
        name, len(pts), miss0, miss1, worst1, infl, above, t1 / t0), flush=True)
