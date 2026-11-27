"""Near-threshold scan of g g > Jpsi g (m_c = 1.5, mu_R = 3) with one build:
per sqrt(s), N random points, MadLoop's default run (CTModeRun -1) and a
QP-only run (CTModeRun 4) as reference.

usage: scan.py BUILD N SQRTS...
"""
import json
import os
import sys

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import run as R  # noqa: E402

build, n = sys.argv[1], int(sys.argv[2])
energies = [float(e) for e in sys.argv[3:]]
out = R.outdir(build)
pdir = olc.subprocess_dir(out)
masses = olc.driver_masses(pdir)
res = {}
for e in energies:
    pts = [p['momenta'] for p in olc.points_lib.generate_points(
        masses[2:], count=n, energy=e, seed=11, incoming_masses=tuple(masses[:2]))]
    rows = {}
    for tag, ct in (('default', -1), ('qp', 4)):
        olc.set_reduction(out, [9], dict(R.PARAMS, CTModeRun=ct))
        rows[tag] = olc.run_points(pdir, pts, mu=3.0, warmup=0)
    res['%g' % e] = dict(points=pts, **rows)
    bad = []
    for i, (d, q) in enumerate(zip(rows['default'], rows['qp'])):
        err = abs(d['finite'] / q['finite'] - 1) if q['finite'] else float('nan')
        perr = abs(d['single_norm'] / d['predicted_single_norm'] - 1)
        qperr = abs(q['single_norm'] / q['predicted_single_norm'] - 1)
        stable = d['ret_code'] // 100 < 4
        if (not stable) or err > max(10 * d['accuracy'], 1e-6) or perr > 1e-6:
            bad.append((i, d['ret_code'], d['accuracy'], err, perr, q['ret_code'], qperr))
    print('%s sqrt(s) %g: %d points, %d unstable, %d flagged' % (
        build, e, n, sum(d['ret_code'] // 100 >= 4 for d in rows['default']), len(bad)))
    for b in bad[:12]:
        print('   pt %3d ret %d acc %.1e |fin/qp-1| %.1e |c1/pred-1| %.1e  qp ret %d qp c1 %.1e' % b)
with open(os.path.join(SP, 'scan_%s.json' % build), 'w') as f:
    json.dump(res, f)
