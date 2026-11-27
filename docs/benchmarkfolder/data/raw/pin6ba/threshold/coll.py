"""g g > Jpsi g with the gluon at small angles theta to the beam (small jet
pT, as an FKS run without cuts samples it): per build, MadLoop's default
run against a QP-only run of da7902b + v16 and against ONIA_IR_POLES.

usage: coll.py BUILD...
"""
import json
import math
import os
import sys

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import run as R  # noqa: E402

M = 3.0


def point(rs, theta, phi):
    eg = (rs * rs - M * M) / (2 * rs)
    n = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
    g = [eg] + [eg * c for c in n]
    psi = [rs - eg] + [-eg * c for c in n]
    return [[rs / 2, 0, 0, rs / 2], [rs / 2, 0, 0, -rs / 2], psi, g]


PTS = []
for rs in (3.3, 5., 10., 20.):
    for th in (1e-1, 1e-2, 1e-3, 1e-4):
        for k, phi in enumerate((0.3, 2.1)):
            PTS.append(dict(rs=rs, theta=th if k == 0 else math.pi - th, phi=phi))
MOM = [point(p['rs'], p['theta'], p['phi']) for p in PTS]

ref_out = R.outdir('v16')
ref_dir = olc.subprocess_dir(ref_out)
olc.set_reduction(ref_out, [9], dict(R.PARAMS, CTModeRun=4))
ref = olc.run_points(ref_dir, MOM, mu=3.0, warmup=0)
olc.set_reduction(ref_out, [9], dict(R.PARAMS, CTModeRun=-1))
res = {'points': PTS, 'qp_v16': ref}
for build in sys.argv[1:]:
    out = R.outdir(build)
    pdir = olc.subprocess_dir(out)
    olc.set_reduction(out, [9], dict(R.PARAMS, CTModeRun=-1))
    r = olc.run_points(pdir, MOM, mu=3.0, warmup=0)
    res[build] = r
    for p, x, q in zip(PTS, r, ref):
        err = abs(x['finite'] / q['finite'] - 1)
        perr = abs(x['single_norm'] / x['predicted_single_norm'] - 1)
        flag = ''
        if x['ret_code'] // 100 < 3 and (err > max(10 * x['accuracy'], 1e-6) or perr > 1e-6):
            flag = '  STABLE BUT WRONG'
        print('%-4s rs %4g theta %.1e ret %d acc %.1e |fin/qp-1| %.1e |c1/pred-1| %.1e  qp(v16) ret %d%s'
              % (build, p['rs'], p['theta'] if p['theta'] < 1 else math.pi - p['theta'],
                 x['ret_code'], x['accuracy'], err, perr, q['ret_code'], flag))
with open(os.path.join(SP, 'coll.json'), 'w') as f:
    json.dump(res, f)
