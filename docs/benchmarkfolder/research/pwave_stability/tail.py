"""The S-wave tail of a stability profile against quadruple precision: the
first N points of a profile whose estimate is at least DELTA, evaluated in
double precision (CTModeRun 1, primary pass only) by one or more outputs, and
in quadruple precision (CTModeRun 4) by the first.

    tail.py CASE SETUP PROFILE.json DELTA N OUTDIR [OUTDIR ...]

The points are those of research/stability/stability.py (same generator and
seed), so PROFILE.json, written by that script for CASE:SETUP, gives their
estimates. Writes tail_CASE_SETUP.json next to the first OUTDIR."""
import json
import os
import sys

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/docs/benchmarkfolder/research/stability')
import onia_loop_check as olc
import stability as st

case, setup, profile, delta, n = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
outdirs = sys.argv[6:]
prof = json.load(open(profile))
idx = [i for i, a in enumerate(prof['accuracy']) if a >= delta][:n]
energy, ptmin = st.SETUPS[setup]
pdirs = [olc.subprocess_dir(o) for o in outdirs]
masses = olc.driver_masses(pdirs[0])
points = st.make_points(masses, prof['meta']['npoints'], energy, ptmin)
sel = [points[i] for i in idx]


def run(outdir, pdir, mode):
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99',
                                            'NRotations_DP': '0', 'NRotations_QP': '0'})
    return [r['finite_norm'] for r in olc.run_points(pdir, sel, None, warmup=0, timeout=36000)]


qp = run(outdirs[0], pdirs[0], '4')
dp = {o: run(o, p, '1') for o, p in zip(outdirs, pdirs)}
out = dict(case=case, setup=setup, delta=delta, indices=idx, estimate=[prof['accuracy'][i] for i in idx],
           qp=qp, dp=dp)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(outdirs[0])),
                                 'tail_%s_%s.json' % (case, setup)), 'w'))
for k, i in enumerate(idx):
    errs = '  '.join('%.1e' % (abs(dp[o][k] - qp[k]) / abs(qp[k])) for o in outdirs)
    print('p%-5d estimate %.1e | true error per output: %s' % (i, prof['accuracy'][i], errs), flush=True)
