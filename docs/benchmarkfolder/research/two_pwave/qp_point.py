"""One CTModeRun mode at one point of an output, the points of qp_check.py
(seed 1); for the long quadruple-precision passes at a single point.

    qp_point.py OUTDIR OUT.json ENERGY POINT MODE
"""
import json, sys, time
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
outdir, out, energy, k, mode = sys.argv[1], sys.argv[2], float(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
pdir = olc.subprocess_dir(outdir); olc.build_driver(pdir); masses = olc.driver_masses(pdir)
p = olc.points_lib.generate_points(masses[2:], count=k + 1, energy=energy, seed=1, incoming_masses=tuple(masses[:2]))[k]['momenta']
olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d-3'})
r = olc.run_points(pdir, [p], None, warmup=0, timeout=10 ** 6)[0]
print('%g/%d mode %s c1 %.14f pred %.14f (%.1e)  c2 %.14f pred %.14f  fin %.13e  code %d  %.1f s' % (
    energy, k, mode, r['single_norm'], r['predicted_single_norm'],
    abs(r['single_norm'] - r['predicted_single_norm']) / abs(r['predicted_single_norm']),
    r['double_norm'], r['predicted_double_norm'], r['finite_norm'], r['ret_code'], r['time']), flush=True)
json.dump(dict(energy=energy, point=k, mode=mode, result=r), open(out, 'w'), indent=1)
