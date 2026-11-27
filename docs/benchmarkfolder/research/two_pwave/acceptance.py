"""Acceptance of a TIRust revision for two P-wave states: the virtual of a
two-P-wave output at points, in both double-precision passes of MadLoop's
stability test (CTModeRun 1, and 2 = the loop direction reversed, TIRust's
check pass) and the test itself (CTModeRun -1, the mean of the two). Each
point prints the computed poles in units of Born * alpha_s/(2 pi) against
ONIA_IR_POLES, the finite part of both passes, MadLoop's return code (4xx =
unstable, which a refused raised-power call forces) and the time.

    acceptance.py OUTDIR OUT.json ENERGY:NPOINTS [ENERGY:NPOINTS ...]

The points are those of fixtures.py (seed 1). MLStabThres is set to 1d99, so
that no point goes to quadruple precision (qp_check.py tests that path); the
accuracy estimate of CTModeRun -1 is still MadLoop's.
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'tools'))
import onia_loop_check as olc

outdir, out = os.path.abspath(sys.argv[1]), sys.argv[2]
runs = [(float(a.split(':')[0]), int(a.split(':')[1])) for a in sys.argv[3:]]
pdir = olc.subprocess_dir(outdir)
olc.build_driver(pdir)
masses = olc.driver_masses(pdir)

records = []
for energy, npts in runs:
    pts = [p['momenta'] for p in olc.points_lib.generate_points(masses[2:], count=npts, energy=energy, seed=1,
                                                            incoming_masses=tuple(masses[:2]))]
    res = {}
    for mode in ('1', '2', '-1'):
        olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
        t = time.time()
        res[mode] = olc.run_points(pdir, pts, None, warmup=0, timeout=36000)
        print('sqrt(s) %g CTModeRun %s: %d points in %.0f s' % (energy, mode, len(pts), time.time() - t),
              flush=True)
    for k in range(npts):
        a, b, s = res['1'][k], res['2'][k], res['-1'][k]
        rec = dict(energy=energy, point=k, momenta=[[float.hex(c) for c in v] for v in pts[k]],
                   ct1=a, ct2=b, stab=s)
        records.append(rec)
        for name, r in (('ct1', a), ('ct2', b), ('stab', s)):
            print('  %g/%d %-4s born %.10e  c1 %.14f pred %.14f (%.1e)  c2 %.14f pred %.14f  '
                  'fin %.12e  code %d  acc %.1e  %.2f s'
                  % (energy, k, name, r['born'], r['single_norm'], r['predicted_single_norm'],
                     abs(r['single_norm'] - r['predicted_single_norm'])
                     / max(abs(r['predicted_single_norm']), 1e-300),
                     r['double_norm'], r['predicted_double_norm'], r['finite_norm'], r['ret_code'],
                     r['accuracy'], r['time']), flush=True)
        print('  %g/%d finite, reversed vs primary %.1e' % (
            energy, k, abs(b['finite_norm'] - a['finite_norm']) / abs(a['finite_norm'])), flush=True)
json.dump(dict(output=os.path.basename(pdir), masses=masses, note=__doc__, records=records),
          open(out, 'w'), indent=1)
print('wrote', out)
