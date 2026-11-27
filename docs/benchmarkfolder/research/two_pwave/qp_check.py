"""The quadruple-precision path of a two-P-wave output against its double
precision: the virtual at points in CTModeRun 1 (double precision), 4
(quadruple precision) and 5 (quadruple precision, loop direction reversed),
with MLStabThres 1d99, and in CTModeRun -1 with MLStabThres 1d-30, so that
MadLoop's stability test sends every point to quadruple precision (the
rescue: return code 2xx when the two quadruple-precision passes agree).
`--thres X` sets the MLStabThres of CTModeRun -1 instead (MadLoop's default
is 1.0d-3).
Each point prints the poles in units of Born * alpha_s/(2 pi) against
ONIA_IR_POLES, the finite parts, the differences between the passes, the
return codes and the times.

    qp_check.py OUTDIR OUT.json ENERGY:NPOINTS [ENERGY:NPOINTS ...] [--modes 1,4,5,-1] [--thres 1.0d-30]

The points are those of fixtures.py and acceptance.py (seed 1).
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'tools'))
import onia_loop_check as olc

args = sys.argv[1:]
modes = ['1', '4', '5', '-1']
if '--modes' in args:
    i = args.index('--modes')
    modes = args[i + 1].split(',')
    del args[i:i + 2]
THRES = {'-1': '1.0d-30'}
if '--thres' in args:
    i = args.index('--thres')
    THRES['-1'] = args[i + 1]
    del args[i:i + 2]
outdir, out = os.path.abspath(args[0]), args[1]
runs = [(float(a.split(':')[0]), int(a.split(':')[1])) for a in args[2:]]
pdir = olc.subprocess_dir(outdir)
olc.build_driver(pdir)
masses = olc.driver_masses(pdir)

records = []
for energy, npts in runs:
    pts = [p['momenta'] for p in olc.points_lib.generate_points(masses[2:], count=npts, energy=energy, seed=1,
                                                            incoming_masses=tuple(masses[:2]))]
    res = {}
    for mode in modes:
        olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode,
                                                'MLStabThres': THRES.get(mode, '1.0d99')})
        t = time.time()
        res[mode] = olc.run_points(pdir, pts, None, warmup=0, timeout=10 ** 6)
        print('sqrt(s) %g CTModeRun %s: %d points in %.0f s' % (energy, mode, len(pts), time.time() - t),
              flush=True)
    for k in range(npts):
        rec = dict(energy=energy, point=k, momenta=[[float.hex(c) for c in v] for v in pts[k]])
        for mode in modes:
            r = res[mode][k]
            rec['ctmode%s' % mode] = r
            print('  %g/%d mode %-2s born %.10e  c1 %.14f pred %.14f (%.1e)  c2 %.14f pred %.14f  '
                  'fin %.13e  code %d  acc %.1e  %.1f s'
                  % (energy, k, mode, r['born'], r['single_norm'], r['predicted_single_norm'],
                     abs(r['single_norm'] - r['predicted_single_norm'])
                     / max(abs(r['predicted_single_norm']), 1e-300),
                     r['double_norm'], r['predicted_double_norm'], r['finite_norm'], r['ret_code'],
                     r['accuracy'], r['time']), flush=True)
        ref = res[modes[0]][k]['finite_norm']
        print('  %g/%d finite against mode %s: %s' % (energy, k, modes[0], '  '.join(
            'mode %s %.1e' % (m, abs(res[m][k]['finite_norm'] - ref) / abs(ref)) for m in modes[1:])),
              flush=True)
        records.append(rec)
json.dump(dict(output=os.path.basename(pdir), masses=masses, modes=modes, note=__doc__,
               records=records), open(out, 'w'), indent=1)
print('wrote', out)
