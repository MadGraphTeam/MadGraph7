"""Scratch: P-wave output in DP, forced QP and MadLoop's stability/rescue modes."""
import os, sys, shutil
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
outdir = sys.argv[1]
pdir = olc.subprocess_dir(outdir)
gen = os.path.join(pdir, 'onia_loop.f.gen')
if os.path.exists(gen):
    shutil.copy(gen, os.path.join(pdir, 'onia_loop.f'))
card = os.path.join(outdir, 'Cards', 'MadLoopParams.dat')
if os.path.exists(card + '.orig'):
    shutil.copy(card + '.orig', card)
olc.build_driver(pdir)
masses = olc.driver_masses(pdir)
final = masses[2:]
import math
if len(sys.argv) > 4 and sys.argv[4] == 'onshell':
    q2 = sum(float(x)**2 for x in sys.argv[3].split(','))
    final = [2*math.sqrt((m/2.)**2 + q2) if i == 0 else m for i, m in enumerate(final)]
energy = 1000. if len(final) > 1 else final[0]
points = olc.points_lib.generate_points(final, count=3, energy=energy,
                                        incoming_masses=tuple(masses[:2]))
modes = []
for m in sys.argv[2].split(','):
    k = m.split(':')
    e = {'CTModeRun': k[0], 'ImprovePSPoint': '-1'}
    if len(k) > 1:
        e['MLStabThres'] = k[1]
    modes.append((m, e))
relmom = [float(x) for x in sys.argv[3].split(',')] if len(sys.argv) > 3 else None
print('final masses', final)
for name, extra in modes:
    olc.set_reduction(outdir, ['9'], extra=extra)
    res = olc.run_points(pdir, [p['momenta'] for p in points], None, timeout=900, relmom=relmom)
    for i, r in enumerate(res):
        print('%-7s pt%d born %.15e fin %.15f c1 %.15f c2 %.15f acc %s code %s' % (
            name, i, r['born'], r['finite_norm'], r['single_norm'], r['double_norm'],
            r.get('accuracy'), r['ret_code']), flush=True)
shutil.copy(card + '.orig', card) if os.path.exists(card + '.orig') else None
