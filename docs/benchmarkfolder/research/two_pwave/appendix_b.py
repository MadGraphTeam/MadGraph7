"""MadLoop's two-P-wave poles against app. B of the two-P-wave manuscript
(/shared/external/madloop_onia/Manuscript/Quarkonium_FKS/JHEP_2Pwave,
eq. (B.2) = eq:Vpoles of paper.tex).

    appendix_b.py table2 OUTDIR OUT.json MODE m_c:ENERGY[:NPTS] ...
    appendix_b.py terms  OUTDIR OUT.json ENERGY:NPTS ...

table2: e+ e- > chic0(1|3P01) hc(1|1P11) [virt=QCD] at the points of the
manuscript's table 2 (tab:chic0hc): the computed 1/eps per Born alpha_s/2pi
and ONIA_IR_POLES, in CTModeRun MODE (MLStabThres 1d99), with m_c set in
the param_card (the bound-state masses follow it).

terms: patches OUTDIR in place (pass a copy) so that ONIA_IR_POLES prints
the parts of its 1/eps in the order of eq. (B.2):
  APPB_T1  POLES(1), POLES(2), BORN2   lines 1-4 (point-like quarkonia)
  APPB_T3  4 f1 Re Z_G, 4 f2 Re Z_V, Im Z_G, Im Z_V, w   line 6 (dipole-dipole)
  APPB_T2  C1, C2, PDD                 lines 4-5 per state, and line 6
and evaluates the output in CTModeRun 1 at the points of qp_check.py
(seed 1), next to the computed 1/eps.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'tools'))
import onia_loop_check as olc


def points(masses, energy, n):
    return [p['momenta'] for p in olc.points_lib.generate_points(masses[2:], count=n, energy=energy, seed=1,
                                                                incoming_masses=tuple(masses[:2]))]


def table2(outdir, out, mode, specs):
    pdir = olc.subprocess_dir(outdir)
    olc.build_driver(pdir)
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
    rows = []
    for a in specs:
        f = a.split(':')
        mc, energy, n = float(f[0]), float(f[1]), int(f[2]) if len(f) > 2 else 2
        olc.set_masses(outdir, {4: mc})
        masses = olc.driver_masses(pdir)
        res = olc.run_points(pdir, points(masses, energy, n), None, warmup=0, timeout=10 ** 6)
        for k, r in enumerate(res):
            M = masses[2]
            w = (energy ** 2 - 2 * M ** 2) / (2 * M ** 2)
            dev = abs(r['single_norm'] - r['predicted_single_norm']) / abs(r['predicted_single_norm'])
            print('m_c %.2f sqrt(s) %7g w %.6f pt %d mode %s: c1 %.14f pred %.14f (%.1e)  c2 %.1e  code %d  %.0f s'
                  % (mc, energy, w, k, mode, r['single_norm'], r['predicted_single_norm'], dev,
                     r['double_norm'], r['ret_code'], r['time']), flush=True)
            rows.append(dict(m_c=mc, energy=energy, omega=w, point=k, mode=mode, c1=r['single_norm'],
                             predicted=r['predicted_single_norm'], c2=r['double_norm'],
                             predicted_c2=r['predicted_double_norm'], code=r['ret_code'], time=r['time']))
    json.dump(dict(output=os.path.basename(pdir), note=__doc__, rows=rows), open(out, 'w'), indent=1)
    print('wrote', out)


def patch(pdir):
    path = os.path.join(pdir, 'onia_loop.f')
    s = open(path).read()
    if 'APPB_T1' in s:
        return
    pre = [l for l in s.splitlines() if 'SUBROUTINE' in l and 'ONIA_IR_POLES(' in l][0].split()[1]
    pre = pre[:pre.index('ONIA_IR_POLES')]
    a = "      CALL %sONIA_PW_SOFT_POLE(P_PHYS,PMASS,CMASS,PSPOLE)\n" % pre
    b = "      CONTR=C1+C2+PDD\n"
    c = "      PDD=4D0*(F1*DBLE(ZG)+F2*DBLE(ZV))\n"
    assert s.count(a) == s.count(b) == s.count(c) == 1
    s = s.replace(a, "      WRITE(*,'(A,3ES26.17)') 'APPB_T1',POLES(1),POLES(2),BORN2\n" + a)
    s = s.replace(b, b + "      WRITE(*,'(A,3ES26.17)') 'APPB_T2',C1,C2,PDD\n")
    s = s.replace(c, c + "      WRITE(*,'(A,5ES26.17)') 'APPB_T3',4D0*F1*DBLE(ZG),\n"
                  "     $ 4D0*F2*DBLE(ZV),DIMAG(ZG),DIMAG(ZV),W\n")
    open(path, 'w').write(s)


def terms(outdir, out, specs):
    pdir = olc.subprocess_dir(outdir)
    patch(pdir)
    olc.build_driver(pdir)
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '1', 'MLStabThres': '1.0d99'})
    masses = olc.driver_masses(pdir)
    rows = []
    for a in specs:
        energy, n = float(a.split(':')[0]), int(a.split(':')[1])
        lines = [str(n)]
        for p in points(masses, energy, n):
            lines.append('%.17e' % -1.0)
            lines.extend(' '.join('%.17e' % c for c in v) for v in p)
        t = time.time()
        o = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                           capture_output=True, text=True, check=True).stdout.splitlines()
        cur, last, k = None, None, 0
        for l in o:
            f = l.split()
            if l.startswith('APPB_T1'):
                cur = dict(t1=float(f[1]), t1_double=float(f[2]), born2=float(f[3]))
            elif l.startswith('APPB_T3') and cur is not None and 't3_f1' not in cur:
                cur.update(t3_f1=float(f[1]), t3_f2=float(f[2]), im_zg=float(f[3]), im_zv=float(f[4]),
                           omega=float(f[5]))
            elif l.startswith('APPB_T2') and cur is not None and 't2_1' not in cur:
                cur.update(t2_1=float(f[1]), t2_2=float(f[2]), t3=float(f[3]))
                last, cur = cur, None
            elif l.startswith('RESULT'):
                born, sp, dp, ao2pi = float(f[2]), float(f[4]), float(f[5]), float(f[8])
                b = last['born2']
                row = dict(energy=energy, point=k, omega=last['omega'], c1=sp / (born * ao2pi),
                           c2=dp / (born * ao2pi), predicted=float(f[11]) / (born * ao2pi), code=int(f[7]),
                           term1=last['t1'] / b, term1_double=last['t1_double'] / b,
                           term2=[last['t2_1'] / b, last['t2_2'] / b], term3_f1=last['t3_f1'] / b,
                           term3_f2=last['t3_f2'] / b, term3=last['t3'] / b,
                           im_z=[last['im_zg'] / b, last['im_zv'] / b])
                rows.append(row)
                k += 1
                print('sqrt(s) %g pt %d w %.6f: c1 %.12f pred %.12f (%.1e) | per Born: T1 %.10f  T2 %.10f + %.10f'
                      '  T3 f1 %.10f f2 %.10f  Im Z %.1e %.1e | code %d'
                      % (energy, row['point'], row['omega'], row['c1'], row['predicted'],
                         abs(row['c1'] - row['predicted']) / abs(row['predicted']), row['term1'], row['term2'][0],
                         row['term2'][1], row['term3_f1'], row['term3_f2'], row['im_z'][0], row['im_z'][1],
                         row['code']), flush=True)
        print('  %d points in %.0f s' % (n, time.time() - t), flush=True)
    json.dump(dict(output=os.path.basename(pdir), masses=masses, note=__doc__, rows=rows), open(out, 'w'),
              indent=1)
    print('wrote', out)


if __name__ == '__main__':
    cmd, outdir, out = sys.argv[1], os.path.abspath(sys.argv[2]), sys.argv[3]
    if cmd == 'table2':
        table2(outdir, out, sys.argv[4], sys.argv[5:])
    else:
        terms(outdir, out, sys.argv[4:])
