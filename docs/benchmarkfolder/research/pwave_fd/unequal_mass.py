"""The self term of a P-wave state with unequal constituent masses: MadLoop's
computed 1/eps against ONIA_IR_POLES (which keeps the self term and LDME
counterterm of arXiv:2402.19221 app. B, coefficient 8/(m_Q m_Qbar)) and
against app. B of the two-P-wave manuscript
(Manuscript/Quarkonium_FKS/JHEP_2Pwave, eq. (B.2)), which has no self-like
pole: its LDME coefficient kappa = 2 M^2/(m_Q m_Qbar)^2 (footnote of its
sect. 5.1) cancels the self term. The two coefficients agree for equal
masses.

    unequal_mass.py OUTDIR OUT.json ENERGY:NPTS ... [--modes 1,2] [--no-improve]

OUTDIR (one P-wave state, e.g. e+ e- > Bc1L+(1|1P11) b c~ / z t t~ QED=2
[virt=QCD]). An output generated before the fix of ONIA_IR_POLES (which has
the self and LDME terms) is patched in place (pass a copy):
ONIA_PW_SOFT_POLE prints, per S-wave daughter S', the line
  UNEQ  label  self  LDME(8/(m_Q m_Qbar))  LDME(kappa)  GMN
with self = -2/(3 P^2) C_eff g_mu nu M^mu nu and LDME = -c M(r_S'), and
ONIA_IR_POLES prints BORN2 before it (UNEQ_B). Per point: the computed 1/eps
(c1), ONIA_IR_POLES (pred), and pred minus self and LDME (no_self, the
prediction of eq. (B.2) if self + LDME(kappa) = 0), all per Born
alpha_s/2pi. An output generated after the fix has neither term: then
no_self = pred, and the check is c1 against ONIA_IR_POLES.

--no-improve: MadLoop's improvement of the phase-space point off
(ImprovePSPoint -1), as for the runs before the fix of the dual spinors: for
these 2 -> 3 points the improvement leaves the incoming e- with p_x, p_y of
1e-16 and p_0 + p_z = 0, and IXXXXX of
aloha/template_files/aloha_functions_dual.f took the dual square root of 0,
whose derivative component is NaN: every P-wave weight of the loop became
NaN (code 410). Without the option the output's MadLoopParams.dat applies.
"""
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'tools'))
import onia_loop_check as olc


def patch(pdir):
    path = os.path.join(pdir, 'onia_loop.f')
    s = open(path).read()
    if 'UNEQ_B' in s:
        return True
    pre = [l for l in s.splitlines() if 'SUBROUTINE' in l and 'ONIA_IR_POLES(' in l][0].split()[1]
    pre = pre[:pre.index('ONIA_IR_POLES')]
    a = "      CALL %sONIA_PW_SOFT_POLE(P_PHYS,PMASS,CMASS,PSPOLE)\n" % pre
    b = "        CONTR=CONTR-2D0/(3D0*KTOT2)*CEFF*GMN-8D0/(MQ*MQB)*CLDME*MR\n"
    assert s.count(a) == 1, s.count(a)
    if s.count(b) == 0:
        return False
    assert s.count(b) == 1, s.count(b)
    s = s.replace(a, "      WRITE(*,'(A,ES26.17)') 'UNEQ_B',BORN2\n" + a)
    s = s.replace(b, b + "        WRITE(*,'(A,1X,A2,4ES26.17)') 'UNEQ',PWS_LABEL(A),\n"
                  "     $   -2D0/(3D0*KTOT2)*CEFF*GMN,-8D0/(MQ*MQB)*CLDME*MR,\n"
                  "     $   -2D0*KTOT2/(MQ*MQB)**2*CLDME*MR,GMN\n")
    open(path, 'w').write(s)
    return True


def main():
    args = sys.argv[1:]
    modes = ['1', '2']
    if '--modes' in args:
        i = args.index('--modes')
        modes = args[i + 1].split(',')
        del args[i:i + 2]
    extra = {}
    if '--no-improve' in args:
        args.remove('--no-improve')
        extra['ImprovePSPoint'] = '-1'
    outdir, out = os.path.abspath(args[0]), args[1]
    pdir = olc.subprocess_dir(outdir)
    has_self = patch(pdir)
    olc.build_driver(pdir)
    masses = olc.driver_masses(pdir)
    rows = []
    for a in args[2:]:
        energy, n = float(a.split(':')[0]), int(a.split(':')[1])
        pts = [p['momenta'] for p in olc.points_lib.generate_points(masses[2:], count=n, energy=energy, seed=1,
                                                                incoming_masses=tuple(masses[:2]))]
        for mode in modes:
            olc.set_reduction(outdir, ['9'], extra=dict(extra, CTModeRun=mode, MLStabThres='1.0d99'))
            lines = [str(n)]
            for p in pts:
                lines.append('%.17e' % -1.0)
                lines.extend(' '.join('%.17e' % c for c in v) for v in p)
            t = time.time()
            o = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                               capture_output=True, text=True, timeout=10 ** 6).stdout.splitlines()
            cur, last, k = None, None, 0
            for l in o:
                f = l.split()
                if l.startswith('UNEQ_B'):
                    cur = dict(born2=float(f[1]), daughters=[])
                elif l.startswith('UNEQ ') and cur is not None:
                    cur['daughters'].append(dict(label=f[1], self=float(f[2]), ldme8=float(f[3]),
                                                 ldme_kappa=float(f[4]), gmn=float(f[5])))
                    last = cur
                elif l.startswith('RESULT'):
                    born, sp, dp, ao2pi = float(f[2]), float(f[4]), float(f[5]), float(f[8])
                    b = last['born2'] if last else 1.0
                    ds = [] if not last else [dict(d, self=d['self'] / b, ldme8=d['ldme8'] / b, ldme_kappa=d['ldme_kappa'] / b,
                               gmn=d['gmn'] / b) for d in last['daughters']]
                    pred = float(f[11]) / (born * ao2pi)
                    se = sum(d['self'] + d['ldme8'] for d in ds)
                    row = dict(energy=energy, point=k, mode=mode, c1=sp / (born * ao2pi), c2=dp / (born * ao2pi),
                               predicted=pred, predicted_c2=float(f[12]) / (born * ao2pi), code=int(f[7]),
                               daughters=ds, self_ldme8=se, no_self=pred - se,
                               self_ldme_kappa=sum(d['self'] + d['ldme_kappa'] for d in ds))
                    row['dev_predicted'] = abs(row['c1'] - pred) / abs(pred)
                    row['dev_no_self'] = abs(row['c1'] - row['no_self']) / abs(row['no_self'])
                    rows.append(row)
                    last, cur = None, None
                    k += 1
                    print('sqrt(s) %g pt %d mode %s: c1 %.12f | pred %.12f dev %.1e | no self %.12f dev %.1e'
                          ' | self+LDME(8/mm) %.6f, self+LDME(kappa) %.2e | %s | c2 %.10f pred %.10f | code %d'
                          % (energy, row['point'], mode, row['c1'], pred, row['dev_predicted'], row['no_self'],
                             row['dev_no_self'], se, row['self_ldme_kappa'],
                             ' '.join('%s: self %.6f LDME %.6f/%.6f' % (d['label'], d['self'], d['ldme8'],
                                                                       d['ldme_kappa']) for d in ds),
                             row['c2'], row['predicted_c2'], row['code']), flush=True)
            print('  sqrt(s) %g mode %s: %d points in %.0f s' % (energy, mode, n, time.time() - t), flush=True)
    json.dump(dict(output=os.path.basename(pdir), masses=masses, note=__doc__, self_terms=has_self,
                   madloop_params=extra, rows=rows), open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
