"""The floor of quadruple precision in e+ e- -> J/psi chi_cJ (and J/psi eta_c)
with heavy b and t quarks (10 and 100 TeV) instead of their exclusion: their
counterterms and loop groups cancel, and MadLoop rounds every loop group to
double precision before it adds them.

    qp_floor.py OUTDIR OUT.json J|etac [--masses 1e4:1e5,1e3:1e3,...] [--exact]

OUTDIR is a generated output with the b and t quarks in the loops (e.g.
e+ e- > Jpsi(1|3S11) chic2(1|3P21) / z QED=2 [virt=QCD]; psi_chic.py excludes
them, with / t t~ b b~, and so has no such floor), patched in place (pass a
copy):
- MP_TIRUST_LOOP (TIRUST_interface.f) adds the quadruple-precision value of
  each loop group, before its rounding to double precision, to QF_SUM, and
  its modulus to QF_ABS;
- MP_COMPUTE_LOOP_COEFS keeps the quadruple-precision counterterm sum QF_CT;
- loop_matrix.f prints, after MadLoop's sum in CTModeRun 4, a line
  QPFLOOR ANS(1,0) QF_CT+QF_SUM sum|parts| sum|groups| QF_CT QF_SUM.

For each (m_b, m_t) of --masses (GeV), the output is evaluated in CTModeRun 4
at the nodes of psi_chic.py (or its --exact points), and the angular average
of the finite part is formed twice: from MadLoop's result (groups rounded to
double precision) and from the quadruple-precision sum (rounded once). Both
are compared with psi_chic_ref.madloop_c0 (J = 0, 1, 2; for etac, Gong-Wang's
n_l = 3 value 24.53642 of research/literature.md C.4, seven digits).
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'tools'))
import onia_loop_check as olc
import psi_chic_ref as ref
from psi_chic import NODES, EXACT, momenta, exact_momenta, fit

DECL = "      REAL*16 QF_SUM(3),QF_ABS,QF_CT(3)\n      COMMON/QFLOOR/QF_SUM,QF_ABS,QF_CT\n"
ETAC_C0 = 2 * 12.268210  # literature.md C.4: sqrt(s) = 10.58, m_c = 1.5, mu = 2 m_c, n_l = 3


def patch(pdir):
    path = os.path.join(pdir, 'TIRUST_interface.f')
    s = open(path).read()
    if 'QFLOOR' in s:
        return
    i = s.index('      SUBROUTINE ML5_0_MP_TIRUST_LOOP(')
    j = s.index('\n      END\n', i)
    body = s[i:j]
    k = body.index('C     A P-wave bound state: the lines and masses before the reduction') \
        if 'C     A P-wave bound state: the lines and masses' in body else body.index('      RES=0\n')
    body = body[:k] + DECL + body[k:]
    old = "        RES(I)=CMPLX(RESULT(I),KIND=KIND(0D0))\n"
    assert body.count(old) == 1
    body = body.replace(old, old + "        QF_SUM(I)=QF_SUM(I)+REAL(RESULT(I),KIND=16)\n"
                        "        IF (I.EQ.1) QF_ABS=QF_ABS+ABS(REAL(RESULT(I),KIND=16))\n")
    open(path, 'w').write(s[:i] + body + s[j:])

    path = os.path.join(pdir, 'mp_compute_loop_coefs.f')
    s = open(path).read()
    k = s.index('      REAL*16 ANS(3,0:NSQUAREDSO)\n')
    s = s[:k] + DECL + s[k:]
    old = "            ANSDP(I,J)=REAL(ANS(I,J),KIND=8)\n"
    assert s.count(old) == 1
    s = s.replace(old, old + "            IF (J.EQ.0) QF_CT(I)=ANS(I,J)\n")
    open(path, 'w').write(s)

    path = os.path.join(pdir, 'loop_matrix.f')
    s = open(path).read()
    k = s.index('      REAL*8 BUFFR(3,0:NSQUAREDSO)')
    s = s[:k] + DECL + "      REAL*8 QF_DPABS\n" + s[k:]
    old = "      DO I_SO=1,NSQUAREDSO\n        DO J=1,NLOOPGROUPS\n          S(I_SO,J)=.TRUE.\n"
    assert s.count(old) == 1
    s = s.replace(old, "      QF_SUM=0\n      QF_ABS=0\n" + old)
    old = "C     Make sure that no NaN is present in the result\n"
    assert s.count(old) == 1
    s = s.replace(old, """      IF (CTMODE.GE.4) THEN
        QF_DPABS=ABS(BUFFR_BIS(1,0))
        DO I=1,NLOOPGROUPS
          QF_DPABS=QF_DPABS+ABS(DBLE(LOOPRES(1,1,I)))
        ENDDO
        WRITE(*,'(A,6ES26.17)') 'QPFLOOR',ANS(1,0),
     $   DBLE(QF_CT(1)+QF_SUM(1)),QF_DPABS,DBLE(QF_ABS),
     $   DBLE(QF_CT(1)),DBLE(QF_SUM(1))
      ENDIF
""" + old)
    open(path, 'w').write(s)


def evaluate(pdir, pts, mu):
    lines = [str(len(pts))]
    for p in pts:
        lines.append('%.17e' % mu)
        lines.extend(' '.join('%.17e' % c for c in v) for v in p)
    out = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir,
                         input='\n'.join(lines) + '\n', capture_output=True, text=True,
                         check=True).stdout.splitlines()
    qf = [[float(x) for x in l.split()[1:]] for l in out if l.startswith('QPFLOOR')]
    res = []
    for l in out:
        if l.startswith('RESULT'):
            f = l.split()
            born, fin, ao2pi = float(f[2]), float(f[3]), float(f[8])
            res.append(dict(born=born, c0=fin / (born * ao2pi)))
    assert len(qf) == len(res), (len(qf), len(res))
    for r, q in zip(res, qf):
        r.update(ans=q[0], qp_sum=q[1], cancellation=q[2] / abs(q[0]),
                 counterterms=q[4], loops=q[5], c0_qp_sum=r['c0'] * q[1] / q[0])
    return res


def average(xs, res, key, exact):
    b = [r['born'] for r in res]
    v = [r['born'] * r[key] for r in res]
    if exact:
        ab, bb, _ = fit(xs, b)
        av, bv, _ = fit(xs, v)
        return (av + bv / 3) / (ab + bb / 3)
    return sum(w * x for (_, w), x in zip(NODES, v)) / sum(w * x for (_, w), x in zip(NODES, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('json')
    ap.add_argument('case')
    ap.add_argument('--masses', default='1e3:1e3,1e4:1e4,1e4:1e5,1e5:1e5')
    ap.add_argument('--exact', action='store_true')
    args = ap.parse_args()
    pdir = olc.subprocess_dir(os.path.abspath(args.outdir))
    patch(pdir)
    olc.build_driver(pdir)
    masses = olc.driver_masses(pdir)
    energy = 18.5 if args.exact else 10.58
    nodes = EXACT if args.exact else NODES
    pts = ([exact_momenta(c, s) for c, s in EXACT] if args.exact
           else [momenta(masses, energy, x) for x, _ in NODES])
    xs = [x for x, _ in nodes]
    if args.case == 'etac':
        reference = ETAC_C0 if not args.exact else None
    else:
        reference = float(ref.madloop_c0(int(args.case), energy, 1.5, 3.0))
    olc.set_reduction(args.outdir, ['9'], extra={'CTModeRun': '4', 'MLStabThres': '1.0d-3'})
    rows = []
    for pair in args.masses.split(','):
        mb, mt = (float(x) for x in pair.split(':'))
        olc.set_masses(args.outdir, {5: mb, 6: mt})
        res = evaluate(pdir, pts, 3.0)
        row = dict(m_b=mb, m_t=mt, points=res,
                   c0_madloop=average(xs, res, 'c0', args.exact),
                   c0_qp_sum=average(xs, res, 'c0_qp_sum', args.exact),
                   max_cancellation=max(r['cancellation'] for r in res),
                   max_rounding=max(abs(r['ans'] - r['qp_sum']) / abs(r['qp_sum']) for r in res))
        if reference is not None:
            row['dev_madloop'] = abs(row['c0_madloop'] - reference) / abs(reference)
            row['dev_qp_sum'] = abs(row['c0_qp_sum'] - reference) / abs(reference)
        rows.append(row)
        print('m_b %.0e m_t %.0e: sum|parts|/|total| %.2e, rounding %.1e; <c0> MadLoop %.12f, '
              'QP sum %.12f%s' % (mb, mt, row['max_cancellation'], row['max_rounding'],
                                  row['c0_madloop'], row['c0_qp_sum'],
                                  '' if reference is None else '; against %.12f: %.1e, %.1e'
                                  % (reference, row['dev_madloop'], row['dev_qp_sum'])), flush=True)
    olc.set_masses(args.outdir, {5: 1e4, 6: 1e5})
    json.dump(dict(note=__doc__, case=args.case, energy=energy, exact=args.exact,
                   reference=reference, rows=rows), open(args.json, 'w'), indent=1)
    print('wrote', args.json)


if __name__ == '__main__':
    main()
