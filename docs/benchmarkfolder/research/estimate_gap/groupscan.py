"""Per loop group at a point: a_g = 2N sum_j |c_j I_j| (finite, DP pass 1), the DP and QP
group values and the true error. groupscan.py OUTDIR POINTS.json INDEX MU [QPREF]"""
import json, os, re, subprocess, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
EPS = 2.220446049250313e-16
out, ptsf, idx, mu = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
pdir = olc.subprocess_dir(out)
p = os.path.join(pdir, 'TIR_interface.f'); s = open(p).read()
if 'GRPABS' not in s:
    a = "        ONIA_TIRABS(CTMODE)=ONIA_TIRABS(CTMODE)+2.0D0*NORMALIZATION\n     $   *ONIA_TIRTERM\n"
    assert s.count(a) == 1, 'tirabs'
    s = s.replace(a, a + """        WRITE(*,'(A,2I6,ES26.17)') 'GRPABS',CTMODE,I_LOOPGROUP,
     $   2.0D0*NORMALIZATION*ONIA_TIRTERM
""")
    open(p, 'w').write(s)
p = os.path.join(pdir, 'loop_matrix.f'); s = open(p).read()
if 'SPLITDBG' not in s:
    old = "C     Make sure that no NaN is present in the result\n"
    s = s.replace(old, """      WRITE(*,'(A,I3,4ES26.17)') 'SPLITDBG CT',CTMODE,BUFFR_BIS(1,0),
     $ BUFFR_BIS(2,0),BUFFR_BIS(3,0),ANS(1,0)
      DO I=1,NLOOPGROUPS
        WRITE(*,'(A,I3,I6,4ES26.17)') 'SPLITDBG LG',CTMODE,I,
     $   DBLE(LOOPRES(1,1,I)),DIMAG(LOOPRES(1,1,I)),
     $   DBLE(LOOPRES(2,1,I)),DBLE(LOOPRES(3,1,I))
      ENDDO
""" + old, 1)
    open(p, 'w').write(s)
olc.build_driver(pdir)
pt = json.load(open(ptsf))['points'][idx]
lines = ['1', '%.17e' % mu if mu > 0 else '-1'] + [' '.join('%.17e' % c for c in v) for v in pt]
def run(mode):
    olc.set_reduction(out, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2'})
    o = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                       capture_output=True, text=True, check=True).stdout.splitlines()
    ct = [l.split() for l in o if l.startswith('SPLITDBG CT')][-1]
    g = {}
    for l in o:
        if l.startswith('SPLITDBG LG') and l.split()[2] == ct[2]:
            f = l.split(); g[int(f[3])] = float(f[4])
    a = {}
    for l in o:
        if l.startswith('GRPABS') and l.split()[1] == '1':
            f = l.split(); a[int(f[2])] = float(f[3])
    return float(ct[6]), g, a
ans, g, a = run('1')
qans, qg, _ = run('4')
olc.set_reduction(out, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
rows = [dict(g=i, v=qg[i], a=a.get(i, 0.0), e=abs(g[i] - qg[i])) for i in g]
tot_a = sum(r['a'] for r in rows)
print('total: |ANS| %.4e, true err %.1e rel, eps*sum a_g/|ANS| %.1e, groups %d' % (abs(qans), abs(ans - qans) / abs(qans), EPS * tot_a / abs(qans), len(rows)))
print('  by a_g (largest 6):  group  a_g/|ANS|  kappa_g=a_g/|v_g|  true e_g/|ANS|  e_g/(eps a_g)')
for r in sorted(rows, key=lambda r: -r['a'])[:6]:
    print('              %5d  %.2e  %.1e  %.1e  %.2g' % (r['g'], r['a'] / abs(qans), r['a'] / max(abs(r['v']), 1e-300), r['e'] / abs(qans), r['e'] / (EPS * r['a']) if r['a'] else 0))
print('  by true error (largest 4):')
for r in sorted(rows, key=lambda r: -r['e'])[:4]:
    print('              %5d  %.2e  %.1e  %.1e  %.2g' % (r['g'], r['a'] / abs(qans), r['a'] / max(abs(r['v']), 1e-300), r['e'] / abs(qans), r['e'] / (EPS * r['a']) if r['a'] else 0))
