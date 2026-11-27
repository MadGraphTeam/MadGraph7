"""Size of the DP coefficient errors of groups 44, 49 and the cancellation of their contraction."""
import json, os, subprocess, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
OUT = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad/trace/uux'
pdir = olc.subprocess_dir(OUT)
p = os.path.join(pdir, 'TIR_interface.f'); s = open(p).read()
if "'LCD'" not in s:
    a = """      IF(LCS_MODE.GT.0.AND.(I_LOOPGROUP.EQ.44.OR.I_LOOPGROUP.EQ.49))
     $  THEN"""
    assert s.count(a) == 1
    s = s.replace(a, """      IF(LCS_MODE.EQ.4.AND.(I_LOOPGROUP.EQ.44.OR.I_LOOPGROUP.EQ.49))
     $  THEN
        DO J=0,NLOOPCOEFS-1
          WRITE(*,'(A,2I5,6ES26.17)') 'LCD',I_LOOPGROUP,J,
     $     LOOPCOEFS(J,I_SQSO,I_LOOPGROUP),TIRCOEFS(J,1),TIRCOEFS(J,2)
        ENDDO
      ENDIF
""" + a.replace('LCS_MODE.GT.0', 'LCS_MODE.GT.0.AND.LCS_MODE.LT.4'))
    open(p, 'w').write(s)
    olc.build_driver(pdir)
pt = json.load(open(OUT + '_modes.json'))['point']
lines = ['1', '%.17e' % 11.0] + [' '.join('%.17e' % c for c in v) for v in pt]
olc.set_reduction(OUT, ['9'], extra={'CTModeRun': '1', 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2'})
o = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                   capture_output=True, text=True, check=True, env=dict(os.environ, LC_SUBST='4')).stdout.splitlines()
olc.set_reduction(OUT, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
dp = {}
for l in o:
    if l.startswith('LCD'):
        f = l.split(); dp[(int(f[1]), int(f[2]))] = [float(x) for x in f[3:]]
qp = {}
for l in open(os.path.join(pdir, 'lcq.txt')):
    f = l.split(); qp[(int(f[0]), int(f[1]))] = complex(float(f[2]), float(f[3]))
for g in (44, 49):
    js = sorted(j for (gg, j) in dp if gg == g)
    cd = [complex(dp[(g, j)][0], dp[(g, j)][1]) for j in js]
    cq = [qp[(g, j)] for j in js]
    ti = [complex(dp[(g, j)][2], dp[(g, j)][3]) for j in js]
    cmax = max(abs(c) for c in cq)
    err = [abs(a - b) for a, b in zip(cd, cq)]
    terms = [c * t for c, t in zip(cq, ti)]
    tot = sum(terms)
    cancel = sum(abs(t) for t in terms) / abs(tot)
    shift = sum((a - b) * t for a, b, t in zip(cd, cq, ti))
    rank = [0] + [1] * 4 + [2] * 10 + [3] * 20
    print('group %d: %d coefficients, max|coef| %.2e, max rel err of a coefficient %.1e (relative to max|coef| %.1e)' % (
        g, len(js), cmax, max(e / abs(b) for e, b in zip(err, cq) if abs(b) > 1e-30 * cmax), max(err) / cmax))
    for r in range(4):
        idx = [k for k in range(len(js)) if rank[k] == r]
        print('   rank %d: max|coef| %.2e  max|err| %.1e  max|tensor integral| %.2e  sum|coef*I| %.2e' % (
            r, max(abs(cq[k]) for k in idx), max(err[k] for k in idx), max(abs(ti[k]) for k in idx),
            sum(abs(terms[k]) for k in idx)))
    print('   contraction (finite part): sum|terms| / |sum| = %.1e; the coefficient errors shift it by %.1e relative' % (
        cancel, abs(shift) / abs(tot)))
