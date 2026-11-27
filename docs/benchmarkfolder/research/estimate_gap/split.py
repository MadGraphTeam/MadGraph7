"""Print, per pass, the counterterm part (BUFFR_BIS) and every loop group (LOOPRES) of the
finite part and of 1/eps, then compare the DP passes with QP at PS.input."""
import json, os, subprocess, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
OUT = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad/trace/uux'
pdir = olc.subprocess_dir(OUT)
path = os.path.join(pdir, 'loop_matrix.f')
s = open(path).read()
if 'SPLITDBG' not in s:
    old = "C     Make sure that no NaN is present in the result\n"
    assert s.count(old) == 1
    s = s.replace(old, """      WRITE(*,'(A,I3,4ES26.17)') 'SPLITDBG CT',CTMODE,BUFFR_BIS(1,0),
     $ BUFFR_BIS(2,0),BUFFR_BIS(3,0),ANS(1,0)
      DO I=1,NLOOPGROUPS
        WRITE(*,'(A,I3,I6,4ES26.17)') 'SPLITDBG LG',CTMODE,I,
     $   DBLE(LOOPRES(1,1,I)),DIMAG(LOOPRES(1,1,I)),
     $   DBLE(LOOPRES(2,1,I)),DBLE(LOOPRES(3,1,I))
      ENDDO
""" + old, 1)
    open(path, 'w').write(s)
olc.build_driver(pdir)
d = json.load(open(OUT + '_modes.json'))
pt = d['point']
lines = ['1', '%.17e' % 11.0] + [' '.join('%.17e' % c for c in v) for v in pt]
out = {}
for mode in ('1', '2', '4'):
    olc.set_reduction(OUT, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2'})
    o = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                       capture_output=True, text=True, check=True).stdout.splitlines()
    ct = [l.split() for l in o if l.startswith('SPLITDBG CT')]
    lg = [l.split() for l in o if l.startswith('SPLITDBG LG')]
    # the last evaluation of each kind (initialisation passes come first)
    n = len([l for l in lg if l[2] == ct[-1][2]]) // max(1, len(ct))
    last_ct = ct[-1]
    groups = {}
    for l in lg:
        if l[2] == last_ct[2]:
            groups[int(l[3])] = [float(x) for x in l[4:]]
    out[mode] = dict(ctmode=last_ct[2], ct=[float(x) for x in last_ct[3:6]], ans=float(last_ct[6]), groups=groups, nct=len(ct))
olc.set_reduction(OUT, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
q = out['4']
tot = q['ans']
print('QP total finite %.17e, CT finite %.17e, sum|groups| %.3e, |CT|/|total| %.2e' % (
    tot, q['ct'][0], sum(abs(g[0]) for g in q['groups'].values()), abs(q['ct'][0] / tot)))
for m in ('1', '2'):
    x = out[m]
    dct = x['ct'][0] - q['ct'][0]
    dct1 = x['ct'][1] - q['ct'][1]
    dl = sum(x['groups'][i][0] for i in x['groups']) - sum(q['groups'][i][0] for i in q['groups'])
    dl1 = sum(x['groups'][i][2] for i in x['groups']) - sum(q['groups'][i][2] for i in q['groups'])
    print('DP mode %s: total diff %.3e (rel %.1e) = CT %.3e + loops %.3e | 1/eps: CT %.3e loops %.3e (total %.3e)' % (
        m, x['ans'] - tot, (x['ans'] - tot) / abs(tot), dct, dl, dct1, dl1, dct1 + dl1))
    worst = sorted(x['groups'], key=lambda i: -abs(x['groups'][i][0] - q['groups'][i][0]))[:8]
    for i in worst:
        a, b = x['groups'][i], q['groups'][i]
        print('   group %4d fin dp %.6e qp %.6e diff %.2e | 1/eps diff %.2e' % (i, a[0], b[0], a[0] - b[0], a[2] - b[2]))
json.dump(out, open(OUT + '_split.json', 'w'))
