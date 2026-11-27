"""calib2.py OUTDIR POINTS.json OUT.json [QP.json]: with ONIA_TIRK=0 in the output (so the
accuracy is MadLoop's estimate without the term), print ONIA_TIRREL(1:2) per point; the true error
against quadruple precision (given or computed)."""
import json, os, re, subprocess, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
EPS = 2.220446049250313e-16
outdir, ptsf, outf = sys.argv[1:4]
d = json.load(open(ptsf)); pts, mu = d['points'], d.get('mu')
pdir = olc.subprocess_dir(outdir)
p = os.path.join(pdir, 'loop_matrix.f'); s = open(p).read()
if 'TIRCAL2' not in s:
    assert s.count('PARAMETER (ONIA_TIRK=10.0D0)') == 1
    s = s.replace('PARAMETER (ONIA_TIRK=10.0D0)', 'PARAMETER (ONIA_TIRK=0.0D0)')
    m = re.search(r"\n( +)ACC\(I\)=MAX\(ACC\(I\),ONIA_CTACC,ONIA_TIRACC\)\n +ENDDO\n", s)
    s = s[:m.end()] + """      WRITE(*,'(A,4ES14.5)') 'TIRCAL2',ONIA_TIRREL(1),ONIA_TIRREL(2),
     $ ONIA_CTACC,ACC(0)
""" + s[m.end():]
    open(p, 'w').write(s)
    olc.build_driver(pdir)
olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2',
                                       'NRotations_DP': '0', 'NRotations_QP': '0'})
lines = [str(len(pts))]
for pt in pts:
    lines.append('%.17e' % (mu if mu else -1.0))
    lines.extend(' '.join('%.17e' % c for c in v) for v in pt)
o = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                   capture_output=True, text=True, check=True).stdout.splitlines()
cal = [[float(x) for x in l.split()[1:]] for l in o if l.startswith('TIRCAL2')][-len(pts):]
dp = olc.run_points(pdir, pts, mu, timeout=36000)
if len(sys.argv) > 4:
    qp = json.load(open(sys.argv[4]))
else:
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '4', 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2'})
    qp = [r['finite_norm'] for r in olc.run_points(pdir, pts, mu, timeout=36000)]
olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
rows = [dict(err=abs(r['finite_norm'] - q) / abs(q), acc=r['accuracy'], code=r['ret_code'],
             rel=EPS * max(c[0], c[1]), ctacc=c[2]) for r, c, q in zip(dp, cal, qp)]
json.dump(rows, open(outf, 'w'), indent=1)
print('%d points: err max %.1e, acc max %.1e, eps*TIRREL max %.1e median %.1e, err>acc at %d' % (
    len(rows), max(r['err'] for r in rows), max(r['acc'] for r in rows), max(r['rel'] for r in rows),
    sorted(r['rel'] for r in rows)[len(rows) // 2], sum(r['err'] > r['acc'] for r in rows)))
