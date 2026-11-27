"""probe.py OUTDIR POINTS.json INDEX MU: groups split DP vs QP; per-N integral comparison; QP
integral substitution per N; QP coefficient substitution of the worst groups."""
import json, os, subprocess, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
out, ptsf, idx, mu = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
pdir = olc.subprocess_dir(out)
pt = json.load(open(ptsf))['points'][idx]
lines = ['1', '%.17e' % mu] + [' '.join('%.17e' % c for c in v) for v in pt]
def run(mode, env={}):
    olc.set_reduction(out, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99', 'ImprovePSPoint': '2'})
    o = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir, input='\n'.join(lines) + '\n',
                       capture_output=True, text=True, check=True, env=dict(os.environ, **env)).stdout.splitlines()
    ct = [l.split() for l in o if l.startswith('SPLITDBG CT')][-1]
    g = {}
    for l in o:
        if l.startswith('SPLITDBG LG') and l.split()[2] == ct[2]:
            f = l.split(); g[int(f[3])] = float(f[4])
    return dict(ct=float(ct[3]), ans=float(ct[6]), g=g, out=o)
q = run('4', {'LCQ_DUMP': '1'})
last = {}
for l in q['out']:
    if l.startswith('LCQ'):
        f = l.split(); last[(int(f[1]), int(f[2]))] = (f[3], f[4])
open(os.path.join(pdir, 'lcq.txt'), 'w').write(''.join('%d %d %s %s\n' % (i, j, a, b) for (i, j), (a, b) in sorted(last.items())))
d = run('1')
err = {i: d['g'][i] - q['g'][i] for i in d['g']}
worst = sorted(err, key=lambda i: -abs(err[i]))[:6]
print('total rel error %.2e | CT %.1e | loops %.2e' % ((d['ans'] - q['ans']) / abs(q['ans']), d['ct'] - q['ct'], sum(err.values())))
print('worst groups:', ', '.join('%d (%.2e of %.3e)' % (i, err[i], q['g'][i]) for i in worst))
for n in ('2', '3', '4', '5'):
    x = run('1', {'TIR_QCMP_N': n})
    cmp_ = [l.split() for l in x['out'] if l.startswith('QSBCMP')]
    if not cmp_: continue
    ok = [c for c in cmp_ if c[5] == 'T']
    mx = max([float(c[6]) for c in ok] or [0])
    y = run('1', {'TIR_QCMP_N': n, 'TIR_QSUBST_USE': '1'})
    print('N=%s: %d calls (%d QP served), DP vs QP integrals up to %.1e (finite, rel. to max); substituted: total %.2e, worst groups %s' % (
        n, len(cmp_), len(ok), mx, (y['ans'] - q['ans']) / abs(q['ans']), ', '.join('%.1e' % (y['g'][i] - q['g'][i]) for i in worst[:3])))
open(os.path.join(pdir, 'lcs_groups.txt'), 'w').write('\n'.join(str(i) for i in worst[:3]) + '\n')
z = run('1', {'LC_SUBST': '1'})
print('QP coefficients for groups %s: total %.2e, those groups %s' % (worst[:3], (z['ans'] - q['ans']) / abs(q['ans']), ', '.join('%.1e' % (z['g'][i] - q['g'][i]) for i in worst[:3])))
olc.set_reduction(out, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'ImprovePSPoint': '2'})
