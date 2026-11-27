"""Reversed-loop pass (CTModeRun 2) with the doubled pentagons (powers, N=5)
replaced by their QP check-pass values (CTModeRun 5), at chosen points."""
import concurrent.futures, json, os, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
outdir, est = sys.argv[1], json.load(open(sys.argv[2]))
idx = [int(x) for x in sys.argv[3].split(',')]
W = os.path.dirname(outdir)
pdir = olc.subprocess_dir(outdir)
def read(path):
    calls, cur = [], None
    for line in open(path):
        if line.startswith('C '):
            f = line.split(); cur = dict(idx=int(f[2]), kind=int(f[3]), n=int(f[4]), rank=int(f[5]), nc=int(f[6]), status=int(f[7]), rows=[]); calls.append(cur)
        else:
            cur['rows'].append(line.split())
    return calls
def run(mode, p, env):
    for k in ('ONIA_DBG_DUMP', 'ONIA_DBG_SUBST'):
        os.environ.pop(k, None)
    os.environ.update(env)
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
    r = olc.run_points(pdir, [p], None, warmup=0, timeout=3600)[0]
    for k in env:
        os.environ.pop(k, None)
    return r['finite_norm']
for i in idx:
    p = est['points'][i]; q = est['qp'][i]; b = os.path.join(W, 'rev_p%d' % i)
    c2 = run('2', p, {'ONIA_DBG_DUMP': b + '_ct2.dump'})
    c5 = run('5', p, {'ONIA_DBG_DUMP': b + '_ct5.dump'})
    d2, d5 = read(b + '_ct2.dump'), read(b + '_ct5.dump')
    ok = len(d2) == len(d5) and all((a['kind'], a['n'], a['rank']) == (c['kind'], c['n'], c['rank']) for a, c in zip(d2, d5))
    sel = [c for c in d5 if c['kind'] == 1 and c['n'] == 5 and c['status'] == 0]
    with open(b + '.subst', 'w') as f:
        f.write('%d\n' % len(sel))
        for c in sel:
            f.write('%d %d\n' % (c['idx'], c['nc']))
            f.write('\n'.join(' '.join(r) for r in c['rows']) + '\n')
    c2s = run('2', p, {'ONIA_DBG_SUBST': b + '.subst'})
    print('p%-3d ct2 err %.1e -> with QP pentagons %.1e (ct5 vs ct4 %.1e, aligned %s, %d pentagons)' % (
        i, abs(c2 - q) / abs(q), abs(c2s - q) / abs(q), abs(c5 - q) / abs(q), ok, len(sel)), flush=True)
