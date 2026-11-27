"""The low-word tests (lowwords.py, ep_retarget.py) at the points of an
estim.py run, against its plain quadruple precision:
- lw1: QP MadLoop, raised-power integrals RAW + RAW_LO from the binary64
  inputs (ONIA_DBG_LW=1, the hybrid test with low words);
- lw2: QP MadLoop, raised-power integrals in binary128 from the binary64
  inputs (ONIA_DBG_LW=2; not usable, binary128 refuses them);
- lw3, lw4: the split of lw1 (ONIA_DBG_LW=3: only the tier-served calls
  take RAW + RAW_LO; =4: only the binary64-served calls take RAW);
- lw5: QP MadLoop, extended inputs (lowwords.py --ext, ONIA_DBG_LW=5);
  lw5u: the same with the unit-power integrals from TIRust's binary64 entry
  (ONIA_DBG_UNIT=1, as in the 80-bit copy); lw5x: with the unit powers
  through the extended entry, 106-bit inputs (ONIA_DBG_UNIT=2);
- ep_hi, ep_lw, ep_ext, ep_all: 80-bit MadLoop with RAW, with RAW + RAW_LO
  (ONIA_EP_LOWWORDS=1), with extended inputs (=2) and with the unit-power
  calls on extended inputs too (=3);
- routes, routes_ext: the TIRust route of every raised-power call at the
  first point.

    lowwords_run.py QP_OUTDIR EP_OUTDIR ESTIM.json OUT.json NWORKERS [KEY ...]

With keys, only those runs are made, merged into an existing OUT.json."""
import collections
import concurrent.futures
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc

qpdir, epdir, est, outjson, nw = sys.argv[1], sys.argv[2], json.load(open(sys.argv[3])), sys.argv[4], int(sys.argv[5])
keys = sys.argv[6:] or ['qp_check', 'lw1', 'lw2', 'ep_hi', 'ep_lw', 'routes']
pts = est['points']


def run(outdir, env, points=pts, mode='4'):
    pdir = olc.subprocess_dir(outdir)
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
    os.environ.update(env)
    try:
        chunks = [points[i::nw] for i in range(nw)]
        with concurrent.futures.ThreadPoolExecutor(nw) as pool:
            res = list(pool.map(lambda c: olc.run_points(pdir, c, None, warmup=0, timeout=36000) if c else [], chunks))
    finally:
        for k in env:
            os.environ.pop(k)
    return [res[i % nw][i // nw] for i in range(len(points))]


def routes(outdir, env):
    with tempfile.NamedTemporaryFile(suffix='.routes', delete=False) as f:
        name = f.name
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '4', 'MLStabThres': '1.0d99'})
    os.environ.update(env, ONIA_DBG_ROUTES=name)
    try:
        olc.run_points(olc.subprocess_dir(outdir), pts[:1], None, warmup=0, timeout=36000)
    finally:
        for k in list(env) + ['ONIA_DBG_ROUTES']:
            os.environ.pop(k)
    count = collections.Counter()
    for line in open(name):
        f = line.split()
        count['N=%s route %s' % (f[1], f[3])] += 1
    os.unlink(name)
    return dict(sorted(count.items()))


out = json.load(open(outjson)) if sys.argv[6:] and os.path.exists(outjson) else {}
out.update(points=pts, qp=est['qp'], ct1=est['ct1'])
if 'qp_check' in keys:
    out['qp_check'] = [r['finite_norm'] for r in run(qpdir, {}, pts[:3])]
for key, outdir, env in (('lw1', qpdir, {'ONIA_DBG_LW': '1'}), ('lw2', qpdir, {'ONIA_DBG_LW': '2'}),
                         ('lw3', qpdir, {'ONIA_DBG_LW': '3'}), ('lw4', qpdir, {'ONIA_DBG_LW': '4'}),
                         ('lw5', qpdir, {'ONIA_DBG_LW': '5'}),
                         ('lw5u', qpdir, {'ONIA_DBG_LW': '5', 'ONIA_DBG_UNIT': '1'}),
                         ('lw5x', qpdir, {'ONIA_DBG_LW': '5', 'ONIA_DBG_UNIT': '2'}),
                         ('ep_hi', epdir, {}), ('ep_lw', epdir, {'ONIA_EP_LOWWORDS': '1'}),
                         ('ep_ext', epdir, {'ONIA_EP_LOWWORDS': '2'}),
                         ('ep_all', epdir, {'ONIA_EP_LOWWORDS': '3'})):
    if key not in keys:
        continue
    res = run(outdir, env)
    out[key] = [r['finite_norm'] for r in res]
    out[key + '_time'] = [r['time'] for r in res]
    out[key + '_codes'] = sorted(set(r['ret_code'] for r in res))
if 'routes' in keys:
    out['routes_lw1'] = routes(qpdir, {'ONIA_DBG_LW': '1'})
    out['routes_ep_lw'] = routes(epdir, {'ONIA_EP_LOWWORDS': '1'})
if 'routes_ext' in keys:
    out['routes_lw5'] = routes(qpdir, {'ONIA_DBG_LW': '5'})
    out['routes_ep_ext'] = routes(epdir, {'ONIA_EP_LOWWORDS': '2'})
json.dump(out, open(outjson, 'w'))

rel = lambda k: [abs(x - q) / abs(q) for x, q in zip(out[k], out['qp'])]
if 'qp_check' in out:
    print('QP of this build against estim QP (3 points): largest %.1e' % max(rel('qp_check')))
names = (('ct1', 'DP production'),
         ('lw2', 'QP MadLoop, binary128 integrals of binary64 inputs'),
         ('lw1', 'QP MadLoop, RAW + RAW_LO'),
         ('lw3', 'QP MadLoop, RAW + RAW_LO of tier-served calls only'),
         ('lw4', 'QP MadLoop, RAW of binary64-served calls only'),
         ('lw5', 'QP MadLoop, extended inputs (106 bits)'),
         ('lw5u', '... with binary64 unit-power integrals'),
         ('lw5x', '... with unit powers through the extended entry'),
         ('ep_hi', '80-bit MadLoop, RAW'),
         ('ep_lw', '80-bit MadLoop, RAW + RAW_LO'),
         ('ep_ext', '80-bit MadLoop, extended inputs (64 bits)'),
         ('ep_all', '80-bit MadLoop, extended inputs, unit powers too'))
for k, name in names:
    if k not in out:
        continue
    e = rel(k)
    t = ('  %.3f s/pt' % statistics.mean(out[k + '_time'])) if k + '_time' in out else ''
    c = ('  codes %s' % out[k + '_codes']) if k + '_codes' in out else ''
    print('%-52s largest %.1e  median %.1e%s%s' % (name, max(e), statistics.median(e), t, c))
for k in ('routes_lw1', 'routes_ep_lw', 'routes_lw5', 'routes_ep_ext'):
    if k in out:
        print('routes at point 0 (%s):' % k[7:], out[k])
cols = [k for k, _ in names if k in out]
for i in range(len(pts)):
    print('p%-3d ' % i + '  '.join('%s %.1e' % (k, rel(k)[i]) for k in cols))
