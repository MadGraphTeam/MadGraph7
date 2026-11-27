"""Cost of one TIRust build against another on the same points: alternating
batches of the two outputs, the median time per point of each batch as the
driver measures it (initialisation excluded), and in production mode the
number of points MadLoop rescues in quadruple precision.

    cost.py CASE SETUP OLD_OUTDIR NEW_OUTDIR NPOINTS PAIRS MODE [NWORKERS]

MODE 'profile' uses the settings of the stability profiles (CTModeRun -1,
MLStabThres 1e99, NRotations_DP 1); 'production' MadLoop's defaults
(CTModeRun -1, MLStabThres 1e-3, no rotations, quadruple-precision rescue).
With the suffix '+ep' (profile+ep, production+ep) NEW_OUTDIR runs with the
80-bit switch (#PWaveExtendedPrecision .TRUE.) and OLD_OUTDIR without it: the
cost of the switch, for two copies of one output. Each batch is split over
NWORKERS drivers (1 by default). Writes cost_CASE_SETUP_MODE.json next to
NEW_OUTDIR."""
import concurrent.futures
import json
import os
import statistics
import sys

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/docs/benchmarkfolder/research/stability')
import onia_loop_check as olc
import stability as st

case, setup, old, new, npts, pairs, mode = sys.argv[1:8]
npts, pairs = int(npts), int(pairs)
nw = int(sys.argv[8]) if len(sys.argv) > 8 else 1
energy, ptmin = st.SETUPS[setup]
switch = mode.endswith('+ep')
SETTINGS = {
    'profile': {'CTModeRun': '-1', 'MLStabThres': '1.0d99', 'NRotations_DP': '1', 'NRotations_QP': '0'},
    'production': {'CTModeRun': '-1', 'MLStabThres': '1.0d-3', 'NRotations_DP': '0', 'NRotations_QP': '0'},
}[mode[:-3] if switch else mode]
pdirs = {}
for name, outdir in (('old', old), ('new', new)):
    pdirs[name] = olc.subprocess_dir(outdir)
    extra = dict(SETTINGS)
    if switch:
        extra['PWaveExtendedPrecision'] = '.TRUE.' if name == 'new' else '.FALSE.'
    olc.set_reduction(outdir, ['9'], extra=extra)
masses = olc.driver_masses(pdirs['new'])
points = st.make_points(masses, npts, energy, ptmin)


def batch(name):
    chunks = [points[i::nw] for i in range(nw)]
    with concurrent.futures.ThreadPoolExecutor(nw) as pool:
        res = list(pool.map(lambda c: olc.run_points(pdirs[name], c, None, warmup=3, timeout=36000)
                            if c else [], chunks))
    rows = [res[i % nw][i // nw] for i in range(npts)]
    return dict(time=[r['time'] for r in rows], ret_code=[r['ret_code'] for r in rows],
                accuracy=[r['accuracy'] for r in rows], finite_norm=[r['finite_norm'] for r in rows])


out = dict(case=case, setup=setup, mode=mode, npoints=npts, pairs=pairs, nworkers=nw,
           old=old, new=new, batches=[])
for k in range(pairs):
    for name in (('old', 'new') if k % 2 == 0 else ('new', 'old')):
        b = batch(name)
        b['build'] = name
        out['batches'].append(b)
        print('%s %s %s pair %d %s: median %.1f ms/pt, mean %.1f ms/pt, QP-rescued %d/%d' % (
            case, setup, mode, k, name, 1e3 * statistics.median(b['time']),
            1e3 * statistics.mean(b['time']), sum(1 for c in b['ret_code'] if (c // 10) % 10 == 2),
            npts), flush=True)
for name in ('old', 'new'):
    med = [statistics.median(b['time']) for b in out['batches'] if b['build'] == name]
    mean = [statistics.mean(b['time']) for b in out['batches'] if b['build'] == name]
    out[name + '_median_ms'] = 1e3 * statistics.median(med)
    out[name + '_mean_ms'] = 1e3 * statistics.median(mean)
print('%s %s %s: old %.1f / new %.1f ms per point (median of batch medians), means %.1f / %.1f' % (
    case, setup, mode, out['old_median_ms'], out['new_median_ms'], out['old_mean_ms'], out['new_mean_ms']))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(new)),
                                 'cost_%s_%s_%s.json' % (case, setup, mode)), 'w'))
