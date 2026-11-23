"""The generated extended-precision switch (#PWaveExtendedPrecision,
madgraph/loop/onia_ep.py) at the points of an estim.py run, against its plain
quadruple precision: for each MODE, given as SWITCH:CTMODERUN (off:1, on:1,
on:2, on:-1), the error of the finite part of each point, the time the driver
measures and the return codes.

    ep_switch.py OUTDIR ESTIM.json NWORKERS MODE [MODE ...]

CTModeRun 1 and 2 are the two double-precision passes of MadLoop's stability
test (2: the loop direction reversed, TIRust's check pass); -1 is the test
itself (MLStabThres 1e-3), which returns the mean of the two. The results go
to OUTDIR_ep_switch.json."""
import concurrent.futures
import json
import statistics
import sys

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc

outdir, est, nw = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3])
modes = sys.argv[4:]
pts = est['points']
pdir = olc.subprocess_dir(outdir)


def run(on, mode):
    olc.set_reduction(outdir, ['9'], extra={
        'CTModeRun': mode, 'MLStabThres': '1.0d-3' if mode == '-1' else '1.0d99',
        'PWaveExtendedPrecision': '.TRUE.' if on else '.FALSE.'})
    chunks = [pts[i::nw] for i in range(nw)]
    with concurrent.futures.ThreadPoolExecutor(nw) as pool:
        res = list(pool.map(lambda c: olc.run_points(pdir, c, None, warmup=0, timeout=36000)
                            if c else [], chunks))
    return [res[i % nw][i // nw] for i in range(len(pts))]


out = {}
for m in modes:
    switch, mode = m.split(':')
    res = run(switch == 'on', mode)
    err = [abs(r['finite_norm'] - q) / abs(q) for r, q in zip(res, est['qp'])]
    out[m] = dict(err=err, time=[r['time'] for r in res],
                  codes=sorted(set(r['ret_code'] for r in res)),
                  accuracy=[r['accuracy'] for r in res])
    print('%-6s largest %.1e  median %.1e  >=1e-6: %2d/%d  %.3f s/pt  codes %s'
          % (m, max(err), statistics.median(err), sum(e >= 1e-6 for e in err), len(err),
             statistics.mean(out[m]['time']), out[m]['codes']), flush=True)
json.dump(dict(points=pts, qp=est['qp'], modes=out), open(outdir.rstrip('/') + '_ep_switch.json', 'w'))
