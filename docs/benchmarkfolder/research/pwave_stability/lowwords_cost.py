"""Paired cost of two run-time modes of one 80-bit output (ep_retarget.py
after lowwords.py): alternating batches over the points of an estim.py run,
CTModeRun 4, the median time per point of each batch as the driver measures
it. Modes are values of ONIA_EP_LOWWORDS ('' = RAW, 1 = RAW + RAW_LO,
2 = extended inputs).

    lowwords_cost.py EP_OUTDIR ESTIM.json PAIRS NWORKERS MODE_A MODE_B"""
import concurrent.futures
import json
import os
import statistics
import sys

sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc

outdir, est, pairs, nw, a, b = sys.argv[1], json.load(open(sys.argv[2])), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
pts = est['points']
pdir = olc.subprocess_dir(outdir)
olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '4', 'MLStabThres': '1.0d99'})


def batch(mode):
    if mode:
        os.environ['ONIA_EP_LOWWORDS'] = mode
    else:
        os.environ.pop('ONIA_EP_LOWWORDS', None)
    chunks = [pts[i::nw] for i in range(nw)]
    with concurrent.futures.ThreadPoolExecutor(nw) as pool:
        res = list(pool.map(lambda c: olc.run_points(pdir, c, None, warmup=1, timeout=36000) if c else [], chunks))
    os.environ.pop('ONIA_EP_LOWWORDS', None)
    return statistics.median(r['time'] for c in res for r in c)


med = {a: [], b: []}
for i in range(pairs):
    for mode in ((a, b) if i % 2 == 0 else (b, a)):
        med[mode].append(batch(mode))
ratios = [y / x for x, y in zip(med[a], med[b])]
print('mode %r: %s s/pt' % (a, ' '.join('%.3f' % t for t in med[a])))
print('mode %r: %s s/pt' % (b, ' '.join('%.3f' % t for t in med[b])))
print('ratio %r/%r: median %.3f (%s)' % (b, a, statistics.median(ratios), ' '.join('%.3f' % r for r in ratios)))
json.dump(dict(modes=[a, b], medians=med, ratios=ratios), open(os.path.join(os.path.dirname(os.path.abspath(outdir)),
          'cost_ep_%s_%s_%s.json' % (os.path.basename(outdir.rstrip('/')), a or 'raw', b or 'raw')), 'w'))
