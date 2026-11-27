import json, os, sys, time
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
outdir, est = sys.argv[1], json.load(open(sys.argv[2]))
W = os.path.dirname(outdir); pdir = olc.subprocess_dir(outdir)
olc.build_driver(pdir)
for mode in ('4', '1'):
    os.environ['ONIA_DBG_DUMP'] = os.path.join(W, 'time_ct%s.dump' % mode)
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
    t0 = time.time(); r = olc.run_points(pdir, [est['points'][0]], None, warmup=0, timeout=3600)[0]
    os.environ.pop('ONIA_DBG_DUMP')
    calls = [l.split() for l in open(os.path.join(W, 'time_ct%s.dump' % mode)) if l.startswith('C ')]
    ts = [float(c[8]) for c in calls]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    pent = sum(g for g, c in zip(gaps, calls[1:]) if c[3] == '1' and c[4] == '5')
    pw = sum(g for g, c in zip(gaps, calls[1:]) if c[3] == '1')
    print('CTModeRun %s: point %.2f s (driver), calls span %.2f s; doubled pentagons %.2f s, all powers calls %.2f s' % (
        mode, r['time'], ts[-1] - ts[0], pent, pw))
