"""per point at setup A, MadLoop's DP primary (CTModeRun 1), its
check pass (CTModeRun 2: reversed loops, TIRust's shifted invariants), the
primary at the momenta rotated as MadLoop's first DP rotation (x->z, y->x,
z->y), and QP (CTModeRun 4). Usage: estim.py OUTDIR NPOINTS ENERGY PTMIN NWORKERS"""
import concurrent.futures, json, os, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import importlib.util
import onia_loop_check as olc
spec = importlib.util.spec_from_file_location('st', '/shared/external/madloop_onia/madgraph7_madloop_onia/docs/benchmarkfolder/research/stability/stability.py')
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)
outdir, n, energy, ptmin, nw = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
pdir = olc.subprocess_dir(outdir)
masses = olc.driver_masses(pdir)
pts = st.make_points(masses, n, energy, ptmin)
rot = [[[p[0], p[2], p[3], p[1]] for p in pt] for pt in pts]
def run(mode, points):
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
    chunks = [points[i::nw] for i in range(nw)]
    with concurrent.futures.ThreadPoolExecutor(nw) as pool:
        res = list(pool.map(lambda c: olc.run_points(pdir, c, None, warmup=0, timeout=36000) if c else [], chunks))
    return [res[i % nw][i // nw]['finite_norm'] for i in range(len(points))]
out = {}
for name, mode, points in (('ct1', '1', pts), ('ct2', '2', pts), ('rot', '1', rot), ('qp', '4', pts)):
    out[name] = run(mode, points)
    print(name, 'done', flush=True)
tag = os.path.join(os.path.dirname(outdir), 'estim_%s_%g.json' % (os.path.basename(outdir), energy))
json.dump(dict(points=pts, **out), open(tag, 'w'))
for i in range(n):
    q = out['qp'][i]
    e = lambda k: abs(out[k][i] - q) / abs(q)
    print('p%-3d true %.1e | ct2 %.1e (|ct1-ct2| %.1e) | rot %.1e (|ct1-rot| %.1e)' % (
        i, e('ct1'), e('ct2'), abs(out['ct1'][i] - out['ct2'][i]) / abs(q), e('rot'),
        abs(out['ct1'][i] - out['rot'][i]) / abs(q)), flush=True)
