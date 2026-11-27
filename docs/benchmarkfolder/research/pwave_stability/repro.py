"""Reproducer for TIRust: the DP calls with powers and five lines (the doubled
pentagons) of u u~ > chic0 g at chosen points of setup A, in the primary pass
(CTModeRun 1) and the reversed-loop pass (CTModeRun 2), with their inputs, the
DP arrays and the QP arrays of the same calls (CTModeRun 4 and 5), as hex
doubles. The output must carry instrument.py's patch.

    repro.py OUTDIR OUT.json.gz POINT [POINT ...]

The reversed pass's inputs are those MadLoop passes (after SWITCH_ORDER); TIRust
moves the invariants of a check pass itself (CHECK_PASS), and its DP array
includes that shift. QP arrays are rounded to double."""
import gzip, json, os, struct, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'tools'))
import importlib.util
import onia_loop_check as olc
spec = importlib.util.spec_from_file_location('st', os.path.join(HERE, '..', 'stability', 'stability.py'))
st = importlib.util.module_from_spec(spec); spec.loader.exec_module(st)

def read_dump(path):
    calls, cur = [], None
    for line in open(path):
        if line.startswith('C '):
            f = line.split()
            cur = dict(prec=f[1], idx=int(f[2]), kind=int(f[3]), n=int(f[4]), rank=int(f[5]), nc=int(f[6]), status=int(f[7]), rows=[])
            calls.append(cur)
        else:
            cur['rows'].append(['0x' + h for h in line.split()])
    return calls

def hexd(h):
    return float.hex(struct.unpack('>d', bytes.fromhex(h[2:]))[0])

def read_inp(path):
    recs, cur = [], None
    for line in open(path):
        tag = line.split()[0]
        if tag == 'P':
            f = [int(x) for x in line.split()[1:]]
            cur = dict(ctmode=f[1], n=f[2], rank=f[3], powers=f[4:4 + f[2]], masses2=[], invariants=[], shifts=[])
            recs.append(cur)
        elif tag == 'MU2':
            cur['mu2'] = float.hex(float(line.split()[1]))
        elif tag == 'M':
            v = [float(x) for x in line.split()[2:]]; cur['masses2'].append([float.hex(v[0]), float.hex(v[1])])
        elif tag == 'I':
            cur['invariants'].append([float.hex(float(x)) for x in line.split()[2:]])
        elif tag == 'S':
            cur['shifts'].append([float.hex(float(x)) for x in line.split()[2:]])
    return recs

outdir, out = sys.argv[1], sys.argv[2]
idx = [int(x) for x in sys.argv[3:]]
pdir = olc.subprocess_dir(outdir)
olc.build_driver(pdir)
masses = olc.driver_masses(pdir)
pts = st.make_points(masses, max(idx) + 1, 1000., 50.)
W = os.path.dirname(os.path.abspath(outdir))

def run(mode, p, dump, inp=None):
    for k in ('ONIA_DBG_DUMP', 'ONIA_DBG_INP'):
        os.environ.pop(k, None)
    os.environ['ONIA_DBG_DUMP'] = dump
    if inp:
        os.environ['ONIA_DBG_INP'] = inp
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d99'})
    r = olc.run_points(pdir, [p], None, warmup=0, timeout=3600)[0]
    for k in ('ONIA_DBG_DUMP', 'ONIA_DBG_INP'):
        os.environ.pop(k, None)
    return r['finite_norm']

res = dict(process='u u~ > chic0(1|3P01) g [virt=QCD]', model='loop_sm_onia-c_mass', setup='sqrt(s) = 1000 GeV, pT > 50 GeV',
           array_layout='rows k = 1..NC in TIRust order; per row 8 hex doubles: (re, im) of RAW(k,1..4); MadLoop uses RAW(:,1)+RAW(:,2) as 1/eps, RAW(:,3) as 1/eps^2, RAW(:,4) as the finite part',
           points=[])
for k in idx:
    p = pts[k]; b = os.path.join(W, 'repro_p%d' % k)
    entry = dict(point=k, momenta=[[float.hex(x) for x in v] for v in p], passes={})
    for name, dpmode, qpmode in (('primary', '1', '4'), ('reversed', '2', '5')):
        fdp = run(dpmode, p, b + '_%s_dp.dump' % name, b + '_%s.inp' % name)
        fqp = run(qpmode, p, b + '_%s_qp.dump' % name)
        dp, qp, inp = read_dump(b + '_%s_dp.dump' % name), read_dump(b + '_%s_qp.dump' % name), read_inp(b + '_%s.inp' % name)
        powers_dp = [c for c in dp if c['kind'] == 1]
        assert len(powers_dp) == len(inp)
        assert len(dp) == len(qp)
        calls = []
        for c, i in zip(powers_dp, inp):
            if c['n'] != 5:
                continue
            q = qp[c['idx']]
            assert (q['kind'], q['n'], q['rank']) == (c['kind'], c['n'], c['rank'])
            dpv = [[struct.unpack('>d', bytes.fromhex(h[2:]))[0] for h in row] for row in c['rows']]
            qpv = [[struct.unpack('>d', bytes.fromhex(h[2:]))[0] for h in row] for row in q['rows']]
            big = max(abs(complex(r[6], r[7])) for r in qpv) or 1.
            dev = max(abs(complex(a[6] - b_[6], a[7] - b_[7])) for a, b_ in zip(dpv, qpv)) / big
            calls.append(dict(call=c['idx'], status_dp=c['status'], status_qp=q['status'], max_finite_dev_rel_to_largest=dev,
                              **i, dp_array=[[hexd(h) for h in row] for row in c['rows']],
                              qp_array=[[hexd(h) for h in row] for row in q['rows']]))
        entry['passes'][name] = dict(finite_norm_dp=fdp, finite_norm_qp=fqp, rel_error=abs(fdp - fqp) / abs(fqp), pentagons=calls)
        print('p%d %s: DP %.10f QP %.10f rel %.1e, %d doubled pentagons, worst array dev %.1e' % (
            k, name, fdp, fqp, abs(fdp - fqp) / abs(fqp), len(calls), max(x['max_finite_dev_rel_to_largest'] for x in calls)), flush=True)
    res['points'].append(entry)
with gzip.open(out, 'wt') as f:
    json.dump(res, f)
print('wrote', out)
