"""Outputs (current templates, TIRust da7902b + v16) and points for the recalibration."""
import json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
R = '/shared/external/madloop_onia/madgraph7_madloop_onia'
B = R + '/docs/benchmarkfolder'
sys.path.insert(0, R + '/tools'); sys.path.insert(0, B + '/research/mg255_swave'); sys.path.insert(0, B + '/research/stability')
import onia_loop_check as olc, compare, ml_side, stability as st
SP = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad'
W = SP + '/est4'; LIB = SP + '/pw73/install/lib'; PY = sys.executable
def gen(name, model, proc, masses):
    out = olc.output_process(model, proc, os.path.join(W, 'out', name), extra=['set tirust %s --no_save' % LIB], regenerate=True)
    if masses: olc.set_masses(out, masses)
    olc.build_driver(olc.subprocess_dir(out))
    return out
def cmd(name, out, qp=True):
    c = [PY, W + '/calib2.py', out, W + '/pts_%s.json' % name, W + '/cal_%s.json' % name]
    if qp: c.append(W + '/qp_%s.json' % name)
    return name, c
def mg255(case):
    model, proc, masses = ml_side.CASES[case]
    out = gen(case, model, proc, masses)
    src, mq, en = compare.CASES[case][0], compare.CASES[case][2], compare.CASES[case][6]
    pts = compare.ref_points(compare.POINTS_FROM.get(case, src)) + compare.random_points(mq, en)
    json.dump(dict(points=[[p[0], p[1], [a + b for a, b in zip(p[2], p[3])], p[4]] for _, p in pts], mu=11.0), open(W + '/pts_%s.json' % case, 'w'))
    json.dump([r['qp']['fin'] for r in json.load(open(B + '/data/raw/v16all/mg255excl_%s.json' % case))['rows']], open(W + '/qp_%s.json' % case, 'w'))
    return cmd(case, out)
def s18(tag, proc):
    out = gen('s18_' + tag, 'loop_sm_onia-c_mass', proc, None)
    pdir = olc.subprocess_dir(out)
    # the onia card as in q3s18 (LDME 1, mass 3.1): the finite part ratios do not depend on it
    d = json.load(open(SP + '/q3s18/%s/results.json' % tag))
    json.dump(dict(points=d['points'], mu=11.0), open(W + '/pts_s18_%s.json' % tag, 'w'))
    json.dump([r['finite_norm'] for r in d['results']['4']], open(W + '/qp_s18_%s.json' % tag, 'w'))
    return cmd('s18_' + tag, out)
def pwave(tag, proc):
    out = gen(tag, 'loop_sm_onia-c_mass', proc, None)
    e = json.load(open(SP + '/ep16/estim_%s.json' % tag[:2]))
    json.dump(dict(points=e['points']), open(W + '/pts_%s.json' % tag, 'w'))
    json.dump(e['qp'], open(W + '/qp_%s.json' % tag, 'w'))
    return cmd(tag, out)
def random_case(name, model, proc, n):
    out = gen(name, model, proc, None)
    masses = olc.driver_masses(olc.subprocess_dir(out))
    json.dump(dict(points=st.make_points(masses, n, 1000., 50.)), open(W + '/pts_%s.json' % name, 'w'))
    return cmd(name, out, qp=False)
with ThreadPoolExecutor(18) as ex:
    futs = [ex.submit(mg255, c) for c in sorted(ml_side.CASES)]
    futs += [ex.submit(s18, 'gg', 'g g > Jpsi(1|3S18) g / t t~ b b~ [virt=QCD]'),
             ex.submit(s18, 'ug', 'u g > Jpsi(1|3S18) u / t t~ b b~ [virt=QCD]'),
             ex.submit(s18, 'uux', 'u u~ > Jpsi(1|3S18) g / t t~ b b~ [virt=QCD]'),
             ex.submit(pwave, 'uu_chic0g', 'u u~ > chic0(1|3P01) g [virt=QCD]'),
             ex.submit(pwave, 'gg_chic0g', 'g g > chic0(1|3P01) g [virt=QCD]'),
             ex.submit(random_case, 'jpsig', 'loop_sm_onia-c_mass', 'g g > Jpsi(1|3S11) g [virt=QCD]', 200),
             ex.submit(random_case, 'etac8g', 'loop_sm_onia-c_mass', 'g g > etac(1|1S08) g [virt=QCD]', 60),
             ex.submit(random_case, 'cc3p08g', 'loop_sm_onia-c_mass', 'e+ e- > Jpsi(1|3P08) g [virt=QCD]', 60)]
    cmds = [f.result() for f in futs]
def runc(nc):
    name, c = nc
    p = subprocess.run(c, capture_output=True, text=True)
    return name, (p.stdout + p.stderr[-1500:]).strip()
with ThreadPoolExecutor(len(cmds)) as ex:
    for name, txt in ex.map(runc, cmds):
        print('==', name); print(txt[-800:], flush=True)
