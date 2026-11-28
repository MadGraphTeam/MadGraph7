"""Colour-octet decays at one loop: each 1 -> 2 decay against the 2 -> 1
production with the same kinematics (time reversal: the same virtual), its
Born against the tree-level decay output, and the poles of 1 -> 3 decays
against ONIA_IR_POLES.

Usage: python3 -I octet.py <case> ...   (cases below; 'all')
"""
import os, re, sys, json, subprocess, math
REPO = '/shared/external/madloop_onia/madgraph7_madloop_onia'
sys.path.insert(0, os.path.join(REPO, 'tools'))
import onia_loop_check as olc
# the outputs and octet_results.json go to $OCTET_WORK (default: the current
# directory), not next to this script
HERE = os.path.abspath(os.environ.get('OCTET_WORK', os.getcwd()))
M = 3.1
TIRUST = os.path.join(REPO, 'HEPTools', 'tirust', 'lib')

PAIRS = {  # name: (decay, production, pdg, tree check)
    '1s08_gg': ('etac(1|1S08) > g g', 'g g > etac(1|1S08)', None),
    '3s18_uu': ('Jpsi(1|3S18) > u u~', 'u u~ > Jpsi(1|3S18)', None),
    '3p08_gg': ('Jpsi(1|3P08) > g g', 'g g > Jpsi(1|3P08)', None),
    '3p28_gg': ('Jpsi(1|3P28) > g g', 'g g > Jpsi(1|3P28)', None),
}
POLES = {  # name: decay 1 -> 3
    '1s08_uug': 'etac(1|1S08) > u u~ g',
    '3s18_ggg': 'Jpsi(1|3S18) > g g g',
    '3p18_uug': 'Jpsi(1|3P18) > u u~ g',
}


def build(name, proc):
    d = os.path.join(HERE, 'oc_' + name)
    pdir_ok = False
    try:
        pdir = olc.subprocess_dir(d)
        pdir_ok = os.path.isfile(os.path.join(pdir, 'ml_points_driver'))
    except Exception:
        pass
    if not pdir_ok:
        olc.output_process('loop_sm_onia-c_mass', proc + ' [virt=QCD]', d,
                           extra=['set tirust %s --no_save' % TIRUST], regenerate=True)
        pdir = olc.subprocess_dir(d)
        olc.build_driver(pdir)
    olc.set_reduction(d, [9])
    return d, olc.subprocess_dir(d)


def tree_born(name, proc, point):
    """|M|^2 of the tree-level standalone_fortran decay at the point (its
    onium mass set to m1+m2 = M)."""
    d = os.path.join(HERE, 'ot_' + name)
    if not os.path.isdir(d):
        olc.run_madgraph(['import model sm_onia-c_mass', 'generate %s' % proc,
                          'output standalone_fortran %s -f' % d], d + '.log')
    card = os.path.join(d, 'Cards', 'onia_card.dat')
    text = open(card).read()
    text = re.sub(r'(?ims)(^block\s+onium_mass\s*\n(?:\s*#.*\n)*)((?:\s*\d+\s+\S+.*\n)+)',
                  lambda m: m.group(1) + re.sub(r'(?m)^(\s*\d+\s+)\S+', r'\g<1>%.16f' % M,
                                                m.group(2)), text)
    open(card, 'w').write(text)
    pdir = olc.subprocess_dir(d)
    with open(os.path.join(pdir, 'PS.input'), 'w') as f:
        for v in point:
            f.write(' '.join('%.17e' % x for x in v) + '\n')
    olc.make(pdir, 'check')
    out = subprocess.run(['./check'], cwd=pdir, capture_output=True, text=True).stdout
    return float(re.search(r'Matrix element\s*=\s*(\S+)', out).group(1))


def keys(r):
    return dict((k, r[k]) for k in ('born', 'finite_norm', 'single_norm', 'double_norm',
                                    'predicted_single_norm', 'predicted_double_norm',
                                    'ret_code', 'accuracy'))


def run_pair(name):
    dec, prod, _ = PAIRS[name]
    dd, dp = build(name + '_dec', dec)
    pd, pp = build(name + '_prod', prod)
    rest_d = [[M, 0, 0, 0], [M / 2, 0, 0, M / 2], [M / 2, 0, 0, -M / 2]]
    rest_p = [[M / 2, 0, 0, M / 2], [M / 2, 0, 0, -M / 2], [M, 0, 0, 0]]
    bd = [olc.boost_vector(v, olc.DECAY_BOOST) for v in rest_d]
    bp = [olc.boost_vector(v, olc.DECAY_BOOST) for v in rest_p]
    rd = olc.run_points(dp, [rest_d, bd], mu=M)
    rp = olc.run_points(pp, [rest_p, bp], mu=M)
    tb = tree_born(name, dec, rest_d)
    out = dict(decay=[keys(r) for r in rd], production=[keys(r) for r in rp], tree_born=tb)
    print('==', name, dec, '|', prod)
    for k, (a, b) in enumerate(zip(rd, rp)):
        print('  %s  decay fin %.12f pole %.10f/%.10f (pred %.10f/%.10f) code %d | prod fin %.12f '
              'pole %.10f/%.10f code %d' % ('rest' if k == 0 else 'boost', a['finite_norm'],
              a['single_norm'], a['double_norm'], a['predicted_single_norm'],
              a['predicted_double_norm'], a['ret_code'], b['finite_norm'], b['single_norm'],
              b['double_norm'], b['ret_code']))
    print('  Born decay %.15e  tree %.15e  ratio %.15f' % (rd[0]['born'], tb, rd[0]['born'] / tb))
    return out


def run_poles(name):
    dec = POLES[name]
    dd, dp = build(name, dec)
    masses = olc.driver_masses(dp)
    pts = olc.decay_points(masses[1:], masses[0], count=2, boost=olc.DECAY_BOOST)
    res = olc.run_points(dp, pts, mu=M)
    print('==', name, dec)
    for r in res:
        print('  fin %.10f single %.10f pred %.10f double %.10f pred %.10f code %d' % (
            r['finite_norm'], r['single_norm'], r['predicted_single_norm'], r['double_norm'],
            r['predicted_double_norm'], r['ret_code']))
    return dict(points=[keys(r) for r in res])


if __name__ == '__main__':
    names = sys.argv[1:]
    if names == ['all']:
        names = list(PAIRS) + list(POLES)
    results = {}
    path = os.path.join(HERE, 'octet_results.json')
    if os.path.isfile(path):
        results = json.load(open(path))
    for n in names:
        try:
            results[n] = run_pair(n) if n in PAIRS else run_poles(n)
        except Exception as e:
            print('FAIL', n, repr(e)[:300], flush=True)
        json.dump(results, open(path, 'w'), indent=1)
        sys.stdout.flush()
