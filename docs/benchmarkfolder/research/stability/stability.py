"""Stability profiles of MadLoop's one-loop virtuals, after appendix C of
arXiv:1604.01363 (Hirschi, Peraro): for N random phase-space points with
p_T > PTMIN for every final-state particle and Delta R > 0.5 between them,
MadLoop's double-precision accuracy estimate. That estimate combines its two
stability tests: the reversed loop direction and one Lorentz rotation of the
point (NRotations_DP = 1). Outputs with bound states add the counterterm
rounding estimate. The quadruple-precision rescue is switched off
(MLStabThres = 1e99) so that the double-precision estimate is what is
recorded. Points MadLoop evaluates without a test (return code 1xx,
initialisation) are dropped.

    stability.py WORKDIR OUTJSONDIR CASE:SETUP [CASE:SETUP ...]

CASE and SETUP are keys of CASES and SETUPS. Environment: TIRUST_LIB48958
(the TIRust build for the P-wave case), TIRUST_LIB08 (TIRust 08edee3 with the
bridge gate patch, for jpsig08), NWORKERS (16), NPOINTS (to override the
number of points)."""
import concurrent.futures
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'tools'))
import onia_loop_check as olc

CASES = {
    # name: (model, process, TIRust build or None, points per setup)
    'ttg': ('loop_sm', 'g g > t t~ g [virt=QCD]', None, 10000),
    'etacg': ('loop_sm_onia-c_mass', 'g g > etac(1|1S01) g [virt=QCD]', None, 10000),
    'jpsig': ('loop_sm_onia-c_mass', 'g g > Jpsi(1|3S11) g [virt=QCD]', None, 10000),
    # the same with TIRust 08edee3 (PRs #47 to #51) and the bridge gate patch 0006
    'jpsig08': ('loop_sm_onia-c_mass', 'g g > Jpsi(1|3S11) g [virt=QCD]', 'TIRUST_LIB08', 10000),
    'etac8g': ('loop_sm_onia-c_mass', 'g g > etac(1|1S08) g [virt=QCD]', None, 10000),
    'chic0g': ('loop_sm_onia-c_mass', 'u u~ > chic0(1|3P01) g [virt=QCD]', 'TIRUST_LIB48958', 2000),
    # bottomonium, four-flavour scheme (massless charm, m_b = 4.7, m_t = 173)
    'etabg': ('loop_sm_onia-no_tau_mass', 'g g > etab(1|1S01) g [virt=QCD]', None, 10000),
    'upsg': ('loop_sm_onia-no_tau_mass', 'g g > Upsilon(1|3S11) g [virt=QCD]', None, 10000),
    'upsg08': ('loop_sm_onia-no_tau_mass', 'g g > Upsilon(1|3S11) g [virt=QCD]', 'TIRUST_LIB08', 10000),
    'etab8g': ('loop_sm_onia-no_tau_mass', 'g g > etab(1|1S08) g [virt=QCD]', None, 10000),
    'chib0g': ('loop_sm_onia-no_tau_mass', 'u u~ > chib0(1|3P01) g [virt=QCD]', 'TIRUST_LIB48958', 2000),
}
SETUPS = {
    # name: (sqrt(s) in GeV, p_T cut in GeV)
    'A': (1000., 50.),   # the setup of arXiv:1604.01363
    'B': (20., 2.),      # a quarkonium-scale setup
}
DR_MIN = 0.5
SEED = 20261122


def pt_eta_phi(p):
    pt = math.hypot(p[1], p[2])
    pz = p[3]
    eta = math.asinh(pz / pt) if pt > 0 else math.copysign(1e9, pz)
    return pt, eta, math.atan2(p[2], p[1])


def passes(momenta, ptmin):
    final = momenta[2:]
    kin = [pt_eta_phi(p) for p in final]
    if any(k[0] < ptmin for k in kin):
        return False
    for i in range(len(kin)):
        for j in range(i + 1, len(kin)):
            dphi = abs(kin[i][2] - kin[j][2])
            dphi = min(dphi, 2 * math.pi - dphi)
            if math.hypot(dphi, kin[i][1] - kin[j][1]) < DR_MIN:
                return False
    return True


def make_points(masses, n, energy, ptmin):
    out, seed = [], SEED
    while len(out) < n:
        batch = olc.points_lib.generate_points(masses[2:], count=max(2 * n, 1000), energy=energy,
                                               seed=seed, incoming_masses=tuple(masses[:2]))
        out.extend(p['momenta'] for p in batch if passes(p['momenta'], ptmin))
        seed += 1
    return out[:n]


def card_libs(outdir):
    text = open(os.path.join(outdir, 'Cards', 'MadLoopParams.dat')).read().split('\n')
    for i, line in enumerate(text):
        if line.strip() == '#MLReductionLib':
            return text[i + 1].strip().split('|')
    raise RuntimeError('no MLReductionLib in the card')


def run_case(work, outjson, case, setup, nworkers):
    model, process, tirust_env, npts = CASES[case]
    npts = int(os.environ.get('NPOINTS', npts))
    energy, ptmin = SETUPS[setup]
    extra = []
    tirust = None
    if tirust_env:
        tirust = os.environ[tirust_env]
        extra = ['set tirust %s --no_save' % tirust]
    outdir = olc.output_process(model, process, os.path.join(work, case), extra=extra,
                                regenerate=not os.path.isdir(os.path.join(work, case)))
    pdir = olc.subprocess_dir(outdir)
    olc.build_driver(pdir)
    libs = card_libs(outdir)
    olc.set_reduction(outdir, libs, extra={'CTModeRun': '-1', 'MLStabThres': '1.0d99',
                                           'NRotations_DP': '1', 'NRotations_QP': '0'})
    masses = olc.driver_masses(pdir)
    points = make_points(masses, npts, energy, ptmin)
    # initialisation (filters, caches) before the parallel chunks
    olc.run_points(pdir, points[:1], None, warmup=10, timeout=3600)
    chunks = [points[i::nworkers] for i in range(nworkers)]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(nworkers) as pool:
        results = list(pool.map(lambda c: olc.run_points(pdir, c, None, warmup=3, timeout=36000)
                                if c else [], chunks))
    wall = time.time() - t0
    rows = []
    for i in range(npts):
        r = results[i % nworkers][i // nworkers]
        rows.append(dict(accuracy=r['accuracy'], ret_code=r['ret_code'], finite_norm=r['finite_norm'],
                         single_norm=r['single_norm'], time=r['time']))
    meta = dict(case=case, setup=setup, model=model, process=process, libs='|'.join(libs),
                tirust=tirust or 'pinned', energy=energy, ptmin=ptmin, dr_min=DR_MIN, npoints=npts,
                settings='CTModeRun -1, MLStabThres 1e99, NRotations_DP 1', wall=wall)
    # compact: per point the accuracy estimate, return code and time (3 digits)
    json.dump(dict(meta=meta, accuracy=[float('%.3g' % r['accuracy']) for r in rows],
                   ret_code=[r['ret_code'] for r in rows],
                   time=[float('%.3g' % r['time']) for r in rows]),
              open(os.path.join(outjson, '%s_%s.json' % (case, setup)), 'w'), separators=(',', ':'))
    tested = [r for r in rows if r['ret_code'] // 100 != 1]
    above = sum(1 for r in tested if r['accuracy'] >= 1e-3)
    print('%-7s %s: %d points (%d tested), %.2f%% at or above 1e-3, wall %.0f s' % (
        case, setup, npts, len(tested), 100. * above / max(len(tested), 1), wall), flush=True)


if __name__ == '__main__':
    work, outjson = sys.argv[1], sys.argv[2]
    os.makedirs(outjson, exist_ok=True)
    nworkers = int(os.environ.get('NWORKERS', 16))
    for item in sys.argv[3:]:
        case, setup = item.split(':')
        run_case(work, outjson, case, setup, nworkers)
