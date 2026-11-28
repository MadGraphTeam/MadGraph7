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
(the TIRust build for the P-wave cases), TIRUST_LIB370 (TIRust 370b632, for
ggchic0g and ggchib0g), TIRUST_LIB08 (TIRust 08edee3 with the bridge gate patch, for
jpsig08 and upsg08), TIRUST_LIB56 (TIRust 5025e12, the head of PR #56,
with the bridge patch for the binary64 gate 0006 and the relation tiers, for the cases with the
suffix 56), TIRUST_LIBEP (TIRust d17085a with the bridge patch v2 for the gate, the relation
tiers, the low words and the extended inputs, whose metadata has raised_power_extended_inputs:
the cases with the suffix ep, which run with #PWaveExtendedPrecision .TRUE.), TIRUST_LIB16
(TIRust da7902b with the bridge patch v16: the suffixes 16 and 16ep), NWORKERS (16),
NPOINTS (to override the number of points). The cases with the suffix 6ba run on the configured
TIRust (HEPTools), which must be the pin 6ba4b53; the S-wave cases without a suffix ran on the
pin of their date, 1c130a81."""
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
    # the same with TIRust 08edee3 (PRs #47, #55 and part of #56) and the bridge gate patch 0006
    'jpsig08': ('loop_sm_onia-c_mass', 'g g > Jpsi(1|3S11) g [virt=QCD]', 'TIRUST_LIB08', 10000),
    'etac8g': ('loop_sm_onia-c_mass', 'g g > etac(1|1S08) g [virt=QCD]', None, 10000),
    'chic0g': ('loop_sm_onia-c_mass', 'u u~ > chic0(1|3P01) g [virt=QCD]', 'TIRUST_LIB48958', 2000),
    # the gg channel needs TIRust 370b632 (PR #56; 48958d1 refuses its doubled-gluon pentagons)
    'ggchic0g': ('loop_sm_onia-c_mass', 'g g > chic0(1|3P01) g [virt=QCD]', 'TIRUST_LIB370', 2000),
    # bottomonium, four-flavour scheme (massless charm, m_b = 4.7, m_t = 173)
    'etabg': ('loop_sm_onia-no_tau_mass', 'g g > etab(1|1S01) g [virt=QCD]', None, 10000),
    'upsg': ('loop_sm_onia-no_tau_mass', 'g g > Upsilon(1|3S11) g [virt=QCD]', None, 10000),
    'upsg08': ('loop_sm_onia-no_tau_mass', 'g g > Upsilon(1|3S11) g [virt=QCD]', 'TIRUST_LIB08', 10000),
    'etab8g': ('loop_sm_onia-no_tau_mass', 'g g > etab(1|1S08) g [virt=QCD]', None, 10000),
    'chib0g': ('loop_sm_onia-no_tau_mass', 'u u~ > chib0(1|3P01) g [virt=QCD]', 'TIRUST_LIB48958', 2000),
    'ggchib0g': ('loop_sm_onia-no_tau_mass', 'g g > chib0(1|3P01) g [virt=QCD]', 'TIRUST_LIB370', 2000),
    # P-wave colour octets: the e+e- -> QQbar[n(8)] g processes of the P-wave pole checks
    'cc3p08g': ('loop_sm_onia-c_mass', 'e+ e- > Jpsi(1|3P08) g [virt=QCD]', 'TIRUST_LIB48958', 10000),
    'cc3p18g': ('loop_sm_onia-c_mass', 'e+ e- > Jpsi(1|3P18) g [virt=QCD]', 'TIRUST_LIB48958', 10000),
    'cc3p28g': ('loop_sm_onia-c_mass', 'e+ e- > Jpsi(1|3P28) g [virt=QCD]', 'TIRUST_LIB48958', 10000),
    'cc1p18g': ('loop_sm_onia-c_mass', 'e+ e- > etac(1|1P18) g [virt=QCD]', 'TIRUST_LIB48958', 10000),
    'bb3p08g': ('loop_sm_onia-no_tau_mass', 'e+ e- > Upsilon(1|3P08) g [virt=QCD]', 'TIRUST_LIB48958', 10000),
    'bb3p18g': ('loop_sm_onia-no_tau_mass', 'e+ e- > Upsilon(1|3P18) g [virt=QCD]', 'TIRUST_LIB48958', 10000),
    'bb3p28g': ('loop_sm_onia-no_tau_mass', 'e+ e- > Upsilon(1|3P28) g [virt=QCD]', 'TIRUST_LIB48958', 10000),
    'bb1p18g': ('loop_sm_onia-no_tau_mass', 'e+ e- > etab(1|1P18) g [virt=QCD]', 'TIRUST_LIB48958', 10000),
}
# every P wave again with TIRust 5025e12 (head of PR #56) and the gate + relation-tier bridge patch
PWAVE = ['chic0g', 'ggchic0g', 'chib0g', 'ggchib0g', 'cc3p08g', 'cc3p18g', 'cc3p28g', 'cc1p18g',
         'bb3p08g', 'bb3p18g', 'bb3p28g', 'bb1p18g']
CASES.update({c + '56': CASES[c][:2] + ('TIRUST_LIB56', CASES[c][3]) for c in PWAVE})
# and every S wave (the relation tiers act on raised powers only, the gate on every call)
SWAVE = ['etacg', 'jpsig', 'etac8g', 'etabg', 'upsg', 'etab8g']
CASES.update({c + '56': CASES[c][:2] + ('TIRUST_LIB56', CASES[c][3]) for c in SWAVE})
# every P wave with the 80-bit switch (#PWaveExtendedPrecision, madgraph/loop/onia_ep.py): its
# double-precision passes in extended precision, TIRust with the extended inputs
EXTENDED = [c + 'ep' for c in PWAVE]
CASES.update({c + 'ep': CASES[c][:2] + ('TIRUST_LIBEP', CASES[c][3]) for c in PWAVE})
# every S and P wave on TIRust da7902b (PR #71, on TIRust main) with the bridge patch v16, and every
# P wave there with the 80-bit switch too (v16 has the extended inputs)
CASES.update({c + '16': CASES[c][:2] + ('TIRUST_LIB16', CASES[c][3]) for c in PWAVE + SWAVE})
EXTENDED += [c + '16ep' for c in PWAVE]
CASES.update({c + '16ep': CASES[c][:2] + ('TIRUST_LIB16', CASES[c][3]) for c in PWAVE})
# every S and P wave on the configured TIRust, which must be the pin 6ba4b53 (since 2026-11-27;
# the cases without a suffix and a build ran on the pin before it, 1c130a81)
PIN = '6ba4b53'
CASES.update({c + '6ba': CASES[c][:2] + (None, CASES[c][3]) for c in PWAVE + SWAVE})
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
    elif case.endswith('6ba'):
        config = json.load(open(os.path.join(ROOT, 'HEPTools', 'tirust', 'lib', 'tirust-config.json')))
        revision = config['local_sources']['tirust']['revision']
        if not revision.startswith(PIN):
            raise RuntimeError('the configured TIRust is %s, not the pin %s' % (revision, PIN))
        tirust = 'pinned %s (HEPTools)' % revision[:9]
    outdir = olc.output_process(model, process, os.path.join(work, case), extra=extra,
                                regenerate=not os.path.isdir(os.path.join(work, case)))
    pdir = olc.subprocess_dir(outdir)
    olc.build_driver(pdir)
    libs = card_libs(outdir)
    settings = {'CTModeRun': '-1', 'MLStabThres': '1.0d99', 'NRotations_DP': '1', 'NRotations_QP': '0'}
    if case in EXTENDED:
        settings['PWaveExtendedPrecision'] = '.TRUE.'
    olc.set_reduction(outdir, libs, extra=settings)
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
                settings='CTModeRun -1, MLStabThres 1e99, NRotations_DP 1' +
                (', PWaveExtendedPrecision' if case in EXTENDED else ''), wall=wall)
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
