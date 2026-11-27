"""e+ e- -> J/psi chi_cJ (J = 0, 1, 2), an S-wave and a P-wave bound state,
against the analytic NLO cross sections of Wang, Ma and Chao
(arXiv:1107.2646; psi_chic_ref.py) and the helicity tables of Dong, Feng and
Jia (arXiv:1107.4351).

    psi_chic.py --work DIR --json OUT.json [--J 0,1,2] [--tirust LIB --tirust-label REV]
                [--energy 10.58] [--mu 3.0] [--modes -1,4] [--exact]

For each J, the process e+ e- > Jpsi(1|3S11) chicJ(1|3PJ1) / z t t~ b b~ QED=2
[virt=QCD] of loop_sm_onia-c_mass (m_c = 1.5 GeV; b and t excluded from the
loops, the limit m_b, m_t -> infinity of the n_l = 3 reference; photon
exchange only, no QED fragmentation) is evaluated at the
four Gauss-Legendre nodes in cos(theta) (azimuth 0.7) in each CTModeRun mode
(-1: MadLoop's stability test at its default threshold 1.0d-3; 4: quadruple
precision). Born and virtual are of the form alpha + beta cos^2(theta) (one
photon, spin 1), so the four nodes integrate them exactly, and the fit
alpha + beta x^2 through them checks that form. Recorded per mode:
- the cross-section average of the finite part, <c0> = int B c0 / int B,
  against psi_chic_ref.madloop_c0 (units of Born * alpha_s/(2 pi), n_l = 3);
- the split into helicity differences lambda = 0 (sin^2 theta) and
  |lambda| = 1 (1 + cos^2 theta): the LO shares against the helicity
  amplitudes of Dong et al. (exact), and the virtual of each part against
  their table 2 (two to three digits);
- the poles against zero and against ONIA_IR_POLES.

--exact replaces the nodes by points whose momenta are exact in binary64:
sqrt(s) = 18.5 GeV, E = 9.25 and |p| = 8.75 GeV for both onia of mass 3 GeV,
and p = 8.75 (sin, 0, cos) with (cos, sin) = (-0.8, 0.6), (-0.6, 0.8), (0, 1),
(0.6, 0.8), (0.8, 0.6). The three values of cos^2 test the form
alpha + beta cos^2(theta), and its fit gives the integrals.
"""
import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', '..', '..', 'tools'))
import onia_loop_check as olc
import psi_chic_ref as ref

NODES = ((-0.8611363115940526, 0.3478548451374538), (-0.3399810435848563, 0.6521451548625461),
         (0.3399810435848563, 0.6521451548625461), (0.8611363115940526, 0.3478548451374538))
PHI = 0.7
EXACT = ((-0.8, 0.6), (-0.6, 0.8), (0., 1.), (0.6, 0.8), (0.8, 0.6))
PROCESS = 'e+ e- > Jpsi(1|3S11) chic%d(1|3P%d1) / z t t~ b b~ QED=2 [virt=QCD]'
MASSES = {4: 1.5}


def momenta(masses, energy, cos):
    m3, m4 = masses[2], masses[3]
    s = energy ** 2
    p = math.sqrt((s - (m3 + m4) ** 2) * (s - (m3 - m4) ** 2)) / (2 * energy)
    e3 = (s + m3 ** 2 - m4 ** 2) / (2 * energy)
    sin = math.sqrt(1 - cos ** 2)
    k = [p * sin * math.cos(PHI), p * sin * math.sin(PHI), p * cos]
    half = energy / 2
    return [[half, 0., 0., half], [half, 0., 0., -half],
            [e3] + k, [energy - e3] + [-x for x in k]]


def exact_momenta(cos, sin):
    p = [[9.25, 0., 0., 9.25], [9.25, 0., 0., -9.25], [9.25, 8.75 * sin, 0., 8.75 * cos],
         [9.25, -8.75 * sin, 0., -8.75 * cos]]
    assert p[2][1] ** 2 + p[2][3] ** 2 == 8.75 ** 2 and p[2][0] ** 2 - 8.75 ** 2 == 9.
    return p


def fit(xs, ys):
    """Least squares y = alpha + beta x^2; returns alpha, beta, the largest
    residual relative to max |y|."""
    u = [x * x for x in xs]
    n, su, suu = len(u), sum(u), sum(t * t for t in u)
    sy, suy = sum(ys), sum(t * y for t, y in zip(u, ys))
    det = n * suu - su * su
    alpha = (suu * sy - su * suy) / det
    beta = (n * suy - su * sy) / det
    res = max(abs(alpha + beta * t - y) for t, y in zip(u, ys)) / max(abs(y) for y in ys)
    return alpha, beta, res


def analyse(J, out, energy, mu, exact=False):
    xs = [x for x, _ in (EXACT if exact else NODES)]
    born = [r['born'] for r in out]
    virt = [r['born'] * r['finite_norm'] for r in out]
    ab, bb, rb = fit(xs, born)
    av, bv, rv = fit(xs, virt)
    if exact:
        c0 = (av + bv / 3) / (ab + bb / 3)
    else:
        c0 = (sum(w * v for (_, w), v in zip(NODES, virt))
              / sum(w * b for (_, w), b in zip(NODES, born)))
    c0_ref = float(ref.madloop_c0(J, energy, 1.5, mu))
    # B = S0 (1 - x^2) + S1 (1 + x^2); integrated: S0 4/3 and S1 8/3
    lo = [(ab - bb) / 2 * 4 / 3, (ab + bb) / 2 * 8 / 3]
    vi = [(av - bv) / 2 * 4 / 3, (av + bv) / 2 * 8 / 3]
    shares = [x / sum(lo) for x in lo]
    r = 4 * 1.5 ** 2 / energy ** 2
    shares_ref = [float(x) for x in ref.dfj_lo_shares(J, r)]
    conv = 2 * float(ref.TF) / 3 * math.log((mu / 1.5) ** 2)
    groups = [vi[i] / lo[i] / 2 - conv if lo[i] else None for i in (0, 1)]
    return dict(
        c0=c0, c0_reference=c0_ref, c0_relative_deviation=abs(c0 - c0_ref) / abs(c0_ref),
        form_residual_born=rb, form_residual_virtual=rv,
        lo_shares=shares, lo_shares_reference=shares_ref,
        lo_share_deviation=max(abs(a - b) for a, b in zip(shares, shares_ref)),
        group_A_nf4=groups, group_A_dfj_table2=ref.dfj_table_groups(J),
        max_pole=max(max(abs(p['single_norm']), abs(p['double_norm'])) for p in out),
        max_pole_deviation=max(max(abs(p['single_norm'] - p['predicted_single_norm']),
                                   abs(p['double_norm'] - p['predicted_double_norm'])) for p in out),
        max_predicted_pole=max(max(abs(p['predicted_single_norm']), abs(p['predicted_double_norm']))
                               for p in out),
        codes=sorted(set(p['ret_code'] for p in out)),
        max_accuracy=max(p['accuracy'] for p in out),
        ms_per_point=1e3 * sum(p['time'] for p in out) / len(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', required=True)
    ap.add_argument('--json', required=True)
    ap.add_argument('--J', default='0,1,2')
    ap.add_argument('--tirust')
    ap.add_argument('--tirust-label')
    ap.add_argument('--energy', type=float, default=10.58)
    ap.add_argument('--mu', type=float, default=3.0)
    ap.add_argument('--modes', default='-1,4')
    ap.add_argument('--exact', action='store_true')
    args = ap.parse_args()
    if args.exact:
        args.energy = 18.5
    extra = ['set tirust %s --no_save' % args.tirust] if args.tirust else []
    results = dict(note=__doc__, energy=args.energy, mu=args.mu, mc=1.5,
                   nodes=EXACT if args.exact else NODES, phi=0. if args.exact else PHI, exact=args.exact,
                   tirust=args.tirust_label or 'pinned', reference=ref.selfcheck(), cases={})
    for J in [int(j) for j in args.J.split(',')]:
        proc = PROCESS % (J, J)
        outdir = olc.output_process('loop_sm_onia-c_mass', proc,
                                    os.path.join(os.path.abspath(args.work), 'psi_chic%d' % J),
                                    extra=extra, regenerate=bool(args.tirust))
        pdir = olc.subprocess_dir(outdir)
        olc.set_masses(outdir, MASSES)
        olc.build_driver(pdir)
        masses = olc.driver_masses(pdir)
        pts = ([exact_momenta(c, sn) for c, sn in EXACT] if args.exact
               else [momenta(masses, args.energy, x) for x, _ in NODES])
        case = dict(process=proc, masses=masses, modes={})
        for mode in args.modes.split(','):
            olc.set_reduction(outdir, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d-3'})
            out = olc.run_points(pdir, pts, args.mu, timeout=10 ** 5)
            a = analyse(J, out, args.energy, args.mu, args.exact)
            a['points'] = out
            case['modes'][mode] = a
            print('J=%d CTModeRun %s: <c0> %.10f ref %.10f (%.1e)  form %.1e/%.1e  LO shares %.1e  '
                  'groups A %s vs DFJ %s  poles %.1e (pred %.1e)  codes %s  acc %.1e  %.0f ms/pt'
                  % (J, mode, a['c0'], a['c0_reference'], a['c0_relative_deviation'],
                     a['form_residual_born'], a['form_residual_virtual'], a['lo_share_deviation'],
                     ['%.3f' % g for g in a['group_A_nf4']], ['%.3f' % g for g in a['group_A_dfj_table2']],
                     a['max_pole'], a['max_predicted_pole'], a['codes'], a['max_accuracy'],
                     a['ms_per_point']), flush=True)
        if '4' in case['modes'] and '5' in case['modes']:
            case['qp_directions_max_difference'] = max(
                abs(p['finite_norm'] - q['finite_norm']) / abs(p['finite_norm'])
                for p, q in zip(case['modes']['4']['points'], case['modes']['5']['points']))
            print('J=%d CTModeRun 4 against 5: %.1e' % (J, case['qp_directions_max_difference']))
        results['cases'][J] = case
    json.dump(results, open(args.json, 'w'), indent=1)
    print('wrote', args.json)


if __name__ == '__main__':
    main()
