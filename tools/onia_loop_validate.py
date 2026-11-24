#!/usr/bin/env python3
################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""Validation campaign of the one-loop virtuals of quarkonium processes.

    tools/onia_loop_validate.py --work WORKDIR [--cases NAME ...] [--libs 9 3]
                                [--npoints 10] [--tirust LIB --tirust-label REV]
                                --json results.json

Each case is a process of loop_sm_onia generated through the MadGraph CLI and
evaluated by tools/onia_loop_check.py at deterministic points, compared with
its reference: the finite part c0 and the poles c1, c2 of the virtual in
units of Born * alpha_s/(2 pi) (MadLoop/BLHA normalisation, 't Hooft-Veltman
scheme, (4 pi)^eps/Gamma(1-eps) (mu^2/Q^2)^eps with Q = mu). Literature
values in units of alpha_s/pi are doubled. The references and their
conventions are collected in docs/benchmarkfolder/research/literature.md.
"""

import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'tools'))

import onia_loop_check as olc

PI2 = math.pi ** 2
CF, CA, TF, NL = 4. / 3., 3., 0.5, 3
LOG2 = math.log(2.)


def vp(r, n=20000):
    """int_0^1 x(1-x) ln(1 - x(1-x) r) dx (r < 4): the finite part of a heavy
    quark bubble of mass m in a gluon of virtuality r m^2 (decoupling)."""
    return sum((x * (1 - x)) * math.log(1 - x * (1 - x) * r)
               for x in ((i + 0.5) / n for i in range(n))) / n


def petrelli(bn, a):
    """c0 of a 2 -> 1 Born at mu = M from Petrelli's constant B_n and double
    pole coefficient A (alpha_s/pi units, f_eps normalisation)."""
    return 2 * bn + a * PI2 / 3


BN_1S0_1 = CF * (-5 + PI2 / 4) + CA * (1 + 5 * PI2 / 12)
BN_1S0_8 = CF * (-5 + PI2 / 4) + CA * (2 + 7 * PI2 / 24)
BN_3S1_8 = CF * (-8 + 2 * PI2 / 3) + CA * (50. / 9 + 2. / 3 * LOG2 - PI2 / 4)
MC, MB, MT = 1.55, 4.7, 173.

CASES = {
    'aa_etac': dict(
        process='a a > etac(1|1S01) [virt=QCD]',
        ref=(2 * CF * (PI2 / 4 - 5), 0., 0.),
        source='eta_c -> gamma gamma: C_F (pi^2/4 - 5) alpha_s/pi (Barbieri et al.)'),
    'ee_jpsi': dict(
        process='e+ e- > Jpsi(1|3S11) [virt=QCD]',
        ref=(-32. / 3, 0., 0.),
        source='J/psi -> e+ e-: -16/3 alpha_s/pi (Barbieri et al., Celmaster)'),
    'gg_etac1': dict(
        process='g g > etac(1|1S01) [virt=QCD]',
        ref=(petrelli(BN_1S0_1, -CA), -9., -6.),
        source='Petrelli et al. hep-ph/9707223 eq. (125), mu = M, n_l = 3'),
    'gg_etac8': dict(
        process='g g > etac(1|1S08) [virt=QCD]',
        ref=(petrelli(BN_1S0_8, -CA), -12., -6.),
        source='Petrelli et al. hep-ph/9707223 eq. (128), mu = M, n_l = 3'),
    'uu_psi8': dict(
        process='u u~ > Jpsi(1|3S18) [virt=QCD]',
        ref=(2 * (BN_3S1_8 - 10. / 9 * NL * TF - 16. / 9 * TF) - CF * PI2 / 3
             + 4 * vp((2 * MC / MB) ** 2) + 4 * vp((2 * MC / MT) ** 2), -7., -8. / 3),
        source='Petrelli et al. eq. (133) + bound charm bubble (Hagiwara et al. '
               '0804.1014) + massive b, t bubbles of the model (decoupling)'),
    'uu_ps': dict(
        process='u u~ > Ps(1|3S1) [virt=QCD]', model='loop_sm_onia-lepton_masses',
        ref=(CF * (PI2 - 8), -3 * CF, -2 * CF),
        source='q qbar -> gamma* vertex C_F (pi^2 - 8) (Altarelli, Ellis, Martinelli); '
               'leptonium as a colourless 3S1 final state'),
    'ee_etac_a': dict(
        process='e+ e- > etac(1|1S01) a / z t t~ b b~ [virt=QCD]', relmom=True, onium_legs=[2],
        colour=1, masses={4: 1.5}, energy=10.58,
        ref=(2 * -2.573030, 0., 0.), tolerance=1e-6,
        source='Sang & Chen 0910.4071 C(r) at sqrt(s) = 10.58, m_c = 1.5 '
               '(literature.md C.1), b and t excluded from the loops'),
    'ee_psi_etac': dict(
        process='e+ e- > Jpsi(1|3S11) etac(1|1S01) / z t t~ b b~ QED=2 [virt=QCD]',
        masses={4: 1.5}, energy=10.58, mu=3.0,
        ref=(2 * 12.268210, 0., 0.), tolerance=1e-6,
        source='Gong & Wang 0712.4220 K1, n_l = 3 decoupling scheme, mu = 2 m_c '
               '(literature.md C.4), b and t excluded from the loops'),
    'gg_psi_g': dict(
        process='g g > Jpsi(1|3S11) g [virt=QCD]', energy=20.,
        relmom=True, onium_legs=[2], colour=1,
        ref=None, source='poles: arXiv:2402.19221 eq. (216); TIRust onia_swave'),
    'gg_etac8_g': dict(
        process='g g > etac(1|1S08) g [virt=QCD]', energy=20.,
        relmom=True, onium_legs=[2], colour=8,
        ref=None, source='poles: arXiv:2402.19221 eq. (217)'),
    'gg_psi8_g': dict(
        process='g g > Jpsi(1|3S18) g [virt=QCD]', energy=20.,
        relmom=True, onium_legs=[2], colour=8,
        ref=None, source='poles: arXiv:2402.19221 eq. (217)'),
}


def coulomb_coefficient(case_colour):
    """C of the Coulomb singularity C pi^2/v of c0 (alpha_s/2pi units): C_F
    for a colour singlet, C_F - C_A/2 for an octet."""
    return CF if case_colour == 1 else CF - CA / 2


def relmom_check(name, case, outdir, pdir, masses, v0_point, energy, timeout,
                 libs=('9', '7'), velocities=(0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015, 0.02)):
    """Independent check of the threshold value of the finite part: the
    virtual at a relative momentum of the constituents (ordinary, non
    degenerate integrals) extrapolated to v -> 0 with the Coulomb term
    C pi^2/v subtracted, against the v = 0 result at the same point."""
    onium_legs = case['onium_legs']
    res = dict(velocities=list(velocities), libs={})
    for lib in libs:
        olc.set_reduction(outdir, lib.split('|'), {'CTModeRun': 1})
        scan = olc.relative_momentum_scan(pdir, masses, onium_legs, velocities,
                                          energy=energy, mu=case.get('mu'), timeout=timeout)
        fin = [scan[v][0]['finite_norm'] for v in velocities]
        c = coulomb_coefficient(case['colour'])
        y = [f - c * PI2 / v for f, v in zip(fin, velocities)]
        fit = olc.fit_threshold(list(velocities), y, (0, 1, 2, 3, 4))
        free = olc.fit_threshold(list(velocities), fin, (-1, 0, 1, 2, 3, 4))
        res['libs'][lib] = dict(finite=fin, intercept=fit[0], free_coulomb=free[-1] / PI2,
                                expected_coulomb=c, v0=v0_point,
                                relative_difference=abs(fit[0] - v0_point) / max(abs(v0_point), 1.))
        print('%-12s relmom lib %-3s intercept %.8f  v=0 %.8f  rel %.1e  C(fit) %.5f (exp %.5f)'
              % (name, lib, fit[0], v0_point, res['libs'][lib]['relative_difference'],
                 free[-1] / PI2, c), flush=True)
    olc.set_reduction(outdir, ['9'], {'CTModeRun': -1})
    return res


def run_case(name, case, work, libs, npoints, timeout, tirust=None, tirust_label=None):
    # tirust: another TIRust build (its lib folder) than the pinned one, for
    # candidate revisions; the output is then regenerated with it
    extra = ['set tirust %s --no_save' % tirust] if tirust else []
    outdir = olc.output_process(case.get('model', 'loop_sm_onia-c_mass'), case['process'],
                                os.path.join(work, name), extra=extra, regenerate=bool(tirust))
    pdir = olc.subprocess_dir(outdir)
    if case.get('masses'):
        olc.set_masses(outdir, case['masses'])
    olc.build_driver(pdir)
    masses = olc.driver_masses(pdir)
    final = masses[2:]
    energy = case.get('energy', 1000.) if len(final) > 1 else final[0]
    points = olc.points_lib.generate_points(final, count=npoints, energy=energy,
                                            incoming_masses=tuple(masses[:2]))
    momenta = [p['momenta'] for p in points]
    res = dict(name=name, process=case['process'], source=case['source'],
               reference=case['ref'], masses=masses, energy=energy,
               mu=case.get('mu'), libs={})
    if tirust:
        res['tirust'] = tirust_label or 'unpinned build'
    for lib in libs:
        olc.set_reduction(outdir, lib.split('|'))
        try:
            out = olc.run_points(pdir, momenta, case.get('mu'), timeout=timeout)
        except Exception as error:
            res['libs'][lib] = dict(status='failed: %s' % str(error)[:200])
            print('%-12s lib %-6s failed: %s' % (name, lib, str(error)[:120]), flush=True)
            continue
        entry = dict(status='ok', points=out)
        # the poles against the universal prediction (ONIA_IR_POLES)
        entry['max_pole_deviation'] = max(
            max(abs(r['single_norm'] - r['predicted_single_norm']),
                abs(r['double_norm'] - r['predicted_double_norm'])) for r in out)
        entry['stable_points'] = sum(1 for r in out if r['ret_code'] // 100 == 2)
        if case['ref']:
            fin, single, double = case['ref']
            dev = [max(abs(r['finite_norm'] - fin) / max(abs(fin), 1.),
                       abs(r['single_norm'] - single), abs(r['double_norm'] - double))
                   for r in out]
            entry['max_deviation'] = max(dev)
            entry['passed'] = max(dev) < case.get('tolerance', 1e-8)
        res['libs'][lib] = entry
        summary = entry.get('max_deviation')
        print('%-12s lib %-6s %s  poles vs prediction %.2e  stable %d/%d  codes %s' % (
            name, lib, 'max dev %.2e' % summary if summary is not None else 'no reference',
            entry['max_pole_deviation'], entry['stable_points'], len(out),
            sorted(set(r['ret_code'] for r in out))), flush=True)
    if case.get('relmom') and res['libs'].get('9', {}).get('status') == 'ok':
        res['relmom'] = relmom_check(name, case, outdir, pdir, masses,
                                     res['libs']['9']['points'][0]['finite_norm'],
                                     energy, timeout)
    return res


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--work', required=True)
    parser.add_argument('--cases', nargs='+', default=sorted(CASES))
    parser.add_argument('--libs', nargs='+', default=['9'])
    parser.add_argument('--npoints', type=int, default=10)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--tirust', help='lib folder of a TIRust build to use instead of '
                        'the pinned one (set tirust)')
    parser.add_argument('--tirust-label', help='its revision, recorded in the results')
    parser.add_argument('--json', required=True)
    args = parser.parse_args(argv)
    results = []
    for name in args.cases:
        results.append(run_case(name, CASES[name], args.work, args.libs,
                                args.npoints, args.timeout, args.tirust, args.tirust_label))
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
