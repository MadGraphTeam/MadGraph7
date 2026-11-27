"""P-wave checks of MadLoop's onia layer (scratch): run cases at points and
print the virtual in units of Born * alpha_s/(2 pi) against a reference."""
import json, math, os, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc

SP = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad'
TIRUST = os.environ.get('TIRUST_LIB', SP + '/pw/install/lib')
PI2 = math.pi ** 2
CF = 4. / 3.
CASES = {
    'uu_ps_3p1': dict(process='u u~ > Ps(2|3P1) [virt=QCD]', model='loop_sm_onia-lepton_masses',
                      ref=(CF * (PI2 - 8), -3 * CF, -2 * CF), masses={11: 10.}),
    'uu_ps_3s1h': dict(process='u u~ > Ps(1|3S1) [virt=QCD]', model='loop_sm_onia-lepton_masses',
                      ref=(CF * (PI2 - 8), -3 * CF, -2 * CF), masses={11: 10.}),
    'uu_ps_3s1': dict(process='u u~ > Ps(1|3S1) [virt=QCD]', model='loop_sm_onia-lepton_masses',
                      ref=(CF * (PI2 - 8), -3 * CF, -2 * CF)),
    # Born and virtual vanish: massless q q~ (J_z = +-1) make no J = 0 state
    # through a vector boson
    'uu_ps_3p0': dict(process='u u~ > Ps(2|3P0) [virt=QCD]', model='loop_sm_onia-lepton_masses',
                      ref=None),
    'aa_chic0': dict(process='a a > chic0(1|3P01) [virt=QCD]', ref=None),
    'gg_chic0': dict(process='g g > chic0(1|3P01) [virt=QCD]', ref=None),
    'aa_chic2': dict(process='a a > chic2(1|3P21) [virt=QCD]', ref=None),
    'gg_chic2': dict(process='g g > chic2(1|3P21) [virt=QCD]', ref=None),
    'gg_3p08': dict(process='g g > Jpsi(1|3P08) [virt=QCD]', ref=(11.7597173, -12., -6.)),
    'gg_3p28': dict(process='g g > Jpsi(1|3P28) [virt=QCD]', ref=(15.1829174, -12., -6.)),
    'ag_3p08': dict(process='a g > Jpsi(1|3P08) [virt=QCD]', ref=(3.3575140, -7.5, -3.)),
    'ag_3p28': dict(process='a g > Jpsi(1|3P28) [virt=QCD]', ref=(10.7171815, -7.5, -3.)),
    'gg_1p18': dict(process='g g > etac(1|1P18) [virt=QCD]', ref=None),
    'gg_chic0g': dict(process='g g > chic0(1|3P01) g [virt=QCD]', ref=None),
    # the P -> S pole with a final-state gluon (88: P[8] -> S[8])
    'ee_3p08g': dict(process='e+ e- > Jpsi(1|3P08) g [virt=QCD]', ref=None),
    'ee_3p18g': dict(process='e+ e- > Jpsi(1|3P18) g [virt=QCD]', ref=None),
    'ee_3p28g': dict(process='e+ e- > Jpsi(1|3P28) g [virt=QCD]', ref=None),
    'ee_1p18g': dict(process='e+ e- > etac(1|1P18) g [virt=QCD]', ref=None),
    # the P -> S pole with three coloured partons (18: P[1] -> S[8])
    'uu_chic0g': dict(process='u u~ > chic0(1|3P01) g [virt=QCD]', ref=None),
    'gu_chic0u': dict(process='g u > chic0(1|3P01) u [virt=QCD]', ref=None),
}

def main(names, npoints=4, regenerate=os.environ.get('REGEN', '1') == '1', ctmode=None):
    work = os.environ.get('RUNS', SP + '/pw/runs')
    out = {}
    for name in names:
        case = CASES[name]
        outdir = olc.output_process(case.get('model', 'loop_sm_onia-c_mass'), case['process'],
                                    os.path.join(work, name),
                                    extra=['set tirust %s --no_save' % TIRUST],
                                    regenerate=regenerate)
        pdir = olc.subprocess_dir(outdir)
        if case.get('masses'):
            olc.set_masses(outdir, case['masses'])
        olc.build_driver(pdir)
        masses = olc.driver_masses(pdir)
        final = masses[2:]
        energy = float(os.environ.get('ENERGY', 1000.)) if len(final) > 1 else final[0]
        points = olc.points_lib.generate_points(final, count=npoints, energy=energy,
                                                incoming_masses=tuple(masses[:2]))
        extra = {'CTModeRun': ctmode} if ctmode is not None else None
        olc.set_reduction(outdir, ['9'], extra=extra)
        res = olc.run_points(pdir, [p['momenta'] for p in points], None, timeout=600)
        out[name] = res
        for r in res:
            line = '%-10s born %.10e fin %.12f c1 %.12f c2 %.12f  pred c1 %.12f c2 %.12f code %s' % (
                name, r['born'], r['finite_norm'], r['single_norm'], r['double_norm'],
                r['predicted_single_norm'], r['predicted_double_norm'], r['ret_code'])
            print(line, flush=True)
        if case['ref']:
            f, s, d = case['ref']
            print('%-10s reference   fin %.12f c1 %.12f c2 %.12f' % (name, f, s, d), flush=True)
    json.dump(out, open(os.path.join(work, 'pw_check_%s.json' % '_'.join(names)), 'w'), indent=1)

if __name__ == '__main__':
    args = sys.argv[1:]
    ctmode = None
    if args and args[0].startswith('--ctmode='):
        ctmode = args.pop(0).split('=')[1]
    main(args, ctmode=ctmode)
