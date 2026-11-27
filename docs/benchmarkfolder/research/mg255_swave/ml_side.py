"""MadLoop side of the comparison with the MG5 2.5.5 onia virtuals:
generate (once) and evaluate an onia output at points read from a JSON
file [[p1, p2, p3, p4, p5], ...] (constituent momenta, as the reference
takes them); prints born, finite/1eps/2eps normalised to Born*as/2pi.

The heavy quarks the reference decouples are excluded from the loops
('/ t t~ b b~' for charmonium, '/ t t~' for bottomonium), the exact
m -> infinity limit. CASES_1E4 is the earlier setup with them at 1e4 GeV,
whose counterterm rounding is M^2-enhanced at soft jets (it is kept for
pol_jpsi.py and the record of Table 5)."""
import json, os, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
import onia_loop_check as olc
SP = '/tmp/claude-1134/-shared-external-madloop-onia/6136fd6c-8e41-47c7-a479-0995fb3044a9/scratchpad'
TIRUST = os.environ.get('TIRUST_LIB')
CASES_1E4 = {
    'gg_etacg': ('loop_sm_onia-c_mass', 'g g > etac(1|1S01) g [virt=QCD]', {5: 1e4, 6: 1e4}),
    'gu_etacu': ('loop_sm_onia-c_mass', 'g u > etac(1|1S01) u [virt=QCD]', {5: 1e4, 6: 1e4}),
    'ug_etacu': ('loop_sm_onia-c_mass', 'u g > etac(1|1S01) u [virt=QCD]', {5: 1e4, 6: 1e4}),
    'uux_etacg': ('loop_sm_onia-c_mass', 'u u~ > etac(1|1S01) g [virt=QCD]', {5: 1e4, 6: 1e4}),
    'gg_jpsig': ('loop_sm_onia-c_mass', 'g g > Jpsi(1|3S11) g [virt=QCD]', {5: 1e4, 6: 1e4}),
    'gg_etabg': ('loop_sm_onia-no_tau_mass', 'g g > etab(1|1S01) g [virt=QCD]', {5: 4.75, 6: 1e4}),
    'gu_etabu': ('loop_sm_onia-no_tau_mass', 'g u > etab(1|1S01) u [virt=QCD]', {5: 4.75, 6: 1e4}),
    'ug_etabu': ('loop_sm_onia-no_tau_mass', 'u g > etab(1|1S01) u [virt=QCD]', {5: 4.75, 6: 1e4}),
    'uux_etabg': ('loop_sm_onia-no_tau_mass', 'u u~ > etab(1|1S01) g [virt=QCD]', {5: 4.75, 6: 1e4}),
    'gg_upsg': ('loop_sm_onia-no_tau_mass', 'g g > Upsilon(1|3S11) g [virt=QCD]', {5: 4.75, 6: 1e4}),
}


def excluded(case):
    model, process, masses = case
    heavy = '/ t t~ b b~' if masses[5] > 1e3 else '/ t t~'
    return (model, process.replace(' [virt=QCD]', ' %s [virt=QCD]' % heavy),
            dict((k, v) for k, v in masses.items() if v < 1e3))


CASES = dict((name, excluded(case)) for name, case in CASES_1E4.items())

def main(name, points_file, mu, out_file):
    model, process, masses = CASES[name]
    extra = ['set tirust %s --no_save' % TIRUST] if TIRUST else []
    outdir = olc.output_process(model, process, os.path.join(SP, 'swj', 'ml', name), extra=extra,
                                regenerate=os.environ.get('REGEN', '0') == '1')
    pdir = olc.subprocess_dir(outdir)
    if masses:
        olc.set_masses(outdir, masses)
    olc.build_driver(pdir)
    olc.set_reduction(outdir, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3',
                                            'ImprovePSPoint': '2'})
    pts = json.load(open(points_file))
    phys = [[p[0], p[1], [a + b for a, b in zip(p[2], p[3])], p[4]] for p in pts]
    res = olc.run_points(pdir, phys, mu, timeout=3600)
    json.dump(res, open(out_file, 'w'), indent=1)
    for r in res:
        print('ML born %.12e fin %.12f c1 %.12f c2 %.12f pred %.12f %.12f code %s' % (
            r['born'], r['finite_norm'], r['single_norm'], r['double_norm'],
            r['predicted_single_norm'], r['predicted_double_norm'], r['ret_code']))

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4])
