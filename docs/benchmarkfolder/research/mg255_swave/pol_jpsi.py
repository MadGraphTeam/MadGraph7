"""MadLoop's g g > Jpsi g per J/psi polarisation against the transverse and
longitudinal virtuals of Jpsi_double_cs_NEW (compare.py gg_jpsig first).

    pol_jpsi.py OUTDIR TIRUST_LIB|- MODE[,MODE...] JSON

generates g g > Jpsi(1|3S11) g in OUTDIR (with the TIRust installation
TIRUST_LIB, or MadGraph's own with -), and evaluates it at the points of the
2.5.5 comparison for every MODE (CTModeRun: -1 with MadLoop's stability
checks, 4 forced quadruple precision), unpolarised and with
ONIA_SELECT_JZ: J_z = +-1 (T) and 0 (L), the helicities in the partonic
centre-of-mass frame of the points. Writes {"<pol>_<mode>": [points]}."""
import json, os, sys
sys.path.insert(0, '/shared/external/madloop_onia/madgraph7_madloop_onia/tools')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import onia_loop_check as olc
import ml_side


def main(outdir, tirust, modes, out_file):
    model, process, masses = ml_side.CASES['gg_jpsig']
    extra = ['set tirust %s --no_save' % tirust] if tirust != '-' else []
    out = olc.output_process(model, process, outdir, extra=extra, regenerate=True)
    pdir = olc.subprocess_dir(out)
    olc.set_masses(out, masses)
    olc.build_driver(pdir)
    pts = json.load(open(os.path.join(ml_side.SP, 'swj', 'pts_gg_jpsig.json')))
    phys = [[p[0], p[1], [a + b for a, b in zip(p[2], p[3])], p[4]] for p in pts]
    res = {}
    for mode in modes:
        olc.set_reduction(out, ['9'], extra={'CTModeRun': mode, 'MLStabThres': '1.0d-3',
                                             'ImprovePSPoint': '2'})
        for name, jz in (('all', None), ('T', (1, [-1, 1])), ('L', (1, [0]))):
            res['%s_%s' % (name, mode)] = olc.run_points(pdir, phys, 11., timeout=3600, jz=jz)
    olc.set_reduction(out, ['9'], extra={'CTModeRun': '-1', 'MLStabThres': '1.0d-3',
                                         'ImprovePSPoint': '2'})
    json.dump(res, open(out_file, 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3].split(','), sys.argv[4])
