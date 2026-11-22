"""The size of the P -> S term of MadLoop's pole prediction (ONIA_PW_SOFT_POLE)
at the points of pw_check.py: a print of PSPOLE/BORN2 is added to the
onia_loop.f of an existing pw_check output, the points driver is rebuilt,
and the term is written in units of Born * alpha_s/(2 pi).

    python3 ps_term.py --json OUT.json OUTDIR:ENERGY [OUTDIR:ENERGY ...]

ENERGY is the centre-of-mass energy of pw_check (ignored for 2 -> 1).
The patch only prints; it changes no result."""
import argparse, json, os, subprocess, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..', '..', '..', 'tools'))
import onia_loop_check as olc

MARK = "      WRITE(*,*) 'PSDEBUG',PSPOLE/BORN2\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', required=True)
    parser.add_argument('--npoints', type=int, default=4)
    parser.add_argument('outputs', nargs='+')
    args = parser.parse_args()
    res = {}
    for item in args.outputs:
        outdir, energy = item.rsplit(':', 1)
        pdir = olc.subprocess_dir(outdir)
        path = os.path.join(pdir, 'onia_loop.f')
        src = open(path).read()
        if MARK not in src:
            anchor = "      POLES(1)=POLES(1)+PSPOLE\n"
            if anchor not in src:
                raise RuntimeError('%s: no P -> S term in ONIA_IR_POLES' % path)
            open(path, 'w').write(src.replace(anchor, anchor + MARK))
        olc.build_driver(pdir)
        masses = olc.driver_masses(pdir)
        final = masses[2:]
        en = float(energy) if len(final) > 1 else final[0]
        pts = olc.points_lib.generate_points(final, count=args.npoints, energy=en,
                                             incoming_masses=tuple(masses[:2]))
        lines = [str(len(pts))]
        for p in pts:
            lines.append('-1')
            lines.extend(' '.join('%.17e' % c for c in v) for v in p['momenta'])
        out = subprocess.run([os.path.join(pdir, 'ml_points_driver')], cwd=pdir,
                             input='\n'.join(lines) + '\n', capture_output=True,
                             text=True, check=True).stdout
        terms = [float(l.split()[1].replace('D', 'E')) for l in out.splitlines()
                 if 'PSDEBUG' in l]
        name = os.path.basename(os.path.normpath(outdir))
        res[name] = dict(energy=en, ps_term=terms)
        print(name, ' '.join('%.10g' % t for t in terms), flush=True)
    json.dump(res, open(args.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
