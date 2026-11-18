#!/usr/bin/env python3
"""Focused real-CLI bare-loop UV/IR checks against MadLoop's COLLIER wrapper."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parents[3]
HERE = None
NATIVE_PATHS = []
sys.path.insert(0, str(ROOT))
from madgraph.various.banner import MadLoopParam

POINTS = ASSETS / 'points'
MODEL = ROOT / 'models/loop_qcd_qed_sm_Gmu'
CASES = {
    'qcd-z': ('loop_sm', 'u u~ > z [virt=QCD]', 'qcd-1', 3),
    'qcd-tt': ('loop_sm', 'g g > t t~ [virt=QCD]', 'qcd-2', 4),
    'ew-mumu': (str(MODEL), 'e+ e- > mu+ mu- [virt=QED]', 'ew-2', 4),
    'li-h': ('loop_sm', 'g g > h [sqrvirt=QCD]', 'control-gg-h', 3),
}


def run(command, cwd, log):
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    env['MAKEFLAGS'] = 'SHELL=' + shutil.which('bash')
    generated_lib = cwd.parents[1] / 'lib'
    linked_dirs = [p.resolve() for p in generated_lib.iterdir() if p.is_dir()] if generated_lib.is_dir() else []
    env['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in NATIVE_PATHS + [generated_lib] + linked_dirs) + ':' + env.get('LD_LIBRARY_PATH', '')
    with log.open('w') as stream:
        subprocess.run(command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(name, ninjars):
    model, process, point_name, nexternal = CASES[name]
    work = HERE / 'processes' / name
    work.mkdir(parents=True, exist_ok=True)
    output = work / 'output'
    command_card = work / 'generate.mg5'
    commands = ['set automatic_html_opening False --no_save',
                'set notification_center False --no_save',
                'set loop_optimized_output True --no_save']
    if ninjars:
        commands.append(f'set ninjars {ninjars} --no_save')
    if COLLIER is not None:
        commands.append(f'set collier {COLLIER} --no_save')
    commands += [f'import model {model}', f'generate {process}', f'output standalone {output} -f', 'quit']
    command_card.write_text('\n'.join(commands) + '\n')
    if not output.exists():
        run([sys.executable, '-B', 'bin/madgraph', str(command_card)], ROOT, work / 'generation.log')
    subs = sorted((output / 'SubProcesses').glob('P*_*'))
    assert len(subs) == 1, subs
    sub = subs[0]
    loop = sub / 'loop_matrix.f'
    prefix = re.search(r'SUBROUTINE\s+(\w+)SLOOPMATRIX\(', loop.read_text(), re.I).group(1)
    metadata = {'process': process, 'model': model, 'prefix': prefix, 'generated_bare_patch': []}
    # Remove all UFO counterterms only in this disposable generated validation copy.
    # The loop numerator, reduction wrappers and production source remain intact.
    for filename, marker, extra in [
            ('loop_matrix.f', '          UVCT_REQ_SO_DONE=.TRUE.', ''),
            ('mp_compute_loop_coefs.f', '          MP_UVCT_REQ_SO_DONE=.TRUE.',
             '\n          DP_AMPL(:,1:NCTAMPS)=(0.0D0,0.0D0)')]:
        path = sub / filename
        text = path.read_text()
        if 'UVIR_VALIDATION_BARE_LOOP' not in text:
            matches = list(re.finditer(r'^[ \t]*' + re.escape(marker.strip()) + r'\s*$', text, re.M))
            assert len(matches) == 1, (path, len(matches))
            match = matches[0]
            replacement = match.group(0) + '\nC         UVIR_VALIDATION_BARE_LOOP: discard UFO UV/R2 counterterms.\n          AMPL(:,1:NCTAMPS)=0' + extra
            original = path.with_suffix('.f.original')
            original.write_text(text)
            text = text[:match.start()] + replacement + text[match.end():]
            path.write_text(text)
        metadata['generated_bare_patch'].append({'path': filename, 'original_sha256': sha(path.with_suffix('.f.original')), 'bare_sha256': sha(path)})
    # Use the fixed point stream from the previous benchmark and exactly zero widths.
    card = output / 'Cards/param_card.dat'
    if not (work / 'param_card.original.dat').exists():
        shutil.copy2(card, work / 'param_card.original.dat')
    text = (ASSETS / 'cards' / (name + '.dat')).read_text()
    card.write_text(text)
    shutil.copy2(card, sub / 'param_card.dat')
    stream = (POINTS / (point_name + '.dat')).read_text().splitlines()
    (work / 'points.dat').write_text('3\n' + '\n'.join(stream[1:1 + 3*nexternal]) + '\n')
    driver = (ASSETS / 'bare_driver.f').read_text().replace('@NEXTERNAL@', str(nexternal)).replace('@NBORN@', '0' if name.startswith('li-') else '1').replace('@PREFIX@', prefix)
    (sub / 'check_sa.f').write_text(driver)
    run(['make', '-j2', 'check'], sub, work / 'build.log')
    metadata['point_sha256'] = sha(work / 'points.dat')
    metadata['parameter_card_sha256'] = sha(card)
    (work / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return work, output, sub


def evaluate(name, ninjars, backend, mode, ctmode, switch=False, filters=False, reuse_filters=False):
    work = HERE / 'processes' / name
    output = work / 'output'
    sub = sorted((output / 'SubProcesses').glob('P*_*'))[0]
    target = work / (('filtered-' if filters else '') + f'backend-{backend}-mode-{mode}-ct-{ctmode}')
    target.mkdir(exist_ok=True)
    card = MadLoopParam(str(output / 'Cards/MadLoopParams.dat'))
    values = dict(MLReductionLib=str(backend), CTModeRun=ctmode,
                  CTModeInit=1, UseLoopFilter=filters, LoopPoleMode=mode,
                  COLLIERComputeUVpoles=True, COLLIERComputeIRpoles=True,
                  UseQPIntegrandForNinjaRS=True)
    for key, value in values.items():
        card[key] = value
    template = str(ROOT / 'Template/loop_material/StandAlone/Cards/MadLoopParams.dat')
    for path in [output / 'Cards/MadLoopParams.dat', *(output / 'SubProcesses').rglob('MadLoopParams.dat')]:
        card.write(str(path), template=template)
    if not reuse_filters:
        for pattern in ['*HelFilter.dat', '*LoopFilter.dat']:
            for path in (output / 'SubProcesses').rglob(pattern):
                path.unlink()
    shutil.copy2(output / 'Cards/MadLoopParams.dat', target / 'MadLoopParams.dat')
    command = [str(sub / 'check'), str(work / 'points.dat'), '1']
    if switch:
        command.append('--switch')
    run(command, sub, target / 'evaluation.log')
    for filename in ['values.dat', 'dimensions.dat']:
        shutil.copy2(sub / filename, target / filename)
    if switch:
        shutil.copy2(sub / 'switch_values.dat', target / 'switch_values.dat')
    records = []
    for line in (target / 'values.dat').read_text().splitlines():
        fields = line.split()
        records.append(dict(point=int(fields[0]), split_order=int(fields[1]), status=int(fields[2]),
                            born=float(fields[3]), finite=float(fields[4]), single=float(fields[5]),
                            double=float(fields[6]), accuracy=float(fields[7])))
    (target / 'result.json').write_text(json.dumps(dict(case=name, backend=backend, mode=mode, ctmode=ctmode,
                                                     filters=filters, reused_filters=reuse_filters, records=records), indent=2) + '\n')
    print(name, backend, mode, ctmode, 'OK', flush=True)


def main():
    global HERE, POINTS, COLLIER, NATIVE_PATHS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ew-model', type=Path, default=MODEL)
    parser.add_argument('--points', type=Path, default=POINTS)
    parser.add_argument('--collier', type=Path)
    parser.add_argument('--campaign', action='store_true')
    parser.add_argument('--case', choices=CASES, action='append')
    parser.add_argument('--ninjars')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--backend', type=int, default=8)
    parser.add_argument('--mode', type=int, action='append')
    parser.add_argument('--ctmode', type=int, default=1)
    parser.add_argument('--switch', action='store_true')
    parser.add_argument('--filters', action='store_true')
    parser.add_argument('--reuse-filters', action='store_true')
    args = parser.parse_args()
    HERE, POINTS = args.output.resolve(), args.points.resolve()
    if HERE == ASSETS or ASSETS in HERE.parents:
        parser.error('--output must be outside the retained source directory')
    HERE.mkdir(parents=True, exist_ok=True)
    COLLIER = args.collier.resolve() if args.collier else None
    NATIVE_PATHS = [COLLIER, COLLIER / 'lib'] if COLLIER else []
    if args.ninjars:
        args.ninjars = str(Path(args.ninjars).resolve())
    CASES['ew-mumu'] = (str(args.ew_model.resolve()), *CASES['ew-mumu'][1:])
    if (args.campaign or args.prepare) and ('ew-mumu' in (args.case or CASES)) and not args.ew_model.is_dir():
        parser.error('EW source is absent: supply --ew-model with loop_qcd_qed_sm_Gmu')
    if args.campaign:
        if args.case:
            parser.error('--campaign runs the complete four-process matrix; omit --case')
        for name in CASES:
            prepare(name, args.ninjars)
        for name in CASES:
            for backend in (7, 8):
                for mode in range(4):
                    evaluate(name, args.ninjars, backend, mode, 1)
            for ctmode in ([-1, 2, 4, 5] if name.startswith('qcd-') else [4]):
                for mode in range(4):
                    evaluate(name, args.ninjars, 8, mode, ctmode)
        for backend in (7, 8):
            evaluate('qcd-tt', args.ninjars, backend, 0, 1, switch=True)
        evaluate('qcd-tt', args.ninjars, 8, -1, 1, switch=True, filters=True)
        evaluate('qcd-tt', args.ninjars, 8, 1, 1, filters=True, reuse_filters=True)
        subprocess.run([sys.executable, '-B', str(ASSETS / 'compare_madloop.py'), '--output', str(HERE)], check=True)
        return
    for name in args.case or CASES:
        if args.prepare:
            prepare(name, args.ninjars)
        for mode in args.mode or ([] if args.prepare else range(4)):
            evaluate(name, args.ninjars, args.backend, mode, args.ctmode, args.switch, args.filters, args.reuse_filters)


if __name__ == '__main__':
    main()
