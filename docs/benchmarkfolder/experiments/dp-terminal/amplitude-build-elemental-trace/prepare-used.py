#!/usr/bin/env python3
"""Build isolated DP terminal variants using the exact frozen compiler recipes."""
import argparse
import difflib
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
PROTOCOL = HERE.parent / 'n1-profile/fortran'
sys.path[:0] = [str(PROTOCOL), str(PROTOCOL / 'protocol')]
from prepare import clone
from tirust_benchmark_compiler import _compile_command, COMPILER_ENVIRONMENT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected one source anchor: ' + repr(old))
    return text.replace(old, new)


def fixed_form(text):
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith('!'):
            lines.append('C ' + line.lstrip()[1:])
        elif line.strip():
            label = re.match(r'^(\d+)\s+(.*)$', line)
            prefix = '      '
            if label:
                prefix = label[1].rjust(5) + ' '
                line = label[2]
            lines.append(prefix + line[:66])
            line = line[66:]
            while line:
                lines.append('     $' + line[:66])
                line = line[66:]
    return '\n'.join(lines) + '\n'


def transform(caller, polynomial, terminal):
    caller = once(caller, '      USE ALOHA_OBJECT\n',
                  '      USE ALOHA_OBJECT\n'
                  '      USE, INTRINSIC :: IEEE_EXCEPTIONS, ONLY: IEEE_GET_FLAG,\n'
                  '     $ IEEE_INEXACT\n')
    caller = once(caller, '      INTEGER I,J,K\n',
                  '      INTEGER I,J,K\n'
                  '      LOGICAL DP_TERMINAL_ENTRY_INEXACT,DP_TERMINAL_FAST\n')
    call = '      CALL ML5_0_UPDATE_WL_1_1(WL(1,0,1,2),4,COEFS,4,4,WL(1,0,1,3))\n'
    caller = once(caller, call,
                  '      CALL IEEE_GET_FLAG(IEEE_INEXACT,DP_TERMINAL_ENTRY_INEXACT)\n'
                  '      CALL ML5_0_DP_TERMINAL_1_1(WL(1,0,1,2),4,COEFS,4,4,\n'
                  '     $ WL(1,0,1,3),DP_TERMINAL_ENTRY_INEXACT,DP_TERMINAL_FAST)\n')
    terminal = once(terminal, 'subroutine DP_TERMINAL(', 'subroutine ML5_0_DP_TERMINAL_1_1(')
    terminal = once(terminal, 'call DP_DENSE(', 'call ML5_0_UPDATE_WL_1_1(')
    return caller, polynomial + '\n' + fixed_form(terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True, choices=('elemental', 'packed'))
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--trace-admission', action='store_true',
                        help='Separate diagnostic executable, excluded from timing comparisons')
    args = parser.parse_args()
    source = ROOT / '.benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z'
    frozen_path = ROOT / 'docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3/candidate-inputs.json'
    frozen = json.loads(frozen_path.read_text())
    original_process = source / 'SubProcesses/P0_uux_z'
    for name, expected in frozen['compilation']['sources'].items():
        if digest(source / name) != expected:
            raise RuntimeError('Frozen generated source changed: ' + name)
    if digest(original_process / 'check') != frozen['compilation']['executable_sha256']:
        raise RuntimeError('Frozen executable changed')
    suffix = args.variant + ('-trace' if args.trace_admission else '')
    report = HERE / ('amplitude-build-' + suffix)
    destination = ROOT / '.tirust-build/dp-terminal' / ('amplitude-' + suffix)
    report.mkdir(exist_ok=False)
    shutil.copy2(__file__, report / 'prepare-used.py')
    clone(source, destination)
    process = destination / 'SubProcesses/P0_uux_z'
    make = ['make', '--no-print-directory', '--dry-run', '--always-make',
            'SHELL=' + shutil.which('bash'), 'FC=' + os.environ['FC'], 'check']
    dry = subprocess.check_output(make, cwd=process, text=True, stderr=subprocess.STDOUT)
    (report / 'original-expanded-recipes.log').write_text(dry)
    commands = [shlex.split(line) for line in dry.splitlines() if line.strip()]
    names = ('polynomial.f', 'coef_construction_1.f')
    compiles = {}
    for name in names:
        selected = [row for row in commands if '-c' in row and name in row]
        if len(selected) != 1:
            raise RuntimeError('Expected one compile recipe for ' + name)
        compiles[name] = selected[0]
        parsed = _compile_command(shlex.join(selected[0]), process, dict(os.environ), {})
        expected = frozen['compilation']['effective_compiler_flags']['scopes']['process']['recipes'][name]
        if parsed != (name, expected):
            raise RuntimeError('Compiler recipe changed for ' + name)
    links = [row for row in commands if '-o' in row and row[row.index('-o') + 1] == 'check']
    if len(links) != 1:
        raise RuntimeError('Expected one final link recipe')
    injected = set(COMPILER_ENVIRONMENT) | {k for k in os.environ if k.startswith(('NIX_CFLAGS_', 'NIX_LDFLAGS_'))}
    environment = {k: os.environ[k] for k in sorted(injected) if k in os.environ}
    if environment != frozen['compilation']['effective_compiler_flags']['environment']:
        raise RuntimeError('Injected compiler environment changed')
    archives = {}
    for name, expected in frozen['runtime']['archives'].items():
        path = Path(name)
        if path.is_relative_to(source):
            path = destination / path.relative_to(source)
        if digest(path) != expected:
            raise RuntimeError('Static link input changed: ' + str(path))
        archives[str(path)] = expected
    terminal_path = HERE / ('packed-scan/terminal.f90' if args.variant == 'packed' else 'terminal.f90')
    original = {name: (process / name).read_text() for name in names}
    caller, polynomial = transform(original['coef_construction_1.f'], original['polynomial.f'], terminal_path.read_text())
    if args.trace_admission:
        caller = once(caller,
                      '      CALL ML5_0_CREATE_LOOP_COEFS(WL(1,0,1,3),2,4,1,1,1,2,H)\n',
                      "      WRITE(*,*) 'DP_TERMINAL_ADMISSION',\n"
                      '     $ DP_TERMINAL_ENTRY_INEXACT,DP_TERMINAL_FAST\n'
                      '      CALL ML5_0_CREATE_LOOP_COEFS(WL(1,0,1,3),2,4,1,1,1,2,H)\n')
    transformed = {'coef_construction_1.f': caller, 'polynomial.f': polynomial}
    for name in names:
        (report / ('original-' + name)).write_text(original[name])
        (report / ('terminal-' + name)).write_text(transformed[name])
        (report / (name + '.diff')).write_text(''.join(difflib.unified_diff(
            original[name].splitlines(True), transformed[name].splitlines(True),
            fromfile='original/' + name, tofile=args.variant + '/' + name)))
    before = {p.name: digest(p) for p in sorted(process.glob('*.o'))}
    manifest = {
        'performance_eligible': False, 'variant': args.variant,
        'trace_admission': args.trace_admission,
        'scope': 'Private final DP wavefunction updater, unchanged polynomial -O3 and caller flags',
        'output': str(destination), 'source': str(source),
        'prepare_sha256': digest(__file__), 'terminal_source_sha256': digest(terminal_path),
        'frozen_input_sha256': digest(frozen_path),
        'original_executable_sha256': digest(original_process / 'check'),
        'original_source_sha256': {name: digest(process / name) for name in names},
        'source_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in transformed.items()},
        'compile_commands': compiles, 'link_command': links[0],
        'compiler_environment': environment, 'static_archive_pins': archives,
        'objects_before': before, 'built': False,
    }
    def save():
        (report / 'build-metadata.json').write_text(json.dumps(manifest, indent=2) + '\n')
    save()
    if not args.build:
        print('Prepared only; no compiler or native execution.')
        return
    for name in ('polynomial.o', 'coef_construction_1.o', 'check'):
        if not (process / name).resolve().is_relative_to(destination):
            raise RuntimeError('Build output escapes isolated clone: ' + name)
    with (report / 'compile-link.log').open('w') as log:
        log.write('Untouched reconstruction:\n' + shlex.join(links[0]) + '\n'); log.flush()
        subprocess.run(links[0], cwd=process, stdout=log, stderr=subprocess.STDOUT, check=True)
        manifest['reconstructed_original_executable_sha256'] = digest(process / 'check')
        save()
        if manifest['reconstructed_original_executable_sha256'] != manifest['original_executable_sha256']:
            raise RuntimeError('Untouched relink failed exact executable reconstruction')
        for name, text in transformed.items():
            (process / name).write_text(text)
        for command in [compiles[name] for name in names] + links:
            log.write(shlex.join(command) + '\n'); log.flush()
            subprocess.run(command, cwd=process, stdout=log, stderr=subprocess.STDOUT, check=True)
    if any(digest(path) != expected for path, expected in archives.items()):
        raise RuntimeError('Static archive changed during compilation')
    after = {p.name: digest(p) for p in sorted(process.glob('*.o'))}
    changed = [name for name in before if before[name] != after[name]]
    if changed != ['coef_construction_1.o', 'polynomial.o']:
        raise RuntimeError('Unexpected changed objects: ' + repr(changed))
    manifest.update(built=True, objects_after=after, changed_objects=changed,
                    executable_sha256=digest(process / 'check'), driver_sha256=digest(process / 'check_sa.f'))
    save()
    print(json.dumps({'variant': args.variant, 'changed_objects': changed,
                      'executable_sha256': manifest['executable_sha256']}, indent=2))


if __name__ == '__main__':
    main()
