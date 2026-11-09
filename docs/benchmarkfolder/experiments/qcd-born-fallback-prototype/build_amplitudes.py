#!/usr/bin/env python3
"""Link private original/prototype drivers; preserve all historical inputs."""
from pathlib import Path
import hashlib
import json
import re
import shlex
import shutil
import subprocess

ROOT = Path.cwd().resolve()
REPORT = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
OUT = REPORT / 'amplitude-native-1'
WORK = ROOT / '.tirust-build/qcd-born-fallback-prototype/amplitude-native-1'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
SP = PROC / 'SubProcesses/P0_uux_zgg'
PHYSICAL = ROOT / 'docs/benchmarkfolder/experiments/physical-stress'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def relative(p):
    return str(p.relative_to(ROOT))


def main():
    OUT.mkdir()
    WORK.mkdir(parents=True)
    preparation = json.loads((REPORT / 'preparation.json').read_text())
    checks = dict(preparation['files_sha256'])
    for name in ('preparation.json', 'build_amplitudes.py', 'amplitude_driver.f'):
        checks[relative(REPORT / name)] = sha(REPORT / name)
    tree = json.loads((REPORT / 'tree-native-2/execution.json').read_text())
    assert tree['complete'] and tree['protected_inputs_unchanged']
    make_text = (SP / 'makefile').read_text()
    process = make_text.split('PROCESS= ', 1)[1].split('\nOLP_PROCESS=', 1)[0]
    objects = [SP / 'MadLoopParamReader.o', SP / 'MadLoopCommons.o']
    for pattern in re.findall(r'\$\(wildcard ([^)]+)\)', process):
        objects.extend(path.with_suffix('.o') for path in sorted(SP.glob(pattern)))
    assert len(objects) == len(set(objects)) and SP / 'loop_matrix.o' in objects
    original_compilation = json.loads((PROC / 'compilation.json').read_text())
    for entries in original_compilation['link_inputs_after'].values():
        for path, expected in entries.items():
            assert sha(Path(path)) == expected, path
            checks[path] = expected
    historical_dso = ROOT / 'HEPTools/tirust/lib/libmadloop_tirust.so'
    assert sha(historical_dso) == '49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
    checks[relative(historical_dso)] = sha(historical_dso)
    # Bind all actual original objects as well as the original library record.
    for path in objects:
        assert path.is_file(), path
        checks[relative(path)] = sha(path)
    checks[relative(PROC / 'compilation.json')] = sha(PROC / 'compilation.json')
    checks[relative(PROC / 'SubProcesses/MadLoop_makefile_definitions')] = sha(PROC / 'SubProcesses/MadLoop_makefile_definitions')
    for name in ('param_card.dat', 'ident_card.dat'):
        checks[relative(PROC / 'Cards' / name)] = sha(PROC / 'Cards' / name)
        shutil.copy2(PROC / 'Cards' / name, WORK / name)
    points = PHYSICAL / 'points/qcd_n3_uux_zgg/selected.dat'
    checks[relative(points)] = sha(points)
    shutil.copy2(points, WORK / 'points.dat')
    for name in ('amplitude_driver.f', 'born_fallback.f', 'loop_matrix.f'):
        shutil.copy2(REPORT / name, WORK / name)
    for path, expected in checks.items():
        assert sha(ROOT / path) == expected, path

    definitions = (PROC / 'SubProcesses/MadLoop_makefile_definitions').read_text()
    libdir = str(PROC / 'lib') + '/'
    links = shlex.split(re.search(r'^LINK_LOOP_LIBS = (.*)$', definitions, re.M)[1].replace('$(LIBDIR)', libdir))
    include = shlex.split(re.search(r'^LOOP_INCLUDE = (.*)$', definitions, re.M)[1])
    fc = shutil.which('gfortran')
    assert fc
    flags = ['-w', '-fPIC', '-ffixed-line-length-132', '-I' + str(PROC / 'Source/MODEL'),
             '-I' + str(PROC / 'Source/DHELAS'), '-I' + str(SP), *include]
    commands = [['taskset', '-c', '11', fc, *flags, '-c', str(WORK / file),
                 '-o', str(WORK / (Path(file).stem + '.o'))]
                for file in ('amplitude_driver.f', 'born_fallback.f', 'loop_matrix.f')]
    for mode in ('original', 'prototype'):
        obj = [WORK / 'amplitude_driver.o']
        obj.extend(WORK / 'loop_matrix.o' if mode == 'prototype' and path == SP / 'loop_matrix.o' else path
                   for path in objects)
        if mode == 'prototype':
            obj.append(WORK / 'born_fallback.o')
        commands.append(['taskset', '-c', '11', fc, *flags, '-o', str(WORK / ('amplitude-' + mode)),
                         *map(str, obj), '-L' + libdir, '-ldhelas', '-lmodel', *links, '-lstdc++'])
    record = {'schema': 1, 'complete': False, 'native_executed': False, 'cpu': 11,
              'commands': commands, 'protected_inputs_before': checks,
              'original_library_record': original_compilation['link_inputs_after'],
              'scope': 'Private generated-host Born fallback, original numerical libraries and immutable objects.'}
    try:
        with (OUT / 'build.log').open('x') as log:
            for cmd in commands:
                subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=120)
        record['artifacts_sha256'] = {relative(path): sha(path) for path in sorted(WORK.iterdir()) if path.is_file()}
        record['complete'] = True
    except BaseException as e:
        record['error'] = {'type': type(e).__name__, 'message': str(e)}
        raise
    finally:
        record['protected_inputs_after'] = {p: sha(ROOT / p) for p in checks}
        record['protected_inputs_unchanged'] = record['protected_inputs_after'] == checks
        (OUT / 'build.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        assert record['protected_inputs_unchanged']
    print('Private original/prototype amplitudes linked; no native calls performed.')


if __name__ == '__main__':
    main()
