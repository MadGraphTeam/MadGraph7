#!/usr/bin/env python3
"""One private build of the exact frozen observer sources; never execute reducers."""
from pathlib import Path
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys

ROOT = Path.cwd().resolve()
HERE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer-native'
DESIGN = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer'
BORN = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
OUT = HERE / 'attempt1'
WORK = ROOT / '.tirust-build/qcd-loop-kinematics-observer/attempt1'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
SP = PROC / 'SubProcesses/P0_uux_zgg'
DESIGN_SHA = '03dcf0b638a32edbd8362fa8cf6d12148250f276f095a53b7579e392aae4ad90'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rel(p):
    return str(p.relative_to(ROOT))


def main():
    assert sha(DESIGN / 'evidence-index.json') == DESIGN_SHA
    assert not OUT.exists() and not WORK.exists(), 'No automatic rebuild or replacement.'
    OUT.mkdir(parents=True)
    WORK.mkdir(parents=True)
    record = {'schema': 1, 'complete': False, 'native_executed': False, 'cpu': 11,
              'scope': 'Exact frozen observer source build; historical numerical libraries and other objects retained.',
              'design_index_sha256': DESIGN_SHA, 'commands': [], 'protected_inputs_before': {}}
    checks = record['protected_inputs_before']
    try:
        verify = subprocess.run([sys.executable, '-B', str(DESIGN / 'verify.py')], cwd=ROOT,
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        (OUT / 'frozen-verifier.log').write_text(verify.stdout)
        assert verify.returncode == 0, 'Frozen source verification failed.'
        born_index = json.loads((BORN / 'evidence-index.json').read_text())
        old_manifest = BORN / 'amplitude-native-1/build.json'
        old_entry = born_index['files']['amplitude-native-1/build.json']
        assert old_manifest.stat().st_size == old_entry['bytes'] and sha(old_manifest) == old_entry['sha256']
        old = json.loads(old_manifest.read_text())
        assert old['complete'] and old['protected_inputs_unchanged']
        checks.update(old['protected_inputs_before'])
        checks.update(old['artifacts_sha256'])
        checks[rel(BORN / 'amplitude-native-1/build.json')] = sha(BORN / 'amplitude-native-1/build.json')
        for name, entry in json.loads((DESIGN / 'evidence-index.json').read_text())['files'].items():
            checks[rel(DESIGN / name)] = entry['sha256']
        checks[rel(DESIGN / 'evidence-index.json')] = DESIGN_SHA
        checks[rel(HERE / 'build.py')] = sha(HERE / 'build.py')
        for p, expected in checks.items():
            assert sha(ROOT / p) == expected, p
        for module in ('mtirust.mod', 'mninjars.mod', 'qcd_kinematics_observer.mod'):
            assert not (ROOT / module).exists(), 'Unexpected current-directory module: ' + module
        for name in ('CT_interface.f', 'TIRUST_interface.f', 'loop_matrix.f', 'amplitude_driver.f'):
            shutil.copyfile(DESIGN / 'sources' / name, WORK / name)
        for name in ('observer.f90', 'observer_context.f'):
            shutil.copyfile(DESIGN / name, WORK / name)
        old_work = ROOT / '.tirust-build/qcd-born-fallback-prototype/amplitude-native-1'
        for name in ('param_card.dat', 'ident_card.dat', 'points.dat'):
            shutil.copyfile(old_work / name, WORK / name)
        make = (SP / 'makefile').read_text()
        process = make.split('PROCESS= ', 1)[1].split('\nOLP_PROCESS=', 1)[0]
        objects = [SP / 'MadLoopParamReader.o', SP / 'MadLoopCommons.o']
        for pattern in re.findall(r'\$\(wildcard ([^)]+)\)', process):
            objects.extend(path.with_suffix('.o') for path in sorted(SP.glob(pattern)))
        assert len(objects) == len(set(objects))
        for p in objects:
            assert rel(p) in checks and sha(p) == checks[rel(p)], p
        definitions = (PROC / 'SubProcesses/MadLoop_makefile_definitions').read_text()
        libdir = str(PROC / 'lib') + '/'
        links = shlex.split(re.search(r'^LINK_LOOP_LIBS = (.*)$', definitions, re.M)[1].replace('$(LIBDIR)', libdir))
        includes = shlex.split(re.search(r'^LOOP_INCLUDE = (.*)$', definitions, re.M)[1])
        fc = shutil.which('gfortran')
        assert fc
        assert all(command[:3] == ['taskset', '-c', '11'] and command[3] == fc
                   for command in old['commands']), 'Compiler differs from the frozen original/prototype build.'
        record['compiler'] = fc
        record['compiler_version'] = subprocess.check_output([fc, '--version'], cwd=ROOT, text=True)
        flags = ['-w', '-fPIC', '-ffixed-line-length-132', '-J' + str(WORK), '-I' + str(WORK),
                 '-I' + str(PROC / 'Source/MODEL'), '-I' + str(PROC / 'Source/DHELAS'), '-I' + str(SP), *includes]
        commands = [['taskset', '-c', '11', fc, *flags, '-c', str(WORK / name),
                     '-o', str(WORK / (Path(name).stem + '.o'))]
                    for name in ('observer.f90', 'observer_context.f', 'CT_interface.f', 'TIRUST_interface.f',
                                 'loop_matrix.f', 'amplitude_driver.f')]
        replaced = {'CT_interface.o', 'TIRUST_interface.o', 'loop_matrix.o'}
        linked = [WORK / 'amplitude_driver.o'] + [WORK / p.name if p.name in replaced else p for p in objects]
        linked += [old_work / 'born_fallback.o', WORK / 'observer.o', WORK / 'observer_context.o']
        old_link = old['commands'][-1]
        first_object = old_link.index('-o') + 2
        first_library = next(i for i in range(first_object, len(old_link)) if old_link[i].startswith('-L'))
        expected_objects = []
        for value in old_link[first_object:first_library]:
            path = Path(value)
            if path.name in replaced or path == old_work / 'amplitude_driver.o':
                path = WORK / path.name
            expected_objects.append(path)
        expected_objects += [WORK / 'observer.o', WORK / 'observer_context.o']
        assert linked == expected_objects, 'Original object order/subset changed beyond declared replacements.'
        library_args = ['-L' + libdir, '-ldhelas', '-lmodel', *links, '-lstdc++']
        assert old_link[first_library:] == library_args, 'Historical library order or linkage flags changed.'
        commands.append(['taskset', '-c', '11', fc, *flags, '-o', str(WORK / 'amplitude-observed'),
                         *map(str, linked), *library_args])
        record['commands'] = commands
        record['linked_objects'] = list(map(rel, linked))
        with (OUT / 'build.log').open('x') as log:
            for command in commands:
                log.write('COMMAND ' + shlex.join(command) + '\n')
                log.flush()
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
        record['artifacts_sha256'] = {rel(p): sha(p) for p in sorted(WORK.iterdir()) if p.is_file()}
        record['linked_objects_sha256'] = {rel(p): sha(p) for p in linked}
        record['complete'] = True
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        record['protected_inputs_after'] = {p: sha(ROOT / p) for p in checks}
        record['protected_inputs_unchanged'] = record['protected_inputs_after'] == checks
        (OUT / 'build.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        assert record['protected_inputs_unchanged']
    print('Private frozen observer compiled and linked; no native reducer execution.')


if __name__ == '__main__':
    main()
