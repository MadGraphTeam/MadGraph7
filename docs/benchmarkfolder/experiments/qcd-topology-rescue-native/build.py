#!/usr/bin/env python3
"""Build one frozen private arm with the historical compiler and link order."""
from pathlib import Path
import argparse
import json
import os
import shlex
import shutil
import subprocess
import time

from common import (ROOT, HERE, DESIGN, OLD, WORKBASE, OUTBASE, ARMS,
                    sha, rel, require, pins, controller_pins, validate_pins,
                    frozen_verify, run_capped)

SOURCE_ORDER = ('observer.f90', 'observer_context.f', 'CT_interface.f',
                'TIRUST_interface.f', 'loop_matrix.f', 'amplitude_driver.f', 'born_fallback.f')


def plan(arm):
    require(arm in ARMS, 'Unknown/nonselected arm.')
    arm_info = json.loads((DESIGN / 'arms.json').read_text())['arms'][arm]
    old = json.loads((OLD / 'attempt1/build.json').read_text())
    require(old['complete'] and old['protected_inputs_unchanged'], 'Historical build incomplete.')
    work = WORKBASE / arm
    old_work = ROOT / '.tirust-build/qcd-loop-kinematics-observer/attempt1'
    prefix = old['commands'][0][:old['commands'][0].index('-c', 4)]
    # Only the isolated module/output directory changes in the historical flags.
    prefix = [value.replace(str(old_work), str(work)) for value in prefix]
    require(prefix[:3] == ['taskset', '-c', '11'], 'Historical CPU contract changed.')
    fc = prefix[3]
    require(shutil.which('gfortran') == fc == old['compiler'], 'Compiler differs from historical build.')
    old_prefix = old['commands'][0][:old['commands'][0].index('-c', 4)]
    require(all(command[:command.index('-c', 4)] == old_prefix for command in old['commands'][:-1]),
            'Historical compile prefixes differ.')
    # Born originally had the same numerical flags, before observer module isolation.
    born_manifest = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype/amplitude-native-1/build.json'
    born_old = json.loads(born_manifest.read_text())
    born_command = next(command for command in born_old['commands']
                        if '-c' in command[4:] and command[command.index('-c', 4) + 1].endswith('/born_fallback.f'))
    born_prefix = born_command[:born_command.index('-c', 4)]
    expected_born_prefix = [value for value in old_prefix
                            if value not in ('-J' + str(old_work), '-I' + str(old_work))]
    require(born_prefix == expected_born_prefix, 'Born numerical flags differ from historical observer flags.')
    names = list(SOURCE_ORDER)
    if arm_info['topology_gate']:
        names.append('shell_classes.f')
    require(set(names) == set(arm_info['sources']), 'Arm source set is unexpected.')
    commands = [prefix + ['-c', str(work / name), '-o', str(work / (Path(name).stem + '.o'))]
                for name in names]
    old_link = old['commands'][-1]
    output_at = old_link.index('-o')
    require(old_link[:output_at] == old_prefix, 'Historical link flags differ.')
    first_object = output_at + 2
    first_library = next(i for i in range(first_object, len(old_link)) if old_link[i].startswith('-L'))
    objects = old_link[first_object:first_library]
    require(len(objects) == 24 and len(set(objects)) == 24, 'Historical 24-object link order changed.')
    replaced = {Path(name).with_suffix('.o').name for name in SOURCE_ORDER}
    require(sum(Path(name).name in replaced for name in objects) == len(replaced),
            'Every declared object replacement must occur exactly once.')
    linked = [str(work / Path(name).name) if Path(name).name in replaced else name for name in objects]
    if arm_info['topology_gate']:
        linked.append(str(work / 'shell_classes.o'))
    libraries = old_link[first_library:]
    commands.append(prefix + ['-o', str(work / 'amplitude-private'), *linked, *libraries])
    return {'arm': arm, 'sources': arm_info['sources'], 'commands': commands,
            'compiler': fc, 'linked_objects': linked, 'library_args': libraries,
            'historical_object_list': objects, 'object_replacements': sorted(replaced),
            'source_order': names, 'topology_gate': arm_info['topology_gate']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    checks = pins()
    recipe = plan(args.arm)
    if args.plan_only:
        print(json.dumps(recipe, indent=2, sort_keys=True))
        return
    checks.update(controller_pins())
    require(recipe == json.loads((HERE / 'plans.json').read_text())[args.arm], 'Reviewed build recipe changed.')
    out, work = OUTBASE / args.arm, WORKBASE / args.arm
    require(not out.exists() and not work.exists(), 'No automatic rebuild or replacement.')
    out.mkdir(parents=True)
    work.mkdir(parents=True)
    record = {'schema': 1, 'complete': False, 'native_executed': False,
              'arm': args.arm, 'cpu': 11, 'commands': recipe['commands'],
              'protected_inputs_before': checks, 'controller_index_sha256': sha(HERE / 'source-index.json'),
              'historical_build_sha256': sha(OLD / 'attempt1/build.json'),
              'compiler': recipe['compiler'], 'source_order': recipe['source_order'],
              'linked_objects': [rel(Path(p)) for p in recipe['linked_objects']],
              'scope': 'Private source arm; unchanged historical numerical flags, ordered objects/libraries except declared replacements.'}
    try:
        (out / 'frozen-verifier.log').write_text(frozen_verify())
        validate_pins(checks)
        for module in ('mtirust.mod', 'mninjars.mod', 'qcd_kinematics_observer.mod'):
            require(not (ROOT / module).exists(), 'Unexpected current-directory module: ' + module)
        copied = {}
        for name, entry in recipe['sources'].items():
            source = ROOT / entry['path']
            require(source.stat().st_size == entry['bytes'] and sha(source) == entry['sha256'], 'Source arm changed.')
            shutil.copyfile(source, work / name)
            copied[rel(work / name)] = entry['sha256']
        protocol = json.loads((DESIGN / 'protocol.json').read_text())
        for name in ('param_card.dat', 'ident_card.dat', 'points.dat'):
            entry = protocol['frozen_inputs'][name]
            shutil.copyfile(ROOT / entry['path'], work / name)
            copied[rel(work / name)] = entry['sha256']
        record['private_copies_expected'] = copied
        validate_pins(copied)
        (work / 'tmp').mkdir()
        # Use a positive environment allowlist: never inherit NIX_* flag hooks,
        # CPATH/LIBRARY_PATH/compiler search overrides, preload hooks, or FPE policy.
        # The old manifests did not record their environment; equality is claimed
        # for historical argv and input bytes, not an unknowable old environment.
        env = {'PATH': os.environ['PATH'], 'LC_ALL': 'C', 'LANG': 'C',
               'TMPDIR': str(work / 'tmp'), 'OMP_NUM_THREADS': '1'}
        record['compiler_environment'] = env
        record['compiler_environment_policy'] = 'Positive allowlist; historical environment was not retained.'
        record['compiler_version'] = subprocess.check_output([recipe['compiler'], '--version'], cwd=ROOT,
                                                            env=env, text=True, timeout=30)
        record['command_records'] = []
        with (out / 'build.log').open('x') as log:
            for command in recipe['commands']:
                log.write('COMMAND ' + shlex.join(command) + '\n')
                log.flush()
                item = {'command': command, 'start_unix': time.time()}
                record['command_records'].append(item)
                try:
                    validate_pins(copied)
                    item['returncode'] = run_capped(command, log, env=env, seconds=180)
                    require(item['returncode'] == 0, 'Compile/link failed; no retry.')
                finally:
                    item['end_unix'] = time.time()
        record['artifacts_sha256'] = {rel(p): sha(p) for p in sorted(work.iterdir()) if p.is_file()}
        record['linked_objects_sha256'] = {rel(Path(p)): sha(Path(p)) for p in recipe['linked_objects']}
        record['complete'] = True
    except BaseException as exc:
        record['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        raise
    finally:
        if 'private_copies_expected' in record:
            record['private_copies_after'] = {p: sha(ROOT / p) for p in record['private_copies_expected']}
            record['private_copies_unchanged'] = record['private_copies_after'] == record['private_copies_expected']
            if not record['private_copies_unchanged']:
                record['complete'] = False
        record['artifacts_sha256'] = {rel(p): sha(p) for p in sorted(work.iterdir()) if p.is_file()}
        record['protected_inputs_after'] = {p: sha(ROOT / p) for p in checks}
        record['protected_inputs_unchanged'] = record['protected_inputs_after'] == checks
        (out / 'build.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
        require(record['protected_inputs_unchanged'], 'Protected inputs changed during build.')
        require(record.get('private_copies_unchanged', False), 'Frozen private copies changed or were not established.')
    print(args.arm + ': private build completed; no native reducer execution.')


if __name__ == '__main__':
    main()
