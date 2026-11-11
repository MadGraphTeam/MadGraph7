"""Portable, text-only replay of the direct emitted-wrapper ABI qualification.

Only included analyzer/reference code executes. Recorded commands and absolute
paths are inert strings; no compiler, loader, reducer or historical script runs.
"""
import argparse
from collections import Counter
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import types

SOURCE_SHA = '35018c3a05f2e863be061c68bbba9fc64ccf15009c94f68ec3012f955601394d'
SOURCE3_SHA = 'c99ce0ffb80a1dabe90a109cc958305efabe1af7968b97ec0adc65a16b009dc2'
CLOSURE_SHA = '52a4b01ca99d5ad6132c1f9fa0ca9b50a471213aab542b34434e8fa48a94a92e'
SOURCE2_SHA = '0e93995f751c7a0f3dfae1e39748714afe5a194ed6dbc1c9e873ca9d89117685'
AUDIT_SHA = 'a396a888a83ecce6bf143cf89c9901bf251d447692c327a7a0f8638a41ddb88d'
BUILD_NAMES = ('environment_token.c', 'aloha_object.f', 'invariant_bounds.f90',
               'invariant_bounds_data.inc', 'MadLoopParams.inc',
               'recording.f90', 'wrapper.f', 'driver.f90')
BUILD_LOGS = ('01-c.log', '02-aloha_object.log', '03-invariant_bounds.log',
              '04-recording.log', '05-wrapper.log', '06-driver.log', '07-link.log')


def need(condition, message):
    if not condition:
        raise ValueError(message)


def identity(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def relative(name):
    need(isinstance(name, str), 'nonstring payload path')
    p = PurePosixPath(name)
    need(name and p.parts and not p.is_absolute() and '..' not in p.parts
         and str(p) == name and '\\' not in name, 'unsafe payload path')
    return p


class Package:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.cache = {}
        need(self.root.is_dir() and not self.root.is_symlink(), 'invalid package directory')

    def bytes(self, name):
        if name in self.cache:
            return self.cache[name]
        p = relative(name)
        current = self.root
        for part in p.parts:
            current = current / part
            need(not current.is_symlink(), 'symlink payload or directory')
        need(stat.S_ISREG(current.stat().st_mode), 'nonregular payload')
        raw = current.read_bytes()
        self.cache[name] = raw
        return raw

    def json(self, name):
        return json.loads(self.bytes(name))

    def check(self, name, pin):
        need(identity(self.bytes(name)) == pin, 'changed payload: ' + name)

    def inventory(self):
        files = set()
        for directory, folders, names in os.walk(self.root, followlinks=False):
            for name in folders:
                need(not (Path(directory) / name).is_symlink(), 'symlink directory')
            for name in names:
                path = Path(directory) / name
                need(not path.is_symlink() and stat.S_ISREG(path.stat().st_mode), 'nonregular package entry')
                files.add(path.relative_to(self.root).as_posix())
        return files


def expected_rows(cases, geometry, reference):
    """Reproduce the admitted source.prepare.expected formula, without importing it."""
    rows = []
    for c in cases:
        mode = c['mode']
        width = 128 if mode >= 4 else 64
        mode_ok = mode in (1, 2, 4, 5)
        route_ok = reference.route_matches(geometry, c['route'], 3, c['we'], [1, 1, 1])
        slots_ok = mode_ok and c['route'] == 1
        env_ok = c['token_mode'] == 0 and c['round_index'] == 0
        admitted = route_ok and mode_ok and c['fresh'] and env_ok
        active = c['qp_words' if width == 128 else 'dp_words']
        external = [[active[e][mu] for e in range(3)] for mu in range(4)]
        bounds = reference.build(geometry, c['route'], mode, width, 3, external, admitted)
        rows.append(dict(id=c['id'], label=c['label'], mode=mode, width=width,
                         route=c['route'], fresh=c['fresh'], supplied_we=c['we'],
                         slots_ok=slots_ok, slots=[1, 2, 3] if slots_ok else [0, 0, 0],
                         have=[bounds is not None] * 3,
                         errors=bounds['chord_error'] if bounds else [[reference.ZERO] * 4 for _ in range(3)],
                         classes=bounds['pair_class'] if bounds else [[0] * 3 for _ in range(3)],
                         flags=[int(bool(c['sticky_mask'] & (1 << i))) for i in range(5)],
                         halts=[int(c['halt_flag'] == i) for i in range(1, 6)],
                         round_index=c['round_index'], token_mode=c['token_mode']))
    return rows


def rendered_inputs(cases):
    lines = [str(len(cases))]
    for c in cases:
        lines.append(' '.join(map(str, [c['id'], c['mode'], c['route'], *c['we'],
            int(c['fresh']), c['token_mode'], c['sticky_mask'], c['round_index'], c['halt_flag']])))
        for field, digits in [('dp_words', 16), ('qp_words', 32)]:
            for e in range(3):
                for mu in range(4):
                    lines.append(' '.join(f'{v:0{digits}X}' for v in c[field][e][mu]))
    return ('\n'.join(lines) + '\n').encode('ascii')


def load_analyzer(package):
    """Compile exact included code; satisfy its reference import from memory."""
    path = package.root / 'source/analyze.py'
    module = types.ModuleType('included_generic_wrapper_analyzer')
    module.__file__ = str(path)
    reference_path = package.root / 'source/reference.py'
    reference = types.ModuleType('wrapper_reference_for_analysis')
    reference.__file__ = str(reference_path)
    exec(compile(package.bytes('source/reference.py'), str(reference_path), 'exec'),
         reference.__dict__)
    class IncludedReference:
        def create_module(self, spec):
            return reference
        def exec_module(self, candidate):
            need(candidate is reference, 'unexpected reference module')
    def included_spec(name, location, *args, **kwargs):
        need(name == 'wrapper_reference_for_analysis' and str(location) == str(reference_path),
             'unexpected analyzer file import')
        return importlib.machinery.ModuleSpec(name, IncludedReference(), origin=str(reference_path))
    previous = sys.dont_write_bytecode
    previous_spec = importlib.util.spec_from_file_location
    try:
        sys.dont_write_bytecode = True
        importlib.util.spec_from_file_location = included_spec
        exec(compile(package.bytes('source/analyze.py'), str(path), 'exec'), module.__dict__)
    finally:
        importlib.util.spec_from_file_location = previous_spec
        sys.dont_write_bytecode = previous
    return module


def sealed_source(package, name, digest, count):
    raw = package.bytes(name)
    need(identity(raw)['sha256'] == digest, 'changed original source seal')
    index = json.loads(raw)
    need(len(index['files']) == count, 'original source inventory count changed')
    prefix = str(PurePosixPath(name).parent)
    for child, pin in index['files'].items():
        relative(child)
        package.check(prefix + '/' + child, pin)
    return index


def stage_receipt(package, name, action, count, source_sha):
    row = package.json(name)
    need(row['schema'] == 'generic-wrapper-abi-stage-v1' and row['stage'] == action
         and row['passed'] is True, 'failed or wrong stage receipt')
    need(row['source_index_sha256'] == source_sha, 'stage source seal mismatch')
    need(row['scientific_evaluations'] == 0 and row['full_sloop_lifetime_coverage'] is False,
         'stage overstates scientific scope')
    need(len(row['commands']) == count, 'stage command count mismatch')
    for command in row['commands']:
        progress = command['progress']
        need(command['returncode'] == 0 and progress['returncode'] == 0
             and progress['process_spawned'] is True and progress['process_reaped'] is True,
             'native command not successfully closed')
        need(progress['spawn_unix'] <= progress['exit_observed_unix'], 'command chronology mismatch')
    need(0 <= row['elapsed_seconds'] <= row['budget_seconds'], 'stage deadline mismatch')
    return row


def runtime_receipt(package, built, ran):
    runtime = ran['runtime']
    raw = package.bytes('run/runtime-maps.txt')
    need(identity(raw)['sha256'] == runtime['maps_sha256'], 'runtime map hash mismatch')
    paths = set()
    for line in raw.decode('utf8').splitlines():
        columns = line.split(None, 5)
        if len(columns) == 6 and columns[5].startswith('/'):
            need(not columns[5].endswith(' (deleted)'), 'deleted mapped file')
            paths.add(columns[5])
    observed = runtime['observed_file_backed_ELF']
    need(len(observed) == 6 and paths == set(observed), 'captured map/provider path mismatch')
    # These are retained provider identities, never filesystem paths to open.
    for name, pin in observed.items():
        need(isinstance(name, str) and name.startswith('/') and isinstance(pin, dict)
             and type(pin['bytes']) is int and pin['bytes'] > 0
             and len(pin['sha256']) == 64, 'invalid retained provider identity')
        need(name.rsplit('/', 1)[-1] not in ('libmadloop_tirust.so', 'libninjars.so'),
             'unexpected scientific provider')
    need(observed.get(runtime['executable']) == built['executable'] == ran['executable'],
         'executable identity join mismatch')
    need(runtime['actual_interpreter'] == built['PT_INTERP']['path']
         and observed.get(runtime['actual_interpreter']) == built['PT_INTERP']['identity'],
         'interpreter identity join mismatch')
    need(runtime['observed_file_backed_ELF_rehashed_after_exit'] is True,
         'missing original after-exit provider rehash claim')
    progress = ran['commands'][0]['progress']
    need(progress['mapping_capture']['released_once'] is True
         and progress['mapping_capture']['actual_file_backed_elf_count'] == 6,
         'runtime capture/release mismatch')
    need(runtime['pid'] == progress['pid']
         and runtime['process_exit_observed_unix'] == progress['exit_observed_unix']
         and runtime['process_wall_seconds'] == progress['process_wall_seconds'],
         'runtime command join mismatch')
    return 6


def verify(root, index_sha256):
    package = Package(root)
    index_raw = package.bytes('evidence-index.json')
    need(identity(index_raw)['sha256'] == index_sha256, 'evidence index digest mismatch')
    index = json.loads(index_raw)
    need(index.get('schema') == 'madloop-tirust-generic-wrapper-abi-evidence-v1',
         'unknown evidence schema')
    files = index['files']
    need(isinstance(files, dict) and files, 'missing payload inventory')
    need(package.inventory() == set(files) | {'evidence-index.json'}, 'extra/missing package files')
    for name, pin in files.items():
        relative(name)
        raw = package.bytes(name)
        need(identity(raw) == pin, 'payload identity mismatch: ' + name)
        raw.decode('utf8')
        need(b'\x00' not in raw and not raw.startswith(b'\x7fELF'), 'binary payload prohibited')
    source = sealed_source(package, 'source/index.json', SOURCE_SHA, 28)
    sealed_source(package, 'source3/source-index.json', SOURCE3_SHA, 20)
    need(source['source3_index_sha256'] == SOURCE3_SHA, 'source3 ancestor join mismatch')
    closure_raw = package.bytes('closed-qualification.json')
    need(identity(closure_raw)['sha256'] == CLOSURE_SHA, 'closed qualification seal mismatch')
    closure = json.loads(closure_raw)
    need(closure['status'] == 'pass' and closure['direct_wrapper_cases'] == 240
         and closure['available'] == 180 and closure['rejected'] == 60,
         'closed qualification summary mismatch')
    for name, key in [('source/index.json', 'source_index'), ('build/result.json', 'build_receipt'),
                      ('run/result.json', 'run_receipt'), ('run/native.log', 'raw_protocol'),
                      ('run/runtime-maps.txt', 'raw_maps')]:
        package.check(name, closure[key])
    need(closure['full_sloop_lifetime_coverage'] is False
         and closure['physical_amplitude_qualified'] is False
         and closure['production_changes'] is False and closure['scientific_evaluations'] == 0,
         'closed qualification overstates scope')
    scope = package.json('scope.json')
    for key in ('parent_instruction_intervals_qualified', 'full_geometry_window_qualified',
                'full_sloop_lifetime_coverage', 'physical_amplitude_qualified',
                'parent_producer_to_consumer_no_overwrite_qualified',
                'global_full_producer_environment_support', 'scientific_dispatch_enabled',
                'production_changes', 'full_source3_compiler_or_physical_promotion'):
        need(scope.get(key) is False, 'scope overstatement: ' + key)
    need(scope['tensor_or_scientific_evaluations'] == 0 and scope['binary_payloads'] == 0
         and scope['source_fixture_nsquaredso'] == 0
         and scope['binary_identities_are_receipts_only'] is True, 'scope identity/evaluation mismatch')
    built = stage_receipt(package, 'build/result.json', 'build', 7, SOURCE_SHA)
    ran = stage_receipt(package, 'run/result.json', 'run', 1, SOURCE_SHA)
    command_plan = package.json('source/command-plan.json')
    need([command['argv'] for command in built['commands']] ==
         command_plan['build']['serial_commands'], 'build command-plan mismatch')
    objects = ('environment_token.o', 'aloha_object.o', 'invariant_bounds.o',
               'recording.o', 'wrapper.o', 'driver.o')
    loads = [line.split(None, 1)[1] for line in
             package.bytes('build/probe.map').decode('utf8').splitlines()
             if line.startswith('LOAD ')]
    need(set(built['objects']) == set(objects)
         and all(loads.count(name) == 1 for name in objects), 'link map/object join mismatch')
    need(set(built['sources']) == set(BUILD_NAMES), 'consumed source list changed')
    for name in BUILD_NAMES:
        package.check('build/consumed/' + name, built['sources'][name])
        need(package.bytes('build/consumed/' + name) == package.bytes('source/' + name),
             'build did not consume sealed source: ' + name)
    for command, name in zip(built['commands'], BUILD_LOGS):
        package.check('build/' + name, command['log'])
    package.check('run/native.log', ran['commands'][0]['log'])
    need(built['PT_INTERP'] == source['expected_interpreter'], 'compiler interpreter differs from source plan')
    runtime_count = runtime_receipt(package, built, ran)
    generations = package.json('source3/generation-results.json')
    generated = next(row for row in generations if row['case'] == 'born-grouped')['generated']
    for name, generated_name in {
            'geometry.json': 'SubProcesses/P0/invariant_provenance.json',
            'invariant_bounds.f90': 'SubProcesses/P0/invariant_bounds.f90',
            'invariant_bounds_data.inc': 'SubProcesses/P0/invariant_bounds_data.inc',
            'environment_token.c': 'SubProcesses/P0/invariant_environment.c',
            'full-CT_interface.f': 'SubProcesses/P0/CT_interface.f',
            'full-aloha_functions.f': 'Source/DHELAS/aloha_functions.f'}.items():
        package.check('source/' + name, generated[generated_name])
    bindings = package.json('source/source-bindings.json')
    for name, full, start_pattern, end_pattern in [
            ('wrapper.f', 'full-CT_interface.f',
             r'^      SUBROUTINE ML5_0_LOOP_3_PROVENANCE\(', r'^      END\s*$'),
            ('aloha_object.f', 'full-aloha_functions.f',
             r'^      module ALOHA_OBJECT\s*$', r'^      end module ALOHA_OBJECT\s*$')]:
        selection = bindings[name]['selection']
        text = package.bytes('source/' + full).decode('utf8')
        first = re.search(start_pattern, text, re.M | re.I)
        need(first is not None, 'missing emitted source span start')
        last = re.search(end_pattern, text[first.start():], re.M | re.I)
        need(last is not None, 'missing emitted source span end')
        end = first.start() + last.end()
        selected = text[first.start():end]
        if not selected.endswith('\n'):
            selected += '\n'
        coordinates = dict(start_line=text[:first.start()].count('\n') + 1,
                           end_line=text[:end].count('\n'))
        need(coordinates == selection, 'emitted source span metadata mismatch')
        need(selected.encode('utf8') == package.bytes('source/' + name),
             'emitted caller/module extraction mismatch')
    cases = package.json('source/cases.json')
    expected = package.json('source/expected.json')
    need([c['id'] for c in cases] == list(range(1, 241)), 'case order mismatch')
    need(rendered_inputs(cases) == package.bytes('source/inputs.txt') == package.bytes('run/inputs.txt'),
         'native input rendering mismatch')
    analyzer = load_analyzer(package)
    need(expected_rows(cases, package.json('source/geometry.json'), analyzer.REF) == expected,
         'independent expected-bound recomputation mismatch')
    replay = analyzer.parse(package.bytes('run/native.log'), cases, expected)
    need(replay == ran['qualification'], 'retained/raw qualification mismatch')
    need(sum(r['source_bound_chords_available'] for r in replay['observations']) == 180,
         'raw accepted-bound count mismatch')
    audit_raw = package.bytes('independent-audit/receipt.json')
    need(identity(audit_raw)['sha256'] == AUDIT_SHA, 'independent audit seal mismatch')
    audit = json.loads(audit_raw)
    need(audit['schema'] == 'independent-generic-wrapper-abi-text-audit-v1'
         and audit['status'] == 'pass' and audit['source_index_sha256'] == SOURCE_SHA,
         'independent audit identity mismatch')
    for name, pin in audit['evidence_pins'].items():
        package.check(name, pin)
    need(identity(package.bytes('independent-audit/review.py'))['sha256'] == audit['review_source_sha256'],
         'independent audit source mismatch')
    need(audit['cases'] == 240 and audit['available'] == 180 and audit['rejected'] == 60
         and audit['command_receipts_checked'] == 8 and audit['source_payloads'] == 28,
         'independent audit count mismatch')
    direct_counts = dict(Counter(line.split()[0] for line in
                                package.bytes('run/native.log').decode('ascii').splitlines()
                                if line.strip()))
    need(audit['direct_counts'] == direct_counts and audit['raw_protocol_replay_exact'] is True
         and audit['commands_recorded_closed_and_reaped'] is True
         and audit['runtime_text_paths_equal_recorded_6_ELFs'] is True,
         'independent audit evidence mismatch')
    need(audit['compiled_or_native_executions_by_audit'] == 0
         and audit['live_ELF_or_library_bytes_read'] is False
         and audit['full_sloop_lifetime_coverage'] is False
         and audit['physical_amplitude_qualified'] is False
         and audit['scientific_dispatch'] == 'recording legacy stub only',
         'independent audit scope mismatch')
    diagnostic_source2 = package.bytes('diagnostics/source2-index.json')
    need(identity(diagnostic_source2)['sha256'] == SOURCE2_SHA, 'diagnostic source2 seal mismatch')
    diagnostic_index = json.loads(diagnostic_source2)
    for name in ('invariant_environment_source.py', 'invariant_window.py'):
        package.check('diagnostics/' + name,
                      diagnostic_index['files']['src/madgraph/loop/' + name])
    diagnostic_path = package.root / 'diagnostics/check_parent_interval_gap.py'
    diagnostic = types.ModuleType('included_parent_interval_gap_diagnostic')
    diagnostic.__file__ = str(diagnostic_path)
    exec(compile(package.bytes('diagnostics/check_parent_interval_gap.py'),
                 str(diagnostic_path), 'exec'), diagnostic.__dict__)
    gap = diagnostic.check(package.root)
    need(gap == package.json('diagnostics/gap-result.json'), 'parent interval gap replay mismatch')
    need(gap['parent_root_complete'] is False and gap['full_geometry_window_qualified'] is False
         and gap['native_or_environment_setter_executions'] == 0,
         'diagnostic overstates qualification')
    return dict(schema='generic-emitted-wrapper-abi-portable-verification-v1', status='pass',
                evidence_index_sha256=index_sha256, payloads=len(files), cases=240,
                source_bound_available=180, conservative_rejections=60,
                raw_protocol_replay_exact=True, expected_bounds_recomputed=True,
                independent_text_audit_bound=True,
                closed_native_commands=8, consumed_source_copies=8,
                retained_runtime_file_identities=runtime_count,
                runtime_paths_opened=False, new_native_executions=0,
                source_parent_interval_gap_reproduced=True,
                parent_instruction_intervals_qualified=False,
                full_geometry_window_qualified=False,
                full_sloop_lifetime_coverage=False, physical_amplitude_qualified=False,
                scientific_evaluations=0, performance_acceptance=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', nargs='?', default=str(Path(__file__).resolve().parents[1]))
    p.add_argument('--index-sha256', required=True)
    a = p.parse_args()
    print(json.dumps(verify(a.root, a.index_sha256), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
