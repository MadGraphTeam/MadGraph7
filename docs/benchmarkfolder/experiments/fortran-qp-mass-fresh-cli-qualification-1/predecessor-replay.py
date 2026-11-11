"""Portable verification of authenticated text; historical programs stay inert."""
import argparse
import ast
import builtins
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import posixpath
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import types

sys.dont_write_bytecode = True
SOURCES = {
    'implementation': '75f32af86266ed01ad06a3b42caabf12897ce6f6c5cefac4032ccf6d77cd9d25',
    'source-1': '9e27928625b59cc76e71a657799c18f21844050e6e79271ac8810f3f6a85ff06',
    'source-2': '3c0f53a6e5526535b6fae98b0f055466b429a9a9cfc863fb22fb999a5c8a0815',
}
BUILD = '74fd12c5bb49d62a59eee94657e25cb9b27760beb90901f6b2ac07c3416a070a'
BUILD_RESULT = '17f0057e011791eb9d75b9ff5529e3a1542e9d5a86ef390462dfb43a716b0917'
FAILED = 'fa0c14036f9ea341ec3f25f679fe9966c81b32dad245c1ebe976c050a267d5b5'
FOLLOWUP = 'ff32a834e10c08a26f9842f0e0448f928e979f8581c77c5de09b886719184b7c'
AUX_SOURCE = 'ea987abb0c313cfe2d01423ca26b3231ff40b9ba1af6896143f70cee594f00e1'
AUX_FAILED = '61b000fbf56ab27dd32f37accf37ce69fc73635254ceb76fcc9a0c5616a47304'
AUX_AUDIT = '720ed92486e4d82f5c976a31c7f5b883cd4c6069624b4dc14a93a9539dd183dd'
LIMITS = {'address_space_bytes': [17179869184] * 2,
          'file_bytes': [536870912] * 2, 'core_bytes': [0, 0]}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def load_source(name, data, logical_path):
    module = types.ModuleType(name)
    module.__file__ = str(logical_path)
    exec(compile(data, str(logical_path), 'exec'), module.__dict__)
    return module


@contextmanager
def blocked_apis():
    saved, attempts = [], []
    def deny(*args, **kwargs):
        attempts.append(repr(args[:1]))
        raise RuntimeError('Forbidden API during portable text replay')
    def patch(obj, name):
        if hasattr(obj, name):
            saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, deny)
    try:
        for name in ('open', 'stat', 'lstat', 'resolve', 'exists', 'is_file', 'is_dir', 'is_symlink',
                     'iterdir', 'glob', 'rglob', 'read_bytes', 'read_text', 'write_bytes', 'write_text',
                     'mkdir', 'unlink', 'rename', 'replace', 'touch', 'chmod', 'symlink_to', 'hardlink_to'):
            patch(Path, name)
        patch(builtins, 'open')
        patch(io, 'open')
        patch(io, 'open_code')
        for name in ('open', 'stat', 'lstat', 'access', 'listdir', 'scandir', 'readlink', 'system', 'popen',
                     'posix_spawn', 'posix_spawnp', 'execv', 'execve', 'execvp', 'execvpe', 'spawnv',
                     'spawnve', 'spawnvp', 'spawnvpe', 'fork', 'forkpty', 'mkdir', 'makedirs', 'unlink',
                     'remove', 'rename', 'replace', 'rmdir', 'removedirs', 'chmod', 'chown', 'truncate',
                     'utime', 'link', 'symlink'):
            patch(os, name)
        for name in ('Popen', 'run', 'call', 'check_call', 'check_output'):
            patch(subprocess, name)
        for name in ('CDLL', 'PyDLL', '_dlopen'):
            patch(ctypes, name)
        if '_ctypes' in sys.modules:
            patch(sys.modules['_ctypes'], 'dlopen')
        yield attempts
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def source_derivation(blobs):
    def data(name):
        return blobs[name]
    def text(name):
        return data(name).decode('utf-8')
    pins = json.loads(data('source-2/input-pins.json'))['cases']['n3']
    prefix = 'original-cli/SubProcesses/' + pins['subprocess'] + '/'
    original = {name: text(prefix + name) for name in
                ('CT_interface.f', 'TIRUST_interface.f', 'loop_CT_calls_1.f')}
    matrix = load_source('portable_mass_materialize', data('source-2/materialize.py'), 'source-2/materialize.py')
    policy = load_source('portable_mass_policy', data('source-2/policy.py'), 'source-2/policy.py')
    driver_base = load_source('portable_mass_driver_base', data('source-2/driver_base.py'), 'source-2/driver_base.py')
    drivers = load_source('portable_mass_drivers', data('source-2/drivers.py'), 'source-2/drivers.py')
    # This replaces only the frozen helper's filesystem read. Its literal and
    # the exact original replacement remain authenticated and unchanged.
    capture_raw = data('support/' + driver_base.CAPTURE_HELPER)
    require(sha(capture_raw) == driver_base.CAPTURE_HELPER_SHA256, 'Read-only IEEE source changed')
    values = [ast.literal_eval(n.value) for n in ast.parse(capture_raw).body
              if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CAPTURE' for t in n.targets)]
    require(len(values) == 1 and isinstance(values[0], str), 'Ambiguous captured IEEE literal')
    driver_base.capture_source = lambda: values[0].replace('QP_CAPTURE_FP', 'O3_CAPTURE_FP')
    candidate, proof = matrix.candidate(original['CT_interface.f'], original['TIRUST_interface.f'],
                                         original['loop_CT_calls_1.f'], text('original-cli/Source/MODEL/mp_coupl.inc'))
    proof.update(complete_original_source_bindings=195, original_numerical_object_bindings=144,
                 actual_CLI_generation_sha256=sha(data('original-cli/generation.json')),
                 current_n3_mass_arguments={name: sum(m['model_symbol'] == name for g in proof['groups']
                                                      for m in g['mass_pairs'])
                                            for name in ('ZERO', 'MDL_MB', 'MDL_MT')})
    require(proof == json.loads(data('source-2/source-proof.json')), 'Actual CLI mass derivation differs')
    variants = {'original': original, 'candidate': candidate,
                'original-observed': matrix.observed(original, 0),
                'candidate-observed': matrix.observed(candidate, 1)}
    for variant, contents in variants.items():
        contents['check_sa.f'] = drivers.worker(text(prefix + 'check_sa.f'), driver_base, variant.endswith('-observed'))
        for name, value in contents.items():
            require(value.encode() == data('source-2/source-preview/'+variant+'/'+name), 'Rendered source differs')
    return variants, policy, proof


def verify(root, *, draft=False, expected_index_sha256=None):
    root = Path(root).resolve()
    name = 'draft-inputs.json' if draft else 'index.json'
    raw_index = (root/name).read_bytes()
    if expected_index_sha256:
        require(sha(raw_index) == expected_index_sha256, 'Portable index seal changed')
    index = json.loads(raw_index)
    require(index['schema'] == ('current-qp-mass-portable-draft-v1' if draft else
                                'current-qp-mass-portable-qualification-v1'), 'Wrong capsule schema')
    require(index['sealed'] is (not draft), 'Draft/final admission differs')
    require(index['source_seals'] == SOURCES and index['native_binaries_packaged'] == 0 and
            index['new_native_executions'] == 0 and index['public_writes'] is False,
            'Contradictory source/native/publication metadata')
    require(index['auxiliary_CLI_status'] == 'failed-startup-unqualified' and
            index['auxiliary_source_sha256'] == AUX_SOURCE and index['auxiliary_terminal_sha256'] == AUX_FAILED
            and index['fresh_CLI_correspondence_qualified'] is False,
            'Auxiliary CLI failure scope was relabelled')
    actual = set()
    for item in root.rglob('*'):
        require(not item.is_symlink(), 'Capsule symlinks forbidden')
        if item.is_file():
            actual.add(str(item.relative_to(root)))
    require(actual == set(index['files']) | {name}, 'Capsule exact inventory changed')
    physical = {}
    for relative, row in index['files'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe capsule payload')
        value = (root/relative).read_bytes()
        require(len(value) == row['bytes'] and sha(value) == row['sha256'], 'Owned payload changed: '+relative)
        require(not value.startswith((b'\x7fELF', b'!<arch>\n')), 'Binary payload packaged')
        physical[relative] = value
    blobs = {}
    for logical, row in index['logical_files'].items():
        value = physical[row['path']]
        require(row['compression'] in ('none', 'gzip'), 'Unknown retention encoding')
        if row['compression'] == 'gzip':
            require(value[:3] == b'\x1f\x8b\x08' and value[4:8] == b'\0'*4, 'Non-deterministic gzip header')
            value = gzip.decompress(value)
        require(len(value) == row['raw_bytes'] and sha(value) == row['raw_sha256'], 'Logical text changed')
        value.decode('utf-8')
        blobs[logical] = value
    def data(logical, checksum=None):
        value = blobs[logical]
        if checksum:
            require(sha(value) == checksum, 'Frozen evidence identity changed: '+logical)
        return value
    def record(logical, checksum=None):
        return json.loads(data(logical, checksum))
    math_replay = load_source('portable_mass_math', physical['math_replay.py'], root/'math_replay.py')
    with blocked_apis() as attempts:
        for prefix, checksum in SOURCES.items():
            source = record(prefix+'/source-index.json', checksum)
            owned = {n[len(prefix)+1:] for n in blobs if n.startswith(prefix+'/')}
            require(owned == set(source['files']) | {'source-index.json'}, 'Nested source inventory changed')
            for relative, row in source['files'].items():
                require(len(data(prefix+'/'+relative, row['sha256'])) == row['bytes'], 'Nested source size changed')
            for relative, expected in source.get('support', {}).items():
                data('support/'+relative, expected)
        auxiliary = record('auxiliary-cli/source-2/source-index.json', AUX_SOURCE)
        auxiliary_names = {n[len('auxiliary-cli/source-2/'):] for n in blobs if n.startswith('auxiliary-cli/source-2/')}
        require(auxiliary_names == set(auxiliary['files']) | {'source-index.json'}, 'Auxiliary source inventory changed')
        for name, row in auxiliary['files'].items():
            require(len(data('auxiliary-cli/source-2/'+name, row['sha256'])) == row['bytes'],
                    'Auxiliary source payload changed')
        auxiliary_failed = record('auxiliary-cli/execution-2/terminal.json', AUX_FAILED)
        require(auxiliary_failed['status'] == 'failed' and auxiliary_failed['returncode'] == 1 and
                not auxiliary_failed['cleanup']['survivors'] and auxiliary_failed['cap_seconds'] == 300 and
                auxiliary_failed['cleanup_reserve_seconds'] == 60 and auxiliary_failed['source_sha256'] == AUX_SOURCE,
                'Auxiliary startup failure no longer closed/failed')
        audit = record('auxiliary-cli/independent-failed-audit-2/receipt.json', AUX_AUDIT)
        require(audit['status'] == 'confirmed_failed_startup_no_qualification' and
                audit['source_index_sha256'] == AUX_SOURCE and audit['terminal_sha256'] == AUX_FAILED,
                'Auxiliary failure audit changed')
        auxiliary_raw = {n[len('auxiliary-cli/execution-2/'):] for n in blobs
                         if n.startswith('auxiliary-cli/execution-2/')}
        require(auxiliary_raw == set(audit['raw_execution_files']), 'Auxiliary failure logs incomplete')
        for name, row in audit['raw_execution_files'].items():
            require(len(data('auxiliary-cli/execution-2/'+name, row['sha256'])) == row['bytes'],
                    'Auxiliary raw failure evidence changed')
        log = data('auxiliary-cli/execution-2/generation.log', audit['generation_log_sha256']).decode()
        require("ModuleNotFoundError: No module named 'mg5decay'" in log,
                'Missing auxiliary startup failure witness')
        audit_preflight = record('auxiliary-cli/independent-failed-audit-2/protected-preflight.json',
                                 audit['protected_state']['protected_preflight_sha256'])
        require(record('auxiliary-cli/postfailure-preflight-2.json') == audit_preflight and
                record('auxiliary-cli/preflight-2.json') == audit_preflight and
                audit_preflight['CLI_invocations'] == audit_preflight['compiler_invocations'] ==
                audit_preflight['compiler_queries'] == audit_preflight['scientific_native_calls'] == 0,
                'Auxiliary post-failure read-only preflight changed')
        input_pins = record('source-2/input-pins.json')
        pins = input_pins['cases']['n3']
        excluded = index['excluded_binaries']
        for original, row in input_pins['files'].items():
            location = ('inputs/external/'+sha(original.encode())+'/'+Path(original).name
                        if Path(original).is_absolute() else 'inputs/'+original)
            if location in blobs:
                require(len(data(location, row['sha256'])) == row['bytes'], 'Pinned input size differs')
            else:
                require(excluded[location]['sha256'] == row['sha256'] and
                        excluded[location]['bytes'] == row['bytes'], 'Excluded input identity differs')
        for original, checksum in input_pins['ancestry'].items():
            data('ancestry/frozen/'+original, checksum)
        original_root = pins['output'].split('/.benchmark-work/', 1)[0]
        def pinned_alias(logical, original):
            relative = original[len(original_root)+1:] if original.startswith(original_root+'/') else original
            row = input_pins['files'].get(original, input_pins['files'].get(relative))
            require(row is not None and len(data(logical, row['sha256'])) == row['bytes'],
                    'Ordinary source/input alias lacks its frozen binding: '+logical)
        for logical, original in (
                ('ancestry/ordinary-warmup.dat', pins['closed_warmup']),
                ('ancestry/ordinary-observation.json', pins['closed_observation']),
                ('ancestry/ordinary-input.json', pins['closed_input']),
                ('ancestry/ordinary-points.dat', pins['points']),
                ('original-cli/generation.json', pins['output']+'/generation.json'),
                ('original-cli/compilation.json', pins['output']+'/compilation.json')):
            pinned_alias(logical, original)
        require(len(pins['generated_sources']) == 195 and len(pins['numerical_objects']) == 144,
                'Complete frozen CLI source/object inventory differs')
        for relative, expected in pins['generated_sources'].items():
            data('original-cli/'+relative, expected)
        variants, policy, proof = source_derivation(blobs)
        failed = record('failed-build-1/terminal.json', FAILED)
        require(failed['status'] == 'failed' and failed['returncode'] != 0 and not failed['cleanup']['survivors']
                and 'failed-build-1/result.json' not in blobs, 'Historical failure was relabelled')
        failed_commands = [n for n in blobs if n.startswith('failed-build-1/') and n.endswith('.command.json')]
        require(len(failed_commands) == 67, 'Failed BUILD command history incomplete')
        commands = [n for n in blobs if n.startswith('stages/build/') and n.endswith('.command.json')]
        require(len(commands) == 297, 'Successful BUILD command coverage differs')
        for command in failed_commands + commands:
            stem = command[:-len('.command.json')]
            closed = record(stem+'.terminal.json')
            require(closed['returncode'] == 0 and closed['stdout_sha256'] == sha(data(stem+'.stdout.txt')),
                    'Recorded build command/stdout does not match')
            if 'stderr_sha256' in closed:
                data(stem+'.stderr.txt', closed['stderr_sha256'])
        for name in blobs:
            if name.startswith('stages/') and name.endswith('.command.json'):
                stem = name[:-len('.command.json')]
                if stem+'.terminal.json' not in blobs:
                    continue  # In-process worker protocol record, not a spawned command.
                closed = record(stem+'.terminal.json')
                require(closed['returncode'] == 0 and closed['stdout_sha256'] == sha(data(stem+'.stdout.txt')),
                        'Closed-stage command failed or lost stdout')
                if 'stderr_sha256' in closed:
                    data(stem+'.stderr.txt', closed['stderr_sha256'])
        terminal = record('stages/build/terminal.json', BUILD)
        built = record('stages/build/result.json', BUILD_RESULT)
        require(terminal['status'] == built['status'] == 'passed' and not terminal['cleanup']['survivors'],
                'Successful BUILD no longer closed')
        require(built['native_amplitude_calls'] == built['unit_executions'] == 0 and
                built['compiler_policy_change'] is False and built['private_MTIrust_rebuild'] is False,
                'BUILD scope inflated')
        compiler = load_source('portable_mass_compiler_policy', data('support/tools/tirust_benchmark_compiler.py'),
                               'support/tools/tirust_benchmark_compiler.py')
        prefix = 'SubProcesses/'+pins['subprocess']+'/'
        replacements = {prefix+Path(n).with_suffix('.o').name for n in policy.NEEDED}
        for variant, row in built['variants'].items():
            observed = variant.endswith('-observed')
            require(set(row['objects']) == set(pins['numerical_objects']) | (
                {prefix+'mgmass_observer.o', prefix+'mgmass_observer_state.mod'} if observed else set()),
                'Canonical144 object/observer inventory differs')
            for object_name, expected in pins['numerical_objects'].items():
                if object_name not in replacements:
                    require(row['objects'][object_name] == expected, 'Unselected physics object changed')
            require(row['unchanged_objects']['unchanged'] == 140, 'Unchanged object count differs')
            expected_sources = dict(pins['generated_sources'])
            for source_name, value in variants[variant].items():
                expected_sources[prefix+source_name] = sha(value.encode())
            original_make = data('original-cli/'+prefix+'makefile').decode()
            expected_sources[prefix+'makefile'] = sha(policy.private_makefile(original_make, observed).encode())
            require(row['sources'] == expected_sources, 'Compiled source family differs')
            policy.verify_policy(pins['compiler_flags'], row['compiler_flags'], compiler)
            require(row['policy_evidence']['TIRUST_interface_flags'] == ['-w', '-fPIC', '-ffixed-line-length-132'],
                    'Original TIRust compiler flags changed')
            require(len(row['MTIRUST_definitions']) == 23 and set(row['MTIRUST_definitions'].values()) == {1}
                    and len(row['private_definitions']) == (9 if variant.startswith('candidate') else 0),
                    'Public/private symbol ownership changed')
            stem = 'stages/build/'+variant+'/'
            pre = record(stem+'prelink.json', row['prelink_sha256'])
            data(stem+'private-link.map', row['link_map_sha256'])
            require(pre['completed_before_link'] and len(row['runtime']['files']) == 15 and
                    pre['explicit_current_runtime_before_link'] == row['runtime']['files'] and
                    all(pre['bound'].get(n) == h for n, h in
                        {**row['actual_link_loads'], **row['runtime']['files']}.items()),
                    'Recorded actual LOAD/runtime was not prebound')
            require(excluded[stem+'check']['sha256'] == row['executable_sha256'], 'Executable identity absent')
            for object_name, expected in row['objects'].items():
                require(excluded[stem+'objects/'+object_name]['sha256'] == expected, 'Object identity absent')
        unit = built['units']
        data('stages/build/units/unit-link.map', unit['link_map_sha256'])
        require(len(unit['runtime']['files']) == 5 and
                not any(re.search('tirust|ninja|collier', n, re.I) for n in unit['runtime']['files']),
                'Recording unit runtime scope inflated')
        require(excluded['stages/build/units/unit_mass_check']['sha256'] == unit['executable_sha256'],
                'Recording unit executable identity absent')
        require(set(index['stage_status']) == set(math_replay.STAGES), 'Planned stage coverage differs')
        for stage, state in index['stage_status'].items():
            if state['status'] == 'pending':
                require('stages/'+stage+'/terminal.json' not in blobs, 'Closed stage labelled pending')
                continue
            closed = record('stages/'+stage+'/terminal.json', state['terminal_sha256'])
            cap = 180 if stage == 'units' else 600
            require(closed['cap_seconds'] == cap and closed['cleanup_reserve_seconds'] == 60 and
                    not closed['cleanup']['survivors'], 'Stage limits/closure differ')
            require(record('stages/'+stage+'/actual-limits.json')['actual'] == LIMITS,
                    'Actual stage resource limits differ')
            if stage in ('cache0', 'cacheall', 'observeoriginal', 'observecandidate') and state['status'] == 'passed':
                physical_result = record('stages/'+stage+'/result.json')
                allowed_data = record('source-2/source-index.json')['allowed_extra_mappings']
                for variant in physical_result['lanes']:
                    built_row = built['variants'][variant]
                    worker_pid = None
                    for edge in ('before', 'after'):
                        mapped = record('stages/'+stage+'/'+variant+'/maps-'+edge+'.json')
                        require(worker_pid is None or worker_pid == mapped['pid'], 'Mapped worker PID changed')
                        worker_pid = mapped['pid']
                        candidates = {n: h for n, h in mapped['files'].items()
                                      if n.endswith('/SubProcesses/'+pins['subprocess']+'/check')}
                        require(len(candidates) == 1 and list(candidates.values()) == [built_row['executable_sha256']],
                                'Mapped executable differs from reviewed BUILD')
                        required = {**built_row['runtime']['files'], **candidates}
                        allowed = {**required, **allowed_data}
                        require(all(mapped['files'].get(n) == h for n, h in required.items()) and
                                all(allowed.get(n) == h for n, h in mapped['files'].items()),
                                'Initialized worker map/provider identities changed')
                        raw_paths = {posixpath.normpath(line.split(None, 5)[5]) for line in mapped['maps'].splitlines()
                                     if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')}
                        require(raw_paths == set(mapped['files']) and
                                any(row['pid'] == worker_pid for row in closed['cleanup']['observed_owned_processes']),
                                'Recorded raw worker maps or owned cleanup identity differs')
        data('followup/matched-input/README.md', FOLLOWUP)
        binding = json.loads(physical['followup/source-binding.json'])
        require(binding['source_only_followup_sha256'] == FOLLOWUP and
                not binding['matched_input_replay_executed'] and not binding['independent_finite_truth_qualified'],
                'Source-only oracle follow-up relabelled as executed')
        handoff = data('followup/matched-input/README.md').decode()
        handoff_pins = dict(re.findall(r'\| `(observer(?:_analysis)?\.(?:f90|py))` \| `([0-9a-f]{64})`', handoff))
        require(set(handoff_pins) == set(binding['observer_bindings']) == {'observer.f90', 'observer_analysis.py'},
                'Exact final observer/handoff binding coverage differs')
        for name, row in binding['observer_bindings'].items():
            require(row['handoff_sha256'] == handoff_pins[name] and
                    row['final_source_sha256'] == sha(data('source-2/'+name)) and
                    row['matches'] == (row['handoff_sha256'] == row['final_source_sha256']),
                    'Final observer/handoff identity join differs')
        numerical = math_replay.replay(blobs, load_source)
        require({stage: value['status'] for stage, value in index['stage_status'].items()} ==
                numerical['stage_status'], 'Retained stage status differs from raw closure')
        require(index['candidate_promoted'] is False and index['performance_acceptance'] is False and
                index['independent_physical_finite_truth_qualified'] is False,
                'Unqualified publication claim')
        require(not attempts, 'Portable replay attempted forbidden effects')
        if not draft:
            require(numerical['all_planned_stages_closed_pass'], 'Final packet lacks a passed required mass stage')
    result = {'verified': True, 'draft': draft, 'source_derivation': {'groups': len(proof['groups']),
             'helpers': len(proof['helpers']), 'complete_CLI_sources': 195, 'canonical_objects': 144},
            'historical_failed_BUILD_retained': True, 'build_commands': 297,
            'numerical': numerical, 'native_binaries_packaged': 0,
            'forbidden_API_attempts': len(attempts), 'new_native_executions': 0,
            'auxiliary_CLI_status': 'failed-startup-unqualified', 'fresh_CLI_correspondence_qualified': False,
            'runtime_scope': 'Recorded LOAD/PT_INTERP/maps identities; no live provider or binary reads.'}
    if not draft:
        require(index['claims'] == summary(result), 'Final claims differ from replayed evidence')
    return result


def summary(result):
    numerical = result['numerical']
    compact = {key: value for key, value in numerical.items() if key != 'numerical'}
    checked = {}
    for stage, value in numerical['numerical'].items():
        if stage == 'units':
            checked[stage] = {key: value[key] for key in
                              ('recorded_helper_calls', 'actual_selected_backend',
                               'genuine_113bit_mass_input_indices', 'bottom_square_significand_bits')}
        elif stage in ('cache0', 'cacheall'):
            comparison = value['comparison']
            checked[stage] = {key: value[key] for key in ('points', 'DP_points', 'automatic_QP_points')}
            checked[stage].update({key: comparison[key] for key in
                                  ('DP_exact', 'coefficient_values_compared', 'all_coefficient_words_equal',
                                   'all_accuracy_words_equal', 'passed')})
        else:
            checked[stage] = value
    compact['checks'] = checked
    return {**result, 'numerical': compact}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', action='store_true')
    parser.add_argument('--index-sha256')
    args = parser.parse_args()
    print(json.dumps(summary(verify(Path(__file__).resolve().parent, draft=args.draft,
                                    expected_index_sha256=args.index_sha256)), sort_keys=True, indent=2))
