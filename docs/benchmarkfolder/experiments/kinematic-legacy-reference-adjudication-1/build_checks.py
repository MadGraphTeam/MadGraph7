"""Pure joins for the closed legacy-policy reference build.

Authenticated packet callbacks supply all bytes. External source, object,
archive and provider paths are historical identities, never opened here.
Only the two retained source transformations are freshly replayed; the full
269-source generation and routing derivation remains historical.
"""
import ast
import difflib
import hashlib
import json
import re

SOURCE_SHA = '0a0dba9baf12ff5a988e4d459141a48140f285a396658ee10f7f09099848dbd8'
BUILD_SHA = '08a5b485954e20dc25fcffc2c26edddd73520b982e9daa380e1d5eca06709f22'
PARENT_SOURCE_SHA = 'ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74'
PARENT_BUILD_SHA = '8cab371e9affd5297c0f48c30184361959d20f3d3a48c6dbd6109b86bf49bacb'
ROOT = '/shared/external/madloop_tirust/madgraph7_madloop_tirust/'
WORK = '.tirust-build/kinematic-invariant-reference-adjudication-executor-1/'
OUTPUT = ROOT + WORK + 'work/builds/cms/reference'
SUB = 'SubProcesses/P0_epem_mupmum/'


def require(ok, message):
    if not ok:
        raise ValueError('Closed reference build: ' + message)


def pin(raw):
    require(isinstance(raw, bytes), 'callback must return bytes')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def checked(data, name, expected):
    raw = data(name)
    actual = pin(raw)
    require(actual['sha256'] == (expected if isinstance(expected, str) else expected['sha256']) and
            (isinstance(expected, str) or actual['bytes'] == expected['bytes']), 'changed payload ' + name)
    return raw


def identity(record):
    return {key: record[key] for key in ('bytes', 'sha256')}


def suffix(name, prefix):
    require(name.startswith(prefix), 'foreign recorded path')
    result = name[len(prefix):]
    require(result and all(part not in ('', '.', '..') for part in result.split('/')), 'unsafe recorded suffix')
    return result


def source_inventory(build):
    prefix = suffix(build['output'], ROOT) + '/'
    return {suffix(name, prefix): value for name, value in build['sources'].items()}


def reverse_patch(candidate, patch, name):
    """Undo exact canonical hunks in memory, rejecting fuzzy/partial matches."""
    current = candidate.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    require(lines[:2] == ['--- sealed-original/' + name + '\n',
                          '+++ private-reference/' + name + '\n'], 'patch header differs')
    original, position, cursor = [], 0, 2
    while cursor < len(lines):
        match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n', lines[cursor])
        require(match is not None, 'invalid hunk header')
        old_start, old_count = int(match[1]), int(match[2] or 1)
        new_start, new_count = int(match[3]), int(match[4] or 1)
        start = new_start - (1 if new_count else 0)
        require(position <= start <= len(current), 'overlapping/out-of-range hunk')
        original.extend(current[position:start]); position = start
        require(len(original) == old_start - (1 if old_count else 0), 'old hunk position differs')
        cursor += 1
        old_seen = new_seen = 0
        while cursor < len(lines) and not lines[cursor].startswith('@@ '):
            line = lines[cursor]
            require(line and line[0] in ' +-', 'invalid hunk record')
            marker, body = line[0], line[1:]
            if marker in (' ', '+'):
                require(position < len(current) and current[position] == body, 'hunk context differs')
                position += 1; new_seen += 1
            if marker in (' ', '-'):
                original.append(body); old_seen += 1
            cursor += 1
        require((old_seen, new_seen) == (old_count, new_count), 'hunk counts differ')
    original.extend(current[position:])
    result = ''.join(original)
    regenerated = ''.join(difflib.unified_diff(result.splitlines(True), candidate.splitlines(True),
        fromfile='sealed-original/' + name, tofile='private-reference/' + name))
    require(regenerated == patch, 'canonical patch differs')
    return result


def replay_sources(data, parent_data, parent_document, bindings, donor):
    old = {}
    for name, key, relative in [('CT_interface.f', 'CT_interface', SUB + 'CT_interface.f'),
                                ('driver-reference.f', 'driver', SUB + 'check_sa.f')]:
        text = data('source/' + name).decode('ascii')
        old[name] = reverse_patch(text, data('source/' + name + '.patch').decode('ascii'), name)
        require(pin(old[name].encode('ascii')) == identity(bindings[key]) == source_inventory(donor)[relative],
                'reconstructed donor/source binding differs')
    require(old['driver-reference.f'].encode('ascii') == parent_data('proposal/driver-amplitude.f'),
            'driver donor is not the retained parent driver')
    names = {'require', 'replace_once', 'hook', 'derive_adapter', 'derive_driver', 'card'}
    selected = []
    found = set()
    logger = None
    for node in ast.parse(data('source/prepare.py')).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            require(not node.decorator_list and not node.args.defaults and not any(node.args.kw_defaults) and
                    node.returns is None and not any(arg.annotation for arg in node.args.args),
                    'unexpected definition-time expression')
            selected.append(node); found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'LOGGER':
            logger = ast.literal_eval(node.value)
            require(isinstance(logger, str), 'logger must be literal text')
    require(found == names and logger is not None, 'pure source transform selection differs')
    namespace = {'__builtins__': {'ValueError': ValueError}, 'LOGGER': logger}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'sealed-reference-source-transform', 'exec'), namespace)
    require(namespace['derive_adapter'](old['CT_interface.f']).encode('ascii') == data('source/CT_interface.f') and
            namespace['derive_driver'](old['driver-reference.f']).encode('ascii') == data('source/driver-reference.f'),
            'fresh source transformation differs')
    card = parent_data('proposal/inputs/cms-backend1-mode4-MadLoopParams.dat')
    require(pin(card) == identity(bindings['card']), 'card donor binding differs')
    for backend in (6, 8):
        require(namespace['card'](card.decode('ascii'), backend).encode('ascii') ==
                data('source/backend' + str(backend) + '-mode4-MadLoopParams.dat'), 'backend card transform differs')
    return old


def verify_build(data, document, parent_data, parent_document):
    """Verify only the explicit closed build and retained parent source ancestry."""
    source = json.loads(checked(data, 'source/index.json', SOURCE_SHA))
    require(len(source['files']) == source['payload_count'] == 29 and source['payload_bytes'] == 238242 and
            source['build_and_run_separate'] and source['native_actions_at_seal'] == 0, 'source scope differs')
    require(sum(value['bytes'] for value in source['files'].values()) == source['payload_bytes'], 'source bytes differ')
    for name, expected in source['files'].items():
        checked(data, 'source/' + name, expected)
    bindings = document('source/bindings.json')['files']
    parent_source = json.loads(checked(parent_data, 'proposal/index.json', PARENT_SOURCE_SHA))
    parent_closure = json.loads(checked(parent_data, 'results/build-closure.json', PARENT_BUILD_SHA))
    donor_raw = checked(parent_data, 'results/builds/original/build.json', identity(bindings['original_build']))
    donor = json.loads(donor_raw)
    require(pin(donor_raw) == parent_closure['payloads']['builds/original/build.json'], 'donor receipt not in parent closure')
    parent_bindings = {
        'source_index': 'proposal/index.json', 'build_closure': 'results/build-closure.json',
        'driver': 'proposal/driver-amplitude.f', 'card': 'proposal/inputs/cms-backend1-mode4-MadLoopParams.dat',
        'old_build_helper': 'proposal/build_stage.py', 'old_comparator': 'proposal/amplitude_compare.py',
        'old_controller': 'proposal/run.py', 'ordinary_closure': 'results/ordinary-closure.json',
        'ordinary_comparison': 'results/ordinary-comparison.json', 'stress_closure': 'results/stress-closure.json',
        'stress_terminal': 'results/stress-terminal-review.json', 'point20_comparison': 'results/root-point20-comparison.json',
        'physical_card': 'proposal/inputs/cms-param_card.dat', 'point1': 'proposal/inputs/cms-point1.dat',
        'point20': 'proposal/inputs/cms-point20.dat',
    }
    for key, logical in parent_bindings.items():
        require(pin(parent_data(logical)) == identity(bindings[key]), 'parent ancestry differs: ' + key)
    require(len(parent_source['files']) == 55 and parent_closure['passed'], 'parent build scope differs')
    for name in ('cms-param_card.dat', 'cms-point1.dat', 'cms-point20.dat'):
        require(data('source/inputs/' + name) == parent_data('proposal/inputs/' + name), 'physical input differs')
    old = replay_sources(data, parent_data, parent_document, bindings, donor)
    for key in ('CT_interface', 'loop_matrix'):
        require(donor['sources'][bindings[key]['path']] == identity(bindings[key]),
                'recorded donor source binding differs: ' + key)
    unmaterialized = sorted(set(bindings) - set(parent_bindings) -
                            {'original_build', 'CT_interface', 'loop_matrix'})
    require(unmaterialized == ['conditional_geometry', 'conditional_geometry_index',
            'ninjars_dependency_provenance', 'ninjars_fortran_source',
            'ninjars_install_metadata', 'universal_double_pole_index'],
            'unmaterialized historical source scope differs')

    closure = json.loads(checked(data, 'results/build-closure.json', BUILD_SHA))
    require(closure['source_index_sha256'] == SOURCE_SHA and closure['stage'] == 'build' and closure['passed'] and
            closure['failure'] is None and closure['postcheck_failure'] is None and closure['reviewed_build_sha256'] is None and
            not closure['host_stability_qualified'] and not closure['performance_measurement'] and
            len(closure['payloads']) == 138 and closure['owned_command_count'] == 64 and closure['loader_query_count'] == 1,
            'closed build scope differs')
    commands = {}
    for name, expected in closure['payloads'].items():
        raw = checked(data, 'results/' + name, expected)
        if name.startswith('execution/') and name.endswith('.json'):
            command = json.loads(raw)
            require(command['returncode'] == 0 and command['exception'] is None and command['cleanup_exception'] is None and
                    command['cleanup']['survivors'] == [], 'recorded command/cleanup failed')
            log = name[:-5] + '.txt'
            require(log in closure['payloads'] and pin(data('results/' + log))['sha256'] == command['log_sha256'],
                    'command/log join differs')
            require(command['cpu'] == 12 and command['cpu_exclusive'] is False and command['argv'][:3] == ['taskset', '-c', '12'],
                    'recorded command placement differs')
            commands[name] = command
    require(len(commands) == 64 and sum('--list' in command['argv'] for command in commands.values()) == 1,
            'command count differs')
    require(data('results/build-before.json') == data('results/build-after.json'), 'protected state changed')
    protected = document('results/build-before.json')
    require(protected['source_index'] == pin(data('source/index.json')) and protected['bindings'] == pin(data('source/bindings.json')) and
            protected['upstream'] == parent_document('results/build-before.json'), 'protected parent identity differs')
    before = document('results/builds/reference/before-compile.json')
    attempt = document('results/builds/reference/attempt.json')
    build = document('results/builds/reference/build.json')
    link = document('results/builds/reference/link-variant-receipt.json')
    prelink = document('results/builds/reference/prelink.json')
    runtime = document('results/runtime/reference.json')
    summary = document('results/build-summary.json')
    require(summary['reference_build'] == pin(data('results/builds/reference/build.json')) and summary['executables'] == 1 and
            summary['actual_make_jobs'] == 1 and summary['native_main_calls'] == summary['tensor_calls'] == 0 and
            not summary['scientific_library_rebuilds'] and not summary['numerical_expressions_changed'], 'build summary differs')
    require(build['case'] == link['case'] == 'cms' and build['variant'] == link['variant'] == 'reference' and
            build['output'] == link['output'] == OUTPUT and build['compiled_targets'] == summary['compiled_objects'] == ['check_sa.o', 'CT_interface.o'],
            'build target scope differs')
    require(build['fortran_main_calls'] == build['tensor_evaluations'] == 0 and not build['library_rebuilds'] and
            not build['independent_accuracy_qualified'] and not build['performance_measurement'] and
            build['unmodified_numerical_arithmetic'] is link['unmodified_numerical_arithmetic'] is False,
            'build execution/source-change scope differs')
    require(build['sources'] == before['sources'] == attempt['sources_before'] == link['sources'], 'source inventory changed during build')
    current, previous = source_inventory(build), source_inventory(donor)
    require(len(current) == 269 and set(current) == set(previous) and
            {name for name in current if current[name] != previous[name]} == {SUB + name for name in ('CT_interface.f', 'check_sa.f', 'makefile')},
            'source change set differs')
    require(current[SUB + 'CT_interface.f'] == pin(data('source/CT_interface.f')) and
            current[SUB + 'check_sa.f'] == pin(data('source/driver-reference.f')) and
            previous[SUB + 'CT_interface.f'] == pin(old['CT_interface.f'].encode('ascii')) and
            previous[SUB + 'check_sa.f'] == pin(old['driver-reference.f'].encode('ascii')), 'built source ancestry differs')
    require(build['physical_card'] == link['physical_card'] == pin(data('source/inputs/cms-param_card.dat')), 'built physical card differs')
    require(attempt['objects_before'] == donor['all_objects_after'] and len(build['all_objects_after']) == 207 and
            set(build['all_objects_after']) == set(attempt['objects_before']), 'object clone inventory differs')
    changed = {SUB + name for name in ('check_sa.o', 'CT_interface.o')}
    require({name for name in build['all_objects_after'] if build['all_objects_after'][name] != attempt['objects_before'][name]} == changed and
            len(build['preserved_object_hashes']) == 205 and
            build['preserved_object_hashes'] == {name: value for name, value in attempt['objects_before'].items() if name not in changed},
            'preserved object identity differs')
    require(build['objects'] == link['objects'] and len(build['objects']) == 19 and
            all(build['all_objects_after'][name]['sha256'] == value for name, value in build['objects'].items()), 'link object inventory differs')
    require(build['reused_libraries'] == attempt['reused_libraries_before'] == donor['reused_libraries'] and
            len(build['reused_libraries']) == 4, 'archive identity differs')
    require(build['module_inputs'] == build['modules_after'] == donor['module_inputs'] and len(build['module_inputs']) == 7,
            'module identity differs')
    for name, value in before['module_inputs'].items():
        if name.startswith(OUTPUT + '/'):
            require(build['module_inputs'][suffix(name, OUTPUT + '/')] == value, 'module admission differs')
        else:
            require(name == ROOT + '.tirust-build/kinematic-provenance-current-executor-1/work/mg-current/vendor/CutTools/includects/mpmodule.mod' and
                    build['module_inputs'][SUB + 'mpmodule.mod'] == build['module_inputs']['lib/mpmodule.mod'] == value and
                    prelink['bound'].get(name) == value['sha256'], 'CutTools module alias differs')
    require(build['compiler_flags'] == before['compiler_flags'] == link['compiler_flags'], 'compiler flag snapshots differ')
    require(build['build_commands'] == attempt['commands'] and len(build['build_commands']) == 5 and attempt['failure'] is None and
            attempt['compile_seen'] and attempt['link_seen'] and attempt['runtime_seen'] and
            attempt['fortran_main_calls'] == attempt['tensor_evaluations'] == 0 and
            build['c_dependencies'] is attempt['c_dependencies'] is None, 'build attempt incomplete')
    transforms = {entry['label']: entry for entry in summary['inherited_make_job_requests_and_actual_transform']}
    require(set(transforms) == {'compile-cms-reference', 'link-cms-reference'}, 'make transform set differs')
    for request in build['build_commands']:
        command = commands['execution/build-' + request['label'] + '.json']
        actual = ['-j1' if arg == '-j2' else arg for arg in request['argv']]
        # The inherited owned runner appends the recorded make shell after
        # the proposal's -j2 to -j1 transformation. This is an argv join,
        # not a fresh external shell-binary identity check.
        tail = ['SHELL=/nix/store/1mv3qz005gkbalxfghk4y3s8ayv4d2dl-bash-interactive-5.3p15/bin/bash'] if actual[0] == 'make' else []
        require(command['argv'] == ['taskset', '-c', '12'] + actual + tail and
                command['cwd'] == request['cwd'], 'executed build request differs')
        if request['label'] in transforms:
            require(transforms[request['label']]['inherited_request'] == request['argv'] and
                    transforms[request['label']]['actual_argv'] == actual and '-j1' in actual, 'actual one-job override differs')
    require(prelink['completed_before_link'] and len(prelink['bound']) == 147 and
            prelink['bound'] == build['prelink_bound'] == link['prelink_bound'] and
            all(prelink['bound'].get(name) == value for name, value in prelink['closure']['files'].items()), 'prelink binding differs')
    require(build['actual_link_loads'] == link['actual_link_loads'] and len(build['actual_link_loads']) == 52 and
            all(prelink['bound'].get(name) == value['sha256'] for name, value in build['actual_link_loads'].items()),
            'recorded LOAD identity is not prebound')
    require(build['runtime'] == link['runtime'] == runtime['runtime'] and runtime['passed'] and runtime['failure'] is None and
            runtime['loader_query_started'] and runtime['actual_interpreter'] == build['runtime']['loader'] and
            build['runtime']['files'] == donor['runtime']['files'] and len(build['runtime']['files']) == 15,
            'runtime provider/admission differs')
    require(build['executable'] == link['executable'] == runtime['executable'] == summary['executable'] == OUTPUT + '/' + SUB + 'check' and
            build['executable_pin'] == link['executable_pin'] == runtime['executable_pin'] == summary['executable_pin'], 'executable identity differs')
    provider_extra = parent_document('proposal/bindings.json')['fixed_supplemental_providers']
    allowed = {**prelink['bound'], **{name: value['sha256'] for name, value in provider_extra.items()}}
    require(all(allowed.get(name) == value for name, value in build['runtime']['files'].items()) and
            build['runtime']['files'][build['runtime']['loader']] == build['runtime']['loader_sha256'], 'runtime provider was not admitted')
    loader_commands = [(name, command) for name, command in commands.items() if '--list' in command['argv']]
    name, loader = loader_commands[0]
    require(loader['argv'] == ['taskset', '-c', '12', build['runtime']['loader'], '--list', build['executable']] and
            data('results/' + name[:-5] + '.txt').decode('utf-8') == build['runtime']['loader_listing'], 'loader raw-output join differs')
    return {'source_payloads': 29, 'closed_build_payloads': 138, 'owned_commands': 64,
            'loader_queries': 1, 'recorded_ELFs': 1, 'recorded_sources': 269,
            'fresh_two_file_transform_replay': True, 'fresh_269_source_DAG_reparse': False,
            'compiled_objects': ['check_sa.o', 'CT_interface.o'], 'preserved_objects': 205,
            'existing_modules_preserved': 7, 'reused_archives': 4,
            'prelink_dependencies': 147, 'recorded_LOADs': 52, 'runtime_providers': 15,
            'new_native_executions': 0, 'external_binary_reads': 0,
            'unmaterialized_historical_source_bindings': unmaterialized,
            'fresh_external_artifact_proof': False, 'amplitude_accuracy_qualified': False,
            'performance_qualified': False}
