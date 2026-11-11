"""Pure matched-input source/build receipt joins; never opens external artifacts.

All input bytes come from authenticated portable-packet callbacks. The sealed
scientific text transformations are freshly replayed; source-generation/DAG
inventories and native artifact identities remain historical evidence.
"""
import ast
import difflib
import hashlib
import json
import re

SCIENCE = '357e116b51879cda1e1145c9d040194732c21ffd9fcd2557d9f68451901b262b'
CONTROLLER = '9076a51a0be56c57fb7dda32cfec3ef9b038bec2e73dc5e751bf31dc5ac3bc41'
OLD_CONTROLLER = '9ff20966e4d24a3062a730a6037e9c9e50ab22d0007f343fdf6b4cee84cfaae7'
FAILURE = 'd0343e3ec8f8b4db046a5fc3ac7c0dd2cace9c5a271fffb00394e1e7ca16af41'
FAILURE_TERMINAL = '8e6c66cce304c7496fc341cfc8f9167f9a6a75e548d148b5fa972a84e2cf041b'
BUILD = 'c09358a689b2b856b81580a7d0371b73a3d13edb3a173e9b8aed4b8f143ad00d'
AMPLITUDE = 'ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74'
LEGACY = '0a0dba9baf12ff5a988e4d459141a48140f285a396658ee10f7f09099848dbd8'
ROOT = '/shared/external/madloop_tirust/madgraph7_madloop_tirust/'
WORK = '.tirust-build/kinematic-matched-native-input-executor-2/'
OUTPUT = ROOT + WORK + 'work/builds/cms/matched'
SUB = 'SubProcesses/P0_epem_mupmum/'
SIX = ('CT_interface.f', 'TIRUST_interface.f', 'TIR_interface.f',
       'loop_matrix.f', 'loop_CT_calls_1.f', 'mp_compute_loop_coefs.f')


def require(value, message):
    if not value:
        raise ValueError('Closed matched build: ' + message)


def pin(raw):
    require(isinstance(raw, bytes), 'byte callback must return bytes')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def identity(value):
    return {name: value[name] for name in ('bytes', 'sha256')}


def checked(data, name, expected):
    raw = data(name)
    actual = pin(raw)
    require(actual['sha256'] == (expected if isinstance(expected, str) else expected['sha256']) and
            (isinstance(expected, str) or actual['bytes'] == expected['bytes']), 'changed payload ' + name)
    return raw


def seal(data, prefix, checksum, count):
    result = json.loads(checked(data, prefix + '/index.json', checksum))
    require(len(result['files']) == count, 'sealed source count differs')
    for name, expected in result['files'].items():
        checked(data, prefix + '/' + name, expected)
    return result


def suffix(name, prefix):
    require(name.startswith(prefix), 'foreign recorded path')
    result = name[len(prefix):]
    require(result and all(part not in ('', '.', '..') for part in result.split('/')), 'unsafe recorded suffix')
    return result


def source_inventory(build):
    prefix = suffix(build['output'], ROOT) + '/'
    return {suffix(name, prefix): value for name, value in build['sources'].items()}


def pure_definitions(raw, functions, constants=()):
    selected, found, values = [], set(), {}
    for node in ast.parse(raw).body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            require(not node.decorator_list and not node.args.defaults and not any(node.args.kw_defaults) and
                    node.returns is None and not any(arg.annotation for arg in node.args.args),
                    'definition-time expression in pure transform')
            selected.append(node); found.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants:
            value = ast.literal_eval(node.value)
            require(isinstance(value, str), 'transform constant must be literal text')
            values[node.targets[0].id] = value
    require(found == set(functions) and set(values) == set(constants), 'pure source function selection differs')
    namespace = {'__builtins__': {'ValueError': ValueError}, 're': re, **values}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'sealed-matched-source-transform', 'exec'), namespace)
    return namespace


def replay_science(data, amplitude_data, legacy_data):
    amplitude = json.loads(checked(amplitude_data, 'proposal/index.json', AMPLITUDE))
    legacy = json.loads(checked(legacy_data, 'source/index.json', LEGACY))
    donors = {name: checked(amplitude_data, 'proposal/candidate/' + name,
        amplitude['files']['candidate/' + name]).decode('ascii') for name in SIX}
    driver = checked(legacy_data, 'source/driver-reference.f', legacy['files']['driver-reference.f']).decode('ascii')
    card = checked(legacy_data, 'source/backend8-mode4-MadLoopParams.dat',
                   legacy['files']['backend8-mode4-MadLoopParams.dat']).decode('ascii')
    namespace = pure_definitions(data('science/prepare.py'),
        {'require', 'once', 'sub', 'replace_sub', 'private_nrs', 'route8', 'trace9', 'derive_driver'}, {'USE', 'DECL'})
    outputs = dict(donors)
    outputs['CT_interface.f'] = namespace['route8'](donors['CT_interface.f']) + '\n' + namespace['private_nrs'](donors['CT_interface.f']) + '\n'
    outputs['TIRUST_interface.f'] = namespace['trace9'](donors['TIRUST_interface.f'])
    products = {'candidate/' + name: text for name, text in outputs.items()}
    for name in ('CT_interface.f', 'TIRUST_interface.f'):
        products[name + '.patch'] = ''.join(difflib.unified_diff(donors[name].splitlines(True), outputs[name].splitlines(True),
            fromfile='sealed-candidate2/' + name, tofile='private-matched/' + name))
    products['driver-matched.f'] = namespace['derive_driver'](driver)
    products['driver-matched.f.patch'] = ''.join(difflib.unified_diff(driver.splitlines(True),
        products['driver-matched.f'].splitlines(True), fromfile='sealed-reference/driver-reference.f',
        tofile='private-matched/driver-matched.f'))
    products['backend8-mode4-MadLoopParams.dat'] = card
    products['backend9-mode4-MadLoopParams.dat'] = namespace['once'](card, '#MLReductionLib\n8\n', '#MLReductionLib\n9\n')
    require(len(products) == 12, 'derived product count differs')
    for name, text in products.items():
        require(text.encode('ascii') == data('science/' + name), 'fresh scientific source derivation differs: ' + name)
    return products


def verify_source_proof(data, document, amplitude_data, amplitude_document):
    """Join the historical preflight; replay only retained source text proofs."""
    preflight = document('science/audit/final-source-preflight.json')
    proof = document('science/source-proof.json')
    review = document('science/audit/candidate-source-review.json')
    require(preflight['schema'] == 'private-matched-input-source-preflight-v1' and preflight['passed'] and
            preflight['sealed_index'] is None and preflight['bound_files'] == 17 and preflight['generated_products'] == 12 and
            preflight['routes'] == 26 and preflight['native_actions'] == 0 and
            not any(preflight[key] for key in ('accuracy_qualified', 'compiled', 'native_controller_present', 'physics_evaluated')),
            'historical preflight scope differs')
    source_pins = None
    for node in ast.parse(data('science/matched_inputs.py')).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'SOURCE_PINS' for target in node.targets):
            require(source_pins is None, 'duplicate source-contract constant')
            source_pins = ast.literal_eval(node.value)
    contract = preflight['source_contract_pins']
    require(source_pins == {name: value['sha256'] for name, value in contract.items()} and len(contract) == 5,
            'retained decoder/preflight source-contract pins differ')
    materialized = '.tirust-build/kinematic-invariant-amplitude-proposal-2/'
    polynomial = '.tirust-build/kinematic-provenance-current-executor-1/work/exports/cms/' + SUB + 'polynomial.f'
    donor = amplitude_document('results/builds/candidate/build.json')
    for name, expected in contract.items():
        if name.startswith(materialized):
            require(pin(amplitude_data('proposal/' + suffix(name, materialized))) == expected,
                    'materialized source-contract dependency differs')
        else:
            require(name == polynomial and source_inventory(donor)[SUB + 'polynomial.f'] == expected,
                    'historical polynomial inventory identity differs')
    require(proof['source_only'] and proof['mode'] == 4 and proof['native_actions'] == 0 and
            proof['derived_products_exact'] == 12 and proof['source_table_routes'] == 26 and
            proof['reviewed_file_pins_reauthenticated'] == 17 and proof['old_candidate_source_index'] == AMPLITUDE and
            proof['old_reference_source_index'] == LEGACY and proof['independent_review'] ==
            pin(data('science/audit/candidate-source-review.json')) and
            not any(proof[key] for key in ('compiled', 'native_controller_present', 'physics_evaluated')),
            'historical source-proof scope/parent joins differ')
    require(len(review['pins']) == 17 and review['sealed_candidate2_index'] == AMPLITUDE and
            review['sealed_reference_driver_index'] == LEGACY and review['derived_products_exact'] == 12,
            'independent source review identity differs')
    for name, expected in review['pins'].items():
        require(pin(data('science/' + name)) == expected, 'historical independent review payload pin differs')
    extract = pure_definitions(data('science/prepare.py'), {'require', 'sub'})['sub']
    oldct = amplitude_data('proposal/candidate/CT_interface.f').decode('ascii')
    oldtir = amplitude_data('proposal/candidate/TIRUST_interface.f').decode('ascii')
    newct = data('science/candidate/CT_interface.f').decode('ascii')
    newtir = data('science/candidate/TIRUST_interface.f').decode('ascii')
    routines = proof['legacy_numerics_preserved']
    require(routines == review['byte_preserved_subroutines'] and len(routines) == 10 and all(routines.values()),
            'preserved routine claim set differs')
    for name in routines:
        old, new = (oldtir, newtir) if 'TIRUST' in name else (oldct, newct)
        require(extract(old, name) == extract(new, name), 'retained original routine differs: ' + name)
    unchanged = ['TIR_interface.f', 'loop_matrix.f', 'loop_CT_calls_1.f', 'mp_compute_loop_coefs.f']
    require(proof['unchanged_candidate2_files'] == unchanged and
            all(data('science/candidate/' + name) == amplitude_data('proposal/candidate/' + name) for name in unchanged),
            'producer source changed')
    old = extract(oldtir, 'ML5_0_MP_PRIVATE_TIRUST_DIAGNOSTIC')
    new = extract(newtir, 'ML5_0_MATCHED_MP_TIRUST_DIAGNOSTIC')
    start = '      STABLE=.FALSE.'
    require(old.count(start) == new.count(start) == 1, 'diagnostic executable boundary differs')
    new = new[new.index(start):]
    new, begins = re.subn(r'      CALL PC_MATCH_BEGIN\(.*?\n(?=      CALL TIRUST_TENSOR_EVALUATE)', '', new, flags=re.S)
    new, ends = re.subn(r'      CALL PC_MATCH_END\(.*?\n(?=      IF\(STATUS.NE.TIRUST_SUCCESS\))', '', new, flags=re.S)
    require(begins == ends == 1 and new.rstrip() == old[old.index(start):].rstrip(),
            'recorder-stripped diagnostic arithmetic differs')
    body_hash = hashlib.sha256(new.rstrip().encode('ascii')).hexdigest()
    require(body_hash == 'd5b6e71ed15d6e56c54dcc2e3304f96afeb1e5f44f3e7763612d1ea89f02e35f',
            'diagnostic executable body identity differs')
    old = extract(oldtir, 'ML5_0_PRIVATE_MP_TIRUST_LOOP')
    new = extract(newtir, 'ML5_0_PRIVATE_MP_TIRUST_LOOP')
    anchor = '      IF(.NOT.STABLE) THEN'
    require(old.count(anchor) == new.count(anchor) == 2 and old[old.index(anchor):] == new[new.index(anchor):],
            'failure/contraction/normalization tail changed')
    caller = data('science/candidate/loop_CT_calls_1.f').decode('ascii')
    fixed = ' '.join(line[6:].strip() for line in caller.splitlines() if len(line) >= 6 and line[0] not in 'Cc*!')
    require(len(re.findall(r'\bNSQUAREDSO\s*=\s*1\b', fixed)) == 1, 'declared squared-order domain differs')
    routes = {}
    for call in re.finditer(r'\bCALL\s+ML5_0_LOOP_(1_2|2|3|4)\s*\(', fixed):
        depth, start, args, end = 0, call.end(), [], None
        for position in range(call.end(), len(fixed)):
            char = fixed[position]
            if char == '(':
                depth += 1
            elif char == ')':
                if depth == 0:
                    args.append(fixed[start:position].strip()); end = position; break
                depth -= 1
            elif char == ',' and depth == 0:
                args.append(fixed[start:position].strip()); start = position + 1
        require(end is not None and args[-2:] == ['PC_ENV', 'PC_PRODUCER_OK'], 'generated call syntax/provenance differs')
        wrapper = call.group(1)
        if wrapper == '1_2':
            require(len(args) == 9 and args[0] == '2' and args[5] == 'I_SO', 'single-propagator wrapper differs')
            n, nw, waves, rank, group, pairing = 1, 2, args[1:3], args[4], args[6], [2]
        else:
            n = int(wrapper)
            require(len(args) == 2 * n + 5 and args[2 * n + 1] == 'I_SO', 'generated wrapper argument layout differs')
            nw, waves, rank, group, pairing = n, args[:n], args[2 * n], args[2 * n + 2], [1] * n
        require(all(re.fullmatch(r'[0-9]+', value) for value in waves + [rank, group]) and group not in routes,
                'nonliteral or duplicate generated route')
        routes[group] = {'n': n, 'nw': nw, 'we': [int(value) for value in waves], 'rank': int(rank),
                         'sqso': 1, 'pairing': pairing}
    require(set(routes) == {str(i) for i in range(1, 27)} and routes == document('science/expected-routes.json'),
            'fresh caller-derived route table differs')
    ordering = extract(data('science/candidate/TIR_interface.f').decode('ascii'), 'ML5_0_MP_SWITCH_ORDER')
    require('      IF (CTMODE.NE.2.AND.CTMODE.NE.5) THEN\n        RETURN\n      ENDIF\n' in ordering,
            'mode4 ordering early-return differs')
    return {'preflight_historical_sealed_index': None, 'materialized_source_contract_pins': 4,
            'historical_polynomial_inventory_pin': contract[polynomial], 'preserved_routines_rechecked': 10,
            'diagnostic_body_sha256': body_hash, 'diagnostic_recorder_calls_removed': 2,
            'postdiagnostic_contraction_tail_unchanged': True, 'fresh_caller_route_rows': len(routes),
            'source_sqso_is_declared_domain_not_runtime_observation': True,
            'external_preflight_reexecuted': False}


def closed_stage(data, prefix, checksum, source, passed, payloads, command_count, loaders):
    closure = json.loads(checked(data, prefix + '/build-closure.json', checksum))
    require(closure['source_index_sha256'] == source and closure['stage'] == 'build' and
            closure['passed'] is passed and closure['postcheck_failure'] is None and closure['reviewed_build_sha256'] is None and
            not closure['host_stability_qualified'] and not closure['performance_measurement'] and
            len(closure['payloads']) == payloads and closure['owned_command_count'] == command_count and
            closure['loader_query_count'] == loaders, 'closed build scope differs')
    commands = {}
    for name, expected in closure['payloads'].items():
        raw = checked(data, prefix + '/' + name, expected)
        if name.startswith('execution/') and name.endswith('.json'):
            command = json.loads(raw)
            require(command['returncode'] == 0 and command['exception'] is None and command['cleanup_exception'] is None and
                    command['cleanup']['survivors'] == [], 'recorded command/cleanup failed')
            log = name[:-5] + '.txt'
            require(log in closure['payloads'] and pin(data(prefix + '/' + log))['sha256'] == command['log_sha256'], 'command/log join differs')
            require(command['cpu'] == 12 and command['cpu_exclusive'] is False and command['argv'][:3] == ['taskset', '-c', '12'],
                    'recorded placement differs')
            commands[name] = command
    require(len(commands) == command_count and sum('--list' in value['argv'] for value in commands.values()) == loaders,
            'owned command count differs')
    require(data(prefix + '/build-before.json') == data(prefix + '/build-after.json'), 'protected state changed')
    return closure, commands


def verify_build(data, document, amplitude_data, amplitude_document, legacy_data, legacy_document):
    """Authenticate sources, failed predecessor, successor and closed build joins."""
    science = seal(data, 'science', SCIENCE, 35)
    controller = seal(data, 'controller', CONTROLLER, 31)
    previous = seal(data, 'predecessor/controller', OLD_CONTROLLER, 27)
    require(science['source_only'] and not science['compiled'] and not science['native_admitted'] and not science['accuracy_qualified'],
            'historical scientific seal scope differs')
    for index in (controller, previous):
        require(index['scientific_source_sha256'] == SCIENCE and index['native_executions_at_seal'] == 0 and
                index['build_and_run_are_separate'] and not index['accuracy_qualified'], 'controller source scope differs')
    require(controller['previous_controller_sha256'] == OLD_CONTROLLER and controller['previous_failed_build_sha256'] == FAILURE,
            'successor ancestry differs')
    proof = document('controller/successor-change-proof.json')
    require(proof['previous_index'] == pin(data('predecessor/controller/index.json')) and proof['scientific_index_sha256'] == SCIENCE and
            proof['numerical_sources_reused_from_same_immutable_packet'] and not proof['scientific_source_changes'] and
            not proof['reduction_libraries_rebuilt'] and proof['actual_native_commands_so_far'] == 0, 'successor scientific change scope differs')
    common = {name for name in previous['files'] if name in controller['files'] and not name.startswith('audit/')}
    same = set(proof['unchanged_predecessor_payloads']); changed = set(proof['changed_predecessor_payloads'])
    require(len(same) == 10 and len(changed) == 7 and same.isdisjoint(changed) and same | changed == common,
            'successor payload partition differs')
    for name in same:
        require(data('controller/' + name) == data('predecessor/controller/' + name), 'unchanged controller payload differs')
    for name in changed:
        old = data('predecessor/controller/' + name).decode('utf-8')
        new = data('controller/' + name).decode('utf-8')
        require(old != new, 'declared changed controller is unchanged')
        patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
            fromfile='sealed-controller1/' + name, tofile='successor-controller2/' + name))
        require(patch.encode('utf-8') == data('controller/' + name + '.successor.patch'), 'controller canonical delta differs')
    products = replay_science(data, amplitude_data, legacy_data)
    source_proof = verify_source_proof(data, document, amplitude_data, amplitude_document)
    for name in ('backend8-mode4-MadLoopParams.dat', 'backend9-mode4-MadLoopParams.dat',
                 'inputs/cms-param_card.dat', 'inputs/cms-point1.dat', 'inputs/cms-point20.dat'):
        require(data('controller/' + name) == data('predecessor/controller/' + name) == data('science/' + name), 'scientific trial input changed')
    for name in ('limit_exec.py', 'native_job.py'):
        require(data('controller/' + name) == legacy_data('source/' + name), 'retained ownership helper changed')
    for version in (1, 2):
        prefix = 'predecessor/controller/' if version == 1 else 'controller/'
        expected = legacy_data('source/worker_entry.py').decode('ascii').replace(
            "root=ROOT/'.tirust-build/kinematic-invariant-reference-adjudication-executor-1'",
            "root=ROOT/'.tirust-build/kinematic-matched-native-input-executor-" + str(version) + "'")
        require(expected.encode('ascii') == data(prefix + 'worker_entry.py'), 'worker ownership/root derivative differs')
    bindings = document('controller/bindings.json')['files']
    ancestry = {
        'scientific_source': (data, 'science/index.json'),
        'previous_controller': (data, 'predecessor/controller/index.json'),
        'previous_build_closure': (data, 'predecessor/results/build-closure.json'),
        'previous_terminal': (data, 'predecessor/results/build-terminal-review.json'),
        'reference_controller': (legacy_data, 'source/index.json'),
        'amplitude_source': (amplitude_data, 'proposal/index.json'),
        'candidate_build': (amplitude_data, 'results/builds/candidate/build.json'),
        'candidate_build_closure': (amplitude_data, 'results/build-closure.json'),
        'ordinary_closure': (amplitude_data, 'results/ordinary-closure.json'),
        'ordinary_comparison': (amplitude_data, 'results/ordinary-comparison.json'),
    }
    require(set(bindings) == set(ancestry), 'controller source ancestry set differs')
    for name, (reader, logical) in ancestry.items():
        require(pin(reader(logical)) == identity(bindings[name]), 'retained source ancestry differs: ' + name)
    science_bindings = document('science/bindings.json')['files']
    for name, (reader, logical) in {
        'candidate2_source': (amplitude_data, 'proposal/index.json'),
        'candidate2_build': (amplitude_data, 'results/builds/candidate/build.json'),
        'candidate2_build_closure': (amplitude_data, 'results/build-closure.json'),
        'candidate2_ordinary': (amplitude_data, 'results/ordinary-closure.json'),
        'reference_source': (legacy_data, 'source/index.json'),
        'legacy_reference_run': (legacy_data, 'results/run-closure.json'),
    }.items():
        require(pin(reader(logical)) == identity(science_bindings[name]), 'scientific source parent pin differs')
    unmaterialized = sorted(set(science_bindings) - {'candidate2_source', 'candidate2_build', 'candidate2_build_closure',
        'candidate2_ordinary', 'reference_source', 'legacy_reference_run'})
    require(len(unmaterialized) == 11, 'external historical scientific dependency scope differs')

    failed, failed_commands = closed_stage(data, 'predecessor/results', FAILURE, OLD_CONTROLLER, False, 8, 2, 0)
    reason = 'ValueError("Unresolved numerical compilation recipes: [\'amplitude_support.f90\', \'caller_bounds.f90\', \'certificate.f90\']")'
    require(failed['failure'] == reason, 'initial recipe-audit failure differs')
    terminal = json.loads(checked(data, 'predecessor/results/build-terminal-review.json', FAILURE_TERMINAL))
    attempt1 = document('predecessor/results/builds/matched/attempt.json')
    require(attempt1['failure'] == reason and attempt1['commands'] == [] and not attempt1['compile_seen'] and
            not attempt1['link_seen'] and not attempt1['runtime_seen'] and
            attempt1['fortran_main_calls'] == attempt1['tensor_evaluations'] == 0, 'first attempt scope differs')
    require(terminal['build_closure'] == pin(data('predecessor/results/build-closure.json')) and
            terminal['source_index'] == pin(data('predecessor/controller/index.json')) and terminal['failure'] == reason and
            terminal['attempt'] == pin(data('predecessor/results/builds/matched/attempt.json')) and
            terminal['closed_payloads_verified'] == 8 and terminal['source_payloads_verified'] == 27 and
            terminal['terminal_exit_code'] == 1 and not terminal['build_passed'] and
            not any(terminal[key] for key in ('compile_invoked', 'link_invoked', 'run_invoked', 'retry_invoked', 'accuracy_qualified')) and
            terminal['fortran_main_calls'] == terminal['tensor_calls'] == terminal['loader_queries'] == 0 and
            terminal['remaining_owned_identities'] == [] and terminal['shared_lock_free_after_terminal'] and
            terminal['protected_equal'] and terminal['protected_before'] == terminal['protected_after'] ==
            pin(data('predecessor/results/build-before.json')), 'first terminal scope/cleanup differs')
    require(len(terminal['commands']) == 2 and {row['path'] for row in terminal['commands']} == set(failed_commands),
            'first terminal command set differs')
    for row in terminal['commands']:
        command = failed_commands[row['path']]
        require(identity(row) == pin(data('predecessor/results/' + row['path'])) and row['argv'] == command['argv'] and
                row['returncode'] == command['returncode'] == 0, 'first command receipt join differs')
    first = failed_commands['execution/build-query-build-0001.json']['argv']
    second = failed_commands['execution/build-query-build-0002.json']['argv']
    require(first[3:7] == ['make', '--no-print-directory', '--dry-run', '--always-make'] and
            second[-1] == '--version', 'first attempt executed beyond recipe queries')

    closure, commands = closed_stage(data, 'results', BUILD, CONTROLLER, True, 138, 64, 1)
    require(closure['failure'] is None, 'successful build has a failure')
    protected = document('results/build-before.json')
    protected1 = document('predecessor/results/build-before.json')
    donor = amplitude_document('results/builds/candidate/build.json')
    donor_raw = amplitude_data('results/builds/candidate/build.json')
    parent_closure = amplitude_document('results/build-closure.json')
    require(pin(donor_raw) == parent_closure['payloads']['builds/candidate/build.json'] and parent_closure['passed'], 'donor build receipt not closed')
    for before, prefix in ((protected, 'controller'), (protected1, 'predecessor/controller')):
        require(before['source_index'] == pin(data(prefix + '/index.json')) and before['bindings'] == pin(data(prefix + '/bindings.json')) and
                before['scientific_source_index'] == pin(data('science/index.json')) and
                before['upstream'] == amplitude_document('results/build-before.json'), 'protected controller/science/ancestor pin differs')
        current = before['candidate_donor_current']
        require(current['receipt'] == pin(donor_raw) and current['executable'] == donor['executable_pin'] and
                current['current_files_verified'] and current['native_actions'] == 0 and
                (current['libraries'], current['modules'], current['objects'], current['prelink_inputs'], current['providers']) == (4, 10, 211, 151, 15),
                'historical donor admission inventory differs')
    before = document('results/builds/matched/before-compile.json')
    attempt = document('results/builds/matched/attempt.json')
    build = document('results/builds/matched/build.json')
    link = document('results/builds/matched/link-variant-receipt.json')
    prelink = document('results/builds/matched/prelink.json')
    runtime = document('results/runtime/matched.json')
    summary = document('results/build-summary.json')
    require(summary['matched_build'] == pin(data('results/builds/matched/build.json')) and summary['executables'] == 1 and
            summary['actual_make_jobs'] == 1 and summary['native_main_calls'] == summary['tensor_calls'] == 0 and
            not summary['scientific_library_rebuilds'] and summary['private8_geometry_policy_changed'] and
            summary['existing_modules_preserved'] and summary['scientific_source_index_sha256'] == SCIENCE, 'build summary differs')
    targets = ['check_sa.o', 'CT_interface.o', 'TIRUST_interface.o', 'matched_trace.o']
    require(build['case'] == link['case'] == 'cms' and build['variant'] == link['variant'] == 'matched' and
            build['output'] == link['output'] == OUTPUT and build['compiled_targets'] == summary['compiled_objects'] == targets,
            'build target scope differs')
    require(build['fortran_main_calls'] == build['tensor_evaluations'] == 0 and not build['library_rebuilds'] and
            not build['independent_accuracy_qualified'] and not build['performance_measurement'] and
            build['unmodified_numerical_arithmetic'] is link['unmodified_numerical_arithmetic'] is False, 'build scope differs')
    require(build['sources'] == before['sources'] == attempt['sources_before'] == link['sources'], 'source inventory changed during build')
    current, old = source_inventory(build), source_inventory(donor)
    require(len(current) == 275 and len(old) == 274 and set(current) - set(old) == {SUB + 'matched_trace.f90'} and
            set(old) <= set(current) and {name for name in old if current[name] != old[name]} ==
            {SUB + name for name in ('CT_interface.f', 'TIRUST_interface.f', 'check_sa.f', 'makefile')}, 'built source change set differs')
    for name in SIX:
        require(current[SUB + name] == pin(data('science/candidate/' + name)), 'built scientific source differs')
    require(current[SUB + 'check_sa.f'] == pin(data('science/driver-matched.f')) and
            current[SUB + 'matched_trace.f90'] == pin(data('science/matched_trace.f90')), 'built driver/recorder source differs')
    require(build['physical_card'] == link['physical_card'] == pin(data('science/inputs/cms-param_card.dat')), 'built physical card differs')
    require(attempt['objects_before'] == donor['all_objects_after'] and len(attempt['objects_before']) == 211 and
            len(build['all_objects_after']) == 212 and set(build['all_objects_after']) - set(attempt['objects_before']) == {SUB + 'matched_trace.o'},
            'object clone/addition differs')
    changed = {SUB + name for name in targets[:-1]}
    require({name for name in attempt['objects_before'] if attempt['objects_before'][name] != build['all_objects_after'][name]} == changed and
            len(build['preserved_object_hashes']) == 208 and
            build['preserved_object_hashes'] == {name: value for name, value in attempt['objects_before'].items() if name not in changed},
            'preserved object inventory differs')
    require(build['objects'] == link['objects'] and len(build['objects']) == 24 and
            all(build['all_objects_after'][name]['sha256'] == value for name, value in build['objects'].items()), 'link object inventory differs')
    require(build['reused_libraries'] == attempt['reused_libraries_before'] == donor['reused_libraries'] and
            len(build['reused_libraries']) == 4, 'reused archive identity differs')
    require(build['module_inputs'] == donor['modules_after'] and len(build['module_inputs']) == 10 and len(build['modules_after']) == 11 and
            set(build['modules_after']) - set(build['module_inputs']) == {SUB + 'ml_private_matched_inputs.mod'} and
            all(build['modules_after'][name] == value for name, value in build['module_inputs'].items()), 'existing module identity differs')
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
    transforms = {row['label']: row for row in summary['inherited_make_job_requests_and_actual_transform']}
    require(set(transforms) == {'compile-cms-matched', 'link-cms-matched'}, 'make transform set differs')
    for request in build['build_commands']:
        command = commands['execution/build-' + request['label'] + '.json']
        actual = ['-j1' if arg == '-j2' else arg for arg in request['argv']]
        tail = ['SHELL=/nix/store/1mv3qz005gkbalxfghk4y3s8ayv4d2dl-bash-interactive-5.3p15/bin/bash'] if actual[0] == 'make' else []
        require(command['argv'] == ['taskset', '-c', '12'] + actual + tail and command['cwd'] == request['cwd'],
                'executed build request differs')
        if request['label'] in transforms:
            require(transforms[request['label']]['inherited_request'] == request['argv'] and
                    transforms[request['label']]['actual_argv'] == actual and '-j1' in actual, 'actual one-job override differs')
    require(prelink['completed_before_link'] and len(prelink['bound']) == 152 and
            prelink['bound'] == build['prelink_bound'] == link['prelink_bound'] and
            all(prelink['bound'].get(name) == value for name, value in prelink['closure']['files'].items()), 'prelink binding differs')
    require(build['actual_link_loads'] == link['actual_link_loads'] and len(build['actual_link_loads']) == 57 and
            all(prelink['bound'].get(name) == value['sha256'] for name, value in build['actual_link_loads'].items()), 'recorded LOAD identity is not prebound')
    require(build['runtime'] == link['runtime'] == runtime['runtime'] and runtime['passed'] and runtime['failure'] is None and
            runtime['loader_query_started'] and runtime['actual_interpreter'] == build['runtime']['loader'] and
            build['runtime']['files'] == donor['runtime']['files'] and len(build['runtime']['files']) == 15, 'provider/admission differs')
    require(build['executable'] == link['executable'] == runtime['executable'] == summary['executable'] == OUTPUT + '/' + SUB + 'check' and
            build['executable_pin'] == link['executable_pin'] == runtime['executable_pin'] == summary['executable_pin'], 'executable receipt identity differs')
    supplementary = amplitude_document('proposal/bindings.json')['fixed_supplemental_providers']
    allowed = {**prelink['bound'], **{name: value['sha256'] for name, value in supplementary.items()}}
    require(all(allowed.get(name) == value for name, value in build['runtime']['files'].items()) and
            build['runtime']['files'][build['runtime']['loader']] == build['runtime']['loader_sha256'], 'runtime provider not admitted')
    loaders = [(name, row) for name, row in commands.items() if '--list' in row['argv']]
    name, loader = loaders[0]
    require(loader['argv'] == ['taskset', '-c', '12', build['runtime']['loader'], '--list', build['executable']] and
            data('results/' + name[:-5] + '.txt').decode('utf-8') == build['runtime']['loader_listing'], 'loader raw-output join differs')
    return {'scientific_payloads': 35, 'controller_payloads': 31, 'predecessor_controller_payloads': 27,
            'scientific_products_freshly_replayed': len(products), 'scientific_source_changed_by_successor': False,
            'source_proof': source_proof,
            'predecessor_failure_preserved': True, 'predecessor_query_commands': 2, 'predecessor_compile_calls': 0,
            'closed_build_payloads': 138, 'owned_build_commands': 64, 'loader_queries': 1, 'recorded_ELFs': 1,
            'compiled_objects': targets, 'recorded_source_files': 275, 'preserved_objects': 208,
            'existing_modules_preserved': 10, 'new_recorder_modules': 1, 'reused_archives': 4,
            'prelink_dependencies': 152, 'recorded_LOADs': 57, 'runtime_providers': 15,
            'unmaterialized_historical_scientific_bindings': unmaterialized,
            'fresh_269_source_DAG_reparse': False, 'fresh_external_artifact_proof': False,
            'new_native_executions': 0, 'external_binary_reads': 0, 'native_input_equality_qualified': False,
            'amplitude_accuracy_qualified': False, 'performance_qualified': False}
