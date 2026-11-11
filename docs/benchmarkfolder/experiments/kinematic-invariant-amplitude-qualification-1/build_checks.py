"""Pure joins for the closed private CMS build and its immutable source ancestry.

All bytes arrive through authenticated portable-packet callbacks. Recorded
binary identities are joined but never reopened. Six donor source texts are
recovered from reversible patches and replayed through the sealed pure source
transform; the full 269-source routing/DAG derivation remains historical.
"""
import ast
import difflib
import hashlib
import json
import re
import shlex

SOURCE_SHA = 'ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74'
PARENT_SHA = '286ecfe5f5fab199dee34e4401cb5154ba1212870eaf2b921d3f44536c251b0e'
BUILD_SHA = '8cab371e9affd5297c0f48c30184361959d20f3d3a48c6dbd6109b86bf49bacb'
TERMINAL_SHA = '9191a4513b79c5f55c0a34bde328c71516ad82b64ce0f81d098af2d41965b44d'
FAILURE_SHA = '99cbb544826b741c030708ce4ab5bc01db83eac0ec49c4dce02a6be6858d76aa'
DERIVED_SHA = '25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184'
FACTORY_SHA = 'aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047'
KERNEL_SHA = '1665eff3188c947f35673b958cd0f02525fd031dc814b708dfdda69160b57964'
ROOT = '/shared/external/madloop_tirust/madgraph7_madloop_tirust/'
WORK = '.tirust-build/kinematic-invariant-amplitude-executor-2/'
SUB = 'SubProcesses/P0_epem_mupmum/'
SIX = ('CT_interface.f', 'TIRUST_interface.f', 'TIR_interface.f',
       'loop_CT_calls_1.f', 'loop_matrix.f', 'mp_compute_loop_coefs.f')
SUPPORT = ('amplitude_support.f90', 'caller_bounds.f90', 'certificate.f90',
           'cms_bounds_data.inc', 'environment_token.c')


def require(value, message):
    if not value:
        raise ValueError('Closed build: ' + message)


def pin(raw):
    require(isinstance(raw, bytes), 'byte callback did not return bytes')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def checked(data, name, expected):
    raw = data(name)
    identity = pin(raw)
    require(identity['sha256'] == (expected if isinstance(expected, str) else expected['sha256']) and
            (isinstance(expected, str) or identity['bytes'] == expected['bytes']), 'payload differs: ' + name)
    return raw


def strip_prefix(name, prefix):
    require(name.startswith(prefix), 'foreign recorded path: ' + name)
    value = name[len(prefix):]
    require(value and not any(part in ('', '.', '..') for part in value.split('/')), 'unsafe recorded suffix')
    return value


def relative_sources(build):
    prefix = strip_prefix(build['output'], ROOT) + '/'
    return {strip_prefix(name, prefix): identity for name, identity in build['sources'].items()}


def reverse_patch(candidate, patch, name):
    """Reverse canonical unified hunks in memory; reject fuzzy or partial application."""
    current = candidate.splitlines(keepends=True)
    lines = patch.splitlines(keepends=True)
    require(lines[:2] == ['--- original/' + name + '\n', '+++ candidate/' + name + '\n'], 'patch headers differ')
    result, position, cursor = [], 0, 2
    while cursor < len(lines):
        match = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@\n', lines[cursor])
        require(match is not None, 'unexpected patch hunk header')
        old_start, old_count = int(match[1]), int(match[2] or 1)
        new_start, new_count = int(match[3]), int(match[4] or 1)
        start = new_start - (1 if new_count else 0)
        require(position <= start <= len(current), 'overlapping/out-of-range patch hunk')
        result.extend(current[position:start]); position = start
        require(len(result) == old_start - (1 if old_count else 0), 'old hunk position differs')
        cursor += 1
        old_seen = new_seen = 0
        while cursor < len(lines) and not lines[cursor].startswith('@@ '):
            line = lines[cursor]
            require(line and line[0] in ' +-\n' and line[0] != '\n', 'invalid patch record')
            marker, body = line[0], line[1:]
            if marker in (' ', '+'):
                require(position < len(current) and current[position] == body, 'candidate hunk/context differs')
                position += 1; new_seen += 1
            if marker in (' ', '-'):
                result.append(body); old_seen += 1
            cursor += 1
        require((old_seen, new_seen) == (old_count, new_count), 'patch hunk lengths differ')
    result.extend(current[position:])
    original = ''.join(result)
    canonical = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                                            fromfile='original/' + name, tofile='candidate/' + name))
    require(canonical == patch, 'patch is not the exact canonical source transformation')
    return original


def source_replay(data, document, build_data, build_document):
    proof = document('proposal/source-proof.json')
    binding = build_document('results/derived/cms-source-bindings.json')
    require(len(binding['files']) == 269 and proof['source_index'] == DERIVED_SHA and
            proof['mathematical_index'] == KERNEL_SHA, 'historical source scope differs')
    recovered = {}
    for name in SIX:
        candidate = data('proposal/candidate/' + name)
        require(pin(candidate) == proof['candidate'][name], 'candidate source proof differs')
        original = reverse_patch(candidate.decode('ascii'), data('proposal/' + name + '.patch').decode('ascii'), name)
        matching = [(path, identity) for path, identity in proof['donors'].items() if path.endswith('/' + name)]
        require(len(matching) == 1, 'donor source identity is ambiguous')
        path, identity = matching[0]
        require(pin(original.encode('ascii')) == identity == binding['files'][path], 'reconstructed historical donor differs')
        recovered[name] = original
    names = {'require', 'once', 'subroutine', 'replace_subroutine', 'masked_helper', 'tensor_core',
             'dp_contraction', 'qp_contraction', 'amplitude_wrappers', 'caller_chain', 'derive'}
    constants = {'CONTRACT_ARGS', 'CONTRACT_DECL', 'CONTRACT_USE'}
    selected = []
    found_functions, found_constants = set(), set()
    for node in ast.parse(data('proposal/derive_candidate.py')).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            require(not node.decorator_list and not node.args.defaults and not any(node.args.kw_defaults),
                    'transform function has executable definition-time expressions')
            selected.append(node); found_functions.add(node.name)
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id in constants:
            require(isinstance(ast.literal_eval(node.value), str), 'transform constant is not literal text')
            selected.append(node); found_constants.add(node.targets[0].id)
    require(found_functions == names and found_constants == constants, 'pure transform selection differs')
    namespace = {'__builtins__': {'ValueError': ValueError},
                 're': re, 'donor_sources': lambda: dict(recovered)}
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'sealed-six-file-transform', 'exec'), namespace)
    original, actual = namespace['derive']()
    require(original == recovered and set(actual) == set(SIX), 'source transform donor/output set differs')
    for name, body in actual.items():
        require(body.encode('ascii') == data('proposal/candidate/' + name), 'fresh six-file transformation differs')
    return recovered


def verify_build(data, document, build_data, build_document, factory_data, factory_document):
    """Verify recorded closed build, schema successor and six-file text replay."""
    source = json.loads(checked(data, 'proposal/index.json', SOURCE_SHA))
    parent = json.loads(checked(data, 'predecessor/proposal/index.json', PARENT_SHA))
    require(len(source['files']) == 55 and len(parent['files']) == 49, 'source payload counts differ')
    for prefix, index in [('proposal/', source), ('predecessor/proposal/', parent)]:
        for name, identity in index['files'].items():
            checked(data, prefix + name, identity)
    require(sum(p['bytes'] for p in source['files'].values()) == source['payload_bytes'] == 604697,
            'source byte count differs')
    successor = document('proposal/successor-change-proof.json')
    same = successor['byte_identical_scientific_and_build_payloads']
    require(len(same) == 26 and not successor['numerical_changes'] and successor['native_executions'] == 0,
            'successor scientific change scope differs')
    for name, identity in same.items():
        require(source['files'][name] == parent['files'][name] == identity and
                data('proposal/' + name) == data('predecessor/proposal/' + name), 'scientific successor payload changed')
    failure_raw = checked(data, 'predecessor/results/build-closure.json', FAILURE_SHA)
    failure = json.loads(failure_raw)
    require(not failure['passed'] and failure['failure'] == "KeyError('size')" and
            failure['source_index_sha256'] == PARENT_SHA and failure['owned_command_count'] == failure['loader_query_count'] == 0 and
            failure['builds'] == {}, 'preserved initial controller failure differs')
    require(pin(failure_raw) == successor['preserved_failed_build_closure'], 'successor failure ancestry differs')
    for name, identity in failure['payloads'].items():
        checked(data, 'predecessor/results/' + name, identity)
    require(data('predecessor/results/build-before.json') == data('predecessor/results/build-after.json'), 'failed-attempt protected state differs')
    checked(data, 'predecessor/results/build-terminal-review.json', successor['preserved_terminal_review'])
    bindings = document('proposal/bindings.json')
    require(pin(data('proposal/bindings.json')) == document('results/build-before.json')['proposal_bindings'], 'protected proposal binding differs')
    checked(build_data, 'results/derived/index.json', DERIVED_SHA)
    checked(factory_data, 'factory/index.json', FACTORY_SHA)
    checked(factory_data, 'kernel/index.json', KERNEL_SHA)
    for name in ('caller_bounds.f90', 'cms_bounds_data.inc', 'environment_token.c'):
        require(data('proposal/dependencies/' + name) == factory_data('factory/' + name), 'copied factory source differs')
    require(data('proposal/dependencies/certificate.f90') == factory_data('kernel/certificate.f90'), 'copied certificate kernel differs')
    recovered = source_replay(data, document, build_data, build_document)

    closure = json.loads(checked(data, 'results/build-closure.json', BUILD_SHA))
    terminal = json.loads(checked(data, 'results/build-terminal-review.json', TERMINAL_SHA))
    require(closure['source_index_sha256'] == SOURCE_SHA and closure['passed'] and closure['failure'] is None and
            closure['postcheck_failure'] is None and closure['owned_command_count'] == 129 and closure['loader_query_count'] == 2,
            'closed build scope/exit differs')
    require(terminal['source_index_sha256'] == SOURCE_SHA and terminal['build_closure'] == pin(data('results/build-closure.json')) and
            terminal['terminal_exit_code'] == 0 and terminal['verified_closed_payloads'] == 275 and
            terminal['observed_owned_identities'] == 160 and terminal['remaining_owned_identities'] == [] and
            terminal['shared_lock_free_after_terminal'] and terminal['all_commands_returned_zero_and_cleanup_clear'] and
            terminal['protected_equal'] and terminal['fortran_main_calls'] == terminal['tensor_evaluations'] == terminal['amplitude_evaluations'] == 0,
            'build terminal ownership/scope differs')
    require(data('results/build-before.json') == data('results/build-after.json'), 'protected build state changed')
    protected = document('results/build-before.json')
    require(len(protected['installed_dependencies']) == 147 and len(protected['external_private_files']) == 3,
            'protected dependency inventory differs')
    require(protected['external_private_files'] == bindings['external_private_files'], 'private prerequisite bytes/hash contract differs')
    summary = document('results/build-summary.json')
    require(summary['executables'] == 2 and summary['loader_queries'] == 2 and summary['native_main_calls'] == summary['tensor_evaluations'] == 0 and
            not summary['library_rebuilds'] and not summary['independent_accuracy_qualified'] and not summary['performance_measurement'], 'build summary scope differs')
    donor = build_document('results/builds/cms-unmodified/build.json')
    require(pin(build_data('results/builds/cms-unmodified/build.json')) == summary['donor_build'] ==
            {k: bindings['donor_build'][k] for k in ('bytes', 'sha256')}, 'donor build receipt differs')
    require(pin(build_data('source/stages_native.py')) == summary['retained_link_helper'], 'retained link helper differs')
    supplementary = {name: identity['sha256'] for name, identity in bindings['fixed_supplemental_providers'].items()}
    builds, attempts, normalized, result = {}, {}, {}, {}
    for variant, expected in [('original', (147, 52, 206, 207, 269)), ('candidate', (151, 56, 201, 211, 274))]:
        base = 'results/builds/' + variant + '/'
        build = document(base + 'build.json'); before = document(base + 'before-compile.json')
        attempt = document(base + 'attempt.json'); prelink = document(base + 'prelink.json')
        link = document(base + 'link-variant-receipt.json'); runtime = document('results/runtime/' + variant + '.json')
        require(pin(data(base + 'build.json')) == {k: closure['builds'][variant][k] for k in ('bytes', 'sha256')} ==
                summary['variants'][variant]['receipt'], 'closed build receipt pin differs')
        require(build['case'] == 'cms' and build['variant'] == variant and
                build['fortran_main_calls'] == build['tensor_evaluations'] == 0 and not build['library_rebuilds'] and
                not build['independent_accuracy_qualified'] and not build['performance_measurement'], 'variant build scope differs')
        require(prelink['completed_before_link'] and prelink['bound'] == build['prelink_bound'] == link['prelink_bound'], 'prelink inventory differs')
        require(all(prelink['bound'][name] == digest for name, digest in prelink['closure']['files'].items()), 'linker closure is not prebound')
        require(build['actual_link_loads'] == link['actual_link_loads'] and
                all(prelink['bound'].get(name) == identity['sha256'] for name, identity in build['actual_link_loads'].items()), 'actual LOAD identity not prebound')
        require(build['sources'] == before['sources'] == attempt['sources_before'] == link['sources'], 'source inventory changed during build')
        require(build['build_commands'] == attempt['commands'] and attempt['failure'] is None and
                attempt['compile_seen'] and attempt['link_seen'] and attempt['runtime_seen'] and
                attempt['fortran_main_calls'] == attempt['tensor_evaluations'] == 0, 'build attempt incomplete')
        require(build['runtime'] == link['runtime'] == runtime['runtime'] and runtime['passed'] and runtime['failure'] is None and
                runtime['loader_query_started'] and runtime['actual_interpreter'] == build['runtime']['loader'], 'runtime admission differs')
        require(build['executable'] == link['executable'] == runtime['executable'] == summary['variants'][variant]['executable'] and
                build['executable_pin'] == link['executable_pin'] == runtime['executable_pin'] == summary['variants'][variant]['executable_pin'],
                'executable receipt identity differs')
        allowed = {**prelink['bound'], **supplementary}
        require(len(build['runtime']['files']) == 15 and all(allowed.get(n) == v for n, v in build['runtime']['files'].items()), 'unadmitted runtime provider')
        require(build['reused_libraries'] == attempt['reused_libraries_before'] and len(build['reused_libraries']) == 4, 'archive identity changed')
        require(all(build['all_objects_after'].get(n) == p == attempt['objects_before'][n] for n, p in build['preserved_object_hashes'].items()),
                'preserved object identity changed')
        require(len(build['module_inputs']) == 7 and all(build['modules_after'][n] == p for n, p in build['module_inputs'].items()),
                'existing module changed')
        for name, identity in before['module_inputs'].items():
            if name.startswith(build['output'] + '/'):
                require(build['module_inputs'][strip_prefix(name, build['output'] + '/')] == identity, 'local module admission differs')
            else:
                require(name == ROOT + '.tirust-build/kinematic-provenance-current-executor-1/work/mg-current/vendor/CutTools/includects/mpmodule.mod' and
                        prelink['bound'].get(name) == identity['sha256'] and
                        build['module_inputs'][SUB + 'mpmodule.mod'] == build['module_inputs']['lib/mpmodule.mod'] == identity,
                        'resolved CutTools module alias differs')
        require(all(build['objects'][n] == p['sha256'] for n, p in build['all_objects_after'].items() if n in build['objects']), 'link object inventory differs')
        require((len(prelink['bound']), len(build['actual_link_loads']), len(build['preserved_object_hashes']),
                 len(build['all_objects_after']), len(build['sources'])) == expected, 'variant inventory counts differ')
        require(build['physical_card'] == pin(data('proposal/inputs/cms-param_card.dat')), 'physical card changed')
        builds[variant], attempts[variant], normalized[variant] = build, attempt, relative_sources(build)
        result[variant] = {'prelink_files': expected[0], 'actual_LOADs': expected[1], 'preserved_objects': expected[2],
                           'all_objects': expected[3], 'recorded_source_files': expected[4], 'reused_archives': 4, 'runtime_providers': 15}
    original, candidate = builds['original'], builds['candidate']
    old_sources = relative_sources(donor)
    a, b = normalized['original'], normalized['candidate']
    require(set(a) == set(old_sources) and {n for n in a if a[n] != old_sources[n]} == {SUB + 'check_sa.f', SUB + 'makefile'},
            'original clone differs outside its diagnostic driver/makefile')
    require(a[SUB + 'check_sa.f'] == b[SUB + 'check_sa.f'] == pin(data('proposal/driver-amplitude.f')), 'diagnostic driver source differs')
    require({n for n in a if a[n] != b[n]} == {SUB + name for name in SIX} | {SUB + 'makefile'} and
            set(b) - set(a) == {SUB + name for name in SUPPORT}, 'candidate source change set differs')
    for name in SIX:
        require(a[SUB + name] == pin(recovered[name].encode('ascii')) and b[SUB + name] == pin(data('proposal/candidate/' + name)), 'built candidate/donor source identity differs')
    for name in SUPPORT:
        request = bindings['support_sources'][name]
        logical = 'proposal/' + strip_prefix(request['path'], '.tirust-build/kinematic-invariant-amplitude-proposal-2/')
        require(b[SUB + name] == pin(data(logical)) == {k: request[k] for k in ('bytes', 'sha256')}, 'built support source differs')
    require(attempts['candidate']['objects_before'] == original['all_objects_after'], 'candidate clone did not preserve original object inputs')
    expected_changed = {SUB + name.replace('.f', '.o') for name in SIX}
    require({n for n, p in attempts['original']['objects_before'].items() if p != original['all_objects_after'][n]} == {SUB + 'check_sa.o'} and
            {n for n, p in attempts['candidate']['objects_before'].items() if p != candidate['all_objects_after'][n]} == expected_changed,
            'unexpected reused-object replacement')
    added_objects = {SUB + name for name in ('amplitude_support.o', 'caller_bounds.o', 'certificate.o', 'environment_token.o')}
    require(set(original['all_objects_after']) == set(attempts['original']['objects_before']) and
            set(candidate['all_objects_after']) - set(attempts['candidate']['objects_before']) == added_objects, 'unexpected added object')
    require(original['reused_libraries'] == candidate['reused_libraries'] and original['module_inputs'] == candidate['module_inputs'] and
            set(candidate['modules_after']) - set(candidate['module_inputs']) == {SUB + name for name in
                ('ml_private_amplitude_support.mod', 'ml_private_cms_caller_bounds.mod', 'ml_private_spacelike_certificate.mod')},
            'cross-variant archive/module preservation differs')
    cdeps = document('results/builds/candidate/c-dependencies.json')
    require(cdeps == candidate['c_dependencies'] == attempts['candidate']['c_dependencies'] and cdeps['passed'] and cdeps['failure'] is None,
            'C probe dependency admission failed')
    joined = cdeps['raw'].replace('\\\n', ' ')
    require(joined.count(':') == 1 and joined.startswith('pc-environment-token:'), 'C dependency syntax differs')
    names = shlex.split(joined.split(':', 1)[1])
    paths = [n if n.startswith('/') else cdeps['cwd'] + '/' + n for n in names]
    require(len(paths) == len(set(paths)) and set(paths) == set(cdeps['actual_dependencies']), 'C dependency set differs')
    for path, identity in cdeps['actual_dependencies'].items():
        expected = pin(data('proposal/dependencies/environment_token.c')) if path == cdeps['cwd'] + '/environment_token.c' else bindings['c_headers'].get(path)
        require(identity == expected, 'unbound C source/header identity')
    return {'source_payloads': 55, 'predecessor_payloads': 49, 'unchanged_scientific_payloads': 26,
            'reconstructed_historical_donors': 6, 'fresh_six_file_transform_replay': True,
            'fresh_269_source_DAG_reparse': False, 'prior_schema_failure_preserved': True, 'builds': result,
            'recorded_owned_identities_absent': 160, 'external_binary_reads': 0, 'native_executions': 0,
            'amplitude_accuracy_qualified': False, 'performance_qualified': False}
