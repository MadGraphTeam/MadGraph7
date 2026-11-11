"""Pure verification of closed original-input EW replay evidence; no native/external reads."""
import ast
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_TEXT = '/shared/external/madloop_tirust/madgraph7_madloop_tirust/'
WORK = '.tirust-build/ew-refusal-replay-executor-1/'
SOURCE = '.tirust-build/ew-refusal-replay-proposal-1/'
SOURCE_SHA = '9384471349f462e7f028161c5b9a1673027698d930e82ac7da0aa6910c08ad61'
BUILD_SHA = 'c195e4de9aa301fb3f17b00e3a89a06fca948352f359bbd212b1d6ac2dad75f3'
RUN_SHA = 'ec0dcaf70e5921fca8b32041bfde688430cf07c347b5ed31275ecb5845f1ce04'
DEPENDENCIES = {
    'current-kinematic-trace-build-qualification-1': '89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033',
    'current-kinematic-trace-runtime-qualification-1': '730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def packet(root, expected_seal=None):
    raw = (root / 'evidence-index.json').read_bytes()
    if expected_seal:
        require(sha(raw) == expected_seal, 'Build sibling seal changed')
    index = json.loads(raw)
    paths = list(root.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Packet contains a symlink')
    require({str(p.relative_to(root)) for p in paths if p.is_file()} ==
            set(index['files']) | {'evidence-index.json'}, 'Packet file set changed')
    for name, pin in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe stored path')
        content = (root / name).read_bytes()
        require(sha(content) == pin['sha256'] and len(content) == pin['bytes'], 'Stored bytes changed: ' + name)
        decoded = gzip.decompress(content) if name.endswith('.gz') else content
        require(b'\0' not in decoded and not decoded.startswith((b'\x7fELF', b'!<arch>')), 'Native payload retained')
        decoded.decode('utf-8')

    def data(name):
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe logical path')
        compressed = index['compressed_text'].get(name)
        if compressed:
            require(compressed['path'] in index['files'], 'Unindexed compressed path')
            content = gzip.decompress((root / compressed['path']).read_bytes())
            require(len(content) == compressed['raw_bytes'] and sha(content) == compressed['raw_sha256'],
                    'Raw compressed identity changed')
            return content
        require(name in index['files'], 'Unindexed logical path')
        return (root / name).read_bytes()

    for name, pin in index['origins'].items():
        content = data(name)
        require(len(content) == pin['bytes'] and sha(content) == pin['sha256'], 'Original copy identity changed')
    return index, data, lambda name: json.loads(data(name))

def identity(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


def main():
    index, data, document = packet(HERE)
    require(index['dependencies'] == DEPENDENCIES, 'Declared sibling dependencies changed')
    build_index, build_data, build_document = packet(HERE.parent / 'current-kinematic-trace-build-qualification-1',
        DEPENDENCIES['current-kinematic-trace-build-qualification-1'])
    runtime_index, runtime_data, runtime_document = packet(HERE.parent / 'current-kinematic-trace-runtime-qualification-1',
        DEPENDENCIES['current-kinematic-trace-runtime-qualification-1'])
    require(sha(data('proposal/index.json')) == SOURCE_SHA, 'Proposal source changed')
    for name, checksum in {
        'results/build-terminal-review.json': 'dc5e27e4a1fc13e8033ce90b8e20f0778efd58467d8797d91122457294443a87',
        'results/root-build-review.json': '70a5f4052a5b784bc21d87fc84c845d13974b88049c778ae5f92a72cc3075cd8',
        'results/root-run-review.json': '344db4e1c506486cba50faf4883c5aaf7ad877ef25a554708a0b6804676f67c4',
        'reviews/source-review.json': '4e0e7beb6cd4b3f47f204a4fc3c784545f18e36988597637802d52c254bb7d0d',
    }.items():
        require(sha(data(name)) == checksum, 'Pinned review changed')
    source = document('proposal/index.json')
    require(len(source['files']) == 17, 'Proposal payload count changed')
    for name, pin in source['files'].items():
        require(identity(data('proposal/' + name)) == pin, 'Sealed source payload differs')
    require(sha(data('results/build-closure.json')) == BUILD_SHA and
            sha(data('results/run-terminal-closure.json')) == RUN_SHA, 'Root-reviewed closures changed')
    bclose = document('results/build-closure.json')
    closed = document('results/run-terminal-closure.json')
    build = document('results/build.json')
    prelink = document('results/prelink.json')
    bindings = document('proposal/bindings.json')
    # The siblings retain exact helper/source bytes. Native dependency identities
    # remain receipts only; never follow an original pathname on this machine.
    sibling_origins = {pin['path']: (name, build_data) for name, pin in build_index['origins'].items()}
    sibling_origins.update({pin['path']: (name, runtime_data) for name, pin in runtime_index['origins'].items()})
    for original, expected in bindings['files'].items():
        if original in sibling_origins:
            name, reader = sibling_origins[original]
            require(identity(reader(name)) == expected, 'Retained dependency/helper source differs')
    expected_origins = {SOURCE + 'index.json'} | {SOURCE + name for name in source['files']}
    expected_origins.update(WORK + 'results/' + name for name in [
        'build-closure.json', 'run-terminal-closure.json', 'root-build-review.json', 'root-run-review.json',
        'build-terminal-review.json', 'build-before.json', 'build-after.json',
        'build.json', 'prelink.json', 'build-runtime.json'])
    expected_origins.update(WORK + 'results/execution/' + label + suffix
        for label in ['compile-driver', 'link-driver', 'query-build-0001'] for suffix in ['.json', '.txt'])
    for original in closed['commands']:
        expected_origins.update([original, str(Path(original).with_suffix('.txt'))])
    expected_origins.add('.tirust-build/ew-refusal-replay-independent-review-1.json')
    expected_origins.update(closed['closed_receipts'])
    for item in closed['workers'].values():
        expected_origins.update(item['files'])
    require({pin['path'] for pin in index['origins'].values()} == expected_origins,
            'Captured origins differ from explicit source/closed receipt lists')
    require(bclose['source_index_sha256'] == SOURCE_SHA and bclose['passed'] and
            bclose['failure'] is None and bclose['postcheck_failure'] is None, 'Build closure failed')
    for key, name in [('before', 'build-before.json'), ('after', 'build-after.json'), ('build_receipt', 'build.json')]:
        require(identity(data('results/' + name)) == bclose[key], 'Closed build receipt differs')
    require(data('results/build-before.json') == data('results/build-after.json') ==
            data('results/run-before.json') == data('results/run-after.json'), 'Protected state changed')
    require(build['fortran_main_runs'] == build['tensor_calls'] == 0 and len(build['prelink_bound']) == 149 and
            len(build['actual_link_loads']) == 20, 'Build-only/link scope changed')
    require(prelink['completed_before_link'] and prelink['bound'] == build['prelink_bound'] and
            prelink['driver_object'] == build['driver_object'], 'Prelink receipt differs')
    donor = build_data('results/builds/cms-unmodified/prelink.json')
    require(identity(donor) == prelink['donor_prelink'], 'Donor inventory differs')
    for name, pin in build['actual_link_loads'].items():
        require(build['prelink_bound'].get(name) == pin['sha256'], 'Actual link load was not prebound')
    for tail, checksum in [('libtirust.a', '0c86349a4fd0dadd918476731e899529ce1c7b17f5892a4af2c22dd762250c1a'),
                           ('libmadloop_tirust.so', 'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9')]:
        matches = [pin['sha256'] for name, pin in build['actual_link_loads'].items() if name.endswith('/' + tail)]
        require(matches == [checksum], 'Original adapter/library link identity differs')
    providers = build['runtime']['files']
    require(len(providers) == 6 and build['runtime']['loader'] == bindings['declared_loader'] and
            providers[bindings['declared_loader']] == build['runtime']['loader_sha256'], 'Runtime identity differs')
    allowed = dict(build['prelink_bound'])
    allowed.update({name: pin['sha256'] for name, pin in bindings['fixed_supplemental_providers'].items()})
    require(all(allowed.get(name) == checksum for name, checksum in providers.items()), 'Unbound provider')
    expected_runtime = {**providers, build['executable']: build['executable_pin']['sha256']}

    commands = {}
    def check_command(logical, expected=None):
        raw = data(logical)
        receipt = json.loads(raw)
        log = data(str(Path(logical).with_suffix('.txt')))
        if expected is not None:
            require(identity(raw) == expected['receipt'] and identity(log) == expected['log'] and
                    receipt['argv'] == expected['argv'], 'Closed command identity differs')
        require(sha(log) == receipt['log_sha256'] and receipt['returncode'] == 0 and
                receipt['exception'] is None and receipt['cleanup_exception'] is None and
                receipt['cleanup']['survivors'] == [] and receipt['cpu'] == 12 and not receipt['cpu_exclusive'],
                'Owned command did not close cleanly')
        commands[logical] = receipt
        return receipt

    build_terminal = document('results/build-terminal-review.json')
    require(build_terminal['terminal_exit_code'] == 0 and build_terminal['build_closure'] == identity(data('results/build-closure.json')) and
            build_terminal['prelink_receipt'] == identity(data('results/prelink.json')) and
            build_terminal['runtime_receipt'] == identity(data('results/build-runtime.json')) and
            build_terminal['cleanup_exceptions'] == build_terminal['matching_live_owned_processes'] == build_terminal['owned_survivors'] == [] and
            build_terminal['shared_lock_nonblocking_acquire_release'] and build_terminal['protected_before_after_equal'],
            'Build terminal closure differs')
    for label in ['compile-driver', 'link-driver', 'query-build-0001']:
        name = 'results/execution/' + label + '.json'
        check_command(name, build_terminal['commands'][WORK + name])
    plan = document('proposal/stage-plan.json')
    for label, key in [('compile-driver', 'compile_argv'), ('link-driver', 'link_argv')]:
        require(commands['results/execution/' + label + '.json']['argv'][6:] == plan['build_gate'][key],
                'Build command differs from sealed plan')
    require(data('results/execution/query-build-0001.txt').decode() == build['runtime']['loader_listing'],
            'Build loader transcript differs')
    bruntime = document('results/build-runtime.json')
    require(bruntime['passed'] and bruntime['failure'] is None and bruntime['loader_query_started'] and
            bruntime['runtime'] == build['runtime'] and bruntime['actual_interpreter'] == build['runtime']['loader'] and
            bruntime['executable_pin'] == build['executable_pin'], 'Build runtime admission differs')
    require(len(closed['commands']) == 4 and len(closed['closed_receipts']) == 7 and
            len(closed['workers']) == 2 and closed['terminal_exit_code'] == 0 and
            closed['source_index_sha256'] == SOURCE_SHA and closed['reviewed_build_closure'] == identity(data('results/build-closure.json')),
            'Run closure scope differs')
    for original, pins in closed['commands'].items():
        require(original.startswith(WORK), 'Foreign command')
        check_command(original[len(WORK):], pins)
    for original, pin in closed['closed_receipts'].items():
        require(original.startswith(WORK) and identity(data(original[len(WORK):])) == pin, 'Closed receipt differs')
    require(closed['actual_context_generations'] == [1] * 2 and closed['actual_fortran_tensor_calls'] == 2 and
            closed['actual_fresh_processes'] == 2 and closed['verified_live_map_boundaries'] == 4 and
            closed['all_pre_post_inputs_unchanged'] and closed['all_owned_commands_exit_zero'] and closed['no_retry'] and
            closed['cleanup_exceptions'] == closed['matching_live_owned_processes'] == closed['owned_survivors'] == [] and
            closed['shared_lock_nonblocking_acquire_release'] and closed['protected_before_after_equal'] and
            not closed['coefficient_buffers_inspected'] and not closed['independent_accuracy_qualified'] and
            not closed['performance_measurement'], 'Closed run accounting or qualification changed')
    require(closed['stage']['passed'] and closed['stage']['failure'] is None and
            closed['stage']['postcheck_failure'] is None and closed['stage']['reviewed_build_sha256'] == BUILD_SHA and
            closed['stage'] == document('results/stage-run.json'), 'Run stage failed or differs')
    # Reuse only the sealed in-memory record parser. No imports, globals, main,
    # subprocess, filesystem helpers or native controller functions are executed.
    tree = ast.parse(data('proposal/worker.py'))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in {'Protocol', 'require'}]
    require({n.name for n in nodes} == {'Protocol', 'require'}, 'Protocol selection differs')
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'retained-pure-protocol', 'exec'), namespace)
    requests = document('proposal/input-certificate.json')['requests']
    require(len(requests) == 2 and set(closed['workers']) == {r['id'] for r in requests}, 'Request set differs')
    response_rows, pids, statuses = [], set(), []
    for position, request in enumerate(requests, 1):
        name = request['id']
        saved = closed['workers'][name]
        prefix = 'results/runs/' + name + '/'
        for original, pin in saved['files'].items():
            require(original.startswith(WORK) and identity(data(original[len(WORK):])) == pin, 'Closed worker file differs')
        require(set(saved['files']) == {WORK + prefix + leaf for leaf in
                ['result.json', 'specification.json', 'stdout.txt', 'maps-READY.txt', 'maps-DONE.txt']}, 'Worker payload set differs')
        spec = document(prefix + 'specification.json')
        result = document(prefix + 'result.json')
        require(spec['request'] == request == result['request'] and spec['proposal_index_sha256'] == SOURCE_SHA and
                spec['executable'] == build['executable'] and spec['executable_pin'] == build['executable_pin'] and
                spec['expected_runtime'] == expected_runtime and
                result['specification_sha256'] == sha(data(prefix + 'specification.json')), 'Worker source/spec differs')
        wordsraw = data('proposal/inputs/' + name + '.words')
        require(identity(wordsraw) == spec['input_pin'] == saved['input_pin'] and
                spec['input_file'] == saved['input_file'] == ROOT_TEXT + WORK + 'work/workers/' + name + '/input.words',
                'Worker literal input differs')
        lines = wordsraw.decode().splitlines()
        count = 41 if request['mode'] == 1 else 82
        require(lines[:2] == ['KPROJ_WORDS_V1', f"{request['point']} {request['mode']} 4 3 {count}"] and
                len(lines[2:]) == count, 'Literal word schema differs')
        protocol = namespace['Protocol'](request, lines[2:])
        stdout = data(prefix + 'stdout.txt')
        for line in stdout.decode().splitlines():
            protocol.line(line)
        require(protocol.phase == 'closed' and not protocol.error_open and protocol.events == ['READY', 'DONE', 'CLOSED'] and
                protocol.fields == result['driver_fields'] and '\n'.join(protocol.error_lines) == result['native_error'] and
                result['protocol'] == protocol.events and result['commands_sent'] == ['EVALUATE', 'EXIT'], 'Raw protocol replay differs')
        status = int(protocol.fields['FORTRAN_STATUS'][0])
        expected_status = 1
        expected_error = {
            1: 'TIRust DP: Unstable("ill-conditioned Gram matrix for [1, 2, 3]: conditioning 2.0047533625855067e-14")',
            4: 'TIRust QP: Unstable("ill-conditioned Gram matrix for [0, 1, 2, 3]: conditioning 5.22407811366341444758736094099234985181085186192010198472166824e-30")',
        }[request['mode']]
        require(status == expected_status == result['fortran_status'] and result['native_error'] == expected_error,
                'Controlled status/error result differs')
        require(result['orchestration_complete'] and result['failure'] is None and result['cleanup_failure'] is None and
                result['postcheck_failures'] == result['cleanup']['survivors'] == [] and result['native_returncode'] == 0 and
                result['stdout'] == identity(stdout) and result['stdout_observed_bytes'] == len(stdout) and
                not result['stdout_overflow'] and not result['coefficient_buffers_read'] and not result['retry_performed'] and
                not result['independent_accuracy_qualified'] and not result['performance_measurement'], 'Worker completion differs')
        require(result['native_pid'] == saved['native_pid'] and result['native_pid'] not in pids, 'Fresh process identity duplicated')
        pids.add(result['native_pid'])
        require([m['boundary'] for m in result['runtime_maps']] == ['READY', 'DONE'], 'Map boundaries differ')
        for m in result['runtime_maps']:
            raw = data(prefix + m['file'])
            require(identity(raw) == {k: m[k] for k in ('bytes', 'sha256')} and m['verified'] and
                    m['elf'] == expected_runtime and m['data'] == {}, 'Live map receipt differs')
            names, inode_keys = set(), {}
            for line in raw.decode().splitlines():
                fields = line.split(None, 5)
                if len(fields) != 6 or not fields[5].startswith('/'):
                    continue
                mapped = fields[5]
                require(not mapped.endswith(' (deleted)') and mapped in expected_runtime and int(fields[4]) > 0,
                        'Unexpected/deleted/unbacked mapped provider')
                names.add(mapped)
                inode_keys.setdefault(mapped, set()).add((fields[3], fields[4]))
            require(names == set(expected_runtime) and all(len(v) == 1 for v in inode_keys.values()),
                    'Raw map provider paths/inode consistency differs')
        admission_name = 'results/' + name + '-runtime.json'
        admission = document(admission_name)
        require(identity(data(admission_name)) == saved['runtime_admission'] and admission['passed'] and admission['failure'] is None and
                admission['actual_interpreter'] == build['runtime']['loader'] and admission['executable_pin'] == build['executable_pin'] and
                admission['runtime']['files'] == providers and admission['runtime']['loader_sha256'] == build['runtime']['loader_sha256'],
                'Per-worker runtime admission differs')
        query = f'results/execution/query-run-{position:04d}'
        require(WORK + query + '.json' == closed['actual_loader_queries'][position - 1] and
                commands[query + '.json']['argv'][-3:] == [build['runtime']['loader'], '--list', build['executable']] and
                data(query + '.txt').decode() == admission['runtime']['loader_listing'], 'Worker loader query join differs')
        command = commands['results/execution/worker-' + name + '.json']
        require(command['argv'][-2:] == [ROOT_TEXT + WORK + prefix + 'specification.json', sha(data(prefix + 'specification.json'))],
                'Worker outer command/spec join differs')
        response = {'coefficients_inspected': False, 'fortran_evaluation_calls': 1, 'fortran_status': status,
                    'fresh_context_generation': 1, 'id': name, 'independent_accuracy_qualified': False,
                    'input_words_unchanged': True, 'mode': request['mode'], 'original_error': expected_error,
                    'point': request['point'], 'variant': request['variant']}
        require(saved['response'] == response, 'Closed response differs from raw stdout')
        response_rows.append(response)
        statuses.append(status)
    require(closed['responses'] == response_rows and statuses == [1, 1], 'Response totals differ')
    compact = [{k: r[k] for k in ('id', 'point', 'mode', 'variant', 'fortran_status')} for r in response_rows]
    responses = document('results/responses.json')
    account = document('results/request-accounting.json')
    require(responses['completed'] == account['completed'] == compact and account['failure'] is None and
            account['not_completed'] == [] and responses['success_status_is_not_accuracy_proof'] and
            responses['diagnostic_original_inputs_only'] and not responses['coefficient_buffers_inspected'], 'Accounting diverges')
    require(closed['diagnostic_original_inputs_only'] and not closed['input_interventions'] and
            closed['both_original_refusal_statuses_reproduced'], 'Original-only diagnostic scope differs')
    source_review = document('reviews/source-review.json')
    require(source_review['source_index_sha256'] == SOURCE_SHA and source_review['reviewed_files'] == source['files'] and
            source_review['additive_resource_clarification']['metadata_is_hard_whole_controller_cap'] is False,
            'Source review/deadline qualification differs')
    spec = importlib.util.spec_from_file_location('retained_ew_input_checks', HERE / 'input_checks.py')
    checks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checks)
    input_summary = checks.verify_inputs(data, document, runtime_data, runtime_document, build_data, build_document)
    print(json.dumps({'retained_payloads': len(index['files']), 'closed_build_commands': 3,
                      'closed_run_commands': 4, 'fresh_contexts_and_tensor_calls': 2, 'raw_protocols_replayed': 2,
                      'live_maps_joined': 4, 'reproduced_original_refusals': 2, 'input_interventions': 0,
                      'native_executions': 0, 'external_binary_reads': 0, 'independent_accuracy_qualified': False,
                      'generalized_fix_qualified': False, 'performance_qualified': False, 'inputs': input_summary}, indent=2))


if __name__ == '__main__':
    main()
