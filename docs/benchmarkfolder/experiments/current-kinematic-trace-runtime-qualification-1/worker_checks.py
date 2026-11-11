"""Pure replay of closed trace workers; no filesystem or controller imports.

Matching observations do not qualify a failed contribution. Runtime ELF and
data identities below are retained observations, not fresh external rehashes.
"""
import hashlib
import re


PREFIX = '.tirust-build/kinematic-provenance-current-executor-1/'
EQUAL_FILES = ('bootstrap-values.dat', 'driver-state.dat', 'driver-bits.dat',
               'qualification.dat', 'dimensions.dat', 'values.dat', 'timed_status.dat')
OUTPUT_FILES = set(EQUAL_FILES) | {'timing.dat', 'trace.dat'}
WORKER_FILES = OUTPUT_FILES | {'result.json', 'specification.json', 'stdout.txt',
                               'maps-0.txt', 'maps-1.txt'}
VARIANTS = ('unmodified', 'disabled', 'enabled')
GROUPS = {'ordinary': [('cms', 1, 1), ('cms', 1, 4), ('ew', 1, 1), ('ew', 1, 4)],
          'stress': [('cms', 20, 1), ('cms', 20, 4), ('cms', 21, 1),
                     ('cms', 21, 4), ('ew', 2, 1), ('ew', 2, 4)]}
FAILURE = 'Equal variants retain a failed contribution; interpretation blocked'
LOCALE = {'/nix/store/ig5ylfz7qcgpmvzip4pxfkp2pfi76s2j-glibc-locales-2.44-25/lib/locale/locale-archive':
          {'bytes': 233352016,
           'sha256': '53851ab7caa5a0f893056762a6abff4e383ea06c0e6b276ee16876499e712d27'}}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def logical(original):
    require(original.startswith(PREFIX), 'Path outside the closed executor')
    name = original[len(PREFIX):]
    require(name and not name.startswith('/') and '..' not in name.split('/'),
            'Unsafe closed logical path')
    return name


def match(raw, pin, name):
    require(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'],
            'Closed raw identity differs: ' + name)


def state_rows(raw, point, mode, stress):
    rows = []
    for line in raw.decode().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        require(len(fields) == 16 and fields[0] == 'STATE' and fields[1] in ('A', 'B'),
                'Driver state schema changed')
        numbers = list(map(int, fields[2:12]))
        bits = fields[12:]
        require(all(re.fullmatch('[0-9A-F]{8}', value) for value in bits), 'Logical bit schema')
        row = dict(zip(('phase', 'call', 'point', 'requested_mode', 'code', 'actual_mode',
                        'checkphase', 'helicity_checked', 'fpe_dp', 'fpe_qp'), numbers))
        row.update(stage=fields[1], bits=bits)
        require(bits == [f'{row[key]:08X}' for key in
                         ('checkphase', 'helicity_checked', 'fpe_dp', 'fpe_qp')],
                'Logical values differ from their retained native words')
        rows.append(row)
    require([(r['phase'], r['call'], r['stage']) for r in rows] ==
            [(0, 1, 'B'), (0, 1, 'A'), (1, 2, 'B'), (1, 2, 'A')],
            'Incomplete or repeated outer calls')
    for i, row in enumerate(rows):
        require(row['point'] == (1 if i < 2 else point) and row['requested_mode'] == mode and
                row['checkphase'] == 1 and row['helicity_checked'] == 0, 'Fixed call state differs')
        if row['stage'] == 'B':
            require(row['code'] == -987654321 and row['actual_mode'] == -999 and
                    row['fpe_dp'] == row['fpe_qp'] == 0, 'Before-call sentinel/state differs')
        else:
            require(row['code'] == {1: 110, 4: 120}[mode] and row['actual_mode'] == mode,
                    'Unchecked fixed-mode status differs')
            expected = ((1, 0) if mode == 1 else (0, 1)) if stress and i == 3 else (0, 0)
            require((row['fpe_dp'], row['fpe_qp']) == expected, 'Contribution failure flags differ')
    return rows


def coverage(raws, point, mode):
    require(raws['dimensions.dat'].decode().split() == ['1', '1', '1'], 'Split-order dimensions')
    heads = []
    for line in raws['driver-bits.dat'].decode().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        fields = line.split()
        require(len(fields) == 10 and all(re.fullmatch('[0-9A-F]{16}', x) for x in fields[5:]),
                'Raw answer word schema')
        head = list(map(int, fields[:5]))
        require(head[4] == {1: 110, 4: 120}[mode] and fields[-1] == 'BFF0000000000000',
                'Per-order unchecked status/accuracy differs')
        require(all(int(word, 16) & 0x7ff0000000000000 != 0x7ff0000000000000
                    for word in fields[5:9]), 'Nonfinite retained coefficient')
        heads.append(tuple(head[:4]))
    require(heads == [(0, 1, 1, 0), (0, 1, 1, 1), (1, 2, point, 0), (1, 2, point, 1)],
            'Missing or duplicate output columns')
    values = [line.split() for line in raws['values.dat'].decode().splitlines() if line.strip()]
    require(len(values) == 2 and all(len(row) == 8 for row in values) and
            [tuple(map(int, row[:3])) for row in values] ==
            [(1, 0, {1: 110, 4: 120}[mode]), (1, 1, {1: 110, 4: 120}[mode])],
            'Decimal answer coverage differs')
    require(raws['timed_status.dat'].decode().split() == [str({1: 110, 4: 120}[mode]), '1'],
            'Measured outer-entry count/status differs')
    return {'orders': [0, 1], 'bootstrap_columns': 2, 'measurement_columns': 2}


def ordinary_trace(raw, case, point, mode):
    count, schema, seen, serials = 0, 0, set(), {}
    groups = {3, 4, 8, 12, 24} if case == 'cms' else {5}
    for line in raw.decode().splitlines():
        fields = line.split()
        require(fields, 'Empty trace row')
        if fields[0] == 'KT_SCHEMA_V2':
            schema += 1
            require(fields[1:] == [str(point), '64', '128', '64', '32'], 'Trace native word schema')
            continue
        require(fields[0].startswith('KT_') and len(fields) >= 11, 'Trace row schema')
        header = list(map(int, fields[1:11]))
        require(header[0] == point and header[2] in groups and header[3] == mode,
                'Trace request differs')
        serials.setdefault(header[1], set()).add(fields[0])
        if fields[0].startswith('KT_STATUS_'):
            require(len(fields) == 12 and int(fields[11]) == 0, 'Ordinary tensor evaluation failed')
        else:
            require(all(re.fullmatch('[0-9A-F]{16}', word) for word in fields[11:]), 'Trace word payload')
        seen.add(header[2])
        count += 1
    suffix = 'DP' if mode == 1 else 'QP'
    required = {'KT_LOOP_DP', 'KT_MATRIX_' + suffix, 'KT_RESIDUAL_' + suffix,
                'KT_TIRUST_' + suffix, 'KT_STATUS_' + suffix}
    require(schema == 1 and count > 0 and seen == groups and
            all(required <= tags for tags in serials.values()), 'Incomplete ordinary trace')
    return {'records': count, 'groups': sorted(seen), 'actual_mode': mode,
            'evaluation_status_is_not_host_stability': True}


def verify_workers(data, document, build_data, build_document, closures):
    """Verify only named retained records; callbacks supply already sealed bytes."""
    require(set(closures) == {'ordinary', 'stress'}, 'Wrong closed-stage scope')
    totals = {'commands': 0, 'workers': 0, 'outer_entries': 0, 'map_boundaries': 0,
              'equal_observation_groups': 0, 'ordinary_passed_groups': 0,
              'stress_failed_contribution_groups': 0}
    stage_summaries = {}
    for stage, count in [('ordinary', 12), ('stress', 18)]:
        closed = closures[stage]
        expected_ids = {f'{case}-point{point}-mode{mode}-{variant}'
                        for case, point, mode in GROUPS[stage] for variant in VARIANTS}
        require(set(closed['workers']) == expected_ids and closed['actual_native_workers'] == count and
                closed['actual_outer_entries_observed'] == 2 * count and
                closed['verified_runtime_map_boundaries'] == 2 * count and
                closed['command_count'] == len(closed['commands']) == 2 * count, 'Stage worker/count scope')
        commands, identities = {}, set()
        for original, pins in closed['commands'].items():
            name = logical(original)
            require(name.startswith('results/execution/') and name.endswith('.json'), 'Command namespace')
            match(data(name), pins['receipt'], name)
            log_name = name[:-5] + '.txt'
            match(data(log_name), pins['log'], log_name)
            row = document(name)
            require(row['returncode'] == 0 and row['exception'] is None and
                    row['cleanup_exception'] is None and not row['cleanup']['survivors'] and
                    sha(data(log_name)) == row['log_sha256'], 'Owned command/cleanup failure')
            require(row['cpu'] == 12 and row['cpu_exclusive'] is False and
                    row['argv'][:3] == ['taskset', '-c', '12'], 'Worker affinity scope')
            commands[original] = row
            identities.update((r['pid'], r['start_time_ticks'], r['session'])
                              for r in row['cleanup']['observed_owned_processes'])
        require(len(identities) == closed['observed_process_identities'] == 3 * count and
                not closed['matching_live_owned_processes'] and closed['shared_lock_nonblocking_acquire_release'] and
                closed['protected_before_after_equal'], 'Closed ownership/protection record differs')
        queries = {}
        require(len(closed['actual_loader_queries']) == count, 'Loader query count')
        for query in closed['actual_loader_queries']:
            command = commands[query['receipt']]
            require(command['argv'] == query['argv'] and query['argv'][:3] == ['taskset', '-c', '12'] and
                    query['argv'][-2] == '--list' and query['argv'][-1] not in queries, 'Loader query binding')
            queries[query['argv'][-1]] = query
        raw_workers, states = {}, {}
        for label, worker in closed['workers'].items():
            prefix = 'results/runs/' + label + '/'
            raws = {}
            for original, pin in worker['files'].items():
                name = logical(original)
                require(name.startswith(prefix) and '/' not in name[len(prefix):], 'Worker payload namespace')
                raw = data(name)
                match(raw, pin, name)
                raws[name[len(prefix):]] = raw
            require(set(raws) == WORKER_FILES, 'Worker file inventory differs')
            result = document(prefix + 'result.json')
            spec = document(prefix + 'specification.json')
            request = result['request']
            require(request == spec['request'] and request['id'] == label and request['stage'] == stage and
                    request['maximum_outer_amplitude_entries'] == 2 and
                    request['bootstrap_outer_entries'] == request['bootstrap_point'] == 1 and
                    request['trace_enabled'] == (request['variant'] == 'enabled'), 'Worker request binding')
            case, point, mode, variant = (request[k] for k in ('case', 'point', 'requested_mode', 'variant'))
            require(label == f'{case}-point{point}-mode{mode}-{variant}' and worker['actual_mode'] == mode,
                    'Worker identity/mode differs')
            for name_key, pin_key in [('bootstrap', 'bootstrap_pin'), ('measurement', 'point_pin'),
                                      ('numerical_card', 'numerical_card'), ('physical_card', 'physical_card')]:
                name = request[name_key]
                require('/' not in name and name not in ('', '.', '..'), 'Input basename differs')
                raw = data('inputs/' + name)
                require(raw == build_data('source/inputs/' + name), 'Literal input differs from sealed source')
                match(raw, spec[pin_key], name)
            require(result['orchestration_complete'] and result['runtime_map_verified'] and result['trace_complete'] and
                    not result['stdout_overflow'] and not result['collection_errors'] and result['failure'] is None and
                    result['cleanup_failure'] is None and not result['numerically_qualified'] and
                    not result['performance_measurement'], 'Worker scope/completion differs')
            require(result['native_pid'] == worker['native_pid'] and
                    result['native_returncode'] == worker['native_returncode'] == 0 and
                    worker['outer_entries_observed'] == 2 and not worker['independent_accuracy_qualified'],
                    'Native worker closure differs')
            require(len(result['close_receipts']) == 1 and
                    all(c['complete'] and c['worker_closed'] and c['graceful_requested'] and
                        c['pid'] == worker['native_pid'] and c['returncode'] == 0 for c in result['close_receipts']),
                    'Native close receipt differs')
            require([row[0] for row in result['protocol']] == ['READY', 'DONE', 'BYE'], 'Protocol boundaries differ')
            require(set(result['outputs']) == OUTPUT_FILES, 'Worker output inventory differs')
            for name, pin in result['outputs'].items():
                match(raws[name], pin, prefix + name)
            worker_command = commands[PREFIX + 'results/execution/worker-' + label + '.json']
            require(worker_command['argv'][-3] == '-B' and
                    worker_command['argv'][-2].endswith('/' + PREFIX + 'source/native_job.py') and
                    worker_command['argv'][-1] == spec['evidence'] + '/specification.json' and
                    worker_command['argv'][-1].endswith('/' + PREFIX + prefix + 'specification.json') and
                    spec['executable'] == spec['cwd'] + '/check', 'Command/specification path binding differs')
            require(any(p['pid'] == worker['native_pid'] for p in
                        worker_command['cleanup']['observed_owned_processes']), 'Native PID missing from ownership')
            build_label = case + '-' + ('unmodified' if variant == 'unmodified' else 'instrumented')
            built = build_document('results/builds/' + build_label + '/build.json')
            require(spec['executable_pin'] == built['executable_pin'] and
                    spec['physical_card'] == built['physical_card'] and
                    spec['source_index_sha256'] == sha(build_data('source/index.json')),
                    'Worker executable/card/source differs from closed build')
            admission_name = 'results/' + stage + '-admission-1/' + label + '-runtime.json'
            match(data(admission_name), worker['loader_admission'], admission_name)
            require(closed['closed_receipts'][PREFIX + admission_name] == worker['loader_admission'],
                    'Loader admission not in closed receipt inventory')
            admission = document(admission_name)
            require(admission['passed'] and admission['failure'] is None and admission['loader_query_started'] and
                    admission['worker_id'] == label and admission['executable'] == spec['executable'] and
                    admission['closed_runtime'] == admission['runtime']['files'] == built['runtime']['files'] and
                    admission['actual_interpreter'] == admission['runtime']['loader'] == built['runtime']['loader'] and
                    admission['runtime']['loader_sha256'] == built['runtime']['loader_sha256'] ==
                    admission['closed_runtime'][admission['actual_interpreter']],
                    'Worker runtime differs from build/admission')
            runtime = dict(admission['closed_runtime'])
            runtime[spec['executable']] = spec['executable_pin']['sha256']
            require(len(runtime) == 16 and spec['expected_runtime'] == runtime, 'Expected ELF identity closure')
            query = queries[spec['executable']]
            require(query['argv'][3] == admission['actual_interpreter'] and
                    data(logical(query['receipt'])[:-5] + '.txt').decode() == admission['runtime']['loader_listing'],
                    'Worker loader query log differs from admission')
            require(worker['maps_boundaries'] == [m['boundary'] for m in result['runtime_maps']] == ['READY', 'DONE'],
                    'Runtime map boundaries differ')
            for i, mapped in enumerate(result['runtime_maps']):
                name = f'maps-{i}.txt'
                require(mapped['file'] == name and sha(raws[name]) == mapped['sha256'] and
                        mapped['elf'] == runtime and mapped['data'] == LOCALE, 'Runtime map identity differs')
                paths = set()
                for line in raws[name].decode().splitlines():
                    fields = line.split(None, 5)
                    require(len(fields) >= 5, 'Malformed retained proc map')
                    if len(fields) == 6 and fields[5].startswith('/'):
                        paths.add(fields[5])
                        if fields[5] in LOCALE:
                            require(fields[1] == 'r--p' and fields[5].startswith('/nix/store/') and
                                    fields[5].endswith('/lib/locale/locale-archive'),
                                    'Observed locale mapping violates the retained data policy')
                require(paths == set(runtime) | set(LOCALE), 'Raw map path set differs from recorded identities')
            observations = state_rows(raws['driver-state.dat'], point, mode, stage == 'stress')
            cov = coverage(raws, point, mode)
            if stage == 'stress':
                require(worker['driver_after_states'] == [r for r in observations if r['stage'] == 'A'] and
                        worker['coverage'] == cov, 'Stress closure state/coverage differs from raw words')
            if variant != 'enabled':
                require(raws['trace.dat'] == b'', 'Inactive instrumentation produced trace output')
            match(raws['trace.dat'], worker['trace_pin'], prefix + 'trace.dat')
            raw_workers[label], states[label] = raws, observations
        pairs, failures = [], []
        for case, point, mode in GROUPS[stage]:
            labels = [f'{case}-point{point}-mode{mode}-{v}' for v in VARIANTS]
            for name in EQUAL_FILES:
                require(len({raw_workers[label][name] for label in labels}) == 1, 'U/D/E bytes differ: ' + name)
            final = states[labels[0]][-1]
            failed = bool(final['fpe_dp'] or final['fpe_qp'] or final['code'] < 0 or final['code'] >= 400)
            if stage == 'stress':
                require(failed, 'Recorded stress contribution failure was lost')
                failures.append({'case': case, 'point': point, 'mode': mode, 'failure': FAILURE})
            else:
                require(not failed and all('##W03' not in raw_workers[label]['stdout.txt'].decode()
                                           for label in labels), 'Ordinary contribution gate failed')
                pairs.append({'case': case, 'point': point, 'mode': mode, 'exact_observation_parity': True,
                              'trace': ordinary_trace(raw_workers[labels[2]]['trace.dat'], case, point, mode),
                              'coverage': coverage(raw_workers[labels[0]], point, mode),
                              'independent_accuracy_qualified': False})
        replayed = {'passed': stage == 'ordinary', 'comparisons': pairs, 'failures': failures,
                    'interpretation_gate': 'Exact U/D/E observations, explicit full-pole bypass contract and actual fixed mode',
                    'independent_accuracy_qualified': False, 'performance_measurement': False}
        require(replayed == closed['parity'] == document('results/' + stage + '-parity.json'),
                'Replayed contribution/parity gate differs')
        require(closed['terminal_exit_code'] == (0 if stage == 'ordinary' else 1) and
                closed['stage']['passed'] == (stage == 'ordinary') and
                not closed['independent_accuracy_qualified'] and not closed['performance_measurement'],
                'Failed stress stage was promoted or stage scope changed')
        stage_summaries[stage] = {'workers': count, 'commands': 2 * count, 'outer_entries': 2 * count,
                                 'map_boundaries': 2 * count, 'equal_observation_groups': len(GROUPS[stage]),
                                 'contribution_gate_passed': stage == 'ordinary'}
        totals['commands'] += 2 * count
        totals['workers'] += count
        totals['outer_entries'] += 2 * count
        totals['map_boundaries'] += 2 * count
        totals['equal_observation_groups'] += len(GROUPS[stage])
        totals['ordinary_passed_groups'] += len(pairs)
        totals['stress_failed_contribution_groups'] += len(failures)
    require(totals['commands'] == 60 and totals['workers'] == 30 and
            totals['ordinary_passed_groups'] == 4 and totals['stress_failed_contribution_groups'] == 6,
            'Combined closed-stage accounting differs')
    return {'passed': True, 'scope': 'Closed raw evidence and recorded mapped identities only',
            **totals, 'stages': stage_summaries, 'stress_success_qualified': False,
            'independent_accuracy_stability_performance_qualified': False,
            'locale_identity_scope': 'Observed and hashed at READY/DONE, not prebound by exact hash',
            'stress_trace_interpretation_performed': False,
            'external_files_read': 0, 'native_processes_launched': 0}
