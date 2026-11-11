"""Portable text-only replay of one closed, strictly gated matched-input experiment."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
AMP = 'kinematic-invariant-amplitude-qualification-1'
LEGACY = 'kinematic-legacy-reference-adjudication-1'
WORK = '.tirust-build/kinematic-matched-native-input-executor-2/'
OLD_WORK = '.tirust-build/kinematic-matched-native-input-executor-1/'
COMPARISON = 'af4b58a379ac35a78066a1bcad38ed3f4fe9e16e6b0d4583dbd4e6c940eba6d5'


def require(value, message):
    if not value:
        raise ValueError(message)


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def safe(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute() and
            all(part not in ('', '.', '..') for part in name.split('/')), 'Unsafe packet path')
    return name


def packet(directory, expected=None):
    require(not directory.is_symlink(), 'Symlink packet')
    raw = (directory / 'evidence-index.json').read_bytes()
    require(expected is None or pin(raw)['sha256'] == expected, 'Sibling seal differs')
    index = json.loads(raw)
    paths = list(directory.rglob('*'))
    require(not any(path.is_symlink() for path in paths), 'Symlink payload')
    require({str(path.relative_to(directory)) for path in paths if path.is_file()} ==
            set(index['files']) | {'evidence-index.json'}, 'Packet file set differs')
    stored = {}
    for name, identity in index['files'].items():
        raw = (directory / safe(name)).read_bytes()
        require(pin(raw) == identity, 'Stored payload differs: ' + name)
        text = gzip.decompress(raw) if name.endswith('.gz') else raw
        require(b'\0' not in text and not text.startswith((b'\x7fELF', b'!<arch>')), 'Native bytes retained')
        text.decode('utf-8')
        stored[name] = raw

    def data(name):
        name = safe(name)
        compressed = index.get('compressed_text', {}).get(name)
        if compressed:
            require(compressed['path'] in stored, 'Unindexed compressed text')
            raw = gzip.decompress(stored[compressed['path']])
            require(pin(raw) == {'bytes': compressed['raw_bytes'], 'sha256': compressed['raw_sha256']},
                    'Compressed text identity differs')
            return raw
        require(name in stored, 'Unindexed text: ' + name)
        return stored[name]

    for name, identity in index.get('origins', {}).items():
        require(pin(data(name)) == {key: identity[key] for key in ('bytes', 'sha256')}, 'Origin identity differs')
    return index, data, lambda name: json.loads(data(name))


def module(name):
    spec = importlib.util.spec_from_file_location('matched_retention_' + name, HERE / (name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def parity(data, document, amplitude_data, amplitude_document, build_summary):
    receipt = document('results/prior-candidate-word-parity.json')
    comparison = document('results/matched-comparison.json')
    prior_comparison = amplitude_document('results/root-point20-comparison.json')
    prior_closed = amplitude_document('results/stress-closure.json')
    current_id = 'cms-p20-mode4-lib9-matched'
    prior_id = 'cms-p20-mode4-lib9-candidate'
    require(receipt['current_worker'] == current_id and receipt['prior_worker'] == prior_id and
            receipt['passed'] and prior_closed['passed'] is False and
            prior_closed['failure'] == receipt['eligibility']['prior_stress_aggregate_failure'] and
            receipt['eligibility']['prior_stress_aggregate_passed'] is False and
            prior_comparison['candidate_gate_passed'] is False, 'Prior failure was reclassified')
    current = comparison['observations']['9']
    prior = prior_comparison['observations'][prior_id]
    require(all(comparison[key] for key in ('passed', 'native_input_gate_passed', 'native_inputs_finite',
            'outer_output_gate_passed', 'bootstrap_controls_passed', 'measurement_interpretation_allowed')) and
            current['id'] == current_id and current['structural_complete'] and current['eligible'] and prior['eligible'] and
            current['failures'] == prior['failures'] == [] and
            current['incomplete_contribution_warnings'] == prior['incomplete_contribution_warnings'] == [],
            'Selected output parity eligibility failed')
    for observation in (current, prior):
        require(observation['protocol_complete'] and
                observation['outer_entries_observed'] == observation['ready_done_maps'] == 2 and
                observation['request']['backend'] == 9 and observation['request']['requested_mode'] == 4 and
                observation['request']['point'] == 20 and observation['request']['expected_fixed_unchecked_return'] == 120,
                'Selected output protocol differs')
    # Authenticate every explicit closed payload without numerically interpreting other rows.
    for name, identity in prior_closed['payloads'].items():
        require(pin(amplitude_data('results/' + safe(name))) == identity, 'Parent closed payload differs')
    for prefix, index_name, callback, count in (
        ('science/', 'science/index.json', data, 35),
        ('controller/', 'controller/index.json', data, 31),
        ('proposal/', 'proposal/index.json', amplitude_data, 55)):
        source = json.loads(callback(index_name))
        require(len(source['files']) == count, 'Parity source inventory differs')
        for name, identity in source['files'].items():
            require(pin(callback(prefix + safe(name))) == identity, 'Parity source bytes differ')
    routes = (
        (WORK, '', data),
        ('.tirust-build/kinematic-matched-native-input-proposal-1/', 'science/', data),
        ('.tirust-build/kinematic-matched-native-input-controller-2/', 'controller/', data),
        ('.tirust-build/kinematic-invariant-amplitude-executor-2/', '', amplitude_data),
        ('.tirust-build/kinematic-invariant-amplitude-proposal-2/', 'proposal/', amplitude_data))
    require(len(receipt['evidence_pins']) == 22, 'Parity pin count differs')
    for record in receipt['evidence_pins']:
        matches = [(prefix, logical, callback) for prefix, logical, callback in routes if record['path'].startswith(prefix)]
        require(len(matches) == 1, 'Parity path outside declared packets')
        prefix, logical, callback = matches[0]
        require(pin(callback(logical + safe(record['path'][len(prefix):]))) ==
                {key: record[key] for key in ('bytes', 'sha256')}, 'Parity explicit pin differs')
    current_base, prior_base = 'results/runs/' + current_id + '/', 'results/runs/' + prior_id + '/'
    for observation, base, callback in ((current, current_base, data), (prior, prior_base, amplitude_data)):
        for name, identity in observation['source_evidence'].items():
            require(pin(callback(base + safe(name))) == identity, 'Selected observation file differs')
    current_spec, prior_spec = document(current_base + 'specification.json'), amplitude_document(prior_base + 'specification.json')
    for key, identity in receipt['input_pins_equal'].items():
        require(current_spec[key] == prior_spec[key] == identity, 'Parity point/card identity differs')
    for key, filename in (('bootstrap_pin', 'inputs/cms-point1.dat'), ('point_pin', 'inputs/cms-point20.dat'),
                          ('physical_card', 'inputs/cms-param_card.dat'), ('numerical_card', 'backend9-mode4-MadLoopParams.dat')):
        require(pin(data('controller/' + filename)) == receipt['input_pins_equal'][key], 'Parity retained input differs')
    runtimes = []
    for specification in (current_spec, prior_spec):
        providers = dict(specification['expected_runtime'])
        require(providers.pop(specification['executable']) == specification['executable_pin']['sha256'], 'Executable provider differs')
        runtimes.append(providers)
    proof = receipt['source_binding_proof']
    require(runtimes[0] == runtimes[1] and len(runtimes[0]) == proof['non_executable_runtime_providers'] == 15 and
            proof['all_non_executable_runtime_provider_pins_identical'], 'Historical runtime provider identities differ')
    tirust = '/shared/external/madloop_tirust/madgraph7_madloop_tirust/' + proof['same_TIRust_runtime_provider']['path']
    require(runtimes[0][tirust] == proof['same_TIRust_runtime_provider']['sha256'] ==
            'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9', 'TIRust provider identity differs')
    # The retained successful rows and integer-only raw parser gate all coefficient words first.
    def rows(raw):
        result = []
        for line in raw.decode().splitlines():
            if not line or line.startswith('#'):
                continue
            fields = line.split()
            require(len(fields) == 10 and all(re.fullmatch('[0-9A-F]{16}', word) for word in fields[5:]), 'Bad output word row')
            numbers = list(map(int, fields[:5])); words = fields[5:]
            require(all((int(word, 16) >> 52) & 2047 != 2047 for word in words), 'Nonfinite output encoding')
            result.append(dict(zip(('phase', 'call', 'point', 'order', 'code'), numbers), words=words))
        require([(row['phase'], row['call'], row['point'], row['order'], row['code']) for row in result] ==
                [(0, 1, 1, 0, 120), (0, 1, 1, 1, 120), (1, 2, 20, 0, 120), (1, 2, 20, 1, 120)], 'Output row domain differs')
        return result
    current_raw, prior_raw = data(current_base + 'driver-bits.dat'), amplitude_data(prior_base + 'driver-bits.dat')
    require(current_raw == prior_raw and data(current_base + 'driver-state.dat') == amplitude_data(prior_base + 'driver-state.dat'),
            'Raw eligible output/state bytes differ')
    current_rows, prior_rows = rows(current_raw), rows(prior_raw)
    require(current_rows == current['raw_rows'] and prior_rows == prior['raw_rows'], 'Observation/raw output join differs')
    for observation in (current, prior):
        after = [state for state in observation['states'] if state['stage'] == 'A']
        require(len(after) == 2 and all(state['actual_mode'] == state['requested_mode'] == 4 and state['code'] == 120 and
                state['fpe_dp'] == state['fpe_qp'] == 0 for state in after), 'Selected software flags invalid')
    words, ancillary = [], []
    for left, right in zip(current_rows, prior_rows):
        base = {key: left[key] for key in ('phase', 'point', 'order')}
        for column, label in enumerate(('Born', 'finite', 'single_pole', 'double_pole')):
            words.append({**base, 'coefficient': label, 'current_word': left['words'][column],
                          'prior_word': right['words'][column], 'equal': left['words'][column] == right['words'][column]})
        ancillary.append({**base, 'current_word': left['words'][4], 'prior_word': right['words'][4],
                          'equal': left['words'][4] == right['words'][4]})
    require(words == receipt['word_rows'] and ancillary == receipt['ancillary_accuracy_words'] and
            all(row['equal'] for row in words + ancillary), 'Parity receipt raw words differ')
    require(proof['current_controller_source_index_sha256'] == pin(data('controller/index.json'))['sha256'] and
            proof['current_scientific_source_index_sha256'] == pin(data('science/index.json'))['sha256'] and
            proof['prior_candidate_source_index_sha256'] == pin(amplitude_data('proposal/index.json'))['sha256'] and
            proof['diagnostic_executable_body_sha256'] == build_summary['source_proof']['diagnostic_body_sha256'] ==
            'd5b6e71ed15d6e56c54dcc2e3304f96afeb1e5f44f3e7763612d1ea89f02e35f' and
            build_summary['source_proof']['postdiagnostic_contraction_tail_unchanged'] and
            build_summary['source_proof']['diagnostic_recorder_calls_removed'] == 2,
            'Parity source ancestry differs')
    for key in ('floating_point_decodes', 'new_library_loads', 'new_native_calls', 'failed_reference_coefficients_decoded'):
        require(receipt[key] == 0, 'Parity report interpreted failed coefficients or launched native work')
    for key in ('accuracy_qualified', 'normal_host_stability_qualified', 'performance_measurement'):
        require(receipt[key] is False, 'Parity scope widened')
    return {'measurement_words_equal': 8, 'bootstrap_words_equal': 8, 'ancillary_words_equal': 4,
            'floating_point_decodes_for_word_parity': 0, 'prior_aggregate_failure_preserved': True,
            'historical_native_input_word_parity_proven': False, 'retained_provider_identity_matches': 15}


def main():
    index, data, document = packet(HERE)
    config = module('capture')  # Import local constants only; capture.main is never called.
    require(index['dependencies'] == config.DEPENDENCIES and index['source_indices'] ==
            {prefix: seal for prefix, (_, seal) in config.SOURCES.items()} and index['closed_stages'] == config.CLOSURES and
            index['failed_predecessor_build'] == config.FAILED, 'Packet scope differs')
    _, amplitude_data, amplitude_document = packet(HERE.parent / AMP, config.DEPENDENCIES[AMP])
    _, legacy_data, legacy_document = packet(HERE.parent / LEGACY, config.DEPENDENCIES[LEGACY])
    origins = {}
    for prefix, (origin, seal) in config.SOURCES.items():
        logical = prefix + '/index.json'
        require(pin(data(logical))['sha256'] == seal, 'Source seal differs')
        source = document(logical)
        require(len(source['files']) == {'science': 35, 'controller': 31, 'predecessor/controller': 27}[prefix], 'Source count differs')
        origins[logical] = origin + '/index.json'
        for name, identity in source['files'].items():
            require(pin(data(prefix + '/' + safe(name))) == identity, 'Source payload differs')
            origins[prefix + '/' + name] = origin + '/' + name
    counts = {}
    stages = [('predecessor/', OLD_WORK, 'build', config.FAILED, False),
              ('', WORK, 'build', config.CLOSURES['build'], True), ('', WORK, 'run', config.CLOSURES['run'], True)]
    for prefix, origin, stage, seal, success in stages:
        logical = prefix + 'results/' + stage + '-closure.json'
        require(pin(data(logical))['sha256'] == seal, 'Closed stage differs')
        closed = document(logical); origins[logical] = origin + 'results/' + stage + '-closure.json'
        source_seal = config.SOURCES['predecessor/controller' if prefix else 'controller'][1]
        require(closed['source_index_sha256'] == source_seal and closed['stage'] == stage and
                closed['passed'] is success and closed['postcheck_failure'] is None and
                closed['host_stability_qualified'] is False and closed['performance_measurement'] is False and
                closed['reviewed_build_sha256'] == (config.CLOSURES['build'] if stage == 'run' else None), 'Closed stage scope differs')
        require(closed['failure'] == (None if success else
                "ValueError(\"Unresolved numerical compilation recipes: ['amplitude_support.f90', 'caller_bounds.f90', 'certificate.f90']\")"),
                'Historical stage failure changed')
        commands = loaders = 0
        for name, identity in closed['payloads'].items():
            raw = data(prefix + 'results/' + safe(name)); require(pin(raw) == identity, 'Closed payload differs')
            origins[prefix + 'results/' + name] = origin + 'results/' + name
            if name.startswith('execution/') and name.endswith('.json'):
                receipt = json.loads(raw); commands += 1; loaders += '--list' in receipt['argv']
                require(receipt['returncode'] == 0 and receipt['exception'] is None and receipt['cleanup_exception'] is None and
                        receipt['cleanup']['survivors'] == [], 'Owned command incomplete')
                log = str(Path(name).with_suffix('.txt'))
                require(log in closed['payloads'] and pin(data(prefix + 'results/' + log))['sha256'] == receipt['log_sha256'],
                        'Closed command/log join differs')
        expected = (2, 0, 8) if prefix else (64, 1, 138) if stage == 'build' else (4, 2, 43)
        require((commands, loaders, len(closed['payloads'])) == expected and closed['owned_command_count'] == commands and
                closed['loader_query_count'] == loaders, 'Closed stage command inventory differs')
        require(data(prefix + 'results/' + stage + '-before.json') == data(prefix + 'results/' + stage + '-after.json') ==
                data(prefix + 'results/build-before.json'), 'Protected state changed')
        counts[prefix + stage] = {'payloads': len(closed['payloads']), 'commands': commands, 'loader_queries': loaders}
    for prefix, origin, extras in (('predecessor/', OLD_WORK, config.OLD_EXTRAS), ('', WORK, config.EXTRAS)):
        for name, seal in extras.items():
            require(pin(data(prefix + name))['sha256'] == seal, 'Closed extra receipt differs')
            origins[prefix + name] = origin + name
    require({name: record['path'] for name, record in index['origins'].items()} == origins, 'Explicit retained origin set differs')
    require(pin(data('results/matched-comparison.json'))['sha256'] == COMPARISON, 'Exact comparison differs')
    for stage in ('build', 'run'):
        terminal = document('results/' + stage + '-terminal-review.json')
        require(terminal[stage + '_closure'] == pin(data('results/' + stage + '-closure.json')) and terminal['terminal_exit_code'] == 0 and
                terminal['passed'] and terminal['protected_before'] == pin(data('results/' + stage + '-before.json')) and
                terminal['protected_after'] == pin(data('results/' + stage + '-after.json')) and terminal['protected_equal'] and
                terminal['remaining_owned_identities'] == [] and terminal['shared_lock_free_after_terminal'] and
                terminal['root_actual_file_review'] == pin(data('results/root-' + stage + '-review.json')), 'Terminal joins/cleanup differ')
    build_terminal, run_terminal = document('results/build-terminal-review.json'), document('results/run-terminal-review.json')
    require(build_terminal['source_index'] == pin(data('controller/index.json')) and
            build_terminal['build_receipt'] == pin(data('results/builds/matched/build.json')) and
            build_terminal['fortran_main_calls'] == build_terminal['tensor_calls'] == 0 and build_terminal['run_invoked'] is False and
            run_terminal['comparison'] == pin(data('results/matched-comparison.json')) and
            (run_terminal['actual_native_calls'], run_terminal['actual_outer_entries'], run_terminal['actual_ready_done_maps']) == (104, 4, 4) and
            all(run_terminal['gates'].values()) and run_terminal['no_further_native_invocations'], 'Terminal qualification differs')
    ib, ir = document('results/independent-build-review.json'), document('results/independent-run-review.json')
    require(ib['source_index'] == ir['controller_source'] == pin(data('controller/index.json')) and
            ib['build_closure'] == ir['closed_build'] == pin(data('results/build-closure.json')) and
            ib['build_receipt'] == pin(data('results/builds/matched/build.json')) and
            ib['terminal_review'] == pin(data('results/build-terminal-review.json')) and
            ir['terminal_review'] == pin(data('results/run-terminal-review.json')) and
            ir['closed_run'] == pin(data('results/run-closure.json')) and ir['comparison'] == pin(data('results/matched-comparison.json')) and
            ib['blocking_findings'] == ir['blocking_findings'] == [] and ib['native_actions_by_reviewer'] == ir['native_actions_by_reviewer'] == 0,
            'Independent review authority differs')
    for review in (ib, ir):
        require(review['protected_before'] == review['protected_after'] == pin(data('results/build-before.json')) and
                review['protected_snapshots_identical'], 'Independent protected identity differs')
    require(ir['exact_comparator_replay']['canonical_serialized_sha256'] == COMPARISON and ir['exact_comparator_replay']['passed'] and
            ir['accuracy_qualified'] is False and ir['normal_host_stability_qualified'] is False and ir['performance_measurement'] is False,
            'Independent run scope widened')
    source_admission, run_admission = document('results/source-admission.json'), document('results/run-admission.json')
    require(source_admission['controller_index'] == pin(data('controller/index.json')) and
            source_admission['scientific_index_sha256'] == config.SOURCES['science'][1] and
            source_admission['previous_failed_build_sha256'] == config.FAILED and
            source_admission['build_invocations_admitted'] == 1 and source_admission['run_invocations_admitted'] == 0 and
            source_admission['independent_review'] == pin(data('controller/audit/successor-review.json')) and
            run_admission['controller_index_sha256'] == config.SOURCES['controller'][1] and
            run_admission['reviewed_build_sha256'] == config.CLOSURES['build'] and
            run_admission['build_terminal_review'] == pin(data('results/build-terminal-review.json')) and
            run_admission['root_actual_file_review'] == pin(data('results/root-build-review.json')), 'Separate admissions differ')
    rb, rr = document('results/root-build-review.json'), document('results/root-run-review.json')
    require(rb['build_closure_sha256'] == config.CLOSURES['build'] and rb['closed_payloads'] == 138 and rb['owned_commands'] == 64 and
            rb['passed'] and rb['physics_calls'] == 0 and rb['numerical_accuracy_qualified'] is False and
            rr['run_closure_sha256'] == config.CLOSURES['run'] and rr['comparison_sha256'] == COMPARISON and
            rr['closed_payloads'] == 43 and rr['native_calls'] == 104 and rr['native_input_pairs'] == 52 and
            rr['outer_calls'] == rr['runtime_maps'] == rr['owned_commands'] == 4 and rr['passed'] and rr['native_actions_by_root'] == 0 and
            rr['independent_accuracy_qualified'] is False and rr['normal_host_stability_qualified'] is False and rr['performance_qualified'] is False,
            'Root review authority or scope differs')
    callbacks = (data, document, amplitude_data, amplitude_document, legacy_data, legacy_document)
    build = module('build_checks').verify_build(*callbacks)
    workers = module('worker_checks').verify_workers(*callbacks)
    raw_parity = parity(data, document, amplitude_data, amplitude_document, build)
    print(json.dumps({'payloads': len(index['files']), 'closed_counts': counts, 'build': build, 'workers': workers,
        'prior_candidate_word_parity': raw_parity, 'retained_receipts_authenticated': True,
        'external_artifact_hashes_rechecked': False, 'historical_full_source_DAG_reparsed': False,
        'normal_checked_stability_qualified': False, 'independent_accuracy_qualified': False,
        'production_promotion': False, 'performance_qualified': False,
        'new_native_executions': 0, 'external_binary_reads': 0}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
