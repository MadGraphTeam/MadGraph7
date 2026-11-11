"""Pure verification of retained closed-build text, never external/native bytes."""
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = '.tirust-build/kinematic-provenance-current-executor-1/'
PINS = {
    'results/build-terminal-closure.json': '564688cfb10fa7a05f4349600209299751a91159632047377b59ebe1d7fe2185',
    'results/root-build-review.json': '36c3ec42425ee8cd96c72a8ecee9efffc95c00f61e7b0462bc2dbdc5e1204559',
    'results/build-postterminal-ownership.json': 'e4849a7dbce603f4659ce878865e4bc3bd8bb8b2c16a3bd7aa410af7c648770f',
    'source/index.json': '7ea709b48d8de462b754fabfb47dcbd5d21d7847458382f5844190028e8c1bed',
    'results/derived/index.json': '25c2114756ed3eb968c311945a27d175c5c5a0af54d4ae21158abc8ed42a4184',
    'build-admission-1/index.json': 'c30fefa33c13dc04cbeed5b12a6a44284d1fc4858c12aa569382ea8a8f8bf813',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    index = json.loads((HERE / 'evidence-index.json').read_bytes())
    paths = list(HERE.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Symlink in portable packet')
    require({str(p.relative_to(HERE)) for p in paths if p.is_file()} ==
            set(index['files']) | {'evidence-index.json'}, 'Packet file set changed')
    for name, pin in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe stored path')
        raw = (HERE / name).read_bytes()
        require(len(raw) == pin['bytes'] and sha(raw) == pin['sha256'], 'Stored bytes changed: ' + name)
        text = gzip.decompress(raw) if name.endswith('.gz') else raw
        require(not text.startswith((b'\x7fELF', b'!<arch>')) and b'\0' not in text, 'Native payload retained')
        text.decode('utf-8')

    def data(name):
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe logical path')
        if name in index['compressed_text']:
            pin = index['compressed_text'][name]
            raw = gzip.decompress((HERE / pin['path']).read_bytes())
            require(len(raw) == pin['raw_bytes'] and sha(raw) == pin['raw_sha256'], 'Decompressed bytes changed')
            return raw
        return (HERE / name).read_bytes()

    def document(name):
        return json.loads(data(name))

    def match(name, pin):
        raw = data(name)
        checksum = pin if isinstance(pin, str) else pin['sha256']
        require(sha(raw) == checksum, 'Recorded input identity mismatch: ' + name)
        if isinstance(pin, dict):
            require(len(raw) == pin['bytes'], 'Recorded input size mismatch: ' + name)
        return raw

    for name, pin in index['compressed_text'].items():
        require(pin['path'] in index['files'], 'Unindexed compressed text')
        data(name)
    for name, pin in index['origins'].items():
        match(name, pin)
    for name, pin in PINS.items():
        match(name, pin)
    source = document('source/index.json')
    derived = document('results/derived/index.json')
    guard_index = document('build-admission-1/index.json')
    for directory, packet, count in [('source', source, 65), ('results/derived', derived, 17),
                                     ('build-admission-1', guard_index, 5)]:
        require(len(packet['files']) == count, 'Source packet file count')
        for name, pin in packet['files'].items():
            match(directory + '/' + name, pin)
    match('ancestry/parent-proposal-index.json', source['parent_proposal_index_sha256'])
    helpers = document('source/helper-bindings.json')
    require(len(helpers['files']) == 26, 'Helper scope changed')
    for name, pin in helpers['files'].items():
        match('helpers/' + name, pin)
    context = document('helpers/.tirust-build/kinematic-provenance-current-proposal-1/proposal/context-bindings.json')
    for key in ['ew_compilation', 'historical_packet', 'campaign_manifest']:
        match('provenance/' + context[key]['path'], context[key])
    bindings = document('build-admission-1/bindings.json')
    for name, pin in bindings['evidence'].items():
        match('provenance/' + name, pin)
    generation = document('results/generation.json')
    match('results/generation.json', derived['generated_source_receipt'])
    require(generation['generation_only'] and generation['amplitude_evaluations'] == 0 and
            generation['source_anchor'] == context['production_anchor'], 'Generation provenance changed')
    require(generation['raw_sources'] == document('results/derived/cms-source-bindings.json')['files'],
            'Derived CMS source inventory differs from generation')

    closure = document('results/build-terminal-closure.json')
    require(closure['terminal_exit_code'] == 0 and closure['command_count'] == len(closure['commands']) == 258 and
            closure['all_commands_exit_zero'] and closure['no_retry'] and closure['amplitude_evaluations'] == 0 and
            not closure['ordinary_stress_admitted'] and not closure['cleanup_exceptions'] and
            not closure['owned_survivors'] and closure['protected_before_after_equal'], 'Build did not close cleanly')
    require(closure['stage'] == document('results/stage-build.json') and closure['stage']['passed'] and
            not closure['stage']['performance_measurement'], 'Build stage mismatch')
    guard_result = document('results/build-admission-1/result.json')
    require(guard_result == closure['guard_result'] and guard_result['passed'] and
            guard_result['failure'] is None and guard_result['postcheck_failure'] is None and
            guard_result['amplitude_evaluations'] == 0 and not guard_result['ordinary_or_stress_admitted'],
            'Guard did not pass')
    for key, name in [('source_index_sha256', 'source/index.json'),
                      ('derived_index_sha256', 'results/derived/index.json'),
                      ('guard_index_sha256', 'build-admission-1/index.json')]:
        require(guard_result[key] == PINS[name], 'Guard source seal mismatch')
    before = document('results/build-admission-1/before.json')
    require(before == document('results/build-admission-1/after.json'), 'Protected inputs changed')
    for key, name in [('source_index', 'source/index.json'), ('derived_index', 'results/derived/index.json'),
                      ('guard_index', 'build-admission-1/index.json'), ('generation', 'results/generation.json')]:
        match(name, before[key])
    require(before['supplemental_providers'] == bindings['supplemental_providers'],
            'Protected provider supplements differ from admission')

    commands, identities = {}, set()
    for original, pins in closure['commands'].items():
        require(original.startswith(WORK + 'results/execution/'), 'Command outside closed scope')
        name = original[len(WORK):]
        raw = match(name, pins['receipt'])
        log_name = str(Path(name).with_suffix('.txt'))
        log = match(log_name, pins['log'])
        row = json.loads(raw)
        require(row['returncode'] == 0 and row['exception'] is None and row['cleanup_exception'] is None and
                not row['cleanup']['survivors'] and sha(log) == row['log_sha256'], 'Owned command failed')
        require(row['cpu'] == '12,13' and row['cpu_exclusive'] is False, 'Affinity scope changed')
        commands[original] = row
        for process in row['cleanup']['observed_owned_processes']:
            identities.add((process['pid'], process['start_time_ticks'], process['session']))
    retained_commands = {pin['path'] for name, pin in index['origins'].items()
                         if pin['path'].startswith(WORK + 'results/execution/') and pin['path'].endswith('.json')}
    require(retained_commands == set(closure['commands']), 'Extra/unclosed command receipt retained')
    require(len(closure['actual_loader_queries']) == 4, 'Wrong actual loader-query count')
    queries_by_executable = {}
    for original, argv in closure['actual_loader_queries'].items():
        require(original in commands and commands[original]['argv'] == argv and
                argv[:3] == ['taskset', '-c', '12,13'] and argv[-2] == '--list', 'Actual loader query mismatch')
        require(argv[-1] not in queries_by_executable, 'Repeated executable loader query')
        queries_by_executable[argv[-1]] = original

    labels = {'cms-unmodified', 'cms-instrumented', 'ew-unmodified', 'ew-instrumented'}
    require(set(closure['builds']) == labels, 'Wrong four-build scope')
    for label, pin in closure['builds'].items():
        prefix = 'results/builds/' + label
        build = json.loads(match(prefix + '/build.json', pin['receipt']))
        prelink = document(prefix + '/prelink.json')
        precompile = document(prefix + '/before-compile.json')
        guard = document('results/build-admission-1/' + label + '-runtime.json')
        require(build['executable'] == pin['executable'] and build['executable_pin'] == pin['executable_pin'] ==
                guard['executable_pin'] and build['unmodified_numerical_arithmetic'], 'Executable identity mismatch')
        require(precompile['sources'] == build['sources'] and precompile['compiler_flags'] == build['compiler_flags'],
                'Compilation input receipt mismatch')
        require(prelink['completed_before_link'] and prelink['bound'] == build['prelink_bound'], 'Prelink mismatch')
        match(prefix + '/prelink.json', guard['prelink_pin'])
        for path, loaded in build['actual_link_loads'].items():
            require(prelink['bound'].get(path) == loaded['sha256'], 'Unbound recorded link input')
        require(guard['passed'] and guard['failure'] is None and guard['runtime'] == build['runtime'] and
                guard['admitted_supplements'] == bindings['supplemental_providers'], 'Loader guard failed')
        runtime = build['runtime']
        require(len(runtime['files']) == pin['runtime_providers'] == 15 and
                guard['actual_interpreter'] == runtime['loader'] == bindings['loader'], 'Interpreter/provider count')
        require(runtime['loader_sha256'] == bindings['supplemental_providers'][runtime['loader']]['sha256'],
                'Interpreter hash mismatch')
        query = queries_by_executable[build['executable']]
        argv = closure['actual_loader_queries'][query]
        require(argv[3] == runtime['loader'], 'Query used a different loader')
        require(data(str(Path(query[len(WORK):]).with_suffix('.txt'))).decode() == runtime['loader_listing'],
                'Raw --list output differs from recorded resolution')
        for path, checksum in runtime['files'].items():
            supplemental = bindings['supplemental_providers'].get(path, {}).get('sha256')
            require(checksum == prelink['bound'].get(path) or checksum == supplemental,
                    'Recorded provider was not prebound')
    ownership = document('results/build-postterminal-ownership.json')
    require(ownership['observed_process_identities'] == len(identities) == 661 and
            not ownership['matching_live_owned_processes'] and ownership['shared_lock_nonblocking_acquire_release'] and
            ownership['terminal_exit_code'] == closure['terminal_exit_code'] and
            ownership['terminal_session_id'] == closure['terminal_session_id'], 'Ownership closure mismatch')
    review = document('results/root-build-review.json')
    require(review['build_closure_sha256'] == PINS['results/build-terminal-closure.json'] and
            review['commands_checked'] == 258 and review['actual_loader_queries'] == 4 and
            review['executables_checked'] == 4 and review['providers_per_executable'] == 15 and
            review['amplitudes_executed'] == 0 and not review['ordinary_or_stress_admitted'], 'Root review mismatch')
    summary = document('qualification-summary.json')
    require(summary['amplitude_evaluations'] == 0 and not summary['ordinary_or_stress_evidence_retained'] and
            not summary['observational_parity_qualified'] and not summary['accuracy_qualified'] and
            not summary['performance_qualified'] and not summary['external_binary_bytes_retained'] and
            not summary['external_binary_bytes_rehashed_by_capture'], 'Scope was broadened')
    print(json.dumps({'portable_closed_build_verified': True, 'commands': 258, 'executables_recorded': 4,
                      'loader_resolution_queries': 4, 'providers_per_executable': 15,
                      'recorded_process_identities': 661, 'amplitude_evaluations_in_closed_stage': 0,
                      'external_binary_reads_by_verifier': 0, 'new_processes_launched': 0,
                      'ordinary_stress_parity_accuracy_performance_qualified': False}, indent=2))


if __name__ == '__main__':
    main()
