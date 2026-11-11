"""Portable closed-reference proof; invalid coefficient buffers stay uninterpreted."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = '.tirust-build/kinematic-invariant-reference-adjudication-proposal-1/'
WORK = '.tirust-build/kinematic-invariant-reference-adjudication-executor-1/'
SOURCE_SEAL = '0a0dba9baf12ff5a988e4d459141a48140f285a396658ee10f7f09099848dbd8'
PARENT = 'kinematic-invariant-amplitude-qualification-1'
PARENT_SEAL = 'c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4'
CLOSURES = {
    'build': '08a5b485954e20dc25fcffc2c26edddd73520b982e9daa380e1d5eca06709f22',
    'run': 'b94dca1a42b7acd2c835ce6b1bf04fe9036b0d64755ec1348240235a76b06462',
}
REVIEWS = {
    'results/build-terminal-review.json': '3f2ecb8eb90ac3b3e6b8730a4fbf17e0bba76cff57d624ae37666b4b6230bf72',
    'results/independent-build-review.json': '2e59e4146439991a9ceb1d7b53b6c36824bc82a51b341174befbbb051d5ebea9',
    'results/run-terminal-review.json': 'ae4d531856143c775222efc3464acd8da9efc43c456c41cec091e60a572da5a9',
    'results/independent-run-review.json': '7fd4c73c86880e95ecebe5dc8a1171db317e6d6803be36d863dcd6af1f75a710',
    'results/root-review.json': '1ec1a16b7deed7f80e81ad32cc17ac6405d536efffdf78a25d6222798e0e50bb',
}


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
    raw = (directory / 'evidence-index.json').read_bytes()
    require(expected is None or pin(raw)['sha256'] == expected, 'Parent seal differs')
    index = json.loads(raw)
    paths = list(directory.rglob('*'))
    require(not any(path.is_symlink() for path in paths), 'Symlink in packet')
    require({str(path.relative_to(directory)) for path in paths if path.is_file()} ==
            set(index['files']) | {'evidence-index.json'}, 'Packet file set differs')
    for name, identity in index['files'].items():
        raw = (directory / safe(name)).read_bytes()
        require(pin(raw) == identity, 'Stored payload differs: ' + name)
        text = gzip.decompress(raw) if name.endswith('.gz') else raw
        require(b'\0' not in text and not text.startswith((b'\x7fELF', b'!<arch>')), 'Native payload retained')
        text.decode('utf-8')

    def data(name):
        name = safe(name)
        compressed = index.get('compressed_text', {}).get(name)
        if compressed:
            require(compressed['path'] in index['files'], 'Unindexed compressed payload')
            raw = gzip.decompress((directory / safe(compressed['path'])).read_bytes())
            require(pin(raw) == {'bytes': compressed['raw_bytes'], 'sha256': compressed['raw_sha256']}, 'Raw payload differs')
            return raw
        require(name in index['files'], 'Unindexed payload: ' + name)
        return (directory / name).read_bytes()

    for name, identity in index.get('origins', {}).items():
        require(pin(data(name)) == {k: identity[k] for k in ('bytes', 'sha256')}, 'Original retained identity differs')
    return index, data, lambda name: json.loads(data(name))


def main():
    require(not any('PENDING' in name or 'PENDING' in seal for name, seal in REVIEWS.items()), 'Review still pending')
    index, data, document = packet(HERE)
    require(index['source_index_sha256'] == SOURCE_SEAL and index['dependencies'] == {PARENT: PARENT_SEAL} and
            index['closed_stages'] == CLOSURES, 'Retention scope differs')
    _, parent_data, parent_document = packet(HERE.parent / PARENT, PARENT_SEAL)
    require(pin(data('source/index.json'))['sha256'] == SOURCE_SEAL, 'Source seal differs')
    source = document('source/index.json')
    require(source['payload_count'] == len(source['files']) == 29 and source['payload_bytes'] == 238242 and
            source['native_actions_at_seal'] == 0, 'Original source scope differs')
    origins = {'source/index.json': SOURCE + 'index.json'}
    for name, identity in source['files'].items():
        require(pin(data('source/' + name)) == identity, 'Source payload differs')
        origins['source/' + name] = SOURCE + name
    counts = {}
    for stage, seal in CLOSURES.items():
        logical = 'results/' + stage + '-closure.json'
        require(pin(data(logical))['sha256'] == seal, 'Closed stage seal differs')
        closed = document(logical)
        origins[logical] = WORK + logical
        require(closed['source_index_sha256'] == SOURCE_SEAL and closed['stage'] == stage and
                closed['postcheck_failure'] is None and closed['host_stability_qualified'] is False and
                closed['performance_measurement'] is False, 'Closed stage authority/scope differs')
        require(closed['passed'] is (stage == 'build') and closed['failure'] == (None if stage == 'build' else
                "ValueError('Reference eligibility/agreement failed; evidence retained')") and
                closed['reviewed_build_sha256'] == (None if stage == 'build' else CLOSURES['build']),
                'Closed stage success/failure was changed')
        commands = loaders = 0
        for name, identity in closed['payloads'].items():
            raw = data('results/' + safe(name))
            require(pin(raw) == identity, 'Closed payload differs')
            origins['results/' + name] = WORK + 'results/' + name
            if name.startswith('execution/') and name.endswith('.json'):
                receipt = json.loads(raw)
                commands += 1
                loaders += '--list' in receipt['argv']
                require(receipt['returncode'] == 0 and receipt['exception'] is None and receipt['cleanup_exception'] is None and
                        receipt['cleanup']['survivors'] == [], 'Owned command failed or cleanup incomplete')
                log = str(Path(name).with_suffix('.txt'))
                require(log in closed['payloads'] and pin(data('results/' + log))['sha256'] == receipt['log_sha256'],
                        'Owned command/log join differs')
        require((commands, loaders, len(closed['payloads'])) == ((64, 1, 138) if stage == 'build' else (4, 2, 43)) and
                commands == closed['owned_command_count'] and loaders == closed['loader_query_count'], 'Command scope differs')
        require(data('results/' + stage + '-before.json') == data('results/' + stage + '-after.json') ==
                data('results/build-before.json'), 'Protected state differs across closed stages')
        counts[stage] = {'payloads': len(closed['payloads']), 'commands': commands, 'loader_queries': loaders}
    for name, seal in REVIEWS.items():
        require(pin(data(name))['sha256'] == seal, 'Closed review differs')
        origins[name] = WORK + name
    for stage in ('build', 'run'):
        terminal = document('results/' + stage + '-terminal-review.json')
        require(terminal['source_index_sha256'] == SOURCE_SEAL and
                terminal[stage + '_closure'] == pin(data('results/' + stage + '-closure.json')) and
                terminal['terminal_exit_code'] == (0 if stage == 'build' else 1) and
                terminal['protected_before'] == pin(data('results/' + stage + '-before.json')) and
                terminal['protected_after'] == pin(data('results/' + stage + '-after.json')) and
                terminal['protected_equal'] and terminal['remaining_owned_identities'] == [] and
                terminal['shared_lock_free_after_terminal'], 'Terminal authority/cleanup differs')
        independent = document('results/independent-' + stage + '-review.json')
        require(independent['source_index_sha256'] == SOURCE_SEAL and
                independent[stage + '_closure'] == pin(data('results/' + stage + '-closure.json')) and
                independent['terminal_review'] == pin(data('results/' + stage + '-terminal-review.json')) and
                independent['native_actions_by_review'] == 0, 'Independent review authority differs')
    terminal = document('results/run-terminal-review.json')
    require(terminal['execution_complete'] and terminal['all_coefficients_comparisons_blocked'] and
            terminal['reference_gate_passed'] is False and terminal['bootstrap_numeric_comparison_performed'] is False and
            terminal['failed_coefficients_numerically_decoded'] is False and terminal['retry_performed'] is False and
            terminal['further_native_invocation'] is False and terminal['unexecuted_requests'] == [],
            'Invalid reference buffers were promoted')
    root_review = document('results/root-review.json')
    require(root_review['source_index'] == pin(data('source/index.json')) and
            root_review['closed_stages'] == {stage: pin(data('results/' + stage + '-closure.json')) for stage in CLOSURES} and
            root_review['closed_payloads'] == 181 and root_review['owned_commands'] == {'build': 64, 'run': 4} and
            root_review['passed_integrity_review_of_unqualified_reference_run'] and
            root_review['reference_gate_passed'] is False and root_review['coefficient_comparisons_interpreted'] is False and
            root_review['new_native_calls'] == 0 and root_review['performance_measurement'] is False,
            'Root review authority/qualification differs')
    require({name: identity['path'] for name, identity in index['origins'].items()} == origins, 'Explicit closed origin set differs')

    def helper(name):
        spec = importlib.util.spec_from_file_location('legacy_reference_' + name, HERE / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    build = helper('build_checks').verify_build(data, document, parent_data, parent_document)
    workers = helper('worker_checks').verify_workers(data, document, parent_data, parent_document)
    print(json.dumps({'payloads': len(index['files']), 'closed_counts': counts, 'build': build, 'workers': workers,
        'reference_gate_passed': False, 'coefficient_comparisons_performed': False,
        'bootstrap_numeric_comparisons_performed': False, 'checked_host_stability_qualified': False,
        'performance_qualified': False, 'actual_internal_working_bits_proven': False,
        'fresh_269_source_reparse': False, 'new_native_executions': 0, 'external_binary_reads': 0}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
