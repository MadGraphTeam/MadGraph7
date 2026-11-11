"""Portable closed amplitude evidence verification; no native or binary reads."""
import ast
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORK = '.tirust-build/kinematic-invariant-amplitude-executor-2/'
OLD_WORK = '.tirust-build/kinematic-invariant-amplitude-executor-1/'
SOURCES = {
    'proposal': ('.tirust-build/kinematic-invariant-amplitude-proposal-2/', 'ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74'),
    'predecessor/proposal': ('.tirust-build/kinematic-invariant-amplitude-proposal-1/', '286ecfe5f5fab199dee34e4401cb5154ba1212870eaf2b921d3f44536c251b0e'),
}
DEPENDENCIES = {
    'current-kinematic-trace-build-qualification-1': '89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033',
    'current-kinematic-trace-runtime-qualification-1': '730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4',
    'caller-bound-native-qualification-1': '51e1ba911d1bbe19c83141777f76fc967b0c876630fa3cef60f818adda205174',
}
CLOSURES = {
    'build': '8cab371e9affd5297c0f48c30184361959d20f3d3a48c6dbd6109b86bf49bacb',
    'ordinary': 'b364bd8535d0f5850cb5a21390c5678cbb1c1dd8fddfae6f0174cf50d261f7a7',
    'stress': 'c696421eb4948b4d2de5993cc84455b1ac04d0f659b245341729714f91ae8482',
}
REVIEWS = {
    'results/build-terminal-review.json': '9191a4513b79c5f55c0a34bde328c71516ad82b64ce0f81d098af2d41965b44d',
    'results/root-build-review.json': 'f517efdb36ab976bcebdc1612a45d4e1b5ad77c08d9835e276355be6e0c0e70c',
    'results/ordinary-terminal-review.json': 'a0ae797b55a2c9d620246fdafd74f4120ff9a132101d016be926724c1203289f',
    'results/root-ordinary-review.json': 'bc2b479c61cbec2b1f8a7df743dab3653dc5388f93870825bd5f1d3c4c91eed3',
    'results/stress-terminal-review.json': '84e365f3a5338acc3b35457a0d7a841e0e53743debc4a5710ae03bc9e948bfff',
    'results/stress-active-owned-sample.json': '22440ae66d2fc64e8a2318ce8e03629667a146c48dccc4141c48aa146484dce3',
    'results/root-stress-review.json': '725af61aa460a1f1f02f98032be989c2daa707664ef8c408043b40685cd4f579',
    'results/root-point20-comparison.json': '7b2390b98193dd11c66ae8eae64ec7b36f5497accd9e2e355de03ccfacda5d6d',
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

    for name, pin in index.get('origins', {}).items():
        content = data(name)
        require(len(content) == pin['bytes'] and sha(content) == pin['sha256'], 'Original copy identity changed')
    return index, data, lambda name: json.loads(data(name))
def identity(raw):
    return {'bytes': len(raw), 'sha256': sha(raw)}


def main():
    require(not any(value.startswith('PENDING') for value in [*CLOSURES.values(), *REVIEWS.values()]),
            'All admitted stages must be closed before verification')
    index, data, document = packet(HERE)
    require(index['dependencies'] == DEPENDENCIES and index['closed_stages'] == CLOSURES, 'Evidence scope differs')
    siblings = {name: packet(HERE.parent / name, checksum) for name, checksum in DEPENDENCIES.items()}
    _, build_data, build_document = siblings['current-kinematic-trace-build-qualification-1']
    _, runtime_data, runtime_document = siblings['current-kinematic-trace-runtime-qualification-1']
    _, factory_data, factory_document = siblings['caller-bound-native-qualification-1']
    origins = {}
    source_indices = {}
    for prefix, (original_prefix, checksum) in SOURCES.items():
        raw = data(prefix + '/index.json')
        require(sha(raw) == checksum, 'Source seal changed')
        source = json.loads(raw)
        source_indices[prefix] = source
        origins[prefix + '/index.json'] = original_prefix + 'index.json'
        for name, pin in source['files'].items():
            require(identity(data(prefix + '/' + name)) == pin, 'Source payload differs')
            origins[prefix + '/' + name] = original_prefix + name
    require(len(source_indices['proposal']['files']) == 55 and len(source_indices['predecessor/proposal']['files']) == 49,
            'Source inventory count differs')
    failed_name = 'predecessor/results/build-closure.json'
    require(sha(data(failed_name)) == '99cbb544826b741c030708ce4ab5bc01db83eac0ec49c4dce02a6be6858d76aa', 'Preserved first attempt differs')
    failed = document(failed_name)
    origins[failed_name] = OLD_WORK + 'results/build-closure.json'
    require(failed['passed'] is False and failed['failure'] == "KeyError('size')" and failed['owned_command_count'] == 0 and
            failed['loader_query_count'] == 0 and failed['builds'] == {} and failed['postcheck_failure'] is None and
            failed['source_index_sha256'] == SOURCES['predecessor/proposal'][1], 'First failure scope changed')
    for name, pin in failed['payloads'].items():
        require(identity(data('predecessor/results/' + name)) == pin, 'First failure receipt differs')
        origins['predecessor/results/' + name] = OLD_WORK + 'results/' + name
    terminal_name = 'predecessor/results/build-terminal-review.json'
    require(sha(data(terminal_name)) == 'b504c91987b6939bea4b2ebf2e4d9ab122f88ecfe1cf76c0a83628cdf1901564', 'First terminal review differs')
    old_terminal = document(terminal_name)
    origins[terminal_name] = OLD_WORK + 'results/build-terminal-review.json'
    require(old_terminal['terminal_exit_code'] == 1 and not old_terminal['native_children_created'] and
            old_terminal['shared_lock_nonblocking_acquisition_after_terminal'] and
            data('predecessor/results/build-before.json') == data('predecessor/results/build-after.json'),
            'First failure cleanup/state differs')
    closures, command_counts, payload_counts = {}, {}, {}
    for stage, checksum in CLOSURES.items():
        logical = 'results/' + stage + '-closure.json'
        require(sha(data(logical)) == checksum, 'Closed stage seal changed')
        closed = document(logical)
        closures[stage] = closed
        origins[logical] = WORK + logical
        require(closed['source_index_sha256'] == SOURCES['proposal'][1] and closed['postcheck_failure'] is None and
                not closed['host_stability_qualified'] and not closed['performance_measurement'], 'Closed stage authority/scope differs')
        if stage != 'build':
            require(closed['reviewed_build_sha256'] == CLOSURES['build'], 'Runtime build authority differs')
        if stage == 'stress':
            require(closed['reviewed_ordinary_sha256'] == CLOSURES['ordinary'], 'Stress ordinary authority differs')
        command_count = loader_count = 0
        failed_commands = []
        for name, pin in closed['payloads'].items():
            raw = data('results/' + name)
            require(identity(raw) == pin, 'Closed stage payload changed: ' + name)
            origins['results/' + name] = WORK + 'results/' + name
            if name.startswith('execution/') and name.endswith('.json'):
                receipt = json.loads(raw)
                command_count += 1
                require(receipt['cleanup_exception'] is None and receipt['cleanup']['survivors'] == [],
                        'Owned command cleanup incomplete')
                if receipt['returncode'] != 0 or receipt['exception'] is not None:
                    failed_commands.append(name)
                    require(stage == 'stress' and name == 'execution/stress-worker-cms-p21-mode1-lib1-original.json' and
                            receipt['returncode'] == 1, 'Unexpected owned command failure')
                log_name = str(Path(name).with_suffix('.txt'))
                require(log_name in closed['payloads'] and sha(data('results/' + log_name)) == receipt['log_sha256'], 'Command/log join differs')
                if '--list' in receipt['argv']:
                    loader_count += 1
        require(len(failed_commands) == (1 if stage == 'stress' else 0), 'Closed failure count changed')
        require(command_count == closed['owned_command_count'] and loader_count == closed['loader_query_count'], 'Command count differs')
        require(data('results/' + stage + '-before.json') == data('results/' + stage + '-after.json'), 'Protected stage state changed')
        command_counts[stage] = command_count
        payload_counts[stage] = len(closed['payloads'])
    require(command_counts == {'build': 129, 'ordinary': 16, 'stress': 18}, 'Declared command campaign differs')
    require(closures['build']['passed'] and closures['ordinary']['passed'], 'A required prior stage failed')
    require(closures['stress']['passed'] is False and closures['stress']['failure'] is not None,
            'Retained stress stage failure was erased')
    for name, checksum in REVIEWS.items():
        require(sha(data(name)) == checksum, 'Independent/terminal review changed')
        origins[name] = WORK + name
    for stage in CLOSURES:
        terminal = document('results/' + stage + '-terminal-review.json')
        require(terminal['terminal_exit_code'] == (1 if stage == 'stress' else 0) and
                terminal['remaining_owned_identities'] == [] and terminal['shared_lock_free_after_terminal'] and
                terminal['protected_equal'],
                'Terminal cleanup/accounting differs')
        require(terminal['source_index_sha256'] == SOURCES['proposal'][1] and
                terminal[stage + '_closure'] == identity(data('results/' + stage + '-closure.json')) and
                terminal['owned_command_count'] == command_counts[stage] and
                terminal['verified_closed_payloads'] == payload_counts[stage] and
                terminal['protected_before'] == identity(data('results/' + stage + '-before.json')) and
                terminal['protected_after'] == identity(data('results/' + stage + '-after.json')),
                'Terminal authority/closed-file joins differ')
        require(data('results/' + stage + '-before.json') == data('results/build-before.json'),
                'Protected state changed between stages')
    terminal = document('results/stress-terminal-review.json')
    require(terminal['live_owned_sample'] == identity(data('results/stress-active-owned-sample.json')) and
            terminal['observed_owned_identities'] == 27 and terminal['workers_launched'] == 9 and
            terminal['workers_orchestration_complete'] == 8 and terminal['actual_runtime_maps'] == 17 and
            terminal['outer_entries_begun'] == 18 and terminal['outer_entries_completed'] == 17 and
            terminal['all_owned_cleanup_clear'] and terminal['full_comparison_reached'] is False and
            terminal['retry_performed'] is False,
            'Incomplete stress cleanup/scope differs')
    sample = document('results/stress-active-owned-sample.json')
    failed_result = document('results/runs/cms-p21-mode1-lib1-original/result.json')
    require(sample['worker'] == 'cms-p21-mode1-lib1-original' and sample['signals_sent'] is False and
            sample['cause_inferred'] is False and len(sample['rows']) == 1 and
            sample['rows'][0]['pid'] == failed_result['native_pid'] and
            sample['rows'][0]['argv'][0] == sample['exact_executable'],
            'Supplemental historical ownership sample differs')
    # The source comparator's pinned references are text attribution only.
    for original, expected in {
        'tools/tirust_benchmark.py': {'bytes': 96661, 'sha256': 'bc405b44d8d794cc4396ef3da3d48af238ac4dfb619c58901d7bab910a7e1370'},
        'tests/parallel_tests/loop_me_comparator.py': {'bytes': 74750, 'sha256': 'b6da773dadbe1e202639a865cbe3f29f65af8147a250a8bb05e756d537548354'},
    }.items():
        logical = 'comparison-sources/' + original
        require(identity(data(logical)) == expected, 'Comparator attribution source differs')
        origins[logical] = original
    require({name: pin['path'] for name, pin in index['origins'].items()} == origins, 'Explicit origin selection differs')

    def helper(name):
        spec = importlib.util.spec_from_file_location('closed_amplitude_' + name, HERE / (name + '.py'))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    build_result = helper('build_checks').verify_build(data, document, build_data, build_document, factory_data, factory_document)
    worker_result = helper('worker_checks').verify_workers(data, document, build_data, build_document,
        {stage: closures[stage] for stage in ('ordinary', 'stress')})
    print(json.dumps({'payloads': len(index['files']), 'closed_payload_counts': payload_counts, 'closed_command_counts': command_counts,
        'build': build_result, 'workers': worker_result, 'predecessor_failure_preserved': True,
        'new_native_executions': 0, 'external_binary_reads': 0, 'performance_qualified': False,
        'checked_host_stability_qualified': False, 'production_candidate_promoted': False,
        'fresh_269_source_rederivation': False}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
