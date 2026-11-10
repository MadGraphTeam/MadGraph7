#!/usr/bin/env python3
"""Portable current-source lifecycle replay; no compiler, loader or reducer runs."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import struct
import types

ANALYZER_SHA = '0014bbf833cce64344ffcf38e2cb8c9a92605eeeb697fcf260553a13e5d1d40b'
SOURCE_INDEX_SHA = '95046dc54616a48266020d9b43d81f953c6374140017cd87eaf8bbfb011aedd6'
MANIFEST_SHA = 'ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0'
REVISION = '2c13246ffa63aa357ad1e29e5e79aff63d88dc7e'
STAGES = ('freeze', 'materialize', 'generate', 'prepare-build', 'build', 'smoke', 'run')


def verify(root=None, *, read_bytes=None, derive_summary=False):
    """Read indexed relative names only; recorded absolute paths remain strings.

    derive_summary is for the offline producer before publishing a final index.
    The report uses the strict default, which requires the exact saved summary.
    """
    def require(value, message):
        if not value:
            raise ValueError(message)
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()
    def safe(name):
        require(isinstance(name, str) and name and '\\' not in name, 'Invalid evidence name')
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and str(path) == name,
                'Nonportable evidence name')
        return name
    if read_bytes is None:
        root = Path(root or Path(__file__).resolve().parent).absolute()
        require(all(not path.is_symlink() for path in (root, *root.parents)), 'Symlink evidence root')
        def local_read(name):
            path = root
            for part in PurePosixPath(name).parts:
                path /= part
                require(not path.is_symlink(), 'Symlink evidence path')
            require(path.is_file(), 'Missing regular evidence file')
            return path.read_bytes()
        read_bytes = local_read
    raw_reader = read_bytes
    index_raw = raw_reader('evidence-index.json')
    index = json.loads(index_raw)
    require(index.get('schema_version') == 1 and index.get('files'), 'Invalid evidence index')
    packed = index.get('compressed_text', {})
    def physical(name):
        safe(name)
        item = index['files'].get(name)
        require(item is not None and item['kind'] == 'file', 'Unindexed regular file: ' + name)
        raw = raw_reader(name)
        require(len(raw) == item['size'] and sha(raw) == item['sha256'], 'Changed evidence: ' + name)
        return raw
    def read(name):
        safe(name)
        if name in packed:
            item = packed[name]
            raw = gzip.decompress(physical(item['path']))
            require(len(raw) == item['raw_bytes'] and sha(raw) == item['raw_sha256']
                    and b'\0' not in raw, 'Compressed text binding differs')
            raw.decode('utf-8')
            return raw
        return physical(name)
    def invalid(value):
        raise ValueError('Nonfinite JSON value: ' + value)
    def load(name):
        return json.loads(read(name), parse_constant=invalid)
    compressed_paths = [item['path'] for item in packed.values()]
    require(len(set(compressed_paths)) == len(compressed_paths), 'Aliased compressed payload')
    for name in index['files']:
        raw = physical(name)
        if name not in compressed_paths:
            require(b'\0' not in raw, 'Binary in portable text packet')
            raw.decode('utf-8')
    for name in packed:
        require(name not in index['files'], 'Compressed/raw path collision')
        read(name)
    class ReadPath:
        def __init__(self, name): self.name = safe(str(name))
        def __truediv__(self, name): return ReadPath(str(PurePosixPath(self.name) / str(name)))
        def __str__(self): return self.name
        def read_bytes(self): return read(self.name)
        def read_text(self, encoding='utf-8'): return self.read_bytes().decode(encoding)
        def stat(self): return types.SimpleNamespace(st_size=len(self.read_bytes()))

    source_index = read('source-proposal/source-index.json')
    require(sha(source_index) == SOURCE_INDEX_SHA, 'Different executed source protocol')
    for name, item in json.loads(source_index)['files'].items():
        raw = read('source-proposal/' + safe(name))
        require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Protocol source changed')
    manifest_raw = read('source-proposal/campaign-manifest.json')
    require(sha(manifest_raw) == MANIFEST_SHA, 'Different current campaign')
    manifest = json.loads(manifest_raw)
    source = load('source-freeze.json')
    materialized = load('materialized-source.json')
    generated = load('generated-source-freeze.json')
    build = load('build.json')
    retention = load('retention-binding.json')
    require(retention['schema'] == 1 and retention['no_native_execution'] is True
            and retention['failed_predecessor_index_sha256'] ==
                '22deeb09e727fb4f687e3c78c2bc215afd86cca81a933b74cf259597453c296b'
            and retention['capture_sha256'] == sha(read('retention-capture.py')),
            'Closed retention provenance differs')
    for name, checksum in retention['closed_original_files'].items():
        require(sha(read(safe(name))) == checksum, 'Original closed evidence changed: ' + name)
    retained_generated = load('retained-generated-index.json')
    expected_generated = {name: item for name, item in generated['source_inventory'].items()
                          if 'sha256' in item}
    require(set(retained_generated) == set(expected_generated), 'Generated retention coverage differs')
    for name, item in expected_generated.items():
        retained = retained_generated[name]
        require(retained['sha256'] == item['sha256'], 'Retained generated inventory differs')
        if retained['retained'] is True:
            raw = read('retained-generated/' + safe(name))
            require(sha(raw) == item['sha256'] and len(raw) == item['size'],
                    'Retained generated source differs: ' + name)
            require(b'\0' not in raw, 'Binary in retained generated text')
            raw.decode('utf-8')
        else:
            require(retained['retained'] is False
                    and retained['reason'] == 'Binary represented by source inventory hash only'
                    and PurePosixPath(name).suffix not in ('.f', '.f90', '.inc', '.dat', '.py', '.h', '.c'),
                    'Required generated text was not retained')
    require(source['source_index_sha256'] == SOURCE_INDEX_SHA
            and source['campaign_manifest_sha256'] == MANIFEST_SHA
            and source['implementation_revision'] == materialized['revision'] == REVISION,
            'Source/campaign identity differs')
    production = dict(manifest['candidate']['exporter_identity']['files'])
    for name, item in manifest['candidate']['numerical_sources'].items():
        require(name not in production or production[name] == item['sha256'], 'Conflicting registered source')
        production[name] = item['sha256']
    require(source['registered_sources'] == production, 'Registered source closure differs')
    for name, checksum in production.items():
        item = materialized['files'][name]
        require(item.get('sha256') == checksum and
                (checksum is not None or item['kind'] == 'link'), 'Materialized source differs: ' + name)
    for name, checksum in source['protocol'].items():
        require(sha(read(name)) == checksum == sha(read('source-proposal/' + name)), 'Execution protocol changed')
    stages = {name: load('stage-' + name + '.json') for name in STAGES}
    for name, item in stages.items():
        cap = 30 if name == 'smoke' else 750 if name == 'run' else 600
        require(item['stage'] == name and item['status'] == 'passed' and item['exception'] is None
                and item['source_index_sha256'] == SOURCE_INDEX_SHA
                and item['campaign_manifest_sha256'] == MANIFEST_SHA
                and item['cap_seconds'] == cap and 0 <= item['elapsed_seconds'] <= cap
                and item['native_execution'] == (name in ('smoke', 'run'))
                and item['performance_eligible'] is False, 'Stage did not close successfully: ' + name)
    require(build['source_freeze_sha256'] == sha(read('source-freeze.json'))
            and build['generated_source_freeze_sha256'] == sha(read('generated-source-freeze.json'))
            and build['linker_prebinding_sha256'] == sha(read('linker-prebinding.json'))
            and build['campaign_manifest_sha256'] == MANIFEST_SHA
            and build['implementation_revision'] == REVISION
            and build['fresh_numerical_source_build'] is True, 'Build/source identity differs')
    closure = load('linker-prebinding.json')
    require(closure['completed_before_any_lifecycle_link'] is True, 'No pre-link closure')
    bound = build['all_link_inputs_before_after']
    for name, checksum in {**closure['closure']['files'], **build['numerical_objects']}.items():
        require(bound.get(name) == checksum, 'Prebound build input differs')
    for name, item in closure['wrapper']['inputs'].items():
        require(bound.get(name) == item['sha256'] == sha(item['text'].encode()), 'Wrapper binding differs')
    for name, checksum in generated['rendered'].items():
        require(sha(read('retained-rendered/' + name)) == checksum, 'Rendered driver differs')
    require(generated['installation'] == source['installation'], 'Generated installation differs')
    runtimes = [item['runtime']['files'] for item in build['executables'].values()]
    require(len(runtimes) == 3 and all(value == runtimes[0] for value in runtimes), 'Runtime closures differ')
    map_resolutions = load('link-map-resolution.json')
    for variant, binary in build['executables'].items():
        runtime = binary['runtime']
        require(runtime['files'].get(runtime['loader']) == runtime['loader_sha256'], 'PT_INTERP loader binding differs')
        for name, checksum in binary['link_loads'].items():
            require(bound.get(name) == checksum, 'Actual linked input was not prebound')
        map_name = 'execution/lifecycle_' + ('card_smoke' if variant == 'card-smoke' else variant) + '.map'
        require(sha(read(map_name)) == binary['map_sha256'], 'Actual link map differs')
        spellings = [line[len('LOAD '):] for line in read(map_name).decode().splitlines() if line.startswith('LOAD ')]
        resolution = map_resolutions['executables'][variant]
        require(resolution['map_sha256'] == binary['map_sha256']
                and [item['spelling'] for item in resolution['loads']] == spellings
                and {item['resolved']: item['sha256'] for item in resolution['loads']} == binary['link_loads'],
                'Actual LOAD spellings/resolution differ')
    installed = source['installation']
    prefixes = [name[:-len('lib/libmadloop_tirust.so')] for name in installed if name.endswith('/lib/libmadloop_tirust.so')]
    require(len(prefixes) == 1, 'Nonunique installation prefix')
    for name, checksum in manifest['candidate']['native']['files'].items():
        if name not in ('Cargo.lock', 'Cargo.toml'):
            require(installed.get(prefixes[0] + name, {}).get('sha256') == checksum,
                    'Retained native artifact differs from campaign')
    for name, item in installed.items():
        if name.endswith('/libmadloop_tirust.so'):
            require(runtimes[0].get(name) == item['sha256'], 'Runtime does not bind installed DSO')

    analyzer_raw = read('analyze.py')
    require(sha(analyzer_raw) == ANALYZER_SHA, 'Different numerical analyzer')
    namespace = {'__name__': 'portable_current_lifecycle_analysis', '__file__': 'analyze.py'}
    exec(compile(analyzer_raw, 'analyze.py', 'exec'), namespace)
    namespace['Path'] = ReadPath
    analysis = types.SimpleNamespace(**namespace)
    replay = json.loads(json.dumps(analysis.analyze('runs'), allow_nan=False))
    require(replay == load('results.json') and replay['passed'], 'Raw lifecycle analysis failed')
    smoke_source = read('source-proposal/smoke.py')
    functions = [node for node in ast.parse(smoke_source).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'check_smoke']
    require(len(functions) == 1, 'Missing smoke checker')
    smoke_namespace = {'analyzer': lambda: analysis}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'smoke.py', 'exec'), smoke_namespace)
    smoke = smoke_namespace['check_smoke'](ReadPath('runs/card-smoke/records.txt'), ReadPath('runs/card-smoke/trace.txt'))
    smoke_result = load('card-smoke-result.json')
    require(smoke == smoke_result['analysis'] and smoke['passed']
            and smoke_result['build_sha256'] == sha(read('build.json'))
            and smoke_result['execution_sha256'] == sha(read('execution/native-card-smoke.json')), 'Smoke prerequisite differs')
    for name, item in smoke_result['evidence'].items():
        require(sha(read('runs/card-smoke/' + safe(name))) == item['sha256'], 'Smoke evidence changed')

    labels = [('card-smoke', 'card-smoke', 0), ('plain-cache0', 'plain', 0),
              ('plain-cache-all', 'plain', -1), ('observed-cache0', 'observed', 0),
              ('observed-cache-all', 'observed', -1)]
    output_names = {'records.txt', 'trace.txt', 'param.log', 'runtime-maps-open.txt', 'runtime-maps-close.txt'}
    data_mappings = load('non-elf-map-classification.json')
    require(data_mappings['scope'] == 'Post-run classification of read-only immutable Nix locale data; no additional ELF mapping admitted.',
            'Unknown supplemental map classification scope')
    for label, variant, cache in labels:
        prefix = 'runs/' + label + '/'
        inputs, outputs = load(prefix + 'inputs.json'), load(prefix + 'outputs.json')
        execution = load('execution/native-' + label + '.json')
        binary = build['executables'][variant]
        require(inputs['build_sha256'] == sha(read('build.json')) and inputs['executable'] == binary
                and (inputs['label'], inputs['variant'], inputs['cache']) == (label, variant, cache), 'Run input role/build differs')
        require(execution['argv'] == ['taskset', '-c', '12', binary['path'], str(cache)]
                and execution['cwd'].endswith('/subprocess-cache-lifecycle-current-2/runs/' + label)
                and execution['returncode'] == 0 and execution['exception'] is None
                and execution['cleanup_exception'] is None and execution['cleanup']['survivors'] == [],
                'Unsuccessful or different native invocation')
        cap = 30 if label == 'card-smoke' else 180
        require(0 < execution['command_work_cap_seconds'] <= cap - 10
                and 0 <= execution['elapsed_seconds'] <= cap, 'Native command exceeded bound')
        console = read('execution/native-' + label + '.txt')
        require(sha(console) == execution['log_sha256'] and b'##W03' not in console, 'Changed/incomplete native console')
        require(outputs['inputs_unchanged'] and outputs['execution_exception'] is None
                and outputs['input_files_after'] == inputs['input_files']
                and set(outputs['declared_outputs']) == output_names, 'Runtime input/output mutation')
        for name, item in outputs['declared_outputs'].items():
            raw = read(prefix + name)
            require(item['kind'] == 'file' and sha(raw) == item['sha256'] and len(raw) == item['size'], 'Changed output bytes')
        for name, item in inputs['input_files'].items():
            if name == 'resources/MadLoopParams.dat':
                raw = read(prefix + 'MadLoopParams.dat')
            elif name.startswith('resources/'):
                expected = generated['source_inventory']['SubProcesses/MadLoop5_resources/' + name[len('resources/'):]]
                require(item['sha256'] == expected['sha256'] and item['size'] == expected['size'], 'Changed generated resource')
                continue
            else:
                raw = read(prefix + safe(name))
            require(item['kind'] == 'file' and sha(raw) == item['sha256'] and len(raw) == item['size'], 'Changed runtime input')
        expected_maps = set(binary['runtime']['files']) | {binary['path']}
        for name in ('runtime-maps-open.txt', 'runtime-maps-close.txt'):
            paths = set()
            for line in read(prefix + name).decode().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith('/'):
                    require(not fields[5].endswith(' (deleted)'), 'Deleted runtime mapping')
                    path = fields[5]
                    if path in expected_maps:
                        paths.add(path)
                    else:
                        item = data_mappings['files'].get(path)
                        require(item is not None and path.startswith('/nix/store/') and path.endswith('/lib/locale/locale-archive')
                                and fields[1] == 'r--p' and fields[3] == item['map_device']
                                and int(fields[4]) == item['inode'] and item['size'] > 0
                                and item['header64_hex'][:8] != '7f454c46'
                                and len(item['sha256']) == 64, 'Unknown/non-data runtime mapping')
            require(paths == expected_maps, 'Actual runtime mappings differ from bound ELF closure')

    workers, traces, parsed = {}, {}, {}
    for label in analysis.LABELS:
        data = analysis.parse_records(ReadPath('runs/' + label + '/records.txt'))
        trace = analysis.parse_trace(ReadPath('runs/' + label + '/trace.txt'))
        parsed[label], traces[label] = data, trace
        maximum = max(struct.unpack('>d', bytes.fromhex(value[-1]))[0]
                      for frame in data['amplitudes'].values() for value in frame['values'])
        require(all(frame['status'] == 219 for frame in data['amplitudes'].values()), 'Scope requires native-DP status219')
        workers[label] = {'warmup_calls': data['warm'], 'measured_physical_calls': 19,
            'all_physical_calls_including_warmup': len(data['amplitudes']), 'tensor_key_calls': len(data['keys']),
            'status_counts_including_warmup': dict(Counter(str(frame['status']) for frame in data['amplitudes'].values())),
            'maximum_reported_accuracy': maximum, 'trace_rows': len(trace['records']), 'trace_flushes': len(trace['flushes']),
            'native_event_counts': dict(Counter(str(row['kind']) for row in trace['records'] if row['kind'] != 1))}
    for label in ('observed-cache0', 'observed-cache-all'):
        evaluations = [row for action in analysis.PHYSICAL for row in traces[label]['groups'][action] if row['kind'] in (6, 7)]
        require(len(evaluations) == 38 and all(row['kind'] == 6 for row in evaluations), 'Physical precision differs')
    review = load('independent-review.json')
    require(review['status'] == 'PASS' and review['results_sha256'] == sha(read('results.json'))
            and review['analyzer_sha256'] == ANALYZER_SHA and review['build_sha256'] == sha(read('build.json')),
            'Independent review missing or bound to other results')
    locale = review['non_elf_mapping']
    require(set(data_mappings['files']) == {locale['path']}, 'Supplemental non-ELF mapping scope differs from independent review')
    for key in ('sha256', 'size', 'inode', 'map_device'):
        require(data_mappings['files'][locale['path']][key] == locale[key], 'Non-ELF classification differs from independent review')
    references = ['source-freeze.json', 'materialized-source.json', 'generated-source-freeze.json',
                  'generation.json', 'build.json', 'linker-prebinding.json', 'card-smoke-result.json',
                  'independent-review.json', 'link-map-resolution.json', 'non-elf-map-classification.json',
                  'retention-binding.json', 'retained-generated-index.json'] + ['stage-' + name + '.json' for name in STAGES]
    summary = {'schema_version': 1, 'kind': 'tirust-generated-subprocess-cache-lifecycle', 'passed': True,
        'implementation_revision': REVISION, 'campaign': 'invariant-symmetry-1', 'campaign_manifest_sha256': MANIFEST_SHA,
        'source_index_sha256': SOURCE_INDEX_SHA, 'results_sha256': sha(read('results.json')),
        'source_and_build_evidence': {name: sha(read(name)) for name in references},
        'independent_review_path': 'independent-review.json', 'independent_review_status': 'PASS',
        'installed_artifacts': {PurePosixPath(name).name: {'recorded_path': name, **item} for name, item in installed.items()},
        'runtime_files': runtimes[0], 'workers': workers,
        'coverage': {'physical_workers': 4, 'warmup_calls_total': sum(sum(data['warm']) for data in parsed.values()),
                     'measured_physical_calls_total': 76, 'tensor_key_calls_total': 96,
                     'tensor_keys_per_worker_per_precision': 12, 'generated_rank_max': 2, 'generated_propagator_max': 3},
        'processes': [{'id': 101, 'process': 'u u~ > z [virt=QCD]', 'prefix': 'ML5_101_'},
                      {'id': 202, 'process': 'd d~ > z [virt=QCD]', 'prefix': 'ML5_202_'}],
        'model': 'loop_sm', 'backend': 9, 'pole_mode': 0, 'compute_finite': True,
        'parity': {'bitwise_coefficients_status_accuracy_inputs_ieee': True, 'plain_observed_parity': True,
                   'cache_enabled_disabled_parity': True, 'physical_status': 219,
                   'physical_precision': 'native double precision; no QP rescue occurred',
                   'qp_scope': 'Separate genuine-QP generated tensor diagnostics; DP inputs promoted exactly.'},
        'card_smoke': {'passed': True, 'actual_reader_calls': 3, 'mass_sequence_GeV': [91.188, 92, 91.188],
                       'context_generations': [-1, -1, -1], 'native_exports': 0},
        'observed_lifecycle': {'context_generations_per_worker': 2, 'prefix_registrations_per_worker': 4,
            'registered_topology_rank': [3, 2], 'automatic_clears_per_measured_physical_call': 2,
            'evaluations_per_measured_physical_call': 2, 'primary_and_stability_check_stores_distinguished': True,
            'seeded_entry_before_automatic_clear': {'entries': 1, 'bytes': 144},
            'seeded_entry_after_automatic_clear': {'entries': 0, 'bytes': 0},
            'cache_enabled_nonempty_final_clears': 19, 'cache_disabled_nonempty_final_clears': 0,
            'recreated_context_precedes_automatic_entry_clear': True, 'key_cold_requests_per_precision': 5,
            'key_repeats_per_precision': 7, 'disabled_qp_bypasses_per_worker': 12, 'all_native_evaluation_statuses_zero': True},
        'limitations': ['Untimed two-subprocess real-SM lifecycle qualification; no performance acceptance.',
            'N<=3/rank<=2; no physical-QP rescue, rank-six, EW, loop-induced or poles-only qualification.',
            'QP key probes use DP values promoted exactly; no binary128-only input perturbation.',
            'Portable replay reads retained text only; actual runtime validation is bound to the executed controller.',
            'The failed predecessor remains separate and does not provide a successful build or numerical result.'],
        'runtime_maps_captured': True, 'performance_eligible': False}
    if not derive_summary:
        require(load('qualification-summary.json') == summary, 'Saved qualification summary differs from raw derivation')
    return {'passed': True, 'index_sha256': sha(index_raw), 'results_sha256': summary['results_sha256'],
            'physical_workers': 4, 'measured_physical_calls': 76, 'tensor_key_calls': 96,
            'card_smoke_passed': True, 'historical_symlinks_are_metadata_only': True,
            'summary': summary, 'scope': 'Portable current-source retained-text/runtime-map replay; no live binary requalification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?', type=Path, default=Path(__file__).resolve().parent)
    result = verify(parser.parse_args().root)
    print(json.dumps({name: value for name, value in result.items() if name != 'summary'}, indent=2, sort_keys=True))
