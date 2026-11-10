#!/usr/bin/env python3
"""Portable preparation/preflight proof; no compiler, loader or worker execution."""
import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

SOURCE_INDEX_SHA = 'c192146977008a7dec8c3d1dce479eeaaf53e382ab4aad5599db6665dfe42f25'
PROPOSAL_SHA = '5478869fa719db000df7df825202f3ae827080d1f00b43f38bad1506f5ab54ac'
MANIFEST_SHA = 'ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0'
GENERATOR_SHA = 'a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19'
REVISION = '2c13246ffa63aa357ad1e29e5e79aff63d88dc7e'
BASELINE = '9228fc58e8f8e48dadc70a6fcfb6617845aaeb1f'
DSO = 'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'

def verify(root=None, *, read_bytes=None, derive_summary=False):
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

    names = set(index['files']) | set(packed)
    for prefix, checksum in (('source', SOURCE_INDEX_SHA), ('proposal', PROPOSAL_SHA)):
        raw = read(prefix + '/source-index.json')
        require(sha(raw) == checksum, 'Different executed source/proposal')
        for name, item in json.loads(raw)['files'].items():
            raw = read(prefix + '/' + safe(name))
            require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Source payload differs')
    plan = load('proposal/proposal.json')
    require(plan['candidate_revision'] == REVISION and plan['baseline_revision'] == BASELINE
            and plan['candidate_generator_sha256'] == GENERATOR_SHA, 'Different scientific source identity')
    bindings = load('proposal/input-bindings.json')
    for name, checksum in bindings.items():
        require(sha(read('inputs/' + safe(name))) == checksum, 'Frozen input binding differs')
    campaign_name = 'inputs/docs/benchmarkfolder/campaigns/invariant-symmetry-1/'
    manifest_raw = read(campaign_name + 'manifest.json')
    require(sha(manifest_raw) == MANIFEST_SHA, 'Different campaign')
    manifest = json.loads(manifest_raw)
    frozen = load('closed/source-freeze.json')
    require(frozen['source_index_sha256'] == SOURCE_INDEX_SHA
            and frozen['proposal_index_sha256'] == PROPOSAL_SHA
            and frozen['campaign_manifest_sha256'] == MANIFEST_SHA
            and frozen['native_execution'] is False, 'Freeze differs')
    retention = load('retention.json')
    require(retention['scope'] == 'preparation-and-preflight-only' and retention['amplitude_evaluations'] == 0
            and retention['capture_sha256'] == sha(read('capture.py')), 'Retention scope differs')
    for name, item in retention['closed_files'].items():
        raw = read('closed/' + safe(name))
        require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Closed original differs')
    require(not any(name.startswith('closed/runs/') or name in ('closed/stage-run.json', 'closed/summary.json') for name in names),
            'Numerical execution included in preflight capsule')
    stages = {name: load('closed/stage-' + name + '.json') for name in ('prepare', 'preflight')}
    for name, stage in stages.items():
        cap = 3600 if name == 'prepare' else 600
        require(stage['stage'] == name and stage['status'] == 'passed' and stage['exception'] is None
                and stage['source_index_sha256'] == SOURCE_INDEX_SHA
                and stage['campaign_manifest_sha256'] == MANIFEST_SHA
                and stage['performance_eligible'] is False
                and stage['native_execution'] == (name == 'preflight')
                and stage['cap_seconds_excluding_lock_wait'] == cap
                and 0 <= stage['elapsed_seconds_including_wait'] <= cap, 'Stage admission failed')
    materialized = {source: load('closed/materialized-' + source + '.json') for source in ('baseline', 'candidate')}
    for source, revision in (('baseline', BASELINE), ('candidate', REVISION)):
        require(materialized[source]['revision'] == revision, 'Different materialized revision')
        prerequisites = load('source/prerequisites.json')
        require(materialized[source]['prebuilt_copies'] == prerequisites['sources'][source]['copies'], 'Private dependency copy binding differs')
        raw = read('entrypoints/' + source + '-madgraph.py')
        require(sha(raw) == materialized[source]['files']['bin/madgraph']['sha256'], 'Actual tracked CLI differs')
    candidate_files = dict(manifest['candidate']['exporter_identity']['files'])
    candidate_files.update({name: item['sha256'] for name, item in manifest['candidate']['numerical_sources'].items()})
    baseline_files = load(campaign_name + 'inputs/baseline.json')['files']
    for source, expected in (('baseline', baseline_files), ('candidate', candidate_files)):
        for name, checksum in expected.items():
            item = materialized[source]['files'][name]
            require(item.get('sha256') == checksum and (checksum is not None or item['kind'] == 'link'), 'Materialized source closure differs')
    # Reuse only the two pure compiler-policy functions, not its command runner.
    module = ast.parse(read('proposal/protocol/tirust_benchmark_compiler.py'))
    chosen = [node for node in module.body if isinstance(node, ast.FunctionDef)
              and node.name in ('_signature', 'validate_compiler_equality')]
    schema = next(node.value.value for node in module.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'SCHEMA' for target in node.targets))
    compiler = {'SCHEMA': schema}
    exec(compile(ast.Module(body=chosen, type_ignores=[]), '<frozen pure compiler policy>', 'exec'), compiler)
    generated = load('retained-generated.json')
    maps = load('link-map-resolution.json')
    expected_labels = {source + '-' + case['case']['id'] for case in plan['cases'] for source in ('baseline', 'candidate')}
    require(set(generated) == set(maps) == expected_labels and len(expected_labels) == 8, 'Build coverage differs')
    commands = {}
    for name in sorted(names):
        if not name.startswith('closed/execution/') or not name.endswith('.json'):
            continue
        command = load(name)
        require(command['returncode'] == 0 and command['exception'] is None
                and command['cleanup_exception'] is None and not command['cleanup']['survivors'], 'Owned command did not close cleanly')
        require(sha(read(name[:-5] + '.txt')) == command['log_sha256'], 'Command log differs')
        require(command['cpu_exclusive'] is False and command['cpu'] in (12, '12-15'), 'CPU policy differs')
        commands[PurePosixPath(name).stem] = command
    mass_commands = [cmd for cmd in commands.values() if './check' in cmd['argv']]
    require(len(mass_commands) == 8 and all(cmd['argv'] == ['taskset', '-c', '12', './check', '--masses'] for cmd in mass_commands),
            'Unexpected amplitude/executable invocation')
    require(not any('--masses' in cmd['argv'] for name, cmd in commands.items() if not name.startswith('masses-')),
            'Mass command mislabeled')
    candidate_dso = set()
    for case in plan['cases']:
        cid = case['case']['id']
        point = load('proposal/points/' + cid + '/selected.json')['inputs']
        for source in ('baseline', 'candidate'):
            label = source + '-' + cid
            prefix = 'closed/builds/' + label + '/'
            generation = load(prefix + 'generation.json'); build = load(prefix + 'build.json')
            prelink = load(prefix + 'prelink.json'); preflight = load(prefix + 'preflight.json')
            require(generation['source'] == source and generation['case'] == case['case']
                    and generation['revision'] == (REVISION if source == 'candidate' else BASELINE), 'Generation identity differs')
            expected_dp = [True, False, True, False, False, True, True, True] + ([True] if source == 'candidate' else [])
            require(generation['availability']['available'] == expected_dp
                    and generation['availability']['qp_available'] == {str(i): i in (1, 6, 8, 9) for i in range(len(expected_dp) + 1)},
                    'Generated backend/precision availability differs')
            require(build['fresh_full_objects'] is True and build['loader_executed'] is False and build['masses_executed'] is False
                    and build['generation_sha256'] == sha(read(prefix + 'generation.json'))
                    and build['prelink_sha256'] == sha(read(prefix + 'prelink.json'))
                    and preflight['build_sha256'] == sha(read(prefix + 'build.json'))
                    and prelink['completed_before_link'] is True, 'Build/preflight closure differs')
            expected_flags = load(campaign_name + 'inputs/' + source + '/' + cid + '/compilation.json')['effective_compiler_flags']
            flags = load(prefix + 'compiler-flags.json')
            compiler['validate_compiler_equality'](expected_flags, flags)
            compiler['validate_compiler_equality'](flags, build['compiler_flags'])
            after_name = 'closed/builds/after-exports-' + source + '-' + cid + '-' + case['process_subdirectory'].replace('/', '-') + '/compiler-flags.json'
            compiler['validate_compiler_equality'](flags, load(after_name))
            require(build['scientific_sources'] == generation['scientific_sources'], 'Numerical source receipt differs')
            for name, checksum in generation['scientific_sources'].items():
                record = generated[label][name]
                raw = read(record['retained'])
                require(sha(raw) == checksum == record['sha256'], 'Generated source differs')
            for name in ('Cards/param_card.dat', 'Cards/MadLoopParams.dat'):
                record = generated[label][name]
                require(sha(read(record['retained'])) == generation['all_generation_files'][name]['sha256'], 'Original exported card differs')
            require(build['physical_card_sha256'] == generation['physical_card_sha256'] == case['physical_card_sha256'], 'Physical input differs')
            require(preflight['masses'] == point['incoming_masses'] + point['masses'], 'Actual masses disagree with frozen point')
            masses = [float(x) for x in read('masses/' + label + '.dat').decode().split()]
            require(masses == preflight['masses'], 'Mass output differs')
            admission = generation['cli_admission']
            require(admission == load(prefix + 'cli-admission.json') and admission['entrypoint'] == 'bin/madgraph'
                    and admission['sha256'] == materialized[source]['files']['bin/madgraph']['sha256']
                    and admission['materialized_receipt_sha256'] == sha(read('closed/materialized-' + source + '.json'))
                    and admission['argv'][1:3] == ['-B', './bin/madgraph'], 'CLI admission differs')
            require(commands['generate-' + label]['argv'] == ['taskset', '-c', '12-15'] + admission['argv'], 'Executed CLI differs')
            require(generation['command_sha256'] == sha(read('source/commands/' + label + '.mg5')), 'Executed CLI card differs')
            bound = build['bound_inputs_before_after']
            for name, checksum in {**prelink['bound'], **build['objects'], **prelink['closure']['files']}.items():
                require(bound.get(name) == checksum, 'Prebound numerical/link input differs')
            for name, checksum in build['actual_link_loads'].items():
                require(bound.get(name) == checksum, 'Actual LOAD was not prebound')
            resolution = maps[label]; map_raw = read(prefix + 'stress.map')
            spellings = [line[5:] for line in map_raw.decode().splitlines() if line.startswith('LOAD ')]
            require(sha(map_raw) == resolution['map_sha256'] and spellings == [v['spelling'] for v in resolution['loads']]
                    and {v['resolved']: v['sha256'] for v in resolution['loads']} == build['actual_link_loads'], 'Link-map resolution differs')
            runtime = preflight['runtime']
            require(runtime['files'].get(runtime['loader']) == runtime['loader_sha256'], 'Actual PT_INTERP identity differs')
            matches = [cmd for cmd in commands.values() if cmd['argv'][3:] == [runtime['loader'], '--list', build['executable']]]
            require(len(matches) == 1, 'Actual loader command absent or duplicated')
            if source == 'candidate':
                observed = [checksum for name, checksum in runtime['files'].items() if name.endswith('/libmadloop_tirust.so')]
                require(observed == [DSO], 'Candidate runtime DSO differs'); candidate_dso.update(observed)
    installation = frozen['installation']
    prefixes = [name[:-len('lib/libmadloop_tirust.so')] for name in installation if name.endswith('/lib/libmadloop_tirust.so')]
    require(len(prefixes) == 1, 'Nonunique installed bridge')
    for name, checksum in plan['candidate_native_files'].items():
        require(installation[prefixes[0] + name]['sha256'] == checksum, 'Installed campaign artifact differs')
    summary = {'scope': 'current-source preparation and preflight only', 'passed': True,
               'role': 'current-physical-stress-preparation', 'preparation_qualified': True,
               'closed_stages': ['prepare', 'preflight'], 'native_state': 'pending',
               'candidate_revision': REVISION, 'candidate_generator_sha256': GENERATOR_SHA,
               'campaign_manifest_sha256': MANIFEST_SHA, 'source_index_sha256': SOURCE_INDEX_SHA,
               'baseline_revision': BASELINE, 'fresh_CLI_exports': 8, 'fresh_full_builds': 8,
               'mass_preflights': 8, 'candidate_DSO_sha256': DSO, 'owned_commands': len(commands),
               'closed_original_files': len(retention['closed_files']), 'generated_text_files': sum(map(len, generated.values())),
               'preparation_seconds': stages['prepare']['elapsed_seconds_including_wait'],
               'preflight_seconds': stages['preflight']['elapsed_seconds_including_wait'],
               'amplitude_evaluations': 0, 'physical_stress_qualified': False, 'pending_native_requests': 24,
               'performance_eligible': False, 'EW_analytic_oracle_qualified': False,
               'request_rows': [{'case': request['case'], 'label': request['label'], 'source': request['source'],
                                 'backend': request['backend'], 'state': 'not_started', 'numerically_qualified': False,
                                 'observed_statuses': None, 'incomplete_warnings': None, 'failure': None}
                                for request in plan['requests']]}
    if not derive_summary:
        require(summary == load('qualification-summary.json'), 'Saved summary differs')
    return {'verified': True, 'passed': True, 'index_sha256': sha(index_raw), 'summary': summary}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', nargs='?')
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2, sort_keys=True))
