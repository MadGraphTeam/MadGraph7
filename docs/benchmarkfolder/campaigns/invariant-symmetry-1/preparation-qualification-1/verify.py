#!/usr/bin/env python3
"""Offline verifier: only reads the portable packet, never the recorded paths."""
import gzip
import hashlib
import json
from datetime import datetime
from pathlib import Path, PurePosixPath

CAMPAIGN = 'invariant-symmetry-1'
BASE = 'docs/benchmarkfolder/campaigns/' + CAMPAIGN + '/'
PREP = BASE + 'preparation-1/'
MANIFEST = 'ce49d147c1117875cd7e9692223bec546c100b3a4f0c3f89eef9206c323223d0'
SOURCE_INDEX = '0757f9f829fbc4d472823cee226bb54a0a62efcbf70fbab59c137beeb030aa87'
GENERATOR = 'a90f80545696a648dbcfdfb8de73b1e319364edf2b93d02d4458c7b9656b8b19'
STAGES = ['reuse-baseline', 'build-baseline', 'generate-candidate',
          'assert-declared-cards', 'build-candidate']


def sha(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def duration(record):
    return (datetime.fromisoformat(record['finished_utc']) -
            datetime.fromisoformat(record['started_utc'])).total_seconds()


def compiler_equal(left, right):
    """Same pure normalization as the retained compiler validation helper."""
    require(left['schema'] == right['schema'] == 'madloop-effective-compiler-flags-v1', 'compiler schema')
    require(left['environment'] == right['environment'], 'compiler environment')
    require(set(left['scopes']) == set(right['scopes']) != set(), 'compiler scopes')
    def signature(recipe):
        return (tuple((c['sha256'], c['version']) for c in recipe['compiler']), tuple(recipe['flags']))
    for scope in left['scopes']:
        a, b = left['scopes'][scope]['recipes'], right['scopes'][scope]['recipes']
        require(a and b, 'empty compiler recipes')
        require({signature(v) for v in a.values()} == {signature(v) for v in b.values()}, 'compiler policy: ' + scope)
        for name in a.keys() & b.keys():
            require(signature(a[name]) == signature(b[name]), 'compiler source: ' + scope + '/' + name)


def verify(folder):
    folder = Path(folder)
    index = json.loads((folder / 'index.json').read_text())
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(actual == set(index['files']) | {'index.json'}, 'unindexed or missing packet files')
    for name, record in index['files'].items():
        p = folder / name
        require(not p.is_symlink() and sha(p.read_bytes()) == record['sha256'] and p.stat().st_size == record['bytes'], 'packet bytes: ' + name)
    packed = (folder / 'evidence.json.gz').read_bytes()
    bundle = json.loads(gzip.decompress(packed))
    require(bundle['schema'] == 'tirust-deduplicated-utf8-evidence-v1', 'bundle schema')
    files, objects = bundle['files'], bundle['objects']
    inventory = json.loads((folder / 'inventory.json').read_text())
    require(files == inventory['files'], 'inventory mapping')
    require(len(files) == 316 and len(objects) == inventory['unique_text_contents'] == 241, 'bundle completeness')
    require(set(objects) == {r['sha256'] for r in files.values()}, 'orphan or missing text content')
    for digest, value in objects.items():
        require(isinstance(value, str) and '\0' not in value and sha(value.encode('utf-8')) == digest, 'text content hash')
    for name, record in files.items():
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts, 'nonrelative logical path')
        require(len(objects[record['sha256']].encode('utf-8')) == record['bytes'], 'content length: ' + name)
    def text(name):
        return objects[files[name]['sha256']]
    def doc(name):
        return json.loads(text(name))
    def digest(name):
        return files[name]['sha256']
    require(digest(BASE + 'manifest.json') == MANIFEST, 'canonical manifest')
    require(digest(PREP + 'source-index.json') == SOURCE_INDEX, 'original preparation source seal')
    source_index = doc(PREP + 'source-index.json')
    for name, expected in source_index['files'].items():
        require(digest(PREP + name) == expected, 'original preparation source: ' + name)
    for name, expected in source_index['inputs'].items():
        require(digest(name) == expected, 'original preparation input: ' + name)
    require(sum(name.startswith(PREP) for name in files) == 41, 'closed preparation file count')
    manifest = doc(BASE + 'manifest.json')
    config = doc(BASE + 'cases.json')
    for name, expected in manifest['shared_inputs']['files'].items():
        require(digest(BASE + name) == expected, 'frozen shared input: ' + name)
    for name, record in manifest['candidate']['numerical_sources'].items():
        require(digest(BASE + manifest['candidate']['source_snapshot'] + '/' + name) == record['sha256'], 'registered numerical source: ' + name)
    specification = doc(PREP + 'specification.json')
    status = doc(PREP + 'status.json')
    require(specification['manifest_sha256'] == MANIFEST and specification['source_index_sha256'] == SOURCE_INDEX, 'specification seals')
    require(specification['supervisor_sha256'] == digest(PREP + 'run.py') and specification['cleanup_sha256'] == digest(PREP + 'process_cleanup.py'), 'supervisor bindings')
    require(specification['cpus'] == [12, 13, 14, 15] and specification['jobs'] == 4 and specification['affinity_exclusive'] is False, 'actual resource scope')
    require(specification['native_numerical_evaluations_authorized'] is False and specification['performance_acceptance_eligible'] is False, 'preparation-only authorization')
    require(status['status'] == 'passed' and status['amplitude_evaluations'] == 0, 'overall preparation result')
    require([s['id'] for s in status['stages']] == [s['id'] for s in specification['stages']] == STAGES, 'stage order and completeness')
    initial = doc(PREP + 'configuration-initial.json')
    require({k: v['sha256'] for k, v in initial.items()} == specification['configuration_hashes'], 'initial configuration binding')
    for value in initial.values():
        require(sha(value['text'].encode()) == value['sha256'], 'initial configuration bytes')
    source = specification['source_identity']
    require(source['files_sha256'] == GENERATOR and source['dirty'] is False, 'initial generator identity')
    derived_stages = []
    for wanted, got in zip(specification['stages'], status['stages']):
        name = got['id']
        require(got['command'] == wanted['command'], 'actual stage command: ' + name)
        require(got['status'] == 'passed' and got['returncode'] == 0, 'stage outcome: ' + name)
        require(got['cap_seconds'] == specification['per_stage_seconds'] == 3600 and 0 <= duration(got) < 3600, 'stage bound: ' + name)
        require(got['console_sha256'] == digest(PREP + name + '/console.log'), 'stage console: ' + name)
        if name != 'assert-declared-cards':
            require(got['cleanup']['survivors'] == [], 'child process cleanup: ' + name)
            require(got['cleanup']['observed_owned_processes'], 'child ownership evidence: ' + name)
        else:
            require(got['action'] == 'assert_declared_cards' and 'child_pid' not in got, 'internal assertion stage')
        for phase in ['before', 'after']:
            require(doc(PREP + name + '/configuration-' + phase + '.json') == initial, 'configuration stability: ' + name)
            snapshot = doc(PREP + name + '/source-' + phase + '.json')
            require(snapshot['files'] == source['files'] and snapshot['files_sha256'] == GENERATOR and snapshot['dirty'] is False, 'source stability: ' + name)
        derived_stages.append({k: got[k] for k in ['id', 'status', 'returncode', 'cap_seconds', 'console_sha256']} | {
            'elapsed_seconds': duration(got), 'cleanup_survivors': got.get('cleanup', {}).get('survivors')})
    require(specification['case_ids'] == [c['id'] for c in config['cases']] and len(config['cases']) == 11, 'eleven cases')
    expected_cards, derived_cases = {}, []
    for case in config['cases']:
        name = case['id']
        origin = manifest['physical_card_origins']['cards'][name]['param_card_sha256']
        point = doc(BASE + 'points/' + name + '.json')
        require(point['sha256'] == digest(BASE + 'points/' + name + '.dat') and point['inputs']['count'] == 100, 'frozen physical point stream: ' + name)
        row, compilations = {'id': name, 'sources': {}}, {}
        for who in ['baseline', 'candidate']:
            prefix = BASE + 'inputs/' + who + '/' + name + '/'
            generation, compilation = doc(prefix + 'generation.json'), doc(prefix + 'compilation.json')
            compilations[who] = compilation
            binding = {'id': CAMPAIGN, 'manifest_sha256': MANIFEST}
            require(generation['campaign_binding'] == compilation['campaign_binding'] == binding, 'case campaign binding')
            require(generation['case'] == case and generation['source'] == who, 'case identity')
            require(digest(prefix + 'param_card.dat') == generation['parameter_card_sha256'] == origin, 'physical card ancestry: ' + who + '/' + name)
            expected_cards[who + '/' + name] = {'card_sha256': origin, 'generation_sha256': digest(prefix + 'generation.json')}
            require(compilation['numerical_build_from_sources'] is True and compilation['full_numerical_rebuild_this_invocation'] is (who == 'candidate'), 'object build policy')
            command = ['make', '--always-make', '-j4', 'check'] if who == 'candidate' else ['make', '-j4', 'check']
            require(compilation['process_build_command'] == command, 'actual build command')
            require(compilation['link_inputs_after'] == compilation['link_inputs_before'], 'link inputs stable')
            require(compilation['masses'] == point['inputs']['incoming_masses'] + point['inputs']['masses'], 'mass probe and frozen stream')
            for scope, log_hash in compilation['build_logs'].items():
                require(digest(prefix + 'compilation-' + scope + '.log') == log_hash, 'compiler log binding')
            template = text('tools/tirust_benchmark_worker.f')
            require(sha(template.encode()) == compilation['worker_template_sha256'], 'worker template binding')
            for token, key in [('@PREFIX@', 'prefix'), ('@NEXTERNAL@', 'nexternal'), ('@NBORN@', 'nborn')]:
                template = template.replace(token, str(compilation['worker_render_parameters'][key]))
            require(sha(template.encode()) == compilation['driver_sha256'], 'rendered worker binding')
            drivers = [h for p, h in compilation['sources'].items() if p.endswith('/check_sa.f')]
            require(drivers == [compilation['driver_sha256']], 'compiled driver source inventory')
            if who == 'candidate':
                require('reused_from' not in generation and compilation['previous_compilation_sha256'] is None, 'fresh candidate objects')
                require(generation['source_identity']['files_sha256'] == GENERATOR, 'candidate generator source')
                require(generation['command_sha256'] == digest(BASE + 'commands/candidate-' + name + '.mg5'), 'actual CLI command')
                require(generation['export_environment'] == {'PYTHONHASHSEED': '0'}, 'deterministic CLI export')
                require(set(compilation['build_logs']) == {'MODEL', 'DHELAS', 'process'}, 'full candidate compilation logs')
            else:
                require(generation['source_identity']['revision'] == config['baseline_revision'], 'frozen baseline revision')
                reuse = generation['reused_from']
                proof = doc(prefix + 'object-relink-proof.json')
                require(proof == reuse['object_relink_proof'], 'baseline proof record binding')
                require(proof['passed'] and proof['objects_unchanged'] and proof['original_executable_unchanged'], 'baseline object proof result')
                history = [p for p in files if p.startswith(prefix + 'build-history/') and p.endswith('/compilation.json')]
                require(len(history) == 1, 'unique baseline compilation predecessor')
                old = doc(history[0])
                require(digest(history[0]) == reuse['compilation_sha256'], 'reused compilation bytes')
                require(proof['expected_executable_sha256'] == proof['relinked_executable_sha256'] == old['executable_sha256'], 'object relink reproduces predecessor executable')
                require(reuse['source_generation_repeated'] is False and reuse['worker_only_relink_required'] is True and reuse['numerical_objects'], 'baseline reuse scope')
                compiler_equal(old['effective_compiler_flags'], compilation['effective_compiler_flags'])
            row['sources'][who] = {
                'generation_sha256': digest(prefix + 'generation.json'),
                'compilation_sha256': digest(prefix + 'compilation.json'),
                'param_card_sha256': origin,
                'generation_seconds': generation['seconds'],
                'generation_repeated_this_campaign': not bool(generation.get('reused_from')),
                'observed_generator_revision': generation['source_identity']['revision'],
                'generator_files_sha256': generation['source_identity']['files_sha256'],
                'make_seconds': compilation['seconds'],
                'numerical_build_from_sources': compilation['numerical_build_from_sources'],
                'full_numerical_rebuild_this_invocation': compilation['full_numerical_rebuild_this_invocation'],
                'process_build_command': compilation['process_build_command'],
                'executable_sha256_identity_only': compilation['executable_sha256']}
        compiler_equal(compilations['baseline']['effective_compiler_flags'], compilations['candidate']['effective_compiler_flags'])
        derived_cases.append(row)
    for suffix in ['assert-declared-cards/cards.json', 'assert-declared-cards/cards-after.json',
                   'build-candidate/cards-before.json', 'build-candidate/cards-after.json']:
        require(doc(PREP + suffix) == expected_cards and len(expected_cards) == 22, 'all 22 cards before/after: ' + suffix)
    results = json.loads((folder / 'results.json').read_text())
    require(results['cases'] == derived_cases and results['stages'] == derived_stages, 'rederived result rows')
    require(results['total_elapsed_seconds'] == duration(status), 'rederived total wall time')
    require(results['status'] == 'passed' and results['amplitude_evaluations'] == results['native_files_included'] == 0, 'preparation-only result')
    require(results['performance_acceptance_eligible'] is False and results['overall_goal_qualified'] is False, 'no numerical or timing qualification')
    require(results['file_count'] == len(files) and results['unique_text_contents'] == len(objects) and results['gzip_bytes'] == len(packed), 'result inventory counts')
    return {'status': 'passed', 'retained_paths': len(files), 'unique_text_contents': len(objects),
            'stages': len(STAGES), 'physical_cards': len(expected_cards), 'cases': len(derived_cases),
            'amplitude_evaluations': 0, 'total_preparation_seconds': duration(status),
            'native_runtime_revalidated': False}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), sort_keys=True, indent=2))
