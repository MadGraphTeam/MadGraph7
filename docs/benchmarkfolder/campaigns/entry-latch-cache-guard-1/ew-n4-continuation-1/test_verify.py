"""Pure continuation verification tests; synthetic completion never reaches disk."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SPEC = importlib.util.spec_from_file_location('ew_continuation_verify', HERE / 'verify.py')
V = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(V)
INPUT_SHA = '6517159cbdea243e0767955d35f5935b09f56d53021720487e5f11803346225a'


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False).encode()


def put(evidence, name, value):
    evidence.blobs[name] = encoded(value)


def clone(evidence):
    result = copy.copy(evidence)
    result.index = copy.deepcopy(evidence.index)
    result.blobs = dict(evidence.blobs)
    return result


@pytest.fixture(scope='module')
def original():
    return V.Evidence(ROOT, V.HERE + '/inputs-index.json', INPUT_SHA)


def test_original_five_remain_incomplete(original):
    lanes, baseline, _, interruption = V.original(original, original.index)
    assert len(lanes) == 5 and len(baseline) == 200
    assert all(row['points'] == 100 for row in lanes)
    assert interruption['original_case_status'] == 'failed'
    assert interruption['last_saved_point'] == 74
    assert interruption['partial_raw_lines'] == 151


def test_original_replay_from_relocated_retained_files(original, tmp_path):
    for name, raw in original.blobs.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    index = tmp_path / V.HERE / 'inputs-index.json'
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_bytes((HERE / 'inputs-index.json').read_bytes())
    result = V.replay_original(tmp_path, V.HERE + '/inputs-index.json', INPUT_SHA)
    assert result['lanes_replayed'] == 5 and result['full_case_qualified'] is False
    assert result['native_executions'] == 0


@pytest.mark.parametrize('path', ['', '.', '../other', '/absolute', 'a/../b', './a', 'a//b', 'a\\b'])
def test_noncanonical_paths_rejected(path):
    with pytest.raises(ValueError):
        V.relative(path)


def test_index_hash_bytes_and_symlink_guards(tmp_path):
    (tmp_path / 'raw').write_bytes(b'closed')
    payload = {'files': {'raw': {'bytes': 6, 'sha256': V.sha(b'closed')}}, 'file_count': 1, 'total_bytes': 6}
    raw = encoded(payload)
    (tmp_path / 'index.json').write_bytes(raw)
    with pytest.raises(ValueError, match='hash mismatch'):
        V.Evidence(tmp_path, 'index.json', '0' * 64)
    assert V.Evidence(tmp_path, 'index.json', V.sha(raw)).data('raw') == b'closed'
    (tmp_path / 'raw').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Payload'):
        V.Evidence(tmp_path, 'index.json', V.sha(raw))
    (tmp_path / 'raw').unlink()
    (tmp_path / 'target').write_bytes(b'closed')
    (tmp_path / 'raw').symlink_to(tmp_path / 'target')
    with pytest.raises(ValueError, match='Symlink'):
        V.Evidence(tmp_path, 'index.json', V.sha(raw))


def test_unknown_file_not_accessible(original):
    with pytest.raises(ValueError, match='allowlist'):
        original.data('/proc/self/maps')


def test_original_interruption_cannot_be_relabelled(original):
    evidence = clone(original)
    name = V.OLD_CASE + '/status.json'
    status = evidence.load(name)
    status['status'] = 'passed'
    put(evidence, name, status)
    with pytest.raises(ValueError, match='interruption evidence changed|relabelled'):
        V.original(evidence, evidence.index)


def test_original_sixth_lane_not_admitted(original):
    evidence = clone(original)
    evidence.index['selected_original_lanes'] = V.descriptors(True)
    with pytest.raises(ValueError, match='selection'):
        V.original(evidence, evidence.index)


def test_original_aggregate_contradiction_rejected(original):
    evidence = clone(original)
    path = V.ORIGINAL + '/results.json'
    aggregate = evidence.load(path)
    aggregate['results'].append(aggregate['results'][-1])
    put(evidence, path, aggregate)
    with pytest.raises(ValueError, match='exactly its five'):
        V.original(evidence, evidence.index)


@pytest.fixture(scope='module')
def combined(original):
    """A fake fresh lane made from closed data; never a measured result."""
    evidence = clone(original)
    initial, baseline, records, _ = V.original(evidence, evidence.index)
    original_index = copy.deepcopy(evidence.index)
    input_name = V.HERE + '/inputs-index.json'
    evidence.blobs[input_name] = (HERE / 'inputs-index.json').read_bytes()
    prefix = V.HERE + '/lane'
    old_lane, new_lane = V.ORIGINAL + '/candidate-8', prefix + '/candidate-9'
    for name, raw in list(evidence.blobs.items()):
        if name.startswith(old_lane + '/'):
            evidence.blobs[new_lane + name[len(old_lane):]] = raw
    spec = evidence.load(V.ORIGINAL + '/specification.json')
    for name in ['specification.json', 'cases.json', 'campaign-manifest.json', *spec['protocols']]:
        evidence.blobs[prefix + '/' + name] = evidence.data(V.ORIGINAL + '/' + name)
    for name in ('raw/warmup.dat', 'candidate-initialization/warmup.dat'):
        lines = []
        for line in evidence.data(new_lane + '/' + name).decode().splitlines():
            if line.startswith('#'):
                lines.append(line)
            else:
                fields = line.split()
                fields[3] = {'218': '219', '322': '323'}.get(fields[3], fields[3])
                lines.append(' '.join(fields))
        evidence.blobs[new_lane + '/' + name] = ('\n'.join(lines) + '\n').encode()
    inputs = evidence.load(new_lane + '/candidate-inputs.json')
    inputs['settings']['mlreductionlib'] = '9'
    inputs['numerical_cards'] = evidence.load(V.ORIGINAL + '/candidate-9/candidate-inputs.json')['numerical_cards']
    put(evidence, new_lane + '/candidate-inputs.json', inputs)
    after = evidence.load(new_lane + '/inputs-after.json')
    after['numerical_cards'] = inputs['numerical_cards']
    put(evidence, new_lane + '/inputs-after.json', after)
    _, case, _, compare, _ = V.helpers(evidence, prefix)
    # Use the same extracted parser as production, with its real status rules.
    worker, parser = {}, {'math': V.math, 'Path': lambda name: V.RetainedPath(name, evidence)}
    V.extract(evidence.data(prefix + '/tirust_benchmark_worker.py'),
              {'BACKEND_CODES', 'MeasurementError', 'validate_status'}, worker)
    parser.update(worker=V.SimpleNamespace(**worker), MeasurementError=worker['MeasurementError'])
    V.extract(evidence.data(prefix + '/tirust_benchmark_warmup.py'), {'_call_groups', 'parse_warmup'}, parser)
    observation = parser['parse_warmup'](new_lane + '/raw', [9], 100, evidence.load(new_lane + '/raw/ready.json'))
    value = evidence.load(new_lane + '/result.json')
    value.update(backend=9, observation=observation, comparison=compare(baseline, observation['rows'], 3e-6))
    put(evidence, new_lane + '/result.json', value)
    put(evidence, new_lane + '/raw/observation.json', observation)
    for name in ('compiled-inputs.json', 'verified-builds.json'):
        put(evidence, V.STAGE + '/' + name, records)
    for name in ('configuration-before.json', 'configuration-after.json'):
        put(evidence, V.STAGE + '/' + name, {'synthetic': 'unchanged'})
    evidence.blobs[V.STAGE + '/console.log'] = b'synthetic closed stage\n'
    put(evidence, V.HERE + '/status.json', {'status': 'passed', 'stages': [{
        'id': 'numerical-' + V.CASE, 'status': 'passed', 'returncode': 0,
        'cleanup': {'survivors': []}, 'console_sha256': evidence.digest(V.STAGE + '/console.log')}]})
    seals = [
        ('source-index.json', {'schema': 'madloop-tirust-ew-single-lane-source-v1',
            'native_lanes': [['candidate', 9]], 'points': 100, 'worker_timeout_seconds': 14400,
            'stage_cap_seconds': 28800, 'cleanup_reserve_seconds': 60, 'automatic_retry': False,
            'timing_authorized': False}, ['README.md', 'inputs-index.json', 'preflight.json', 'run.py', 'test_run.py']),
        ('replay-index.json', {'schema': 'madloop-tirust-ew-continuation-replay-source-v1'}, ['verify.py', 'test_verify.py'])]
    for name, seal, names in seals:
        for item in names:
            if item != 'inputs-index.json':
                evidence.blobs[V.HERE + '/' + item] = b'synthetic sealed source\n'
        seal['files'] = {item: {'bytes': len(evidence.data(V.HERE + '/' + item)),
                              'sha256': evidence.digest(V.HERE + '/' + item)} for item in names}
        if name == 'source-index.json':
            dependency = 'tools/synthetic-closed-dependency.py'
            evidence.blobs[dependency] = b'synthetic dependency\n'
            seal['dependencies'] = {dependency: {'bytes': len(evidence.data(dependency)),
                                                 'sha256': evidence.digest(dependency)}}
        put(evidence, V.HERE + '/' + name, seal)
    fresh, _ = V.replay_lanes(evidence, V.descriptors(True)[-1:], records, baseline)
    evidence.index = {'schema': 'madloop-tirust-combined-one-case-closure-v1', 'campaign': V.CAMPAIGN,
        'case': case, 'cases': [V.case_summary(initial + fresh)], 'performance_acceptance_eligible': False,
        'manifest_sha256': V.MANIFEST, 'source_index_sha256': evidence.digest(V.HERE + '/source-index.json'),
        'replay_index_sha256': evidence.digest(V.HERE + '/replay-index.json'), 'lanes': V.descriptors(True),
        'interruption': original_index['interruption'], 'files': original_index['files'],
        'continuation': {'input_index_path': input_name, 'input_index_sha256': INPUT_SHA,
            'status_path': V.HERE + '/status.json', 'stage_directory': V.STAGE,
            'configuration_before_path': V.STAGE + '/configuration-before.json',
            'configuration_after_path': V.STAGE + '/configuration-after.json'}}
    return evidence


def check_combined(monkeypatch, evidence):
    # The constructor's independent filesystem/byte guards are tested above.
    monkeypatch.setattr(V, 'Evidence', lambda *args: evidence)
    return V.verify(ROOT, V.HERE + '/checkpoint.json', 'a' * 64)


def test_synthetic_complete_continuation_preserves_original_failure(combined, monkeypatch):
    result = check_combined(monkeypatch, combined)
    assert result['verified'] and result['lanes_replayed'] == 6
    assert result['native_executions'] == 0 and result['performance_acceptance'] is False
    assert result['interruption']['original_case_status'] == 'failed'
    assert result['interruption']['last_saved_point'] == 74
    assert result['continuation'] == {
        'reason': 'user-requested daemon update',
        'original_attempt': {'path': V.OLD_CASE, 'status': 'interrupted',
            'selected_completed_lanes': ['baseline-1', 'candidate-1', 'candidate-6', 'candidate-7', 'candidate-8'],
            'partial_backend9_selected': False, 'last_saved_point': 74},
        'fresh_attempt': {'path': V.HERE + '/lane', 'selected_lane': 'candidate-9', 'points': 100},
        'concatenated': False}
    assert result['continuation_bindings'] == combined.index['continuation']


@pytest.mark.parametrize('change,match', [
    ('five', 'exact five original'), ('old-nine', 'exact five original'),
    ('partial', '100-point|traversal|warmup|warm-up|Warm'), ('survivor', 'cleanup'),
    ('runtime', 'admitted build/runtime'), ('configuration', 'Configuration changed'),
    ('missing', 'allowlist'), ('summary', 'summary differs'),
    ('parser', 'protocol differs'), ('settings', 'card/settings differ'),
    ('missing-dependency', 'allowlist'), ('mutated-dependency', 'dependency changed'),
])
def test_reject_invalid_combined_evidence(combined, monkeypatch, change, match):
    evidence = clone(combined)
    lane = V.HERE + '/lane/candidate-9'
    if change == 'five':
        evidence.index['lanes'].pop()
    elif change == 'old-nine':
        evidence.index['lanes'][-1]['directory'] = V.ORIGINAL + '/candidate-9'
    elif change == 'partial':
        for name in ('raw/warmup.dat', 'candidate-initialization/warmup.dat'):
            lines = evidence.data(lane + '/' + name).decode().splitlines()
            evidence.blobs[lane + '/' + name] = ('\n'.join(lines[:-2]) + '\n').encode()
    elif change == 'survivor':
        name = V.HERE + '/status.json'
        value = evidence.load(name)
        value['stages'][0]['cleanup']['survivors'] = [42]
        put(evidence, name, value)
    elif change == 'runtime':
        name = lane + '/inputs-after.json'
        value = evidence.load(name)
        value['runtime']['libraries']['/different/library.so'] = '0' * 64
        put(evidence, name, value)
    elif change == 'configuration':
        put(evidence, V.STAGE + '/configuration-after.json', {'synthetic': 'changed'})
    elif change == 'missing':
        del evidence.blobs[lane + '/raw/warmup.dat']
    elif change == 'summary':
        evidence.index['cases'][0]['lanes'][-1]['matched_saved_points'] = 74
    elif change == 'parser':
        name = V.HERE + '/lane/tirust_benchmark.py'
        evidence.blobs[name] += b'\n# changed protocol\n'
        spec_name = V.HERE + '/lane/specification.json'
        spec = evidence.load(spec_name)
        spec['protocols']['tirust_benchmark.py'] = evidence.digest(name)
        put(evidence, spec_name, spec)
    elif change == 'settings':
        name = lane + '/candidate-inputs.json'
        inputs = evidence.load(name)
        inputs['settings']['extra_option'] = True
        put(evidence, name, inputs)
    elif change == 'missing-dependency':
        del evidence.blobs['tools/synthetic-closed-dependency.py']
    elif change == 'mutated-dependency':
        evidence.blobs['tools/synthetic-closed-dependency.py'] = b'mutated dependency\n'
    with pytest.raises(ValueError, match=match):
        check_combined(monkeypatch, evidence)
