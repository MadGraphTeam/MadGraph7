#!/usr/bin/env python3
"""Reparse frozen EW4 observations and its incomplete initialization trace."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import runpy

HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]
FOLDER = HERE / 'strict-policy-ew-n4-warmup-v2'
OUTPUT = HERE / 'ew-n4-warmup-v2-index.json'
README = HERE / 'README-ew-n4-warmup-v2.md'


def require(value, message):
    if not value:
        raise ValueError(message)


def record(path):
    data = path.read_bytes()
    return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def load(path):
    return json.loads(path.read_text())


def inventory():
    files = {}
    for path in sorted(FOLDER.rglob('*')):
        require(not path.is_symlink(), 'Unexpected evidence symlink')
        if path.is_file():
            files[path.relative_to(REPORT).as_posix()] = record(path)
    return files


def audit():
    files = inventory()
    helper_path = HERE / 'audit_completed_pilots.py'
    helper = runpy.run_path(str(helper_path), run_name='ew4_audit_helpers')
    spec, config, outcomes = (load(FOLDER / name) for name in
                              ('specification.json', 'cases.json', 'results.json'))
    case = spec['case']
    require(outcomes['case'] == case and spec['performance_eligible'] is False
            and outcomes['performance_eligible'] is False, 'Scope mismatch')
    require(record(FOLDER / 'cases.json')['sha256'] == spec['configuration_sha256'], 'Config hash')
    require(config['points_per_case'] == 100 and config['relative_tolerance'] == 3e-6, 'Numerical policy')
    for name, digest in spec['protocols'].items():
        require(record(FOLDER / name)['sha256'] == digest, 'Frozen protocol hash: ' + name)
    expected = [('baseline', 1), ('candidate', 1), ('candidate', 6),
                ('candidate', 7), ('candidate', 8), ('candidate', 9)]
    require([(r['source'], r['backend']) for r in outcomes['results']] == expected, 'Lane coverage')
    rows_out, identities, reference, generations, physical = [], {}, None, {}, set()
    for (source, backend), result in zip(expected, outcomes['results']):
        lane = FOLDER / f'{source}-{backend}'
        inp = load(lane / f'{source}-inputs.json')
        gen, comp = inp['generation'], inp['compilation']
        require(gen['source'] == source and gen['case'] == case, 'Generated case/source')
        require(inp['settings']['mlreductionlib'] == str(backend)
                and inp['settings']['mlstabthres'] == 1e-6, 'Explicit strict backend policy')
        policy = dict(config['madloop_parameters'])
        if source == 'candidate':
            policy.update(config['candidate_parameters'])
        require(all(inp['settings'][key.lower()] == value for key, value in policy.items()), 'Card policy')
        for path, expected_hash in (
            (REPORT / 'points' / (case['id'] + '.dat'), inp['points_sha256']),
            (REPORT / 'inputs' / source / case['id'] / 'param_card.dat', inp['param_card_sha256']),
            (REPORT / 'commands' / f"{source}-{case['id']}.mg5", gen['command_sha256'])):
            require(record(path)['sha256'] == expected_hash, 'Physical input hash: ' + str(path))
            files[path.relative_to(REPORT).as_posix()] = record(path)
        require(inp['param_card_sha256'] == gen['parameter_card_sha256'], 'Parameter-card binding')
        physical.add((inp['points_sha256'], inp['param_card_sha256']))
        require(comp['numerical_build_from_sources'] is True
                and comp['full_numerical_rebuild_this_invocation'] is False
                and re.fullmatch('[0-9a-f]{64}', comp['previous_compilation_sha256']), 'Inherited numerical build provenance')
        require(comp['link_inputs_before'] == comp['link_inputs_after'], 'Link inputs changed')
        for kind in ('archives', 'metadata'):
            require(inp['runtime'][kind] == comp['link_inputs_after'][kind], 'Runtime/link binding')
        require(comp['worker_template_sha256'] == spec['protocols']['tirust_benchmark_worker.f'], 'Native worker template')
        rendered = (FOLDER / 'tirust_benchmark_worker.f').read_text()
        for key, token in [('prefix', 'PREFIX'), ('nexternal', 'NEXTERNAL'), ('nborn', 'NBORN')]:
            rendered = rendered.replace('@' + token + '@', str(comp['worker_render_parameters'][key]))
        driver_hash = hashlib.sha256(rendered.encode()).hexdigest()
        require(driver_hash == comp['driver_sha256'], 'Compiled worker rendering')
        require([value for name, value in comp['sources'].items()
                 if re.fullmatch(r'SubProcesses/P[^/]+/check_sa\.f', name)] == [driver_hash], 'Driver source inventory')
        if source in generations:
            require(generations[source] == (gen, comp), 'Backend source/build differs')
        generations[source] = (gen, comp)
        identities[lane.name] = {
            'generator_revision': gen['source_identity']['revision'],
            'generator_files_sha256': gen['source_identity']['files_sha256'],
            'executable_sha256_recorded': comp['executable_sha256'],
            'driver_sha256': driver_hash,
            'previous_compilation_sha256_recorded': comp['previous_compilation_sha256'],
            'full_numerical_rebuild_this_invocation': False,
            'numerical_build_from_sources_recorded': True,
            'runtime_libraries_recorded': inp['runtime']['libraries'],
            'historical_binary_bytes_revalidated': False}
        if backend == 9:
            require(result == load(lane / 'failure.json') and result['status'] == 'failed', 'TIRust timeout outcome')
            require('Worker protocol timeout' in result['error'], 'Expected initialization timeout')
            require(not (lane / 'raw').exists() and not (lane / 'result.json').exists(), 'Unexpected completed TIRust stream')
            continue
        require(result == load(lane / 'result.json') and result['status'] == 'complete', 'Completed lane outcome')
        rows, observation, counts, usage = helper['parse_observation'](lane, backend)
        require(observation == result['observation'], 'Observation duplicate')
        if reference is None:
            reference = rows
        comparison, misses = helper['compare'](reference, rows, backend, result['comparison'])
        rows_out.append({'source': source, 'backend': backend, 'points': 100,
                         'coefficient_rows': len(rows), 'status_counts': dict(counts),
                         'backend_precision_counts': dict(usage), **comparison,
                         'ordinary_target_miss_points': sorted({r['point'] for r in misses})})
    require(len(physical) == 1, 'Physical input identity differs between lanes')
    require(not helper['ERRORS'], '; '.join(helper['ERRORS']))
    warm = FOLDER / 'candidate-9/candidate-initialization/warmup.dat'
    trace = []
    for line in warm.read_text().splitlines():
        if line.startswith('#'):
            continue
        fields = line.split()
        require(len(fields) == 9, 'Malformed warmup row')
        integers, numbers = tuple(map(int, fields[:4])), tuple(map(float, fields[4:]))
        require(all(map(math.isfinite, numbers)), 'Nonfinite warmup trace')
        trace.append((integers, numbers))
    require([item[0][:3] for item in trace] ==
            [(call, 1 if call == 0 else call, order) for call in range(66) for order in range(2)],
            'Expected bootstrap0 then completed calls1..65, both answer orders')
    require(all(item[0][3] in (219, 323) for item in trace), 'Unexpected partial warmup status')
    for first, second in zip(trace[::2], trace[1::2]):
        require(first[0][3] == second[0][3], 'Warmup split-order status disagreement')
    warm_counts = Counter(str(item[0][3]) for item in trace if item[0][2] == 0)
    fail = load(FOLDER / 'candidate-9/candidate-initialization/failure.json')
    require(fail['error_class'] == 'ProtocolError' and 'Worker protocol timeout' in fail['error'], 'Initialization failure binding')
    require('READY' not in (FOLDER / 'candidate-9/candidate-output.txt').read_text(), 'Unexpected stdout READY')
    require('timeout' not in spec, 'Original specification unexpectedly has timeout')
    require(inventory() == {k: v for k, v in files.items() if k.startswith('experiments/numerical/' + FOLDER.name + '/')},
            'Original evidence changed during audit')
    files[helper_path.relative_to(REPORT).as_posix()] = record(helper_path)
    files[Path(__file__).relative_to(REPORT).as_posix()] = record(Path(__file__))
    files[README.relative_to(REPORT).as_posix()] = record(README)
    return {'schema_version': 1, 'label': FOLDER.name, 'case': case,
            'scope': 'Five completed reference observations and one incomplete TIRust initialization; no acceptance timing',
            'performance_eligible': False, 'all_five_candidate_backends_qualified': False,
            'native_executions_by_audit': 0, 'original_files_modified': False,
            'completed_lanes': rows_out, 'completed_observation_point_evaluations': 500,
            'candidate9': {'classification': 'initialization_protocol_timeout',
                'ready_reached': False, 'completed_observation_points': 0,
                'initialization_coefficient_rows': len(trace), 'completed_public_calls': 66,
                'bootstrap_call0_count': 1, 'subsequent_completed_warmup_calls': 65,
                'unique_input_points_seen': 65, 'status_counts_per_public_call': dict(warm_counts),
                'actual_backend_precision_per_public_call': {'9:DP': warm_counts['219'], '9:QP': warm_counts['323']},
                'accuracy_claim': 'Unqualified partial trace; no final 100-point TIRust accuracy result',
                'requested_timeout_seconds_session_reported': 1800,
                'timeout_numeric_value_recorded_in_original_attempt': False,
                'timeout_evidence_limit': 'The original specification omits the CLI timeout; failure records prove a protocol timeout but do not encode its duration. 1800 seconds is the coordinator-reported --timeout command, not reconstructed from timestamps.',
                'next_call_completion_known': False},
            'physical_inputs': {'points_sha256': next(iter(physical))[0], 'param_card_sha256': next(iter(physical))[1]},
            'retained_protocols': spec['protocols'], 'identities': identities,
            'evidence_limits': ['Complete observations pass the 3e-6 comparator; candidate6/7/8 miss the ordinary1e-8 target against the retained CutTools reference.',
                'Native driver bytes are reconstructed from the frozen template and checked against compilation records. Python protocol copies are hashed as retained; no new native execution is performed.',
                'This invocation rebuilt the worker against previously compiled numerical objects; prior compilation hashes and numerical-build provenance are retained, not relabelled as a fresh full rebuild.',
                'Loaded-library/executable hashes are historical recorded identities; this audit does not re-read mutable installation binaries.'],
            'retained_attempt_file_count': len(inventory()), 'files': files, 'all_audit_checks_pass': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Create the index; refuse to overwrite')
    args = parser.parse_args()
    result = audit()
    if args.write:
        with OUTPUT.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
    else:
        require(result == load(OUTPUT), 'Audit differs from frozen index')
    print(json.dumps({'checks_pass': True, 'completed_points': 500,
                      'tirust_completed_observation_points': 0,
                      'tirust_partial_warmup_calls': 66, 'index_sha256': record(OUTPUT)['sha256']}))
