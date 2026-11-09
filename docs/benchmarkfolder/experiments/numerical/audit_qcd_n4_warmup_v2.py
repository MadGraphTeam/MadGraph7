#!/usr/bin/env python3
"""Hash/reparse frozen QCD4 evidence; never run native code or postchecks.

Only --write creates the adjacent index, exclusively. Default mode verifies
that index. Original attempts and the separate backend-8 forensic stay intact.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPORT = HERE.parents[1]
ROOT = REPORT.parents[1]
FOLDER = HERE / 'strict-policy-qcd-n4-warmup-v2'
FORENSIC = HERE / 'qcd-n4-backend8-resource-failure'
OUTPUT = HERE / 'qcd-n4-warmup-v2-index.json'
README = HERE / 'README-qcd-n4-warmup-v2.md'
HELPER = HERE / 'audit_completed_pilots.py'
HELPER_SHA = 'beee4203c0864fe1f21c01495334aefc47c0d66d1ca210692e5201d5af70f38f'
FORENSIC_SHA = 'cead7b6c7000317f35d7c07ffba43009d451a3283042d1d39b516a621b0bcb4f'
WARM_SHA = '33968008a3cd21ec02d49735eab04a44ea434c315115b02a09373a289f1eddf2'
PROTOCOLS = {'tirust_benchmark.py', 'tirust_benchmark_compiler.py',
             'tirust_benchmark_numerical.py', 'tirust_benchmark_worker.f', 'tirust_benchmark_worker.py'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(path.is_absolute() and path.is_relative_to(REPORT) and '..' not in path.parts and
            path.resolve().is_relative_to(REPORT), 'Evidence path escapes report tree')
    require(not any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(REPORT)),
            'Unexpected evidence symlink')
    require(path.is_file(), 'Missing evidence file: ' + str(path))
    return path.read_bytes()


def record(path):
    data = read(path)
    return dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def load(path):
    return json.loads(read(path))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inventory():
    files = {}
    for path in sorted(FOLDER.rglob('*')):
        require(not path.is_symlink(), 'Unexpected attempt symlink')
        if path.is_file():
            require('__pycache__' not in path.parts and path.suffix != '.pyc', 'Unexpected bytecode in attempt')
            files[path.relative_to(REPORT).as_posix()] = record(path)
        else:
            require(path.is_dir(), 'Nonregular attempt entry')
    return files


def bind(files, path):
    files[path.relative_to(REPORT).as_posix()] = record(path)
    return files[path.relative_to(REPORT).as_posix()]


def audit_backend8(files, result):
    lane = FOLDER / 'candidate-8'
    require(result == load(lane / 'failure.json') == {
        'source': 'candidate', 'backend': 8, 'status': 'failed', 'performance_eligible': False,
        'error': "BlockingIOError(11, 'Resource temporarily unavailable')"}, 'Backend8 failure changed')
    for name in ('result.json', 'raw/observation.json', 'inputs-after.json'):
        require(not (lane / name).exists(), 'Unexpected backend8 completed evidence: ' + name)
    require(record(FORENSIC / 'forensic.json')['sha256'] == FORENSIC_SHA, 'Retained forensic identity')
    forensic = load(FORENSIC / 'forensic.json')
    for name in ('forensic.py', 'forensic.json', 'README.md'):
        bind(files, FORENSIC / name)
    for name, digest in forensic['bound_evidence_sha256'].items():
        require(bind(files, ROOT / name)['sha256'] == digest, 'Forensic bound evidence mismatch: ' + name)
    require(forensic['original_status'] == 'failed' and forensic['original_error'] == result['error'] and
            not any(forensic[key] for key in ('formal_numerical_qualification', 'performance_eligible',
                'retrospective_promotion', 'native_or_compiler_jobs_started', 'postchecks_rerun',
                'original_completed_postverification_evidence')), 'Forensic qualification scope')
    dimensions = tuple(map(int, read(lane / 'raw/dimensions.dat').split()))
    require(dimensions == (1, 1, 1), 'Backend8 raw dimensions')
    rows = []
    for line in read(lane / 'raw/values.dat').decode().splitlines():
        fields = line.split()
        require(len(fields) == 8, 'Backend8 raw row shape')
        point, order, status = map(int, fields[:3])
        numbers = list(map(float, fields[3:]))
        require(status in (218, 322) and all(map(math.isfinite, numbers)) and numbers[-1] >= 0,
                'Backend8 raw status/value')
        rows.append((point, order, status))
    require([(p, o) for p, o, _ in rows] == [(p, o) for p in range(1, 101) for o in range(2)],
            'Backend8 raw point/order coverage')
    require(all(left[2] == right[2] for left, right in zip(rows[::2], rows[1::2])), 'Backend8 split statuses')
    counts = Counter(str(status) for _, order, status in rows if order == 0)
    histogram = [tuple(map(int, line.split())) for line in read(lane / 'raw/timed_status.dat').decode().splitlines()]
    require(len(histogram) == len({code for code, _ in histogram}) and
            {str(code): count for code, count in histogram} == dict(counts) == {'218': 96, '322': 4},
            'Backend8 timed/raw status counts')
    qp_points = [p for p, order, status in rows if order == 0 and status == 322]
    require(qp_points == [84, 89, 93, 94] == forensic['accepted_qp_points'], 'Backend8 QP point identities')
    require(forensic['accepted_status_counts'] == dict(counts) and
            forensic['accepted_backend_precision_counts'] == {'8:DP': 96, '8:QP': 4} and
            forensic['raw_shape'] == dict(points=100, coefficient_rows=200,
                answer_dimensions=dict(allocated=1, loop_orders=1, born_orders=1)), 'Forensic raw-summary equality')
    return dict(classification='failed_controller_result_with_retained_raw_outputs', original_status='failed',
        original_error=result['error'], completed_controller_observation_points=0,
        retained_raw_points=100, retained_coefficient_rows=200, status_counts=dict(counts),
        actual_backend_precision_counts={'8:DP': 96, '8:QP': 4}, qp_points=qp_points,
        completed_post_measurement_verification_evidence=False,
        exact_failing_operation_known=False,
        forensic_record=(FORENSIC / 'forensic.json').relative_to(REPORT).as_posix(),
        forensic_sha256=FORENSIC_SHA, forensic_bound_file_count=len(forensic['bound_evidence_sha256']),
        accuracy_claim='None from this failed lane. The separately retained raw-data forensic is bound, not promoted to a completed numerical qualification.',
        error_evidence_limit='The failure has no traceback. The archived controller has no explicit post-observation verification step; the failing operation cannot be identified.')


def audit():
    original_files = inventory()
    files = dict(original_files)
    require(bind(files, HELPER)['sha256'] == HELPER_SHA, 'Pinned read-only parser helper identity')
    helper = runpy.run_path(str(HELPER), run_name='qcd4_audit_helpers')
    spec, config, outcomes = (load(FOLDER / name) for name in ('specification.json', 'cases.json', 'results.json'))
    case = spec['case']
    require(case['id'] == 'qcd_n4_uux_zzgg' and case['nfinal'] == 4 and case in config['cases'] and
            outcomes['case'] == case and spec['performance_eligible'] is False and
            outcomes['performance_eligible'] is False, 'Case/scope mismatch')
    require(record(FOLDER / 'cases.json')['sha256'] == spec['configuration_sha256'], 'Configuration hash')
    require(config['points_per_case'] == 100 and config['relative_tolerance'] == 3e-6 and
            config['ordinary_target_tolerance'] == 1e-8, 'Numerical comparison policy')
    require(set(spec['protocols']) == PROTOCOLS, 'Protocol inventory')
    for name, digest in spec['protocols'].items():
        require(record(FOLDER / name)['sha256'] == digest, 'Frozen protocol hash: ' + name)
    points = REPORT / 'points' / (case['id'] + '.dat')
    point_meta = points.with_suffix('.json')
    point_data, metadata = read(points), load(point_meta)
    require(int(point_data.splitlines()[0]) == 100 and len(point_data.splitlines()) == 601 and
            metadata['inputs']['count'] == 100 and bind(files, points)['sha256'] == metadata['sha256'],
            'Physical point stream shape/hash')
    bind(files, point_meta)
    baseline_source = load(REPORT / 'inputs/baseline.json')
    bind(files, REPORT / 'inputs/baseline.json')
    expected = [('baseline', 1), ('candidate', 1), ('candidate', 6),
                ('candidate', 7), ('candidate', 8), ('candidate', 9)]
    require([(r['source'], r['backend']) for r in outcomes['results']] == expected, 'Lane inventory')
    rows_out, identities, reference, generations, physical, all_settings = [], {}, None, {}, set(), {}
    backend8 = None
    for (source, backend), result in zip(expected, outcomes['results']):
        lane = FOLDER / f'{source}-{backend}'
        inp = load(lane / f'{source}-inputs.json')
        gen, comp = inp['generation'], inp['compilation']
        require(gen['source'] == source and gen['case'] == case, 'Generated case/source')
        require(inp['settings']['mlreductionlib'] == str(backend) and inp['settings']['mlstabthres'] == 1e-6,
                'Strict explicit-backend policy')
        policy = dict(config['madloop_parameters'])
        if source == 'candidate':
            policy.update(config['candidate_parameters'])
        require(all(inp['settings'][key.lower()] == value for key, value in policy.items()), 'Recorded reduction policy')
        normalized_settings = {key: value for key, value in inp['settings'].items() if key != 'mlreductionlib'}
        if source in all_settings:
            require(normalized_settings == all_settings[source], 'Backend settings differ beyond selection')
        all_settings[source] = normalized_settings
        require(inp['points_sha256'] == metadata['sha256'], 'Input point binding')
        for path, expected_hash in (
                (REPORT / 'inputs' / source / case['id'] / 'param_card.dat', inp['param_card_sha256']),
                (REPORT / 'commands' / f"{source}-{case['id']}.mg5", gen['command_sha256'])):
            require(bind(files, path)['sha256'] == expected_hash, 'Physical input hash: ' + str(path))
        require(inp['param_card_sha256'] == gen['parameter_card_sha256'], 'Generation/card binding')
        physical.add((inp['points_sha256'], inp['param_card_sha256']))
        identity = gen['source_identity']
        if source == 'candidate':
            require(canonical(identity['files']) == identity['files_sha256'], 'Generator source inventory digest')
            snapshot = REPORT / 'inputs/source-identities' / (canonical(identity) + '.json')
            require(load(snapshot) == identity, 'Retained generator source identity')
            bind(files, snapshot)
        else:
            require(identity['revision'] == baseline_source['revision'] == config['baseline_revision'] and
                    identity['files_sha256'] == canonical(baseline_source['files']) and
                    identity['archive_sha256'] == baseline_source['archive_sha256'], 'Baseline source identity')
        require(comp['numerical_build_from_sources'] is True and comp['full_numerical_rebuild_this_invocation'] is False and
                re.fullmatch('[0-9a-f]{64}', comp['previous_compilation_sha256']), 'Inherited numerical build provenance')
        require(comp['masses'] == metadata['inputs']['incoming_masses'] + metadata['inputs']['masses'], 'Compiled masses')
        require(comp['link_inputs_before'] == comp['link_inputs_after'], 'Link dependencies changed')
        for kind in ('archives', 'metadata'):
            require(inp['runtime'][kind] == comp['link_inputs_after'][kind], 'Runtime/link identity')
        for name, value in comp['settings'].items():
            require(hashlib.sha256(value.encode()).hexdigest() == comp['sources'][name], 'Recorded build-source identity')
        require(comp['worker_template_sha256'] == spec['protocols']['tirust_benchmark_worker.f'], 'Native worker template')
        rendered = read(FOLDER / 'tirust_benchmark_worker.f').decode()
        for key, token in [('prefix', 'PREFIX'), ('nexternal', 'NEXTERNAL'), ('nborn', 'NBORN')]:
            rendered = rendered.replace('@' + token + '@', str(comp['worker_render_parameters'][key]))
        driver_hash = hashlib.sha256(rendered.encode()).hexdigest()
        require(driver_hash == comp['driver_sha256'] and [value for name, value in comp['sources'].items()
                if re.fullmatch(r'SubProcesses/P[^/]+/check_sa\.f', name)] == [driver_hash], 'Rendered driver/source inventory')
        if source in generations:
            require(generations[source] == (gen, comp), 'Backend source/build differs')
        generations[source] = (gen, comp)
        identities[lane.name] = dict(generator_revision=identity['revision'], generator_files_sha256=identity['files_sha256'],
            generator_dirty=identity['dirty'], generated_source_count=len(comp['sources']),
            generated_sources_sha256=canonical(comp['sources']), compiler_policy_sha256=canonical(comp['effective_compiler_flags']),
            recorded_settings_sha256=canonical(inp['settings']), executable_sha256_recorded=comp['executable_sha256'],
            driver_sha256=driver_hash, previous_compilation_sha256_recorded=comp['previous_compilation_sha256'],
            full_numerical_rebuild_this_invocation=False, numerical_build_from_sources_recorded=True,
            runtime_libraries_recorded=inp['runtime']['libraries'], historical_binary_bytes_revalidated=False)
        if backend == 8:
            backend8 = audit_backend8(files, result)
            continue
        if backend == 9:
            require(result == load(lane / 'failure.json') and result['status'] == 'failed' and
                    'Worker protocol timeout' in result['error'], 'TIRust timeout outcome')
            require(not (lane / 'raw').exists() and not (lane / 'result.json').exists(), 'Unexpected completed TIRust stream')
            continue
        require(result == load(lane / 'result.json') and result['status'] == 'complete' and
                result['performance_eligible'] is False and not (lane / 'failure.json').exists(), 'Completed lane outcome')
        rows, observation, counts, usage = helper['parse_observation'](lane, backend)
        require(observation == result['observation'], 'Observation duplicate mismatch')
        if reference is None:
            reference = rows
        comparison, misses = helper['compare'](reference, rows, backend, result['comparison'])
        rows_out.append(dict(source=source, backend=backend, points=100, coefficient_rows=len(rows),
            status_counts=dict(counts), backend_precision_counts=dict(usage), **comparison,
            ordinary_target_miss_points=sorted({row['point'] for row in misses})))
    shared = all_settings['baseline'].keys() & all_settings['candidate'].keys()
    require(all(all_settings['baseline'][key] == all_settings['candidate'][key] for key in shared), 'Shared numerical policy differs')
    require(len(physical) == 1 and backend8 is not None, 'Physical identity or forensic coverage')
    require(not helper['ERRORS'], '; '.join(helper['ERRORS']))
    warm = FOLDER / 'candidate-9/candidate-initialization/warmup.dat'
    require(record(warm)['sha256'] == WARM_SHA, 'TIRust warmup trace identity')
    require(tuple(map(int, read(warm.with_name('dimensions.dat')).split())) == (1, 1, 1), 'Warmup dimensions')
    trace = []
    for line in read(warm).decode().splitlines():
        if line.startswith('#'):
            continue
        fields = line.split()
        require(len(fields) == 9, 'Malformed warmup row')
        integers, numbers = tuple(map(int, fields[:4])), tuple(map(float, fields[4:]))
        require(all(map(math.isfinite, numbers)) and numbers[-1] >= 0, 'Nonfinite/invalid warmup values')
        trace.append((integers, numbers))
    require([item[0][:3] for item in trace] == [(call, 1 if call == 0 else call, order)
            for call in range(28) for order in range(2)], 'Bootstrap0 + calls1..27 paired coverage')
    require(all(row[0][3] in (219, 323) for row in trace) and
            all(left[0][3] == right[0][3] for left, right in zip(trace[::2], trace[1::2])), 'Warmup statuses')
    warm_counts = Counter(str(item[0][3]) for item in trace if item[0][2] == 0)
    require(dict(warm_counts) == {'219': 6, '323': 22}, 'Warmup precision counts')
    fail = load(FOLDER / 'candidate-9/candidate-initialization/failure.json')
    require(fail['error_class'] == 'ProtocolError' and 'Worker protocol timeout' in fail['error'], 'Startup failure binding')
    require(re.search(r'^READY(?:\s|$)', read(FOLDER / 'candidate-9/candidate-output.txt').decode(), re.M) is None,
            'Unexpected stdout READY')
    require('timeout' not in spec, 'Specification unexpectedly records timeout')
    require(inventory() == original_files, 'Original evidence changed during audit')
    bind(files, Path(__file__))
    bind(files, README)
    return dict(schema_version=1, label=FOLDER.name, case=case,
        scope='Four completed 100-point observations, one failed controller lane with retained raw values, and one partial TIRust initialization; no acceptance timing.',
        performance_eligible=False, all_five_candidate_backends_qualified=False,
        native_executions_by_audit=0, original_files_modified=False, completed_lanes=rows_out,
        completed_observation_point_evaluations=400, candidate8=backend8,
        candidate9=dict(classification='initialization_protocol_timeout', ready_reached=False,
            completed_observation_points=0, initialization_coefficient_rows=len(trace), completed_public_calls=28,
            bootstrap_call0_count=1, subsequent_completed_warmup_calls=27, unique_input_points_seen=27,
            incomplete_coefficient_pairs=0, status_counts_per_public_call=dict(warm_counts),
            actual_backend_precision_per_public_call={'9:DP': warm_counts['219'], '9:QP': warm_counts['323']},
            warmup_sha256=WARM_SHA, accuracy_claim='Unqualified partial initialization trace; no completed 100-point TIRust observation.',
            requested_timeout_seconds_session_reported=1800, timeout_numeric_value_recorded_in_original_attempt=False,
            timeout_evidence_limit='The specification omits the CLI timeout. Failure records prove a protocol timeout without encoding its duration; 1800 seconds is session-reported command provenance, not a timestamp inference.',
            next_call_completion_known=False),
        physical_inputs=dict(points_sha256=next(iter(physical))[0], param_card_sha256=next(iter(physical))[1]),
        retained_protocols=spec['protocols'], identities=identities,
        evidence_limits=[
            'Only baseline1 and candidate1/6/7 are completed controller observations. Candidate6/7 pass the 3e-6 comparator but miss the ordinary 1e-8 target against CutTools.',
            'Backend8 remains failed. Its 100 raw points support only the separately bound forensic; they do not supply missing controller completion or post-measurement verification evidence.',
            'Backend9 contributes partial initialization only, including a separate bootstrap call0 on point1; its values are not a completed accuracy observation.',
            'DP/QP counts describe final public-call statuses, not internal reduction attempts, fallback calls, or retries.',
            'Native driver bytes are reconstructed from the frozen template. Retained parser/protocol identities and source inventories are hashed without new native execution.',
            'This invocation rebuilt the worker against earlier numerical objects, not a fresh full numerical rebuild. Historical executable/library hashes are retained records, not revalidated mutable binaries.',
            'Reduction policies are checked against retained effective-settings records; physical parameter-card bytes and point bytes are separately hashed.'],
        retained_attempt_file_count=len(original_files), files_base='docs/benchmarkfolder', files=files, all_audit_checks_pass=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Create the index exclusively; refuse to overwrite')
    args = parser.parse_args()
    require(Path.cwd().resolve() == ROOT, 'Run from the repository root')
    if not args.write:
        expected = load(OUTPUT)
        for name, value in expected['files'].items():
            require(record(REPORT / name) == value, 'Indexed evidence changed: ' + name)
    result = audit()
    if args.write:
        with OUTPUT.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    else:
        require(result == expected, 'Audit differs from frozen index')
    print(json.dumps(dict(checks_pass=True, completed_points=400, backend8_controller_status='failed',
        backend8_retained_raw_points=100, tirust_completed_observation_points=0, tirust_partial_warmup_calls=28,
        retained_attempt_files=result['retained_attempt_file_count'], indexed_files=len(result['files']),
        index_sha256=record(OUTPUT)['sha256'])))
