"""Portable, text-only reanalysis of the one guarded physical replay.

The caller verifies the capture manifest before supplying UTF-8 bytes keyed by
repository-relative paths. Only archived Python parsers and pure comparison
helpers are called. Native files and their historical absolute paths are never
opened, and controller verify/bind/run functions are never called.
"""
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tempfile


SOURCE = 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-physical-proposal-2'
OLD_SOURCE = 'docs/benchmarkfolder/experiments/adaptive-qp-checked-trap-proposal-1'
RUN = '.tirust-build/cayley-guarded-adaptive-physical-proposal-2/run1'
BENCHMARK = 'tools/tirust_benchmark.py'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def relative_name(name):
    require(type(name) is str and bool(name), 'Invalid payload name')
    path = PurePosixPath(name)
    require(not path.is_absolute() and '..' not in path.parts and
            str(path) == name and name != '.', 'Noncanonical payload name: ' + name)
    return name


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def first_named_flags(control):
    """Retain named observations; do not infer names from opaque journal bits."""
    first = {}
    for state in control['states']:
        for name, value in state['flags'].items():
            if value['set'] and name not in first:
                first[name] = {'phase': state['phase'], **value}
    return first


def inspect_worker(base, name, row, parser, controls, assessment):
    """Parse one archived worker, preserving missing and interrupted journals."""
    directory = base / RUN / name
    code = row.get('exit_code', row.get('progress', {}).get('returncode'))
    result = {'kind': row['kind'], 'mode': row.get('mode'),
              'policy': row['policy'], 'exit_code': code,
              'retained_process_failure': row.get('process_failure'),
              'retained_offline_assessment_failure': row.get('offline_assessment_failure'),
              'native_or_loader_executed_by_analyzer': False}
    stdout = directory / 'stdout.log'
    result['W03'] = ([{'line': i, 'text': line}
                      for i, line in enumerate(stdout.read_text().splitlines(), 1)
                      if '##W03' in line] if stdout.exists() else [])
    result['stdout_present'] = stdout.exists()
    capture = None
    try:
        named = controls.parse(directory / 'checked-control.log', allow_partial=code != 0)
        result['named_control'] = named
        result['control_assessment'] = controls.assess(named, exit_code=code)
        result['first_named_set_flags'] = first_named_flags(named)
        if row['kind'] != 'physical':
            result['journal_disposition'] = 'not applicable to deliberate fault control'
            return result, capture
        journal = directory / 'mass-observer.hex'
        if not journal.exists():
            missing = RUN + '/' + name + '/mass-observer.hex'
            result['missing_journal_path'] = missing
            failure = row.get('offline_assessment_failure')
            result['retained_missing_journal_failure_matches'] = (
                failure.get('type') == 'FileNotFoundError' and missing in failure.get('message', '')
                if failure is not None else None)
            result['journal_disposition'] = (
                'absent after interrupted worker; no observed journal or physical qualification'
                if code != 0 else 'missing required journal after zero exit')
            result['physical_outputs_qualified'] = False
            return result, capture
        capture = parser.parse(journal, allow_partial=code != 0)
        result['journal_disposition'] = 'complete' if capture['complete'] else 'interrupted prefix; unqualified'
        result['journal_summary'] = parser.summarize(capture)
        if not stdout.exists():
            result['numerical_assessment_disposition'] = 'stdout absent; omission scan unavailable'
            result['physical_outputs_qualified'] = False
            return result, capture
        if row['mode'] == 'checked':
            result['numerical_assessment'] = parser.qualify_checked(
                capture, named['observed'], omissions=len(result['W03']))
        else:
            result['numerical_assessment'] = assessment.qualify_fixed(
                capture, named['observed'], parser, omissions=len(result['W03']))
    except Exception as error:
        result['portable_parse_failure'] = {'type': type(error).__name__, 'message': str(error)}
        result['physical_outputs_qualified'] = False
    return result, capture


def compare_captures(spec, captures, workers, references, parser, old, assessment, benchmark):
    """Reproduce the frozen controller's pure comparisons and observations."""
    comparisons = {}
    pending = {}
    for mode in ('checked', 'forced-qp'):
        left, right = mode + '-masked', mode + '-trapped'
        observations = {'numerical_to_closed_adaptive': {'masked': False, 'trapped': False},
                        'host_stability_acceptance': {'masked': False, 'trapped': False},
                        'observed_QP_path_coverage': {'masked': False, 'trapped': False},
                        'IEEE_policy_worker_completion': {'masked': False, 'trapped': False},
                        'policy_numerical_agreement': False}
        for policy in ('masked', 'trapped'):
            row = workers.get(mode + '-' + policy, {})
            observed = row.get('numerical_assessment', {})
            observations['host_stability_acceptance'][policy] = bool(
                mode == 'checked' and observed.get('checked_outputs_qualified', False))
            observations['observed_QP_path_coverage'][policy] = bool(
                observed.get('qp_coverage_observed', False) if mode == 'checked'
                else observed.get('fixed_qp_outputs_qualified', False))
            observations['IEEE_policy_worker_completion'][policy] = row.get(
                'control_assessment', {}).get('physical_worker_completed_with_requested_IEEE_policy', False)
        if left in captures and right in captures:
            inputs = parser.compare_inputs_and_born(captures[left], captures[right])
            numerical = assessment.compare_amplitudes(captures[left], captures[right], parser, benchmark)
            comparisons[mode + '_policy_comparison'] = {
                'inputs_and_born': inputs, 'numerical_outputs': numerical,
                'path_changes': assessment.classify_paths(captures[left], captures[right], parser),
                'strict_bitwise_diagnostic': old.compare_policy_outputs(captures[left], captures[right])}
            observations['policy_numerical_agreement'] = bool(
                inputs['inputs_and_born_exact'] and numerical['passed'])
        for reference in spec['comparators']:
            if reference['mode'] != mode:
                continue
            previous = references[reference['label']]
            for policy in ('masked', 'trapped'):
                current = mode + '-' + policy
                if current not in captures:
                    continue
                key = reference['label'] + '__' + current
                inputs = parser.compare_inputs_and_born(previous, captures[current])
                numerical = assessment.compare_amplitudes(previous, captures[current], parser, benchmark)
                comparisons[key] = {
                    'inputs_and_born': inputs, 'numerical_outputs': numerical,
                    'path_changes': assessment.classify_paths(previous, captures[current], parser),
                    'previously_served_bitwise_diagnostic': assessment.previously_served_preserved(
                        previous, captures[current], parser),
                    'strict_bitwise_diagnostic': old.compare_policy_outputs(previous, captures[current]),
                    'numerical_agreement_reference': reference['numerical_agreement_reference']}
                if reference['numerical_agreement_reference']:
                    observations['numerical_to_closed_adaptive'][policy] = bool(
                        inputs['inputs_and_born_exact'] and numerical['passed'])
        pending[mode] = observations
    return comparisons, pending


def composite_gates(gates):
    checked = gates.get('checked', {})
    fixed = gates.get('forced-qp', {})
    return {
        'checked_trap_gate_qualified': bool(
            checked.get('numerical_agreement', False) and
            checked.get('host_stability_acceptance', False) and
            checked.get('trap_completion', False) and
            checked.get('masked_trapped_numerical_agreement', False)),
        'forced_qp_trap_gate_qualified': bool(
            fixed.get('numerical_agreement', False) and
            fixed.get('QP_path_coverage', False) and
            fixed.get('trap_completion', False) and
            fixed.get('masked_trapped_numerical_agreement', False))}


def saved_equal(actual, saved, label):
    require(canonical(actual) == canonical(saved), 'Retained derived output differs: ' + label)


def published_transport(record, transport):
    """Final publication may rescind success after every child was reaped."""
    if record['execution_complete']:
        require(transport and 'closure_failure' not in record and
                record.get('protected_inputs_after') == 'unchanged',
                'Successful final publication lacks transport/closure')
    return bool(transport and record['execution_complete'] and 'closure_failure' not in record)


def reanalyze(blobs):
    """Reparse a terminal text capture without opening historical native files.

    blobs maps canonical repository-relative names to already verified UTF-8
    bytes. Missing worker journals are evidence outcomes, while missing frozen
    source/comparator text is a malformed capture and raises ValueError.
    """
    require(type(blobs) is dict and bool(blobs), 'Empty or invalid text capture')
    for name, data in blobs.items():
        relative_name(name)
        require(type(data) is bytes, 'Payload is not bytes: ' + name)
        data.decode('utf-8')
    required = [SOURCE + '/prepared.json', SOURCE + '/controller.py', SOURCE + '/assessment.py',
                OLD_SOURCE + '/controller.py', OLD_SOURCE + '/parser.py',
                OLD_SOURCE + '/control_log.py', BENCHMARK, RUN + '/execution.json']
    require(all(name in blobs for name in required), 'Missing frozen source or final execution text')
    record = json.loads(blobs[RUN + '/execution.json'])
    require(type(record.get('execution_complete')) is bool and 'elapsed_seconds' in record,
            'Execution receipt is not a terminal publication')
    spec = json.loads(blobs[SOURCE + '/prepared.json'])
    source_hashes = {name: hashlib.sha256(blobs[name]).hexdigest() for name in required[:-1]}
    old_path = list(sys.path)
    old_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with tempfile.TemporaryDirectory(prefix='guarded-physical-offline-') as directory:
            base = Path(directory) / 'repository'
            for name, data in blobs.items():
                target = base / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            # Loading these modules defines functions and constants only. The
            # imported controllers' operational entrypoints are never called.
            load('retained_guarded_physical_controller', base / SOURCE / 'controller.py')
            old = load('retained_guarded_old_controller', base / OLD_SOURCE / 'controller.py')
            parser = load('retained_guarded_journal', base / OLD_SOURCE / 'parser.py')
            controls = load('retained_guarded_controls', base / OLD_SOURCE / 'control_log.py')
            assessment = load('retained_guarded_assessment', base / SOURCE / 'assessment.py')
            benchmark = load('retained_guarded_regression', base / BENCHMARK)
            workers = {}
            captures = {}
            names = ['fault-' + n for n in spec['fault_controls']]
            names += [mode + '-' + policy for mode in ('checked', 'forced-qp')
                      for policy in ('masked', 'trapped')]
            require(set(record['workers']) <= set(names), 'Unexpected retained worker')
            for name in names:
                if name not in record['workers']:
                    workers[name] = {'disposition': 'not launched; no physical qualification'}
                    continue
                row = record['workers'][name]
                result, capture = inspect_worker(base, name, row, parser, controls, assessment)
                workers[name] = result
                if ('missing_journal_path' in result and
                        row.get('offline_assessment_failure', {}).get('type') == 'FileNotFoundError'):
                    require(result['retained_missing_journal_failure_matches'],
                            'Retained missing-journal error points elsewhere: ' + name)
                if capture is not None:
                    captures[name] = capture
                for key in ('control_assessment', 'numerical_assessment'):
                    if key in row:
                        require(key in result, 'Retained assessment could not be replayed: ' + name + '/' + key)
                        saved_equal(result[key], row[key], name + '/' + key)
                    elif key in result:
                        # An outer deadline may stop the controller before it
                        # parses already written text. Retain any new offline
                        # derivation without imputing an executed acceptance.
                        result.setdefault('additional_offline_derivations', []).append(key)
                prefix = RUN + '/' + name + '/'
                for key, filename in [('named_control', 'named-control.json'),
                                      ('journal_summary', 'journal-summary.json')]:
                    if prefix + filename in blobs:
                        require(key in result, 'Derived artifact without parsed input: ' + prefix + filename)
                        saved_equal(result[key], json.loads(blobs[prefix + filename]), prefix + filename)
                if 'omissions' in row:
                    require(result['stdout_present'], 'Retained omission scan has no stdout text: ' + name)
                    require(row['omissions'] == len(result['W03']), 'Retained W03 count differs: ' + name)
                if 'native_output_closure' in row:
                    closure = old.output_closure(row['inputs_before'], row['files_after'],
                                                 row['kind'], row.get('exit_code', row['progress'].get('returncode')))
                    saved_equal(closure, row['native_output_closure'], name + '/native_output_closure')
                else:
                    require(record['execution_complete'] is False,
                            'Complete execution lacks worker output closure: ' + name)
                    result['output_closure_disposition'] = 'not published before retained execution failure'
            references = {}
            for reference in spec['comparators']:
                name = relative_name(reference['journal'])
                require(name in blobs, 'Missing closed comparator text: ' + name)
                require(hashlib.sha256(blobs[name]).hexdigest() == reference['journal_sha256'],
                        'Closed comparator identity differs: ' + name)
                references[reference['label']] = parser.parse(base / name)
            comparisons, pending = compare_captures(
                spec, captures, workers, references, parser, old, assessment, benchmark)
            comparison_path = RUN + '/comparisons.json'
            if comparison_path in blobs:
                saved_equal(comparisons, json.loads(blobs[comparison_path]), 'comparisons.json')
            else:
                require(record['execution_complete'] is False,
                        'Complete execution is missing comparison publication')
            if 'gate_observations_before_transport_closure' in record:
                saved_equal(pending, record['gate_observations_before_transport_closure'], 'gate observations')
            controls_valid = all(workers.get('fault-' + name, {}).get('control_assessment', {}).get(
                'fault_control_demonstrated', False) for name in spec['fault_controls'])
            if 'all_trap_controls_demonstrated' in record:
                require(controls_valid == record['all_trap_controls_demonstrated'], 'Fault-control aggregate differs')
            transport_error = None
            try:
                transport = old.transport_complete(record['workers'], top_failure='failure' in record)
            except (KeyError, TypeError) as error:
                require(record['execution_complete'] is False,
                        'Successful execution has incomplete transport metadata')
                transport = False
                transport_error = {'type': type(error).__name__, 'message': str(error)}
            # Final publication failure may rescind an otherwise complete
            # transport. Never reconstruct success from healthy rows alone.
            valid = published_transport(record, transport)
            gates = assessment.separate_gates(pending, valid, controls_valid)
            saved_equal(gates, record['qualified_gates'], 'qualified_gates')
            composite = composite_gates(gates)
            for name, value in composite.items():
                require(record[name] is value, 'Composite gate differs: ' + name)
            result = {'schema': 'guarded-adaptive-physical-offline-analysis-v1',
                      'workers': workers, 'comparisons': comparisons,
                      'gate_observations': pending, 'qualified_gates': gates, **composite,
                      'retained_execution_complete': record['execution_complete'],
                      'transport_metadata_replayed': transport,
                      'incomplete_transport_metadata': transport_error,
                      'final_publication_qualified': valid,
                      'fault_controls_demonstrated': controls_valid,
                      'retained_failure': record.get('failure'),
                      'retained_closure_failure': record.get('closure_failure'),
                      'source_sha256': source_hashes,
                      'independent_physical_truth_qualified': False,
                      'per_call_working_precision_qualified': False,
                      'performance_qualified': False,
                      'historical_wall_time_remeasured': False,
                      'native_or_loader_executed_by_analyzer': False,
                      'historical_native_files_opened': False}
            return canonical(result)
    finally:
        sys.path[:] = old_path
        sys.dont_write_bytecode = old_bytecode
