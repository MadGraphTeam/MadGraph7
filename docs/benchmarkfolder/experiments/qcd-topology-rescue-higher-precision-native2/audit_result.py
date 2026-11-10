#!/usr/bin/env python3
"""Recheck a completed runner2 record offline; never load a native library."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd().resolve()
BASE = ROOT / 'docs/benchmarkfolder/experiments'
RUNNER = BASE / 'qcd-topology-rescue-higher-precision-runner2'
RUNNER_SHA = '69f76225a439dd325c4c08369a46ae5654102f1a95c19d6c3e7ca83682f0a3f4'
FAILED = BASE / 'qcd-topology-rescue-higher-precision-native'
FAILED_SHA = 'd5b67551d3c9353dbd472a0d272710b63bf220bdcab6e186944713ed07e9b14c'
COMPARISON = BASE / 'qcd-topology-rescue-higher-precision-comparison'
COMPARISON_SHA = '9102583bbe668322ed284a2970c24e538a5f737a485b57e0484fc0d78acc8b9c'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inventory(directory, expected, name='evidence-index.json'):
    require(sha(directory / name) == expected, 'Wrong frozen dependency index.')
    entries = read(directory / name)['files']
    paths = list(directory.rglob('*'))
    require(not any(p.is_symlink() for p in paths), 'Linked dependency path.')
    require({str(p.relative_to(directory)) for p in paths if p.is_file()} == set(entries) | {name},
            'Dependency inventory is not exact.')
    for item, entry in entries.items():
        require(not Path(item).is_absolute() and '..' not in Path(item).parts, 'Unsafe inventory member.')
        p = directory / item
        require(p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], 'Changed member: ' + item)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def rows(path):
    return [line.strip() for line in path.read_text().splitlines()
            if line.strip().startswith(('AMPLITUDE_', 'BORN_RESCUE_COUNTS', 'LOOP_NORMALIZATION'))]


def compute(directory, live=False):
    # Authenticate every imported helper package before importing code.
    inventory(RUNNER, RUNNER_SHA, 'source-index.json')
    inventory(FAILED, FAILED_SHA)
    inventory(COMPARISON, COMPARISON_SHA)
    attempt = directory / 'attempt1'
    record = read(attempt / 'execution.json')
    require(record['complete'] and record['native_executed'], 'Native protocol capture is incomplete.')
    require(not record['compiler_executed'] and not record['physics_accepted'] and
            not record['formal_performance_eligible'], 'Scope exceeds this diagnostic.')
    require(record['source_index_sha256'] == RUNNER_SHA and
            record['prior_incomplete_attempt_index_sha256'] == FAILED_SHA, 'Wrong runner/control bindings.')
    require((record['cpu'], record['thread_count'], record['timeout_per_process_seconds'],
             record['automatic_retry_count'], record['input_bits'], record['working_bits'],
             record['guard_exponent']) == (11, 1, 180, 0, 113, 512, -392), 'Protocol changed.')
    require(record['returncode'] == 0 and record['native_process_wall_seconds'] <= 180, 'Native cap/exit failed.')
    require(record['native_launch_progress']['process_spawned'] and
            record['native_launch_progress']['process_reaped'], 'Native child not owned/reaped.')
    require(record['ldd_launch_progress']['process_spawned'] and
            record['ldd_launch_progress']['process_reaped'], 'Inspection child not owned/reaped.')
    pins = {}
    for prefix in ('protected_inputs', 'resource_sha256', 'runtime_sha256', 'ldd_tool_runtime_sha256'):
        before, after = record[prefix + '_before'], record[prefix + '_after']
        require(before == after and record[prefix + '_unchanged'], 'Input/runtime drift: ' + prefix)
        for name, digest in before.items():
            require(name not in pins or pins[name] == digest, 'Conflicting input hash.')
            pins[name] = digest
    if live:
        for name, digest in pins.items():
            require(sha(ROOT / name) == digest, 'Changed live pin: ' + name)
    for name, prefix in (('observed.log', 'log'), ('trace.dat', 'trace'), ('ldd.txt', 'ldd'), ('process-maps.txt', 'maps')):
        require(sha(attempt / name) == record[prefix + '_sha256'] and
                (attempt / name).stat().st_size == record[prefix + '_bytes'], 'Changed output: ' + name)
    for name, entry in record['mapping_snapshots'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe snapshot path.')
        require(sha(attempt / name) == entry['sha256'] and
                (attempt / name).stat().st_size == entry['bytes'], 'Changed mapping snapshot.')
    mapping = record['actual_process_mapping']
    require(mapping['cpu_allowed_list'] == '11', 'Wrong observed leader affinity.')
    require(mapping['expected_runtime_sha256'] == mapping['mapped_files_sha256_after_exit'] ==
            record['runtime_sha256_before'], 'Observed runtime closure mismatch.')
    require(set(mapping['mapped_elf_paths']) == set(record['runtime_sha256_before']) | {mapping['executable']},
            'Unexpected mapped ELF set.')
    require(mapping['selected_library'] in record['runtime_sha256_before'], 'No selected library mapping.')
    require(record['runtime_sha256_before'][mapping['selected_library']] ==
            'e8b65f86b78f77d8090e7c406274da32920d89279ee4d5d3ef40e518fb05afdc', 'Wrong private512 library.')
    raw = mapping['maps_text'].encode()
    require(raw and hashlib.sha256(raw).hexdigest() == mapping['maps_sha256'], 'Missing actual mapping text.')
    require((attempt / 'process-maps.snapshots/complete.txt').read_bytes() == raw,
            'Complete actual mapping snapshot differs.')
    parsed = set()
    for line in raw.decode().splitlines():
        columns = line.split(None, 5)
        if len(columns) == 6 and columns[5].startswith('/'):
            require(not columns[5].endswith(' (deleted)'), 'Deleted mapped file.')
            parsed.add(str(Path(columns[5]).resolve()))
    require(parsed == set(mapping['mapped_paths']), 'Raw and recorded mapping paths differ.')
    require(set(mapping['mapped_elf_paths']) <= parsed, 'Recorded ELF path absent from raw maps.')
    require(rows(attempt / 'observed.log') == rows(FAILED / 'attempt1/observed.log'), 'Prior selected rows differ.')
    require((attempt / 'trace.dat').read_bytes() == (FAILED / 'attempt1/trace.dat').read_bytes(),
            'Prior full raw trace differs.')
    require(record['incomplete_attempt_all_numeric_rows_exact'] and record['incomplete_attempt_full_trace_bytes_exact'],
            'Prior parity certificate missing.')
    for key in ('ordinary_numeric_rows_exact', 'ordinary_repeated_rows_exact',
                'ordinary_raw_words_exact', 'ordinary_repeated_raw_words_exact', 'trace_structure_valid'):
        require(record[key], 'Historical ordinary parity failed: ' + key)
    require(record['health'] == [[1, 0, 0], [2, 0, 0], [3, 0, 0]], 'Trace health failed.')
    admissions = sorted((directory / 'admissions').glob('*.json'))
    admitted = [p for p in admissions if read(p).get('admitted')]
    require(len(admitted) == 1 and str(admitted[0].relative_to(ROOT)) == record['admission_path'],
            'Wrong one-shot admitted record.')
    admission = read(admitted[0])
    require(admission['attempt_created'] and admission['native_executed'] and
            admission['controller_index_sha256'] == RUNNER_SHA, 'Incomplete admission record.')
    comparison = module('frozen_qcd512_compare2', COMPARISON / 'compare2.py').compute(attempt, False)
    require(comparison == read(directory / 'comparison.json'), 'Offline group comparison differs.')
    summary = module('frozen_qcd512_summary', COMPARISON / 'summarize.py').compute(directory / 'comparison.json')
    require(summary == read(directory / 'comparison-summary.json'), 'Offline magnitude summary differs.')
    require(comparison['runtime_capture_complete'] and not comparison['provisional_raw_comparison'], 'Wrong capture scope.')
    require(comparison['all_common_inputs_and_scale_semantics_exact'] and
            comparison['same_original_and_effective_coordinate_words'], 'Input equivalence failed.')
    require(comparison['group_calls'] == 232 and comparison['complex_contracted_laurent_slots'] == 696,
            'Incomplete comparison coverage.')
    require(summary['all_696_complex_slots_included'] and summary['discarded_component_count'] == 0,
            'Some components were discarded.')
    return {'schema': 1, 'verification': 'pass', 'runtime_capture_complete': True,
            'native_executed_by_verifier': False, 'physics_campaign_accepted': False,
            'production_promoted': False, 'formal_performance_eligible': False,
            'live_hashes_checked': len(pins) if live else 0, 'recorded_unchanged_hashes': len(pins),
            'execution_sha256': sha(attempt / 'execution.json'), 'actual_mapping_sha256': mapping['maps_sha256'],
            'native_wall_seconds': record['native_process_wall_seconds'],
            'prior_incomplete_attempt_trace_and_rows_exact': True, 'ordinary_parity_exact': True,
            'group_calls': 232, 'complex_contracted_slots': 696,
            'flagged_relative_slots': len(summary['flagged_slots']),
            'observation_limit': 'One successful mapping and leader-affinity snapshot, not continuous monitoring.',
            'scientific_limit': 'Private topology/C-Born and512 arithmetic/admission experiment; no general physics or production qualification.'}
