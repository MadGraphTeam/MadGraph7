#!/usr/bin/env python3
"""Verify retained common-allocation evidence without builds or native runs.

Default: verify the existing portable index and reparse the numerical records.
--write: create that index exclusively, after all evidence and README are frozen.
--check-static: inspect completed builds/stubs/native parity only; do not read
the timing directory or create an index.  Absolute paths inside old metadata
are treated as recorded identities, never as instructions to open live files.
"""
import argparse
import ast
import difflib
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import struct
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
FORTRAN = HERE.parent / 'fortran'
PROTOCOL = FORTRAN / 'protocol'
FROZEN = ROOT / 'docs/benchmarkfolder/runs/qcd_n1_uux_z/tirust-first-attempt-3'
INDEX = HERE / 'evidence-index.json'
SCHEMA = 'madloop-common-allocation-evidence-v1'
DSO = '1c3688011fb0b423142ab4ac4c832f5b71d8e4409ef323087dfac5ceb2075cd7'
VARIANTS = ('tir-alloc', 'switch-guard', 'combined')
BACKENDS = (1, 6, 7, 8, 9)
LEGACY_BASELINE_INAPPLICABLE = frozenset((
    'tirustcachemaxbytes', 'tirustcomputefinite', 'tirustglobalcache',
    'useqpintegrandfortirust',
))
PROTOCOL_SHA256 = {
    'tirust_benchmark.py': '6f67119e5b3be36406fe9065915fd3e68a090fdf3bcd375416ad3370b9ccdc2d',
    'tirust_benchmark_worker.py': '33c7b986798b468bdd0a5649283ce1ab707d16edc7373e6c8930d134f22d01ee',
    'tirust_benchmark_worker.f': '02a5115406ec38dafb307bdf82879e26f363a5957cfe794121520cfe36b4d678',
    'tirust_benchmark_compiler.py': '86dd3a860a39a1421e035dcf6555575addcbed0c99c80211f9f8b348a3e75e46',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(path.is_absolute() and path.is_relative_to(ROOT) and '..' not in path.parts and
            path.resolve().is_relative_to(ROOT), 'Path escapes evidence repository: ' + str(path))
    require(not any(parent.is_symlink() for parent in path.parents if parent.is_relative_to(ROOT)),
            'Symlink ancestor in evidence path: ' + str(path))
    require(path.is_file() and not path.is_symlink(), 'Missing/nonregular/symlink file: ' + str(path))
    return path.read_bytes()


def digest(path):
    return hashlib.sha256(read(path)).hexdigest()


def document(path):
    return json.loads(read(path))


def same(left, right, label):
    # JSON canonicalization also normalizes tuples returned by old helpers.
    require(json.dumps(left, sort_keys=True, allow_nan=False) ==
            json.dumps(right, sort_keys=True, allow_nan=False), label)


def relative(path, base):
    value = path.relative_to(base).as_posix()
    require(value and '..' not in Path(value).parts and not Path(value).is_absolute(),
            'Unsafe inventory path: ' + value)
    return value


def identity(path, base):
    contents = read(path)
    return dict(path=relative(path, base), bytes=len(contents),
                sha256=hashlib.sha256(contents).hexdigest())


def frozen_helpers():
    """Extract only retained pure parsing/statistics functions, not controllers."""
    for name in ('tirust_benchmark.py', 'tirust_benchmark_worker.py'):
        require(digest(PROTOCOL / name) == PROTOCOL_SHA256[name],
                'Pinned parser source changed before extraction: ' + name)
    worker_path = PROTOCOL / 'tirust_benchmark_worker.py'
    tree = ast.parse(read(worker_path), filename=str(worker_path))
    names = {'OUTPUTS', 'RAW_OUTPUTS', 'BACKEND_CODES', '_T975',
             'MeasurementError', 'raw_outputs', 'validate_status',
             'parse_outputs', 'paired_statistics'}
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets):
            selected.append(node)
    namespace = dict(Path=Path, math=math, statistics=statistics)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(worker_path), 'exec'), namespace)
    campaign_path = PROTOCOL / 'tirust_benchmark.py'
    tree = ast.parse(read(campaign_path), filename=str(campaign_path))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == 'compare_rows']
    require(len(functions) == 1, 'Missing unique retained compare_rows')
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(campaign_path), 'exec'), namespace)
    return namespace


def audit_builds(candidate):
    require(digest(HERE / 'transform.py') == document(HERE / 'stub-parity.json')['transformer_sha256'],
            'Historical transform changed before importing')
    spec = importlib.util.spec_from_file_location('common_allocation_transform_audit', HERE / 'transform.py')
    transform = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(transform)
    records = {}
    source_key = 'SubProcesses/P0_uux_z/TIR_interface.f'
    source_inventory = candidate['compilation']['sources']
    require(len(source_inventory) == 111 and source_key in source_inventory,
            'Unexpected frozen numerical source inventory')
    attribution = document(FORTRAN / 'preparation.json')
    for variant in VARIANTS:
        folder = HERE / (variant + '-build')
        meta = document(folder / 'build-metadata.json')
        require(meta['built'] is True and meta['performance_eligible'] is False,
                variant + ': not a completed isolated diagnostic build')
        require(meta['variant'] == variant, 'Build variant mismatch')
        original = read(folder / 'original-TIR_interface.f').decode()
        modified = read(folder / 'variant-TIR_interface.f').decode()
        same(transform.transform(original, variant), modified, variant + ': transform/source mismatch')
        same(''.join(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                        fromfile='original', tofile=variant)),
             read(folder / 'interface.diff').decode(), variant + ': retained diff mismatch')
        require(meta['original_source_sha256'] == digest(folder / 'original-TIR_interface.f') ==
                source_inventory[source_key] == attribution['source_sha256']['TIR_interface.f'],
                variant + ': original source identity')
        require(meta['source_sha256'] == digest(folder / 'variant-TIR_interface.f'),
                variant + ': replacement source identity')
        require(meta['prepare_sha256'] == digest(HERE / 'prepare_variant.py') and
                meta['transform_sha256'] == digest(HERE / 'transform.py'),
                variant + ': preparation/transform identity')
        require(meta['original_executable_sha256'] == meta['reconstructed_original_executable_sha256'] ==
                candidate['compilation']['executable_sha256'], variant + ': untouched relink identity')
        require(meta['driver_sha256'] == candidate['compilation']['driver_sha256'],
                variant + ': compiled worker identity')
        same(meta['original_compile'], meta['candidate_compile'], variant + ': compile command changed')
        compiler = meta['original_compile'][0]
        logged = [shlex.split(line) for line in read(folder / 'compile-link.log').decode().splitlines()
                  if line.startswith(compiler + ' ')]
        same(logged, [meta['link'], meta['candidate_compile'], meta['link']],
             variant + ': recorded relink/compile/relink commands')
        expanded = [shlex.split(line) for line in read(folder / 'original-expanded-recipes.log').decode().splitlines()
                    if line.startswith(compiler + ' ')]
        same([row for row in expanded if '-c' in row and 'TIR_interface.f' in row],
             [meta['original_compile']], variant + ': expanded original compile recipe')
        same([row for row in expanded if '-o' in row and row[row.index('-o') + 1] == 'check'],
             [meta['link']], variant + ': expanded original link recipe')
        require(not any(arg.startswith('-O') for arg in meta['original_compile']),
                variant + ': unexpected optimization flag')
        recipe = candidate['compilation']['effective_compiler_flags']['scopes']['process']['recipes']['TIR_interface.f']
        same(meta['frozen_recipe_verified'], recipe, variant + ': frozen compiler recipe')
        same(meta['compiler_environment'], candidate['compilation']['effective_compiler_flags']['environment'],
             variant + ': injected compiler environment')
        require(meta['compiler_version'].strip() == recipe['compiler'][0]['version'].strip(),
                variant + ': recorded compiler version')
        require(set(meta['objects_before']) == set(meta['objects_after']) and
                len(meta['objects_before']) == 19, variant + ': object inventory')
        changed = sorted(key for key in meta['objects_before']
                         if meta['objects_before'][key] != meta['objects_after'][key])
        same(changed, ['TIR_interface.o'], variant + ': unexpected changed object')
        same(meta['changed_objects'], changed, variant + ': reported changed objects')
        source, output = meta['source'].rstrip('/'), meta['output'].rstrip('/')
        archives = {(source + path[len(output):] if path.startswith(output + '/') else path): value
                    for path, value in meta['static_archive_pins'].items()}
        same(archives, candidate['runtime']['archives'], variant + ': recorded static dependencies')
        records[variant] = meta
    return records, dict(builds=3, changed_objects=['TIR_interface.o'], objects_per_build=19,
        static_archives_per_build=len(candidate['runtime']['archives']),
        frozen_source_inventory_entries=111,
        frozen_source_inventory_sha256=hashlib.sha256(json.dumps(source_inventory,
            sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        source_inventory_scope='Bound to the retained frozen input manifest; the three build records do not contain a new execution-time snapshot of every original source/include file.',
        live_objects_or_compilers_reopened=False,
        executable_sha256={name: meta['executable_sha256'] for name, meta in records.items()})


def audit_stub():
    report = document(HERE / 'stub-parity.json')
    for key, name in [('program_sha256', 'check-layout-used.py'),
                      ('template_sha256', 'template-used.inc'),
                      ('driver_sha256', 'layout_driver.f'),
                      ('transformer_sha256', 'transform.py')]:
        require(report[key] == digest(HERE / name), 'Historical stub binding: ' + name)
    require(report['outcome'] == 'pass', 'Stub corpus did not pass')
    rows = report['comparisons']
    configurations = ('cache2-interference', 'cache2-amplitude', 'uncached-amplitude')
    expected = {(config, build, variant) for config in configurations
                for build in ('O0', 'O3', 'O3-static') for variant in ('original', *VARIANTS)}
    require(len(rows) == 36 and {(r['configuration'], r['build'], r['variant']) for r in rows} == expected,
            'Incomplete/duplicate stub build matrix')
    for row in rows:
        require(row['outcome'] == 'pass' and row['tir_records'] == 645 and row['switch_records'] == 432,
                'Unexpected stub case count/outcome')
        original = next(r for r in rows if r['configuration'] == row['configuration'] and
                        r['build'] == row['build'] and r['variant'] == 'original')
        require(row['output_sha256'] == original['output_sha256'] and
                row['output_bytes'] == original['output_bytes'], 'Stub output parity hash mismatch')
    controls = {r['name']: r for r in report['mutation_controls']}
    require(set(controls) == {'missing-cleanup', 'wrong-switch-guard'}, 'Missing stub mutation control')
    require(controls['missing-cleanup']['returncode'] == 2 and
            any('already allocated' in line for line in controls['missing-cleanup']['diagnostic']),
            'Missing cleanup negative-control evidence')
    require(controls['wrong-switch-guard']['returncode'] == 71 and
            any('input mutation/canaries' in line for line in controls['wrong-switch-guard']['diagnostic']),
            'Wrong guard negative-control evidence')
    unsupported = report['unsupported_configurations']
    require(len(unsupported) == 1 and unsupported[0]['configuration'] == 'cache0-interference' and
            unsupported[0]['returncode'] != 0 and 'shall not be zero' in unsupported[0]['diagnostic'],
            'Missing original zero-cache compile limitation')
    return dict(builds_passed=36, original_builds=9, variant_comparisons=27,
                tir_cases_per_build=645, switch_cases_per_build=432,
                mutation_controls_detected=2, historical_zero_cache_compile_failure=True,
                raw_output_scope='The portable namespace retains the historical report and raw-output hashes, not the ignored full stub stdout/executables.',
                replay_runner_is_historical_execution=False)


def rows_binary(rows, coefficients_only=False):
    if coefficients_only:
        return b''.join(struct.pack('!d', value) for row in rows for value in row['coefficients'])
    return b''.join(struct.pack('!iiiddddd', row['point'], row['order'], row['status'],
                                *row['coefficients'], row['accuracy']) for row in rows)


def runtime_expected(frozen, library):
    result = dict(frozen['runtime']['libraries'])
    old = [path for path in result if Path(path).name == 'libmadloop_tirust.so']
    require(len(old) == 1, 'Expected one historical TIRust runtime library')
    del result[old[0]]
    result[library['path']] = library['sha256']
    return result


def audit_runtime(folder, expected, wrapped=False):
    recorded = document(folder / 'runtime-identity.json')
    if wrapped:
        same(recorded['expected'], expected, 'Recorded expected runtime: ' + str(folder))
        recorded = recorded['actual']
    same(recorded, expected, 'Recorded actual runtime: ' + str(folder))
    linked = set(re.findall(r'(?:=>\s*)?(/[^\s]+)', read(folder / 'ldd.txt').decode()))
    same(sorted(linked), sorted(expected), 'ldd/runtime path inventory: ' + str(folder))


def parse_outputs_checked(helper, folder, backend, points, loops):
    # The retained parser opens these paths itself; reject unsafe paths before
    # allowing it to read them, including in --check-static mode.
    for name in ('values.dat', 'timed_status.dat', 'dimensions.dat', 'timing.dat'):
        read(folder / name)
    return helper['parse_outputs'](folder, backend, points, loops)


def audit_settings_card(path, expected, allowed_missing=()):
    """Check retained parameter values without importing a live card reader."""
    lines = read(path).decode().splitlines()
    fields = {}
    for position, line in enumerate(lines):
        match = re.fullmatch(r'\s*#([A-Za-z0-9_]+)\s*', line)
        if not match:
            continue
        key = match[1].lower()
        require(key not in fields, 'Duplicate reduction-card setting: ' + key)
        following = next((item.strip().split('!', 1)[0].strip() for item in lines[position + 1:]
                          if item.strip() and not item.lstrip().startswith('!')), None)
        require(following is not None and not following.startswith('#'), 'Missing card value: ' + key)
        fields[key] = following
    omitted = {}
    for key, value in expected.items():
        if key.lower() not in fields and key.lower() in allowed_missing:
            omitted[key.lower()] = value
            continue
        require(key.lower() in fields, 'Missing frozen setting in card: ' + key)
        text = fields[key.lower()]
        if isinstance(value, bool):
            require(text.lower() in ('.true.', '.false.'), 'Invalid logical setting: ' + key)
            observed = text.lower() == '.true.'
        elif isinstance(value, int):
            observed = int(text)
        elif isinstance(value, float):
            observed = float(text.lower().replace('d', 'e'))
        elif isinstance(value, str):
            observed = text
        else:
            raise ValueError('Unexpected retained setting type: ' + key)
        same(observed, value, 'Frozen card value mismatch: ' + key)
    return omitted


def audit_native(candidate, builds, helper):
    folder = HERE / 'combined-native-parity-1'
    result = document(folder / 'results.json')
    require(result['passed'] is True and result['performance_eligible'] is False,
            'Native parity result eligibility/outcome')
    require(result['runner_sha256'] == digest(folder / 'runner-used.py') == digest(HERE / 'validate_backends.py'),
            'Native parity runner identity')
    require(result['frozen_inputs_sha256'] == digest(FROZEN / 'candidate-inputs.json'), 'Native frozen input identity')
    require(result['points_sha256'] == candidate['points_sha256'] ==
            digest(ROOT / 'docs/benchmarkfolder/points/qcd_n1_uux_z.dat'), 'Native point identity')
    require(result['fortran_build_metadata_sha256'] == digest(folder / 'fortran-build-metadata.json') ==
            digest(HERE / 'combined-build/build-metadata.json'), 'Native Fortran build identity')
    same(document(folder / 'fortran-build-metadata.json'), builds['combined'], 'Native build metadata contents')
    require(result['library']['sha256'] == DSO == document(folder / 'bridge-build-metadata.json')['library_sha256'],
            'Native selected DSO identity')
    expected_runtime = runtime_expected(candidate, result['library'])
    expected_order = [(backend, label) for backend in BACKENDS for label in ('original', 'variant')]
    same([(r['backend'], r['variant']) for r in result['runs']], expected_order, 'Native lane order/count')
    parsed = {}
    for record in result['runs']:
        backend, label = record['backend'], record['variant']
        lane = folder / f'{backend}-{label}'
        audit_runtime(lane, expected_runtime)
        observation = parse_outputs_checked(helper, lane, [backend], 100, 1)
        same(observation, record['observation'], 'Native raw/observation mismatch: ' + lane.name)
        same(document(lane / 'observation.json'), record, 'Native observation record: ' + lane.name)
        same(observation['backend_precision_counts'], {f'{backend}:DP': 100}, 'Unexpected native precision/backend')
        same(observation['answer_dimensions'], dict(allocated=1, loop_orders=1, born_orders=1), 'Native dimensions')
        require(record['driver_sha256'] == candidate['compilation']['driver_sha256'], 'Native worker identity')
        expected_exe = builds['combined']['original_executable_sha256' if label == 'original' else 'executable_sha256']
        require(record['executable_sha256'] == expected_exe, 'Native executable identity')
        settings = dict(candidate['settings'], mlreductionlib=str(backend))
        audit_settings_card(lane / 'MadLoopParams.dat', settings)
        parsed[(backend, label)] = observation
    comparisons, summary = [], []
    for backend in BACKENDS:
        original, modified = (parsed[(backend, label)] for label in ('original', 'variant'))
        for name in ('values.dat', 'timed_status.dat', 'dimensions.dat', 'MadLoopParams.dat'):
            require(read(folder / f'{backend}-original' / name) == read(folder / f'{backend}-variant' / name),
                    f'Native original/variant byte mismatch: {backend}/{name}')
        comparison = dict(backend=backend, points=100,
            rows_status_accuracy_bit_identical=rows_binary(original['rows']) == rows_binary(modified['rows']),
            status_histograms_identical=original['timed_status_counts'] == modified['timed_status_counts'],
            original_usage=original['backend_precision_counts'], variant_usage=modified['backend_precision_counts'])
        require(comparison['rows_status_accuracy_bit_identical'] and comparison['status_histograms_identical'],
                'Native full-row parity')
        comparisons.append(comparison)
        summary.append(dict(backend=backend, status_counts=original['timed_status_counts'],
                            original_dp=100, variant_dp=100, qp=0))
    same(comparisons, result['comparisons'], 'Native comparison summary')
    return dict(physical_original_variant_pairs=500, paired_saved_rows=1000,
                timed_dp_amplitudes=1000, timed_qp_amplitudes=0,
                complete_rows_and_status_files_byte_identical=True, backends=summary,
                recorded_dso_sha256=DSO, performance_eligible=False,
                runtime_scope='Retained ldd paths/hash records and build identities are checked; this verifier neither loads the DSO nor reopens the historical runtime files.')


def diagnostic_statistics(helper, left, right):
    result = helper['paired_statistics'](left, right, threshold=1.05, minimum_pairs=7)
    result['minimum_pair_count_satisfied'] = result.pop('acceptance_eligible')
    result['performance_eligible'] = False
    result['note'] = 'Diagnostic comparison, not an acceptance campaign.'
    return result


def audit_timing(candidate, builds, helper):
    folder = HERE / 'combined-triads-1'
    result = document(folder / 'results.json')
    require(result['performance_eligible'] is False and result['pairs'] == 7 and
            result['profile_repetitions'] == 0 and result['original_uses_candidate_library'] is True,
            'Timing scope/eligibility')
    loops = result['loops']
    require(isinstance(loops, int) and not isinstance(loops, bool) and loops > 0, 'Timing loop count')
    require(result['runner_sha256'] == digest(folder / 'runner-used.py') == digest(HERE / 'run_common.py'),
            'Timing runner identity')
    require(result['worker_sha256'] == digest(PROTOCOL / 'tirust_benchmark_worker.py'), 'Timing parser identity')
    require(result['points_sha256'] == candidate['points_sha256'], 'Timing point identity')
    require(result['candidate_library']['sha256'] == DSO, 'Timing selected DSO identity')
    require(result['build_metadata']['sha256'] == digest(folder / 'candidate-build-metadata.json') and
            document(folder / 'candidate-build-metadata.json')['library_sha256'] == DSO, 'Timing bridge build identity')
    require(result['fortran_build_metadata']['sha256'] == digest(folder / 'candidate-fortran-build-metadata.json') ==
            digest(HERE / 'combined-build/build-metadata.json'), 'Timing common Fortran build identity')
    same(document(folder / 'candidate-fortran-build-metadata.json'), builds['combined'], 'Timing build metadata')
    frozen = {'baseline': document(FROZEN / 'baseline-inputs.json'), 'original': candidate, 'split': candidate}
    variants = ['baseline', 'original', 'split']
    same(sorted(result['inputs']), sorted(variants), 'Timing input lane inventory')
    card_audit = {}
    for variant in variants:
        recorded = result['inputs'][variant]
        source = 'baseline' if variant == 'baseline' else 'candidate'
        require(recorded['frozen_input_sha256'] == digest(FROZEN / (source + '-inputs.json')),
                'Timing frozen input binding')
        same(recorded['settings'], frozen[variant]['settings'], 'Timing settings binding')
        allowed_missing = LEGACY_BASELINE_INAPPLICABLE if variant == 'baseline' else ()
        card_path = folder / (variant + '-MadLoopParams.dat')
        omitted = audit_settings_card(card_path, recorded['settings'], allowed_missing)
        require(set(omitted) == set(allowed_missing), 'Unexpected legacy card omissions: ' + variant)
        chain = [int(value) for value in recorded['settings']['mlreductionlib'].split('|')]
        same(chain, [8, 6, 7, 1] if variant == 'baseline' else [9, 8, 6, 7, 1],
             'Unexpected retained reduction chain: ' + variant)
        card_audit[variant] = dict(sha256=digest(card_path), reduction_chain=chain,
            omitted_legacy_inapplicable_settings=omitted,
            omission_scope=('These TIRust-only metadata settings are absent from the legacy '
                            'NinjaRS baseline card and inapplicable; they were not applied '
                            'or checked for equality. Every shared setting and every '
                            'candidate setting is checked against the retained card.'))
        require(recorded['driver_sha256'] == frozen[variant]['compilation']['driver_sha256'] and
                recorded['param_card_sha256'] == frozen[variant]['param_card_sha256'], 'Timing worker/card binding')
        expected_exe = builds['combined']['executable_sha256'] if variant == 'split' else frozen[variant]['compilation']['executable_sha256']
        require(recorded['executable_sha256'] == expected_exe, 'Timing executable binding')
    sequence = []
    for repetition in range(7):
        order = variants[repetition % 3:] + variants[:repetition % 3]
        if repetition % 2:
            order.reverse()
        sequence.extend((repetition, variant) for variant in order)
    same([(r['repetition'], r['variant']) for r in result['runs']], sequence, 'Timing balanced triad order')
    times = {variant: [] for variant in variants}
    rows, reference = {}, None
    for record in result['runs']:
        variant, repetition = record['variant'], record['repetition']
        target = folder / str(repetition) / variant
        expected_runtime = (frozen[variant]['runtime']['libraries'] if variant == 'baseline'
                            else runtime_expected(frozen[variant], result['candidate_library']))
        audit_runtime(target, expected_runtime, wrapped=True)
        resolved = [path for path in expected_runtime if Path(path).name == 'libmadloop_tirust.so']
        same(record['resolved_library'], resolved, 'Timing resolved-library record')
        same(record['resolved_library_sha256'], [expected_runtime[path] for path in resolved],
             'Timing resolved-library hash record')
        chain = [int(value) for value in frozen[variant]['settings']['mlreductionlib'].split('|')]
        observed = parse_outputs_checked(helper, target, chain, 100, loops)
        calibration = parse_outputs_checked(helper, target / 'calibration', chain, 100, loops)
        same(observed, record['observation'], 'Timing raw observation')
        same(calibration, record['calibration'], 'Timing raw calibration')
        same(document(target / 'observation.json'), record, 'Timing retained run record')
        backend = 8 if variant == 'baseline' else 9
        expected_usage = {f'{backend}:DP': 100 * loops}
        same(observed['backend_precision_counts'], expected_usage, 'Timing measured backend/precision')
        same(calibration['backend_precision_counts'], expected_usage, 'Timing calibration backend/precision')
        require(record['expected_backend_usage_passed'] is True, 'Timing usage flag')
        if reference is None:
            reference = observed['rows']
        parity = helper['compare_rows'](reference, observed['rows'], 1e-8)
        warm_parity = helper['compare_rows'](reference, calibration['rows'], 1e-8)
        require(parity['passed'] is True and warm_parity['passed'] is True and
                record['parity_passed'] is True and record['calibration_parity_passed'] is True,
                'Timing numerical parity')
        same(parity['maximum_relative_error'], record['maximum_relative_error'], 'Timing relative-error summary')
        times[variant].append(observed['seconds_per_point'])
        rows[(repetition, variant)] = observed['rows']
    same(times, result['timings'], 'Timing samples')
    medians = {key: statistics.median(values) * 1e6 for key, values in times.items()}
    same(medians, result['median_microseconds_per_amplitude'], 'Timing medians')
    stats = {label: diagnostic_statistics(helper, times[left], times[right])
             for label, left, right in [('original_over_baseline', 'baseline', 'original'),
                                       ('split_over_baseline', 'baseline', 'split'),
                                       ('split_over_original', 'original', 'split')]}
    same(stats, result['paired_statistics'], 'Timing paired Student-t intervals')
    exact = all(rows_binary(rows[(rep, 'original')], True) == rows_binary(rows[(rep, 'split')], True)
                for rep in range(7))
    require(exact is True and result['all_original_split_bit_identical'] is True, 'Timing coefficient bit parity')
    maximum = max(helper['compare_rows'](rows[(rep, 'original')], rows[(rep, 'split')], 1e-8)['maximum_relative_error']
                  for rep in range(7))
    same(maximum, result['maximum_original_split_relative_error'], 'Timing original/split maximum error')
    return dict(triads=7, measured_runs=21, calibration_runs=21, loops_per_run=loops,
                dp_amplitudes_per_measured_run=100 * loops, qp_amplitudes=0,
                all_original_split_coefficients_bit_identical=exact,
                reduction_cards=card_audit,
                medians_microseconds=medians, paired_statistics=stats, performance_eligible=False,
                qualification='Bounded diagnostic triads. The threshold and confidence intervals are descriptive; these runs do not qualify performance acceptance.')


def external_inventory():
    entries = []
    paths = [FROZEN / (name + '-inputs.json') for name in ('baseline', 'candidate')]
    paths += [ROOT / 'docs/benchmarkfolder/points/qcd_n1_uux_z.dat', FORTRAN / 'preparation.json',
              PROTOCOL / 'identity.json']
    for name in ('tirust_benchmark.py', 'tirust_benchmark_worker.py',
                 'tirust_benchmark_worker.f', 'tirust_benchmark_compiler.py'):
        paths.extend([PROTOCOL / name, FROZEN / 'protocol-sources' / name])
    protocol_identity = document(PROTOCOL / 'identity.json')
    require(set(protocol_identity) == set(PROTOCOL_SHA256), 'Unexpected/incomplete protocol identity keys')
    for name, value in protocol_identity.items():
        require(digest(PROTOCOL / name) == value['frozen_sha256'] == PROTOCOL_SHA256[name] ==
                digest(FROZEN / 'protocol-sources' / name), 'Retained protocol identity: ' + name)
    for path in paths:
        row = identity(path, ROOT)
        row['binding_scope'] = 'Retained frozen input, protocol, or attribution identity; read during this audit.'
        entries.append(row)
    for path in (FORTRAN / 'prepare.py', FORTRAN / 'run.py'):
        row = identity(path, ROOT)
        row['binding_scope'] = ('Audit-time imported-helper identity only. No separate execution-time '
                                'snapshot or recorded hash for this helper was captured by the completed run.')
        entries.append(row)
    return sorted(entries, key=lambda row: row['path'])


def helper_snapshot_inventory():
    entries = []
    for original, filename in (('madgraph/various/banner.py', 'banner.py'),
                               ('madgraph/iolibs/file_writers.py', 'file_writers.py')):
        row = identity(HERE / 'audit-helper-snapshots' / filename, HERE)
        row['original_source_path'] = original
        row['binding_scope'] = ('Bytes copied at audit time, not an execution-time snapshot. '
                                'The verifier never imports this snapshot or reads the live original source.')
        entries.append(row)
    return entries


def static_evidence():
    candidate = document(FROZEN / 'candidate-inputs.json')
    helper = frozen_helpers()
    builds, build_summary = audit_builds(candidate)
    summary = dict(builds=build_summary, stub=audit_stub(), native=audit_native(candidate, builds, helper))
    return candidate, helper, builds, summary


def namespace_inventory():
    files = []
    for path in sorted(HERE.rglob('*')):
        if '__pycache__' in path.relative_to(HERE).parts or path == INDEX:
            continue
        require(not path.is_symlink(), 'Symlink in evidence namespace: ' + str(path))
        if path.is_file():
            files.append(identity(path, HERE))
        else:
            require(path.is_dir(), 'Nonregular evidence entry: ' + str(path))
    require(any(row['path'] == 'README.md' for row in files), 'Missing frozen README')
    return files


def collected_index():
    candidate, helper, builds, summary = static_evidence()
    summary['timing'] = audit_timing(candidate, builds, helper)
    files = namespace_inventory()
    return dict(schema=SCHEMA, namespace=relative(HERE, ROOT), files=files,
                external_files=external_inventory(), audit_time_helper_snapshots=helper_snapshot_inventory(),
                evidence=summary,
                performance_eligible=False,
                limitations=[
                    'Verification replays retained parsers/statistics, not native amplitudes, compilation, or performance measurements.',
                    'All physical native-parity and timing samples used DP; QP coverage is confined to the separate recording-stub switch corpus.',
                    'Absolute build/runtime paths in recorded metadata are historical identities; portability does not require those ignored work directories or installed libraries.',
                    'The frozen 111-entry source inventory is bound as recorded input evidence; the new build metadata does not snapshot every source/include file anew at execution time.',
                    'Some imported helper files are bound only at audit time, explicitly identified in external_files or audit_time_helper_snapshots; those hashes must not be described as execution-time snapshots.',
                    'Stub raw outputs and executables remain outside the portable namespace; their historical hashes and outcomes are retained in stub-parity.json.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--write', action='store_true', help='Create a new index exclusively; never overwrite.')
    group.add_argument('--check-static', action='store_true', help='Do not inspect the timing folder or create an index.')
    args = parser.parse_args()
    if args.check_static:
        print(json.dumps(dict(scope='Static evidence only; timing/index untouched',
                              evidence=static_evidence()[3]), indent=2, allow_nan=False))
        return
    if args.write:
        require(not INDEX.exists(), 'Index already exists; refusing to overwrite')
        result = collected_index()
        with INDEX.open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write('\n')
    else:
        expected = document(INDEX)
        require(expected.get('schema') == SCHEMA, 'Unexpected index schema')
        # Check archived hashes before interpreting any retained helper code.
        same(namespace_inventory(), expected['files'], 'Namespace inventory/hash mismatch')
        same(external_inventory(), expected['external_files'], 'External inventory/hash mismatch')
        result = collected_index()
        same(result, expected, 'Inventory, identity, or reparsed-evidence mismatch')
    print(json.dumps(dict(outcome='verified', files=len(result['files']),
                          external_files=len(result['external_files']),
                          index_sha256=digest(INDEX), performance_eligible=False), indent=2))


if __name__ == '__main__':
    main()
