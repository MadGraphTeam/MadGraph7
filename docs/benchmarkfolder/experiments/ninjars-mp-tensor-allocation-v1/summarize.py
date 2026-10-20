#!/usr/bin/env python3
"""Recompute this isolated experiment's correctness summary from retained logs."""
import ast
import gzip
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize():
    provenance = json.loads((HERE / 'provenance.json').read_text())
    capture_path = HERE.parents[3] / provenance['capture']['path']
    assert sha(capture_path) == provenance['capture']['sha256']
    capture_bytes = gzip.decompress(capture_path.read_bytes())
    assert hashlib.sha256(capture_bytes).hexdigest() == provenance['capture']['uncompressed_sha256']
    raw_requests = [json.loads(line) for line in capture_bytes.splitlines()]
    assert len(raw_requests) == 1392 and all(row['status'] == 0 for row in raw_requests)
    for name, digest in provenance['original_files'].items():
        assert sha(HERE / 'original' / name) == digest, name
    for name, record in provenance['dependency_implementations'].items():
        assert sha(HERE / 'dependency' / name) == record['sha256']
    assert sha(HERE / 'tensor.rs') == provenance['prototype_tensor_sha256']
    runs = {}
    signatures = set()
    for name in ['unit-2', 'golden-1', 'capture-all-1']:
        base = HERE / name
        record = json.loads((base / 'result.json').read_text())
        assert record['returncode'] == 0
        for field in ['source_unchanged', 'production_unchanged', 'installed_archive_unchanged', 'executable_unchanged']:
            assert record[field], field
        for stream in ['stdout', 'stderr']:
            assert sha(base / f'{stream}.log') == record[f'{stream}_sha256']
        assert sha(HERE / record['build_record']) == record['build_record_sha256']
        sources = record['source_before']
        for path, digest in sources.items():
            if path == 'src/tensor.rs': retained = HERE / 'tensor.rs'
            elif path in ['src/tensor_allocation_tests.rs', 'src/tensor_capture_tests.rs']: retained = HERE / Path(path).name
            elif path == 'src/tensor_dense_reference.rs': retained = HERE / 'original/src/tensor.rs'
            else: retained = HERE / 'original' / path
            assert sha(retained) == digest, path
        signatures.add((record['executable_sha256'], json.dumps(sources, sort_keys=True)))
        stdout = (base / 'stdout.log').read_text()
        result = re.findall(r'test result: ok\. (\d+) passed; (\d+) failed; (\d+) ignored;', stdout)
        assert len(result) == 1 and int(result[0][1]) == 0
        runs[name] = {'result_sha256': sha(base/'result.json'), 'passed_tests': int(result[0][0]), 'ignored_tests': int(result[0][2]), 'elapsed_seconds': record['elapsed_seconds']}
    assert len(signatures) == 1
    capture = (HERE/'capture-all-1/stderr.log').read_text()
    summary = re.findall(r'captured exact requests=(\d+); expansion/value counts=(\{.*\})', capture)
    assert len(summary) == 1
    inventory = json.loads((HERE/'input-inventory.json').read_text())
    assert inventory['capture'] == provenance['capture']
    assert len({(row['n'],row['rank']) for row in raw_requests}) == len(inventory['groups'])
    assert int(summary[0][0]) == inventory['successful_records'] == 1392
    counts = ast.literal_eval(summary[0][1])
    assert all(counts.get(kind, 0) > 0 for kind in ['numerator_values', 't2', 't3', 'mu'])
    golden = (HERE/'golden-1/stderr.log').read_text()
    golden_summary = re.findall(r'golden exact lanes=(\d+); expansion/value counts=(\{.*\})', golden)
    assert len(golden_summary) == 1 and int(golden_summary[0][0]) == 21
    unit = (HERE/'unit-2/stdout.log').read_text()
    convergence = [json.loads(line.split('NINJARS_CONVERGENCE ', 1)[1]) for line in unit.splitlines() if 'NINJARS_CONVERGENCE ' in line]
    assert len(convergence) == 1 and convergence[0]['passed']
    original_convergence = json.loads((HERE/'original/tests/data/convergence.json').read_text())
    for field in ['input', 'result_256', 'result_384']:
        assert convergence[0][field] == original_convergence[field], field
    failed_base = HERE/'unit-1'
    failed = json.loads((failed_base/'result.json').read_text())
    assert failed['returncode'] == 101
    assert sha(failed_base/'tensor_allocation_tests.rs') == failed['source_before']['src/tensor_allocation_tests.rs']
    for stream in ['stdout','stderr']:
        assert sha(failed_base/f'{stream}.log') == failed[f'{stream}_sha256']
    assert '19 passed; 1 failed; 2 ignored' in (failed_base/'stdout.log').read_text()
    assert 'sign differs' in (failed_base/'stderr.log').read_text()
    scalar = json.loads((HERE/'scalar-result.json').read_text())
    assert scalar['returncode'] == 0
    assert sha(HERE/'scalar-diagnostic.rs') == scalar['source_sha256']
    assert sha(HERE/'scalar-build.sh') == scalar['build_command_sha256']
    for stream in ['stdout','stderr']:
        assert sha(HERE/f'scalar-diagnostic.{stream}') == scalar[f'{stream}_sha256']
    assert (HERE/'scalar-diagnostic.stdout').read_text().endswith('exact_products=245 exact_actual_zero_one_resets=70 unrelated_arbitrary_nan_copy_mismatches=15\n')
    return {'schema':'mp-tensor-allocation-correctness-v1', 'passed':True, 'acceptance_evidence':False,
            'performance_measured':False, 'production_changed':False,
            'provenance_sha256':sha(HERE/'provenance.json'), 'executable_sha256':next(iter(signatures))[0],
            'runs':runs, 'actual_requests':1392, 'actual_shapes':len(inventory['groups']),
            'actual_callback_complex_outputs':counts, 'actual_ascii_fields_compared':1392*8,
            'golden_lanes':21, 'golden_callback_complex_outputs':ast.literal_eval(golden_summary[0][1]),
            'convergence_256_384_matches_original_exactly':True,
            'retained_failed_suite':{'result_sha256':sha(failed_base/'result.json'),'passed':19,'failed':1,'cause':'Direct arbitrary NaN-source clone_from retains destination sign; unused by finite zero/one reset sites. Numerical patch unchanged.'},
            'scalar_diagnostic':{'result_sha256':sha(HERE/'scalar-result.json'),'exact_exceptional_products':245,'exact_zero_one_resets':70,'unrelated_arbitrary_nan_copy_mismatches':15},
            'comparison':'Every callback complex component: exact stored rational value, tracked precision and sign. Every captured final ASCII field: exact text.',
            'limitations':['Actual capture replay creates a new numerator/amplitude per request; changed retained state is tested synthetically and across all three golden lanes per case.', 'No runtime improvement is established.']}


if __name__ == '__main__':
    result = summarize()
    destination = HERE/'summary.json'
    if destination.exists():
        assert json.loads(destination.read_text()) == result
    else:
        destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
