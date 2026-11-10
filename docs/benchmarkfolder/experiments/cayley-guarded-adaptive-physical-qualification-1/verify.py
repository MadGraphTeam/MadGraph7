#!/usr/bin/env python3
"""Verify and reanalyze retained UTF8/gzip evidence without historical natives."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SOURCE_SHA = '7d9a40657b9df702fe1bdd1b3f302363813b1ba4a9514a2bcc88467f371dcd1c'
BINDING_SHA = 'ee2933ed02f2f2c6ba1b004c0feb951553275c2967f829b5bb0efb894de7c61a'
PROPOSAL = 'docs/benchmarkfolder/experiments/cayley-guarded-adaptive-physical-proposal-2'
WORK = '.tirust-build/cayley-guarded-adaptive-physical-proposal-2'
PREFIX = '/shared/external/madloop_tirust/madgraph7_madloop_tirust/'
LIBRARY_SHA = '69722452c234f494f0aa436fa24bbabfeae59100c03b05a1f0b76336cc17886b'
BUILD_RECEIPT_SHA = '59167eec84a767f0712b6905285a99564458ff0667427de87ce1921534b46632'
EXECUTION_SHA = '18f7d89c565a97962871ee428e4edc993e7d33c9e5980da96915cff0bf22ba13'
LOCALIZATION_SHA = '08c221e1a259d0e47595ebac67d04c4471f2efec2713c9afb7afde0d61d31264'

def native_identity_only(name):
    path = Path(name)
    return (path.is_absolute() and not name.startswith(PREFIX) or
            path.suffix in {'.so','.a','.o','.mod','.rlib','.pyc'} or
            '.so.' in path.name or path.name in {'checked','guarded-qp'})

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def safe_relative(name):
    path = Path(name)
    assert name and not path.is_absolute() and '..' not in path.parts
    return path

def load_blobs():
    index = json.loads((HERE / 'evidence-index.json').read_text())
    all_paths = list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in all_paths)
    assert {str(p.relative_to(HERE)) for p in all_paths if p.is_file()} == set(index['files']) | {'evidence-index.json'}
    for name, entry in index['files'].items():
        raw = (HERE / safe_relative(name)).read_bytes()
        assert len(raw) == entry['bytes'] <= 1048576 and digest(raw) == entry['sha256'], name
        assert not raw.startswith((b'\x7fELF', b'!<arch>'))
        assert Path(name).suffix not in ('.so','.rlib','.o','.a','.mod','.pyc')
        (gzip.decompress(raw) if name.endswith('.gz') else raw).decode('utf8')
    manifest = json.loads((HERE / 'manifest.json').read_text())
    assert manifest['schema'] == 'guarded-physical-portable-text-v1'
    assert not manifest['native_payloads_retained'] and not manifest['historical_native_paths_required']
    assert not manifest['native_loader_or_process_executed_by_capture']
    assert manifest['source_index_sha256'] == SOURCE_SHA
    assert manifest['artifact_binding_sha256'] == BINDING_SHA
    blobs = {}
    used = set()
    for name, entry in manifest['text_payloads'].items():
        safe_relative(name)
        parts = []
        for piece in entry['chunks']:
            safe_relative(piece)
            assert piece.startswith('payloads/') and piece.endswith('.gz')
            raw = gzip.decompress((HERE / piece).read_bytes())
            raw.decode('utf8')
            assert Path(piece).stem == digest(raw)
            parts.append(raw)
            used.add(piece)
        raw = b''.join(parts)
        assert len(raw) == entry['bytes'] and digest(raw) == entry['sha256'], name
        blobs[name] = raw
    assert used == {name for name in index['files'] if name.startswith('payloads/')}
    assert digest(blobs[PROPOSAL + '/source-index.json']) == SOURCE_SHA
    assert digest(blobs[WORK + '/binding1.json']) == BINDING_SHA
    assert digest(blobs[WORK + '/run1/execution.json']) == manifest['execution_sha256']
    assert manifest['execution_sha256'] == EXECUTION_SHA
    source = json.loads(blobs[PROPOSAL + '/source-index.json'])
    for name, sha in source['sources'].items():
        assert digest(blobs[PROPOSAL + '/' + name]) == sha, name
    for name, sha in source['dependencies'].items():
        key = name.removeprefix(PREFIX)
        if not native_identity_only(name):
            assert digest(blobs[key]) == sha, key
        else:
            assert key not in blobs
            assert manifest['native_identities'][key]['sha256'] == sha, key
            assert not manifest['native_identities'][key]['payload_retained']
    binding = json.loads(blobs[WORK + '/binding1.json'])
    assert binding['source_index_sha256'] == SOURCE_SHA
    assert binding['build_receipt_sha256'] == BUILD_RECEIPT_SHA
    receipt_raw = blobs[binding['build_receipt_path'].removeprefix(PREFIX)]
    assert digest(receipt_raw) == BUILD_RECEIPT_SHA
    receipt = json.loads(receipt_raw)
    assert receipt['complete'] and not receipt['loader_or_native_test_executed']
    assert receipt['private_library_not_loaded'] == binding['library']
    assert binding['library']['sha256'] == LIBRARY_SHA
    assert binding['library']['bytes'] == 69162336
    record = json.loads(blobs[WORK + '/run1/execution.json'])
    assert record['source_index_sha256'] == SOURCE_SHA
    assert record['artifact_binding_sha256'] == BINDING_SHA
    assert record['guarded_library_sha256'] == LIBRARY_SHA
    for row in [record.get('loader', {})] + list(record['workers'].values()):
        progress = row.get('progress', {})
        assert not progress.get('process_spawned') or progress.get('process_reaped')
    for name, sha in record.get('runtime_sha256_after', {}).items():
        key = name.removeprefix(PREFIX)
        assert native_identity_only(name)
        assert manifest['native_identities'][key]['sha256'] == sha
        assert not manifest['native_identities'][key]['payload_retained']
    if record.get('execution_complete'):
        assert record['protected_inputs_after'] == 'unchanged'
        assert len(record['runtime_sha256_after']) == 15
        bridge_key = WORK + '/run1/runtime-guarded/libmadloop_tirust.so'
        assert record['runtime_sha256_after'][PREFIX + bridge_key] == LIBRARY_SHA
        assert manifest['native_identities'][bridge_key]['sha256'] == LIBRARY_SHA
    for row in record['workers'].values():
        if record.get('execution_complete'):
            assert row['expected_runtime_sha256'] == record['runtime_sha256_after']
        for name, sha in row.get('complete_case_files_sha256', {}).items():
            assert digest(blobs[name]) == sha, name
    assert record['execution_complete'] == manifest['execution_complete_as_recorded']
    assert record.get('failure') == manifest['runtime_failure_retained']
    return blobs, manifest, index

def verify():
    blobs, manifest, index = load_blobs()
    spec = importlib.util.spec_from_file_location('portable_guarded_physical_analysis',HERE / 'analyze.py')
    analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyzer)
    result = analyzer.reanalyze(blobs)
    assert result == json.loads(gzip.decompress((HERE / 'offline-result.json.gz').read_bytes()))
    assert digest((HERE / 'localization/evidence-index.json').read_bytes()) == LOCALIZATION_SHA
    localized_spec = importlib.util.spec_from_file_location('portable_guarded_static_localization',HERE / 'localization/verify.py')
    localized = importlib.util.module_from_spec(localized_spec)
    localized_spec.loader.exec_module(localized)
    assert localized.verify()['verified']
    assert digest(gzip.decompress((HERE / 'localization/inputs/runtime-execution.json.gz').read_bytes())) == EXECUTION_SHA
    localization = json.loads((HERE / 'localization/analysis.json').read_text())
    record = json.loads(blobs[WORK + '/run1/execution.json'])
    assert summary(result,manifest,record,localization) == json.loads((HERE / 'summary.json').read_text())
    return {'verified':True, 'payload_files':len(index['files']),
            'text_inputs':len(blobs), 'native_identity_only':len(manifest['native_identities']),
            'execution_complete_as_recorded':manifest['execution_complete_as_recorded'],
            'native_loader_or_process_executed':False,
            'offline_result':{key: result[key] for key in (
                'gate_observations','qualified_gates','checked_trap_gate_qualified',
                'forced_qp_trap_gate_qualified','fault_controls_demonstrated',
                'independent_physical_truth_qualified','per_call_working_precision_qualified',
                'performance_qualified')}}

def summary(result, manifest, record, localization):
    workers = {}
    for name, row in result['workers'].items():
        workers[name] = {key:row.get(key) for key in (
            'kind','mode','policy','exit_code','journal_disposition',
            'retained_offline_assessment_failure','control_assessment','numerical_assessment')}
    comparisons = {}
    for name, row in result['comparisons'].items():
        comparisons[name] = {
            'numerical_agreement_reference':row.get('numerical_agreement_reference'),
            'passed':row['numerical_outputs']['passed'],
            'ordinary_target_passed':row['numerical_outputs']['ordinary_target_passed'],
            'maximum_relative_error':row['numerical_outputs'].get('maximum_relative_error'),
            'strict_bitwise_outputs_exact':row['strict_bitwise_diagnostic']['outputs_exact'],
            'path_changes':row['path_changes']}
    return {'schema':'guarded-adaptive-physical-closed-summary-v1',
            'execution_sha256':EXECUTION_SHA, 'source_index_sha256':SOURCE_SHA,
            'artifact_binding_sha256':BINDING_SHA, 'library_sha256':LIBRARY_SHA,
            'build_receipt_sha256':BUILD_RECEIPT_SHA,
            'text_inputs':len(manifest['text_payloads']),
            'native_identity_only':len(manifest['native_identities']),
            'execution_complete':record['execution_complete'],
            'diagnostic_elapsed_seconds_not_benchmark':record['elapsed_seconds'],
            'workers':workers,'comparisons':comparisons,
            **{key:result[key] for key in ('gate_observations','qualified_gates',
                'checked_trap_gate_qualified','forced_qp_trap_gate_qualified',
                'fault_controls_demonstrated','independent_physical_truth_qualified',
                'per_call_working_precision_qualified','performance_qualified')},
            'localization_evidence_index_sha256':LOCALIZATION_SHA,
            'trapped_instruction_sites':[{key:case[key] for key in (
                'case','ELF_address','reported_PC','instruction','symbol_offset',
                'not_in_cayley_cancellation_symbol',
                'actual_register_operands_or_exception_subtype_available')} |
                {'symbol':case['symbol']['name']} for case in localization['cases']]}

if __name__ == '__main__':
    print(json.dumps(verify(),indent=2,sort_keys=True))
