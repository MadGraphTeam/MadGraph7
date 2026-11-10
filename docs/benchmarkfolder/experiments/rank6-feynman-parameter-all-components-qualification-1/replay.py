#!/usr/bin/env python3
"""Portable offline replay of closed all-component quadrature evidence."""
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p/'madgraph').is_dir() and (p/'tools').is_dir())
RAW_SHA256 = '3d1c56b4e3fbecc4aac439714cbbe1dd6d273b6a4a5d03e85984f33477168cee'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_runtime():
    receipt = json.loads((HERE/'data/execution.json').read_text())
    assert receipt['root_admitted'] and receipt['complete']
    assert receipt['returncode'] == 0 and receipt['progress']['returncode'] == 0 and receipt['progress']['process_reaped']
    assert receipt['cpu'] == 10 and receipt['cpu_exclusive'] is False
    assert receipt['address_space_limit_bytes'] == 4294967296
    assert receipt['aggregate_worker_wall_cap_seconds'] == 180 and receipt['elapsed_seconds'] < 180
    assert receipt['automatic_retries'] == receipt['native_reducer_calls'] == 0
    assert receipt['source_before'] == receipt['source_after']
    for original, expected in receipt['source_before'].items():
        marker = '/madgraph7_madloop_tirust/'
        relative = original.split(marker, 1)[1]
        path = HERE/'data/controller.py' if relative.endswith('/rank6-fp-all-components-runtime-1/controller.py') else REPO/relative
        assert digest(path) == expected, str(path)
    raw = gzip.decompress((HERE/'data/vector-result.json.gz').read_bytes())
    assert hashlib.sha256(raw).hexdigest() == RAW_SHA256
    for name, metadata in receipt['outputs'].items():
        payload = raw if name == 'vector-result.json' else (HERE/'data'/name).read_bytes()
        assert len(payload) == metadata['bytes'] and hashlib.sha256(payload).hexdigest() == metadata['sha256']
    result = json.loads(raw)
    assert result['status'].startswith('completed estimates;')
    assert [row['n'] for row in result['orders']] == [8, 16, 18]
    assert result['actual_cpu_affinity'] == [10]
    assert result['extended_dtype']['significand_bits'] == 64
    return result


def main():
    index = json.loads((HERE/'index.json').read_text())
    for record in index['files']:
        path = HERE/record['path']; assert path.resolve().is_relative_to(HERE.resolve())
        raw = path.read_bytes()
        assert len(raw) == record['bytes'] and hashlib.sha256(raw).hexdigest() == record['sha256'], str(path)
        if record.get('uncompressed_sha256'):
            raw = gzip.decompress(raw)
            assert len(raw) == record['uncompressed_bytes'] and hashlib.sha256(raw).hexdigest() == record['uncompressed_sha256']
    for record in index['inputs']:
        path = REPO/record['path']; assert path.resolve().is_relative_to(REPO.resolve())
        assert path.stat().st_size == record['bytes'] and digest(path) == record['sha256'], str(path)
    result = verify_runtime()
    spec = importlib.util.spec_from_file_location('all210_offline_replay', HERE/'analyze.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    analysis = module.analyze(result); summary = module.summary(analysis)
    assert json.loads(json.dumps(analysis)) == json.loads(gzip.decompress((HERE/'analysis.json.gz').read_bytes()))
    assert summary == json.loads((HERE/'summary.json').read_text())
    independent = importlib.util.spec_from_file_location('all210_independent_fraction', HERE/'independent_fraction_audit.py')
    witness = importlib.util.module_from_spec(independent); independent.loader.exec_module(witness)
    raw = gzip.decompress((HERE/'data/vector-result.json.gz').read_bytes())
    audit = witness.analyze(raw, witness.ROWS.read_bytes())
    assert audit == json.loads((HERE/'independent-fraction-audit.json').read_text())
    endpoint = summary['orders'][-1]['summary']
    print(json.dumps({'passed': True, 'owned_files': len(index['files']), 'bound_inputs': len(index['inputs']),
                      'components': 210, 'parts': 420, 'orders': [8, 16, 18],
                      'tirust_pass_3e6': endpoint['TIRust_QP']['pass_3e6'],
                      'collier_pass_3e6': endpoint['COLLIER']['pass_3e6'],
                      'new_quadrature_or_native_calls': 0}))


if __name__ == '__main__':
    main()
