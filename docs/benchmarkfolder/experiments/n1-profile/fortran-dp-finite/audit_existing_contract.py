#!/usr/bin/env python3
"""Recheck already retained raw ABI evidence without loading native libraries."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    bridge = HERE.parent / 'bridge'
    controller = bridge / 'split-validation.json'
    worker = bridge / 'split-validation-workers/split.json'
    data = json.loads(worker.read_text())
    expected = json.loads(controller.read_text())['workers']['split']['result_sha256']
    assert digest(worker) == expected
    assert data['identity']['library_sha256'] == (
        '1c3688011fb0b423142ab4ac4c832f5b71d8e4409ef323087dfac5ceb2075cd7')
    records = []
    for event in data['events']:
        if event['operation'] != 'evaluate_dp':
            continue
        raw = bytes.fromhex(data['output_payloads'][event['output_sha256']])
        assert hashlib.sha256(raw).hexdigest() == event['output_sha256']
        assert len(raw) == event['output_bytes']
        values = list(struct.iter_unpack('<dd', raw))
        if event['status'] == 0:
            assert all(math.isfinite(part) for value in values for part in value)
            if not event['options']['finite']:
                assert all(value == (0.0, 0.0) for value in values[3 * len(values) // 4:])
        else:
            assert all(value == (999.0, 999.0) for value in values)
            assert event['failure_sentinel_unchanged']
        records.append({key: event[key] for key in
                        ('scenario', 'label', 'fixture', 'status', 'options',
                         'error', 'before', 'after', 'output_sha256')})
    successful = [row for row in records if row['status'] == 0]
    failed = [row for row in records if row['status'] != 0]
    modes = Counter((row['options']['finite'], row['options']['poles']) for row in successful)
    assert {(1, n) for n in range(4)} | {(0, n) for n in range(3)} <= set(modes)
    assert any(row['options']['check'] for row in successful)
    assert any(not row['options']['cache'] for row in successful)
    assert any(any(row['after'][i] > row['before'][i] for i in (0, 2, 4))
               for row in successful)
    assert any(row['label'] == 'overflow' for row in failed)
    result = {
        'scope': 'Read-only reanalysis of existing native ABI evidence, not a new numerical run.',
        'performance_eligible': False,
        'library_sha256': data['identity']['library_sha256'],
        'evidence_sha256': {str(path.relative_to(HERE.parents[4])): digest(path)
                            for path in (controller, worker, Path(__file__))},
        'calls': len(records), 'successful_calls': len(successful),
        'failed_calls': len(failed),
        'every_returned_success_component_finite': True,
        'every_failed_call_left_all_output_sentinels_unchanged': True,
        'unevaluated_finite_slots_are_zero_at_raw_ABI': True,
        'observed_cache_reuse_and_independent_check_bypass': True,
        'mode_success_counts': [{'compute_finite': mode[0], 'pole_mode': mode[1], 'calls': count}
                                for mode, count in sorted(modes.items())],
        'records': records,
        'limit': 'Examples support the source proof; they cannot establish a universal contract on their own.'}
    (HERE / 'existing-native-contract-evidence.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('calls', 'successful_calls', 'failed_calls')}))


if __name__ == '__main__':
    main()
