"""Read two closed TIRust lanes; no numerical evaluation or source mutation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import types

DATA_SHA = 'ebd8f1faf5228f205b1038a490329037066988dc03e7e774210d648b38dbbb6e'
CODEC_SHA = '33505d7db5186df76c974cf419b2333a8efbc1ce90ecee67a19c9edfd9de7a7a'
CASES = ('qcd_n3_uux_zgg', 'qcd_n4_uux_zzgg')


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def diagnose(data_root):
    root = Path(data_root)
    index_blob = (root / 'index.json').read_bytes()
    assert digest(index_blob) == DATA_SHA
    index = json.loads(index_blob)
    codec_blob = (root / 'journal_codec.py').read_bytes()
    assert digest(codec_blob) == CODEC_SHA
    codec = types.ModuleType('closed_diagnosis_codec')
    exec(compile(codec_blob, str(root / 'journal_codec.py'), 'exec'), codec.__dict__)

    def payload(name):
        item = index['payloads'][name]
        if item['kind'] == 'gzip-chunks':
            blob = codec.decode_journal(item['descriptor'], root, total_raw_budget=512 * 1024 * 1024)
        else:
            path = Path(item['path'])
            assert not path.is_absolute() and '..' not in path.parts
            blob = (root / path).read_bytes()
        assert len(blob) == item['bytes'] and digest(blob) == item['sha256']
        return blob

    result = {}
    for case in CASES:
        prefix = 'requests/' + case + '-backend9/'
        record = json.loads(payload(prefix + 'result.json'))
        parsed = record['diagnostics']
        calls = parsed['loop_group_calls']
        primary = [call for call in calls if call['phase'] == 1]
        failed = [call for call in calls if call['failed_groups']]
        transition = next(call for call in calls if call['fpe_dp_became_set'])
        first_primary_failure = next(call for call in primary if call['failed_groups'])
        latch_only = [call for call in primary if not call['failed_groups']
                      and (call['fpe_dp'] or call['fpe_qp'] or call['fpe_dp_before'] or call['fpe_qp_before'])]
        selected = [transition, first_primary_failure, primary[0], latch_only[0]]
        at = calls.index(transition)
        if at:
            selected.append(calls[at - 1])
        keys = {(c['phase'], c['call'], c['point'], c['method']) for c in selected}
        group_blob = payload(prefix + 'stability-groups.dat')
        group_rows = []
        before_keys = set()
        closed_keys = set()
        for number, line in enumerate(group_blob.decode().splitlines(), 1):
            fields = line.split()
            if not fields or fields[0] not in ('B', 'G', 'C'):
                continue
            key = tuple(map(int, fields[1:5]))
            if fields[0] == 'B':
                before_keys.add(key)
            elif fields[0] == 'C':
                closed_keys.add(key)
            if key not in keys:
                continue
            if fields[0] != 'G' or (int(fields[8]) > 0 and int(fields[10]) == 0):
                group_rows.append({'line': number, 'raw': line})
        console_blob = payload(prefix + 'native/stdout.txt')
        console_rows = []
        active = None
        for number, line in enumerate(console_blob.decode(errors='replace').splitlines(), 1):
            fields = line.split()
            if fields[:1] == ['STABILITY_CALL_BEGIN']:
                active = tuple(map(int, fields[1:5]))
            if active in keys and ('STABILITY_CALL_' in line or '##W03' in line):
                console_rows.append({'line': number, 'raw': line})
            if fields[:1] == ['STABILITY_CALL_END']:
                active = None
        points = sorted({call['point'] for call in primary if call['failed_groups']})
        status_rows = [row for row in parsed['rows'] if row['phase'] == 1 and row['order'] == 0]
        assert all(row['status'] == 110 and all(type(v) in (int, float) and math.isfinite(v)
                   for v in row['coefficients']) for row in status_rows)
        assert all(call['nonfinite_coefficient_groups'] == call['nonfinite_result_groups'] == 0 for call in calls)
        result[case] = {
            'request_sha256': index['payloads'][prefix + 'result.json']['sha256'],
            'group_journal_sha256': digest(group_blob), 'console_sha256': digest(console_blob),
            'state': record['state'], 'native_returncode': record['native']['returncode'],
            'native_error': record['native']['error'], 'diagnostic_error': record.get('diagnostic_error'),
            'first_dp_latch_transition': transition,
            'first_primary_group_failure': first_primary_failure,
            'first_primary_call': primary[0], 'first_latch_only_primary_call': latch_only[0],
            'bootstrap_group_failure_calls': sum(c['phase'] == 0 for c in failed),
            'bootstrap_failed_group_observations': sum(len(c['invalid_groups']) for c in failed if c['phase'] == 0),
            'primary_group_failure_calls': sum(c['phase'] == 1 for c in failed),
            'primary_failed_group_observations': sum(len(c['invalid_groups']) for c in failed if c['phase'] == 1),
            'primary_points_with_failed_groups': points,
            'primary_group_ids_by_point': {str(p): sorted({g['group'] for c in primary if c['point'] == p
                                                         for g in c['invalid_groups']}) for p in points},
            'primary_latch_only_calls': len(latch_only),
            'primary_latch_affected_calls': sum(c['fpe_dp'] or c['fpe_qp'] or c['fpe_dp_before'] or c['fpe_qp_before'] for c in primary),
            'dp_latch_rises': sum(c['fpe_dp_became_set'] for c in calls),
            'qp_latch_rises': sum(c['fpe_qp_became_set'] for c in calls),
            'completed_primary_order0_rows': len(status_rows),
            'all_saved_primary_order0_statuses_110_and_amplitudes_finite': True,
            'observed_nonfinite_coefficient_or_result_group_counts': 0,
            'unclosed_before_call_keys': [list(key) for key in sorted(before_keys - closed_keys)],
            'selected_exact_group_lines': group_rows, 'selected_exact_console_lines': console_rows,
        }
    assert result[CASES[0]]['primary_group_failure_calls'] == 15
    assert result[CASES[0]]['primary_latch_only_calls'] == 213
    assert result[CASES[1]]['bootstrap_group_failure_calls'] == 4
    assert result[CASES[1]]['primary_group_failure_calls'] == 212
    assert result[CASES[1]]['primary_latch_only_calls'] == 42
    return {'schema': 'madloop-raw-stability-first-failure-diagnosis-v1',
            'data_index_sha256': DATA_SHA, 'cases': result,
            'interpretation': 'software failure and latch observations only; bridge status, internal tensor outputs and hardware IEEE flags were not recorded'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    result = diagnose(args.data_root)
    if args.verify:
        expected = json.loads(Path(__file__).with_name('diagnosis.json').read_bytes())
        assert result == expected, 'Derived diagnosis differs from closed evidence'
        print(json.dumps({'verified': True, 'data_index_sha256': DATA_SHA, 'cases': list(result['cases'])}))
    else:
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
