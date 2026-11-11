"""Pure summary of six admitted input/status decodes; never reads tensor outputs."""
from collections import Counter
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEYS = ('section', 'point', 'serial', 'id', 'mode', 'rank', 'n', 'split_order')


def rational(value):
    return F(int(value['numerator_hex'], 16), int(value['denominator_hex'], 16))


def pair_observation(records, call, pair):
    suffix = 'QP' if call['identity']['mode'] == 4 else 'DP'

    def one(tag, selected_pair=None):
        found = [r for r in records if r['tag'] == 'KT_' + tag + '_' + suffix
                 and (selected_pair is None or [r['i'], r['j']] == selected_pair)]
        assert len(found) <= 1
        return found[0]['data'] if found else None

    raw, final = one('RAW', pair), one('FINAL', pair)
    matrix, native = one('MATRIX'), one('TIRUST')
    i, j = pair
    state = next((d for d in call.get('threshold_decisions', []) if d['pair'] == pair), None)
    native_value = native['invariants_column_major'][(j-1)*4+i-1] if native else None
    masses = matrix['squared_masses'] if matrix else None
    sequence = [{'tag': r['tag'], 'line': r['line'], 'matrix_entry': r['data']['matrix_entry']}
                for r in records if [r['i'], r['j']] == pair and any(
                    r['tag'] == 'KT_' + branch + '_' + edge + '_' + suffix
                    for branch in ('MASSI', 'MASSJ', 'ZERO') for edge in ('BEFORE', 'AFTER'))]
    return {'pair': pair, 'raw_diffsq': raw['diffsq'] if raw else None,
        'raw_matrix_entry': raw['matrix_entry'] if raw else None,
        'final_matrix_entry': final['matrix_entry'] if final else None,
        'original_branch_sequence': sequence,
        'threshold_decision': state,
        'mass_i': masses[2*(i-1):2*i] if masses else None,
        'mass_j': masses[2*(j-1):2*j] if masses else None,
        'native_input_invariant': native_value,
        'native_input_invariant_zero': rational(native_value['value']) == 0 if native_value else None,
        'native_input_equals_mass_i_real': rational(native_value['value']) == rational(masses[2*(i-1)]['value']) if native_value and masses else None}


def summarize():
    receipt = json.loads((HERE / 'receipt.json').read_bytes())
    review_raw = (HERE / 'root-stress-review.json').read_bytes()
    assert hashlib.sha256(review_raw).hexdigest() == receipt['root_review_sha256'] == '2954d77a73bf5a31ff97b4e8b4f52b39989b5a62be17fd0e301c8a157c2b2484'
    review = json.loads(review_raw)
    assert len(receipt['runs']) == len(review['enabled_traces']) == 6
    results, total_status = [], Counter()
    for run in receipt['runs']:
        assert run['trace'] in review['enabled_traces']
        result = {'name': run['name'], 'parser_returncode': run['returncode'], 'parser_log': run['log']}
        if run['returncode'] != 0:
            result['parser_failed'] = True
            results.append(result)
            continue
        raw = (HERE / run['output']).read_bytes()
        value = json.loads(raw)
        assert value['trace_sha256'] == run['trace']['sha256']
        assert value['accuracy_qualified'] is False and value['native_executions'] == 0
        calls, statuses = [], Counter()
        for call in value['calls']:
            identity = call['identity']
            records = [r for r in value['records'] if all(r[k] == identity[k] for k in KEYS)]
            statuses[str(call['native_status'])] += 1
            total_status[str(call['native_status'])] += 1
            calls.append({'identity': identity, 'native_status': call['native_status'],
                'boundaries': call['boundary_observed'], 'missing_boundary_scope': call.get('missing_boundary_scope'),
                'rational_analysis_unavailable': call.get('rational_analysis_unavailable'),
                'exact_mass_transport_match': call.get('matrix_masses_equal_dp_squared_inputs_exactly'),
                'critical_pair': pair_observation(records, call, call['critical_pair']),
                'photon_control': pair_observation(records, call, [3, 4]) if value['case'] == 'ew' else None,
                'conservation': call.get('conservation'),
                'actual_matrix_pair': call.get('matrix_shift_pair'),
                'symbolic_routed_values': call.get('exact_routing_on_captured_external_words'),
                'refusal_cause_established': False, 'accuracy_qualified': False})
        result.update(output_sha256=hashlib.sha256(raw).hexdigest(), record_count=len(value['records']),
                      selected_calls=len(calls), native_status_counts=dict(statuses), calls=calls)
        results.append(result)
    return {'schema': 'closed-stress-kinematic-input-status-summary-v1',
        'root_review_sha256': receipt['root_review_sha256'],
        'root_verified_run_closure_sha256': review['closure_sha256'],
        'decoder_source_index_sha256': receipt['source_index_sha256'],
        'parser_failures': sum(r['parser_returncode'] != 0 for r in results),
        'selected_call_status_counts': dict(total_status),
        'root_reported_global_flags': receipt['root_reported_global_flags'],
        'global_flags_scope': receipt['global_flags_scope'], 'results': results,
        'failed_numerical_gate_preserved': review['failed_numerical_gate_preserved'],
        'accuracy_qualified': False, 'refusal_cause_established': False,
        'tensor_outputs_read': False, 'new_native_executions': 0,
        'scope': 'Only six admitted stress traces of input construction, original branch decisions and native integer status. Input projection and subsequent refusal are observed; no counterfactual causal test or tensor-output comparison was performed.'}


if __name__ == '__main__':
    print(json.dumps(summarize(), sort_keys=True, indent=2, allow_nan=False))
