"""Summarize only the four already decoded ordinary traces; no native work."""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED = {'cms-point1-mode1-enabled': 5, 'cms-point1-mode4-enabled': 5,
            'ew-point1-mode1-enabled': 1, 'ew-point1-mode4-enabled': 1}


def fraction(value):
    return Fraction(int(value['numerator_hex'], 16), int(value['denominator_hex'], 16))


def summarize():
    receipt = json.loads((HERE / 'receipt.json').read_bytes())
    assert {run['name'] for run in receipt['runs']} == set(EXPECTED)
    results = []
    for run in receipt['runs']:
        assert run['returncode'] == 0
        raw = (HERE / run['output']).read_bytes()
        value = json.loads(raw)
        assert value['trace_sha256'] == run['trace_sha256']
        assert value['native_executions'] == 0 and value['accuracy_qualified'] is False
        assert len(value['calls']) == EXPECTED[run['name']]
        states, native, mismatches, observations = Counter(), Counter(), [], []
        for call in value['calls']:
            native[str(call['native_status'])] += 1
            for decision in call.get('threshold_decisions', []):
                for branch, state in decision['original_branches'].items():
                    states[branch + ':' + state] += 1
                    ideal = decision['exact_rational_predicates_not_fortran_replay'][branch]
                    if state not in ('taken', 'not taken (complete pair boundaries)') or (state == 'taken') != ideal:
                        mismatches.append({'id': call['identity'], 'pair': decision['pair'], 'branch': branch,
                                           'observed': state, 'ideal_rational_predicate': ideal})
            observations.append({'identity': call['identity'],
                'boundary_observed': call['boundary_observed'], 'native_status': call['native_status'],
                'mass_inputs_numerically_equal_exact_dp_squared_values': call.get('matrix_masses_equal_dp_squared_inputs_exactly'),
                'rational_analysis_unavailable': call.get('rational_analysis_unavailable'),
                'conservation': call.get('conservation'),
                'critical_routing': call.get('exact_routing_on_captured_external_words'),
                'actual_matrix_pair': call.get('matrix_shift_pair'),
                'critical_native_invariant': call.get('critical_native_invariant'),
                'critical_invariant_rational_differences': call.get('critical_invariant_rational_differences'),
                'critical_original_decisions': [d['original_branches'] for d in call.get('threshold_decisions', []) if d['critical_pair']]})
        results.append({'name': run['name'], 'output_sha256': hashlib.sha256(raw).hexdigest(),
            'record_count': len(value['records']), 'selected_calls': len(value['calls']),
            'native_status_counts': dict(native), 'all_observed_boundaries_complete': all(all(c['boundary_observed'].values()) for c in value['calls']),
            'exact_dp_squared_mass_transport_matches': sum(c.get('matrix_masses_equal_dp_squared_inputs_exactly') is True for c in value['calls']),
            'exact_routed_vector_matches': sum(all(fraction(v) == 0 for v in c['matrix_shift_pair']['difference_from_symbolic_routed_vector']) for c in value['calls']),
            'original_branch_states': dict(states), 'ideal_rational_vs_observed_disagreements': mismatches,
            'calls': observations})
    return {'schema': 'ordinary-current-kinematic-observations-summary-v1',
        'closed_trace_count': len(results), 'selected_tensor_calls': sum(r['selected_calls'] for r in results),
        'source_decoder_seal_sha256': receipt['source_index_sha256'],
        'root_verified_run_closure_sha256': receipt['root_verified_run_closure_sha256'],
        'abi_receipt': receipt['root_authenticated_abi'], 'results': results,
        'new_native_executions': 0, 'independent_accuracy_qualified': False,
        'threshold_or_model_changes': False,
        'scope': 'Ordinary point1 only; selected groups/modes1,4. Status0 observations and exact kinematic transport comparisons are not an independent integral or amplitude accuracy qualification. Original branch tags remain authoritative; ideal rational comparisons do not replay Fortran rounding.'}


if __name__ == '__main__':
    print(json.dumps(summarize(), indent=2, sort_keys=True, allow_nan=False))
