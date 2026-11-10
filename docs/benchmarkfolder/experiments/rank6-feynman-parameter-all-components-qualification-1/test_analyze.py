"""Pure bookkeeping/zero/guard tests; no integration or native calls."""
import copy
from decimal import Decimal, localcontext
import importlib.util
import gzip
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('full210_offline', HERE/'analyze.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class OfflineBookkeeping(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parser, cls.core, cls.vector, cls.snapshot = a.dependencies()

    def test_native_historical420_part_count(self):
        by = {row['action']: row for row in self.snapshot['rows']}
        counts = {'3e-6': 0, '1e-8': 0}
        zero = 0
        with localcontext() as context:
            context.prec = 180
            for i in range(210):
                for offset in (0, 2):
                    x, y = (self.parser.binary128(by[action]['values'][i][0][offset:offset+2]) for action in (3007, 3013))
                    zero += x.is_zero() or y.is_zero()
                    residual = self.parser.symmetric(x, y)
                    for threshold in counts:
                        counts[threshold] += residual > Decimal(threshold)
        self.assertEqual(counts, {'3e-6': 122, '1e-8': 328})
        self.assertEqual(zero, 0)

    def test_zero_diagnostics_do_not_relax_symmetric_failure(self):
        diagnostic = a.zero_diagnostic(Decimal('1e-30'), Decimal(0), Decimal(1))
        self.assertTrue(diagnostic['retained_reference_exactly_zero'])
        self.assertFalse(diagnostic['estimate_exactly_zero'])
        self.assertEqual(Decimal(diagnostic['rank_scaled_absolute_difference']), Decimal('1e-30'))
        self.assertFalse(self.parser.distance(Decimal('1e-30'), Decimal(0))['estimate_agrees_at_3e6'])
        self.assertIsNone(a.zero_diagnostic(Decimal(0), Decimal(0), Decimal(0))['rank_scaled_absolute_difference'])

    def test_three_way_conditional_counts_remain_separate(self):
        parts = []
        for i, (x, e) in enumerate((('1', '1e-12'), ('1.01', '1e-12'), ('1', '1'))):
            x, y, e = Decimal(x), Decimal(1), Decimal(e)
            item = {'retained_value': str(y), **self.parser.distance(x, y),
                    'analytic_error_only_interval': self.parser.analytic_interval(x, y, e)}
            parts.append((i, 'real', {'TIRust_QP': item}))
        summary = a.summarize_parts(parts, 'TIRust_QP')
        for suffix in ('3e6', '1e8'):
            self.assertEqual(summary['analytic_error_only_agree_'+suffix], 1)
            self.assertEqual(summary['analytic_error_only_disagree_'+suffix], 1)
            self.assertEqual(summary['analytic_error_only_inconclusive_'+suffix], 1)
        self.assertEqual(summary['fail_3e6'], 1)

    def test_identity_guard_precedes_component_analysis(self):
        invalid = {'fixture_sha256': 'wrong', 'monomials': []}
        with self.assertRaisesRegex(ValueError, 'fixture identity'):
            a.analyze(invalid)

    def test_terminal_failure_preserves_completed_raw_order(self):
        result = json.loads(gzip.decompress((HERE/'data/vector-result.json.gz').read_bytes()))
        result['orders'] = result['orders'][:1]
        result['status'] = 'failed or timed out; earlier completed orders retained'
        result['failure'] = {'type': 'SyntheticTest', 'message': 'comparison failed after raw persistence'}
        result['orders'][0]['saved_selected_comparison'] = {'status': 'pending'}
        output = a.analyze(result)
        self.assertEqual([row['n'] for row in output['orders']], [8])
        self.assertEqual(output['unavailable_proposed_orders'], [16, 18])
        self.assertEqual(output['failure'], result['failure'])
        self.assertFalse(output['orders'][0]['saved_scalar_control']['worker_comparison_available'])

    def test_tensor_and_bound_tampering_rejected(self):
        for field in ('index', 'bound'):
            result = json.loads(gzip.decompress((HERE/'data/vector-result.json.gz').read_bytes()))
            if field == 'index':
                result['orders'][0]['tensor_values'][0]['index'] = 1
            else:
                result['orders'][0]['tensor_values'][0]['analytic_error_bound_excludes_roundoff']['absolute_analytic_error'] = '0'
            with self.assertRaises(ValueError):
                a.analyze(result)


if __name__ == '__main__':
    unittest.main()
