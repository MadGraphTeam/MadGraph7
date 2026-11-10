"""Pure parser tests using retained numbers as synthetic estimate values."""
import contextlib
import copy
from decimal import Decimal
from fractions import Fraction
import importlib.util
import io
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rank6_fp_compare', HERE/'compare.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class CompareTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = c.retained_snapshot()

    def synthetic_result(self):
        result = json.loads((HERE.parent/'rank6-feynman-parameter-proposal-1/proposal.json').read_text())
        result.update(status='completed estimates; synthetic parser test only', engine='decimal')
        values = next(row for row in self.snapshot['rows'] if row['action'] == 3007)['values']
        targets = {}
        for name, (index, _) in c.TARGETS.items():
            targets[name] = {'real': str(c.binary128(values[index][0][:2])),
                             'imaginary': str(c.binary128(values[index][0][2:])),
                             'bounds': {'absolute_analytic_error': c.analytic_bound(4, name)}}
        result['orders'] = [{'n': 4, 'nodes': 4**5, 'results': targets}]
        return result

    def test_binary128_exact_and_signed_zero(self):
        self.assertEqual(c.binary128(['0000000000000000', '3FFF000000000000']), Decimal(1))
        self.assertEqual(c.binary128(['0000000000000000', 'BFFF000000000000']), Decimal(-1))
        self.assertTrue(c.binary128(['0000000000000000', '8000000000000000']).is_signed())
        with self.assertRaises(ValueError):
            c.binary128(['0000000000000000', '7FFF000000000000'])

    def test_binary128_subnormal_and_hex_contract(self):
        smallest = c.binary128(['0000000000000001', '0000000000000000'])
        self.assertEqual(Fraction(*smallest.as_integer_ratio()), Fraction(1, 2**16494))
        for words in (['0000000000000000', '+FFF000000000000'],
                      ['0000000000000000', 'GGFF000000000000']):
            with self.assertRaises(ValueError):
                c.binary128(words)

    def test_json_float_roundtrip_preserves_binary64(self):
        self.assertEqual(c.estimate_number(0.1, 'float'), Decimal.from_float(0.1))
        for value, engine in ((True, 'float'), ('0.1', 'float'), (1.0, 'decimal'), ('NaN', 'decimal')):
            with self.assertRaises(ValueError):
                c.estimate_number(value, engine)

    def test_symmetric_zero_and_opposite_sign_limits(self):
        self.assertEqual(c.symmetric(Decimal(0), Decimal(0)), 0)
        self.assertTrue(c.symmetric(Decimal(1), Decimal(-1)).is_infinite())

    def test_analytic_interval_keeps_roundoff_exclusion(self):
        interval = c.analytic_interval(Decimal(1), Decimal(1), Decimal('1e-8'))
        self.assertTrue(interval['quadrature_error_only_agreement_at_3e6'])
        self.assertTrue(interval['excludes_finite_precision_node_weight_arithmetic_error'])
        failure = c.analytic_interval(Decimal('1.01'), Decimal(1), Decimal('1e-8'))
        self.assertTrue(failure['quadrature_error_only_disagreement_at_3e6'])
        self.assertEqual(Decimal(c.analytic_interval(Decimal(0), Decimal(0), Decimal(0))['symmetric_residual_upper']), 0)
        self.assertTrue(Decimal(c.analytic_interval(Decimal(1), Decimal(-1), Decimal(0))['symmetric_residual_lower']).is_infinite())

    def test_exact_retained_values_are_not_relabelled_truth(self):
        output = c.analyze(self.synthetic_result(), self.snapshot)
        self.assertFalse(output['finite_truth_assigned'])
        order = output['orders'][0]
        self.assertTrue(order['lowrank_convention_checks_by_estimate']['TIRust_QP']['all_four_controls_agree_at_1e8'])
        for item in order['components'].values():
            for part in item['parts'].values():
                self.assertEqual(Decimal(part['TIRust_QP']['absolute_difference']), 0)
        self.assertGreater(Decimal(order['components']['index161']['parts']['imaginary']['COLLIER']['symmetric_residual']), Decimal('0.0013'))

    def test_running_result_rejected(self):
        result = self.synthetic_result()
        result['status'] = 'running; floating roundoff unqualified'
        with self.assertRaises(ValueError):
            c.analyze(result, self.snapshot)

    def test_tampered_component_or_bound_rejected(self):
        for case in ('powers', 'bound'):
            result = self.synthetic_result()
            if case == 'powers':
                result['targets']['index161']['powers'] = [6, 0, 0, 0]
            else:
                result['orders'][0]['results']['index161']['bounds']['absolute_analytic_error'] = '0'
            with self.assertRaises(ValueError):
                c.analyze(result, self.snapshot)

    def test_duplicate_order_rejected(self):
        result = self.synthetic_result()
        result['orders'].append(copy.deepcopy(result['orders'][0]))
        with self.assertRaises(ValueError):
            c.analyze(result, self.snapshot)

    def test_unreviewed_order_fails_explicitly(self):
        with self.assertRaisesRegex(ValueError, 'no reviewed bound'):
            c.analytic_bound(14, 'index161')

    def test_closed_gate_precedes_any_result_read(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                c.main(['--result', '/nonexistent/active-result.json', '--output', '/nonexistent/ignored.json'])
        self.assertEqual(raised.exception.code, 2)

    def test_arithmetic_join_uses_only_common_orders(self):
        decimal = self.synthetic_result()
        floating = copy.deepcopy(decimal)
        floating['engine'] = 'float'
        for item in floating['orders'][0]['results'].values():
            item['real'], item['imaginary'] = float(item['real']), float(item['imaginary'])
        output = c.compare_arithmetic(floating, decimal)
        self.assertEqual([entry['n'] for entry in output['orders']], [4])
        self.assertIn('no rigorous roundoff bound', output['status'])
        for item in output['orders'][0]['components'].values():
            for part in item.values():
                self.assertLess(Decimal(part['symmetric_residual']), Decimal('1e-15'))


if __name__ == '__main__':
    unittest.main()
