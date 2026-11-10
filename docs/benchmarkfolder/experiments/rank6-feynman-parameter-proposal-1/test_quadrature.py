"""Synthetic/exact-algebra checks only; no fixture quadrature or reducer call."""
import importlib.util
import gzip
import hashlib
import json
import math
from pathlib import Path
from fractions import Fraction
from decimal import Decimal, localcontext
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rank6_fp_proposal', HERE/'quadrature.py')
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)


class ExactRules(unittest.TestCase):
    def test_known_shifted_jacobi_polynomials(self):
        self.assertEqual(q.jacobi_polynomial(1, 0), [-1, 2])
        self.assertEqual(q.jacobi_polynomial(1, 1), [-1, 3])
        self.assertEqual(q.jacobi_polynomial(2, 1), [1, -8, 10])

    def test_root_brackets_are_exact_and_complete(self):
        for n, alpha in ((1, 0), (3, 0), (4, 2), (5, 4)):
            polynomial = q.jacobi_polynomial(n, alpha)
            brackets = q.isolate_roots(polynomial, bits=100)
            self.assertEqual(len(brackets), n)
            for lo, hi in brackets:
                self.assertLess(q.poly_at(polynomial, lo)*q.poly_at(polynomial, hi), 0)
                self.assertLessEqual(hi-lo, Fraction(1, 1 << 100))

    def test_positive_gaussian_weights_exact_moments(self):
        # These are beta-function identities of polynomials. No Feynman
        # denominator, captured coefficient or fixture integration is used.
        for n in (1, 2, 3, 4, 6):
            for alpha in range(5):
                rule = q.gaussian_rule(n, alpha)
                self.assertTrue(all(0 < r['node'] < 1 and r['weight'] > 0 for r in rule))
                for power in range(2*n):
                    expected = Fraction(math.factorial(power)*math.factorial(alpha),
                                        math.factorial(power+alpha+1))
                    observed = math.fsum(r['weight']*r['node']**power for r in rule)
                    self.assertLessEqual(abs(observed-float(expected)), 5e-14*float(expected))


class TensorFormula(unittest.TestCase):
    def terms(self, name):
        return {(r, p): c for c, r, p in q.tensor_terms(q.TARGETS[name])}

    def test_control_formulas(self):
        self.assertEqual(self.terms('scalar'), {((0, 0, 0, 0), 4): 6})
        self.assertEqual(self.terms('odd_time'), {((1, 0, 0, 0), 4): 6})
        self.assertEqual(self.terms('time_time'), {((2, 0, 0, 0), 4): 6, ((0, 0, 0, 0), 3): -1})
        self.assertEqual(self.terms('space_space'), {((0, 2, 0, 0), 4): 6, ((0, 0, 0, 0), 3): 1})

    def test_tensor_domain_is_explicit(self):
        for powers in ((7, 0, 0, 0), (-1, 0, 0, 0), (True, 0, 0, 0), (1.0, 0, 0, 0), (1, 1)):
            with self.assertRaises(ValueError):
                q.tensor_terms(powers)

    def test_disputed_formulas(self):
        self.assertEqual(self.terms('index55'), {((1, 1, 1, 1), 4): 6})
        self.assertEqual(self.terms('index111'), {((1, 1, 1, 2), 4): 6, ((1, 1, 1, 0), 3): 1})
        self.assertEqual(self.terms('index161'), {((3, 1, 1, 1), 4): 6, ((1, 1, 1, 1), 3): -3})

    def test_point_integrand_equal_mass_zero_shift_analytic_control(self):
        synthetic = {'shifts': [[0]*4]*6, 'invariants': [[0]*6]*6,
                     'masses_squared': [[16, 0]]*6}
        value = q.point_integrand((1.0, 0, 0, 0, 0, 0), synthetic)
        self.assertEqual(value['scalar'], 6/16**4)
        self.assertEqual(value['time_time'], -1/16**3)
        self.assertEqual(value['space_space'], 1/16**3)
        for name in ('odd_time', 'index55', 'index111', 'index161'):
            self.assertEqual(value[name], 0)

    def test_decimal_point_analytic_control(self):
        synthetic = {'shifts': [[0]*4]*6, 'invariants': [[0]*6]*6,
                     'masses_squared': [[16, 0]]*6}
        with localcontext() as context:
            context.prec = 80
            value = q.decimal_point_integrand((Decimal(1),)+(Decimal(0),)*5, synthetic)
            self.assertEqual(value['scalar'], (Decimal(6)/16**4, Decimal(0)))
            self.assertEqual(value['time_time'], (-Decimal(1)/16**3, Decimal(0)))
            self.assertEqual(value['space_space'], (Decimal(1)/16**3, Decimal(0)))

    def test_stick_breaking_constant_measure(self):
        for nodes in ((0, 0, 0, 0, 0), (1, 1, 1, 1, 1), (.5,)*5):
            xs = q.stick_breaking(nodes)
            self.assertEqual(math.fsum(xs), 1)
            self.assertTrue(all(0 <= x <= 1 for x in xs))


class BoundsAndAdmission(unittest.TestCase):
    def test_deadline_covers_root_preparation_and_cache_hits(self):
        expired = time.monotonic()-1
        with self.assertRaises(TimeoutError):
            q.isolate_roots(q.jacobi_polynomial(5, 2), deadline=expired)
        q.gaussian_rule(2, 1)
        with self.assertRaises(TimeoutError):
            q.gaussian_rule(2, 1, deadline=expired)

    def test_rational_majorant_and_analytic_bounds(self):
        self.assertGreater(Fraction(118031, 1000000)**2, Fraction(18778, 1161**2))
        for n in (4, 6, 8, 10, 18, 20):
            for powers in q.TARGETS.values():
                bounds = q.exact_bounds(n, powers)
                self.assertEqual(bounds['K'], n-4)
                self.assertGreater(Fraction(bounds['absolute_analytic_error']), 0)
                self.assertGreater(Fraction(bounds['absolute_integral_envelope']), 0)
        for name, maximum in (('index55', Fraction(936, 10**24)),
                              ('index111', Fraction(603, 10**23)),
                              ('index161', Fraction(287, 10**22))):
            self.assertLess(Fraction(q.exact_bounds(20, q.TARGETS[name])['absolute_analytic_error']), maximum)

    def test_fixture_is_mutually_consistent_exact_4d(self):
        self.assertEqual(q.fixture()['id'], 108)

    def test_fixture_rejects_changed_bytes_even_for_consistent_geometry(self):
        with tempfile.TemporaryDirectory() as temporary:
            original = q.FIXTURE
            changed = Path(temporary)/'changed.json'
            changed.write_bytes(original.read_bytes()+b'\n')
            q.FIXTURE = changed
            try:
                with self.assertRaisesRegex(ValueError, 'fixture bytes differ'):
                    q.fixture()
            finally:
                q.FIXTURE = original

    def test_proposal_does_not_generate_gaussian_rules_or_integrate(self):
        original = q.gaussian_rule
        q.gaussian_rule = lambda *args: self.fail('proposal evaluated a Gaussian rule')
        try:
            result = q.proposal()
            self.assertEqual(result['native_reducer_calls'], 0)
            self.assertIn('source-only', result['status'])
            self.assertIn('unqualified', result['roundoff_status'])
        finally:
            q.gaussian_rule = original

    def test_actual_evaluation_requires_explicit_admission(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'result.json'
            with self.assertRaises(SystemExit) as caught:
                q.main(['run', '--output', str(path)])
            self.assertEqual(caught.exception.code, 2)
            self.assertFalse(path.exists())

    def test_owned_derivation_matches_formulas_and_bounds(self):
        d = json.loads((HERE/'derivation.json').read_text())
        self.assertEqual(d['fixture_sha256'], hashlib.sha256(q.FIXTURE.read_bytes()).hexdigest())
        names = {'scalar': 'scalar', 'odd_time': 'q0', 'time_time': 'q0_squared',
                 'space_space': 'q1_squared', 'index55': 'target_55',
                 'index111': 'target_111', 'index161': 'target_161'}
        frozen = json.loads(gzip.decompress((q.COMMON/'finite-audit-1/rows.json.gz').read_bytes()))
        for name, label in names.items():
            terms = d['tensor_terms'][label]
            self.assertEqual(terms['coefficient_index'], q.INDICES[name])
            self.assertEqual(tuple(terms['powers']), q.TARGETS[name])
            self.assertEqual(tuple(frozen['monomials'][q.INDICES[name]]), q.TARGETS[name])
            self.assertEqual([(Fraction(c), tuple(a), p) for c, a, p in terms['terms']],
                             q.tensor_terms(q.TARGETS[name]))
        for entry in d['analytic_quadrature_bounds']:
            for name, label in names.items():
                bound = q.exact_bounds(entry['nodes_per_dimension'], q.TARGETS[name])
                self.assertEqual(bound['absolute_analytic_error'],
                                 entry['absolute_complex_error_bound'][label]['exact_rational'])


if __name__ == '__main__':
    unittest.main()
