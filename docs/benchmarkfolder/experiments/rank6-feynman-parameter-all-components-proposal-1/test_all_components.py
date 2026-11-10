"""Analytic synthetic controls only; never integrate the actual fixture108."""
import contextlib
from decimal import Decimal
from fractions import Fraction
import importlib.util
import io
import itertools
import json
import math
from pathlib import Path
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('rank6_all_components', HERE/'all_components.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def double_factorial_odd(n):
    return math.prod(range(1, n+1, 2))


def analytic_components(monomials, mass, shift_moment):
    """Independent Wick-pair count using even loop powers and binomials."""
    expected = []
    for powers in monomials:
        value = 0j
        for loop_powers in itertools.product(*(range(0, power+1, 2) for power in powers)):
            pairs = sum(loop_powers)//2
            coefficient = Fraction((-1)**(loop_powers[0]//2) * math.factorial(3-pairs), 2**pairs)
            for power, loops in zip(powers, loop_powers):
                coefficient *= math.comb(power, loops)*double_factorial_odd(loops-1)
            residual = tuple(p-l for p, l in zip(powers, loop_powers))
            value += float(coefficient)*complex(shift_moment(residual))*mass**(-(4-pairs))
        expected.append((value.real, value.imag))
    return expected


class AnalyticSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.core = a.load_core()
        cls.monomials = a.load_layout(cls.core)
        cls.layout = a.plan(cls.core, cls.monomials)
        cls.np = a.numpy()

    def selected(self, shifts, mass=(16, -.125)):
        return {'shifts': shifts, 'invariants': [[0]*6]*6, 'masses_squared': [list(mass)]*6}

    def integrate_synthetic(self, selected, n=2, chunk=4096):
        return a.integrate(n, selected, self.core, self.layout, time.monotonic()+30, chunk)

    def assert_close(self, observed, expected, relative=2e-14, absolute=1e-24):
        np = self.np
        np.testing.assert_allclose(observed, np.array(expected, dtype=np.longdouble), rtol=relative, atol=absolute)

    def test_layout_counts_and_exact_sparse_map(self):
        self.assertEqual(len(self.layout['primitives']), 296)
        self.assertEqual(sum(len(t) for t in self.layout['transform']), 660)
        self.assertTrue(all(c.denominator in (1, 2, 4, 8) for terms in self.layout['transform'] for _, c in terms))
        index = self.monomials.index((6, 0, 0, 0))
        terms = {(self.layout['primitives'][p][0], self.layout['primitives'][p][1]): c for p, c in self.layout['transform'][index]}
        self.assertEqual(terms, {(4, (6, 0, 0, 0)): 6, (3, (4, 0, 0, 0)): -15,
                                (2, (2, 0, 0, 0)): Fraction(45, 4), (1, (0, 0, 0, 0)): Fraction(-15, 8)})

    def test_zero_shift_odd_zeros_and_all_even_trace_identities(self):
        _, actual, _ = self.integrate_synthetic(self.selected([[0]*4]*6), chunk=1)
        expected = analytic_components(self.monomials, complex(16, -.125), lambda powers: Fraction(1, 120) if not sum(powers) else 0)
        self.assert_close(actual, expected)
        for powers, value in zip(self.monomials, actual):
            if any(power % 2 for power in powers):
                self.assertTrue(self.np.all(value == 0))

    def test_nonzero_constant_shift_all296_primitives_and210_components(self):
        p = (2, 3, 5, 7); mass = complex(16, -.125)
        primitives, actual, _ = self.integrate_synthetic(self.selected([list(p)]*6))
        expected_primitives = []
        for power, powers in self.layout['primitives']:
            value = math.prod(v**r for v, r in zip(p, powers))*mass**(-power)/120
            expected_primitives.append((value.real, value.imag))
        self.assert_close(primitives, expected_primitives)
        expected = analytic_components(self.monomials, mass, lambda powers: Fraction(math.prod(v**r for v, r in zip(p, powers)), 120))
        self.assert_close(actual, expected, relative=1e-13, absolute=1e-20)

    def test_nonconstant_null_shift_dirichlet_moments_and_partial_chunks(self):
        # p_i=lambda_i*(1,1,0,0) is a genuine null polygon: every S_ij=0,
        # F is constant, while P varies. h_r are exact complete homogeneous
        # polynomials, giving integral L^r = r!*h_r/(r+5)!.
        h = [1]+[0]*6
        for value in range(6):
            for degree in range(1, 7):
                h[degree] += value*h[degree-1]
        moments = [Fraction(math.factorial(r)*h[r], math.factorial(r+5)) for r in range(7)]
        selected = self.selected([[i, i, 0, 0] for i in range(6)])
        _, actual37, meta = self.integrate_synthetic(selected, n=4, chunk=37)
        _, actual4096, _ = self.integrate_synthetic(selected, n=4, chunk=4096)
        self.assertEqual(meta['chunks'], 4*math.ceil(4**4/37))
        self.assert_close(actual37, actual4096, relative=2e-17, absolute=1e-25)
        expected = analytic_components(self.monomials, complex(16, -.125),
                                       lambda powers: 0 if powers[2] or powers[3] else moments[sum(powers)])
        self.assert_close(actual37, expected, relative=1e-13, absolute=1e-22)

    def test_spatial_permutation_and_momentum_reflection(self):
        _, original, _ = self.integrate_synthetic(self.selected([[2, 3, 5, 7]]*6))
        _, swapped, _ = self.integrate_synthetic(self.selected([[2, 7, 5, 3]]*6))
        _, reflected, _ = self.integrate_synthetic(self.selected([[-2, -3, -5, -7]]*6))
        lookup = {powers: i for i, powers in enumerate(self.monomials)}
        expected_swap = [original[lookup[(powers[0], powers[3], powers[2], powers[1])]] for powers in self.monomials]
        self.assert_close(swapped, expected_swap, relative=2e-17, absolute=1e-22)
        self.assertTrue(self.np.array_equal(reflected, self.np.array([(-1)**sum(p)*v for p, v in zip(self.monomials, original)])))

    def test_saved_selected_transport_only_no_fixture_integration(self):
        row, source = a.saved_selected(8)
        values = self.np.zeros((210, 2), dtype=self.np.longdouble)
        for name, target in source['targets'].items():
            values[target['index']] = [self.np.longdouble(str(Decimal.from_float(row['results'][name][part]))) for part in ('real', 'imaginary')]
        comparison = a.compare_saved(8, values, self.core)
        self.assertEqual(len(comparison['parts']), 14)
        self.assertTrue(all(part['agrees_at_1e12'] for part in comparison['parts'].values()))

    def test_extended_precision_and_bounded_work(self):
        np = self.np
        self.assertGreaterEqual(np.finfo(np.longdouble).nmant+1, 64)
        self.assertLessEqual(210*8192*np.dtype(np.longdouble).itemsize, 28_000_000)
        self.assertTrue(all(a.os.environ[name] == '1' for name in a.THREAD_VARS))

    def test_deadline_and_chunk_domain(self):
        selected = self.selected([[0]*4]*6)
        with self.assertRaises(TimeoutError):
            a.integrate(4, selected, self.core, self.layout, time.monotonic()-1)
        with self.assertRaises(ValueError):
            self.integrate_synthetic(selected, chunk=8193)

    def test_source_proposal_generates_no_gaussian_nodes_or_integrals(self):
        original_core, original_numpy = a.load_core, a.numpy
        saved_rule = self.core.gaussian_rule
        self.core.gaussian_rule = lambda *args: self.fail('source proposal generated a rule')
        a.load_core = lambda: self.core
        a.numpy = lambda: self.fail('source proposal imported numeric integration')
        try:
            proposal = a.proposal()
            self.assertEqual(proposal['components'], 210)
            self.assertEqual(proposal['primitive_moments'], 296)
        finally:
            a.load_core, a.numpy, self.core.gaussian_rule = original_core, original_numpy, saved_rule

    def test_cli_admission_before_any_fixture_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'blocked.json'
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    a.main(['run', '--output', str(path)])
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(path.exists())

    def test_completed_raw_order_survives_postprocessing_failure(self):
        # The actual integrator is replaced by zero arrays. This exercises
        # controller persistence only, not a physical parameter evaluation.
        integrate, compare = a.integrate, a.compare_saved
        a.integrate = lambda *args: (self.np.zeros((296, 2), dtype=self.np.longdouble),
                                     self.np.zeros((210, 2), dtype=self.np.longdouble),
                                     {'chunks': 0, 'chunk_size': 4096, 'rules': []})
        a.compare_saved = lambda *args: (_ for _ in ()).throw(ValueError('synthetic comparison failure'))
        try:
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/'mocked-controller.json'
                self.assertEqual(a.main(['run', '--orders', '8', '--output', str(path), '--admit-physics']), 1)
                result = json.loads(path.read_text())
                self.assertEqual(len(result['orders']), 1)
                self.assertEqual(len(result['orders'][0]['primitive_values']), 296)
                self.assertEqual(len(result['orders'][0]['tensor_values']), 210)
                self.assertEqual(result['failure']['message'], 'synthetic comparison failure')
        finally:
            a.integrate, a.compare_saved = integrate, compare


if __name__ == '__main__':
    unittest.main()
