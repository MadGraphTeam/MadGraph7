"""Independent Fraction checks of the algorithm SPEC; no Fortran/native execution."""
from fractions import Fraction
from pathlib import Path
import copy
import json
import random
import re
import unittest

import derive_witnesses
import reference

HERE=Path(__file__).resolve().parent


def value(bits,width):
    # Independent semantic oracle: do not use reference.decode/normalize.
    if type(bits) is not int or not 0<=bits<2**width:
        raise ValueError('bad word')
    fraction_width,exponent_width,bias={64:(52,11,1023),128:(112,15,16383)}[width]
    sign=-1 if bits//2**(width-1) else 1
    stored=(bits//2**fraction_width)%(2**exponent_width)
    fraction=bits%2**fraction_width
    if stored==2**exponent_width-1:
        raise ValueError('not finite')
    if stored:
        return sign*Fraction(2)**(stored-bias)*(1+Fraction(fraction,2**fraction_width))
    return sign*Fraction(fraction,2**fraction_width)*Fraction(2)**(1-bias)


def oracle(width,external,coefficients,q):
    for mu in range(4):
        if value(q[mu][1],width)!=0:
            return False
        if any(value(e[mu][1],width)!=0 for e in external):
            return False
        if sum((c*value(e[mu][0],width) for c,e in zip(coefficients,external)),Fraction(0))!=value(q[mu][0],width):
            return False
    return True


def word(v,width):
    bits=derive_witnesses.exact_word(Fraction(v),width)
    if value(bits,width)!=v:
        raise ValueError('Fixture encoding differs from independent decoder')
    return bits


def vectors(numbers,width):
    return [[[word(v,width),0] for v in row] for row in numbers]


class ExactGeometryTests(unittest.TestCase):
    def assert_true(self,width,external,coefficients,q):
        self.assertTrue(oracle(width,external,coefficients,q))
        got=reference.check(width,width,external,coefficients,q)
        self.assertTrue(got['ok'],got)
        self.assertEqual(got['error_exp'],[reference.ZERO_BOUND]*4)
        self.assertFalse(got['ownership_qualified'])
        self.assertFalse(got['Fortran_executed'])
        return got

    def test_four_retained_conditional_witnesses(self):
        frozen=json.loads((HERE/'witnesses.json').read_text())
        self.assertEqual(derive_witnesses.build(),frozen)
        expected=[[8,54,0,53],[8,54,0,113],[8,54,0,8],[8,54,0,110]]
        for case,bits in zip(frozen['cases'],expected):
            with self.subTest(point=case['point'],width=case['width']):
                self.assertFalse(case['owned_snapshot_observed'])
                self.assertFalse(case['q_component_ABI_observed'])
                result=self.assert_true(case['width'],case['inferred_external_P_bits'],
                                        case['coefficients_relative_to_repaired_P'],
                                        case['reconstructed_q_bits'])
                self.assertEqual(result['absolute_sum_bits'],bits)
                # Current source coefficients already contain the external signs.
                wrong=reference.check(case['width'],case['width'],case['captured_external_W_bits'],
                                      case['coefficients_relative_to_repaired_P'],case['reconstructed_q_bits'])
                self.assertFalse(wrong['ok'])

    def test_random_exact_cases_against_independent_fraction_oracle(self):
        rng=random.Random(0xE0A1)
        for width in (64,128):
            for trial in range(250):
                n=rng.randrange(1,9)
                exponents=[rng.randrange(-18,19) for _ in range(4)]
                numbers=[[Fraction(rng.randrange(-512,513))*Fraction(2)**exponents[mu]
                          for mu in range(4)] for _ in range(n)]
                coefficients=[rng.randrange(-8,9) for _ in range(n)]
                qvalues=[sum((coefficients[e]*numbers[e][mu] for e in range(n)),Fraction(0))
                         for mu in range(4)]
                external=vectors(numbers,width)
                q=vectors([qvalues],width)[0]
                self.assert_true(width,external,coefficients,q)
                changed=copy.deepcopy(q)
                changed[0][0]=word(qvalues[0]+Fraction(2)**(exponents[0]-1),width)
                self.assertFalse(oracle(width,external,coefficients,changed))
                self.assertFalse(reference.check(width,width,external,coefficients,changed)['ok'])

    def test_random_finite_words_have_no_false_acceptance(self):
        rng=random.Random(0x12864)
        for width in (64,128):
            for _ in range(300):
                nf,ne={64:(52,11),128:(112,15)}[width]
                def finite():
                    return (rng.randrange(2)<<(width-1))|(rng.randrange(2**ne-1)<<nf)|rng.getrandbits(nf)
                external=[[[finite(),0] for _ in range(4)] for _ in range(3)]
                q=[[finite(),0] for _ in range(4)]
                coefficients=[rng.randrange(-4,5) for _ in range(3)]
                got=reference.check(width,width,external,coefficients,q)
                if got['ok']:
                    self.assertTrue(oracle(width,external,coefficients,q))

    def test_signed_zero_all_real_and_imaginary_positions(self):
        for width in (64,128):
            negative_zero=1<<(width-1)
            external=[[[negative_zero,0],[0,negative_zero],[negative_zero,negative_zero],[0,0]]]
            q=[[0,negative_zero],[negative_zero,0],[0,0],[negative_zero,negative_zero]]
            self.assert_true(width,external,[-(1<<31)],q)

    def test_extreme_and_subnormal_equalities(self):
        for width in (64,128):
            nf,ne={64:(52,11),128:(112,15)}[width]
            maximum=((2**ne-2)<<nf)|(2**nf-1)
            for bits in (1,2**nf-1,2**nf,maximum,maximum|(1<<(width-1))):
                external=[[[bits,0] for _ in range(4)]]
                self.assert_true(width,external,[1],copy.deepcopy(external[0]))

    def test_minimum_default_integer_and_power_two_coefficient(self):
        for width in (64,128):
            external=vectors([[1,-1,Fraction(1,2),0]],width)
            for coefficient in (-(1<<31),1<<30):
                q=vectors([[coefficient,-coefficient,Fraction(coefficient,2),0]],width)[0]
                self.assert_true(width,external,[coefficient],q)

    def test_mathematically_equal_cancellation_does_not_bypass_product_guard(self):
        width=128
        number=Fraction(2**113-1,2**112)
        external=vectors([[number]*4,[number]*4],width)
        q=vectors([[0]*4],width)[0]
        coefficients=[(1<<31)-1,-((1<<31)-1)]
        self.assertTrue(oracle(width,external,coefficients,q))
        got=reference.check(width,width,external,coefficients,q)
        self.assertEqual(got['reason'],'coefficient-product-capacity')

    def test_exponent_gap_and_full_format_range_fail_closed(self):
        for width in (64,128):
            for exponent in (127,200):
                high=Fraction(2)**exponent
                external=vectors([[high]*4,[high]*4,[1]*4,[1]*4],width)
                q=vectors([[0]*4],width)[0]
                self.assertTrue(oracle(width,external,[1,-1,1,-1],q))
                self.assertEqual(reference.check(width,width,external,[1,-1,1,-1],q)['reason'],
                                 'alignment-capacity')
            nf,ne={64:(52,11),128:(112,15)}[width]
            maximum=((2**ne-2)<<nf)|(2**nf-1)
            external=[[[maximum,0] for _ in range(4)],[[maximum,0] for _ in range(4)],
                      [[1,0] for _ in range(4)]]
            q=[[1,0] for _ in range(4)]
            self.assertTrue(oracle(width,external,[1,-1,1],q))
            self.assertFalse(reference.check(width,width,external,[1,-1,1],q)['ok'])

    def test_alignment_magnitude_and_absolute_sum_guards(self):
        width=128
        q=vectors([[0]*4],width)[0]
        for high,reason in ((Fraction(2)**126,'absolute-sum-capacity'),
                            (Fraction(2**113-1)*2**15,'alignment-capacity')):
            external=vectors([[high]*4,[high]*4,[1]*4,[1]*4],width)
            self.assertTrue(oracle(width,external,[1,-1,1,-1],q))
            self.assertEqual(reference.check(width,width,external,[1,-1,1,-1],q)['reason'],reason)

    def test_nonfinite_and_nonreal_words_are_rejected_even_when_coefficient_zero(self):
        for width in (64,128):
            nf,ne={64:(52,11),128:(112,15)}[width]
            infinity=(2**ne-1)<<nf
            for bad in (infinity,infinity|(1<<(width-1)),infinity|1,infinity|(1<<(nf-1))):
                for location in ('external-real','external-imag','q-real','q-imag'):
                    external=vectors([[0]*4],width);q=vectors([[0]*4],width)[0]
                    if location.startswith('external'):
                        external[0][0][int(location.endswith('imag'))]=bad
                    else:
                        q[0][int(location.endswith('imag'))]=bad
                    self.assertFalse(reference.check(width,width,external,[0],q)['ok'])
            external=vectors([[0]*4],width);q=vectors([[0]*4],width)[0]
            external[0][0][1]=1
            self.assertEqual(reference.check(width,width,external,[0],q)['reason'],'nonreal')

    def test_malformed_shape_kind_and_words(self):
        external=vectors([[1]*4],64);q=copy.deepcopy(external[0])
        bad=[(0,0,external,[1],q),(True,True,external,[1],q),(64,128,external,[1],q),
             (64,64,external,[True],q),(64,64,external,[1<<31],q),
             (64,64,external,[-(1<<31)-1],q),(64,64,[],[],q),
             (64,64,external,[1],q[:3]),(64,64,external[:1]*65,[1]*65,q)]
        for args in bad:
            self.assertFalse(reference.check(*args)['ok'])
        for malformed in (-1,1<<64,True,'0000000000000000'):
            e=copy.deepcopy(external);e[0][0][0]=malformed
            self.assertFalse(reference.check(64,64,e,[1],q)['ok'])

    def test_Fortran_is_unexecuted_integer_only_source(self):
        source=(HERE/'exact_geometry.f90').read_text()
        code='\n'.join(line.split('!',1)[0] for line in source.splitlines())
        self.assertNotRegex(code,re.compile(r'\b(?:REAL|COMPLEX|DOUBLE|IEEE_SET|TRANSFER|ALLOCATE|BIND)\b',re.I))
        self.assertNotRegex(code,re.compile(r'\b(?:DBLE|CMPLX|SQRT|EXP|LOG)\s*\(',re.I))
        self.assertLessEqual(max(map(len,source.splitlines())),132)


if __name__=='__main__':
    unittest.main()
