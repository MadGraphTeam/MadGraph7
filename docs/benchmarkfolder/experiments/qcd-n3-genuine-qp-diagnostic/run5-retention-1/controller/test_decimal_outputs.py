"""Pinned formatter-contract tests; no native formatter or library is invoked."""
from fractions import Fraction
import unittest
import analyze
import test_protocol as fixture


def record(text):
    return text.encode('ascii').ljust(64,b' ')


class DecimalOutputTests(unittest.TestCase):
    def test_canonical_output_zero_and_negative_zero(self):
        data=record('0')+record('-0')
        self.assertEqual(analyze.ascii113(data,canonical_output_zero=True),[0,0])
        signs=analyze.ascii_zero_signs(data)
        self.assertEqual(signs['positive_zero_indices'],[0])
        self.assertEqual(signs['negative_zero_indices'],[1])

    def test_input_grammar_still_requires_exponent(self):
        for token in ('0','-0','+0','1','-1','0.0'):
            with self.subTest(token=token),self.assertRaisesRegex(ValueError,'malformed decimal'):
                analyze.ascii113(record(token))

    def test_nonzero_output_grammar_still_requires_exponent(self):
        for token in ('+0','1','-1','0.0','-0.0','NaN','Inf','-inf','1E','1E+',''):
            with self.subTest(token=token),self.assertRaises(ValueError):
                analyze.ascii113(record(token),canonical_output_zero=True)

    def test_fixed_record_width_remains_mandatory(self):
        for data in (b'0',record('0')[:-1],record('0')+b' '):
            with self.subTest(data=data),self.assertRaisesRegex(ValueError,'record length'):
                analyze.ascii113(data,canonical_output_zero=True)

    def test_lexical_zero_signs_preserved_for_scientific_inputs(self):
        data=record('-0.000E+0000')+record('0D-0001')+record('1.0E0')
        self.assertEqual(analyze.ascii113(data),[0,0,1])
        signs=analyze.ascii_zero_signs(data)
        self.assertEqual(signs['negative_zero_indices'],[0])
        self.assertEqual(signs['positive_zero_indices'],[1])

    def test_values_below_dp_resolution_still_exactly_reconstructed(self):
        data=record('1.000000000000000000000000000000788860905E0')
        decoded=analyze.ascii113(data,canonical_output_zero=True)[0]
        self.assertEqual(decoded,Fraction(1)+Fraction(1,2**100))
        self.assertNotEqual(decoded,Fraction(float(decoded)))

    def test_normal_only_qualification_explicitly_rejects_subnormals_and_underflow(self):
        for token in ('1E-4950','-1E-4950','1E-6000'):
            with self.subTest(token=token),self.assertRaisesRegex(ValueError,'outside qualified normal range'):
                analyze.ascii113(record(token),canonical_output_zero=True)

    def test_overflow_and_rounded_overflow_are_explicit_failures(self):
        with self.assertRaisesRegex(ValueError,'outside qualified normal range'):
            analyze.ascii113(record('1E6000'),canonical_output_zero=True)
        # A 113-bit nearest-even result carrying across the maximal normal binade
        # must fail explicitly rather than masquerading as supported binary128.
        boundary=Fraction(2)**16384-Fraction(2)**(16383-113)
        with self.assertRaisesRegex(ValueError,'rounded value outside qualified normal range'):
            analyze.rounded113(boundary)

    def test_successful_ninjars_zero_output_decodes_but_failed_output_is_not_read(self):
        rows=fixture.sample(8,4,1)
        output=fixture.find(rows,108,10)
        output['data']=record('0')+record('-0')+output['data'][128:]
        parsed=fixture.parse(rows,8,4,1)
        signs=parsed['points'][0]['inputs']['native'][0]['ascii_zero_signs']['output']
        self.assertIn(1,signs['negative_zero_indices'])
        self.assertIn(0,signs['positive_zero_indices'])
        bad=fixture.sample(8,4,1)
        fixture.find(bad,108,9)['data']=fixture.J(2)
        bad.remove(fixture.find(bad,108,10))
        with self.assertRaisesRegex(ValueError,'refusal'):
            fixture.parse(bad,8,4,1)

    def test_tirust_identical_formatter_contract_is_output_only(self):
        rows=fixture.sample(9,4,1)
        output=fixture.find(rows,109,10);output['data']=record('0')+output['data'][64:]
        fixture.parse(rows,9,4,1)
        bad=fixture.sample(9,4,1)
        shifts=fixture.find(bad,109,3);shifts['data']=record('0')+shifts['data'][64:]
        with self.assertRaisesRegex(ValueError,'malformed decimal'):
            fixture.parse(bad,9,4,1)


if __name__=='__main__':unittest.main(verbosity=2)
