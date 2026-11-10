"""Pure decimal/word protocol tests only; never load a reduction library."""
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import unittest
import analyze

class WordTransport(unittest.TestCase):
    def test_signed_zero(self):
        self.assertEqual(analyze.binary128('-0.0000E+0000'),1<<127)
        self.assertEqual(analyze.binary128('0'),0)

    def test_exact_low_bit(self):
        with localcontext() as ctx:
            ctx.prec=180
            value=Decimal(1)+Decimal(2)**-112
            self.assertEqual(analyze.binary128(str(value)),(16383<<112)+1)

    def test_ties_even(self):
        with localcontext() as ctx:
            ctx.prec=180
            self.assertEqual(analyze.binary128(str(Decimal(1)+Decimal(2)**-113)),16383<<112)
            self.assertEqual(analyze.binary128(str(Decimal(1)+3*Decimal(2)**-113)),(16383<<112)+2)

    def test_subnormal(self):
        with localcontext() as ctx:
            ctx.prec=180
            self.assertEqual(analyze.binary128(str(Decimal(2)**-16494)),1)
        self.assertEqual(analyze.fraction(1),Fraction(1,2**16494))

    def test_nonfinite(self):
        for text in ('nan','inf','-inf'):
            with self.assertRaises(ValueError):analyze.binary128(text)
        with self.assertRaises(ValueError):analyze.fraction(32767<<112)

    def test_captured_records(self):
        root=Path(__file__).resolve().parent/'fixtures'
        manifest=json.loads((root/'manifest.json').read_text())
        for key,value in manifest['fields'].items():
            self.assertEqual(analyze.records((root/('group10-'+key+'.dat')).read_bytes().hex(),len(value['raw_words'])//2),
                             analyze.word_bits(value['raw_words']))

    def test_count_and_encoding(self):
        with self.assertRaises(ValueError):analyze.records(b'1'.hex(),1)
        with self.assertRaises(ValueError):analyze.word_bits(['0'])

if __name__=='__main__':unittest.main()
