"""Format/matching regressions; synthetic integer records, no amplitudes."""
import copy
from pathlib import Path
import tempfile
import unittest
import parse_capture


def synthetic():
    lines=['LAYOUT 3FF0000000000000 0000000000000000 3FFF000000000000',
           'COMPLEX64 3FF0000000000000 4000000000000000',
           'COMPLEX128 0000000000000000 3FFF000000000000 0000000000000000 4000000000000000']
    seq=0
    for point in (44,80):
        for stage in (1,2,3):
            bits=128 if stage==3 else 64
            words=' '.join(['0000000000000000']*(bits//64))
            complex_words=words+' '+words
            for direction in (1,2):
                mode=direction+(3 if stage==3 else 0)
                for group in range(1,117):
                    seq+=1
                    lines.extend([f'BEGIN {seq} {point} {mode} {group} 1 1 0 {stage}',
                                  f'PAYLOAD {seq} {bits} 0 1',f'C 0 {complex_words}',f'S 1 1 {words}'])
                    lines.extend(f'P {i} 1 {words}' for i in range(4))
                    lines.extend([f'M 1 {complex_words}',f'U {words}'])
                    lines.extend(f'R {i} {complex_words}' for i in range(3))
                    lines.extend([f'R1 {complex_words}',f'END {seq}'])
    return '\n'.join(lines)+'\n'


class CaptureTests(unittest.TestCase):
    def parsed(self,text=None):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'capture.hex';path.write_text(synthetic() if text is None else text)
            return parse_capture.parse(path)

    def test_complete_exact_matching(self):
        result=parse_capture.match(self.parsed())
        self.assertEqual(result['matched_records'],464)
        self.assertTrue(result['native_compensated_inputs_bit_identical'])

    def test_partial_payload_is_rejected(self):
        with self.assertRaises((AssertionError,StopIteration)):
            self.parsed(synthetic().rsplit('END',1)[0])

    def test_duplicate_group_is_rejected(self):
        records=self.parsed();records[1]['loop_id']=1
        with self.assertRaises(AssertionError):parse_capture.match(records)

    def test_compensated_coefficient_change_is_rejected(self):
        records=self.parsed()
        next(r for r in records if r['stage']==2)['data']['C:0'][0]='3FF0000000000000'
        with self.assertRaises(AssertionError):parse_capture.match(records)

    def test_qp_rank_is_not_a_matching_shortcut(self):
        records=self.parsed()
        next(r for r in records if r['stage']==3)['n']=2
        with self.assertRaises(AssertionError):parse_capture.match(records)

    def test_nonfinite_and_layout_are_rejected(self):
        with self.assertRaises(AssertionError):
            self.parsed(synthetic().replace('C 0 0000000000000000','C 0 7FF0000000000000',1))
        with self.assertRaises(AssertionError):
            self.parsed(synthetic().replace('LAYOUT 3FF0000000000000','LAYOUT 3FF0000000000001',1))

    def test_missing_or_extra_record_is_rejected(self):
        text=synthetic()
        with self.assertRaises(AssertionError):self.parsed(text[:text.rfind('BEGIN')])
        with self.assertRaises(AssertionError):self.parsed(text+'unexpected\n')


if __name__=='__main__':
    unittest.main()
