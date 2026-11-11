"""Synthetic trace arithmetic only; never imports or loads the native library."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('private_shadow',HERE/'analyze_shadow.py')
shadow=importlib.util.module_from_spec(spec)
spec.loader.exec_module(shadow)


def trace(prefix, cases):
    rows,hexes=[],[]
    byte_offset=0
    for call,(epoch,mode,status,native) in enumerate(cases,1):
        def row(kind,label='-',length=0):
            nonlocal byte_offset
            number=len(rows)+1
            values=[number,kind,call,epoch,mode,3,2,2,0,1,1,268435456,1,
                    status if kind==4 else 0,int(status==0) if kind==4 else -1,
                    0,0,0,0,0,0,byte_offset,length]
            rows.append(' '.join(map(str,values))+f' {label}')
            return number
        row(1)
        for label in ('pre_pl','pre_m2l','pre_mur','pre_osthres',
                      'encoded_p','encoded_s','encoded_m','encoded_scale'):
            payload=(b'A' if label.startswith('pre_') else native)
            number=row(6,label,len(payload))
            hexes.append(f'{number} {byte_offset} {payload.hex()}')
            byte_offset+=len(payload)
        row(4)
    prefix.with_suffix('.calls.tsv').write_text('\n'.join(rows)+'\n')
    prefix.with_suffix('.shadow.hex').write_text('\n'.join(hexes)+'\n')


class ShadowTests(unittest.TestCase):
    def inspect(self,cases):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'point-16';trace(prefix,cases)
            return shadow.inspect(prefix)

    def test_natural_primary_repeat_only(self):
        result=self.inspect([(1,4,0,b'B'),(1,5,0,b'B'),(1,4,0,b'B')])
        self.assertEqual(result['eligible_calls'],2)
        self.assertEqual(result['potential_prior_exact_key_repeats'],1)
        self.assertFalse(result['performance_acceptance'])

    def test_no_cross_event_or_failed_seed(self):
        result=self.inspect([(1,4,0,b'B'),(1,4,2,b'B'),(1,4,0,b'B'),(2,4,0,b'B')])
        self.assertEqual(result['potential_prior_exact_key_repeats'],0)

    def test_geometry_change_for_same_early_key_rejected(self):
        with self.assertRaisesRegex(ValueError,'different native geometry'):
            self.inspect([(1,4,0,b'B'),(1,4,0,b'C')])

    def test_truncated_payload_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix=Path(directory)/'point-16';trace(prefix,[(1,4,0,b'B')])
            prefix.with_suffix('.shadow.hex').write_text('')
            with self.assertRaisesRegex(ValueError,'Malformed tensor byte payload'):
                shadow.inspect(prefix)


if __name__ == '__main__':
    unittest.main()
