"""Pure word-layout, exact-enclosure and closed-input regression checks."""
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
import extract_inputs as e


class InputTests(unittest.TestCase):
    def test_dp_only_two_symmetric_column_major_scalars_change(self):
        words=[f'{i:016X}' for i in range(41)]
        changed,proof=e.replace_pair(words,['BE123456789ABCDE'],1)
        self.assertEqual([i+1 for i in range(41) if words[i]!=changed[i]],[32,38])
        self.assertEqual(changed[31],changed[37]);self.assertEqual(changed[:24],words[:24])
        self.assertEqual(changed[40],words[40])

    def test_qp_low_word_first_and_unchanged_low_half(self):
        words=[f'{i:016X}' for i in range(82)]
        words[62]=words[74]='0000000000000000'
        raw=['0000000000000000','BFE9123456789ABC']
        changed,proof=e.replace_pair(words,raw,4)
        self.assertEqual(changed[62:64],raw);self.assertEqual(changed[74:76],raw)
        self.assertEqual(proof['allowed_word_positions_one_based'],[63,64,75,76])
        self.assertEqual(proof['changed_word_positions_one_based'],[64,76])
        self.assertEqual(changed[:48],words[:48]);self.assertEqual(changed[80:],words[80:])

    def test_invalid_shape_and_noop_intervention_rejected(self):
        with self.assertRaisesRegex(ValueError,'shape differs'):e.replace_pair(['0']*40,['0'],1)
        with self.assertRaisesRegex(ValueError,'Both selected'):e.replace_pair(['0']*82,['0','0'],4)

    def test_strict_negative_enclosures_without_float_conversion(self):
        for value in (-F(1,2**16494),-F(5,10**7),-F(3,2),-F(2**112+1,2**120)):
            bound=e.negative_bounds(value)
            self.assertLess(bound['strict_lower'],value)
            self.assertLessEqual(value,bound['inclusive_upper'])
            self.assertLess(bound['inclusive_upper'],0)
        for value in (F(0),F(1,2)):
            with self.assertRaisesRegex(ValueError,'strictly negative'):e.negative_bounds(value)

    def test_wrong_hash_fails_before_json_parse(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'bad';p.write_text('not json')
            with self.assertRaisesRegex(ValueError,'Bound bytes differ'):
                e.checked(p,{'bytes':0,'sha256':'wrong'})

    def test_ten_saved_files_have_independent_symmetric_differences(self):
        data=json.loads((e.HERE/'input-certificate.json').read_bytes())
        self.assertEqual(len(data['requests']),10)
        groups={}
        for request in data['requests']:
            p=e.ROOT/request['input']['path'];e.checked(p,request['input'])
            lines=p.read_text().splitlines();self.assertEqual(lines[0],'KPROJ_WORDS_V1')
            point,mode,n,rank,count=map(int,lines[1].split())
            self.assertEqual((point,mode),(request['point'],request['mode']))
            self.assertEqual((n,rank),(4,2));self.assertEqual(len(lines[2:]),count)
            groups.setdefault((point,mode),{})[request['variant']]=lines[2:]
        self.assertEqual(set(groups),{(p,m) for p in (1,20,21) for m in (1,4)})
        for (point,mode),variants in groups.items():
            if point==1:
                self.assertEqual(set(variants),{'stock'});continue
            self.assertEqual(set(variants),{'stock','raw24'})
            a,b=variants['stock'],variants['raw24']
            expected=[32,38] if mode==1 else [63,64,75,76]
            self.assertEqual([i+1 for i in range(len(a)) if a[i]!=b[i]],expected)
            width=1 if mode==1 else 2
            matrix=[tuple(b[24*width+j*width:24*width+(j+1)*width]) for j in range(16)]
            self.assertEqual(matrix[13],matrix[7])
            self.assertEqual(a[:24*width],b[:24*width]);self.assertEqual(a[40*width:],b[40*width:])


if __name__=='__main__':unittest.main()
