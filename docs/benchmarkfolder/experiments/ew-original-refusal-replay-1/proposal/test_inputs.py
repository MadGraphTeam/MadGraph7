"""Exact captured-input transport and original-only EW scope checks."""
import json
from math import comb
from pathlib import Path
import tempfile
import unittest

import extract_inputs as e
import worker

class InputTests(unittest.TestCase):
    def test_two_unchanged_rank_three_inputs(self):
        certificate=json.loads((e.HERE/'input-certificate.json').read_bytes())
        self.assertEqual(len(certificate['requests']),2)
        self.assertFalse(certificate['input_interventions'])
        self.assertEqual(certificate['tensor_component_count'],comb(3+4,4))
        for request,capture in zip(certificate['requests'],certificate['certificates']):
            self.assertEqual((request['point'],request['variant']),(2,'stock'))
            p=e.ROOT/request['input']['path']
            e.checked(p,request['input'])
            words=worker.input_words(p,request)
            self.assertEqual(words,capture['native_words'])
            self.assertEqual(len(words),41 if request['mode']==1 else 82)
            self.assertEqual(capture['original_native_status'],1)
            self.assertEqual(capture['identity']['rank'],3)
            self.assertFalse(capture['intervention'])

    def test_wrong_rank_or_word_count_rejected(self):
        request={'point':2,'mode':4}
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.words'
            for header,count in [('2 4 4 2 82',82),('2 4 4 3 82',81),('2 4 4 3 41',41)]:
                p.write_text('KPROJ_WORDS_V1\n'+header+'\n'+('0000000000000000\n'*count))
                with self.assertRaises(ValueError):worker.input_words(p,request)

    def test_plan_sequence_is_original_source_order(self):
        certificate=json.loads((e.HERE/'input-certificate.json').read_bytes())
        self.assertEqual(certificate['plan_registration_sequence'],[
            [1,0],[2,0],[2,1],[2,2],[3,0],[3,1],[3,2],[3,3],
            [4,1],[4,2],[4,3],[5,2],[5,3]])

    def test_wrong_hash_precedes_input_parse(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad';p.write_text('not an input')
            with self.assertRaisesRegex(ValueError,'Bound bytes differ'):
                e.checked(p,{'bytes':0,'sha256':'wrong'})

if __name__=='__main__':unittest.main()
