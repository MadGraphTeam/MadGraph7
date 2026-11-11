import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('matched_report',HERE/'prepare_matched.py')
M=importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class ClosedMatchedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.public=M.Public(M.ROOT/'docs/benchmarkfolder/experiments/matched-native-input-qualification-1',
                            'eacced6cc887ee586d15c4ceade8ac1f089fec0a36b84708f12086d05ba2f8a0')
        mapping=json.loads((HERE/'public-map.json').read_text())
        cls.raw={n:cls.public.read(mapping[n]) for n in M.PINS}
        cls.decoded={n:M.parsed(v) for n,v in cls.raw.items()}

    def mutated(self, change, target='matched-comparison.json'):
        raw=dict(self.raw)
        value=copy.deepcopy(self.decoded[target]);change(value)
        raw[target]=M.encoded(value).encode()
        return raw

    def test_actual_public_join_and_exact_residuals(self):
        data=M.derive(self.raw,True)
        self.assertTrue(data['ready_to_seal'])
        self.assertEqual([r['binary64_word_ULP_gap'] for r in data['table']],[0,1,1,0])
        self.assertEqual(data['maximum_relative_error'],'2.01363060398197313E-16')

    def test_failed_native_status_blocks_all_interpretation(self):
        raw=self.mutated(lambda d:d['input_match']['parsed'][0]['calls'][0].update(status=1))
        with self.assertRaises(ValueError):M.derive(raw,True)

    def test_nonidentical_actual_input_rejected(self):
        raw=self.mutated(lambda d:d['input_match']['parsed'][0]['calls'][0]['words']['INV'].__setitem__(0,'3FFF'+'0'*28))
        with self.assertRaises(ValueError):M.derive(raw,True)

    def test_group3_direct_use_required(self):
        def change(d):
            for backend in d['input_match']['parsed']:
                for call in backend['calls']:
                    if call['phase']==1 and call['group']==3:call['meta']['USE'][7]=0
        with self.assertRaises(ValueError):M.derive(self.mutated(change),True)

    def test_whole_worker_failure_rejected(self):
        with self.assertRaises(ValueError):
            M.derive(self.mutated(lambda d:d['observations']['8'].update(eligible=False)),True)

    def test_duplicate_order_cannot_change_a_coefficient(self):
        def change(d):d['measurement_comparison']['residuals'][5]['candidate_word']='BFA1A4B0DE963B62'
        with self.assertRaises(ValueError):M.derive(self.mutated(change),True)

    def test_comparison_word_must_match_actual_outer_row(self):
        def change(d):
            for r in d['measurement_comparison']['residuals']:
                if r['coefficient']=='finite':r['candidate_word']='BFA1A4B0DE963B62'
        with self.assertRaises(ValueError):M.derive(self.mutated(change),True)

    def test_prior_parity_not_a_substitute_for_current_words(self):
        raw=self.mutated(lambda d:d['word_rows'][0].update(current_word='0'*16,prior_word='0'*16),
                         'prior-candidate-word-parity.json')
        with self.assertRaises(ValueError):M.derive(raw,True)

    def test_no_accuracy_or_algorithmic_winner_promotion(self):
        for key in ('accuracy_qualified','normal_host_stability_qualified','performance_measurement','winner_selected'):
            with self.assertRaises(ValueError):M.derive(self.mutated(lambda d:d.update({key:True})),True)

    def test_nonfinite_output_words_rejected_without_float_decode(self):
        for word in ('7FF0000000000000','7FF8000000000001','FFF0000000000000'):
            with self.assertRaises(ValueError):M.exact64(word)

    def test_preview_cannot_claim_public_seal(self):
        data=M.derive(self.raw,False)
        self.assertFalse(data['public_packet_verified']);self.assertFalse(data['ready_to_seal'])


if __name__=='__main__':unittest.main()
