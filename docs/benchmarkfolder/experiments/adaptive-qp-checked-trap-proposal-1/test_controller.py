"""Synthetic ownership-receipt/cap/output tests; no child or native work."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

spec=importlib.util.spec_from_file_location('private_checked_controller',Path(__file__).with_name('controller.py'))
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)


def capture():
    data={'AMPLITUDE':[['R8',i,'0000000000000000'] for i in range(1,9)],
          'ACCURACY':[['R8',i,'0000000000000000'] for i in (1,2)],
          'PUBLIC_STATUS':[['INTEGER','323','FFFFF']]}
    group={'key':[1,1,1,4,1,0],'data':{'RAW_FINITE':[['C32',1,*(['0'*16]*4)]],
                                       'GROUP_STABLE':[['LOGICAL','T','FFFFF']]}}
    return {'complete':True,'points':{n:{'data':copy.deepcopy(data),'groups':[copy.deepcopy(group)]} for n in (1,2)}}


def workers():
    expected={'/runtime/lib%d.so'%i:str(i) for i in range(15)}
    row={'progress':{'process_reaped':True},'exit_code':0,'changed_inputs':{},'native_output_closure':{'matched':True},
         'expected_runtime_sha256':expected,'actual_mapping':{'executable':'/checked','expected_runtime_sha256':expected,
         'mapped_files_sha256_after_exit':expected,'mapped_elf_paths':[*expected,'/checked']}}
    return {str(i):copy.deepcopy(row) for i in range(7)}


class ControllerTests(unittest.TestCase):
    def test_masked_trapped_same_outputs_with_flags_observed_separately(self):
        a=capture();b=copy.deepcopy(a);b['points'][1]['data']['PUBLIC_STATUS'][0][-1]='FFFFT'
        self.assertTrue(controller.compare_policy_outputs(a,b)['outputs_exact'])
    def test_changed_finite_or_raw_rejected(self):
        for label in ('AMPLITUDE','RAW_FINITE'):
            with self.subTest(label=label):
                a=capture();b=copy.deepcopy(a)
                target=b['points'][2]['data'] if label=='AMPLITUDE' else b['points'][2]['groups'][0]['data']
                target[label][0][2]='0000000000000001'
                self.assertFalse(controller.compare_policy_outputs(a,b)['outputs_exact'])
    def test_partial_output_never_exact(self):
        a=capture();b=copy.deepcopy(a);b['complete']=False
        self.assertFalse(controller.compare_policy_outputs(a,b)['outputs_exact'])
    def test_transport_requires_every_actual_map_and_unchanged_closure(self):
        rows=workers();self.assertTrue(controller.transport_complete(rows))
        del rows['2']['actual_mapping'];self.assertFalse(controller.transport_complete(rows))
        rows=workers();rows['2']['actual_mapping']['mapped_files_sha256_after_exit']={}
        self.assertFalse(controller.transport_complete(rows))
    def test_sigfpe_is_terminal_science_but_timeout_is_transport_failure(self):
        rows=workers();rows['3']['exit_code']=-8;self.assertTrue(controller.transport_complete(rows))
        rows['3']['process_failure']={'type':'TimeoutExpired'};self.assertFalse(controller.transport_complete(rows))
    def test_unexpected_or_missing_output_closure_rejected(self):
        expected={'stdout.log','checked-control.log','mapping-ready','native-maps.txt','native-maps.snapshots/complete.txt','mass-observer.hex','param.log'}
        after={p:'hash' for p in expected};self.assertTrue(controller.output_closure({},after,'physical',0)['matched'])
        after['unexpected.log']='hash';self.assertFalse(controller.output_closure({},after,'physical',0)['matched'])
        del after['unexpected.log'];del after['param.log'];self.assertFalse(controller.output_closure({},after,'physical',0)['matched'])
    def test_late_publication_rescinds_every_acceptance(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'record.json';record={'execution_complete':True,'qualified':True}
            with mock.patch.object(controller,'remaining',side_effect=[1,1,TimeoutError('late publication')]),mock.patch.object(controller.signal,'setitimer'):
                controller.publish_final(path,record,1,0,'execution_complete',lambda:None,claim_key='qualified')
            saved=json.loads(path.read_text());self.assertFalse(saved['execution_complete']);self.assertFalse(saved['qualified'])
    def test_card_changes_are_exactly_reversible(self):
        original='#CTModeRun\n4\n#NRotations_QP\n0\n'
        new=controller.replace_card(original,'CTModeRun','4','-1')
        new=controller.replace_card(new,'NRotations_QP','0','1')
        self.assertEqual(controller.replace_card(controller.replace_card(new,'CTModeRun','-1','4'),'NRotations_QP','1','0'),original)
        with self.assertRaises(AssertionError):controller.replace_card(original,'CTModeRun','-1','4')


if __name__=='__main__':unittest.main()
