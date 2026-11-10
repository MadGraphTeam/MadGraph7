"""Synthetic parser checks only; these tests launch no process or library."""
import copy
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('adaptive_physical_analysis',Path(__file__).with_name('analyze.py'))
analysis=importlib.util.module_from_spec(spec);spec.loader.exec_module(analysis)


def fixture():
    points={}
    for number in (1,2):
        groups=[]
        for group in range(1,201):
            failed=number==2 and group in analysis.EXPECTED_REFUSALS
            data={'NATIVE_INVARIANTS':[['R16',1,'0000000000000000','3FFF000000000000']],
                  'CHECK_PASS':[['LOGICAL','F','FFFFF']],'CACHE_ENABLED':[['LOGICAL','T','FFFFF']],
                  'GROUP_STABLE':[['LOGICAL','F' if failed else 'T','FFFFF']],
                  'GROUP_RESULT':[['C16',1,'0000000000000000','0000000000000000']]}
            if not failed:
                data.update({key:[['C32',1,*(['0000000000000000']*4)]] for key in analysis.RAW})
            groups.append({'key':[group,group,1,4,4,3],'flags':'FFFFF','data':data,
                           'native':[{'backend':9,'status':int(failed),'flags':'FFFFF'}],
                           'errors':['TIRust QP: Unstable("ill-conditioned Gram matrix for fixture")'] if failed else []})
        pdata={key:[['INTEGER','120','FFFFF']] for key in analysis.POINT_OUTPUTS}
        pdata['INPUT_P']=[['R8',1,'3FF0000000000000']]
        points[number]={'data':pdata,'groups':groups,'begin_flags':'FFFFF','end_flags':'FFFFF'}
    return {'points':points,'footer':{'records':1,'dropped':0,'flag_changes':0,'limit':1000000}}


class AnalysisTests(unittest.TestCase):
    def setUp(self):self.old=fixture();self.new=copy.deepcopy(self.old);self.warning='##W03 omitted\n'*74
    def check(self,newlog=None):return analysis.compare(self.old,self.new,self.warning,self.warning if newlog is None else newlog)
    def test_retained_refusals_are_diagnostic_not_complete(self):
        result=self.check();self.assertTrue(result['invariants_passed']);self.assertEqual(result['refusals_now_served'],0)
        self.assertFalse(result['observed_sums_have_no_recorded_native_omissions']);self.assertFalse(result['physical_accuracy_qualified'])
    def test_recovered_all_does_not_prove_precision_or_truth(self):
        for group in self.new['points'][2]['groups']:
            if group['native'][0]['status']:
                group['native'][0]['status']=0;group['errors']=[];group['data']['GROUP_STABLE'][0][1]='T'
                group['data'].update({key:[['C32',1,*(['0000000000000000']*4)]] for key in analysis.RAW})
        result=self.check('');self.assertTrue(result['invariants_passed']);self.assertEqual(result['refusals_now_served'],37)
        self.assertTrue(result['observed_sums_have_no_recorded_native_omissions'])
        self.assertFalse(result['actual_per_call_working_precision_available']);self.assertFalse(result['fixed_CTMode4_is_checked'])
    def test_served_raw_change_rejected(self):
        self.new['points'][1]['groups'][0]['data']['RAW_FINITE'][0][2]='0000000000000001'
        self.assertFalse(self.check()['invariants_passed'])
    def test_served_failure_rejected(self):
        self.new['points'][1]['groups'][0]['native'][0]['status']=1
        self.assertFalse(self.check()['invariants_passed'])
    def test_input_bits_change_rejected(self):
        self.new['points'][2]['groups'][4]['data']['NATIVE_INVARIANTS'][0][2]='0000000000000001'
        self.assertFalse(self.check()['invariants_passed'])
    def test_cache_policy_change_rejected(self):
        self.new['points'][1]['groups'][0]['data']['CACHE_ENABLED'][0][1]='F'
        self.assertFalse(self.check()['invariants_passed'])
    def test_sticky_event_flags_observed(self):
        self.new['points'][2]['groups'][5]['flags']='FFFFT'
        result=self.check();self.assertTrue(result['invariants_passed']);self.assertEqual(len(result['group_or_native_IEEE_flag_differences']),1)
    def test_absent_warning_is_not_complete_if_refusals_remain(self):
        self.assertFalse(self.check('')['observed_sums_have_no_recorded_native_omissions'])
    def test_changed_original_refusal_pattern_rejected(self):
        self.old['points'][2]['groups'][4]['errors']=['unrelated error']
        self.assertFalse(self.check()['invariants_passed'])
    def test_new_early_return_rejected(self):
        self.new['points'][2]['groups'][4]['native']=[]
        self.assertFalse(self.check()['invariants_passed'])


if __name__=='__main__':unittest.main()
