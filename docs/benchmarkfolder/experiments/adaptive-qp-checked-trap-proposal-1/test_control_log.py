"""Pure synthetic control-log tests; no executable or reducer is launched."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('checked_controls',Path(__file__).with_name('control_log.py'))
control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)


def state(phase,trapped=False):
    return ['ROUND '+phase+' nearest']+['STATE %s %s F %s T'%(phase,name,'T' if trapped and i<3 else 'F') for i,name in enumerate(control.FLAGS)]


def physical():
    rows=['FORMAT ADAPTIVE_CHECKED_CONTROL_V1','READY_FOR_MAPPING',*state('PROGRAM_START'),'MODE physical masked',*state('AFTER_POLICY'),*state('SETPARA_BEFORE'),*state('SETPARA_AFTER')]
    for stage,index,point,code in [('BOOTSTRAP',0,1,130),('WARMUP',1,1,219),('WARMUP',2,2,323),('OBSERVED',1,1,219),('OBSERVED',2,2,323)]:
        if stage=='OBSERVED' and point==1:rows.append('READY_INITIALIZATION_FINISHED_NOT_ACCEPTANCE')
        rows+=['CALL_ENTER %s %s %s'%(stage,index,point),*state(stage+'_PREPROCESS_BEFORE'),*state(stage+'_PREPROCESS_AFTER'),*state(stage+'_BEFORE'),*state(stage+'_AFTER'),
               'CALL_RETURN %s %s %s %s 3D719799812DEA11 3D719799812DEA11'%(stage,index,point,code)]
    return rows+state('EXIT_BEFORE')+state('EXIT_AFTER')+['COMPLETE_CHECKED_DIAGNOSTIC 0']


class ControlTests(unittest.TestCase):
    def parse(self,rows,partial=False):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'control.log';path.write_text('\n'.join(rows)+'\n');return control.parse(path,allow_partial=partial)
    def test_complete_observations(self):
        result=self.parse(physical());self.assertTrue(control.assess(result,exit_code=0)['physical_worker_completed_with_requested_IEEE_policy'])
        self.assertEqual(result['observed'][2]['H'],3)
    def test_truncated_stream_never_qualifies(self):
        result=self.parse(physical()[:-1],True)
        self.assertFalse(control.assess(result,exit_code=0)['physical_worker_completed_with_requested_IEEE_policy'])
        with self.assertRaises(ValueError):self.parse(physical()[:-1])
    def test_fault_requires_actual_signal_and_no_survival(self):
        rows=['FORMAT ADAPTIVE_CHECKED_CONTROL_V1','READY_FOR_MAPPING',*state('PROGRAM_START'),'MODE fault invalid',*state('AFTER_POLICY',True),'FAULT_ENTER invalid']
        result=self.parse(rows,True);self.assertTrue(control.assess(result,exit_code=-8)['fault_control_demonstrated'])
        self.assertFalse(control.assess(result,exit_code=0)['fault_control_demonstrated'])
        result=self.parse(rows+['FAULT_SURVIVED invalid'],True)
        self.assertFalse(control.assess(result,exit_code=-8)['fault_control_demonstrated'])
    def test_halting_policy_change_rejected(self):
        rows=physical();rows[rows.index('STATE AFTER_POLICY invalid F F T')]='STATE AFTER_POLICY invalid F T T'
        result=self.parse(rows);self.assertFalse(control.assess(result,exit_code=0)['physical_worker_completed_with_requested_IEEE_policy'])
    def test_rounding_change_rejected(self):
        rows=physical();rows[rows.index('ROUND AFTER_POLICY nearest')]='ROUND AFTER_POLICY down'
        self.assertFalse(control.assess(self.parse(rows),exit_code=0)['physical_worker_completed_with_requested_IEEE_policy'])
    def test_duplicate_flag_rejected(self):
        rows=physical();index=rows.index('STATE AFTER_POLICY invalid F F T');rows.insert(index,rows[index])
        with self.assertRaises(ValueError):self.parse(rows)
    def test_nonfinite_accuracy_is_retained_as_unevaluated(self):
        rows=[r.replace('3D719799812DEA11','7FF8000000000000') if r.startswith('CALL_RETURN OBSERVED') else r for r in physical()]
        self.assertEqual(self.parse(rows)['observed'][1]['accuracy'],[None,None])
    def test_unsuccessful_warmup_is_not_acceptance(self):
        rows=[r.replace('WARMUP 2 2 323','WARMUP 2 2 420') for r in physical()]
        result=self.parse(rows);self.assertTrue(result['ready']);self.assertEqual(result['calls'][2]['H'],4)
    def test_missing_call_IEEE_observation_rejected(self):
        rows=[r for r in physical() if 'OBSERVED_PREPROCESS_AFTER' not in r]
        with self.assertRaises(ValueError):self.parse(rows)
    def test_fault_missing_program_start_cannot_qualify(self):
        rows=['FORMAT ADAPTIVE_CHECKED_CONTROL_V1','READY_FOR_MAPPING','MODE fault invalid',*state('AFTER_POLICY',True),'FAULT_ENTER invalid']
        self.assertFalse(control.assess(self.parse(rows,True),exit_code=-8)['fault_control_demonstrated'])
    def test_missing_bootstrap_rejected(self):
        rows=physical();start=rows.index('CALL_ENTER BOOTSTRAP 0 1')
        end=next(i for i,r in enumerate(rows) if r.startswith('CALL_RETURN BOOTSTRAP'))
        rows=rows[:start]+rows[end+1:]
        with self.assertRaises(ValueError):self.parse(rows)


if __name__=='__main__':unittest.main()
