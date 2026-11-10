"""Pure source-admission tests; no compiler, loader or numerical evaluation."""
import importlib.util
from pathlib import Path
import re
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
spec = importlib.util.spec_from_file_location('capture_instrument', HERE / 'instrument.py')
instrument = importlib.util.module_from_spec(spec)
spec.loader.exec_module(instrument)


def statements(text):
    rows = []
    for line in text.splitlines():
        if not line or line[0] in 'cC*!':
            continue
        continuation = len(line) > 5 and line[5] not in ' 0'
        code = line[6:].strip().upper()
        if not code:
            continue
        if continuation:
            rows[-1] += code
        else:
            rows.append(code)
    return [re.sub(r'\s+', '', row) for row in rows]


class InstrumentationAdmission(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = (ROOT / 'docs/benchmarkfolder/stability_appendix_c/source5/stability_worker.f').read_text()
        cls.cases = []
        for case,sub,case_id,target,groups,n,rank in (
            ('qcd_n3_uux_zgg','P0_uux_zgg',1,25,[59,61],5,4),
            ('qcd_n4_uux_zzgg','P0_uux_zzgg',2,2,[408,422],6,5)):
            base = ROOT / '.tirust-build/stability-appendix-c-pilot-3/masters' / case / 'SubProcesses' / sub
            params = dict(prefix='ML5_0_',nexternal=n,nborn=1,case_id=case_id,
                          target_point=target,groups=groups,nprop=n,rank=rank)
            cls.cases.append((params,(base/'TIRUST_interface.f').read_text(),(base/'TIR_interface.f').read_text()))

    def test_actual_both_qualified_sources_patch_without_external_execution(self):
        for params, tirust, tir in self.cases:
            with self.subTest(case=params['case_id']):
                result = instrument.instrument(tirust,tir,params)
                self.assertEqual(set(result),{'TIRUST_interface.f','TIR_interface.f'})
                for key,original in (('TIRUST_interface.f',tirust),('TIR_interface.f',tir)):
                    old = statements(original); new = statements(result[key])
                    assignments = lambda rows:[x for x in rows if re.match(r'^[A-Z][A-Z0-9_]*(\(.*\))?=',x)]
                    native_calls = lambda rows:[x for x in rows if x.startswith('CALL') and not x.startswith('CALLML_TIRUST_CAPTURE_')]
                    self.assertEqual(assignments(new),assignments(old))
                    self.assertEqual(native_calls(new),native_calls(old))
                    self.assertEqual([x for x in new if x.startswith('DO')],[x for x in old if x.startswith('DO')])
                    self.assertTrue(all(len(line)<=132 for line in result[key].splitlines()))

    def test_qp_and_order_switch_functions_unchanged(self):
        for params,tirust,tir in self.cases:
            result=instrument.instrument(tirust,tir,params)
            for name,text,marker in (
                ('TIRUST_interface.f',tirust,'      SUBROUTINE ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC'),
                ('TIR_interface.f',tir,'      SUBROUTINE ML5_0_SWITCH_ORDER')):
                self.assertEqual(result[name][result[name].index(marker):],text[text.index(marker):])

    def test_stage_population_and_failed_return_end_hook(self):
        for params,tirust,tir in self.cases:
            result=instrument.instrument(tirust,tir,params)
            for stage in (10,11,12,15,20,22,30):
                self.assertIn('ML_TIRUST_CAPTURE_STAGE('+str(stage)+',',result['TIRUST_interface.f'])
            for stage in (13,14,21):
                self.assertIn('ML_TIRUST_CAPTURE_STATUS('+str(stage)+',STATUS)',result['TIRUST_interface.f'])
            self.assertEqual(result['TIR_interface.f'].count('CALL ML_TIRUST_CAPTURE_END(STABLE)'),2)

    def test_driver_preserves_all_physical_points_and_bootstrap_but_caps_primary_prefix(self):
        for params,unused,unused2 in self.cases:
            text=instrument.render_driver(self.template,params)
            self.assertIn('PARAMETER(REQUIRED_POINTS=100,MAX_BOOTSTRAP=16)',text)
            self.assertEqual(text.count('      DO K=1,NPOINTS\n'),1)
            self.assertIn('      DO K=1,'+str(params['target_point'])+'\n',text)
            self.assertIn('DO BOOTSTRAP=1,MAX_BOOTSTRAP',text)
            self.assertEqual(text.count('CALL ML_TIRUST_CAPTURE_POINT('),1)
            self.assertNotIn('FPE_IN_DP_REDUCTION=',text)
            self.assertNotIn('FPE_IN_QP_REDUCTION=',text)
            self.assertEqual(text.count('CALL ML_TIRUST_CAPTURE_INIT('),1)

    def test_unknown_registry_source_drift_and_double_instrumentation_rejected(self):
        params,tirust,tir=self.cases[0]
        with self.assertRaises(ValueError):
            instrument.render_driver(self.template,{**params,'groups':[59,60]})
        with self.assertRaises(ValueError):
            instrument.instrument(tirust.replace('STABLE=.FALSE.','STABLE=.TRUE.',1),tir,params)
        patched=instrument.instrument(tirust,tir,params)
        with self.assertRaises(ValueError):
            instrument.instrument(patched['TIRUST_interface.f'],patched['TIR_interface.f'],params)


if __name__=='__main__':
    unittest.main()
