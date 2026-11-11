"""Pure controller contract checks; no compiler, loader, or native process."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('four_lepton_controller_tested',HERE/'run.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)

class ControllerTests(unittest.TestCase):
    def test_fixed_scope_and_limits(self):
        self.assertEqual(controller.CASES,('qcd_n4_uux_epem_mupmum',))
        self.assertEqual(controller.BACKENDS,(1,6,7,8,9))
        self.assertEqual((controller.CPU,controller.CASE_SECONDS,controller.TOTAL_SECONDS,controller.CLEANUP_SECONDS),(12,600.,600.,60.))
        self.assertEqual(controller.TARGET_SECONDS,2.)
        self.assertEqual(controller.LIMITS,{'address_space_bytes':16*1024**3,'file_bytes':512*1024**2,'core_bytes':0})

    def test_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'a/result.json';controller.dump(path,{'status':'closed'})
            with self.assertRaises(FileExistsError):controller.dump(path,{'status':'replaced'})
            self.assertEqual(json.loads(path.read_text()),{'status':'closed'})

    def test_actual_loader_excludes_only_historical_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);actual=root/'real/ld-linux-x86-64.so.2';old=root/'old/ld-linux-x86-64.so.2'
            actual.parent.mkdir();old.parent.mkdir();actual.write_bytes(b'actual');old.write_bytes(b'old')
            native=root/'libmadloop_tirust.so';native.write_bytes(b'bound-by-preparation')
            original={'original_output':str(root),'subprocess':'P0','runtime':{
                'libraries':{str(actual):'actual',str(old):'old',str(native):'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9'},
                'ldd':f'\t{actual} => {old} (0x123)\n'}}
            loader=SimpleNamespace(_support=lambda *a:SimpleNamespace(interpreter=lambda p:actual))
            with patch.object(controller,'support',return_value=loader):value=controller.expected_runtime(original)
            self.assertEqual(value['historical_ldd_only_aliases'],{str(old):'old'})
            self.assertEqual(value['actual_interpreter'],str(actual))
            self.assertNotIn(str(old),value['files'])
            original['runtime']['libraries'].pop(str(actual))
            with patch.object(controller,'support',return_value=loader),self.assertRaisesRegex(ValueError,'Actual PT_INTERP absent'):
                controller.expected_runtime(original)

    def test_loader_callback_rejects_wrong_command_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe=Path(tmp)/'check';exe.write_bytes(b'ELF')
            def query(exe,expected,bound,callback):return callback(['/unexpected','--list',str(exe)],'bad')
            with patch.object(controller,'support',return_value=SimpleNamespace(query_runtime=query)),patch.object(controller.subprocess,'check_output') as invoked:
                with self.assertRaisesRegex(ValueError,'Unexpected loader'):
                    controller.loader_query(exe,{'actual_interpreter':'/expected','files':{}},Path(tmp)/'query')
                invoked.assert_not_called()

    def test_prepare_copies_source_parents_without_native_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);here=base/'source';here.mkdir();(here/'nested').mkdir();(here/'nested/payload.txt').write_text('pinned')
            report=base/'report';work=base/'work';campaign=base/'campaign';campaign.mkdir()
            for name in ('manifest.json','cases.json'):(campaign/name).write_text('{}')
            (here/'source-index.json').write_text(json.dumps({'files':{'nested/payload.txt':{}},'support':{}}))
            def prepared(*args):
                controller.dump(report/'preparation.json',{'originals':{},'clones':{}})
                return {'originals':{}}
            helper=SimpleNamespace(prepare=prepared,schedule=lambda:[])
            actual_digest=controller.digest
            def digest(path):
                if 'locale-archive' in str(path):return '53851ab7caa5a0f893056762a6abff4e383ea06c0e6b276ee16876499e712d27'
                return actual_digest(path)
            with patch.multiple(controller,HERE=here,REPORT=report,WORK=work,CASES=(),EXPECTED_SOURCE='s'),patch.object(controller,'preflight',return_value={'originals':{},'references':{}}),patch.object(controller,'local',return_value=helper),patch.object(controller,'digest',side_effect=digest),patch.object(controller.bench,'REPORT',campaign),patch.object(controller.bench,'machine_identity',return_value={}),patch.object(controller.subprocess,'Popen') as native:
                controller.prepare()
                self.assertEqual((report/'source/nested/payload.txt').read_text(),'pinned')
                self.assertIn('source/nested/payload.txt',json.loads((report/'specification.json').read_text())['frozen_files'])
                native.assert_not_called()

    def test_absolute_alarm_bounds_synchronous_checks(self):
        import time
        with self.assertRaises(controller.BudgetExpired):
            with controller.alarm_deadline(time.monotonic()+.02):time.sleep(.2)

    def test_fresh_reference_plan_has_no_historical_observation(self):
        value=controller.reference_identity(controller.CASES[0],{'points_sha256':'points','executable_sha256':'executable'})
        self.assertIsNone(value['historical_reference'])
        self.assertTrue(value['acquisition_pending'])
        self.assertEqual((value['backend'],value['points']),(1,100))

    def test_reference_must_be_formed_by_cuttools_once_before_comparisons(self):
        observation={'rows':[{'point':1}]}
        with patch.object(controller.bench,'compare_rows') as compare:
            reference,check=controller.learning_check(1,observation,None,3e-6)
            self.assertIs(reference,observation)
            self.assertFalse(check['independent_accuracy_claim'])
            self.assertEqual(check['role'],'fresh-reference-acquisition')
            compare.assert_not_called()
            with self.assertRaisesRegex(ValueError,'only once'):
                controller.learning_check(1,observation,reference,3e-6)
            with self.assertRaisesRegex(ValueError,'must precede'):
                controller.learning_check(9,observation,None,3e-6)
            compare.return_value={'passed':False,'ordinary_target_passed':False}
            retained,check=controller.learning_check(9,{'rows':[{'point':2}]},reference,3e-6)
            self.assertIs(retained,reference)
            compare.assert_called_once_with(observation['rows'],[{'point':2}],3e-6)
            self.assertFalse(check['passed'])

    def test_topology_case_keeps_unchanged_full100_parser_before_reference(self):
        import ast
        tree=ast.parse((HERE/'run.py').read_text())
        child=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='case_child')
        calls=[(n.lineno,ast.unparse(n.func)) for n in ast.walk(child) if isinstance(n,ast.Call)]
        parsed=min(n for n,name in calls if name=='parse_warmup')
        adopted=min(n for n,name in calls if name=='learning_check')
        self.assertLess(parsed,adopted)
        source=(HERE/'run.py').read_text()
        self.assertIn("learned=parse_warmup(lane/'candidate-initialization',[backend],100,ready[key])",source)
        self.assertNotIn("reference=read(REPORT/'reference'",source)
        self.assertIsNone(controller.CAMPAIGN)

    def test_controller_owned_cleanup_and_timing_loop_are_inherited(self):
        import ast
        original=HERE.parent/'current-source-five-backend-timing-controller-1/run.py'
        def body(path,name):
            tree=ast.parse(path.read_text())
            return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
        for name in ('loader_query','mapped_files','capture_last'):
            self.assertEqual(body(HERE/'run.py',name),body(original,name),name)
        old=body(original,'execute').replace('madloop-current-five-backend-descriptive-v1','madloop-current-four-lepton-five-backend-descriptive-v1')
        self.assertEqual(body(HERE/'run.py','execute'),old)

if __name__=='__main__':unittest.main()
