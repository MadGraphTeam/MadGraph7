"""Pure controller contract checks; no compiler, loader, or native process."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('five_backend_controller_tested',HERE/'run.py')
controller=importlib.util.module_from_spec(spec);spec.loader.exec_module(controller)

class ControllerTests(unittest.TestCase):
    def test_fixed_scope_and_limits(self):
        self.assertEqual(controller.CASES,('qcd_n1_uux_z','qcd_n2_gg_ttx','ew_n2_epem_mupmum'))
        self.assertEqual(controller.BACKENDS,(1,6,7,8,9))
        self.assertEqual((controller.CPU,controller.CASE_SECONDS,controller.TOTAL_SECONDS,controller.CLEANUP_SECONDS),(12,600.,1800.,60.))
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

if __name__=='__main__':unittest.main()
