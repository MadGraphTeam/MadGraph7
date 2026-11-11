import ast,copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import controller as c
class RunGlueTests(unittest.TestCase):
    def setUp(self):
        self.patches=[patch.object(c,'REVIEWED_VALIDATION',c.VALIDATION_SHA),patch.object(c,'EXPECTED_SOURCE','a'*64)]
        for p in self.patches:p.start()
        self.addCleanup(lambda:[p.stop() for p in reversed(self.patches)])
    def test_validation_gate_exact_success(self):
        r=c.verify_validated();self.assertEqual(r['status'],'passed');self.assertEqual(r['native_tensor_calls'],0)
    def test_validation_hash_is_mandatory(self):
        with patch.object(c,'REVIEWED_VALIDATION','0'*64):
            with self.assertRaisesRegex(ValueError,'separately reviewed'):c.verify_validated()
    def test_numerical_body_bytecode_unchanged_except_provenance_field(self):
        p=c.scientific();ns=c.scientific_run_namespace(Path('/tmp/not-created-run-fixture'))
        self.assertEqual(p.run_locked.__code__.co_code,ns['run_locked'].__code__.co_code)
        self.assertEqual(tuple('REVIEWED_VALIDATION' if n=='REVIEWED_BUILD' else n for n in p.run_locked.__code__.co_names),ns['run_locked'].__code__.co_names)
        self.assertEqual(ns['WORK'],c.COMPILED.parent);self.assertEqual(ns['HERE'],c.SCI)
        self.assertEqual(ns['EXPECTED_SOURCE'],'a'*64)
    def test_wrong_prebound_map_or_elf_rejected_before_callback(self):
        ns=c.scientific_run_namespace(Path('/tmp/not-created-run-fixture'));loader=ns['support']('loader')
        callback=lambda *args:(_ for _ in ()).throw(AssertionError('must not execute'))
        for exe,record in [(c.COMPILED/'other',c.read(c.COMPILED/'prelink.json')),(c.COMPILED/'check',{})]:
            with self.assertRaises(ValueError):loader.query_runtime(exe,c.original_inputs()['expected_runtime'],record,callback)
    def test_numerical_glue_forbids_compiler_metadata_command(self):
        directory=Path('/tmp/not-created-run-fixture');ns=c.scientific_run_namespace(directory)
        with patch.object(c.subprocess,'run',side_effect=AssertionError('must not execute')):
            with self.assertRaisesRegex(ValueError,'exact pre-main loader'):ns['command']([c.original_inputs()['tools']['gfortran'],'--version'],c.ROOT,directory,'rejected')
    def test_only_RUN_entry_option(self):
        source=Path(c.__file__).read_text()
        self.assertNotIn("add_argument('--build'",source);self.assertNotIn("add_argument('--validate'",source)
        self.assertIn("add_argument('--run'",source);self.assertIn("--reviewed-validation",source)
    def test_preflight_failure_never_spawns(self):
        with tempfile.TemporaryDirectory(prefix='rank6-run-source-') as tmp:
            with patch.object(c,'WORK',Path(tmp)/'work'),patch.object(c,'preflight',side_effect=ValueError('synthetic source failure')),patch.object(c.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
                with self.assertRaisesRegex(ValueError,'synthetic source failure'):c.stage('run')
                self.assertEqual(c.read(Path(tmp)/'work/run/terminal.json')['status'],'failed')
    def test_ownership_core_is_identical(self):
        old=(c.VALIDATOR/'controller.py').read_text();new=Path(c.__file__).read_text()
        def select(s):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(s).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in ['cleanup_owned','raw_cleanup','StageDeadline']}
        self.assertEqual(select(old),select(new))
if __name__=='__main__':unittest.main()
