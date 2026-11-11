import ast,copy,hashlib,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import controller as c
class ValidationSourceTests(unittest.TestCase):
    def test_exact_single_provider_join_and_failed_inputs_unchanged(self):
        path=c.COMPILED/'prelink.json';before=c.digest(path)
        record=c.join_record()
        self.assertTrue(record['record_created_after_failed_build'])
        self.assertEqual(record['added_bindings'],{c.EXTRA:c.EXTRA_SHA})
        self.assertEqual(record['bound'][c.EXTRA],c.EXTRA_SHA)
        self.assertNotIn(c.EXTRA,c.read(path)['bound'])
        self.assertEqual(c.digest(path),before)
    def test_extra_not_in_original_source_manifest_is_rejected(self):
        pins=copy.deepcopy(c.original_inputs());pins['files'].pop(c.EXTRA)
        with patch.object(c,'original_inputs',return_value=pins):
            with self.assertRaisesRegex(ValueError,'not in original protected'):c.join_record()
    def test_two_unbound_providers_are_rejected(self):
        pins=copy.deepcopy(c.original_inputs());pins['expected_runtime']['/unbound/second-file']='0'*64
        with patch.object(c,'original_inputs',return_value=pins):
            with self.assertRaisesRegex(ValueError,'exactly the scoped'):c.join_record()
    def test_original_failed_command_is_not_relabelled_success(self):
        original=c.read
        def fake(path):
            result=original(path)
            if str(path).endswith('compiler-version.terminal.json'):result={**result,'returncode':1}
            return result
        with patch.object(c,'read',side_effect=fake):
            with self.assertRaisesRegex(ValueError,'Original command failed'):c.join_record()
    def test_validation_has_no_compiler_or_main_execution_branch(self):
        tree=ast.parse(Path(c.__file__).read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_locked')
        calls=[n for n in ast.walk(fn) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Name) and n.func.id in ('build_locked','run_locked') for n in calls))
        self.assertEqual(sum(isinstance(n.func,ast.Attribute) and n.func.attr=='query_runtime' for n in calls),1)
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='command' for n in calls),2)
        self.assertNotIn("add_argument('--build'",Path(c.__file__).read_text());self.assertNotIn("add_argument('--run'",Path(c.__file__).read_text())
    def test_compiler_and_main_commands_forbidden(self):
        pins=c.original_inputs()
        with tempfile.TemporaryDirectory(prefix='rank6-validation-whitelist-') as tmp,patch.object(c.subprocess,'run',side_effect=AssertionError('must not execute')):
            for argv in ([pins['tools']['gfortran'],'--version'],[str(c.COMPILED/'check')]):
                with self.assertRaisesRegex(ValueError,'Forbidden validation command'):c.command(argv,Path(tmp),'rejected')
    def test_failed_preflight_stops_before_any_process(self):
        with tempfile.TemporaryDirectory(prefix='rank6-validation-source-') as tmp:
            with patch.object(c,'WORK',Path(tmp)/'work'),patch.object(c,'preflight',side_effect=ValueError('synthetic refusal')),patch.object(c.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
                with self.assertRaisesRegex(ValueError,'synthetic refusal'):c.stage('validate')
                self.assertEqual(c.read(Path(tmp)/'work/validate/terminal.json')['status'],'failed')
if __name__=='__main__':unittest.main()
