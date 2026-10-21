"""Real retained-preparation metadata preflight; every native entry is forbidden."""
from contextlib import ExitStack
import copy,importlib.util,json,tempfile,unittest
from pathlib import Path
from unittest import mock
BASE=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('light_metadata_control',BASE/'execute.py');runner=importlib.util.module_from_spec(loader);loader.loader.exec_module(runner)
class PreparationPreflight(unittest.TestCase):
    def setUp(self):
        self.spec=runner.read(BASE/'specification.json');self.temp=tempfile.TemporaryDirectory(dir=runner.ROOT/'.benchmark-work',prefix='light-preflight-');self.addCleanup(self.temp.cleanup)
        self.stack=ExitStack();self.addCleanup(self.stack.close)
        for owner,name in((runner.bench,'run'),(runner.bench,'configure'),(runner.bench,'measure'),(runner.workers.subprocess,'Popen'),(runner.workers,'Worker')):
            self.stack.enter_context(mock.patch.object(owner,name,side_effect=AssertionError('No native or output-producing path may run in metadata preflight')))
    def changed_preparation(self,data):
        p=Path(self.temp.name)/'prepared.json';p.write_text(json.dumps(data));spec=copy.deepcopy(self.spec);spec['compiled_preparation']=runner.bound(p);return spec
    def test_real_identity_and_all_nine_asset_guards_before_any_output_or_native_path(self):
        cards={}
        original=runner.read(runner.policy.path(self.spec['compiled_preparation']['path']))
        for target in self.spec['schedule']:
            p=runner.policy.path(original['variants'][target['variant']]['output'])/'Cards/MadLoopParams.dat';cards[p]=runner.digest(p)
        before=runner.policy.path(self.spec['report']).exists();self.assertFalse(before)
        verified=runner.preparation_preflight(self.spec)
        self.assertIsNone(verified.get('schema'));self.assertEqual(verified['identity_sha256'],original['identity_sha256'])
        self.assertEqual(len(self.spec['schedule']),9);self.assertFalse(runner.policy.path(self.spec['report']).exists())
        self.assertEqual(cards,{p:runner.digest(p) for p in cards})
    def test_missing_sealed_identity_rejects_even_with_matching_file_checksum(self):
        data=runner.read(runner.policy.path(self.spec['compiled_preparation']['path']));data.pop('identity_sha256')
        with self.assertRaises(RuntimeError):runner.preparation_preflight(self.changed_preparation(data))
        self.assertFalse(runner.policy.path(self.spec['report']).exists())
    def test_tampered_declared_executable_binding_rejects_before_native_path(self):
        spec=copy.deepcopy(self.spec);spec['schedule'][0]['executable_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Declared target'):runner.preparation_preflight(spec)
        self.assertFalse(runner.policy.path(self.spec['report']).exists())
if __name__=='__main__':unittest.main()
