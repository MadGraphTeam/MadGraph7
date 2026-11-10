"""Pure Python prerequisite tests; no subprocesses or native calls."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('additive_control_test',HERE/'control.py')
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)

class Base:
    @staticmethod
    def require(test,message):
        if not test:raise ValueError(message)

class PrerequisiteChecks(unittest.TestCase):
    def exercise(self,change=None):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='source-test-') as folder:
            root=Path(folder);parent=root/'parent';parent.mkdir();case=root/'case';case.mkdir()
            observed=case/'mass-observer.hex';observed.write_text('synthetic fixture, not native evidence\n')
            argv=['/synthetic/guarded','/synthetic/points']
            report={'state':'complete','input_hashes_after':'unchanged','manifest_sha256':'m',
                'jobs':[{'name':'validate','returncode':0,'timed_out':False},
                        {'name':'guarded','returncode':0,'timed_out':False,'changed_inputs':{},
                         'expected_output_closure_matched':True,'command':['taskset','-c','8',*argv],
                         'output_files':{'mass-observer.hex':c.sha(observed)}}]}
            expected={'stages':[{'validation':{'name':'validate'},
                                 'jobs':[{'name':'guarded','argv':argv,'cwd':'case'}]}]}
            if change:change(report,observed)
            (parent/'run.json').write_text(json.dumps(report));(parent/'prepared.json').write_text(json.dumps(expected))
            with mock.patch.object(c,'PARENT',parent),mock.patch.object(c,'ROOT',root):
                return c.prior_observations(Base,{'parent_manifest_sha256':'m'})

    def test_complete_observations_bound(self):self.assertEqual(len(self.exercise()),2)
    def test_failed_parent_refused(self):
        with self.assertRaises(ValueError):self.exercise(lambda r,p:r.update(state='failed'))
    def test_failed_parser_refused(self):
        with self.assertRaises(ValueError):self.exercise(lambda r,p:r['jobs'][0].update(returncode=1))
    def test_changed_observation_refused(self):
        with self.assertRaises(ValueError):self.exercise(lambda r,p:p.write_text('changed'))
    def test_wrong_runtime_command_refused(self):
        with self.assertRaises(ValueError):self.exercise(lambda r,p:r['jobs'][1].update(command=['taskset','-c','9','wrong']))

if __name__=='__main__':unittest.main()
