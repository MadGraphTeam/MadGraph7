"""Pure Python checks; bounded process calls are replaced, never executed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

HERE=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('combined_control_test',HERE/'control.py')
c=importlib.util.module_from_spec(loader);loader.loader.exec_module(c)

class ControlChecks(unittest.TestCase):
    def test_runtime_exact(self):
        rows={'ld.so':{'path':'/loader','sha256':'a'},'libc':{'path':'/libc','sha256':'b'}}
        c.verify_runtime(rows,dict(loader='/loader',expected=rows))
        with self.assertRaises(ValueError):
            c.verify_runtime({**rows,'injected':{'path':'/inject','sha256':'x'}},dict(loader='/loader',expected=rows))

    def test_fixture_subset_requires_identities(self):
        rows={'ld.so':{'path':'/loader','sha256':'a'},'libc':{'path':'/libc','sha256':'b'}}
        check=dict(loader='/loader',expected={**rows,'unused':{'path':'/unused','sha256':'c'}},
                   allow_system_subset=True,required=['ELF_INTERPRETER','libc'])
        c.verify_runtime(rows,check)
        with self.assertRaises(ValueError):c.verify_runtime({'ld.so':rows['ld.so']},check)
        with self.assertRaises(ValueError):c.verify_runtime({**rows,'libc':{'path':'/libc','sha256':'bad'}},check)

    def test_wrong_interpreter_refused(self):
        with self.assertRaises(ValueError):c.normalize_runtime({'ld':{'path':'/other'}},'/loader')

    def test_expected_sigfpe_is_explicit_and_retained(self):
        self.exercise_event(-8,False,None,expected=-8,success=True)

    def test_sigfpe_normal_case_fails(self):
        self.exercise_event(-8,False,None,expected=0,success=False)

    def test_timeout_never_accepted(self):
        self.exercise_event(0,True,None,expected=0,success=False)

    def test_launch_error_never_accepted(self):
        self.exercise_event(0,False,{'type':'OSError'},expected=0,success=False)

    def exercise_event(self,returncode,timed_out,error,expected,success):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='control-test-') as folder:
            directory=Path(folder)
            def bounded(argv,cwd,log,deadline,env):
                log.write('retained raw output\n')
                return dict(returncode=returncode,timed_out=timed_out,error=error,elapsed_seconds=0.01)
            cleanup=mock.Mock();cleanup.bounded=bounded
            job=dict(name='case',argv=['never-launched'],cwd=str(directory),expected_exit=expected,
                     log_file=str(directory/'stdout.log'),raw_stdout=True)
            report={'jobs':[]}
            with mock.patch.object(c,'save') as save:
                if success:c.run_job(job,report,{},1,cleanup,'run')
                else:
                    with self.assertRaises(ValueError):c.run_job(job,report,{},1,cleanup,'run')
                self.assertEqual(len(report['jobs']),1)
                self.assertEqual(report['jobs'][0]['returncode'],returncode)
                self.assertTrue(save.called)
            self.assertEqual((directory/'stdout.log').read_text(),'retained raw output\n')

    def test_immutable_parent_sources_bound(self):
        d=json.loads((HERE/'source-manifest.json').read_text())
        self.assertTrue(d)
        self.assertEqual(c.sha(c.OLD),c.OLD_SHA)

if __name__=='__main__':unittest.main()
