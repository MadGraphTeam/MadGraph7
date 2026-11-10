"""Focused pure-source/mocked tests. No external command or reducer is run."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
R=load('stress_test_supervisor',HERE/'run.py')
N=load('stress_test_native',HERE/'native_job.py')

class SourceTests(unittest.TestCase):
    def test_default_never_creates_or_executes(self):
        with patch.object(R.sys,'argv',['run.py']),patch.object(R.L.subprocess,'Popen') as spawn,patch.object(R,'seal') as seal,contextlib.redirect_stdout(io.StringIO()) as out:
            R.main()
        self.assertFalse(spawn.called);self.assertFalse(seal.called);self.assertEqual(json.loads(out.getvalue())['commands_executed'],0)
    def test_execute_requires_stage(self):
        with patch.object(R.sys,'argv',['run.py','--execute']),patch.object(R.L.subprocess,'Popen') as spawn,self.assertRaisesRegex(RuntimeError,'Explicit stage'):
            R.main()
        self.assertFalse(spawn.called)
    def test_only_preflight_has_mass_execution(self):
        tree=ast.parse((HERE/'run.py').read_text())
        owners=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and '--masses' in ast.get_source_segment((HERE/'run.py').read_text(),n)]
        self.assertEqual(owners,['preflight'])
    def test_preparation_does_not_request_loader(self):
        tree=ast.parse((HERE/'run.py').read_text())
        for n in tree.body:
            if isinstance(n,ast.FunctionDef) and n.name in ('prepare','prepare_process','materialize'):
                names=[x.func.id for x in ast.walk(n) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name)]
                self.assertNotIn('runtime',names)
    def test_native_commands_stay_singleton_and_bound(self):
        self.assertEqual(len(R.PLAN['requests']),24)
        for x in R.PLAN['requests']:
            self.assertEqual(x['MLReductionLib'],str(x['backend']))
            self.assertEqual(x['startup_seconds'],180);self.assertEqual(x['inclusive_worker_seconds'],380)
    def test_preparation_owns_only_fresh_roots(self):
        with patch.object(R.REPORT.__class__,'exists',return_value=True),patch.object(R,'materialize') as materialize,self.assertRaisesRegex(RuntimeError,'Fresh namespaces'):
            R.prepare()
        self.assertFalse(materialize.called)
    def test_gate_rejects_w03_even_for_finite_checked_output(self):
        data={'rows':[{'accuracy':1e-12}]}
        self.assertTrue(N.qualified(data,None,[],True,[]))
        self.assertFalse(N.qualified(data,None,['##W03: missing group'],True,[]))
    def test_gate_requires_actual_map_and_strict_accuracy(self):
        self.assertFalse(N.qualified({'rows':[{'accuracy':1e-3}]},None,[],True,[]))
        self.assertFalse(N.qualified({'rows':[{'accuracy':1e-12}]},None,[],False,[]))
        self.assertFalse(N.qualified({'rows':[{'accuracy':1e-12}]},None,[],True,['unknown']))
    def test_120_is_rejected_by_retained_parser(self):
        from tirust_benchmark_worker import validate_status
        with self.assertRaises(ValueError):validate_status(120,[9])
    def test_loader_uses_binary_interpreter_not_ldd(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader=Path(tmp)/'actual-loader';loader.write_bytes(b'ELFsource');exe=Path(tmp)/'check';exe.write_bytes(b'exe')
            with patch.object(R.L,'interpreter',return_value=loader),patch.object(R.L,'capture',return_value=str(loader)) as capture:
                result=R.runtime(exe)
            self.assertEqual(capture.call_args.args[0],[loader,'--list',exe]);self.assertIn(str(loader),result['files'])
    def test_owned_timeout_retains_cleanup_and_reaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=Mock(pid=42,returncode=-9);process.wait.side_effect=R.L.subprocess.TimeoutExpired(['fake'],0)
            owned=Mock();owned.stop.return_value={'survivors':[]}
            active=types.SimpleNamespace(end=100.,handoff=False,deferred=False)
            with patch.object(R.L,'REPORT',Path(tmp)),patch.object(R.L,'ACTIVE',active),patch.object(R.L,'REQUESTS',Mock()),patch.object(R.L,'remaining',return_value=.01),patch.object(R.L,'environment',return_value={}),patch.object(R.L.subprocess,'Popen',return_value=process),patch.object(R.L.cleanup,'OwnedChild',return_value=owned),patch.object(R.L.time,'monotonic',side_effect=[0.,1.,2.]),patch.object(R.L.signal,'getsignal',return_value=0),patch.object(R.L.signal,'signal'),patch.object(R.L.signal,'setitimer'),self.assertRaises(TimeoutError):
                R.L.command(['fake'],'timeout',cpu=False)
            owned.stop.assert_called_once();self.assertTrue((Path(tmp)/'execution/timeout.json').exists())
    def test_owned_nonzero_exit_also_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=Mock(pid=43,returncode=7);process.wait.return_value=7;owned=Mock();owned.stop.return_value={'survivors':[]}
            active=types.SimpleNamespace(end=100.,handoff=False,deferred=False)
            with patch.object(R.L,'REPORT',Path(tmp)),patch.object(R.L,'ACTIVE',active),patch.object(R.L,'REQUESTS',Mock()),patch.object(R.L,'remaining',return_value=10.),patch.object(R.L,'environment',return_value={}),patch.object(R.L.subprocess,'Popen',return_value=process),patch.object(R.L.cleanup,'OwnedChild',return_value=owned),patch.object(R.L.time,'monotonic',return_value=0.),patch.object(R.L.signal,'getsignal',return_value=0),patch.object(R.L.signal,'signal'),patch.object(R.L.signal,'setitimer'),self.assertRaisesRegex(RuntimeError,'returned7|returned 7'):
                R.L.command(['fake'],'failed',cpu=False)
            owned.stop.assert_called_once()

if __name__=='__main__':unittest.main(verbosity=2)
