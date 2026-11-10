"""Focused pure-source/mocked tests. No external command or reducer is run."""
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
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
            with patch.object(R.L,'REPORT',Path(tmp)),patch.object(R.L,'ACTIVE',active),patch.object(R.L,'REQUESTS',Mock()),patch.object(R.L,'remaining',return_value=.01),patch.object(R.L,'environment',return_value={}),patch.object(R.L.subprocess,'Popen',return_value=process),patch.object(R.L.cleanup,'OwnedChild',return_value=owned),patch.object(R.L.time,'monotonic',side_effect=iter(range(1000))),patch.object(R.L.signal,'getsignal',return_value=0),patch.object(R.L.signal,'signal'),patch.object(R.L.signal,'setitimer'),self.assertRaises(TimeoutError):
                R.L.command(['fake'],'timeout',cpu=False)
            owned.stop.assert_called_once();self.assertTrue((Path(tmp)/'execution/timeout.json').exists())
    def test_owned_nonzero_exit_also_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=Mock(pid=43,returncode=7);process.wait.return_value=7;owned=Mock();owned.stop.return_value={'survivors':[]}
            active=types.SimpleNamespace(end=100.,handoff=False,deferred=False)
            with patch.object(R.L,'REPORT',Path(tmp)),patch.object(R.L,'ACTIVE',active),patch.object(R.L,'REQUESTS',Mock()),patch.object(R.L,'remaining',return_value=10.),patch.object(R.L,'environment',return_value={}),patch.object(R.L.subprocess,'Popen',return_value=process),patch.object(R.L.cleanup,'OwnedChild',return_value=owned),patch.object(R.L.time,'monotonic',return_value=0.),patch.object(R.L.signal,'getsignal',return_value=0),patch.object(R.L.signal,'signal'),patch.object(R.L.signal,'setitimer'),self.assertRaisesRegex(RuntimeError,'returned7|returned 7'):
                R.L.command(['fake'],'failed',cpu=False)
            owned.stop.assert_called_once()
    def test_outer_timeout_never_preserves_child_success(self):
        result,stop=R.admit_completion({'numerically_qualified':True},TimeoutError('cap'),{'cleanup':{'survivors':[]},'cleanup_exception':None})
        self.assertFalse(result['numerically_qualified']);self.assertIsNone(stop)
    def test_cleanup_failure_stops_even_after_numerical_timeout(self):
        for execution in (None,{'cleanup':None,'cleanup_exception':'failed'}, {'cleanup':{'survivors':[42]},'cleanup_exception':None}):
            result,stop=R.admit_completion({'numerically_qualified':True},TimeoutError('cap'),execution)
            self.assertTrue(stop);self.assertFalse(result['numerically_qualified'])
    def test_controller_failure_stops(self):
        _,stop=R.admit_completion({'numerically_qualified':False},RuntimeError('python failed'),{'cleanup':{'survivors':[]},'cleanup_exception':None})
        self.assertTrue(stop)
    def map_line(self,path,permissions='r-xp'):
        st=path.stat()
        return '1000-2000 %s 00000000 %x:%x %d %s\n'%(permissions,os.major(st.st_dev),os.minor(st.st_dev),st.st_ino,path)
    def test_same_mapped_path_changed_bytes_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lib.so';path.write_bytes(b'\x7fELFfake')
            with self.assertRaisesRegex(ValueError,'ELF bytes'):
                N.classify_maps(self.map_line(path),{str(path):'0'*64})
            self.assertTrue(N.classify_maps(self.map_line(path),{str(path):R.sha(path)})['exact_expected'])
    def test_unknown_nonelf_and_deleted_maps_are_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data';path.write_bytes(b'notELF')
            with self.assertRaisesRegex(ValueError,'non-ELF'):N.classify_maps(self.map_line(path,'r--p'),{})
            with self.assertRaisesRegex(ValueError,'Deleted'):N.classify_maps(self.map_line(path).rstrip()+' (deleted)\n',{})
    def test_mapped_inode_mismatch_is_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'lib.so';path.write_bytes(b'\x7fELFfake')
            line=self.map_line(path).split();line[4]='0'
            with self.assertRaisesRegex(ValueError,'inode'):N.classify_maps(' '.join(line),{str(path):R.sha(path)})
    def test_build_receipt_mutation_is_rejected_before_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);directory=root/'builds/candidate-case';directory.mkdir(parents=True)
            (directory/'generation.json').write_text('{}');(directory/'prelink.json').write_text('{}')
            record={'generation_sha256':R.sha(directory/'generation.json'),'prelink_sha256':'0'*64}
            (directory/'build.json').write_text(json.dumps(record))
            with patch.object(R,'REPORT',root),self.assertRaisesRegex(RuntimeError,'Prelink receipt'):
                R.build_record('candidate',{'case':{'id':'case'}})
    def test_card_serialization_is_literal_and_explicitly_scoped(self):
        settings={'mlreductionlib':'9','mlstabthres':1e-6,'tirustcomputefinite':True}
        text='#MLReductionLib\n!6|7|1\n#MLStabThres\n!0.001\n#TIRustComputeFinite\n!.TRUE.\n'
        result,ignored=R.render_card(text,settings,'candidate')
        self.assertEqual(result,'#MLReductionLib\n9\n#MLStabThres\n1.000000d-06\n#TIRustComputeFinite\n.true.\n');self.assertEqual(ignored,{})
        with self.assertRaisesRegex(RuntimeError,'Required numerical'):R.render_card('',settings,'candidate')
        with self.assertRaisesRegex(RuntimeError,'Required numerical'):R.render_card('',{'mlstabthres':1e-6},'baseline')
        self.assertEqual(R.render_card('',{'tirustcomputefinite':True},'baseline')[1],{'tirustcomputefinite':True})
    def test_card_writer_does_not_import_shared_banner(self):
        tree=ast.parse((HERE/'run.py').read_text())
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in ('render_card','configure_worker'):
                self.assertFalse(any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(node)))
    def test_all_declared_campaign_native_artifacts_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            lib=Path(tmp)/'lib';lib.mkdir();(Path(tmp)/'Cargo.lock').write_text('changed')
            with patch.object(R.q,'installation',return_value={}),patch.object(R.q,'INSTALL',lib),patch.dict(R.PLAN,{'candidate_native_files':{'Cargo.lock':'0'*64}}),self.assertRaisesRegex(RuntimeError,'campaign artifact'):
                R.installed()
    def test_actual_launch_entrypoint_exists_in_both_archived_git_trees(self):
        self.assertEqual(R.CLI_ENTRYPOINT,'bin/madgraph')
        observed={}
        for source in ('baseline','candidate'):
            data=subprocess.run(['git','cat-file','blob',R.PLAN[source+'_revision']+':'+R.CLI_ENTRYPOINT],cwd=R.ROOT,capture_output=True,check=True,timeout=10).stdout
            ast.parse(data.decode())
            observed[source]=hashlib.sha256(data).hexdigest()
        failure=R.read(HERE.with_name('physical-stress-current-preparation-failure-1')/'retention.json')
        self.assertEqual(observed,{source:item['sha256'] for source,item in failure['entrypoints'].items()})
    def test_all_private_cli_cards_change_only_namespace(self):
        commands=sorted((HERE/'commands').glob('*.mg5'));self.assertEqual(len(commands),8)
        for path in commands:
            original=(R.P/'commands'/path.name).read_text()
            expected=original.replace('/.tirust-build/physical-stress-current-1/','/.tirust-build/physical-stress-current-2/')
            self.assertEqual(path.read_text(),expected)
            self.assertNotIn('/.tirust-build/physical-stress-current-1/',path.read_text())
        self.assertEqual(R.P.name,'physical-stress-current-source-proposal-1')
    def test_cli_history_is_private_without_home_override(self):
        with patch.object(R.q,'env',return_value={'PATH':'/fixture/bin','HOME':'/untouched-home'}),patch.object(R.shutil,'which',return_value='/bin/bash'):
            env=R.environment(R.ROOT)
        self.assertEqual(env['HOME'],'/untouched-home')
        self.assertEqual(env['XDG_STATE_HOME'],str(R.WORK/'xdg/state'))
        self.assertEqual(R.L.recorded_environment(env)['XDG_STATE_HOME'],env['XDG_STATE_HOME'])
    def test_entrypoint_admission_binds_actual_materialized_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);mg=root/'mg';(mg/'bin').mkdir(parents=True);entry=mg/'bin/madgraph';entry.write_text('tracked CLI\n')
            materialized={'revision':R.PLAN['candidate_revision'],'files':{'bin/madgraph':{'kind':'file','sha256':R.sha(entry)}}}
            (root/'materialized-candidate.json').write_text(json.dumps(materialized))
            with patch.object(R,'REPORT',root):
                admission=R.cli_admission(mg,'candidate',root/'input.mg5')
                self.assertEqual(admission['argv'],[R.sys.executable,'-B','./bin/madgraph',str(root/'input.mg5')])
                entry.write_text('changed CLI\n')
                with self.assertRaisesRegex(RuntimeError,'bytes differ'):R.cli_admission(mg,'candidate',root/'input.mg5')
                entry.unlink()
                with self.assertRaisesRegex(RuntimeError,'entrypoint missing'):R.cli_admission(mg,'candidate',root/'input.mg5')

if __name__=='__main__':unittest.main(verbosity=2)
