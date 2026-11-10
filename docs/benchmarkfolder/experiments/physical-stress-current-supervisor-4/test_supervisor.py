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
Q=R.prerequisites

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
            expected=original.replace('/.tirust-build/physical-stress-current-1/','/.tirust-build/physical-stress-current-4/')
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
    def test_output_parent_matches_exporter_mkdir_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'exports/candidate/case'
            with self.assertRaises(FileNotFoundError):os.mkdir(output)
            R.prepare_output_parent(output)
            self.assertTrue(output.parent.is_dir());self.assertFalse(output.exists())
            os.mkdir(output)
            with self.assertRaisesRegex(RuntimeError,'Fresh CLI output'):R.prepare_output_parent(output)
    def test_zero_exit_console_fatal_markers_are_rejected(self):
        entry={'process_subdirectory':'SubProcesses/P0_test'}
        for marker in ('Error detected','interrupted in sub-command','Traceback (most recent call last)'):
            with self.assertRaisesRegex(RuntimeError,'fatal diagnostic'):R.validate_cli_output(Path('/absent'),entry,marker)
    def test_complete_export_files_are_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);entry={'process_subdirectory':'SubProcesses/P0_test'}
            files=['Cards/param_card.dat','Source/make_opts',*['SubProcesses/P0_test/'+n for n in ('check_sa.f','loop_matrix.f','CT_interface.f','improve_ps.f','makefile')]]
            for name in files:
                path=output/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('source fixture\n')
            for scope in ('MODEL','DHELAS'):(output/'Source'/scope).mkdir()
            R.validate_cli_output(output,entry,'Successful export')
            (output/'SubProcesses/P0_test/loop_matrix.f').unlink()
            with self.assertRaisesRegex(RuntimeError,'output incomplete'):R.validate_cli_output(output,entry,'Successful export')

class PrerequisiteTests(unittest.TestCase):
    def fixture(self,root):
        real=root/'real-ct';real.mkdir()
        baseline=root/'baseline/vendor/CutTools';baseline.mkdir(parents=True)
        (baseline/'includects').symlink_to(real,target_is_directory=True)
        copies={}
        payloads={'libcts.a':b'archive','mpmodule.mod':b'module','compiler_version.log':b'15.2.0\n'}
        for name,data in payloads.items():
            path=real/name;path.write_bytes(data)
            donor=baseline/'includects'/name
            copies['vendor/CutTools/includects/'+name]={'donor':str(donor),'resolved_donor':str(path),'sha256':Q.digest(path),'size':len(data)}
        path=root/'iregi.a';path.write_bytes(b'per-revision archive')
        copies['vendor/IREGI/src/libiregi.a']={'donor':str(path),'resolved_donor':str(path),'sha256':Q.digest(path),'size':path.stat().st_size}
        return {'CT_version':'15.2.0','external':{},'sources':{'baseline':{'required':list(copies),'copies':copies}}}
    def test_explicit_symlink_directory_donor_is_dereferenced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root);mg=root/'private';mg.mkdir()
            with patch.object(Q,'DATA',data):
                result=Q.copy_private('baseline',mg);Q.check_private('baseline',mg)
            self.assertEqual(len(result),4)
            self.assertTrue((root/'baseline/vendor/CutTools/includects').is_symlink())
            for name,record in result.items():
                path=mg/name;self.assertTrue(path.is_file());self.assertFalse(path.is_symlink())
                self.assertEqual(Q.digest(path),record['sha256'])
            self.assertFalse((mg/'vendor/CutTools/includects').is_symlink())
    def test_missing_required_module_fails_before_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root);mg=root/'private';mg.mkdir()
            del data['sources']['baseline']['copies']['vendor/CutTools/includects/mpmodule.mod']
            with patch.object(Q,'DATA',data),self.assertRaisesRegex(RuntimeError,'Required prebuilt not declared'):
                Q.copy_private('baseline',mg)
            self.assertEqual(list(mg.iterdir()),[])
    def test_private_module_or_compiler_version_mutation_fails(self):
        for name in ('mpmodule.mod','compiler_version.log'):
            with self.subTest(name=name),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);data=self.fixture(root);mg=root/'private';mg.mkdir()
                with patch.object(Q,'DATA',data):
                    Q.copy_private('baseline',mg)
                    (mg/'vendor/CutTools/includects'/name).write_bytes(b'changed')
                    with self.assertRaisesRegex(RuntimeError,'Private prebuilt changed'):Q.check_private('baseline',mg)
    def test_donor_retarget_or_mutation_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root);mg=root/'private';mg.mkdir()
            path=root/'real-ct/mpmodule.mod';path.write_bytes(b'changed')
            with patch.object(Q,'DATA',data),self.assertRaisesRegex(RuntimeError,'Prebuilt dependency changed'):
                Q.copy_private('baseline',mg)
    def test_private_dependency_link_escape_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=self.fixture(root);mg=root/'private';mg.mkdir();(mg/'vendor').symlink_to(root/'baseline/vendor',target_is_directory=True)
            with patch.object(Q,'DATA',data),self.assertRaisesRegex(RuntimeError,'escapes clone'):Q.copy_private('baseline',mg)
    def test_per_revision_iregi_is_declared_separately(self):
        baseline=Q.DATA['sources']['baseline']['copies']['vendor/IREGI/src/libiregi.a']
        candidate=Q.DATA['sources']['candidate']['copies']['vendor/IREGI/src/libiregi.a']
        self.assertNotEqual(baseline['sha256'],candidate['sha256'])
        for source in ('baseline','candidate'):
            self.assertEqual(len(Q.DATA['sources'][source]['copies']),123)
            self.assertEqual(set(Q.DATA['sources'][source]['required']),{'vendor/CutTools/includects/libcts.a','vendor/CutTools/includects/mpmodule.mod','vendor/CutTools/includects/compiler_version.log','vendor/IREGI/src/libiregi.a'})
    def test_external_helper_and_shared_inputs_are_pinned(self):
        names=list(Q.DATA['external'])
        for suffix in ('ninja/bin/ninja-config','ninja/lib/libninja.so','collier/libcollier.so','ninja/include/mninja.mod','include/collier.mod','ninjars/lib/mninjars.mod'):
            self.assertEqual(sum(p.endswith(suffix) for p in names),1,suffix)
        helper=next(v for p,v in Q.DATA['external'].items() if p.endswith('ninja/bin/ninja-config'))
        self.assertTrue(helper['executable'])
    def export_fixture(self,root,source='candidate'):
        process=root/'SubProcesses/Ptest';process.mkdir(parents=True)
        (root/'SubProcesses/MadLoop_makefile_definitions').write_text('LOOP_INCLUDE = -I/pinned/module\n')
        values=' .TRUE., .FALSE., .TRUE., .FALSE., .FALSE.,\n     $ .TRUE., .TRUE., .TRUE.'+(',.TRUE.' if source=='candidate' else '')
        text='      DATA LOOPLIBS_AVAILABLE /'+values+'/\n'
        text+='      DATA LOOPLIBS_QPAVAILABLE(0:5) /.FALSE.,.TRUE.,4*.FALSE./\n'
        text+='      DATA LOOPLIBS_QPAVAILABLE(6) /.TRUE./\n      DATA LOOPLIBS_QPAVAILABLE(7) /.FALSE./\n      DATA LOOPLIBS_QPAVAILABLE(8) /.TRUE./\n'
        if source=='candidate':text+='      DATA LOOPLIBS_QPAVAILABLE(9) /.TRUE./\n'
        (process/'loop_matrix.f').write_text(text)
        return process/'loop_matrix.f'
    def test_full_generated_capabilities_and_fixed_form_ranges(self):
        for source in ('baseline','candidate'):
            with self.subTest(source=source),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);self.export_fixture(root,source)
                result=Q.check_generated(source,root,'SubProcesses/Ptest')
                self.assertTrue(result['qp_available'][6]);self.assertFalse(result['qp_available'][7])
                self.assertEqual(len(result['available']),9 if source=='candidate' else 8)
    def test_silent_ninja_qp_downgrade_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=self.export_fixture(root)
            path.write_text(path.read_text().replace('LOOPLIBS_QPAVAILABLE(6) /.TRUE.','LOOPLIBS_QPAVAILABLE(6) /.FALSE.'))
            with self.assertRaisesRegex(RuntimeError,'QP capability differs'):Q.check_generated('candidate',root,'SubProcesses/Ptest')
    def test_missing_tirust_or_duplicate_qp_declaration_rejected(self):
        for replacement in ('','      DATA LOOPLIBS_QPAVAILABLE(8) /.TRUE./\n'):
            with self.subTest(replacement=replacement),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path=self.export_fixture(root)
                path.write_text(path.read_text().replace('      DATA LOOPLIBS_QPAVAILABLE(9) /.TRUE./\n',replacement))
                with self.assertRaisesRegex(RuntimeError,'QP capability differs|Malformed QP'):Q.check_generated('candidate',root,'SubProcesses/Ptest')
    def test_unresolved_include_and_internal_dependency_tree_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.export_fixture(root)
            definition=root/'SubProcesses/MadLoop_makefile_definitions';definition.write_text('LOOP_INCLUDE = -I<Not_found_define_it_yourself>')
            with self.assertRaisesRegex(RuntimeError,'external include'):Q.check_generated('candidate',root,'SubProcesses/Ptest')
            definition.write_text('LOOP_INCLUDE = -I/pinned/module');(root/'Source/CutTools').mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError,'internal dependency tree'):Q.check_generated('candidate',root,'SubProcesses/Ptest')
    def test_attempted_dependency_rebuild_rejected(self):
        for marker in ('Compiling CutTools','Compiling IREGI','CutTools was compiled with a different'):
            with self.assertRaisesRegex(RuntimeError,'fatal diagnostic'):R.validate_cli_output(Path('/absent'),{},marker)

if __name__=='__main__':unittest.main(verbosity=2)
