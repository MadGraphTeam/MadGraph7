"""CMS source, receipt and mocked ownership boundaries; no external child runs."""
import ast
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import Mock,patch

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('cms_test_supervisor',HERE/'run.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)

class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.no_child=patch.object(R.L.subprocess,'Popen',side_effect=AssertionError('No real child admitted by source tests'))
        self.no_child.start();self.addCleanup(self.no_child.stop)

    def test_default_is_plan_without_creating_or_sealing(self):
        with patch.object(R.sys,'argv',['run.py']),patch.object(R,'seal') as seal,contextlib.redirect_stdout(io.StringIO()) as out:
            R.main()
        self.assertFalse(seal.called);self.assertEqual(json.loads(out.getvalue())['requests'],6)

    def test_execute_requires_stage(self):
        with patch.object(R.sys,'argv',['run.py','--execute']),self.assertRaisesRegex(RuntimeError,'Explicit stage'):
            R.main()

    def test_fresh_roots_required_before_materializing(self):
        with patch.object(R.prerequisites,'check_donors'),patch.object(R.REPORT.__class__,'exists',return_value=True),patch.object(R,'materialize') as materialize,self.assertRaisesRegex(RuntimeError,'Fresh namespaces'):
            R.prepare()
        materialize.assert_not_called()

    def test_only_preflight_executes_masses_or_model(self):
        source=(HERE/'run.py').read_text();tree=ast.parse(source)
        owners=[n.name for n in tree.body if isinstance(n,ast.FunctionDef) and '--masses' in ast.get_source_segment(source,n)]
        self.assertEqual(owners,['preflight'])
        for name in ('prepare','prepare_process','materialize','build_probe'):
            node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
            self.assertNotIn('runtime',[n.func.id for n in ast.walk(node) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)])

    def test_all_six_frozen_literal_cards_render_exactly(self):
        for request in R.PLAN['requests']:
            with tempfile.TemporaryDirectory() as tmp:
                output=Path(tmp);(output/'Cards').mkdir();(output/'SubProcesses').mkdir()
                expected=R.P/'cards'/(request['source']+'-'+str(request['backend'])+'-MadLoopParams.dat')
                (output/'Cards/MadLoopParams.dat').write_bytes(expected.read_bytes())
                result=R.configure_worker(output,request)
                self.assertEqual((output/'Cards/MadLoopParams.dat').read_bytes(),expected.read_bytes())
                self.assertEqual(result['requested']['mlstabthres'],1e-3)
                self.assertEqual(result['requested']['mlreductionlib'],str(request['backend']))

    def test_unknown_card_field_requires_review(self):
        with self.assertRaisesRegex(RuntimeError,'Uncontrolled'):
            R.render_card('#UnknownFutureField\n1\n',{},'candidate')

    def test_current_model_proof_is_exact_card_and_complex(self):
        raw='1\n80.419 2.09291 6467.215560999999 -168.30972928999998\n91.188 2.49877 8315.251344 -227.85783876\n'
        self.assertEqual(R.validate_model_probe(raw)['CMSParam'],1.)
        for bad in [raw.replace('80.419','80.420'),raw.replace('2.49877','2.49876'),raw.replace('-227.85783876','227.85783876'),raw.replace('1\n','0\n',1),raw+'0\n']:
            with self.assertRaises(RuntimeError):R.validate_model_probe(bad)

    def test_model_proof_rejects_nonfinite(self):
        with self.assertRaisesRegex(RuntimeError,'Nonfinite'):
            R.validate_model_probe('1\n80.419 2.09291 nan -168.30972928999998\n91.188 2.49877 8315.251344 -227.85783876\n')

    def test_native_request_rechecks_its_fresh_model_proof_receipt(self):
        raw='1\n80.419 2.09291 6467.215560999999 -168.30972928999998\n91.188 2.49877 8315.251344 -227.85783876\n'
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'builds/candidate-case';directory.mkdir(parents=True)
            (directory/'build.json').write_text('{}');(directory/'cms_masses.dat').write_text(raw)
            pre={'build_sha256':R.sha(directory/'build.json'),'physical_card_sha256':R.PLAN['physical_card_sha256'],'cms_masses_sha256':R.sha(directory/'cms_masses.dat'),'complex_model':R.validate_model_probe(raw),'probe_runtime':{'files':{}}}
            (directory/'preflight.json').write_text(json.dumps(pre))
            with patch.object(R,'REPORT',Path(tmp)):
                value,checksum=R.preflight_record('candidate','case');self.assertEqual(value,pre)
                self.assertEqual(checksum,R.sha(directory/'preflight.json'))
                (directory/'cms_masses.dat').write_text(raw.replace('80.419','80.420'))
                with self.assertRaisesRegex(RuntimeError,'raw model proof changed'):R.preflight_record('candidate','case')

    def test_loader_uses_actual_interpreter(self):
        with tempfile.TemporaryDirectory() as tmp:
            loader=Path(tmp)/'loader';loader.write_bytes(b'bound');exe=Path(tmp)/'check';exe.write_bytes(b'exe')
            with patch.object(R.L,'interpreter',return_value=loader),patch.object(R.L,'capture',return_value=str(loader)) as capture:
                rt=R.runtime(exe)
            self.assertEqual(capture.call_args.args[0],[loader,'--list',exe]);self.assertIn(str(loader),rt['files'])

    def test_private_links_admit_only_private_or_bound_external(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp);work=base/'work';output=work/'workers/candidate-1-auto/case';output.mkdir(parents=True)
            external=base/'external';external.write_bytes(b'library');(output/'library').symlink_to(external)
            with patch.object(R,'WORK',work),patch.object(R,'installed',return_value={}):
                with self.assertRaisesRegex(RuntimeError,'outside private'):R.validate_private_links(output,'candidate')
            with patch.object(R,'WORK',work),patch.object(R,'installed',return_value={str(external):{'sha256':R.sha(external)}}):
                R.validate_private_links(output,'candidate')
            external.write_bytes(b'changed')
            with patch.object(R,'WORK',work),patch.object(R,'installed',return_value={str(external):{'sha256':'0'*64}}):
                with self.assertRaisesRegex(RuntimeError,'outside private'):R.validate_private_links(output,'candidate')

    def test_build_receipt_mutation_fails_before_using_products(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)/'builds/candidate-case';directory.mkdir(parents=True)
            (directory/'generation.json').write_text('{}');(directory/'prelink.json').write_text('{}')
            (directory/'build.json').write_text(json.dumps({'generation_sha256':R.sha(directory/'generation.json'),'prelink_sha256':'0'*64}))
            with patch.object(R,'REPORT',Path(tmp)),self.assertRaisesRegex(RuntimeError,'Prelink receipt'):
                R.build_record('candidate',{'case':{'id':'case'}})

    def test_raw_cli_driver_bytes_survive_diagnostic_substitution(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';directory=output/'SubProcesses/P0';directory.mkdir(parents=True)
            (output/'Source').mkdir();(output/'Cards').mkdir();driver=directory/'check_sa.f';driver.write_bytes(b'original CLI source')
            (output/'Cards/param_card.dat').write_bytes(b'original physical card')
            with patch.object(R,'REPORT',root/'report'):
                science,archive=R.retain_raw_cli(output,'candidate-case')
            driver.write_bytes(b'diagnostic worker')
            retained=Path(archive['directory'])/'SubProcesses/P0/check_sa.f'
            self.assertEqual(retained.read_bytes(),b'original CLI source')
            self.assertEqual(science['SubProcesses/P0/check_sa.f'],R.sha(retained));R.check_hashes(archive['bound_files'])
            retained.write_bytes(b'corrupt')
            with self.assertRaisesRegex(RuntimeError,'Bound link/compiler input changed'):R.check_hashes(archive['bound_files'])

    def test_probe_is_prebound_before_its_link_and_has_own_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);output=root/'output';directory=output/'SubProcesses/P0';directory.mkdir(parents=True)
            (output/'Source/MODEL').mkdir(parents=True);(output/'lib').mkdir()
            (output/'cms_probe.f').write_bytes((R.P/'protocol/cms_probe.f').read_bytes());(output/'lib/libmodel.a').write_bytes(b'fresh MODEL')
            compiler=root/'gfortran';compiler.write_bytes(b'pinned compiler')
            def command(argv,label,cwd=R.ROOT,seconds=600):
                if '-c' in argv:(output/'cms_probe.o').write_bytes(b'fresh probe object')
                elif '-print-search-dirs' in argv:return 'libraries: ='+str(root)
                elif '-o' in argv:
                    receipt=root/'report/builds/candidate-case/cms-probe-prelink.json'
                    self.assertTrue(receipt.is_file());self.assertTrue(json.loads(receipt.read_text())['completed_before_link'])
                    (output/'cms_probe').write_bytes(b'fresh executable')
                    (output/'cms-probe.map').write_text('LOAD '+str(output/'cms_probe.o')+'\nLOAD '+str(output/'lib/libmodel.a')+'\n')
                return ''
            with patch.object(R,'REPORT',root/'report'),patch.object(R,'cmd',side_effect=command),patch.object(R.q,'external_inputs',return_value={}),patch.object(R.L,'ACTIVE',types.SimpleNamespace(end=100.)),patch.object(R.shutil,'which',return_value=str(compiler)),patch.object(R.L.linker,'wrapper_flags',return_value={'flags':'','inputs':{}}),patch.object(R.L.linker,'prebind',return_value={'files':{}}):
                result=R.build_probe(output,'candidate-case',directory)
                R.check_probe(result)
            self.assertEqual(set(result['actual_LOAD']),{str(output/'cms_probe.o'),str(output/'lib/libmodel.a')})

    def test_failed_outer_execution_never_preserves_success(self):
        result,stop=R.admit_completion({'numerically_qualified':True},TimeoutError('cap'),{'cleanup':{'survivors':[]},'cleanup_exception':None})
        self.assertFalse(result['numerically_qualified']);self.assertIsNone(stop)

    def test_cleanup_controller_and_map_failures_stop(self):
        valid={'cleanup':{'survivors':[]},'cleanup_exception':None}
        for result,error,receipt in [({'numerically_qualified':True},None,None),({'numerically_qualified':True},None,{'cleanup':{'survivors':[1]}}),({'numerically_qualified':False,'fatal_controller_failure':True},None,valid),({'numerically_qualified':False,'runtime_identity_failure':True},None,valid),({'numerically_qualified':False},RuntimeError('bad controller'),valid)]:
            result,stop=R.admit_completion(result,error,receipt);self.assertTrue(stop)

    def test_owned_timeout_cleans_and_retains_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            process=Mock(pid=42,returncode=-9);process.wait.side_effect=R.L.subprocess.TimeoutExpired(['fake'],0)
            owned=Mock();owned.stop.return_value={'survivors':[]};active=types.SimpleNamespace(end=100.,handoff=False,deferred=False)
            with patch.object(R.L,'REPORT',Path(tmp)),patch.object(R.L,'ACTIVE',active),patch.object(R.L,'REQUESTS',Mock()),patch.object(R.L,'remaining',return_value=.01),patch.object(R.L,'environment',return_value={}),patch.object(R.L.subprocess,'Popen',return_value=process),patch.object(R.L.cleanup,'OwnedChild',return_value=owned),patch.object(R.L.time,'monotonic',side_effect=iter(range(1000))),patch.object(R.L.signal,'getsignal',return_value=0),patch.object(R.L.signal,'signal'),patch.object(R.L.signal,'setitimer'),self.assertRaises(TimeoutError):
                R.L.command(['fake'],'timeout',cpu=False)
            owned.stop.assert_called_once();self.assertTrue((Path(tmp)/'execution/timeout.json').exists())

    def test_prelink_requires_completed_inventory_before_actual_LOAD(self):
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'lib.a';file.write_bytes(b'archive')
            with self.assertRaises(Exception):R.L.linker.require_prebound_load(file,{})
            R.L.linker.require_prebound_load(file,{str(file):R.sha(file)})

    def test_primary_agreement_stays_null_without_all_current_backends(self):
        result=R.summarize_results([],0)
        self.assertFalse(result['five_current_backend_complete']);self.assertIsNone(result['primary_policy_agreement'])

    def test_comparison_separates_ordinary_target_and_raw_pole_errors(self):
        left=[{'point':p,'order':o,'coefficients':[1.,2.,3.,4.]} for p in range(1,22) for o in (0,1)]
        right=[dict(row,coefficients=list(row['coefficients'])) for row in left];right[-1]['coefficients'][2]+=1e-5
        result=R.compare(left,right)
        self.assertTrue(result['ordinary_point_comparison']['ordinary_target_passed'])
        self.assertFalse(result['all_stream_1e8_target_passed']);self.assertGreater(result['residuals'][-2]['absolute_error'],0)

if __name__=='__main__':unittest.main(verbosity=2)
