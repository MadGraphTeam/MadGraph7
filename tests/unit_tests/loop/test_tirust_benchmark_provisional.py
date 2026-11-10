"""Provisional observations retain physics and cannot become acceptance gates."""
from contextlib import ExitStack
import json
import math
from pathlib import Path
import shutil
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import tirust_benchmark as bench
import tirust_benchmark_provisional as provisional


class ProvisionalTests(unittest.TestCase):
    def fake_controller(self, mode):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        base=Path(temporary.name); report=base/'report';work=base/'work';report.mkdir();work.mkdir()
        protocols=report/'protocol-sources';protocols.mkdir()
        shutil.copy2(ROOT/'docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-qualification-2/process_cleanup.py',protocols/'process_cleanup.py')
        row={'id':'01-fake','case':'fake','cap_seconds':3.2 if mode=='spawn-timeout' else 1.2}
        spec={'schedule':[row],'comparison_cleanup':{'cooperative_seconds':.05,'terminate_seconds':.05,'kill_seconds':.2}}
        prep={'clones':{'fake':{'baseline':{},'candidate':{}}}}
        stack=ExitStack();self.addCleanup(stack.close)
        stack.enter_context(patch.object(provisional,'locations',return_value=(report,work)))
        stack.enter_context(patch.object(provisional,'verify_prepared',return_value=(spec,prep)))
        stack.enter_context(patch.object(provisional,'verify_prepared_clone'))
        stack.enter_context(patch.object(provisional,'TOTAL_SECONDS',4. if mode=='spawn-timeout' else 2.))
        stack.enter_context(patch.object(provisional,'CLEANUP_RESERVE',2.5 if mode=='spawn-timeout' else .5))
        stack.enter_context(patch.object(bench,'environment',return_value=None))
        real_popen=subprocess.Popen
        created=[]
        self.addCleanup(lambda:[p.poll() for p in created])
        sleeper="import signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
        child="import subprocess,sys,os,time;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c',sys.argv[1]],start_new_session=True);Path(sys.argv[2]).write_text(str(p.pid));time.sleep(30) if sys.argv[3]=='sleep' else None"
        def spawn(command,**kwargs):
            if mode=='postcheck':
                program="from pathlib import Path;import sys;Path(sys.argv[1]).write_text('{}')"
                return real_popen([sys.executable,'-B','-c',program,str(report/'comparisons/01-fake/result.json')],**kwargs)
            process=real_popen([sys.executable,'-B','-c',child,sleeper,str(base/'worker.pid'),
                               'sleep' if mode in ('spawn-timeout','spawn-sigterm') else 'exit'],**kwargs)
            created.append(process)
            if mode in ('spawn-timeout','spawn-sigterm'):
                until=time.monotonic()+.4
                while not (base/'worker.pid').exists() and time.monotonic()<until:time.sleep(.005)
                if mode=='spawn-sigterm':os.kill(os.getpid(),signal.SIGTERM)
                time.sleep(5 if mode=='spawn-timeout' else .01)
            return process
        stack.enter_context(patch.object(provisional.subprocess,'Popen',side_effect=spawn))
        return base,report,protocols

    def test_schedule_contains_every_chain_before_all_standalone_rows(self):
        config = {'cases': [{'id': name} for name in provisional.CASE_ORDER]}
        chains = {name: '8|6|7|1' for name in provisional.CASE_ORDER}
        rows = provisional.schedule(config, chains)
        self.assertEqual(len(rows), 66)
        self.assertEqual([r['case'] for r in rows[:11]], list(provisional.CASE_ORDER))
        self.assertTrue(all(r['kind'] == 'chain' and r['cap_seconds'] == 120 for r in rows[:11]))
        self.assertEqual([(r['case'], r['backend']) for r in rows[11:21]],
            [(name, backend) for name in ('qcd_n1_uux_z', 'ew_n2_epem_mupmum') for backend in (1,6,7,8,9)])
        self.assertEqual(rows[0]['chains'], {'baseline':[8,6,7,1], 'candidate':[9,8,6,7,1]})
        self.assertEqual(rows[15]['chains'], {'baseline':[1], 'candidate':[9]})
        with self.assertRaises(ValueError):
            provisional.schedule({'cases': config['cases'][:-1]}, chains)

    def test_three_pairs_are_descriptive_even_when_slow(self):
        value = provisional.descriptive({'baseline':[1,1,1], 'candidate':[2,2,2]})
        self.assertEqual(value['geometric_mean_paired_ratio'], 2)
        self.assertFalse(value['performance_acceptance_eligible'])
        for key in ('passed','lower_95','upper_95','threshold','acceptance_eligible'):
            self.assertNotIn(key, value)
        for bad in ([1,1], [1,1,0], [1,1,math.inf]):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                provisional.descriptive({'baseline':[1,1,1], 'candidate':bad})

    def test_private_adapter_preserves_all_original_guards_and_restores_on_error(self):
        original = bench.WORK
        calls = []
        def guard(stage, **kwargs):
            calls.append((stage, bench.WORK, kwargs))
            self.assertEqual(bench.WORK, original)
            raise ValueError('guard refusal')
        with patch.object(bench, 'campaign_guard', guard):
            with self.assertRaisesRegex(ValueError, 'guard refusal'):
                with provisional.private_execution(Path('/tmp/private-provisional-test')):
                    self.assertNotEqual(bench.WORK, original)
                    bench.campaign_guard('worker', inputs={'exact':'actual'})
            self.assertIs(bench.campaign_guard, guard)
        self.assertEqual(bench.WORK, original)
        self.assertEqual(calls, [('worker', original, {'inputs':{'exact':'actual'}})])

    def test_capture_never_attributes_an_old_unstarted_stream(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); report=base/'report';report.mkdir()
            directory=base/'work/outputs/candidate/case/SubProcesses/P0_case';directory.mkdir(parents=True)
            (directory/'warmup.dat').write_text('previous comparison')
            provisional.capture_failure_raw(base/'work', 'case', report)
            self.assertFalse((report/'last-raw').exists())
            (report/'candidate-raw-cleared.json').write_text('{}')
            provisional.capture_failure_raw(base/'work', 'case', report)
            self.assertEqual((report/'last-raw/candidate/warmup.dat').read_text(), 'previous comparison')

    def test_comparison_deadline_cleans_observed_detached_workers(self):
        # This is a Python sleeper, not an amplitude or native reducer.
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); report=base/'report'; work=base/'work';report.mkdir();work.mkdir()
            protocols=report/'protocol-sources';protocols.mkdir()
            shutil.copy2(ROOT/'docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-qualification-2/process_cleanup.py',
                         protocols/'process_cleanup.py')
            row={'id':'01-fake','case':'fake','cap_seconds':1.2}
            spec={'schedule':[row], 'comparison_cleanup':{'cooperative_seconds':.1,'terminate_seconds':.1,'kill_seconds':.1}}
            real_popen=subprocess.Popen
            sleeper="import os,signal,time;signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
            child="import subprocess,sys,time,signal;from pathlib import Path;p=subprocess.Popen([sys.executable,'-B','-c',sys.argv[1]],start_new_session=True);Path(sys.argv[2]).write_text(str(p.pid));signal.signal(signal.SIGINT,signal.SIG_IGN);signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)"
            def spawn(command, **kwargs):
                return real_popen([sys.executable,'-B','-c',child,sleeper,str(base/'worker.pid')], **kwargs)
            with ExitStack() as stack:
                stack.enter_context(patch.object(provisional,'locations',return_value=(report,work)))
                stack.enter_context(patch.object(provisional,'verify_prepared',return_value=(spec,{})))
                stack.enter_context(patch.object(provisional,'TOTAL_SECONDS',2.))
                stack.enter_context(patch.object(provisional,'CLEANUP_RESERVE',.5))
                stack.enter_context(patch.object(bench,'environment',return_value=None))
                stack.enter_context(patch.object(provisional.subprocess,'Popen',side_effect=spawn))
                start=time.monotonic();result=provisional.execute('fake')
            self.assertLess(time.monotonic()-start,3.)
            self.assertEqual(result['rows'][0]['status'],'unavailable_deadline')
            cleanup=json.loads((report/'comparisons/01-fake/cleanup.json').read_text())
            self.assertEqual(cleanup['survivors'], [])
            pid=int((base/'worker.pid').read_text())
            records={r['pid']:r for r in cleanup['observed_owned_processes']}
            self.assertIn(pid,records)
            helper=provisional.checked_module('cleanup_for_test',protocols/'process_cleanup.py',provisional.CLEANUP_SHA)
            self.assertFalse(helper.matches(records[pid]))
            with patch.object(provisional,'locations',return_value=(report,work)):
                with self.assertRaises(FileExistsError):
                    provisional.execute('fake')

    def test_global_budget_retains_unattempted_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/'report';report.mkdir();protocols=report/'protocol-sources';protocols.mkdir()
            shutil.copy2(ROOT/'docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-qualification-2/process_cleanup.py',protocols/'process_cleanup.py')
            spec={'schedule':[{'id':'a'},{'id':'b'}]}
            with patch.object(provisional,'locations',return_value=(report,Path(tmp)/'work')), \
                 patch.object(provisional,'verify_prepared',return_value=(spec,{})), \
                 patch.object(provisional,'TOTAL_SECONDS',.01), \
                 patch.object(provisional.subprocess,'Popen') as launch:
                result=provisional.execute('fake')
            launch.assert_not_called()
            self.assertEqual([r['status'] for r in result['rows']],['not_run_budget','not_run_budget'])
            self.assertFalse(result['performance_acceptance_eligible'])

    def test_stalled_preflight_is_interrupted_by_overall_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp);start=time.monotonic()
            with patch.object(provisional,'locations',return_value=(report,report/'work')), \
                 patch.object(provisional,'TOTAL_SECONDS',.05), \
                 patch.object(provisional,'verify_prepared',side_effect=lambda label:time.sleep(5)):
                result=provisional.execute('fake')
            self.assertLess(time.monotonic()-start,.5)
            self.assertEqual(result['status'],'closed_budget')
            self.assertEqual(result['failure']['error_type'],'BudgetExpired')

    def test_start_publication_is_under_overall_deadline(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp); write=provisional.exclusive
            def stalled(path,value):
                if Path(path).name=='run-start.json':time.sleep(5)
                return write(path,value)
            with patch.object(provisional,'locations',return_value=(report,report/'work')), \
                 patch.object(provisional,'TOTAL_SECONDS',.05), \
                 patch.object(provisional,'exclusive',side_effect=stalled), \
                 patch.object(provisional,'verify_prepared') as verify:
                start=time.monotonic();result=provisional.execute('fake')
            verify.assert_not_called()
            self.assertLess(time.monotonic()-start,.5)
            self.assertIn('administrative-after-deadline',result['publication_scope'])
            self.assertEqual(result['status'],'closed_budget')

    def test_summary_publication_after_alarm_is_explicitly_administrative(self):
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp);protocols=report/'protocol-sources';protocols.mkdir()
            shutil.copy2(ROOT/'docs/benchmarkfolder/campaigns/split-dp-terminal-1/numerical-qualification-2/process_cleanup.py',protocols/'process_cleanup.py')
            write=provisional.exclusive; stalled=[False]
            def publish(path,value):
                if Path(path).name=='summary.json' and not stalled[0]:
                    stalled[0]=True;time.sleep(5)
                return write(path,value)
            with patch.object(provisional,'locations',return_value=(report,report/'work')), \
                 patch.object(provisional,'TOTAL_SECONDS',.05), \
                 patch.object(provisional,'exclusive',side_effect=publish), \
                 patch.object(provisional,'verify_prepared',return_value=({'schedule':[]},{})):
                start=time.monotonic();result=provisional.execute('fake')
            self.assertLess(time.monotonic()-start,.5)
            self.assertIn('administrative-after-deadline',result['publication_scope'])
            self.assertFalse(result['performance_acceptance_eligible'])
            self.assertEqual(result['status'],'closed_budget')

    def test_late_cleanup_uses_reserved_original_cap_not_fresh_seconds(self):
        base,report,protocols=self.fake_controller('fast-exit')
        load=provisional.checked_module;stops=[]
        def module(name,path,checksum):
            helper=load(name,path,checksum);original=helper.OwnedChild.stop
            def stop(owned,**options):
                stops.append((time.monotonic(),options))
                if len(stops)<=2:time.sleep(5)
                return original(owned,**options)
            helper.OwnedChild.stop=stop
            return helper
        with patch.object(provisional,'checked_module',side_effect=module):
            start=time.monotonic();result=provisional.execute('fake')
        self.assertLess(time.monotonic()-start,1.4)
        self.assertEqual(result['status'],'closed_budget')
        self.assertEqual(stops[-1][1]['cooperative_seconds'],0.)
        self.assertEqual(stops[-1][1]['terminate_seconds'],0.)
        admission=json.loads((report/'comparisons/01-fake/admission.json').read_text())
        self.assertLessEqual(stops[-1][0]+stops[-1][1]['kill_seconds'],admission['finish_deadline_monotonic']+.001)
        cleanup=json.loads((report/'comparisons/01-fake/cleanup-interruption.json').read_text())
        self.assertEqual(cleanup['survivors'],[])
        helper=load('cleanup_late_test',protocols/'process_cleanup.py',provisional.CLEANUP_SHA)
        self.assertFalse(any(helper.matches(r) for r in cleanup['observed_owned_processes']))

    def test_comparison_publication_cannot_extend_original_cap(self):
        self.fake_controller('postcheck');write=provisional.exclusive
        def publish(path,value):
            if Path(path).name=='completion.json':time.sleep(5)
            return write(path,value)
        with patch.object(provisional,'verify_comparison',return_value={'numerical_pass':True}), \
             patch.object(provisional,'exclusive',side_effect=publish):
            start=time.monotonic();result=provisional.execute('fake')
        self.assertLess(time.monotonic()-start,1.5)
        self.assertEqual(result['status'],'closed_budget')
        self.assertEqual(result['rows'][0]['status'],'not_completed_budget')
        self.assertFalse(result['performance_acceptance_eligible'])

    def test_stalled_postcheck_is_inside_comparison_cap(self):
        self.fake_controller('postcheck');start=time.monotonic()
        with patch.object(provisional,'verify_comparison',side_effect=lambda run:time.sleep(5)):
            result=provisional.execute('fake')
        self.assertLess(time.monotonic()-start,1.5)
        self.assertEqual(result['rows'][0]['status'],'unavailable_deadline')

    def test_fast_exiting_leader_cannot_orphan_detached_worker(self):
        base,report,protocols=self.fake_controller('fast-exit')
        result=provisional.execute('fake')
        pid=int((base/'worker.pid').read_text())
        cleanup=json.loads((report/'comparisons/01-fake/cleanup.json').read_text())
        records={r['pid']:r for r in cleanup['observed_owned_processes']}
        self.assertIn(pid,records)
        helper=provisional.checked_module('cleanup_fast_exit',protocols/'process_cleanup.py',provisional.CLEANUP_SHA)
        self.assertFalse(helper.matches(records[pid]))
        self.assertEqual(cleanup['survivors'],[])
        self.assertEqual(result['rows'][0]['status'],'unavailable_or_numerical_failure')

    def test_deadline_during_popen_return_window_reaps_adopted_children(self):
        base,report,protocols=self.fake_controller('spawn-timeout')
        result=provisional.execute('fake')
        cleanup=json.loads((report/'comparisons/01-fake/cleanup-unassigned.json').read_text())
        helper=provisional.checked_module('cleanup_spawn_timeout',protocols/'process_cleanup.py',provisional.CLEANUP_SHA)
        self.assertFalse(any(helper.matches(r) for r in cleanup['observed_owned_processes']))
        self.assertEqual(cleanup['survivors'],[])
        self.assertEqual(result['rows'][0]['status'],'unavailable_deadline')

    def test_sigterm_during_spawn_unwinds_and_cleans_child(self):
        base,report,protocols=self.fake_controller('spawn-sigterm')
        with self.assertRaises(InterruptedError):
            provisional.execute('fake')
        cleanup=json.loads((report/'comparisons/01-fake/cleanup-interruption.json').read_text())
        helper=provisional.checked_module('cleanup_spawn_sigterm',protocols/'process_cleanup.py',provisional.CLEANUP_SHA)
        self.assertFalse(any(helper.matches(r) for r in cleanup['observed_owned_processes']))
        self.assertEqual(cleanup['survivors'],[])


class PortableReplayTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.report = Path(temp.name)
        self.run = self.report / 'comparisons/01-case'
        self.run.mkdir(parents=True)
        self.row = {'case':'case','chains':{'baseline':[1],'candidate':[9]}}
        self.binding = {'id':'frozen','manifest_sha256':'manifest'}
        bench.dump(self.report/'specification.json', {'schedule':[self.row],'campaign_binding':self.binding})
        config = {'relative_tolerance':3e-6,'madloop_parameters':{},'candidate_parameters':{}}
        bench.dump(self.report/'cases.json', config)
        points = self.report/'points/case.dat'; points.parent.mkdir(); points.write_text('frozen100')
        points.with_suffix('.json').write_text('{"points":100}\n')
        self.result = {'schema':provisional.SCHEMA,'performance_acceptance_eligible':False,
                       'comparison':self.row,'numerical_pass':True,'observations':{},'calibration':{},
                       'calibration_loops':{},'inputs_after':{}}
        prepared = {}
        ready = {'points':100,'ndim':0,'nso':0,'nborn':0,'cold_seconds':1.}
        for source,backend,code in (('baseline',1,211),('candidate',9,219)):
            warm = self.run/(source+'-initialization');warm.mkdir()
            (warm/'dimensions.dat').write_text('0 0 0\n')
            (warm/'warmup.dat').write_text(''.join(f'{p} {p} 0 {code} 1 2 -3 -4 1e-10\n' for p in range(1,101)))
            bench.dump(self.run/(source+'-ready.json'),ready)
            bench.dump(self.run/(source+'-learning.json'),provisional.parse_warmup(warm,[backend],100,ready))
            values = {}
            for phase in ('calibrate-0','calibrate-1','repetition-0','repetition-1','repetition-2'):
                raw=self.run/'raw'/source/phase;raw.mkdir(parents=True)
                (raw/'dimensions.dat').write_text('0 0 0\n')
                (raw/'timing.dat').write_text('1.0 1.1\n')
                (raw/'values.dat').write_text(''.join(f'{p} 0 {code} 1 2 -3 -4 1e-10\n' for p in range(1,101)))
                (raw/'timed_status.dat').write_text(f'{code} 100\n')
                values[phase]=provisional.parse_outputs(raw,[backend],100,1)
            self.result['observations'][source]=[values[f'repetition-{i}'] for i in range(3)]
            self.result['calibration'][source]={'initial':values['calibrate-0'],'sustained':values['calibrate-1'],'sustained_loops':1}
            self.result['calibration_loops'][source]=1
            (self.run/(source+'-param_card.dat')).write_text('physical')
            (self.run/(source+'-MadLoopParams.dat')).write_text('numeric')
            compiled={'executable_sha256':'executable'}; generation={'source':source}
            bench.dump(self.run/(source+'-compilation.json'),compiled)
            bench.dump(self.run/(source+'-generation.json'),generation)
            prepared[source]={name+'_sha256':bench.digest(self.run/(source+'-'+name+'.json')) for name in ('compilation','generation')}
            runtime={'libraries':{},'archives':{},'metadata':{}}
            cards={'Cards/MadLoopParams.dat':bench.digest(self.run/(source+'-MadLoopParams.dat'))}
            inputs={'campaign_binding':self.binding,'compilation':compiled,'generation':generation,'runtime':runtime,
                    'settings':{'mlreductionlib':str(backend)},'numerical_cards':cards,
                    'points_sha256':bench.digest(points),'points_metadata_sha256':bench.digest(points.with_suffix('.json')),
                    'param_card_sha256':bench.digest(self.run/(source+'-param_card.dat'))}
            bench.dump(self.run/(source+'-inputs.json'),inputs)
            self.result['inputs_after'][source]={'runtime':runtime,'executable_sha256':'executable','numerical_cards':cards,'points_sha256':bench.digest(points)}
        bench.dump(self.report/'preparation.json',{'clones':{'case':prepared}})
        left=self.result['observations']['baseline'][0]['rows'];right=self.result['observations']['candidate'][0]['rows']
        self.result['parity']=[bench.compare_rows(left,right)]*4
        self.result['timings']={s:[x['seconds_per_point'] for x in rows] for s,rows in self.result['observations'].items()}
        self.result['statistics']=provisional.descriptive(self.result['timings'])
        self.result['actual_backend_precision_counts']={s:bench.aggregate_usage(v) for s,v in self.result['observations'].items()}
        self.result['evidence_files']={str(p.relative_to(self.run)):bench.digest(p) for p in self.run.rglob('*') if p.is_file()}
        bench.dump(self.run/'result.json',self.result)

    def test_complete_full100_raw_result_replays(self):
        value=provisional.verify_comparison(self.run)
        self.assertEqual(value['statistics']['paired_repetitions'],3)
        self.assertFalse(value['performance_acceptance_eligible'])

    def test_raw_changes_are_detected(self):
        path=self.run/'raw/candidate/repetition-2/values.dat'
        path.write_text(path.read_text().replace('1 2 -3','1 4 -3'))
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            provisional.verify_comparison(self.run)

    def test_fabricated_numerical_verdict_and_ci_are_rejected(self):
        self.result['numerical_pass']=False
        bench.dump(self.run/'result.json',self.result)
        with self.assertRaisesRegex(ValueError,'Numerical comparison'):
            provisional.verify_comparison(self.run)
        self.result['numerical_pass']=True
        self.result['statistics']['upper_95']=1.01
        bench.dump(self.run/'result.json',self.result)
        with self.assertRaisesRegex(ValueError,'Descriptive timing'):
            provisional.verify_comparison(self.run)

    def test_changed_prepared_build_cannot_relabel_worker(self):
        prep=provisional.read(self.report/'preparation.json')
        prep['clones']['case']['candidate']['compilation_sha256']='changed'
        bench.dump(self.report/'preparation.json',prep)
        with self.assertRaisesRegex(ValueError,'prepared clone'):
            provisional.verify_comparison(self.run)


if __name__ == '__main__':
    unittest.main()
