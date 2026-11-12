"""Focused contained-integrity/source/RAW/geometry/cache/runtime negative tests."""
import copy,gzip,json,os,shutil,tempfile
from pathlib import Path
import unittest
import replay as R
HERE=Path(__file__).resolve().parent
class PortableCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ix,cls.c,cls.d,cls.p=R.decode(HERE);cls.j=staticmethod(lambda n:json.loads(cls.d[n]));cls.t=staticmethod(lambda n:cls.d[n].decode())
        with R.blocked_apis(cls.d) as effects:
            cls.m=R.modules(cls.d);cls.points={cohort:cls.m['cache_worker'].point_words('source/run4/inputs/training.dat','source/run4/inputs/'+cohort+'.dat') for cohort in ('controls','critical20','critical21')}
        if effects:raise RuntimeError('Test setup attempted an effect')
        cls.plain=cls.t('run4/evidence/cacheall-plain-controls/driver.raw64');cls.observed=cls.t('run4/evidence/cacheall-observed-controls/driver.raw64');cls.normal=cls.m['normal_run_analysis'].parse_raw(cls.plain,lane='candidate',request_ids=[1,7,12],expected_points=cls.points['controls'])
    def observed_parse(self,text,cohort='controls'):
        ids={'controls':[1,7,12],'critical20':[20],'critical21':[21]}[cohort]
        plain=self.t('run4/evidence/cacheall-plain-'+cohort+'/driver.raw64')
        return self.m['observer_analysis'].analyze_observed(text,plain,lane='candidate',request_ids=ids,observed_pid=self.j('run4/validated-cacheall-observed-'+cohort+'.json')['pid'],plain_pid=self.j('run4/validated-cacheall-plain-'+cohort+'.json')['pid'],parse_normal=self.m['normal_run_analysis'].parse_raw,expected_points=self.points[cohort],compiled_dispatch_source=self.d['compiled-actual/observed/CT_interface.f'],cache_policy=-1,cache_budget=268435456)
    @staticmethod
    def mutate_line(text,tag,edit,*,after_run=False):
        lines=text.splitlines(keepends=True);ready=not after_run
        for i,line in enumerate(lines):
            if line.startswith('CALL RUN '):ready=True
            if ready and line.split() and line.split()[0]==tag:
                words=line.split();edit(words);lines[i]=' '.join(words)+'\n';return ''.join(lines)
        raise AssertionError('Actual mutation tag absent '+tag)
    def test_01_complete_under_effect_guard(self):
        with R.blocked_apis(self.d) as effects:result=R.semantic(self.d,self.c,self.p)
        self.assertFalse(effects);self.assertEqual(result,self.ix['claims']);self.assertEqual(result['observed_genuine_QP_frames'],8)
    def test_02_relocation(self):
        with tempfile.TemporaryDirectory(prefix='CMS-cache-relocated-') as d:
            target=Path(d)/'capsule';shutil.copytree(HERE,target);self.assertEqual(R.verify(target),self.ix['claims'])
    def test_03_physical_tamper(self):
        with tempfile.TemporaryDirectory(prefix='CMS-cache-tamper-') as d:
            target=Path(d)/'capsule';shutil.copytree(HERE,target);f=target/'README.md';f.write_bytes(f.read_bytes()+b'tamper\n')
            with self.assertRaisesRegex(ValueError,'Changed physical'):R.verify(target)
    def test_04_changed_RAW_coefficient(self):
        text=self.mutate_line(self.plain,'ANS',lambda w:w.__setitem__(-1,f'{int(w[-1],16)^1:016X}'),after_run=True);parsed=self.m['normal_run_analysis'].parse_raw(text,lane='candidate',request_ids=[1,7,12],expected_points=self.points['controls']);self.assertFalse(self.m['cache_comparison'].compare_own_parity(self.normal,parsed)['passed'])
    def test_05_ACC_above_threshold(self):
        text=self.mutate_line(self.plain,'ACC',lambda w:w.__setitem__(-1,'3FF0000000000000'),after_run=True);parsed=self.m['normal_run_analysis'].parse_raw(text,lane='candidate',request_ids=[1,7,12],expected_points=self.points['controls'])
        with self.assertRaises(ValueError):self.m['accuracy_policy'].enforce(parsed,word_value=self.m['normal_run_analysis']._value)
    def test_06_false_success_status(self):
        lines=self.plain.splitlines(keepends=True);i=next(i for i,line in enumerate(lines) if line.startswith('CALL RUN '));w=lines[i].split();w[4:6]=['420','0'];lines[i]=' '.join(w)+'\n';parsed=self.m['normal_run_analysis'].parse_raw(''.join(lines),lane='candidate',request_ids=[1,7,12],expected_points=self.points['controls']);self.assertFalse(parsed['accepted_for_continuation'])
    def test_07_bootstrap_index(self):
        a=copy.deepcopy(self.normal['learning']);a['records'][0]['call_index']=1
        with self.assertRaises(ValueError):self.m['protocol_parity'].learning_coordinate(a)
    def test_08_lastK_not_outer_count(self):
        a=copy.deepcopy(self.normal['learning']);a['records'][-1]['call_index']=16
        with self.assertRaises(ValueError):self.m['protocol_parity'].learning_coordinate(a)
    def test_09_learning_point_order(self):
        a=copy.deepcopy(self.normal['learning']);a['records'][2]['point_id']=12
        with self.assertRaises(ValueError):self.m['protocol_parity'].learning_coordinate(a)
    def test_10_dropped_RAW_record(self):
        lines=self.plain.splitlines(keepends=True);del lines[next(i for i,line in enumerate(lines) if line.startswith('P '))]
        with self.assertRaises(ValueError):self.m['normal_run_analysis'].parse_raw(''.join(lines),lane='candidate',request_ids=[1,7,12],expected_points=self.points['controls'])
    def test_11_changed_physical_input_words(self):
        points=copy.deepcopy(self.points['controls']);points[1][0][0]='0000000000000000'
        with self.assertRaises(ValueError):self.m['normal_run_analysis'].parse_raw(self.plain,lane='candidate',request_ids=[1,7,12],expected_points=points)
    def test_12_missing_journals(self):
        normalized,blocks=self.m['observer_analysis']._split(self.observed)
        with self.assertRaises(ValueError):self.observed_parse(normalized)
    def test_13_malformed_journal(self):
        text=self.mutate_line(self.observed,'OBS_BEGIN',lambda w:w.__setitem__(4,'1'))
        with self.assertRaises(ValueError):self.observed_parse(text)
    def test_14_DP_eligibility_not_MP_genuine(self):
        text=self.mutate_line(self.observed,'OBS_FRAME',lambda w:w.__setitem__(12,'0'))
        with self.assertRaises(ValueError):self.observed_parse(text)
    def test_15_stale_current_words(self):
        text=self.mutate_line(self.observed,'OBS_FRAME_CURRENT',lambda w:w.__setitem__(-2,f'{int(w[-2],16)^1:032X}'))
        with self.assertRaises(ValueError):self.observed_parse(text)
    def test_16_changed_consumed_q(self):
        text=self.mutate_line(self.t('run4/evidence/cacheall-observed-critical20/driver.raw64'),'OBS_CONSUMED_Q',lambda w:w.__setitem__(-2,f'{int(w[-2],16)^1:032X}'))
        with self.assertRaises(ValueError):self.observed_parse(text,'critical20')
    def test_17_changed_native_join(self):
        text=self.mutate_line(self.observed,'OBS_NATIVE',lambda w:w.__setitem__(-1,f'{int(w[-1],16)^1:032X}'))
        with self.assertRaises(ValueError):self.observed_parse(text)
    def test_18_wrong_dispatch_source(self):
        with self.assertRaises(ValueError):self.m['observer_analysis'].analyze_observed(self.observed,self.plain,lane='candidate',request_ids=[1,7,12],observed_pid=self.j('run4/validated-cacheall-observed-controls.json')['pid'],plain_pid=self.j('run4/validated-cacheall-plain-controls.json')['pid'],parse_normal=self.m['normal_run_analysis'].parse_raw,expected_points=self.points['controls'],compiled_dispatch_source=b'fake source',cache_policy=-1,cache_budget=268435456)
    def test_19_forged_QP_counter(self):
        a=self.j('run4/validated-cacheall-observed-controls.json')['analysis'];e=next(e for j in a['journals'] for e in j['cache_observations']['events'] if e['kind']==1 and e['width']==64 and e['check_pass']);e['after'][9]+=1
        with self.assertRaises(ValueError):self.m['cache_metrics']._delta(e)
    def test_20_check_pass_store_mutation(self):
        a=self.j('run4/validated-cacheall-observed-controls.json')['analysis'];e=next(e for j in a['journals'] for e in j['cache_observations']['events'] if e['kind']==1 and e['width']==64 and e['check_pass']);e['after'][0]+=1
        with self.assertRaises(ValueError):self.m['cache_metrics']._delta(e)
    def test_21_failed_interval_counted_as_success(self):
        cache=self.j('run4/cache-observations.json');next(e for e in cache['intervals'] if e['status']!=0)['status']=0
        with self.assertRaises(ValueError):R.cache_facts(cache)
    def test_22_missing_clear(self):
        cache=self.j('run4/cache-observations.json');cache['lifecycle_events'].pop(0)
        with self.assertRaises(ValueError):R.cache_facts(cache)
    def test_23_invented_group3_QP_hit(self):
        cache=self.j('run4/cache-observations.json');next(e for e in cache['intervals'] if e['successful_visible_delta'] and e['successful_visible_delta']['QP6'][0])['group']=3
        with self.assertRaises(ValueError):R.cache_facts(cache)
    def test_24_no_lifetime_hit_rate(self):
        cache=self.j('run4/cache-observations.json');cache['hit_rates']=0.9
        with self.assertRaises(ValueError):R.cache_facts(cache)
    def test_25_overbudget_RAW_store(self):
        a=self.j('run4/validated-cacheall-observed-controls.json')['analysis'];event=next(e for j in a['journals'] for e in j['cache_observations']['events'] if e['kind']==1 and e['width']==64);lines=self.observed.splitlines(keepends=True)
        for i,line in enumerate(lines):
            w=line.split()
            if w[:2]==['OBS_CACHE_VALUES',str(event['index'])]:w[25]=f'{268435456+1:016X}';lines[i]=' '.join(w)+'\n';break
        else:self.fail('Actual budget counter absent')
        with self.assertRaises(ValueError):self.observed_parse(''.join(lines))
    def test_26_forged_raw_map(self):
        worker=self.j('run4/validated-cacheall-plain-controls.json');mapping=worker['maps']['ready'];mapping['raw']=mapping['raw'].replace(worker['executable_path'],worker['executable_path']+'-fake')
        with self.assertRaises(ValueError):R.authenticate_maps(worker,self.j('run4/terminal.json'),self.j('source/run4/input-pins.json')['runtime'])
    def test_27_wrong_process_start(self):
        worker=self.j('run4/validated-cacheall-plain-controls.json');worker['maps']['ready']['identity']['start_time_ticks']+=1
        with self.assertRaises(ValueError):R.authenticate_maps(worker,self.j('run4/terminal.json'),self.j('source/run4/input-pins.json')['runtime'])
    def test_28_wrong_provider(self):
        worker=self.j('run4/validated-cacheall-plain-controls.json');worker['maps']['ready']['ELF']['/fake/provider']='0'*64
        with self.assertRaises(ValueError):R.authenticate_maps(worker,self.j('run4/terminal.json'),self.j('source/run4/input-pins.json')['runtime'])
    def test_29_forged_protocol_coordinate(self):
        worker=self.j('run4/validated-cacheall-plain-controls.json');worker['protocol'][2]['fields'][4]='16'
        with self.assertRaises(ValueError):self.m['protocol_parity'].protocol_core(worker['protocol'],pid=worker['pid'],ids=[1,7,12],measurements=self.normal['measurements'],learning=self.normal['learning'])
    def test_30_extra_protocol_tail(self):
        worker=self.j('run4/validated-cacheall-plain-controls.json');worker['protocol'].append({'direction':'RECV','fields':['EXTRA']})
        with self.assertRaises(ValueError):self.m['protocol_parity'].protocol_core(worker['protocol'],pid=worker['pid'],ids=[1,7,12],measurements=self.normal['measurements'],learning=self.normal['learning'])
    def test_31_missing_failure_history(self):
        data=dict(self.d);del data['history/run2/terminal.json']
        with self.assertRaises((ValueError,KeyError)):R.semantic(data,self.c,self.p)
    def test_32_source_pin_drift(self):
        data=dict(self.d);data['source/run4/observer_analysis.py']+=b'\nchanged'
        with self.assertRaisesRegex(ValueError,'Frozen source member'):R.semantic(data,self.c,self.p)
    def test_33_effects_refused(self):
        import ctypes,socket,subprocess
        with R.blocked_apis({}) as attempts:
            for call in (lambda:Path('/outside').read_bytes(),lambda:subprocess.run(['true']),lambda:ctypes.CDLL('/unused'),lambda:socket.socket(),lambda:os.kill(1,0),lambda:Path('/outside').write_text('x')):
                with self.assertRaises(RuntimeError):call()
        self.assertEqual(len(attempts),6)
    def test_34_unmapped_payload(self):
        with tempfile.TemporaryDirectory(prefix='CMS-cache-extra-') as d:
            target=Path(d)/'capsule';shutil.copytree(HERE,target);(target/'extra').write_text('x')
            with self.assertRaisesRegex(ValueError,'Exact physical inventory'):R.verify(target)
    def test_35_symlink(self):
        with tempfile.TemporaryDirectory(prefix='CMS-cache-symlink-') as d:
            target=Path(d)/'capsule';shutil.copytree(HERE,target);(target/'alias').symlink_to('README.md')
            with self.assertRaisesRegex(ValueError,'symlink'):R.verify(target)
    def test_36_safe_native_exclusions(self):
        for name,b in self.d.items():self.assertFalse(b.startswith(b'\x7fELF'),name);b.decode('utf8')
    def test_37_changed_consumed_raw(self):
        text=self.mutate_line(self.t('run4/evidence/cacheall-observed-critical20/driver.raw64'),'OBS_CONSUMED_RAW',lambda w:w.__setitem__(-2,f'{int(w[-2],16)^1:032X}'))
        with self.assertRaises(ValueError):self.observed_parse(text,'critical20')
    def test_38_forged_certifier_veto(self):
        lines=self.t('run4/evidence/cacheall-observed-critical20/driver.raw64').splitlines(keepends=True)
        for i,line in enumerate(lines):
            w=line.split()
            if w[0]=='OBS_PAIR' and w[14]=='1':w[12]='0';lines[i]=' '.join(w)+'\n';break
        else:self.fail('Actual used certificate/veto witness absent')
        with self.assertRaises(ValueError):self.observed_parse(''.join(lines),'critical20')
    def test_39_stale_reference(self):
        text=self.mutate_line(self.observed,'OBS_FRAME_REF',lambda w:w.__setitem__(-2,f'{int(w[-2],16)^1:032X}'))
        with self.assertRaises(ValueError):self.observed_parse(text)
    def test_40_false_precision_width(self):
        text=self.mutate_line(self.observed,'OBS_FRAME',lambda w:w.__setitem__(3,'128'))
        with self.assertRaises(ValueError):self.observed_parse(text)
    def test_41_undeclared_runtime_card_change(self):
        policy=self.j('source/run4/runtime-policies.json')['policies']['cache0']
        with self.assertRaises(ValueError):self.m['cache_worker'].validate_runtime_card(self.d['source/run4/cards/original-MadLoopParams.dat'],self.d['source/run4/cards/cache0.dat']+b'\nforged\n',policy)
    def test_42_path_traversal_inventory(self):
        with tempfile.TemporaryDirectory(prefix='CMS-cache-path-') as d:
            target=Path(d)/'capsule';shutil.copytree(HERE,target);ix=json.loads((target/'index.json').read_text());ix['files']['../README.md']=ix['files'].pop('README.md');(target/'index.json').write_text(json.dumps(ix))
            with self.assertRaises(ValueError):R.verify(target)
    def test_43_false_counter_qualification(self):
        cache=self.j('run4/cache-observations.json');cache['performance_qualified']=True
        with self.assertRaises(ValueError):R.cache_facts(cache)
    def test_44_mixed_valid_mask(self):
        text=self.mutate_line(self.observed,'OBS_FRAME',lambda w:w.__setitem__(13,'0'))
        with self.assertRaises(ValueError):self.observed_parse(text)
if __name__=='__main__':unittest.main(verbosity=2)
