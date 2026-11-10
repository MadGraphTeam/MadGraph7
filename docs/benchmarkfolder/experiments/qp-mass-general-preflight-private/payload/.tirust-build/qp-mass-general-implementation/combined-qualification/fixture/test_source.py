#!/usr/bin/env python3
"""Source/parser tests with synthetic journal strings; never compile or run Fortran."""
import copy
import importlib.util
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import validate as v


class SourceFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.commands=json.loads((HERE/'commands.json').read_text())
        cls.matrix=json.loads((HERE/'argument-matrix.json').read_text())

    def test_01_frozen_generation_bindings(self):
        self.assertEqual(len(self.commands['runs']),218)
        for name,digest in self.commands['inputs'].items():self.assertEqual(v.sha(v.ROOT/name),digest,name)
        self.assertEqual(self.commands['required_reducer_libraries'],[])
        self.assertFalse(self.commands['native_compile_performed'])
        self.assertFalse(self.commands['native_execution_performed'])

    def test_02_actual_guard_and_identical_dp_calls(self):
        for route in ('grouped','ungrouped','loop_induced'):
            old=self.matrix['legacy_'+route]['items'];new=self.matrix['guarded_'+route]['items']
            self.assertEqual(len(old),len(new))
            for a,b in zip(old,new,strict=True):
                if a.startswith('CALL '):
                    lines=b.splitlines()
                    self.assertEqual(lines[0],'IF(CTMODE.LT.4) THEN')
                    self.assertEqual(lines[1],a)
                    self.assertNotIn('MP__',a)
                    self.assertNotIn('_QPMASS',a)
                    self.assertEqual(lines[2],'ELSE');self.assertEqual(lines[-1],'ENDIF')
                else:self.assertEqual(a,b)

    def test_03_concrete_ids_pairings_masses(self):
        for arm in ('legacy','guarded'):
            for route in ('grouped','ungrouped','loop_induced'):
                items=self.matrix[arm+'_'+route]['expected']
                groups=[4,8,12] if route=='grouped' else [7,9,11,13]
                self.assertEqual([x['group'] for x in items],groups)
                self.assertEqual(items[-2]['masses'],['MDL_REAL','MDL_DERIVED','CMASS_MDL_CMS'])
                self.assertEqual(items[-1]['masses'],['MDL_DERIVED','CMASS_MDL_CMS'])
                self.assertEqual(items[-1]['pairing'],[1,2])
                self.assertEqual(items[-1]['mothers'],[11,12,13])
                self.assertTrue(all(x['rank']==2 and x['sqso']==1 for x in items))

    def test_04_real_common_and_formals(self):
        for route in ('grouped','ungrouped','loop_induced'):
            source=(HERE/'src'/('guarded_'+route+'.f')).read_text()
            self.assertEqual(source.count('COMMON/FX_CT/LSCALE,CTMODE'),source.count('SUBROUTINE FX_'))
            self.assertEqual(source.count('CTMODE.LT.4'),len(self.matrix['guarded_'+route]['expected']))
            self.assertTrue(all(len(line)<=132 for line in source.splitlines()))
        for arm in ('legacy','guarded'):
            for route in ('grouped','ungrouped','loop_induced'):
                source=(HERE/'src'/(arm+'_'+route+'.f')).read_text()
                n=source.count('SUBROUTINE FX_')
                compact=re.sub(r'\s+','',source).replace('$','')
                self.assertEqual(compact.count('NLOOPGROUPS=13'),n)
                self.assertEqual(compact.count('NCOMB=1'),n)
                self.assertEqual(compact.count('NSQUAREDSO=1'),n)
                self.assertEqual(compact.count('GOODAMP(NSQUAREDSO,NLOOPGROUPS)'),n)
                self.assertEqual(compact.count('COMMON/FX_FILTERS/GOODAMP,GOODHEL,HELOFFSET'),n)
        helper=(HERE/'src/helpers.f').read_text()
        logical=''.join(line[6:].strip() for line in helper.splitlines())
        self.assertIn('FX_LOOP_2_QPMASS(I1,I2,M1,MP_M1,M2,MP_M2,RANK',logical)
        self.assertIn('FX_LOOP_3_QPMASS(I1,I2,I3,M1,MP_M1,M2,MP_M2,M3,MP_M3,RANK',logical)
        self.assertIn('FX_LOOP_2_3_QPMASS(P1,P2,I1,I2,I3,M1,MP_M1,M2,MP_M2,RANK',logical)
        for legacy in re.split(r'(?=      SUBROUTINE)',helper)[1:]:
            if '_QPMASS(' not in legacy.split('IMPLICIT NONE')[0]:
                self.assertNotIn('MP_M',legacy)
                self.assertNotIn('COMMON/',legacy)

    def test_05_binary_layout_and_derived_not_dp_promotion(self):
        self.assertEqual(v.real_words(1,113),['0000000000000000','3FFF000000000000'])
        self.assertEqual(v.real_words(-2,113),['0000000000000000','C000000000000000'])
        self.assertEqual(v.real_words(3,53),['4008000000000000'])
        self.assertNotEqual(v.derived(53),v.derived(113))
        dp,mp=v.model_words('valid')
        self.assertNotEqual(mp['MP__MDL_DERIVED'],v.complex_words(v.derived(53),p=113))
        for name in ('qnan','snan'):
            _,mp=v.model_words(name)
            self.assertEqual(mp['MP__MDL_REAL'][0],'0000000000000011')
            self.assertEqual(mp['MP__MDL_DERIVED'][0],'0000000000000022')
            self.assertEqual(mp['MP__CMASS_MDL_CMS'][::2],['0000000000000033','0000000000000044'])

    def test_06_all_expected_synthetic_journals_parse(self):
        for job in self.commands['runs']:
            rows=v.expected_rows(job['case'],self.matrix)
            text='\n'.join(' '.join(r) for r in rows)+'\n'
            self.assertEqual(v.parse_case(text,job,self.matrix),rows)
            if job['case']['payload']=='valid':
                rows=v.expected_rows(job['case'],self.matrix,setup='FFFFT')
                v.parse_case('\n'.join(' '.join(r) for r in rows)+'\n',job,self.matrix)

    def test_07_parser_rejects_incomplete_changed_flag_bit_and_order(self):
        job=next(x for x in self.commands['runs'] if x['case']==dict(arm='guarded',mode=4,payload='valid',trap=0,route='grouped'))
        good=v.expected_rows(job['case'],self.matrix)
        variants=[good[:-1],good+[good[-1]],good[:4]+good[5:]]
        for tag,column,value in [('ARG_MP',3,'DEADBEEF00000000'),('ARG_DP',3,'DEADBEEF00000000'),
                                 ('STATUS',2,'1'),('COMPLETE',2,'1'),('MOTHERS',2,'99')]:
            bad=copy.deepcopy(good);next(r for r in bad if r[0]==tag)[column]=value;variants.append(bad)
        bad=copy.deepcopy(good);next(r for r in bad if r[:2]==['FLAGS','AFTER'])[2]='TFFFF';variants.append(bad)
        bad=copy.deepcopy(good);bad[5],bad[6]=bad[6],bad[5];variants.append(bad)
        for rows in variants:
            with self.assertRaises(ValueError):v.parse_case('\n'.join(' '.join(r) for r in rows)+'\n',job,self.matrix)

    def test_08_trap_missing_only_allowed_exact_positive_control(self):
        control=next(x for x in self.commands['runs'] if x['expected_exit']==-8)
        self.assertEqual(control['case']['arm'],'trap_control')
        rows=v.expected_rows(control['case'],self.matrix)
        self.assertEqual(rows[-1],['PRECALL'])
        ordinary=next(x for x in self.commands['runs'] if x['case']['arm']=='guarded')
        truncated=v.expected_rows(ordinary['case'],self.matrix)
        truncated=truncated[:truncated.index(['PRECALL'])+1]
        with self.assertRaises(ValueError):v.parse_case('\n'.join(' '.join(r) for r in truncated),ordinary,self.matrix)

    def test_09_whole_gate_rejects_missing_events(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='parser-test-') as directory:
            path=Path(directory)/'events.json';path.write_text('{"jobs": []}\n')
            with self.assertRaisesRegex(ValueError,'Missing or extra fixture execution events'):v.validate(path)

    def test_10_no_native_commands_in_python_test_execution(self):
        self.assertEqual(self.commands['execution_seconds_max'],60)
        self.assertTrue(all(x['timeout_seconds']==3 for x in self.commands['runs']))
        self.assertEqual(sum(x['expected_exit']==-8 for x in self.commands['runs']),1)
        self.assertEqual({x['case']['mode'] for x in self.commands['runs']},set(range(-1,7)))
        self.assertEqual(len(self.commands['build_jobs']),12)
        self.assertTrue(all(job['argv'][0]==self.commands['compiler'] for job in self.commands['build_jobs']))

    def test_11_whole_gate_synthetic_in_memory_success_and_event_rejection(self):
        # No files are written to future native run directories. This checks the
        # complete gate against explicit in-memory synthetic journals/events.
        memory={};events=[]
        for job in self.commands['runs']:
            path=v.ROOT/job['cwd']/'caller.hex'
            text='\n'.join(' '.join(row) for row in v.expected_rows(job['case'],self.matrix))+'\n'
            memory[path]=text
            events.append(dict(name=job['name'],returncode=job['expected_exit'],timed_out=False,
                cwd=job['cwd'],command=['/pinned/test/taskset','-c','8',*job['argv']],
                changed_inputs={},expected_output_closure_matched=True,elapsed_seconds=0.001,
                output_files={'caller.hex':hashlib.sha256(text.encode()).hexdigest()}))
        events.extend([dict(name='fixture-loader'),dict(name='ew-amplitude'),dict(name='fixture-validation')])
        eventpath=HERE/'synthetic-only-memory-events.json'
        memory[eventpath]=json.dumps({'jobs':events})
        original_read=Path.read_text;original_is_file=Path.is_file;original_sha=v.sha
        def read(path,*args,**kwargs):return memory[path] if path in memory else original_read(path,*args,**kwargs)
        def is_file(path):return path in memory or original_is_file(path)
        def sha(path):return hashlib.sha256(memory[path].encode()).hexdigest() if path in memory else original_sha(path)
        with mock.patch.object(Path,'read_text',read),mock.patch.object(Path,'is_file',is_file),mock.patch.object(v,'sha',sha):
            result=v.validate(eventpath)
            self.assertTrue(result['passed']);self.assertEqual(result['dp_exact_pairs'],90)
            self.assertEqual(result['qp_transport_pairs'],18)
            for mutation in ('exit','timeout','argv','hash','duplicate'):
                broken=copy.deepcopy(events)
                if mutation=='exit':broken[0]['returncode']=1
                if mutation=='timeout':broken[0]['timed_out']=True
                if mutation=='argv':broken[0]['command'][2]='9'
                if mutation=='hash':broken[0]['output_files']['caller.hex']='0'*64
                if mutation=='duplicate':broken.append(broken[0])
                memory[eventpath]=json.dumps({'jobs':broken})
                with self.assertRaises(ValueError):v.validate(eventpath)


def main():
    require_cwd=v.ROOT
    assert Path.cwd().resolve()==require_cwd
    log=io.StringIO()
    with mock.patch.object(subprocess,'Popen',side_effect=AssertionError('Compiler/native subprocess forbidden')):
        result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceFixture))
    print(log.getvalue(),end='')
    if not result.wasSuccessful():raise SystemExit(1)
    summary=dict(schema='qp-mass-fixture-source-tests-v1',tests_passed=result.testsRun,
                 scope='Python generation/parser validation using synthetic expected journal strings only',
                 no_compiler_or_native_execution=True,commands_sha256=v.sha(HERE/'commands.json'),
                 test_sha256=v.sha(Path(__file__)),parser_sha256=v.sha(HERE/'validate.py'))
    text=json.dumps(summary,indent=2,sort_keys=True)+'\n'
    output=HERE/'source-tests.json'
    if output.exists():assert output.read_text()==text
    else:output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
