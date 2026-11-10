#!/usr/bin/env python3
"""Pure Python source/parser tests; historical fixtures are not new native runs."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Source(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=load('current_original_analyzer',HERE/'analyze.py')
        cls.s=json.loads((HERE/'child-prepared.json').read_text())
        parser=cls.s['frozen_parser'];assert cls.a.sha(parser['path'])==parser['sha256']
        p=load('current_original_frozen_parser',parser['path'])
        cls.c={}
        for backend in (8,9):
            cls.c[backend]={arm:p.parse(Path(cls.s['historical_oracles'][str(backend)][arm]['path']),arm,backend)
                            for arm in ('original','candidate')}

    def pair(self):return copy.deepcopy(self.c[9]['original']),copy.deepcopy(self.c[9]['candidate'])

    def test_one_link_and_no_compile(self):
        self.assertEqual(len(self.s['build_jobs']),1)
        argv=self.s['build_jobs'][0]['argv']
        self.assertNotIn('-c',argv)
        self.assertEqual(len([p for p in argv if p.endswith('.o')]),24)
        self.assertFalse(any(p.endswith(('.f','.f90')) for p in argv))
        old=list(self.s['link_delta_proof']['original_link_recipe'])
        old[old.index('-o')+1]=self.s['build_outputs'][0]
        for p in self.s['link_delta_proof']['removed_objects']:old.remove(p)
        old[old.index(str(HERE.parent/'amplitude/build/loop_CT_calls_1.o'))]=self.s['link_delta_proof']['restored_object']
        self.assertEqual(old,argv)

    def test_exact_staged_inputs(self):
        for run in self.s['runs']:
            paired=self.s['counterparts'][str(run['backend'])]
            src=Path(paired['run']['cwd'])
            if not src.is_absolute():src=Path.cwd()/src
            expected={str(Path(p).relative_to(src)):v for p,v in paired['input_sha256'].items()}
            got={str(Path(p).relative_to(run['cwd'])):v for p,v in run['input_sha256'].items()}
            self.assertEqual(expected,got)
            for p,digest in run['input_sha256'].items():self.assertEqual(self.a.sha(p),digest)

    def test_historical_mass_axis_fixtures(self):
        # Tests the comparator only; these inputs use historical49af, and this
        # test is not a claim about the pending same-current-runtime campaign.
        for backend in (8,9):
            result=self.a.compare(self.c[backend]['original'],self.c[backend]['candidate'],True)
            self.assertTrue(result['exact_invariant_inputs_match'])
            self.assertIn('mass_argument_or_squared_mass',result['change_categories'])

    def test_identical_secondary(self):
        result=self.a.compare(self.c[9]['original'],self.c[9]['original'],False)
        self.assertTrue(result['exact_all_observed_records_match'])
        self.assertEqual(result['changes'],[])

    def test_mp_model_change_fails(self):
        a,b=self.pair();row=b['points'][1]['data']['MODEL_MP__MDL_MT'][0]
        row[2]=f'{int(row[2],16)^1:016X}'
        self.assertFalse(self.a.compare(a,b,True)['exact_invariant_inputs_match'])

    def test_numerator_change_fails(self):
        a,b=self.pair();row=b['points'][1]['groups'][0]['data']['NUMERATOR_COEFS'][0]
        row[2]=f'{int(row[2],16)^1:016X}'
        self.assertFalse(self.a.compare(a,b,True)['exact_invariant_inputs_match'])

    def test_kinematic_change_is_classified_on_mass_axis(self):
        a,b=self.pair();row=b['points'][1]['groups'][0]['data']['KINEMATIC_S'][0]
        row[2]=f'{int(row[2],16)^1:016X}'
        result=self.a.compare(a,b,True)
        self.assertTrue(result['exact_invariant_inputs_match'])
        self.assertIn('mass_dependent_kinematic_matrix_or_invariants',result['change_categories'])

    def test_kinematic_change_fails_same_policy(self):
        a=copy.deepcopy(self.c[9]['original']);b=copy.deepcopy(a)
        row=b['points'][1]['groups'][0]['data']['KINEMATIC_S'][0];row[2]=f'{int(row[2],16)^1:016X}'
        self.assertFalse(self.a.compare(a,b,False)['exact_invariant_inputs_match'])

    def test_result_and_status_differences_are_retained(self):
        a,b=self.pair();point=b['points'][1]
        point['data']['PUBLIC_STATUS'][0][1]='999'
        point['groups'][0]['native'][0]['status']=123
        result=self.a.compare(a,b,True)
        self.assertTrue(result['exact_invariant_inputs_match'])
        self.assertEqual(result['points'][0]['right_event_metadata']['point_status_flags']['PUBLIC_STATUS'][0][1],'999')
        self.assertTrue(any(r['native'][0]['status']==123 for r in result['points'][0]['right_native_refusals']))

    def test_group_order_fails(self):
        a,b=self.pair();b['points'][1]['groups'][0:2]=reversed(b['points'][1]['groups'][0:2])
        with self.assertRaises(ValueError):self.a.compare(a,b,True)

    def test_unknown_field_fails(self):
        a,b=self.pair();b['points'][1]['groups'][0]['data']['ARG_MBOGUS']=[]
        with self.assertRaises(ValueError):self.a.compare(a,b,True)

    def test_malformed_flags_fail(self):
        a,b=self.pair();b['points'][1]['end_flags']='UNKNOWN'
        with self.assertRaises(ValueError):self.a.compare(a,b,True)

    def test_entry_square_mismatch_fails(self):
        a,b=self.pair();row=b['points'][1]['groups'][0]['data']['ENTRY_M2_QP'][0];row[2]=f'{int(row[2],16)^1:016X}'
        with self.assertRaises(ValueError):self.a.compare(a,b,True)


if __name__=='__main__':unittest.main(verbosity=2)
