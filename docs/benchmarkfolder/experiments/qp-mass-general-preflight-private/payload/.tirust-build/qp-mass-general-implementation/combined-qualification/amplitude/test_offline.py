#!/usr/bin/env python3
"""Offline synthetic/parser tests only; no observed DP result is fabricated."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class Offline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a=load('combined_amplitude_analyzer',HERE/'analyze.py')
        cls.spec=json.loads((HERE/'prepared.json').read_text())
        parser=cls.spec['frozen_qp_parser']
        assert cls.a.sha(parser['path'])==parser['sha256']
        cls.old=load('frozen_qp_test_parser',parser['path'])
        cls.capture=cls.old.parse(Path(cls.spec['oracles']['9']['path']),'candidate',9)

    def dp_text(self):
        # Deliberately synthetic: shapes/words copied from a QP capture solely
        # to exercise the DP parser. It is never labeled as a native DP result.
        labels={'INPUT_P','RETURN_P','AMPLITUDE','ACCURACY','PUBLIC_STATUS',
                'FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END'}
        data=self.capture['points'][1]['data']
        labels|={s for s in data if s.startswith('MODEL_') and s!='MODEL_UPDATE'}
        records=[]
        for label in sorted(labels):
            for row in data[label]:records.append(' '.join([row[0],label,*map(str,row[1:])]))
        return '\n'.join(['FORMAT EW_QP_MASS_HEX_V1 storage-order-int64-words','BEGIN 1 FFFFF',
                          *records,'END 1 0 0 FFFFF',f'FOOTER {len(records)} 0 0 1000000'])+'\n'

    def parse(self,text):
        with tempfile.TemporaryDirectory(prefix='offline-synthetic-',dir=HERE/'inputs') as directory:
            path=Path(directory)/'synthetic.hex';path.write_text(text)
            return self.a.parse_dp(path)

    def test_synthetic_dp_shape(self):
        self.assertIn('AMPLITUDE',self.parse(self.dp_text())['data'])

    def test_missing_footer(self):
        with self.assertRaises(ValueError):self.parse(self.dp_text().rsplit('FOOTER',1)[0])

    def test_unexpected_native(self):
        with self.assertRaises(ValueError):self.parse(self.dp_text().replace('END 1 0 0','NATIVE 9 0 FFFFF\nEND 1 0 0'))

    def test_missing_model(self):
        with self.assertRaises(ValueError):self.parse('\n'.join(s for s in self.dp_text().splitlines() if not s.startswith('R8 MODEL_MDL_MT ')))

    def test_wrong_footer(self):
        with self.assertRaises(ValueError):self.parse(self.dp_text().replace('0 0 1000000','1 0 1000000'))

    def test_wrong_kind(self):
        with self.assertRaises(ValueError):self.parse(self.dp_text().replace('R8 MODEL_MDL_MT','R16 MODEL_MDL_MT'))

    def test_qp_self_compare_and_refusals(self):
        result=self.a.qp_compare(self.capture,self.capture)
        self.assertTrue(result['exact_mass_nonmass_output_status_final_flags_match'])
        self.assertEqual([len(x) for x in result['refused_groups_by_point']],[0,37])

    def test_qp_mass_bit_change(self):
        other=copy.deepcopy(self.capture)
        row=other['points'][1]['groups'][0]['data']['NATIVE_MASSES'][0]
        row[2]=f'{int(row[2],16)^1:016X}'
        self.assertFalse(self.a.qp_compare(self.capture,other)['exact_mass_nonmass_output_status_final_flags_match'])

    def test_intermediate_flag_timing_is_excluded(self):
        other=copy.deepcopy(self.capture)
        row=other['points'][1]['groups'][0]['data']['ACTUAL_BACKEND'][0]
        row[2]='TTTTT' if row[2]!='TTTTT' else 'FFFFF'
        self.assertTrue(self.a.qp_compare(self.capture,other)['exact_mass_nonmass_output_status_final_flags_match'])

    def test_final_flag_change_fails(self):
        other=copy.deepcopy(self.capture)
        old=other['points'][1]['end_flags']
        other['points'][1]['end_flags']='T'+old[1:] if old[0]=='F' else 'F'+old[1:]
        self.assertFalse(self.a.qp_compare(self.capture,other)['exact_mass_nonmass_output_status_final_flags_match'])

    def test_final_fpe_record_flag_change_fails(self):
        other=copy.deepcopy(self.capture)
        row=other['points'][1]['data']['FPE_QP_END'][0]
        row[2]='TTTTT' if row[2]!='TTTTT' else 'FFFFF'
        self.assertFalse(self.a.qp_compare(self.capture,other)['exact_mass_nonmass_output_status_final_flags_match'])


if __name__=='__main__':unittest.main(verbosity=2)
