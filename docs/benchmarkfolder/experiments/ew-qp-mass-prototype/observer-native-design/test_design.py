#!/usr/bin/env python3
"""Pure-Python source/protocol tests. No Fortran compiler or reducer runs."""
import copy
import difflib
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import analyze
import instrument

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def synthetic_stream():
    rows=['FORMAT EW_QP_MASS_HEX_V1 storage-order-int64-words'];count=0
    def add(line):
        nonlocal count
        rows.append(line);count+=1
    def data(tag,label,n):
        for i in range(1,n+1):add(tag+' '+label+' '+str(i)+' '+(' '.join(['0000000000000000']*analyze.COUNTS[tag])))
    for point in (1,2):
        rows.append(f'BEGIN {point} FFFFFF')
        for label,n in [('INPUT_P',20),('RETURN_P',20),('AMPLITUDE',8),('ACCURACY',2)]:data('R8',label,n)
        data('R16','CONSUMED_MP_PS',20)
        add('INTEGER PUBLIC_STATUS 120 FFFFFF')
        add('INTEGER MODEL_UPDATE 1 FFFFFF');add('LOGICAL GENUINE_QP_INTEGRAND T FFFFFF')
        for label in ('FPE_DP_BEGIN','FPE_QP_BEGIN','FPE_DP_END','FPE_QP_END'):add('LOGICAL '+label+' F FFFFFF')
        for mass in ('MT','MZ','MW','MH'):
            data('R8','MODEL_MDL_'+mass,1);data('R16','MODEL_MP__MDL_'+mass,1)
        for tag,label,n in [('R8','MODEL_MU_R',1),('C32','MODEL_MP__MU_R',1),('C16','MODEL_GAL',2),('C32','MODEL_MP__GAL',2)]:data(tag,label,n)
        for group in range(1,201):
            add(f'GROUP {point} {group} {group} 1 4 1 0 FFFFFF')
            for tag,label,n in [('C16','ARG_M1',1),('C16','DP_M2L',1),('R8','HELPER_DP_PL',4),
                ('R16','HELPER_QP_PL',4),('R16','REPAIRED_MP_PS',20),('R16','CURRENT_MP_P',20),
                ('C32','NUMERATOR_COEFS',1),('R16','ENTRY_PL',4),('C16','ENTRY_M2_DP',1),
                ('R16','ORDERED_PL',4),('C32','ORDERED_MASSES',1),('R16','NATIVE_SHIFTS',4),
                ('C32','KINEMATIC_S',1),('R16','NATIVE_INVARIANTS',1),('C32','NATIVE_MASSES',1),('R16','NATIVE_SCALE',1)]:data(tag,label,n)
            add('INTEGER ACTUAL_BACKEND 9 FFFFFF');add('NATIVE 9 0 FFFFFF')
            add('LOGICAL CHECK_PASS F FFFFFF');add('LOGICAL CACHE_ENABLED T FFFFFF')
            for label in ('RAW_UV1','RAW_IR1','RAW_IR2','RAW_FINITE'):data('C32',label,1)
            data('C16','GROUP_RESULT',3);add('LOGICAL GROUP_STABLE T FFFFFF')
        rows.append(f'END {point} 200 200 FFFFFF')
    rows.append(f'FOOTER {count} 0 0 1000000')
    return '\n'.join(rows)+'\n'


class Source(unittest.TestCase):
    def test_frozen_parent_inventory(self):
        p=HERE.parent;index=json.loads((p/'evidence-index.json').read_text())
        self.assertEqual(hashlib.sha256((p/'evidence-index.json').read_bytes()).hexdigest(),'9c9725ed41fe6484bbfa7eaa14f5143e91bda235771ab281fa7ecfd8ddec2234')
        for name,h in index['files'].items():self.assertEqual(hashlib.sha256((p/name).read_bytes()).hexdigest(),h,name)

    def test_matched_reproduction_and_insertions(self):
        for arm in ('original','candidate'):
            old=ROOT/'.tirust-build/ew-qp-mass-prototype'/arm/'SubProcesses/P0_epem_ttxa'
            names=('CT_interface.f','TIRUST_interface.f','loop_CT_calls_1.f','mp_compute_loop_coefs.f')
            inputs={n:(old/n).read_text() for n in names}
            new=instrument.instrument(inputs,arm=='candidate')
            for name,text in new.items():
                self.assertEqual(text,(HERE/'sources'/arm/name).read_text())
                # No existing line is replaced or deleted by instrumentation.
                for tag,_,_,_,_ in difflib.SequenceMatcher(a=inputs[name].splitlines(),b=text.splitlines(),autojunk=False).get_opcodes():
                    self.assertIn(tag,('equal','insert'),(arm,name,tag))
                self.assertTrue(all(not l or l[0] in 'Cc*!' or len(l)<=72 for l in text.splitlines()))
            self.assertEqual(new['loop_CT_calls_1.f'],inputs['loop_CT_calls_1.f'])

    def test_staged_inventory_and_no_execution_evidence(self):
        p=json.loads((HERE/'prepared.json').read_text())
        for name,h in p['artifacts'].items():
            f=ROOT/name;self.assertFalse(f.is_symlink());self.assertEqual(hashlib.sha256(f.read_bytes()).hexdigest(),h,name)
        self.assertEqual(p['compiler_invocations'],0);self.assertEqual(p['native_reducer_calls'],0)

    def test_legacy_decimal_trace_inactive(self):
        for name in ('check_observed.f','ew_mass_observer.f90','ew_mass_snapshot.f'):
            text=(HERE/name).read_text().upper()
            self.assertNotIn('EW_TRACE_BEGIN',text);self.assertNotIn('EW_TRACE_END',text)
        module=(HERE/'ew_mass_observer.f90').read_text()
        self.assertNotIn('ES45',module);self.assertIn('bits=transfer(values(i),bits)',module)
        self.assertIn('call ieee_set_status(saved)',module)


class Parser(unittest.TestCase):
    def parse_text(self,text,arm='original',backend=9):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'synthetic.hex';p.write_text(text);return analyze.parse(p,arm,backend)

    def test_synthetic_complete_and_exact_comparison(self):
        value=self.parse_text(synthetic_stream())
        self.assertTrue(analyze.compare_pair(value,value,'same_backend')['nonmass_input_bits_match'])
        altered=copy.deepcopy(value)
        altered['points'][1]['groups'][0]['data']['ENTRY_PL'][0][-1]='0000000000000001'
        self.assertFalse(analyze.compare_pair(value,altered,'same_backend')['nonmass_input_bits_match'])

    def test_reject_truncation_cap_flag_changes_and_failed_raw(self):
        text=synthetic_stream()
        bad=[text.rsplit('FOOTER',1)[0],text.replace(' 0 0 1000000',' 1 0 1000000'),
             text.replace(' 0 0 1000000',' 0 1 1000000'),text.replace('NATIVE 9 0','NATIVE 9 2',1),
             text.replace('R16 ENTRY_PL 1','R16 ENTRY_PL 2',1)]
        for malformed in bad:
            with self.assertRaises((ValueError,AssertionError)):self.parse_text(malformed)

    def test_reject_wrong_arm_backend_type_and_absent_policy(self):
        text=synthetic_stream()
        with self.assertRaises(AssertionError):self.parse_text(text,'candidate',9)
        with self.assertRaises(AssertionError):self.parse_text(text,'original',8)
        for before,after in [('R16 ENTRY_PL 1','C16 ENTRY_PL 1'),
                             ('LOGICAL GENUINE_QP_INTEGRAND T','LOGICAL GENUINE_QP_INTEGRAND F'),
                             ('MODEL_MP__MDL_MT','MISSING_MP_MASS'),
                             ('CHECK_PASS F','CHECK_PASS T'),('CACHE_ENABLED T','CACHE_ENABLED F'),
                             ('FPE_DP_BEGIN','MISSING_FPE')]:
            with self.assertRaises(AssertionError):self.parse_text(text.replace(before,after,1))

    def test_reject_raw_without_native_and_opposite_result_family(self):
        text=synthetic_stream()
        absent=text.replace('NATIVE 9 0 FFFFFF\n','',1)
        before=absent.rsplit('FOOTER ',1);fields=before[1].split();fields[0]=str(int(fields[0])-1)
        absent=before[0]+'FOOTER '+' '.join(fields)+'\n'
        with self.assertRaises(AssertionError):self.parse_text(absent)
        with self.assertRaises(AssertionError):self.parse_text(text.replace('RAW_UV1','NRS_RESULT',1))

    def test_ieee_binary_decoding(self):
        self.assertEqual(analyze.finite_value(['3FF0000000000000'],64),1)
        self.assertEqual(analyze.finite_value(['0000000000000000','3FFF000000000000'],128),1)
        self.assertIsNone(analyze.finite_value(['7FF0000000000000'],64))
        self.assertIsNone(analyze.finite_value(['0000000000000000','7FFF000000000000'],128))
        self.assertEqual(analyze.finite_value(['0000000000000001'],64),analyze.Fraction(2)**-1074)


if __name__=='__main__':unittest.main()
