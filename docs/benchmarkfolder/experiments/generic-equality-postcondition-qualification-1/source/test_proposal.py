"""Pure corpus, parser, admission and environment tests; no compiler/native calls."""
from pathlib import Path
import copy
import hashlib
import json
import unittest

import analyze
import controller
import corpus
import tool_environment

HERE=Path(__file__).resolve().parent
CONSTANTS=[0,1024,2048,3072,1,4,8,16,32,61]
BASELINE=[0,0,0,0x1f80,0x037f,0x1f80,0]


def synthetic(cases,expected,skip=None):
    lines=['SCHEMA exact-geometry-fenv-v1','CONSTANTS '+' '.join(map(str,CONSTANTS)),
           'NCASES '+str(len(cases)),'MODES 20',
           'SNAPSHOT_FIELDS round sticky halts mxcsr_control x87cw mxcsr_raw x87sw',
           'CASE_FIELDS id mode status accepted restore_ok request_round request_sticky request_halts '
           'request_ftz request_daz baseline(7) before(7) after(7) restored(7)']
    for case in cases:
        identifier=case['id'];n=len(case['coefficients'])
        lines.append('INPUT '+' '.join(map(str,[identifier,case['width'],case['reference_width'],n,case['shape']])))
        lines.append('COEFF '+str(identifier)+(' '+' '.join(map(str,case['coefficients'])) if n else ''))
        for leg,vector in enumerate(case['external'],1):
            for mu,(real,imag) in enumerate(vector):
                lines.append(f'X {identifier} {leg} {mu} {real:032X} {imag:032X}')
        for mu,(real,imag) in enumerate(case['q']):
            lines.append(f'Q {identifier} {mu} {real:032X} {imag:032X}')
        for mode in range(20):
            request=analyze.request_for(mode,CONSTANTS)
            status=int(skip is not None and skip(identifier,mode))
            before=BASELINE[:]
            if mode and not status:
                rounding,sticky,halts,ftz,daz=request
                csr=(0x1f80&~(halts<<7))|(rounding<<3)|sticky
                if ftz>=0:csr|=(ftz<<15)|(daz<<6)
                cw=(0x037f&~halts)|rounding
                before=[rounding,sticky,halts,csr&~63,cw,csr,0]
            accepted=-1 if status else int(expected['cases'][str(identifier)]['expected_accept'])
            fields=[identifier,mode,status,accepted,1,*request,*BASELINE,*before,*before,*BASELINE]
            lines.append('CASE '+' '.join(map(str,fields)))
    lines.append('DONE '+str(len(cases)))
    return '\n'.join(lines)+'\n'


class PureProposalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus=corpus.build();cls.expected=corpus.expectations(cls.corpus)
        cls.small=cls.corpus['cases'][:2]
        cls.small_expected=copy.deepcopy(cls.expected)
        cls.small_expected['case_count']=2
        cls.small_expected['cases']={str(x['id']):cls.expected['cases'][str(x['id'])] for x in cls.small}

    def test_frozen_fixture_and_independent_oracle_regeneration(self):
        self.assertEqual(corpus.encode(self.corpus),(HERE/'fixtures.hex').read_text())
        self.assertEqual(self.expected,json.loads((HERE/'expectations.json').read_text()))
        self.assertEqual(self.expected['case_count'],1698)
        self.assertEqual(self.expected['counts']['expected_accept'],526)
        self.assertEqual(self.expected['counts']['expected_refuse'],1172)

    def test_capacity_refusals_include_exact_cancellations(self):
        rows=[r for r in self.expected['cases'].values() if r['mathematical_equality'] and not r['expected_accept']]
        self.assertEqual(len(rows),10)
        self.assertTrue(any(r['domain_reason']=='coefficient-product-capacity' for r in rows))
        self.assertTrue(any(r['domain_reason']=='alignment-capacity' for r in rows))
        self.assertTrue(any(r['domain_reason']=='absolute-sum-capacity' for r in rows))

    def test_conditionally_reconstructed_witnesses_are_not_observed_snapshots(self):
        rows=[r for r in self.expected['cases'].values() if r['tag'].startswith('conditional-CMS')]
        self.assertEqual(len(rows),4)
        self.assertTrue(all(r['expected_accept'] for r in rows))
        self.assertTrue(all(not r['provenance']['owned_snapshot_observed'] and
                            not r['provenance']['q_component_ABI_observed'] for r in rows))

    def test_full_binary128_last_bit_and_DP_extension_controls(self):
        rows=[r for r in self.expected['cases'].values() if r['tag'].startswith('QP-113bit')]
        self.assertEqual(len(rows),4)
        self.assertEqual(sum(r['expected_accept'] for r in rows),2)
        bad=[r for r in self.expected['cases'].values() if r['tag'].startswith('DP-')]
        self.assertEqual(len(bad),2)
        self.assertTrue(all(not r['expected_accept'] for r in bad))

    def test_complete_input_echo_and_all_cohorts_parse(self):
        result=analyze.analyze(synthetic(self.small,self.small_expected),self.small_expected)
        self.assertEqual(result['case_count'],2)
        self.assertEqual(result['counts']['executed_mode_observations'],40)
        self.assertFalse(result['snapshot_ownership_qualified'])

    def test_corrupted_native_input_echo_is_not_an_answer(self):
        text=synthetic(self.small,self.small_expected)
        lines=text.splitlines()
        index=next(i for i,line in enumerate(lines) if line.startswith('Q 1 0 '))
        fields=lines[index].split();fields[3]=format(int(fields[3],16)^1,'032X');lines[index]=' '.join(fields)
        with self.assertRaisesRegex(ValueError,'input echo'):
            analyze.analyze('\n'.join(lines)+'\n',self.small_expected)

    def test_wrong_answer_flags_rounding_and_restoration_fail(self):
        original=synthetic(self.small,self.small_expected)
        for kind in ('answer','flags','rounding','restore'):
            lines=original.splitlines();index=next(i for i,s in enumerate(lines) if s.startswith('CASE 1 2 '))
            fields=lines[index].split()
            if kind=='answer':fields[4]=str(1-int(fields[4]))
            if kind=='flags':fields[25]=str(int(fields[25])^32)
            if kind=='rounding':
                for start in (18,25):
                    fields[start+3]=str(int(fields[start+3])&~0x6000)
                    fields[start+5]=str(int(fields[start+5])&~0x6000)
            if kind=='restore':fields[5]='0'
            lines[index]=' '.join(fields)
            with self.assertRaises(ValueError,msg=kind):
                analyze.analyze('\n'.join(lines)+'\n',self.small_expected)

    def test_explicit_unsupported_modes_are_excluded(self):
        result=analyze.analyze(synthetic(self.small,self.small_expected,lambda i,m:m==17),self.small_expected)
        self.assertEqual(result['counts']['skipped_mode_observations'],2)
        self.assertEqual(result['counts']['executed_mode_observations'],38)
        self.assertEqual(result['case_results'][0]['explicitly_unsupported_modes'],[17])
        with self.assertRaisesRegex(ValueError,'support differs'):
            analyze.analyze(synthetic(self.small,self.small_expected,lambda i,m:i==1 and m==17),self.small_expected)
        with self.assertRaisesRegex(ValueError,'Mandatory'):
            analyze.analyze(synthetic(self.small,self.small_expected,lambda i,m:m==2),self.small_expected)

    def test_truncation_and_extra_records_fail(self):
        text=synthetic(self.small,self.small_expected)
        for bad in ('\n'.join(text.splitlines()[:-1])+'\n',text+'DONE 2\n'):
            with self.assertRaises(ValueError):analyze.analyze(bad,self.small_expected)

    def test_frozen_helper_is_byte_exact(self):
        source=(HERE/'exact_geometry.f90').read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         '4dfedb9f940dd8e0565ff1ff34b9aa0d56635a5e5d8c52b328359fbfc754afda')

    def test_compiler_loader_injection_is_replaced(self):
        pin=json.loads((HERE/'toolchain-pins.json').read_text())
        names=json.loads((HERE/'ancestry.json').read_text())['compiler_environment_names']
        env=tool_environment.reconstruct(pin,names,{},[],{'PATH':'/fake','LD_PRELOAD':'/fake.so',
             'LD_LIBRARY_PATH':'/fake','FFLAGS':'-ffast-math','NIX_LDFLAGS':'bad','GFORTRAN_CONVERT_UNIT':'big_endian'})
        self.assertEqual(env['PATH'],pin['path'])
        self.assertEqual(env['LD_PRELOAD'],'')
        self.assertEqual(env['LD_LIBRARY_PATH'],'')
        self.assertNotIn('FFLAGS',env)
        self.assertNotIn('GFORTRAN_CONVERT_UNIT',env)

    def test_separate_RUN_admission_and_budgets(self):
        self.assertEqual(controller.CAPS,{'build':180.,'run':180.})
        self.assertEqual(controller.CLEANUP,60.)
        self.assertEqual(controller.LIMITS['address_space_bytes'],4*1024**3)
        self.assertEqual(controller.CPU,12)
        controller.REVIEWED_BUILD=None
        with self.assertRaisesRegex(ValueError,'reviewed BUILD'):
            controller.verify_build()


if __name__=='__main__':unittest.main()
