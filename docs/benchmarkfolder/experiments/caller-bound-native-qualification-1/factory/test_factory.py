import copy
import json
from fractions import Fraction as F
from pathlib import Path
import unittest
from derive_dag import Expr, derive, zero, sum_seeded, momentum_kind
from emit_data import emit
from runtime_reference import (ZERO,MARK,upper_exp,build,finish,route_matches,
                               environment_ok,producer_kind_ok)

HERE=Path(__file__).resolve().parent
D=json.loads((HERE/'dag-metadata.json').read_text())


def word(value,width):
    value=F(value);nf,bias=(52,1023) if width==64 else (112,16383)
    if not value:return 0
    sign=int(value<0);x=abs(value)
    e=x.numerator.bit_length()-x.denominator.bit_length()
    power=F(2**e) if e>=0 else F(1,2**-e)
    if x<power:e-=1;power/=2
    mant=(x/power-1)*2**nf
    assert mant.denominator==1 and 1<=e+bias<2*bias+1
    return (sign<<(width-1))|((e+bias)<<nf)|int(mant)


def leaves(width,boost=0):
    P=[[8,8,8,8],[1,-1,1,-1],[2,-2,2,-2],[4,-4,4,-4]]
    scale=F(2**boost) if boost>=0 else F(1,2**-boost)
    signed=[-1,-1,1,1]
    return [[(word(scale*x*s,width),0) for x,s in zip(row,signed)] for row in P],P,scale


def shift_words(mode_record,P,scale,width):
    return [[word(scale*sum(x*y for x,y in zip(chord['signed_coefficients'],row)),width)
             for row in P] for chord in mode_record['chords']]


class FactoryTests(unittest.TestCase):
    def test_all269_sources_and_all104_metadata_records_rederive(self):
        self.assertEqual(derive(),D)
        self.assertEqual(D['source_files_verified'],269)
        self.assertEqual(sum(m['supported'] for r in D['routes'] for m in r['modes'].values()),104)

    def test_counts_match_each_admitted_source_operation(self):
        # Independent count oracle for the actual two-term W5:8 assignments.
        wfops={w:(0 if w<=4 else 1) for w in range(1,9)}
        wfmult={1:[1,0,0,0],2:[0,1,0,0],3:[0,0,1,0],4:[0,0,0,1],
                5:[1,1,0,0],6:[1,1,0,0],7:[0,0,1,1],8:[0,0,1,1]}
        for r in D['routes']:
            cursor=0;ops=[];occ=[]
            for count in r['pairing']:
                mothers=r['we'][cursor:cursor+count];cursor+=count
                ops.append(sum(wfops[w] for w in mothers)+count)
                occ.append([sum(wfmult[w][e] for w in mothers) for e in range(4)])
            for mode,record in r['modes'].items():
                reverse=int(mode) in (2,5) and r['n']>2
                actualops=list(reversed(ops)) if reverse else ops
                actualocc=list(reversed(occ)) if reverse else occ
                self.assertEqual([x['expanded_add_sub_operations'] for x in record['corners']],actualops)
                for j,chord in enumerate(record['chords']):
                    count=r['n'] if j==0 else j
                    self.assertEqual(chord['expanded_add_sub_operations'],sum(actualops[:count])+count)
                    self.assertEqual(chord['unsigned_leaf_multiplicities'],
                                     [sum(v[e] for v in actualocc[:count]) for e in range(4)])

    def test_every_route_mode_builds_and_finishes_without_numerical_inputs(self):
        for r in D['routes']:
            self.assertTrue(route_matches(D,r['route'],r['n'],r['we'],r['pairing']))
            for mode,record in r['modes'].items():
                width=record['width'];external,P,scale=leaves(width)
                b=build(D,r['route'],int(mode),width,r['n'],external)
                self.assertIsNotNone(b)
                q=finish(width,shift_words(record,P,scale,width),b)
                self.assertEqual(len(q),r['n']*(r['n']-1))
                self.assertEqual(b['pair_class'],record['pair_class'])

    def test_unsigned_multiplicity_survives_signed_cancellation(self):
        e=Expr((1,0,0,0),(1,0,0,0),0)
        node=e.add(e.neg())
        self.assertEqual(node.signed,(0,0,0,0))
        self.assertEqual(node.occurrences,(2,0,0,0))
        self.assertEqual(node.operations,1)
        actual=D['routes'][2]['modes']['1']['differences'][0]
        self.assertEqual(actual['signed_coefficients'][0],0)
        self.assertEqual(actual['unsigned_leaf_multiplicities'][0],2)

    def test_seeded_zero_additions_count_and_closure_not_deleted(self):
        e=Expr((1,0,0,0),(1,0,0,0),0)
        self.assertEqual(sum_seeded([e]).operations,1)
        for r in D['routes']:
            for mode,m in r['modes'].items():
                closure=m['chords'][0]
                self.assertNotEqual(closure['unsigned_leaf_multiplicities'],[0]*4)
                self.assertEqual(closure['signed_coefficients'],
                    [-1,-1,1,1] if int(mode) in (2,5) and r['n']>2 else [1,1,-1,-1])

    def test_boost_changes_bounds_for_all_routes(self):
        for r in D['routes']:
            for mode,m in r['modes'].items():
                width=m['width'];a=build(D,r['route'],int(mode),width,r['n'],leaves(width)[0])
                b=build(D,r['route'],int(mode),width,r['n'],leaves(width,100)[0])
                for va,vb in zip(a['chord_error'],b['chord_error']):
                    self.assertEqual([y-x for x,y in zip(va,vb)],[100]*4)

    def test_bad_route_shape_pairing_or_mother_is_rejected(self):
        r=D['routes'][2];x=leaves(64)[0]
        for rid in (-1,0,27,999):
            self.assertFalse(route_matches(D,rid,4,r['we'],r['pairing']))
            self.assertIsNone(build(D,rid,1,64,4,x))
        self.assertFalse(route_matches(D,3,4,list(reversed(r['we'])),r['pairing']))
        self.assertFalse(route_matches(D,3,4,r['we'],[2,1,1,0]))
        self.assertIsNone(build(D,3,1,64,3,x))
        for mode,width in ((3,64),(4,64),(1,128),(0,64),(5,80)):
            self.assertIsNone(build(D,3,mode,width,4,x))

    def test_unknown_custom_route_and_mixed_precision_are_ineligible(self):
        bad=copy.deepcopy(D);bad['routes'][2]['modes']['1']['supported']=False
        self.assertIsNone(build(bad,3,1,64,4,leaves(64)[0]))
        self.assertFalse(producer_kind_ok(4,128,False))
        self.assertFalse(producer_kind_ok(5,128,False))
        self.assertTrue(producer_kind_ok(4,128,True))
        self.assertTrue(producer_kind_ok(1,64,False))

    def test_wrong_wavefunction_kind_is_not_silently_promoted(self):
        block={'statements':[(1,'TYPE(MP_ALOHA)F1',[])]}
        with self.assertRaises(ValueError):momentum_kind(block,'F1',64)
        momentum_kind(block,'F1',128)

    def test_environment_source_audit_does_not_claim_runtime_proof(self):
        self.assertEqual(D['environment_source_scan']['source_files'],269)
        self.assertEqual(D['environment_source_scan']['matches'],[])
        self.assertIn('separate preconditions',D['environment_source_scan']['scope'])

    def test_ieee_bit_boundaries_real_and_imaginary(self):
        for width in (64,128):
            mode=1 if width==64 else 4;nf,bias=(52,1023) if width==64 else (112,16383)
            self.assertEqual(upper_exp(0,width),ZERO)
            self.assertEqual(upper_exp(1<<(width-1),width),ZERO)
            self.assertEqual(upper_exp(1,width),2-bias-nf)
            self.assertEqual(upper_exp((1<<nf)-1,width),1-bias)
            self.assertEqual(upper_exp(1<<nf,width),2-bias)
            for value in (0,1,1<<(width-1)):
                x=leaves(width)[0];x[0][0]=(value,0)
                self.assertIsNotNone(build(D,3,mode,width,4,x))
            for imag in (1,word(1,width),((2*bias+1)<<nf)|1):
                x=leaves(width)[0];x[0][0]=(x[0][0][0],imag)
                self.assertIsNone(build(D,3,mode,width,4,x))
            x=leaves(width)[0];x[0][0]=(x[0][0][0],1<<(width-1))
            self.assertIsNotNone(build(D,3,mode,width,4,x))

    def test_nan_inf_and_sign_extended_dp_fail_without_fp_decode(self):
        for width in (64,128):
            mode=1 if width==64 else 4;nf,ne=(52,11) if width==64 else (112,15)
            for frac in (0,1,1<<(nf-1)):
                for sign in (0,1):
                    value=(sign<<(width-1))|(((1<<ne)-1)<<nf)|frac
                    x=leaves(width)[0];x[0][0]=(value,0)
                    self.assertIsNone(build(D,3,mode,width,4,x))
        for value in (-1,1<<64,(1<<128)-1):
            with self.assertRaises(ValueError):upper_exp(value,64)

    def test_environment_control_masks_and_false_contract(self):
        token=MARK|(0x1f80<<16)|0x033f
        self.assertTrue(environment_ok(token))
        for change in (0x40<<16,0x8000<<16,0x2000<<16,0x4000<<16,0x400,0x800):
            self.assertFalse(environment_ok(token|change))
        self.assertFalse(environment_ok(token&~0x300))
        for bad in (0,-1,token|(1<<40),token|1<<16):self.assertFalse(environment_ok(bad))
        self.assertIsNone(build(D,3,1,64,4,leaves(64)[0],False))
        self.assertIsNone(finish(64,[],None,False))

    def test_table_reemission_exact_and_fortran_lines_bounded(self):
        self.assertEqual(emit(D),(HERE/'cms_bounds_data.inc').read_text())
        self.assertLessEqual(max(map(len,(HERE/'cms_bounds_data.inc').read_text().splitlines())),132)


if __name__=='__main__':unittest.main()
