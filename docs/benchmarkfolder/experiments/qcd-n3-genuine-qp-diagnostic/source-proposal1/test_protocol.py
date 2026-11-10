"""Synthetic byte traces and fake supervised children only; no real reducers."""
from fractions import Fraction
from pathlib import Path
from unittest import mock
import copy
import json
import struct
import tempfile
import time
import unittest
import analyze as a
import run as control
import shapes

I=lambda *v:struct.pack('<'+'q'*len(v),*v)
D=lambda *v:struct.pack('<'+'d'*len(v),*v)
J=lambda v:struct.pack('<i',v)
P={p:bytes(160) for p in (1,93,39)}
MODEL=bytes(a.MODEL)
FLAGS=I(0,0,0,0,1,0,0,0,0,0,0)

def q(value):
    value=Fraction(value)
    if not value:return bytes(16)
    sign=int(value<0);value=abs(value);e=value.numerator.bit_length()-value.denominator.bit_length()
    if value<Fraction(2)**e:e-=1
    n=value/(Fraction(2)**(e-112));assert n.denominator==1
    return ((sign<<127)|((e+16383)<<112)|(int(n)-(1<<112))).to_bytes(16,'little')

def qs(values):return b''.join(q(v) for v in values)
def ascii64(values):return b''.join((str(v)+'E0').encode().ljust(64,b' ') for v in values)

def sample(backend=9,mode=4,observed=1):
    rows=[]
    def add(call,kind,part,data,lean=False,group=0):
        if kind>=100:group=group or 1
        rows.append(dict(call=call,kind=kind,part=part,group=group,so=int(kind>=100),rank=0,
                         mode=mode if kind>=100 else 0,data=data if lean else data+MODEL+FLAGS))
    for call in (0,1,101,102,103):
        point=(1,93,39)[call-101] if call>=101 else 1
        policy=[backend,mode,point,call,113,128]
        before=policy+([-1]*5 if call==0 else [mode if call>=101 else -1,1,1,1,1])
        after=policy+[mode if call>=101 else -1,1,1,1,1]
        add(call,1,0,P[point]+I(*before))
        if call>=101 and observed:
            add(call,20,0,bytes(160)+I(*([0]*5)));add(call,20,1,bytes(160)+I(*([0]*5)))
            add(call,10,0,bytes(160)+I(-1,1));add(call,10,1,bytes(160)+I(-1,1)+D(1,1))
            add(call,21,0,bytes(320)+I(*([0]*5)));add(call,21,1,bytes(320)+I(*([0]*5)))
            add(call,30,0,bytes(320)+I(mode,backend))
            add(call,32,0,b'');add(call,32,1,b'')
            helicity=bytes(320)+I(0,0,0,0,0,1,0,0,0,0,0)
            add(call,34,0,helicity);add(call,34,1,helicity+bytes(256))
            add(call,36,0,helicity);add(call,36,1,helicity)
            add(call,30,1,bytes(320)+I(mode,backend)+D(*([0]*6)))
            co=q(1)+bytes(2240*116-16);add(call,31,0,co)
            host=bytes(160)+I(*([0]*11));add(call,40,0,host)
            pl=[1,0,0,0,2,0,0,0,-3,0,0,0];mass=[2,0,3,0,4,0]
            routed=[-v for i in (2,1,0) for v in pl[4*i:4*i+4]] if mode==5 else pl
            rmass=[2,0,4,0,3,0] if mode==5 else mass
            shifts=[0,0,0,0]+routed[:4]+[routed[i]+routed[4+i] for i in range(4)]
            if backend!=1:
                add(call,140,0,I(mode,3)+qs(pl)+qs(mass),lean=True)
                add(call,140,1,I(mode,3)+qs(routed)+qs(rmass),lean=True)
            add(call,110,0,co[:2240])
            tag=backend+100;add(call,tag,0,J(3),lean=True);add(call,tag,1,J(0),lean=True)
            enc=(lambda v:D(*v)) if backend==1 else qs if backend==6 else ascii64
            if backend==1:add(call,tag,2,J(mode),lean=True)
            if backend in (6,8):add(call,tag,2,enc([1,0]),lean=True)
            add(call,tag,3,enc(shifts),lean=True);add(call,tag,4,enc(rmass),lean=True)
            add(call,tag,5,enc([1000 if backend==1 else 1000000]),lean=True)
            add(call,tag,6,enc([1000] if backend==1 else [0]*9),lean=True)
            if backend==8:add(call,tag,7,J(0),lean=True)
            if backend==9:add(call,tag,7,struct.pack('<4i',0,1,int(mode==5),1),lean=True)
            if backend==1:
                add(call,130,0,bytes(128),lean=True);add(call,130,1,bytes(128)+qs([1,0]),lean=True)
            add(call,tag,9,J(1 if backend==1 else 0),lean=True)
            add(call,tag,10,enc([0]*(6 if backend in (1,6) else 8)),lean=True)
            if backend in (1,6):add(call,tag,11,enc([0,0]),lean=True)
            if backend==1:add(call,tag,12,enc([0,0]),lean=True)
            add(call,40,1,host+bytes(696*8)+I(*([1]*116)))
        add(call,1,1,P[point]+I(*after)+D(1,2,3,4,1,2,3,4)+D(-1,-1)+I(120 if call>=101 else 200+10+backend))
    add(-2,9,0,I(backend,mode,observed,1,113,128),lean=True)
    return rows

def trace(rows,observed=1,health=0):
    lines=[f'QG1 {observed} {len(rows)} {sum(len(r["data"]) for r in rows)} {health}']
    for i,r in enumerate(rows,1):
        lines.append('E '+str(i)+' '+' '.join(str(r[k]) for k in ('call','kind','part','group','so','rank','mode'))+f' {len(r["data"])} '+r['data'].hex())
    return '\n'.join(lines+['COMPLETE'])+'\n'

def parse(rows,b=9,m=4,o=1,stdout=None):
    return a.parse_worker(trace(rows,o),stdout or f'QG_DONE {b} {m} {o} 1\n',b,m,o,P)

def find(rows,kind,part,call=101):return next(r for r in rows if (r['call'],r['kind'],r['part'])==(call,kind,part))

class ProtocolTests(unittest.TestCase):
    def test_all_16_synthetic_workers_and_unchecked_label(self):
        workers=[parse(sample(b,m,o),b,m,o) for b in a.BACKENDS for m in (4,5) for o in (0,1)]
        result=a.adjudicate(workers)
        self.assertEqual(result['diagnostic_physical_calls'],48)
        self.assertEqual(result['all_public_calls'],80)
        self.assertTrue(result['all_shared_input_gates_pass'])
        self.assertFalse(result['host_checked']);self.assertFalse(result['timing_or_campaign_acceptance'])
        self.assertEqual(len(result['comparisons']),48)
    def test_113bit_transport_preserves_beyond_dp(self):
        value=Fraction(1)+Fraction(1,2**100)
        self.assertEqual(a.binary(q(value),128),[value]);self.assertNotEqual(float(value),value)
        # 1 + 2^-100 has a terminating decimal representation: no context rounding.
        self.assertEqual(a.rounded113(value),value)
        half=Fraction(1)+Fraction(1,2**113);self.assertEqual(a.rounded113(half),1)
    def test_ascii_native_fraction_does_not_narrow(self):
        data=b'1.0000000000000000000000000000007888609052210118054117E0'.ljust(64,b' ')
        self.assertNotEqual(a.ascii113(data)[0],1)
    def test_truncation_even_exit_zero_rejected(self):
        with self.assertRaisesRegex(ValueError,'truncated'):a.parse_trace(trace(sample())[:-10])
    def test_nonzero_health_rejected(self):
        with self.assertRaisesRegex(ValueError,'overflow'):a.parse_trace(trace(sample(),health=1))
    def test_w03_retained_unchecked120_rejected(self):
        with self.assertRaisesRegex(ValueError,'omitted-contribution'):
            parse(sample(),stdout='##w03 unstable sum omitted\nQG_DONE 9 4 1 1\n')
    def test_stop_zero_rejected(self):
        with self.assertRaisesRegex(ValueError,'STOP'):parse(sample(),stdout='STOP 0\nQG_DONE 9 4 1 1\n')
    def test_extra_or_missing_phase_rejected(self):
        rows=sample();rows.remove(find(rows,21,1))
        with self.assertRaises(ValueError):parse(rows)
    def test_native_failure_rejected_without_reading_output(self):
        rows=sample();find(rows,109,9)['data']=J(2);rows.remove(find(rows,109,10))
        with self.assertRaisesRegex(ValueError,'refusal'):parse(rows)
    def test_mass_after_routing_mismatch_rejected(self):
        rows=sample();find(rows,109,4)['data']=ascii64([3,0,3,0,4,0])
        with self.assertRaisesRegex(ValueError,'squared masses'):parse(rows)
    def test_shift_after_routing_mismatch_rejected(self):
        rows=sample();find(rows,109,3)['data']=ascii64([1]*12)
        with self.assertRaisesRegex(ValueError,'shifts differ'):parse(rows)
    def test_nonzero_missing_group_rejected(self):
        rows=sample();r=find(rows,31,0);r['data']=r['data'][:2240]+q(1)+r['data'][2256:]
        with self.assertRaisesRegex(ValueError,'missing native evaluation'):parse(rows)
    def test_failed_group_rejected_despite120(self):
        rows=sample();r=find(rows,40,1);offset=248+696*8;r['data']=r['data'][:offset]+I(0)+r['data'][offset+8:]
        with self.assertRaisesRegex(ValueError,'omitted host group'):parse(rows)
    def test_model_drift_rejected(self):
        rows=sample();r=find(rows,34,0);r['data']=r['data'][:-a.TAIL]+I(1)+r['data'][-a.TAIL+8:]
        with self.assertRaisesRegex(ValueError,'model differs'):parse(rows)
    def test_qp_repaired_input_drift_rejected(self):
        rows=sample();r=find(rows,30,0);r['data']=q(1)+r['data'][16:]
        with self.assertRaisesRegex(ValueError,'canonical repaired'):parse(rows)
    def test_plain_observed_ieee_difference_blocks_comparison(self):
        workers=[parse(sample(b,m,o),b,m,o) for b in a.BACKENDS for m in (4,5) for o in (0,1)]
        workers[1]['public_raw'][0]='changed'
        with self.assertRaisesRegex(ValueError,'IEEE mismatch'):a.adjudicate(workers)
    def test_input_mismatch_retained_without_coeff_comparison(self):
        workers=[parse(sample(b,m,o),b,m,o) for b in a.BACKENDS for m in (4,5) for o in (0,1)]
        workers[1]['points'][0]['inputs']['shared_input_sha256']='different'
        result=a.adjudicate(workers)
        bad=[r for r in result['comparisons'] if not r['shared_physical_born_repair_numerator_equal']]
        self.assertTrue(bad);self.assertTrue(all(r['coefficients'] is None for r in bad))
    def test_duplicate_worker_rejected(self):
        workers=[parse(sample(b,m,o),b,m,o) for b in a.BACKENDS for m in (4,5) for o in (0,1)]
        workers[-1]=workers[0]
        with self.assertRaisesRegex(ValueError,'duplicate'):a.adjudicate(workers)
    def test_symmetric_signed_denominator(self):
        self.assertEqual(a.delta(2,-1)['symmetric'],3)
        self.assertIsNone(a.delta(1,-1)['symmetric']);self.assertTrue(a.delta(1,-1)['opposite_nonzero_sum'])
        self.assertEqual(a.delta(0,0)['symmetric'],0)
    def test_backend_card_only_changes_value(self):
        raw=b'#MLReductionLib\n9\n#CTModeRun\n-1\n'
        self.assertEqual(control.card_for_backend(raw,6),b'#MLReductionLib\n6\n#CTModeRun\n-1\n')
        with self.assertRaises(ValueError):control.card_for_backend(b'#missing\n9\n',6)
    def test_fake_supervisor_exception_retains_failure(self):
        binding={'runtime':{},'programs':{}}
        def fake(*args,**kwargs):
            kwargs['progress']['process_reaped']=True
            raise InterruptedError('fake handled interruption')
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            with self.assertRaises(InterruptedError):
                control.run_command(['fake'],folder,'fake',binding=binding,deadline=time.monotonic()+1,runner=fake)
            record=json.loads((folder/'fake.json').read_text())
            self.assertTrue(record['process_reaped']);self.assertEqual(record['failure']['type'],'InterruptedError')
    def test_expired_stage_never_calls_supervisor(self):
        fake=mock.Mock()
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(TimeoutError):control.run_command(['fake'],Path(temp),'fake',binding={'runtime':{}},deadline=time.monotonic()-1,runner=fake)
        fake.assert_not_called()
    def test_environment_receipt_omits_secret(self):
        def fake(*args,**kwargs):return 0
        with tempfile.TemporaryDirectory() as temp,mock.patch.dict(control.os.environ,{'TEST_SECRET_TOKEN':'not-for-evidence'}):
            control.run_command(['fake'],Path(temp),'fake',binding={'runtime':{}},deadline=time.monotonic()+1,runner=fake)
            self.assertNotIn('not-for-evidence',(Path(temp)/'fake.json').read_text())
    def test_busy_lock_is_hold_before_attempt(self):
        import fcntl
        with tempfile.TemporaryDirectory() as temp,mock.patch.object(control,'ROOT',Path(temp)):
            (Path(temp)/'.tirust-build').mkdir()
            with (Path(temp)/'.tirust-build/numerical-tests.lock').open('a+') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with self.assertRaises(SystemExit) as raised:
                    with control.numerical_lock(): self.fail('busy lock entered')
                self.assertEqual(raised.exception.code,75)
                self.assertEqual([p.name for p in (Path(temp)/'.tirust-build').iterdir()],['numerical-tests.lock'])
    def test_existing_build_is_one_shot(self):
        with tempfile.TemporaryDirectory() as temp,mock.patch.object(control,'WORK',Path(temp)),mock.patch.object(control,'source_check',return_value={}),mock.patch.object(control,'run_command') as launch:
            (Path(temp)/'build1').mkdir()
            with self.assertRaises(FileExistsError):control.build()
            launch.assert_not_called()
    def test_unrecognized_warmup_rejected(self):
        rows=sample();r=find(rows,1,1,1);r['data']=r['data'][:328]+I(120)+r['data'][336:]
        with self.assertRaisesRegex(ValueError,'warmup status'):parse(rows)
    def test_common_order_and_shape_negative_controls(self):
        # Pure existing-source reads, no compiler or object access.
        binding=json.loads((control.HERE/'bindings.json').read_text());base=Path(binding['process'])
        source=[(base/name).read_text() for name in ('coupl.inc','mp_coupl.inc','loop_matrix.f','mp_compute_loop_coefs.f','mp_helas_calls_ampb_1.f','loop_max_coefs.inc')]
        observer=(control.HERE/'observer.f90').read_text()
        self.assertEqual(shapes.verify(*source,observer)['couplings'],34)
        bad=source.copy();bad[0]=bad[0].replace('COMMON/STRONG/ G, ALL_G','COMMON/STRONG/ ALL_G, G')
        with self.assertRaisesRegex(ValueError,'COMMON order'):shapes.verify(*bad,observer)
        with self.assertRaisesRegex(ValueError,'Observer declaration'):shapes.verify(*source,observer.replace('dc(34)','dc(35)'))

if __name__=='__main__':unittest.main(verbosity=2)
