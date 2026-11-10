"""Fabricated text/source-only tests. These are not numerical evidence."""
import copy
from decimal import Decimal as D, localcontext
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('rank6_analyzer',HERE/'analyze.py')
analyze=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analyze)


class BinaryTransport(unittest.TestCase):
    def test_known_binary128_patterns(self):
        self.assertEqual(analyze.binary128(['0000000000000000','3FFF000000000000']),D(1))
        self.assertEqual(analyze.binary128(['0000000000000000','C000000000000000']),D(-2))
        self.assertTrue(analyze.binary128(['0000000000000001','7FFF000000000000']).is_nan())
    def test_double_promotions(self):
        self.assertEqual(analyze.promoted(1.),['0000000000000000','3FFF000000000000'])
        self.assertEqual(analyze.promoted(-0.),['0000000000000000','8000000000000000'])
        for value in [0.,-0.,1.,-2.,0.1,float.fromhex('0x0.0000000000001p-1022')]:
            with localcontext() as ctx:
                ctx.prec=180
                self.assertEqual(analyze.binary128(analyze.promoted(value)),+D.from_float(value))
    def test_decimal_binary128_rounding(self):
        with localcontext() as ctx:
            ctx.prec=180
            # Exact halfway selects the even significand; the next midpoint
            # selects the adjacent even value, independently of host floats.
            halfway=D(1)+D(2)**-113
            self.assertEqual(analyze.rounded113(halfway),D(1))
            self.assertEqual(analyze.rounded113(D(1)+3*D(2)**-113),D(1)+D(2)**-111)
            for value in [0.1,1000.,float.fromhex('0x0.0000000000001p-1022')]:
                self.assertEqual(analyze.rounded113(D(format(value,'.39e'))),+D.from_float(value))


class ObserverText(unittest.TestCase):
    def frame(self):
        return {'schema':1,'action':1,'present':1,'kind':1,'n':2,'rank':0,
                'mode':0,'finite':1,'check':0,'cache':0,'status':0,
                'flags':[0,32,32],'rounding':[0,0,0],'health':0,
                'p':'00'*64,'s':'00'*32,'m':'00'*32,'mu':'00'*8,
                'out':'00'*64,'error':'','cll_p':'','cll_s':'','cll_m':''}
    def parse(self,frames,expected=None):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observer.txt'
            path.write_text('\n'.join(json.dumps(f) for f in frames)+'\n')
            return analyze.parse_observer(path,expected or [1])
    def test_valid_fabricated_frame(self):
        self.assertEqual(self.parse([self.frame()])[1]['status'],0)
    def test_missing_extra_and_duplicate_records(self):
        for rows in [[],[self.frame(),self.frame()]]:
            with self.subTest(rows=len(rows)),self.assertRaises(ValueError):self.parse(rows)
    def test_unknown_key_and_wrong_schema(self):
        for key,value in [('schema',True),('extra',1),('n',7),('rank',7),('status',9)]:
            frame=self.frame();frame[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.parse([frame])
    def test_fenv_and_health_tampering(self):
        for key,value in [('flags',[0,32,33]),('rounding',[0,0,1024]),('health',1)]:
            frame=self.frame();frame[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.parse([frame])
    def test_native_lengths(self):
        for key in ['p','s','m','mu','out']:
            frame=self.frame();frame[key]=frame[key][:-2]
            with self.subTest(key=key),self.assertRaises(ValueError):self.parse([frame])
    def test_absent_native_metadata_is_clean(self):
        frame=self.frame();frame.update(present=0,kind=0,finite=0,flags=[0,0,0])
        for k in ['p','s','m','mu','out']:frame[k]=''
        self.parse([frame])
        frame['status']=1
        with self.assertRaises(ValueError):self.parse([frame])
    def test_duplicate_json_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'observer.txt'
            path.write_text(json.dumps(self.frame())[:-1]+',"action":1}\n')
            with self.assertRaises(ValueError):analyze.parse_observer(path,[1])


class SourceSchedule(unittest.TestCase):
    def test_fixed_coverage_budget(self):
        rows=json.loads((HERE/'schedule.json').read_text())
        self.assertEqual(len(rows),123)
        self.assertEqual(len({r['action'] for r in rows}),123)
        self.assertEqual(sum(r['backend']==9 for r in rows)+3,105)
        self.assertEqual(sum(r['backend']==7 for r in rows),21)
        self.assertEqual({r['fixture'] for r in rows},set(range(101,108)))
    def test_materialized_fresh_copies_and_call_order(self):
        for model in ['ew','qcd']:
            text=(HERE/'materialized'/model/'driver.f').read_text()
            start=text.index('DO STEP=1,NSTEPS')
            copy=text.index('PL=PL0(:,:,F-100)',start)
            mass=text.index('M2=M20(:,F-100)',start)
            begin=text.index('CALL RP_BEGIN(A)',start)
            self.assertLess(copy,begin);self.assertLess(mass,begin)
            self.assertIn("SETPARA('nominal/param_card.dat')",text)
    def test_missing_records_and_bare_stop_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'records.txt'
            for text in ['','STOP 0\n','HEADER 1 1 126\n']:
                path.write_text(text)
                with self.subTest(text=text),self.assertRaises(ValueError):analyze.parse_records(path,'qcd',[])


class RecordFrames(unittest.TestCase):
    """One fabricated capture and one rank-zero frame, not a fake campaign."""
    def fixture(self):
        zero='0000000000000000'
        def inputs(p,m,n):
            return [line for i in range(1,n+1) for line in
                    [p+' '+str(i)+' '+' '.join([zero]*8),m+' '+str(i)+' '+' '.join([zero]*4)]]
        def values(n):
            return ['V '+str(i)+' '+str(s)+' '+' '.join([zero]*4)+' 0 0'
                    for i in range(n) for s in (1,2,3)]
        spec={'action':1,'fixture':106,'backend':9,'ct':1,'mode':0,'finite':1,
              'cache':0,'budget':268435456,'clear':1,'n':2,'rank':0}
        lines=['HEADER 1 1 126','CAPTURE 903 1 1 1']+inputs('PI','MI',6)+values(126)+['CAPTURE_END 903']
        lines+=['CASE 1 106 9 1 0 1 0 268435456 1 2 0 1','SCALE 4000000000000000']
        lines+=inputs('PI','MI',2)+['CACHE_PRE '+' '.join(['0']*15),'STATE 0 0 0 0 1 0 0']
        lines+=inputs('PO','MO',2)+values(1)+['CACHE_POST '+' '.join(['0']*15),'END 1 0']
        lines+=['FINAL 0 1 -1 '+' '.join(['0']*15),'DONE 0']
        return lines,[spec]
    def parse(self,lines,spec):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'records.txt';path.write_text('\n'.join(lines)+'\n')
            return analyze.parse_records(path,'qcd',spec)
    def test_one_fabricated_frame(self):
        captures,rows=self.parse(*self.fixture())
        self.assertEqual(len(captures),1);self.assertEqual(len(rows),1)
    def test_missing_extra_coefficient_rows(self):
        original,spec=self.fixture()
        i=next(i for i,line in enumerate(original) if line.startswith('V '))
        for lines in [original[:i]+original[i+1:],original[:i]+[original[i]]+original[i:]]:
            with self.assertRaises(ValueError):self.parse(lines,spec)
    def test_missing_final_or_extra_trailing_record(self):
        original,spec=self.fixture()
        for lines in [original[:-1],original[:-2]+original[-1:],original+['STOP 0']]:
            with self.assertRaises(ValueError):self.parse(lines,spec)
    def test_final_context_and_cache_failure(self):
        original,spec=self.fixture()
        i=next(i for i,line in enumerate(original) if line.startswith('FINAL '))
        for field in [1,3,10]:
            lines=original[:];tokens=lines[i].split();tokens[field]='1';lines[i]=' '.join(tokens)
            with self.subTest(field=field),self.assertRaises(ValueError):self.parse(lines,spec)


if __name__=='__main__':unittest.main(verbosity=2)
