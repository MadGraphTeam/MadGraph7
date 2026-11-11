import ast,copy,importlib.util,json,struct,unittest
from decimal import Decimal as D,localcontext
from fractions import Fraction as F
from pathlib import Path
from unittest.mock import patch
import analyze as a
from oracle import ZERO,triangle_poles,scaleless_bubble_poles,monomials
HERE=Path(__file__).resolve().parent

def words(q,dp=False):
    if q is None:return ['0000000000000000','7FFF800000000000']
    q=F(q)
    if dp:q=F.from_float(float(q))
    sign=int(q<0);q=abs(q)
    if not q:return ['0000000000000000','0000000000000000']
    e=q.numerator.bit_length()-q.denominator.bit_length()
    if q<F(2)**e:e-=1
    value=q/F(2)**(e-112);n,r=divmod(value.numerator,value.denominator)
    if 2*r>value.denominator or 2*r==value.denominator and n&1:n+=1
    if n==1<<113:n//=2;e+=1
    bits=(sign<<127)|((e+16383)<<112)|(n-(1<<112))
    return [f'{bits&((1<<64)-1):016X}',f'{bits>>64:016X}']
def component(real,imag=0,dp=False):
    rw,iw=words(real,dp),words(imag,dp)
    return {'words':rw+iw,'value':[a.lp.binary128(rw),a.lp.binary128(iw)]}
def encoded(values,kind):
    if kind==1:return b''.join(struct.pack('<d',float(x)) for x in values).hex()
    return b''.join(str(int(x)).ljust(64).encode() for x in values).hex()
def synthetic():
    cases=[];frames={};errors={}
    for spec in json.loads((HERE/'schedule.json').read_text()):
        fixture=json.loads((HERE/('future-triangle-fixture.json' if spec['fixture']==110 else 'future-scaleless-fixture.json')).read_text())
        row={**spec,'entry':{'p':[[w for x in leg for w in a.lp.promoted(x)] for leg in fixture['external_pl']],
          'm':[[w for x in mass for w in a.lp.promoted(x)] for mass in fixture['masses_squared']]},'stable':True,'state':[32,0,32,0,1,0,0],
          'scale':'4000000000000000','pre':[0]*15,'post':[0]*15}
        row['exit']=copy.deepcopy(row['entry']);row['values']=[]
        for alpha in monomials(6):
            uv,ir1,ir2=triangle_poles(alpha) if spec['fixture']==110 else scaleless_bubble_poles(alpha)
            simple,double=(uv+ir1,ir2) if spec['mode']==0 else (uv,ZERO) if spec['mode']==1 else (ir1,ir2)
            finite=None if not spec['finite'] else 0 if spec['fixture']==111 else 1
            row['values'].append([component(finite,dp=spec['ct']==1),component(simple.re,simple.im,dp=spec['ct']==1),component(double.re,double.im,dp=spec['ct']==1)])
        cases.append(row);kind=1 if spec['ct']==1 else 2
        frame={'present':1,'kind':kind,'n':spec['n'],'rank':6,'mode':spec['mode'],'finite':spec['finite'],'check':0,'cache':0,'status':0,'flags':[32]*3,'rounding':[0]*3,'error':''}
        for key,values in [('p',[x for leg in fixture['shifts'] for x in leg]),('s',[x for leg in fixture['invariants'] for x in leg]),('m',[0]*(2*spec['n'])),('mu',[4])]:frame[key]=encoded(values,kind)
        frames[spec['action']]=frame;errors[spec['action']]=''
    return cases,frames,errors

class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with localcontext() as c:c.prec=180;cls.seed=synthetic()
    def fixture(self):return copy.deepcopy(self.seed)
    def evaluate(self,data,initial=(0,0),model=(32,0)):
        with localcontext() as c:
            c.prec=100
            return a.analyze_rows(*data,list(initial),list(model))
    def test_24_row_matrix_and_positive_oracle_semantics(self):
        result=self.evaluate(self.fixture())
        self.assertEqual(result['status'],'passed');self.assertEqual(len(result['checks']),24)
        self.assertEqual(len(result['full_partial']),12)
        self.assertTrue(all(p['status']=='passed' for p in result['full_partial']))
    def test_native_cache_or_precision_override_rejected(self):
        for field,value in [('cache',1),('kind',2)]:
            d=self.fixture();d[1][4101][field]=value
            with self.assertRaisesRegex(ValueError,'options/precision/cache'):self.evaluate(d)
    def test_native_input_changed_rejected(self):
        d=self.fixture();d[1][4101]['mu']=encoded([16],1)
        with self.assertRaisesRegex(ValueError,'native mu'):self.evaluate(d)
    def test_inherited_or_new_IEEE_failure_retained(self):
        self.assertEqual(self.evaluate(self.fixture(),initial=(1,0))['status'],'open_acceptance_items')
        d=self.fixture();d[1][4101]['flags'][1]=33
        self.assertEqual(self.evaluate(d)['status'],'open_acceptance_items')
    def test_refusal_retained_without_truth_claim(self):
        d=self.fixture();d[0][0]['stable']=False;d[0][0]['state'][4]=0;d[1][4101]['status']=3;d[1][4101]['error']=b'unsupported'.hex()
        result=self.evaluate(d)
        self.assertEqual(result['status'],'open_acceptance_items');self.assertIsNone(result['checks'][0]['oracle'])
        self.assertTrue(any(x['kind']=='native_refusal' for x in result['open_items']))
    def test_bad_pole_and_partial_marker_rejected(self):
        d=self.fixture();d[0][0]['values'][0][2]=component(17)
        self.assertEqual(self.evaluate(d)['status'],'open_acceptance_items')
        d=self.fixture();d[0][1]['values'][0][0]=component(0)
        with self.assertRaisesRegex(ValueError,'finite NaN'):self.evaluate(d)
    def test_source_includes_register_both_plans_and_no_amplitude(self):
        text=(HERE/'driver.f').read_text().upper()
        self.assertIn('CALL TIRUST_REGISTER_PLAN(2,6,STATUS)',text);self.assertIn('CALL TIRUST_REGISTER_PLAN(3,6,STATUS)',text)
        self.assertNotIn('CALL ML5_',text);self.assertNotIn('CALL CAP6_COLLIER',text)
        self.assertNotIn('IEEE_SET',text);self.assertNotIn('FECLEAR',text)
        self.assertTrue(all(len(x)<=132 for x in text.splitlines()))
    def test_syntax_and_reused_ownership_definitions(self):
        for p in HERE.glob('*.py'):ast.parse(p.read_text())
        parent=(HERE.parent/'fortran-qp-output-bulk-unit-controller-2/controller.py').read_text()
        source=(HERE/'controller.py').read_text()
        def selected(s):return {x.name:ast.dump(x,include_attributes=False) for x in ast.parse(s).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name in ['StageDeadline','raw_cleanup','cleanup_owned']}
        self.assertEqual(selected(parent),selected(source))

class AdministrativeTests(unittest.TestCase):
    def test_preflight_failure_never_spawns_and_is_retained(self):
        import controller,tempfile
        with tempfile.TemporaryDirectory(prefix='rank6-source-stage-') as tmp:
            with patch.object(controller,'WORK',Path(tmp)/'work'),patch.object(controller,'preflight',side_effect=ValueError('synthetic preflight failure')),patch.object(controller.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
                with self.assertRaisesRegex(ValueError,'synthetic preflight'):controller.stage('build')
                result=json.loads((Path(tmp)/'work/build/terminal.json').read_text())
                self.assertEqual(result['status'],'failed');self.assertNotIn('child_pid',result)
    def test_existing_stage_directory_untouched(self):
        import controller,tempfile
        with tempfile.TemporaryDirectory(prefix='rank6-source-existing-') as tmp:
            work=Path(tmp)/'work';(work/'build').mkdir(parents=True);(work/'build/sentinel').write_text('untouched')
            with patch.object(controller,'WORK',work),patch.object(controller.subprocess,'Popen',side_effect=AssertionError('must not spawn')):
                with self.assertRaises(FileExistsError):controller.stage('build')
            self.assertEqual([p.name for p in (work/'build').iterdir()],['sentinel'])

if __name__=='__main__':unittest.main()
