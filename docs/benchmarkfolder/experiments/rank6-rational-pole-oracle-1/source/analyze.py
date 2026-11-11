"""Retrospective coefficient audit; stdlib only, no native/process calls."""
import argparse,gzip,hashlib,json
from decimal import Decimal,localcontext
from fractions import Fraction as F
from pathlib import Path
from oracle import C,ZERO,bubble_uv,triangle_poles,monomials,ieee_fraction,symmetric
HERE=Path(__file__).resolve().parent

def identity(path):
    b=path.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def show(q):
    with localcontext() as c:
        c.prec=70
        return str(Decimal(q.numerator)/Decimal(q.denominator))
def analyze():
    pins=json.loads((HERE/'bindings.json').read_text())
    for old,w in pins['files'].items():
        assert identity(HERE/w['copy'])=={k:w[k] for k in ('bytes','sha256')},old
    audit=json.loads((HERE/'inputs/ancestor-finite-audit-index.json').read_text())
    runtime=json.loads((HERE/'inputs/ancestor-runtime-index.json').read_text())
    assert audit['files']['rows.json.gz']==identity(HERE/'inputs/rows.json.gz')['sha256']
    assert audit['retained_runtime_index_sha256']==identity(HERE/'inputs/ancestor-runtime-index.json')['sha256']
    assert audit['retained_runtime_index_sha256']=='b25130768bc8a5f8daabbcfe820160755b41b4dd2e558cd0b1a87aa49e1f6ec7'
    fixture_key='frozen/repo/docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1/common-input-proposal/fixtures.json'
    assert runtime['files'][fixture_key]==identity(HERE/'inputs/fixtures.json')
    fixtures=json.loads((HERE/'inputs/fixtures.json').read_text())['fixtures']
    fixture=next(f for f in fixtures if f['id']==109)
    snap=json.loads(gzip.decompress((HERE/'inputs/rows.json.gz').read_bytes()))
    alpha=monomials(6)
    assert fixture['n']==2 and fixture['rank']==6
    assert snap['monomials']==[list(a) for a in alpha]
    expected=[bubble_uv(a,fixture['shifts'],fixture['masses_squared'],fixture['invariants'][0][1]) for a in alpha]
    assert len(expected)==210 and all(z!=ZERO for z in expected)
    rows=[r for r in snap['rows'] if r['fixture']==109]
    assert [r['action'] for r in rows]==list(range(3016,3031))
    output=[];tol=F(1,10**8)
    for row in rows:
        assert row['n']==2 and row['rank']==6 and row['stable']
        assert row['mode'] in (0,1,2) and row['backend'] in (7,9) and len(row['values'])==210
        worst=F(0);parts=[];zeros=[];zero_scales=[];zero_complex=True;passed=True
        for i,uv in enumerate(expected):
            for slot in (1,2):
                want=uv if slot==1 and row['mode']!=2 else ZERO
                w=row['values'][i][slot];got=C(ieee_fraction(w[:2]),ieee_fraction(w[2:]))
                error=(got.re-want.re)**2+(got.im-want.im)**2
                norm=(got.re+want.re)**2+(got.im+want.im)**2
                if want==ZERO:
                    zero_complex=zero_complex and got==ZERO
                    passed=passed and got==ZERO
                else:
                    assert norm,'opposite nonzero complex values'
                    ratio=error/norm;passed=passed and ratio<tol**2;worst=max(worst,ratio)
                    for part,(a,b) in enumerate(((got.re,want.re),(got.im,want.im))):
                        if b:parts.append((symmetric(a,b),i,part))
                        else:
                            zeros.append((abs(a),i,part))
                            zero_scales.append((abs(a)/max(abs(want.re),abs(want.im)),i,part))
        with localcontext() as c:
            c.prec=70
            residual=(Decimal(worst.numerator)/Decimal(worst.denominator)).sqrt()
        nz=max(parts,default=(F(0),-1,-1));z=max(zeros,default=(F(0),-1,-1));zs=max(zero_scales,default=(F(0),-1,-1))
        output.append({'action':row['action'],'backend':row['backend'],'ctmode':row['ct'],'pole_mode':row['mode'],'compute_finite':bool(row['finite']),
         'adapter_stable':row['stable'],'original_IEEE_state':row['state'],'coefficient_oracle_gate_1e8':passed,'zero_complex_residues_exact':zero_complex,
         'maximum_complex_symmetric_residual':str(residual),'maximum_complex_residual_squared_exact':str(worst),
         'maximum_nonzero_part_symmetric_residual':show(nz[0]),'nonzero_part_worst_index':nz[1],'nonzero_part_worst_part':nz[2],
         'maximum_expected_zero_part_absolute_leakage':show(z[0]),'zero_part_worst_index':z[1],'zero_part_worst_part':z[2],
         'maximum_zero_part_leakage_scaled_to_exact_UV_component':show(zs[0])})
    return {'schema':'rank6-rational-UV-coefficient-corroboration-v1',
     'scope':'Retrospective exact-rational UV coefficient corroboration only; original inherited IEEE failures unchanged.',
     'fixture':109,'n':2,'rank':6,'components':210,'all_UV_components_nonzero':True,
     'pole_complex_coefficients_checked':6300,'rows':output,
     'all_coefficient_oracle_gates_pass':all(r['coefficient_oracle_gate_1e8'] for r in output),
     'new_native_calls':0,'new_compiler_calls':0,'physical_stress_qualified':False,'new_IEEE_or_stability_qualification':False,'performance_acceptance':False,
     'exact_UV_components':[{'index':i,'powers':list(a),'UV1':z.json(),'IR1':['0','0'],'IR2':['0','0']} for i,(a,z) in enumerate(zip(alpha,expected))]}
def future():
    v=(3,1,2,2);w=(0,2,0,0)
    return {'schema':'rank6-rational-IR-triangle-source-fixture-v1','status':'source-only; not compiled or evaluated',
     'n':3,'rank':6,'mu2':4,'masses_squared':[[0,0]]*3,'shifts':[(0,0,0,0),v,w],
     'external_pl':[v,tuple(w[i]-v[i] for i in range(4)),tuple(-x for x in w)],
     'invariants':[[0,0,-4],[0,0,0],[-4,0,0]],'parameter_polynomial':'F=4*x0*x2',
     'all_IR2_components_nonzero':True,'components':[{'index':i,'powers':list(a),'UV1':p[0].json(),'IR1':p[1].json(),'IR2':p[2].json()} for i,a in enumerate(monomials(6)) for p in [triangle_poles(a)]],
     'finite_oracle':None,'actual_native_input_join':False}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--verify-analysis',action='store_true');p.add_argument('--future',action='store_true');a=p.parse_args()
    result=future() if a.future else analyze()
    if a.verify_analysis:
        assert not a.future and result==json.loads((HERE/'retrospective-analysis.json').read_text())
        print(json.dumps({'verified':True,'rows':len(result['rows']),'native_calls':0}))
    elif a.output:
        with a.output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
        print(json.dumps({'output':str(a.output),'identity':identity(a.output)}))
    else:print(json.dumps(result,indent=2,sort_keys=True))
