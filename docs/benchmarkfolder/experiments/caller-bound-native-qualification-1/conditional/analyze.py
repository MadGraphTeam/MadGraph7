"""Pure conditional evaluation on six sealed trace streams; never loads native code."""
import hashlib, gzip, json, importlib.util
from pathlib import Path
from fractions import Fraction as F
from collections import defaultdict, Counter
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
CAP=ROOT/'docs/benchmarkfolder/experiments/current-kinematic-trace-runtime-qualification-1'
FAC=ROOT/'.tirust-build/kinematic-invariant-caller-metadata-1'
GEN=ROOT/'.tirust-build/kinematic-invariant-general-proposal-1'
PINS={}
def pin(p,sha=None):
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
 if sha: assert h==sha,str(p)
 PINS[str(p.relative_to(ROOT))]={'sha256':h,'bytes':len(b)}
 return b
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def load_packet(path,sha):
 d=json.loads(pin(path/'index.json',sha))
 for name,v in d['files'].items(): assert len(pin(path/name,v['sha256']))==v['bytes']
 return d
def frac(v):return v['value']
def word(v):return sum(int(s,16)<<(64*k) for k,s in enumerate(v['words']))
def nearest(v,p,emin):
 if not v:return F(0)
 sign=-1 if v<0 else 1;v=abs(v)
 e=v.numerator.bit_length()-v.denominator.bit_length()
 if v < F(2)**e:e-=1
 step=F(2)**max(e-p+1,emin-p+1)
 a=v/step;q,r=divmod(a.numerator,a.denominator)
 q+=int(2*r>a.denominator or (2*r==a.denominator and q%2))
 return sign*q*step

def main():
 load_packet(FAC,'aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047')
 load_packet(GEN,'1665eff3188c947f35673b958cd0f02525fd031dc814b708dfdda69160b57964')
 idx=json.loads(pin(CAP/'evidence-index.json','730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4'))
 def read(name):
  c=idx['compressed_text'][name];v=idx['files'][c['path']];b=pin(CAP/c['path'],v['sha256']);assert len(b)==v['bytes']
  raw=gzip.decompress(b);assert len(raw)==c['raw_bytes'] and hashlib.sha256(raw).hexdigest()==c['raw_sha256'];return raw
 decpath=CAP/'analysis/decoder/decode_trace.py';pin(decpath,idx['files']['analysis/decoder/decode_trace.py']['sha256'])
 dec=module('trace_decoder',decpath);fac=module('caller_reference',FAC/'runtime_reference.py');cert=module('cert_reference',GEN/'certificate_reference.py')
 dag=json.loads((FAC/'dag-metadata.json').read_bytes())
 closures={}
 for key,sha in [('ordinary','56d4249509bbda4d1aa3c328b18b42176c7419f4805a1d4454a73a7fa3608fcf'),('stress','df0987b3f0c20dd26c4a903ff7c4acc7fb6cce4e433a50071a82918194553e9e')]:
  raw=read('results/'+key+'-terminal-closure.json');assert hashlib.sha256(raw).hexdigest()==sha;closures[key]=json.loads(raw)
 result=[];streams=[]
 for point in (1,20,21):
  for mode in (1,4):
   name=f'cms-point{point}-mode{mode}-enabled';logical=f'results/runs/{name}/trace.dat';raw=read(logical)
   worker=closures['ordinary' if point==1 else 'stress']['workers'][name]
   assert worker['trace_pin']=={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   schemas,records=dec.parse(raw,'low-first');calls=defaultdict(list)
   for r in records:calls[tuple(r[k] for k in ('section','point','serial','id','mode','rank','n','split_order'))].append(r)
   streams.append({'name':name,'trace_sha256':hashlib.sha256(raw).hexdigest(),'calls':len(calls),'groups':sorted({k[3] for k in calls})})
   for key,rows in calls.items():
    section,pt,serial,route,md,rank,n,so=key;assert pt==point and md==mode
    width=64 if mode==1 else 128;p,emin=(53,-1022) if width==64 else (113,-16382);suf='DP' if width==64 else 'QP'
    one=lambda tag,i=None,j=None:dec.one(rows,'KT_'+tag+'_'+suf,i,j)
    ex=[one('EXTERNAL',i)['data']['complex_momentum'] for i in range(1,5)]
    ext=[[(word(ex[e][2*mu]),word(ex[e][2*mu+1])) for e in range(4)] for mu in range(4)]
    mat=one('MATRIX')['data']['shifts_component_major'];shift=[[mat[mu*n+i] for mu in range(4)] for i in range(n)]
    initial=fac.build(dag,route,mode,width,n,ext,True);assert initial is not None
    finished=fac.finish(width,[[word(v) for v in a] for a in shift],initial,True);assert finished is not None
    for i in range(n):
     for j in range(n):
      if i==j:continue
      rr=one('RAW',i+1,j+1);assert rr is not None
      rawparts=rr['data']['diffsq'];rv=frac(rawparts[0]);assert frac(rawparts[1])==0
      q=[nearest(frac(a)-frac(b),p,emin) for a,b in zip(shift[i],shift[j])]
      calc=nearest(q[0]*q[0],p,emin)
      for v in q[1:]:calc=nearest(calc-nearest(v*v,p,emin),p,emin)
      assert calc==rv,('raw nearest-source replay',key,i,j)
      cls=initial['pair_class'][i][j];err=finished[(i,j)]
      mass=any(one(label+'_BEFORE',i+1,j+1) is not None for label in ('MASSI','MASSJ'))
      zero=one('ZERO_BEFORE',i+1,j+1) is not None
      eligible=cert.certificate(q,rv,err,p,emin,cls,mass,True,True)
      qe=[cert.floor2(x)+1 if x else cert.ZERO for x in q];terms=[]
      for x,e in zip(qe,err):
       if e!=cert.ZERO:
        terms.append(2*e)
        if x!=cert.ZERO:terms.append(1+x+e)
      squares=cert.psum([2*x if x!=cert.ZERO else cert.ZERO for x in qe])
      if squares!=cert.ZERO:terms.append(3-p+squares)
      terms.append(4+emin-p)
      result.append({'stream':name,'section':section,'serial':serial,'route':route,'mode':mode,'rank':rank,'split_order':so,'pair':[i+1,j+1],
        'pair_class':cls,'q_error_exp':err,'raw_words':rawparts[0]['words'],'raw_floor_log2':cert.floor2(rv),'bound_upper_log2':cert.psum(terms),
        'raw_decimal':dec.wire(rv)['decimal_approximation_36_digits'],'mass_snap_seen':mass,'original_zero_branch':zero,
        'conditional_certificate':eligible,'conditional_zero_veto':eligible and zero,'raw_nearest_source_replay_exact':True})
 counts=Counter();critical=[]
 for r in result:
  counts['ordered_pairs']+=1;counts['class_'+str(r['pair_class'])]+=1;counts['original_zero_branches']+=r['original_zero_branch'];counts['conditional_certificates']+=r['conditional_certificate'];counts['conditional_zero_vetoes']+=r['conditional_zero_veto']
  if r['route']==3 and r['pair']==[2,4]:critical.append(r)
 out={'schema':'caller-trace-conditional-v1','scope':'Exact pure source-arithmetic replay; eligibility assumes new fresh environment/actual same-kind producer contracts. Old traces do not prove these contracts. No native execution, output accuracy or amplitude qualification.',
  'streams':streams,'counts':dict(counts),'critical_group3_pair24':critical,'rows':result,'input_pins':PINS}
 return out
if __name__=='__main__':
 out=main();(HERE/'results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'counts':out['counts'],'streams':out['streams'],'critical':out['critical_group3_pair24']},indent=2))
