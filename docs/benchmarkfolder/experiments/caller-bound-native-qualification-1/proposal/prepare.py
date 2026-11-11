"""Pure deterministic source/corpus preparation. No compiler, native or subprocess."""
import copy, hashlib, importlib.util, json
from pathlib import Path
from fractions import Fraction as F
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
FACTORY=ROOT/'.tirust-build/kinematic-invariant-caller-metadata-1'
SHA='aba903450f49e0e18ba3fff40dbf6074e1cd67e9af1c67cd83bbd6fd583c1047'
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(b):return hashlib.sha256(b).hexdigest()
def verify_factory():
 b=(FACTORY/'index.json').read_bytes();assert sha(b)==SHA;d=json.loads(b)
 for name,v in d['files'].items():
  b=(FACTORY/name).read_bytes();assert {'sha256':sha(b),'bytes':len(b)}==v
 return d

def word(v,width):
 v=F(v);nf,bias=(52,1023) if width==64 else (112,16383)
 if not v:return 0
 sign=int(v<0);v=abs(v);e=v.numerator.bit_length()-v.denominator.bit_length()
 if v<F(2)**e:e-=1
 m=(v/F(2)**e-1)*(1<<nf);assert m.denominator==1
 return (sign<<(width-1))|((e+bias)<<nf)|int(m)

def corpus():
 verify_factory();ref=load('caller_reference',FACTORY/'runtime_reference.py');dag=json.loads((FACTORY/'dag-metadata.json').read_bytes());cases=[]
 physical=[[8,8,8,8],[1,-1,1,-1],[2,-2,2,-2],[4,-4,4,-4]]
 def base(r,mode,boost=0):
  width=r['modes'][str(mode)]['width'];scale=F(2)**boost
  ext=[[(word(scale*v*s,width),0) for v,s in zip(row,[-1,-1,1,1])] for row in physical]
  shifts=[[word(scale*sum(a*b for a,b in zip(c['signed_coefficients'],row)),width) for row in physical] for c in r['modes'][str(mode)]['chords']]
  return dict(route=r['route'],mode=mode,width=width,n=r['n'],we=r['we'],pairing=r['pairing'],env=True,genuine=True,external=ext,shifts=shifts,label='route-mode')
 def add(c,label=None):
  c=copy.deepcopy(c);c['id']=len(cases)+1
  if label:c['label']=label
  cases.append(c)
 for r in dag['routes']:
  for mode in (1,2,4,5):add(base(r,mode))
 for width,mode in [(64,1),(128,4)]:
  c=base(dag['routes'][2],mode);nf,ne,bias=(52,11,1023) if width==64 else (112,15,16383)
  for boost in (-100,100):add(base(dag['routes'][2],mode,boost),'boost'+str(boost))
  values=[0,1,(1<<nf)-1,1<<nf,1<<(width-1),((1<<ne)-1)<<nf,(((1<<ne)-1)<<nf)|1,(((1<<ne)-1)<<nf)|(1<<(nf-1)),((1<<width)-1)]
  for v in values:
   x=copy.deepcopy(c);x['external'][0][0]=(v,0);add(x,'external-boundary-'+hex(v))
   x=copy.deepcopy(c);x['shifts'][0][0]=v;add(x,'shift-boundary-'+hex(v))
  for im in (1,1<<(width-1),word(1,width)):
   x=copy.deepcopy(c);x['external'][0][0]=(x['external'][0][0][0],im);add(x,'imaginary-'+hex(im))
  for field,v in [('env',False),('genuine',False),('route',0),('route',27),('mode',3),('width',80)]:
   x=copy.deepcopy(c);x[field]=v;add(x,field+'-'+str(v))
  x=copy.deepcopy(c);x['we']=list(reversed(x['we']));add(x,'bad-WE-order')
  x=copy.deepcopy(c);x['pairing']=[2,1,1,0];add(x,'bad-pairing')
  x=copy.deepcopy(c);x['n']=3;x['pairing']=x['pairing'][:3];x['shifts']=x['shifts'][:3];add(x,'bad-n')
  x=copy.deepcopy(c);x['we']=x['we'][:-1];add(x,'bad-nw')
  x=copy.deepcopy(c);x['width']=128 if width==64 else 64;add(x,'wrong-kind')
 for v in ((1<<128)-1,1<<64):
  x=base(dag['routes'][2],1);x['external'][0][0]=(v,0);add(x,'DP-upper-bits')
 expected=[]
 for c in cases:
  route_ok=ref.route_matches(dag,c['route'],c['n'],c['we'],c['pairing']);producer=ref.producer_kind_ok(c['mode'],c['width'],c['genuine'])
  b=ref.build(dag,c['route'],c['mode'],c['width'],c['n'],c['external'],c['env'] and route_ok and producer)
  f=ref.finish(c['width'],c['shifts'],b,c['env'] and route_ok and producer)
  n=c['n'];chords=b['chord_error'] if b else [[ref.ZERO]*4 for _ in range(n)];classes=b['pair_class'] if b else [[0]*n for _ in range(n)]
  expected.append({'id':c['id'],'label':c['label'],'route_ok':route_ok,'producer_ok':producer,
   'chords':[{'have':b is not None,'errors':x} for x in chords],
   'pairs':[{'i':i+1,'j':j+1,'class':classes[i][j],'have':f is not None and (i,j) in f,'errors':f[(i,j)] if f is not None and (i,j) in f else [ref.ZERO]*4} for j in range(n) for i in range(n)]})
 token=ref.MARK|(0x1f80<<16)|0x033f
 tokens=[token,0,token|1<<40,token|1<<16,token&~0x300,*[token|x for x in (0x40<<16,0x8000<<16,0x2000<<16,0x4000<<16,0x400,0x800)]]
 return cases,expected,[{'word':t,'expected':ref.environment_ok(t)} for t in tokens]

def render(cases,tokens):
 lines=[str(len(cases))]
 for c in cases:
  lines.append(' '.join(map(str,[c['id'],c['route'],c['mode'],c['width'],c['n'],len(c['we']),int(c['env']),int(c['genuine'])])))
  lines+=[' '.join(map(str,c['we'])),' '.join(map(str,c['pairing']))]
  lines+=[' '.join(f'{c["external"][mu][leg][part]:032X}' for leg in range(4) for mu in range(4) for part in range(2))]
  lines+=[' '.join(f'{v:032X}' for shift in c['shifts'] for v in shift)]
 return '\n'.join(lines)+'\n',str(len(tokens))+'\n'+'\n'.join(f'{t["word"]:016X}' for t in tokens)+'\n'

def main():
 c,e,t=corpus();raw,tokens=render(c,t)
 for name,value in [('cases.json',c),('expected.json',e),('tokens.json',t)]: (HERE/name).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
 (HERE/'inputs.txt').write_text(raw);(HERE/'tokens.txt').write_text(tokens)
 print(json.dumps({'cases':len(c),'base_routes_modes':104,'token_cases':len(t),'native_executions':0}))
if __name__=='__main__':main()
