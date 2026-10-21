"""Exact dyadic, real Minkowski realization of oneloopmaster STATUS ns9210."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import hashlib,json,os,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
import ninjars_precision_v2 as precision

def bound(p):return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
momenta=[[F(0)]*4,[F(85731,2048),F(92445,2048),F(0),F(0)],
         [F(1899671,32768),F(1993577,32768),F(0),F(0)],
         [F(1838927,32768),F(2054321,32768),F(0),F(0)]]
dot=lambda a,b:a[0]*b[0]-sum(x*y for x,y in zip(a[1:],b[1:]))
matrix=[[dot([x-y for x,y in zip(a,b)],[x-y for x,y in zip(a,b)]) for b in momenta] for a in momenta]
expected=[[0,-74767104,-89257653,-204731997],[-74767104,0,3440331,-27478365],
          [-89257653,3440331,0,0],[-204731997,-27478365,0,0]]
assert matrix==[[F(x,262144) for x in row] for row in expected]
g=[[dot(a,b) for b in momenta[1:]] for a in momenta[1:]]
det=g[0][0]*(g[1][1]*g[2][2]-g[1][2]*g[2][1])-g[0][1]*(g[1][0]*g[2][2]-g[1][2]*g[2][0])+g[0][2]*(g[1][0]*g[2][1]-g[1][1]*g[2][0])
assert det==0
assert all(F(float(x))==x for rows in [momenta,matrix] for row in rows for x in row)
requests=[]
with localcontext() as ctx:
    ctx.prec=180
    ascii=lambda value:format(Decimal(value.numerator)/Decimal(value.denominator),'.39E')
    for negative_zero in [False,True]:
      for mode in [0,1,2]:
        masses=[[6.957667603777042,-4.856459427327334]]*3+[[0.,-0. if negative_zero else 0.]]
        requests.append({'record':len(requests),'case':'ns9210-exact-planar-rank0-box',
            'lane':'negative-mass-zero' if negative_zero else 'positive-mass-zero','mode':mode,'n':4,'rank':0,
            'coefficients':['1','0'],'s_matrix':[ascii(x) for row in matrix for x in row],
            'momenta':[ascii(x) for p in momenta for x in p],
            'masses_squared':[format(x,'.39e') for row in masses for x in row],'scale_squared':['1']})
(REPORT/'planar-exact-requests.json').write_text(json.dumps(requests,indent=2)+'\n')
(REPORT/'planar-exact-requests.txt').write_text(precision.encode(requests))
command=['taskset','-c','190',str(REPORT/'physical-driver')];start=time.monotonic()
with (REPORT/'planar-exact-requests.txt').open() as inp,(REPORT/'planar-exact.stdout').open('x') as out,(REPORT/'planar-exact.stderr').open('x') as err:
    p=subprocess.Popen(command,cwd=ROOT,stdin=inp,stdout=out,stderr=err,start_new_session=True)
    ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split(')',1)[1].split()[19])
    (REPORT/'planar-exact-active.json').write_text(json.dumps({'pid':p.pid,'process_group_id':p.pid,'start_ticks':ticks,'command':command},indent=2)+'\n')
    print('PLANAR_EXACT_PID',p.pid,'START_TICKS',ticks,flush=True)
    try:code=p.wait(timeout=1800)
    except BaseException:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise
outputs=[json.loads(line.split('NINJARS_INSTALLED ',1)[1]) for line in (REPORT/'planar-exact.stdout').read_text().splitlines() if line.startswith('NINJARS_INSTALLED ')]
checks=[]
for request,actual in zip(requests,outputs):
    untouched=actual['output']==(['!'*64]*8 if request['mode']==2 else [None]*8)
    checks.append({'record':request['record'],'mode':request['mode'],'lane':request['lane'],
        'status':actual['status'],'output_untouched':untouched,
        'safely_rejected':actual['status']!=0 and untouched})
proof={'schema':'final-release-exact-planar-full-bridge-v1','requests':bound(REPORT/'planar-exact-requests.json'),
    'request_stream':bound(REPORT/'planar-exact-requests.txt'),'controller':bound(Path(__file__)),
    'stdout':bound(REPORT/'planar-exact.stdout'),'stderr':bound(REPORT/'planar-exact.stderr'),
    'executable':bound(REPORT/'physical-driver'),'archive':bound(ROOT/'.ninjars-build/final-dependency-candidate-20261021/bridge/libninjars.a'),
    'geometry':{'exact_gram_determinant':str(det),'pair_distances_numerators':expected,
        'pair_distances_common_denominator':262144,'all_shifts_and_distances_exactly_binary64':True,
        'coordinate_plane':'real Minkowski time/x, with y=z=0 exactly'},
    'exit_code':code,'outputs':outputs,'checks':checks,'complete':code==0 and len(outputs)==6,
    'all_calls_reject_without_output_commit':len(outputs)==6 and all(row['safely_rejected'] for row in checks),
    'wall_seconds':time.monotonic()-start,'cpu':190,'numerical_threads':1,'performance_claim':False}
(REPORT/'planar-exact-proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:proof[key] for key in ['exit_code','complete','all_calls_reject_without_output_commit','wall_seconds']}),flush=True)
print('STATUSES',[row['status'] for row in outputs],flush=True)
