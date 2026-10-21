"""Consistent planar rank-zero box control for the upstream ns9210 master."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal,localcontext
import hashlib,json,os,signal,subprocess,sys,time

ROOT=Path(__file__).resolve().parents[2]
REPORT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
import ninjars_precision_v2 as precision

def bound(p):
    return {'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}

def round113(value):
    if not value:return F(0)
    negative=value<0;value=abs(value)
    exponent=value.numerator.bit_length()-value.denominator.bit_length()
    power=lambda n:F(2)**n
    if value<power(exponent):exponent-=1
    scaled=value*power(112-exponent)
    quotient,remainder=divmod(scaled.numerator,scaled.denominator)
    if 2*remainder>scaled.denominator or (2*remainder==scaled.denominator and quotient%2):quotient+=1
    result=F(quotient)*power(exponent-112)
    return -result if negative else result

with localcontext() as ctx:
    ctx.prec=180
    decimal=lambda x:Decimal(x.numerator)/Decimal(x.denominator)
    record=lambda x:format(decimal(x),'.39E')
    p1,p2,p3,p4,s12,s23=map(F,[-285.2138671875,13.123821258544922,0.,-780.9905891418457,-340.4909248352051,-104.8216438293457])
    a,b,c=p1,s12,p4;g12=(a+b-p2)/2;g13=(a+c-s23)/2;g23=(b+c-p3)/2
    determinant=a*(b*c-g23*g23)-g12*(g12*c-g23*g13)+g13*(g12*g23-b*g13)
    assert determinant==0
    x1=decimal(-a).sqrt();x2=-decimal(g12)/x1;t2=(decimal(b)+x2*x2).sqrt();x3=-decimal(g13)/x1;t3=(decimal(g23)+x2*x3)/t2
    momenta=[[F(0)]*4,[F(0),round113(F(x1)),F(0),F(0)],
             [round113(F(t2)),round113(F(x2)),F(0),F(0)],
             [round113(F(t3)),round113(F(x3)),F(0),F(0)]]
    exact_matrix=[[F(0),p1,s12,p4],[p1,F(0),p2,s23],[s12,p2,F(0),p3],[p4,s23,p3,F(0)]]
    rounded_matrix=[]
    for left in momenta:
        row=[]
        for right in momenta:
            q=[round113(x-y) for x,y in zip(left,right)]
            value=round113(round113(q[0]*q[0])-round113(q[1]*q[1]))
            row.append(value)
        rounded_matrix.append(row)
    requests=[]
    for matrix_name,matrix in [('exact-STATUS-S',exact_matrix),('rounded-113-momentum-S',rounded_matrix)]:
        for negative_zero in [False,True]:
            masses=[[6.957667603777042,-4.856459427327334]]*3+[[0.,-0. if negative_zero else 0.]]
            requests.append({'record':len(requests),'case':'ns9210-planar-rank0-box','lane':matrix_name+('-negative-zero' if negative_zero else '-positive-zero'),
                'mode':2,'n':4,'rank':0,'coefficients':['1','0'],'s_matrix':[record(x) for row in matrix for x in row],
                'momenta':[record(x) for p in momenta for x in p],
                'masses_squared':[format(x,'.39e') for row in masses for x in row],'scale_squared':['1']})
    consistency=max(abs(x-y) for rowa,rowb in zip(exact_matrix,rounded_matrix) for x,y in zip(rowa,rowb))
    metadata={'exact_gram_determinant':str(determinant),'coordinate_plane':'time/x, with y=z=0 exactly',
        'momentum_rounding':'nearest113 binary bits, ties to even; 40-significant-digit ASCII transport',
        'largest_squared_distance_difference':str(decimal(consistency)),
        'original_S_scale':str(max(abs(x) for row in exact_matrix for x in row)),
        'rounded_momentum_squared_distances':[[record(x) for x in row] for row in rounded_matrix]}
(REPORT/'planar-bridge-requests.json').write_text(json.dumps(requests,indent=2)+'\n')
(REPORT/'planar-bridge-requests.txt').write_text(precision.encode(requests))
command=['taskset','-c','190',str(REPORT/'physical-driver')];start=time.monotonic()
with (REPORT/'planar-bridge-requests.txt').open() as inp,(REPORT/'planar-bridge.stdout').open('x') as out,(REPORT/'planar-bridge.stderr').open('x') as err:
    p=subprocess.Popen(command,cwd=ROOT,stdin=inp,stdout=out,stderr=err,start_new_session=True)
    ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split(')',1)[1].split()[19])
    (REPORT/'planar-bridge-active.json').write_text(json.dumps({'pid':p.pid,'process_group_id':p.pid,'start_ticks':ticks,'command':command},indent=2)+'\n')
    print('PLANAR_BRIDGE_PID',p.pid,'START_TICKS',ticks,flush=True)
    try:code=p.wait(timeout=1800)
    except BaseException:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        raise
outputs=[json.loads(line.split('NINJARS_INSTALLED ',1)[1]) for line in (REPORT/'planar-bridge.stdout').read_text().splitlines() if line.startswith('NINJARS_INSTALLED ')]
proof={'schema':'final-release-planar-full-bridge-v1','requests':bound(REPORT/'planar-bridge-requests.json'),
    'request_stream':bound(REPORT/'planar-bridge-requests.txt'),'geometry':metadata,'controller':bound(Path(__file__)),
    'stdout':bound(REPORT/'planar-bridge.stdout'),'stderr':bound(REPORT/'planar-bridge.stderr'),
    'executable':bound(REPORT/'physical-driver'),'archive':bound(ROOT/'.ninjars-build/final-dependency-candidate-20261021/bridge/libninjars.a'),
    'exit_code':code,'outputs':outputs,'complete':code==0 and len(outputs)==4,'wall_seconds':time.monotonic()-start,
    'all_calls_reject_without_output_commit':len(outputs)==4 and all(x['status']!=0 and x['output']==['!'*64]*8 for x in outputs),
    'performance_claim':False,'independent_master_failure_is_not_assumed_to_be_full_reduction_failure':True}
(REPORT/'planar-bridge-proof.json').write_text(json.dumps(proof,indent=2,allow_nan=False)+'\n')
print(json.dumps({key:proof[key] for key in ['exit_code','complete','all_calls_reject_without_output_commit','wall_seconds']}),flush=True)
print('STATUSES',[row['status'] for row in outputs],flush=True)
