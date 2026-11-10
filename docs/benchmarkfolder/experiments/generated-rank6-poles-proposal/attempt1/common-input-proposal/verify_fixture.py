"""Pure exact-arithmetic fixture and transformation checks, no numerical load."""
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def determinant(matrix):
    a=[[F(v) for v in row] for row in matrix];value=F(1)
    for i in range(len(a)):
        pivot=next((j for j in range(i,len(a)) if a[j][i]),None)
        if pivot is None:return F(0)
        if pivot!=i:a[i],a[pivot]=a[pivot],a[i];value=-value
        v=a[i][i];value*=v;a[i]=[x/v for x in a[i]]
        for j in range(i+1,len(a)):
            v=a[j][i];a[j]=[x-v*y for x,y in zip(a[j],a[i])]
    return value

def dot(a,b):return a[0]*b[0]-sum(x*y for x,y in zip(a[1:],b[1:]))

def verify():
    data=json.loads((HERE/'fixtures.json').read_text())
    schedule=json.loads((HERE/'schedule.json').read_text())
    assert len(schedule)==30 and [r['action'] for r in schedule]==list(range(3001,3031))
    keys=('action','fixture','backend','ct','mode','finite','cache','budget','clear','n','rank')
    assert (HERE/'schedule.dat').read_text()=='30\n'+''.join(' '.join(str(r[k]) for k in keys)+'\n' for r in schedule)
    for fixture in data['fixtures']:
        q,pl,n=fixture['shifts'],fixture['external_pl'],fixture['n']
        assert fixture['rank']==6 and q[0]==[0]*4 and len(q)==len(pl)==n
        assert all(sum(p[i] for p in pl)==0 for i in range(4))
        assert [[q[(j+1)%n][i]-q[j][i] for i in range(4)] for j in range(n)]==pl
        expected=[]
        for a in q:
            row=[]
            for b in q:
                v=[x-y for x,y in zip(a,b)];row.append(dot(v,v))
            expected.append(row)
        assert expected==fixture['invariants']
        assert all(F(value).denominator in (1,2,4,8) for mass in fixture['masses_squared'] for value in mass)
        assert all(mass[0]>200 and mass[1]<0 for mass in fixture['masses_squared'])
    q=data['fixtures'][0]['shifts'];minors=[]
    for n in range(2,6):
        for subset in combinations(range(6),n):
            vectors=[[q[k][i]-q[subset[0]][i] for i in range(4)] for k in subset[1:]]
            det=determinant([[dot(a,b) for b in vectors] for a in vectors]);assert det
            minors.append({'subset':list(subset),'determinant':str(det)})
    assert minors==data['exact_nonzero_gram_minors']
    old=(HERE.parent/'materialized/ew/driver.f').read_text()
    new=old.replace('M20(6,7)','M20(6,9)').replace('PL0(0:3,6,7)','PL0(0:3,6,9)').replace('BASEMU(7)','BASEMU(9)')
    start=new.index("      OPEN(72,FILE='point.dat',STATUS='OLD')");end=new.index('      CALL UPDATE_AS_PARAM()',start)
    new=new[:start]+'      MU_R=8D0\n      MU0=MU_R\n'+new[end:]
    start=new.index('      DO K=1,2\n');end=new.index('      COLLIERGLOBALCACHE=0',start)
    block='C     Independent exact common-input fixtures; no physical capture calls.\n'
    for item in data['fixtures']:
        f=item['id']-100
        for k,leg in enumerate(item['external_pl'],1):block+=f'      PL0(:,{k},{f})=(/'+','.join(str(v)+'D0' for v in leg)+'/)\n'
        for k,(mr,mi) in enumerate(item['masses_squared'],1):block+=f'      M20({k},{f})=CMPLX({mr}D0,{mi}D0,KIND=8)\n'
        block+=f'      BASEMU({f})=8D0\n'
    new=new[:start]+block+new[end:]
    new=new.replace("WRITE(71,*) 'HEADER',1,MODEL,ORIGINAL_NC","WRITE(71,*) 'HEADER',2,MODEL,210").replace('F.GT.107','F.GT.109')
    new=new.replace('      CALL RP_OPEN()\n','      CALL RP_OPEN()\n      CALL RP_STATE(FLAG0,ROUND0)\n      WRITE(71,*) \'INITIAL_IEEE\',FLAG0,ROUND0\n',1)
    new=new.replace('      CALL UPDATE_AS_PARAM()\n','      CALL UPDATE_AS_PARAM()\n      CALL RP_STATE(FLAG0,ROUND0)\n      WRITE(71,*) \'MODEL_IEEE\',FLAG0,ROUND0\n',1)
    assert new==(HERE/'driver.f').read_text()
    return {'passed':True,'rows':30,'nonzero_exact_gram_minors':len(minors),'scope':'source-only exact arithmetic and driver transformation'}

if __name__=='__main__':print(json.dumps(verify(),sort_keys=True))
