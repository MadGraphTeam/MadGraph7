"""Pure exact numeric join from recorded native outputs to Fortran words.

No library loads. Signed-zero transport is deliberately outside this numeric
relation; original words remain retained. Partial finite markers are explicit.
"""
from fractions import Fraction as F
import re,struct

def require(x,m):
    if not x:raise ValueError(m)
def rounded(value,precision):
    q=F(value)
    if not q:return F(0)
    sign=-1 if q<0 else 1;q=abs(q)
    e=q.numerator.bit_length()-q.denominator.bit_length()
    if q<F(2)**e:e-=1
    step=max(e-precision+1,-1074 if precision==53 else -16494)
    unit=F(2)**step;n,r=divmod((q/unit).numerator,(q/unit).denominator)
    if 2*r>(q/unit).denominator or 2*r==(q/unit).denominator and n&1:n+=1
    return sign*n*unit

def mapping(source):
    statements=[]
    for line in source.splitlines():
        if not line or line[0] in 'Cc*!':continue
        if len(line)<6:continue
        code=line[6:].replace(' ','')
        if not code:continue
        if line[5] not in (' ','0'):
            require(bool(statements),'Orphan fixed-form continuation');statements[-1]+=code
        else:statements.append(code)
    groups=[];current={}
    for row in statements:
        m=re.fullmatch(r'DATA\(TIRUST_MAP\(I\),I=(\d+),(\d+)\)/([0-9,]+)/',row)
        if not m:continue
        lo,hi=int(m[1]),int(m[2]);values=list(map(int,m[3].split(',')))
        if lo==0 and current:groups.append(current);current={}
        require(len(values)==hi-lo+1,'Mapping count differs')
        for i,k in zip(range(lo,hi+1),values):
            require(i not in current,'Repeated mapping index');current[i]=k-1
    if current:groups.append(current)
    require(len(groups)==2 and groups[0]==groups[1],'DP/QP mapping tables differ/missing')
    result=[groups[0][i] for i in range(210)]
    require(sorted(result)==list(range(210)),'Mapping is not a complete permutation')
    return result

def outputs(frame):
    raw=bytes.fromhex(frame['out']);precision=53 if frame['kind']==1 else 113
    if precision==53:
        require(len(raw)==8*210*8,'DP native output length')
        values=[F.from_float(v[0]) for v in struct.iter_unpack('<d',raw)]
    else:
        require(len(raw)==8*210*64,'QP native output length')
        values=[rounded(F(raw[i:i+64].decode('ascii').strip()),113) for i in range(0,len(raw),64)]
    return values,precision

def verify(cases,frames,source,ieee_fraction):
    permutation=mapping(source);checked=markers=0
    for row in cases:
        frame=frames[row['action']]
        require(row['stable'] and frame['present']==1 and frame['status']==0,'Cannot join failed native result')
        values,p=outputs(frame)
        def raw(slot,k,part):return values[2*(slot*210+k)+part]
        for i,k in enumerate(permutation):
            for slot in range(3):
                words=row['values'][i][slot]['words']
                if slot==0 and not row['finite']:
                    bits=int(words[0],16)|(int(words[1],16)<<64)
                    require((bits>>112)&32767==32767 and bits&((1<<112)-1) and bits&(1<<111),'Partial finite is not quiet NaN')
                    require(ieee_fraction(words[2:])==0,'Partial finite imaginary marker differs');markers+=1;continue
                for part in range(2):
                    expected=raw(3,k,part) if slot==0 else rounded(raw(0,k,part)+raw(1,k,part),p) if slot==1 else raw(2,k,part)
                    actual=ieee_fraction(words[part*2:part*2+2])
                    require(actual==expected,'Native-to-host numeric transport/mapping differs at action/index/slot/part '+str((row['action'],i,slot,part)))
                    checked+=1
    require(checked==25200 and markers==2520,'Native-to-host coverage differs')
    return {'native_output_buffers':24,'mapping_components':210,'host_real_values_checked':checked,
            'partial_quiet_finite_markers_checked':markers,'native_to_host_numeric_join':True,
            'signed_zero_transport_qualified':False,'new_native_calls':0}
