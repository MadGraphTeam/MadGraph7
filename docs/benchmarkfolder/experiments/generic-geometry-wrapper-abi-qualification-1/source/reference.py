"""Integer-only reference of the uncompiled Fortran factory."""
ZERO=-1000000
MARK=0x4B45000000000000


def producer_kind_ok(mode,width,genuine_qp):
    return (mode in (1,2) and width==64) or (mode in (4,5) and width==128 and genuine_qp)


def environment_ok(token):
    if token<0 or token>>32 != MARK>>32:return False
    csr=(token>>16)&0xffff;cw=token&0xffff
    return not (csr&0x003f or cw&0xe0c0 or csr&0xe040 or cw&0x0c00) and cw&0x0300==0x0300


def upper_exp(bits,width):
    if width not in (64,128) or bits<0 or bits>=1<<width:raise ValueError('format')
    nf,ne,bias=(52,11,1023) if width==64 else (112,15,16383)
    ex=(bits>>nf)&((1<<ne)-1);f=bits&((1<<nf)-1)
    if ex==(1<<ne)-1:raise ValueError('nonfinite')
    if ex:return ex-bias+1
    return 2-bias-nf+(f.bit_length()-1) if f else ZERO


def psum(v):
    v=[x for x in v if x!=ZERO]
    return max(v)+(len(v)-1).bit_length() if v else ZERO


def route_matches(d,route,n,we,pairing):
    if not 1<=route<=len(d['routes']):return False
    r=d['routes'][route-1]
    return r['route']==route and r['n']==n and r['we']==we and r['pairing']==pairing


def build(d,route,mode,width,n,external_bits,env_valid=True):
    if not env_valid or not 1<=route<=len(d['routes']):return None
    r=d['routes'][route-1]
    if r['n']!=n or str(mode) not in ('1','2','4','5'):return None
    m=r['modes'][str(mode)]
    if not m['supported'] or (64 if mode in (1,2) else 128)!=width:return None
    p,emin=(53,-1022) if width==64 else (113,-16382)
    try:
        up=[]
        for component in external_bits:
            mu=[]
            for real,imag in component:
                if upper_exp(imag,width)!=ZERO:return None
                mu.append(upper_exp(real,width))
            if len(mu)!=d['graph']['nexternal']:return None
            up.append(mu)
        if len(up)!=4:return None
    except (ValueError,TypeError):return None
    errors=[]
    for chord in m['chords']:
        L=chord['ops'];mult=chord['mult']
        if not 0<=L<=1000000 or any(not 0<=x<=1000000 for x in mult):return None
        err=[]
        for mu in range(4):
            total=psum([x+(k-1).bit_length() for x,k in zip(up[mu],mult) if x!=ZERO and k])
            terms=[]
            if L:
                logl=(L-1).bit_length()
                if total!=ZERO:terms.append(1+logl-p+total)
                terms.append(1+logl+emin-p)
            err.append(psum(terms))
        errors.append(err)
    return {'chord_error':errors,'pair_class':m['pair_class'],'have_chord':[True]*n}


def finish(width,shifts,initial,env_valid=True):
    if initial is None or not env_valid or width not in (64,128):return None
    n=len(shifts);p,emin=(53,-1022) if width==64 else (113,-16382)
    errors=initial['chord_error'];classes=initial['pair_class']
    if any(x!=ZERO and not -100000<=x<=100000 for row in errors for x in row):return None
    try:up=[[upper_exp(x,width) for x in row] for row in shifts]
    except ValueError:return None
    out={}
    for i in range(n):
        for j in range(n):
            if i==j or not initial['have_chord'][i] or not initial['have_chord'][j]:continue
            if classes[i][j] not in (1,2):continue
            values=[]
            for mu in range(4):
                mag=psum([up[i][mu],up[j][mu]])
                values.append(psum([errors[i][mu],errors[j][mu],emin-p,mag-p if mag!=ZERO else ZERO]))
            out[(i,j)]=values
    return out
