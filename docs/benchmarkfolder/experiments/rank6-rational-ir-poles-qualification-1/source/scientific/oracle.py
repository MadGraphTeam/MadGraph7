"""Exact Cartesian UV residues and a fixed massless-triangle IR oracle.

Pure stdlib Fraction algebra. No reducer/native imports or stored-answer fitting.
The normalized integral uses denominators (q+p_i)^2-m_i^2, (+---), actual
components, and the MadLoop (-1)^rank convention. No amplitude 1/(16*pi^2).
"""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
from math import factorial, comb

@dataclass(frozen=True)
class C:
    re: F = F(0)
    im: F = F(0)
    def __post_init__(self):
        object.__setattr__(self,"re",F(self.re))
        object.__setattr__(self,"im",F(self.im))
    @classmethod
    def of(cls,z):
        return z if isinstance(z,C) else cls(F(z))
    def __add__(a,b):
        b=C.of(b);return C(a.re+b.re,a.im+b.im)
    __radd__=__add__
    def __neg__(a):return C(-a.re,-a.im)
    def __sub__(a,b):return a+-C.of(b)
    def __rsub__(a,b):return C.of(b)+-a
    def __mul__(a,b):
        b=C.of(b);return C(a.re*b.re-a.im*b.im,a.re*b.im+a.im*b.re)
    __rmul__=__mul__
    def __truediv__(a,b):
        b=F(b);return C(a.re/b,a.im/b)
    def __pow__(a,n):
        assert isinstance(n,int) and n>=0
        out=C(1)
        for _ in range(n):out=out*a
        return out
    def json(a):return [str(a.re),str(a.im)]

ZERO=C()

def p_add(a,b):
    n=max(len(a),len(b))
    return tuple((a[i] if i<len(a) else ZERO)+(b[i] if i<len(b) else ZERO) for i in range(n))

def p_mul(a,b):
    out=[ZERO]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]=out[i+j]+x*y
    return tuple(out)

def p_pow(a,n):
    assert n>=0
    out=(C(1),)
    for _ in range(n):out=p_mul(out,a)
    return out

def interval_integral(poly):
    return sum((x/(i+1) for i,x in enumerate(poly)),ZERO)

def pairing_terms(alpha):
    """Cartesian index selections, not external multinomial tensor weights."""
    assert len(alpha)==4 and all(isinstance(a,int) and a>=0 for a in alpha)
    for pairs in product(*(range(a//2+1) for a in alpha)):
        k=sum(pairs)
        beta=tuple(a-2*b for a,b in zip(alpha,pairs))
        ways=1
        for a,b in zip(alpha,pairs):ways*=factorial(a)//(factorial(a-2*b)*2**b*factorial(b))
        ways*=(-1)**sum(pairs[1:])
        yield k,beta,ways

def bubble_uv(alpha,shifts,masses_squared,invariant):
    """Exact N=2 UV residue from its polynomial parameter integral."""
    assert len(shifts)==len(masses_squared)==2 and all(len(p)==4 for p in shifts)
    p0,p1=([F(x) for x in p] for p in shifts)
    m0,m1=(C(*m) if not isinstance(m,C) else m for m in masses_squared)
    s=F(invariant)
    momenta=[(C(p0[mu]),C(p1[mu]-p0[mu])) for mu in range(4)]
    f=(m0,m1-m0-C(s),C(s))
    result=ZERO
    for k,beta,ways in pairing_terms(alpha):
        poly=p_pow(f,k)
        for mu,b in enumerate(beta):poly=p_mul(poly,p_pow(momenta[mu],b))
        result=result+interval_integral(poly)*F(ways,2**k*factorial(k))
    return result

def tadpole_uv(alpha,shift,mass_squared):
    """N=1 UV residue; independently useful zero-shift normalization controls."""
    assert len(shift)==4
    m=C(*mass_squared) if not isinstance(mass_squared,C) else mass_squared
    result=ZERO
    for k,beta,ways in pairing_terms(alpha):
        value=m**(k+1)*F(ways,2**k*factorial(k+1))
        for p,b in zip(shift,beta):value=value*F(p)**b
        result=result+value
    return result

def _triangle_P_terms(beta):
    # p0=0, p1=(3,1,2,2), p2=(0,2,0,0).
    # P^beta = sum_c A_c x1^(rank-c) x2^c.
    rank=sum(beta)
    other=3**beta[0]*2**(beta[2]+beta[3])
    return [(rank-c,c,other*comb(beta[1],c)*2**c) for c in range(beta[1]+1)]

def triangle_poles(alpha):
    """Exact [UV1,IR1,IR2] for the proposed Euclidean fixture at mu^2=4.

    shifts0,(3,1,2,2),(0,2,0,0); all masses0. F=4*x0*x2.
    Nonzero IR2 for all210 components throughrank6. No finite-term oracle.
    """
    rank=sum(alpha)
    a0=3**alpha[0]*2**(alpha[2]+alpha[3])
    harmonic=sum((F(1,k) for k in range(1,rank+1)),F(0))
    ir2=F(-a0,4)
    ir1=2*harmonic*ir2
    for b,c,a in _triangle_P_terms(alpha):
        if c:ir1+=F(a*factorial(b)*factorial(c-1),4*factorial(rank))
    uv=F(0)
    for k,beta,ways in pairing_terms(alpha):
        j=k-1
        if j<0:continue
        for b,c,a in _triangle_P_terms(beta):
            integ=F(factorial(j)*factorial(b)*factorial(c+j),factorial(b+c+2*j+2))
            uv+=F(ways*a*4**j,2**k*factorial(j))*integ
    return (C(uv),C(ir1),C(ir2))

def scaleless_bubble_poles(alpha,lightlike=(3,1,2,2)):
    assert sum(F(x)**2*(1 if i==0 else -1) for i,x in enumerate(lightlike))==0
    residue=F(1,sum(alpha)+1)
    for p,a in zip(lightlike,alpha):residue*=F(p)**a
    return (C(residue),C(-residue),ZERO)

def monomials(maxrank):
    # MadLoop's symmetric component ordering: nondecreasing index tuples.
    import itertools
    out=[]
    for rank in range(maxrank+1):
        for indices in itertools.combinations_with_replacement(range(4),rank):
            out.append(tuple(indices.count(mu) for mu in range(4)))
    # Real MadLoop ordering is reverse lexicographic by the largest component;
    # sort powers by that component count (and recursively lower counts).
    return sorted(out,key=lambda a:(sum(a),a[3],a[2],a[1]))

def ieee_fraction(words):
    """Read binary128 words exactly. Reject nonfinite results."""
    assert len(words)==2
    bits=int(words[0],16)|(int(words[1],16)<<64)
    sign=-1 if bits>>127 else 1
    exponent=(bits>>112)&32767
    significand=bits&((1<<112)-1)
    if exponent==32767:raise ValueError("nonfinite binary128")
    if exponent:significand+=1<<112
    shift=(exponent or 1)-16383-112
    return F(sign*significand)*(F(2)**shift)

def symmetric(actual,expected):
    numerator=abs(actual-expected);denominator=abs(actual+expected)
    if denominator:return numerator/denominator
    return F(0) if not numerator else None
