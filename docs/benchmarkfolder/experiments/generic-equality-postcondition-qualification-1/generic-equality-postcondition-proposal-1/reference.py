"""Checked signed128 algorithm specification, not an executed Fortran result.

Python words are unsigned canonical bit patterns. GNU INTEGER128 carries the
same128 bits as a signed value; callers zero-extend binary64 patterns.
external_bits[leg][component][real/imag], q_bits[component][real/imag].
"""
MAX_INTEGER=(1<<127)-1
MAX_EXTERNAL=64
ZERO_BOUND=-1000000


class Refusal(ValueError):
    pass


def require(value,reason):
    if not value:
        raise Refusal(reason)


def decode(bits,width):
    require(type(bits) is int and 0<=bits<1<<width,'malformed-word')
    nf,ne,bias=(52,11,1023) if width==64 else (112,15,16383)
    stored=(bits>>nf)&((1<<ne)-1)
    require(stored!=(1<<ne)-1,'nonfinite')
    magnitude=bits&((1<<nf)-1)
    exponent=1-bias-nf
    if stored:
        magnitude|=1<<nf
        exponent=stored-bias-nf
    magnitude,exponent=normalize(magnitude,exponent)
    return magnitude,exponent,bool(bits>>(width-1))


def normalize(magnitude,exponent):
    if magnitude:
        zeros=(magnitude&-magnitude).bit_length()-1
        magnitude>>=zeros
        exponent+=zeros
    return magnitude,exponent


def check(width,reference_width,external_bits,coefficients,q_bits):
    """Return mathematical equality only; never infer ownership or FP history."""
    bits_used=[]
    try:
        require(type(width) is int and width in (64,128),'unsupported-width')
        require(type(reference_width) is int and reference_width==width,'reference-width')
        require(type(coefficients) in (list,tuple),'coefficient-shape')
        require(1<=len(coefficients)<=MAX_EXTERNAL,'coefficient-count')
        require(all(type(c) is int and -(1<<31)<=c<(1<<31) for c in coefficients),'coefficient-kind')
        require(type(external_bits) in (list,tuple) and len(external_bits)==len(coefficients),'external-shape')
        require(all(type(e) in (list,tuple) and len(e)==4 and
                    all(type(v) in (list,tuple) and len(v)==2 for v in e)
                    for e in external_bits),'external-shape')
        require(type(q_bits) in (list,tuple) and len(q_bits)==4 and
                all(type(v) in (list,tuple) and len(v)==2 for v in q_bits),'q-shape')
        for mu in range(4):
            terms=[]
            for external,coefficient in zip(external_bits,coefficients):
                imaginary,_,_=decode(external[mu][1],width)
                require(imaginary==0,'nonreal')
                magnitude,exponent,negative=decode(external[mu][0],width)
                c,ce=normalize(abs(coefficient),0)
                if not magnitude or not c:
                    continue
                require(magnitude<=MAX_INTEGER//c,'coefficient-product-capacity')
                terms.append((magnitude*c,exponent+ce,-1 if negative!=(coefficient<0) else 1))
            imaginary,_,_=decode(q_bits[mu][1],width)
            require(imaginary==0,'nonreal')
            magnitude,exponent,negative=decode(q_bits[mu][0],width)
            if magnitude:
                terms.append((magnitude,exponent,1 if negative else -1))
            if not terms:
                bits_used.append(0)
                continue
            common=min(e for _,e,_ in terms)
            absolute_sum=0
            aligned=[]
            for magnitude,exponent,sign in terms:
                shift=exponent-common
                require(0<=shift<=126,'alignment-capacity')
                require(magnitude<=MAX_INTEGER>>shift,'alignment-capacity')
                term=magnitude<<shift
                require(term<=MAX_INTEGER-absolute_sum,'absolute-sum-capacity')
                absolute_sum+=term
                aligned.append(sign*term)
            # All signed arithmetic follows the complete absolute-sum bound.
            require(sum(aligned)==0,'not-equal')
            bits_used.append(absolute_sum.bit_length())
        return {'ok':True,'error_exp':[ZERO_BOUND]*4,'absolute_sum_bits':bits_used,
                'ownership_qualified':False,'Fortran_executed':False}
    except Refusal as error:
        return {'ok':False,'error_exp':None,'reason':str(error),
                'ownership_qualified':False,'Fortran_executed':False}
