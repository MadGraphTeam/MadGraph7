"""Pure exact-arithmetic specification; no scientific/native dependencies."""
from fractions import Fraction as F

ZERO = -1000000


def pow2(n):
    return F(2**n) if n >= 0 else F(1, 2**-n)


def floor2(x):
    x = abs(F(x))
    if not x:
        return ZERO
    n = x.numerator.bit_length() - x.denominator.bit_length()
    return n - (x < pow2(n))


def psum(exponents):
    v = [x for x in exponents if x != ZERO]
    return max(v) + (len(v)-1).bit_length() if v else ZERO


def certificate(q, raw, error_exp, p=53, emin=-1022, pair_class=2,
                mass_snap_seen=False, have_bounds=True, environment_valid=True):
    if (pair_class != 2 or mass_snap_seen or not have_bounds or
            not environment_valid or raw >= 0):
        return False
    if len(q) != 4 or len(error_exp) != 4:
        return False
    if any(x != ZERO and abs(x) > 100000 for x in error_exp):
        return False
    qe = [floor2(x)+1 if x else ZERO for x in q]
    terms = []
    for x, e in zip(qe, error_exp):
        if e != ZERO:
            terms.append(2*e)
            if x != ZERO:
                terms.append(1+x+e)
    squares = psum([2*x if x != ZERO else ZERO for x in qe])
    if squares != ZERO:
        terms.append(3-p+squares)
    terms.append(4+emin-p)
    return floor2(raw) > psum(terms)


def exact_error_bound(q, error_exp, p=53, emin=-1022):
    e = [pow2(x) if x != ZERO else F(0) for x in error_exp]
    return (sum(2*abs(x)*y+y*y for x, y in zip(q, e)) +
            8*pow2(-p)*sum(x*x for x in q) + 16*pow2(emin-p))


def source_chord_bound_exp(leaf_upper_exp, leaf_error_exp, multiplicities,
                           operation_count, p=53, emin=-1022):
    """Upper bound2**e from a validated linear-expression DAG, not final signs.

    Total addition/subtraction count L bounds every leaf path; all reused
    occurrences enter nonnegative multiplicities. L*u <=1/2 is mandatory.
    This helper does not discover/validate the DAG or external input contract.
    """
    if not (len(leaf_upper_exp) == len(leaf_error_exp) == len(multiplicities)):
        raise ValueError('leaf shape')
    L = operation_count
    if not isinstance(L, int) or L < 0 or L >= 2**(p-1):
        raise ValueError('operation bound')
    s, inherited = [], []
    for x, e, m in zip(leaf_upper_exp, leaf_error_exp, multiplicities):
        if not isinstance(m, int) or m < 0:
            raise ValueError('unsigned multiplicity')
        if not m:
            continue
        logm = (m-1).bit_length()
        if x != ZERO:
            s.append(x+logm)
        if e != ZERO:
            inherited.append(e+logm)
    terms = []
    a, b = psum(s), psum(inherited)
    if b != ZERO:
        terms.append(b+1)   # accumulated input errors amplified by <=2
    if L:
        logl = (L-1).bit_length()
        if a != ZERO:
            terms.append(1+logl-p+a)   # gamma_L <=2*L*u
        terms.append(1+logl+emin-p)    # <=2*L*eta
    return psum(terms)


def difference_bound_exp(a_exp, b_exp, a_error, b_error, p=53, emin=-1022):
    terms = [a_error, b_error, emin-p]
    s = psum([a_exp, b_exp])
    if s != ZERO:
        terms.append(s-p)
    return psum(terms)
