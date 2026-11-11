# Independent pole derivation

Write D_i=(q+p_i)^2-m_i² and use g=diag(1,-1,-1,-1), D=4−2ε.
After Feynman parametrization, P=Σx_i p_i, F=Σx_i m_i²−Σi<j x_i x_j s_ij,
l=q+P. The measure is normalized so A0_UV=m² and B0_UV=1; no amplitude
1/(16π²) multiplies these tensor moments. Tensor weights are actual Cartesian
components. MadLoop multiplies a rank-r moment by(−1)^r. The shift numerator
(l−P)^α supplies(−1)^(r−2K), which cancels this sign for every surviving term.
This cancellation must not be applied a second time in the adapter.

For 2K loop indices, rotational invariance gives the sum over metric pairings
with coefficient

    (−1)^(N−K) Γ(N−K−2+ε) F^(K+2−N−ε) / 2^K.

The common normalization is1+O(ε²) after scalar pole conventions are imposed;
its quadratic term affects finite coefficients of double poles but no residue
used here. Let k_mu metric pairs use componentmu, K=Σk_mu, beta=α−2k and
j=K+2−N. The number of surviving choices of Cartesian indices is

    W(α,k)= Πmu α_mu! /[(α_mu−2k_mu)! 2^k_mu k_mu!] * (−1)^(k_1+k_2+k_3).

A UV pole exists iff j>=0. Since Res Γ(ε−j)=(−1)^j/j!, the two signs multiply
to+1. Thus

    UV1_α = Σk with j>=0 W(α,k)/(2^K j!) * ∫simplex P^beta F^j dx.

The Wick factor W is not a multinomial output weight. For N=2, j=K and the
integrand is an ordinary polynomial on[0,1]. Coefficients are rational complex
numbers for the dyadic fixture109. Integrating x^n as1/(n+1) is exact. For N=1,
the simplex is a point and j=K+1. Checks at zero shift/equal m²=4 give tadpole
UV(q0²,q0⁴,q0⁶)=(4,8,20), and equal-mass bubble UV(q0²,q0⁴,q0⁶)=(2,6,20).
Spatial squared components change the appropriate metric sign. Scalar bubble
UV1 is1 and rank1 UV1 is p/2 in MadLoop conventions.

## Proposed N=3 infrared fixture

Take p0=0, v=p1=(3,1,2,2), w=p2=(0,2,0,0), all masseszero and mu²4.
v²=(v−w)²=0, w²=−4, so F=4*x0*x2 and P=x1*v+x2*w. There is no complex
logarithm branch ambiguity in this Euclidean parameter polynomial. The exact
external corners are v,w−v,−w and close tozero without floating-point rounding.

For fixed numerator α of rankr, expand

    P^α=Σc A_c x1^(r−c) x2^c,
    A_c=3^α0 * 2^(α2+α3) * binomial(α1,c) * 2^c, 0<=c<=α1.

Only the K=0 loop moment can have infrared poles. Its parameter integral is

    −4^(−1−ε) Γ(1+ε) Γ(−ε) Γ(r−c+1) Γ(c−ε)
      / Γ(r+1−2ε),

with the common scale/normalization factor understood. At mu²4, c=0 gives
IR2=−A0/4 and IR1=−A0 H_r/2. The c>=1 terms have only a simple pole and add

    A_c * (r−c)! * (c−1)! / (4*r!)

to IR1. Hence IR2 is nonzero for all210 components, including all84 exact-rank6
components. The general-scale IR1 additionally contains−log(4/mu²)*IR2.
The K>=1 terms have no endpoint infrared pole and use the UV polynomial formula
above, with j=K−1. For a term A_c from P^beta, its simplex moment is

    ∫ x0^j x1^(rank(beta)−c) x2^(c+j) dx
      = j! (rank(beta)−c)! (c+j)! / (rank(beta)+2j+2)!.

No finite coefficient is predicted.

## Algebraic cross-checks independent of stored reducer answers

The tests check all component contractions allowed through rank6. With T_α the
MadLoop tensor and B_ab the bubble with shiftsa,b:

    v·T_(α+1) = (B_vw,α − B_0w,α)/2,
    w·T_(α+1) = (B_vw,α − B_0v,α − 4*T_α)/2.

Both B_vw and B_0v are scaleless bubbles with separated UV and opposite IR1;
B_0w has spacelike momentum and no IR pole. These follow directly by cancelling
denominators, and test every UV1/IR1/IR2 slot of all126 lower monomials. The
70 rank≤4 traces obey g·T_(α+2)=B_vw,α at pole order. The D−4 loop-component
remainder affects only finite terms in this fixture; no IR pole survives in
that dimension-shifted remainder. These identities do not inspect native output.

A scaleless alternative with shifts0,(3,1,2,2) has UV1=v^α/(r+1), opposite IR1,
and zero IR2 in MadLoop conventions. It also provides all210 nonzero UV/IR1
witnesses, unlike the older lightlike shift whose vanishing spatial components
leave many unwitnessed entries. It is mathematically useful but is not a new
native qualification in this packet.
