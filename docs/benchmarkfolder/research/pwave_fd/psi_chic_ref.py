"""Reference for e+ e- -> J/psi chi_cJ (J = 0, 1, 2) at one loop: the NLO
QCD correction to the cross section of Wang, Ma and Chao, arXiv:1107.2646
(PRD 84 (2011) 034022), appendix: sigma_NLO = sigma_LO (1 + alpha_s
Re M_J^NLO / (72 a^3 b d^3 e pi M_J^LO)), photon exchange only, CDR, MSbar
alpha_s with n_f = 4 (charm active, beta_0 = 25/3), on-shell charm mass,
Coulomb terms absorbed into the wave functions. The exclusive final state has
no real emission, so the correction is the virtual alone.

The expressions M_J^NLO below were converted mechanically (sympy) from the
LaTeX source of the arXiv version (jpsichicjpub.tex, appendix); only their
real parts are claimed by the authors. They reproduce the
K factors of the paper's table 1 (1.57, 0.91, 0.78 at mu = 2 m_c and 1.79,
1.25, 1.14 at mu = sqrt(s)/2, sqrt(s) = 10.6 GeV, m_c = 1.5 GeV) and those
of Dong, Feng and Jia, arXiv:1107.4351 v3, tables 1 and 2 (sqrt(s) =
10.58 GeV), whose LO helicity amplitudes, eq. (3.3), give the same M_J^LO
ratios to all digits (selfcheck()).

coefficient(J, sqrts, mc, mu) is A_J in sigma_NLO = sigma_LO (1 + (alpha_s/pi) A_J)
in the n_f = 4 scheme; madloop_c0(...) converts it to MadLoop's normalisation
and scheme (n_l = 3, heavy quarks decoupled): c0 = 2 (A_J + (2 T_F/3)
ln(mu^2/m_c^2)), the cross section average of the finite part in units of
Born * alpha_s/(2 pi) (literature.md, section 0.4).
"""
import mpmath as mp

mp.mp.dps = 40

# M_J^NLO of arXiv:1107.2646, appendix (converted from its LaTeX source)
M_NLO = {
    0: ('-a**6*d**3*l4*(8640*a**4 - 864*a**2 - 648) + a**4*d**3*e*l2*(3072*a**8 - 136320*a**6 + '
        '164160*a**4 - 30816*a**2 - 4320) + a**3*b*d**3*e*lnmu*(9600*a**8 + 436800*a**6 - 256800*a**4'
        ' + 22800*a**2 + 5400) - a**3*b*d*e*ln2*(48021504*a**12 + 10614784*a**10 - 23666688*a**8 - '
        '1180608*a**6 + 1590176*a**4 + 252192*a**2 + 11016) - a**3*b*d*e*lna*(24281088*a**12 + '
        '25235456*a**10 - 18106368*a**8 - 2215296*a**6 + 1069504*a**4 + 195648*a**2 + 9000) + '
        'a**2*b*c*d**3*e*ln2ac*(188032*a**8 + 90048*a**6 - 133824*a**4 + 22880*a**2 + 2352) - '
        'a**2*d**3*e*l3*(1536*a**10 + 69696*a**8 + 15264*a**6 - 18816*a**4 + 1008*a**2 + 216) + '
        'a**2*d**3*e*l7*(2304*a**10 + 18432*a**8 - 19488*a**6 + 10320*a**4 - 1584*a**2 - 216) + '
        'a**2*d**3*e*pi2*(384*a**10 + 22400*a**8 - 26508*a**6 + 7766*a**4 - 162*a**2 - 96) + '
        'a**2*d**3*l1*(-6144*a**12 - 188160*a**10 + 342816*a**8 - 200160*a**6 + 46632*a**4 - '
        '1656*a**2 - 720) + a**2*d**3*l5*(3072*a**12 + 231936*a**10 - 307152*a**8 + 130080*a**6 - '
        '18444*a**4 - 540*a**2 + 252) + a**2*d**3*l6*(231936*a**10 - 403728*a**8 + 212976*a**6 - '
        '34404*a**4 - 1332*a**2 + 360) + a*b*d**2*e*(55296*a**12 + 1092608*a**10 - 611744*a**8 + '
        '81040*a**6 + 17504*a**4 - 4236*a**2 - 432) - d**3*e*lnab*(128*a**12 + 10496*a**10 - '
        '32640*a**8 + 65728*a**6 - 28736*a**4 + 2016*a**2 + 432)'),
    1: ('a**4*d**3*e*l2*(122112*a**6 + 51552*a**4 - 67824*a**2 - 11664) + '
        'a**3*b*d**3*e*lnmu*(115200*a**6 - 172800*a**4 + 46800*a**2 + 16200) - '
        'a**3*b*e*ln2*(37748736*a**14 - 213516288*a**12 - 78815232*a**10 + 79994880*a**8 + '
        '49650048*a**6 + 10483200*a**4 + 960048*a**2 + 33048) - a**3*b*e*lna*(18874368*a**14 - '
        '22413312*a**12 - 86114304*a**10 + 24164352*a**8 + 30219264*a**6 + 7543872*a**4 + 749088*a**2'
        ' + 27000) + a**2*b*c*d**3*e*ln2ac*(18432*a**8 - 28416*a**6 - 45408*a**4 + 48480*a**2 + 7056)'
        ' + a**2*d**3*e*l1*(32256*a**8 + 5760*a**6 - 43200*a**4 + 4500*a**2 + 2160) - '
        'a**2*d**3*e*l3*(118656*a**8 - 39600*a**6 - 31392*a**4 + 1296*a**2 + 648) + '
        'a**2*d**3*e*l5*(12672*a**8 - 35568*a**6 + 22860*a**4 + 1314*a**2 - 756) - '
        'a**2*d**3*e*l6*(77184*a**8 + 28656*a**6 - 55512*a**4 - 3582*a**2 + 1080) - '
        'a**2*d**3*e*l7*(11520*a**8 - 7200*a**6 - 11664*a**4 + 3024*a**2 + 648) - '
        'a**2*d**3*e*pi2*(14784*a**8 + 3576*a**6 - 11196*a**4 - 93*a**2 + 288) + '
        'a*b*d*e*(565248*a**12 - 1975296*a**10 + 998784*a**8 + 438288*a**6 - 32784*a**4 - 19188*a**2 '
        '- 1296) - d**3*e*lnab*(5376*a**10 - 37440*a**8 + 56304*a**6 - 26112*a**4 + 2160*a**2 + 1296)'),
    2: ('-a**4*b**2*d**3*l4*(2592*a**2 + 648) + a**4*d**3*e*l2*(6144*a**8 - 73344*a**6 - 219744*a**4 '
        '- 87984*a**2 - 17712) + a**3*b*d**3*e*lnmu*(19200*a**8 + 96000*a**6 - 225600*a**4 + '
        '92400*a**2 + 27000) - a**3*b*e*ln2*(90439680*a**14 + 204144640*a**12 + 349687808*a**10 + '
        '250902528*a**8 + 93341312*a**6 + 17694592*a**4 + 1599216*a**2 + 55080) - '
        'a**3*b*e*lna*(47185920*a**14 + 140705792*a**12 + 96010240*a**10 + 109231104*a**8 + '
        '57695488*a**6 + 12834560*a**4 + 1253280*a**2 + 45000) + a**2*b*c*d**3*e*ln2ac*(28928*a**8 + '
        '51072*a**6 + 84576*a**4 + 71776*a**2 + 11760) - a**2*d**3*e*l3*(3072*a**10 + 14400*a**8 - '
        '187056*a**6 - 44688*a**4 + 720*a**2 + 1080) + a**2*d**3*e*l7*(4608*a**10 + 17280*a**8 - '
        '2400*a**6 + 5520*a**4 - 3600*a**2 - 1080) + a**2*d**3*e*pi2*(768*a**10 + 11680*a**8 + '
        '19704*a**6 + 15160*a**4 + 633*a**2 - 480) + a**2*d**3*l1*(-12288*a**12 - 58368*a**10 - '
        '10752*a**8 - 139104*a**6 + 84840*a**4 + 2988*a**2 - 3600) + a**2*d**3*l5*(6144*a**12 + '
        '80256*a**10 - 97344*a**8 + 107448*a**6 - 32496*a**4 - 5202*a**2 + 1260) + '
        'a**2*d**3*l6*(105600*a**10 + 188448*a**8 + 47664*a**6 - 68700*a**4 - 10350*a**2 + 1800) + '
        'a*b*d*e*(589824*a**14 + 3301376*a**12 - 6177280*a**10 + 1874880*a**8 + 757744*a**6 - '
        '83984*a**4 - 34572*a**2 - 2160) + d**3*e*lnab*(27392*a**12 + 16256*a**10 - 63744*a**8 + '
        '7504*a**6 + 20224*a**4 - 6192*a**2 - 2160)'),
}

M_LO = {0: '16*a**8 + 728*a**6 - 428*a**4 + 38*a**2 + 9',
        1: '192*a**6 - 288*a**4 + 78*a**2 + 27',
        2: '32*a**8 + 160*a**6 - 376*a**4 + 154*a**2 + 45'}

TF = mp.mpf(1) / 2


def _li2(x):
    # real part of Li2 on the real axis (the same on either side of the cut)
    return mp.re(mp.polylog(2, mp.mpf(x)))


def _variables(sqrts, mc, mu):
    """a ... h and l_1 ... l_7 of arXiv:1107.2646, appendix."""
    a = mp.mpf(sqrts) / (4 * mp.mpf(mc))
    b = mp.sqrt(a**2 - 1)
    c = mp.sqrt(4 * a**2 - 1)
    d = 8 * a**2 + 1
    e = 2 * a**2 - 1
    f = 8 * a**4 + 4 * (2 * a**2 + 1) * a * b + 1
    g = 2 * a * b + 1
    h = 8 * a**4 - 8 * a**2 - 4 * a * b * e + 1
    L = _li2
    v = dict(a=a, b=b, c=c, d=d, e=e)
    v['l1'] = L(1 + b / a)
    v['l2'] = L(1 - b / a)
    v['l3'] = L(e + 2 * a * b) - L(e - 2 * a * b)
    v['l4'] = 2 * L(g) - L(d * g / f) + L(1 / h) - L(g / h) - L(h) + L(d * h / f)
    v['l5'] = (L((a + b)**2 / (e + b * c)) - L((a - b)**2 / (e + b * c))
               + L((a + b)**2 / (e - b * c)) - L((a - b)**2 / (e - b * c)))
    v['l6'] = (L(4 * a**2 - 3 - b * c**2 / a) + L((4 * a**3 + 5 * a - b * c**2) / (a * d))
               - L(d / (32 * a**4 - 24 * a**2 - 8 * b * c**2 * a + 1)))
    v['l7'] = (L((a**2 + a * b + 1) / 2) - L((a**2 - a * b + 1) / 2)
               + L((a**2 + a * b + 1) / (2 * a**2)) - L((a**2 - a * b + 1) / (2 * a**2))
               - 2 * L(1 - b / (2 * a)) + L(a**2 - b * a - b / (2 * a))
               + L((14 * a**3 + 4 * a - (2 * a**2 + 1) * b) / (2 * a * d)) - L(f / d))
    v.update(pi2=mp.pi**2, ln2=mp.log(2), lna=mp.log(a), lnab=mp.log(a + b),
             ln2ac=mp.log(2 * a + c), lnmu=mp.log(mp.mpf(mu) / mp.mpf(mc)))
    return v


def coefficient(J, sqrts, mc, mu):
    """A_J: sigma_NLO = sigma_LO (1 + (alpha_s/pi) A_J), n_f = 4 MSbar."""
    v = _variables(sqrts, mc, mu)
    nlo = eval(M_NLO[J], {}, v)
    lo = eval(M_LO[J], {}, v)
    return nlo / (72 * v['a']**3 * v['b'] * v['d']**3 * v['e'] * lo)


def madloop_c0(J, sqrts, mc, mu):
    """The cross-section average of MadLoop's finite part (units of Born *
    alpha_s/(2 pi)) in its scheme: n_l = 3 MSbar, heavy quarks decoupled.
    alpha_s^(4) = alpha_s^(3) (1 + (alpha_s/pi)(T_F/3) ln(mu^2/m^2)) and a
    Born of order alpha_s^2 give A^(3) = A^(4) + (2 T_F/3) ln(mu^2/m^2)."""
    return 2 * (coefficient(J, sqrts, mc, mu) + 2 * TF / 3 * mp.log((mp.mpf(mu) / mp.mpf(mc))**2))


# Dong, Feng, Jia, arXiv:1107.4351 v3: LO helicity amplitudes, eq. (3.3),
# A^J_{l1,l2} ~ r^((1 + |l1 + l2|)/2) c^J_{l1,l2}(r), r = 4 m_c^2/s; entries
# (l1, l2): (multiplicity in sigma_tot, c). lambda = l1 - l2 fixes the angular
# factor: sin^2(theta) for lambda = 0, (1 + cos^2(theta))/2 for |lambda| = 1.
def dfj_helicities(J, r):
    r = mp.mpf(r)
    s2, s3, s6 = mp.sqrt(2), mp.sqrt(3), mp.sqrt(6)
    return {0: {(0, 0): (1, 1 + 10 * r - 12 * r**2), (1, 0): (2, 9 - 14 * r)},
            1: {(1, 0): (2, -s6 * r), (0, 1): (2, -s6 * (2 - 7 * r)), (1, 1): (2, -2 * s6 * (1 - 3 * r))},
            2: {(0, 0): (1, s2 * (1 - 2 * r - 12 * r**2)), (1, 0): (2, s2 * (3 - 11 * r)),
                (0, 1): (2, s6 * (1 - 5 * r)), (1, 1): (2, 2 * s6 * (1 - 3 * r)), (1, 2): (2, 2 * s3)}}[J]


def dfj_lo_shares(J, r):
    """The LO cross section shares of lambda = 0 and |lambda| = 1."""
    out = [mp.mpf(0), mp.mpf(0)]
    for (l1, l2), (n, c) in dfj_helicities(J, r).items():
        out[abs(l1 - l2)] += n * mp.mpf(r)**(1 + abs(l1 + l2)) * c**2
    tot = out[0] + out[1]
    return out[0] / tot, out[1] / tot


# Dong, Feng, Jia, table 2: single-helicity LO and NLO cross sections (fb),
# sqrt(s) = 10.58 GeV, m_c = 1.5 GeV, mu = 2 m_c, alpha_s(mu) = 0.259, n_f = 4
DFJ_TABLE2 = dict(alpha_s=0.259, sqrts=10.58, mc=1.5, mu=3.0, sigma={
    0: {(0, 0): (1.67, 2.50), (1, 0): (2.80, 4.46)},
    1: {(1, 0): (0.0017, -0.016), (0, 1): (0.56, 0.55), (1, 1): (0.050, 0.020)},
    2: {(0, 0): (0.65, 0.40), (1, 0): (0.40, 0.35), (0, 1): (0.097, 0.089), (1, 1): (0.050, 0.041),
        (1, 2): (0.0035, 0.0026)}})


def dfj_table_groups(J):
    """A of the lambda = 0 and |lambda| = 1 groups from table 2 (n_f = 4):
    sigma_NLO/sigma_LO = 1 + (alpha_s/pi) A; two to three digits."""
    lo, nlo = [0., 0.], [0., 0.]
    for (l1, l2), (s0, s1) in DFJ_TABLE2['sigma'][J].items():
        n = dfj_helicities(J, 0.1)[(l1, l2)][0]
        lo[abs(l1 - l2)] += n * s0
        nlo[abs(l1 - l2)] += n * s1
    k = DFJ_TABLE2['alpha_s'] / float(mp.pi)
    return [(nlo[i] / lo[i] - 1) / k for i in (0, 1)]


def selfcheck():
    rows = []
    for sqrts, mu, als, ref, src in (
            (10.6, 3.0, 0.259, (1.57, 0.91, 0.78), 'WMC table 1'),
            (10.6, 5.3, 0.211, (1.79, 1.25, 1.14), 'WMC table 1'),
            (10.58, 3.0, 0.259, (11.43 / 7.26, 1.11 / 1.22, 1.38 / 1.76), 'DFJ table 2'),
            (10.58, 10.58 / 2, 0.211, (8.62 / 4.83, 1.01 / 0.81, 1.33 / 1.17), 'DFJ table 1')):
        k = [float(1 + als / mp.pi * coefficient(J, sqrts, 1.5, mu)) for J in (0, 1, 2)]
        rows.append(dict(sqrts=sqrts, mu=mu, alpha_s=als, K=k, reference=list(ref), source=src))
        print('sqrt(s) %.2f mu %.2f alpha_s %.3f: K %s, %s %s' % (
            sqrts, mu, als, ' '.join('%.3f' % x for x in k), src, ' '.join('%.3f' % x for x in ref)))
    r = 4 * 1.5**2 / 10.58**2
    lo = {J: eval(M_LO[J], {}, dict(a=mp.mpf(10.58) / 6)) for J in (0, 1, 2)}
    dfj = {J: sum(n * mp.mpf(r)**(1 + abs(l1 + l2)) * c**2 for (l1, l2), (n, c) in dfj_helicities(J, r).items())
           for J in (0, 1, 2)}
    ratios = [(float(lo[J] / lo[0]), float(dfj[J] / dfj[0])) for J in (1, 2)]
    print('M_J^LO / M_0^LO (WMC) against DFJ eq. (3.3):', ratios)
    return dict(k_factors=rows, lo_ratios=ratios)


if __name__ == '__main__':
    selfcheck()
    for J in (0, 1, 2):
        print('J = %d: A (n_f = 4) %.10f, MadLoop c0 %.10f (sqrt(s) = 10.58, m_c = 1.5, mu = 3)'
              % (J, coefficient(J, 10.58, 1.5, 3.0), madloop_c0(J, 10.58, 1.5, 3.0)))
