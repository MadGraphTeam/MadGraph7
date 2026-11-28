"""He-Huang (arXiv:2608.23439) H -> l+ l- gamma at LO and NLO (virtual only)
with MadLoop: eta_c, chi_c0,1,2 with l = e, mu and photon-energy cuts 100,
300, 500 MeV in the rest frame of H.

Inputs (their sect. 3): alpha = 1/137, m_Q = m_H/2 per state (p0^2 = m_H^2 =
4 m_Q^2), |R_S(0)|^2 = 0.810 GeV^3, |R'_P(0)|^2 = 0.075 GeV^5, m_e = 0.511e-3,
m_mu = 0.106 GeV, mu_R = m_H, on-shell mass and field renormalisation. QED
only (no Z, no H exchange).

The decay LDME of the onia card (Block ldme_decay) is <O_H> = 2 Nc |R_S(0)|^2
/(4 pi) for an S wave and 2 Nc 3 |R'_P(0)|^2/(4 pi) for a P wave (whatever J),
the normalisation of the tree-level output (checked against eta_c, chi_c0,2 ->
a a, J/psi -> e+ e-).

Gamma = 1/(2M) int |M|^2 dPhi_3, dPhi_3 = p* |p_gamma| /(64 pi^3 M m12)
dm12^2 dcos(theta*), theta* the lepton angle in the pair rest frame (|M|^2 is
summed over helicities and averaged over the 2J+1 states of H). The virtual
is IR finite (checked: the poles vanish), 2 Re(M0* M1) = born * finite_norm *
alpha_s/(2 pi), linear in alpha_s, so the NLO width is given as
Gamma_LO + alpha_s * C with C = int born finite_norm/(2 pi).

Usage: HH_WORK=<scratch dir> [HH_CUT_MODE=mH-m12] python3 -I hh.py
       generate|run|run_rest <state> <lepton> [nodes]
"""
import sys, os, re, json, math
REPO = '/shared/external/madloop_onia/madgraph7_madloop_onia'
sys.path.insert(0, os.path.join(REPO, 'tools'))
import onia_loop_check as olc
# the MadLoop outputs and the result JSON files go to $HH_WORK (default: the
# current directory), not next to this script
HERE = os.path.abspath(os.environ.get('HH_WORK', os.getcwd()))

STATES = {  # name: (process name, pdg, hadron mass, L)
    'etac': ('etac(1|1S01)', 441, 2.984, 0),
    'chic0': ('chic0(1|3P01)', 10441, 3.415, 1),
    'chic1': ('chic1(1|3P11)', 20443, 3.511, 1),
    'chic2': ('chic2(1|3P21)', 445, 3.556, 1),
}
LEPTONS = {'e': ('e+ e-', 11, 0.511e-3), 'mu': ('mu+ mu-', 13, 0.106)}
R2S, R2P = 0.810, 0.075
CUTS = (0.5, 0.3, 0.1)
CUT_MODE = os.environ.get('HH_CUT_MODE', 'Egamma')


def outdir(state, lepton):
    return os.path.join(HERE, 'hh_%s_%s' % (state, lepton))


def generate(state, lepton):
    proc, pdg, mh, L = STATES[state]
    leps, lpdg, ml = LEPTONS[lepton]
    out = olc.output_process('loop_sm_onia-full', '%s > %s a / z h [virt=QCD]' % (proc, leps),
                             outdir(state, lepton), regenerate=True)
    configure(state, lepton)
    pdir = olc.subprocess_dir(out)
    olc.build_driver(pdir)
    return pdir


def set_entry(text, block, key, value):
    """Set the value of entry `key` of `block` in an SLHA-like card."""
    m = re.search(r'(?im)^block\s+%s\b.*$' % re.escape(block), text)
    if not m:
        raise ValueError('no block %s' % block)
    start = m.end()
    nxt = re.search(r'(?im)^(block|decay)\s', text[start:])
    end = start + (nxt.start() if nxt else len(text) - start)
    body = text[start:end]
    pat = re.compile(r'(?m)^(\s*%s\s+)(\S+)' % re.escape(str(key)))
    if not pat.search(body):
        raise ValueError('no entry %s in block %s' % (key, block))
    body = pat.sub(lambda mm: mm.group(1) + ('%.10e' % value), body, count=1)
    return text[:start] + body + text[end:]


def configure(state, lepton):
    proc, pdg, mh, L = STATES[state]
    leps, lpdg, ml = LEPTONS[lepton]
    d = outdir(state, lepton)
    card = os.path.join(d, 'Cards', 'param_card.dat')
    text = open(card).read()
    text = set_entry(text, 'mass', 4, mh / 2)
    text = set_entry(text, 'mass', lpdg, ml)
    text = set_entry(text, 'sminputs', 1, 137.0)
    open(card, 'w').write(text)
    ocard = os.path.join(d, 'Cards', 'onia_card.dat')
    text = open(ocard).read()
    ldme = 6 * (R2S if L == 0 else 3 * R2P) / (4 * math.pi)
    text = set_entry(text, 'ldme_decay', pdg, ldme)
    text = set_entry(text, 'onium_mass', pdg, mh)
    open(ocard, 'w').write(text)


def gauss_legendre(n):
    """Nodes and weights on [-1, 1] (Newton on P_n)."""
    xs, ws = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            dx = p1 / dp
            x -= dx
            if abs(dx) < 1e-16:
                break
        xs.append(x)
        ws.append(2 / ((1 - x * x) * dp * dp))
    return xs, ws


def momenta(M, ml, m12, cth):
    """H at rest; the photon along -z, the pair along +z; l+ at angle theta*
    to +z in the pair rest frame (phi = 0). Order: H, l+, l-, a."""
    eg = (M * M - m12 * m12) / (2 * M)
    epair = M - eg
    ps = math.sqrt(max(m12 * m12 / 4 - ml * ml, 0.0))
    sth = math.sqrt(max(1 - cth * cth, 0.0))
    gam, bg = epair / m12, eg / m12   # pair momentum = eg along +z
    def boost(e, pz):
        return gam * e + bg * pz, bg * e + gam * pz
    el = m12 / 2
    e1, z1 = boost(el, ps * cth)
    e2, z2 = boost(el, -ps * cth)
    return [[M, 0.0, 0.0, 0.0], [e1, ps * sth, 0.0, z1], [e2, -ps * sth, 0.0, z2],
            [eg, 0.0, 0.0, -eg]], ps, eg


def boost(p, b):
    """The 4-vector p boosted with velocity b (3-vector)."""
    b2 = b[0]**2 + b[1]**2 + b[2]**2
    g = 1 / math.sqrt(1 - b2)
    bp = b[0]*p[1] + b[1]*p[2] + b[2]*p[3]
    f = (g - 1) * bp / b2 + g * p[0]
    return [g * (p[0] + bp)] + [p[i + 1] + f * b[i] for i in range(3)]


# a generic boost (|beta| = 0.6): the widths are Lorentz invariant, and a
# P-wave bound state exactly at rest is a degenerate configuration for the
# reduction (TIRust's QP pass hits a floating-point exception there)
BOOST = [0.6 * x / math.sqrt(14) for x in (1.0, 2.0, 3.0)]


def run(state, lepton, nodes=16, ncos=4, boosted=True):
    proc, pdg, mh, L = STATES[state]
    leps, lpdg, ml = LEPTONS[lepton]
    d = outdir(state, lepton)
    pdir = olc.subprocess_dir(d)
    configure(state, lepton)
    olc.set_reduction(d, [9])
    M = mh
    # u = ln(m12^2 - 4 ml^2): smooth at threshold (p* ~ e^{u/2})
    umin = math.log(4 * ml * ml) - 30
    # the cut: E_gamma > c in the rest frame of H, m12^2 < M^2 - 2 M c; or
    # (CUT_MODE 'mH-m12', what reproduces the tables of He-Huang) m12 < M - c
    if CUT_MODE == 'mH-m12':
        tops = [(M - c) ** 2 for c in CUTS]
    else:
        tops = [M * M - 2 * M * c for c in CUTS]
    edges = [umin, math.log(4 * ml * ml) + 2] + [math.log(t - 4 * ml * ml) for t in tops]
    xs, ws = gauss_legendre(nodes)
    cs, cw = gauss_legendre(ncos)
    pts, meta = [], []
    for ip in range(len(edges) - 1):
        a, b = edges[ip], edges[ip + 1]
        for x, w in zip(xs, ws):
            u = 0.5 * (a + b) + 0.5 * (b - a) * x
            m12sq = math.exp(u) + 4 * ml * ml
            m12 = math.sqrt(m12sq)
            for c, wc in zip(cs, cw):
                p, ps, eg = momenta(M, ml, m12, c)
                # dGamma = 1/(2M) |M|^2 p* |p_g|/(64 pi^3 M m12) dm12^2 dcos
                jac = 0.5 * (b - a) * w * math.exp(u) * wc
                meta.append((ip, jac * ps * eg / (64 * math.pi ** 3 * M * m12) / (2 * M)))
                pts.append([boost(v, BOOST) for v in p] if boosted else p)
    res = olc.run_points(pdir, pts, mu=M, warmup=1)
    lo = [0.0] * (len(edges) - 1)
    c_nlo = [0.0] * (len(edges) - 1)
    worst = 0.0
    codes = {}
    for (ip, wt), r in zip(meta, res):
        lo[ip] += wt * r['born']
        c_nlo[ip] += wt * r['born'] * r['finite_norm'] / (2 * math.pi)
        if r['born']:
            worst = max(worst, abs(r['single_norm']), abs(r['double_norm']))
        codes[r['ret_code']] = codes.get(r['ret_code'], 0) + 1
    out = {'state': state, 'lepton': lepton, 'M': M, 'ml': ml, 'nodes': nodes, 'ncos': ncos,
           'boost': BOOST if boosted else None,
           'max_pole_norm': worst, 'ret_codes': codes, 'cuts': {}}
    for k, cut in enumerate(CUTS):
        # panels up to the cut: 0 .. 1+k
        g_lo = sum(lo[:2 + k])
        g_c = sum(c_nlo[:2 + k])
        out['cuts'][str(cut)] = {'LO_GeV': g_lo, 'C_GeV': g_c, 'LO_eV': g_lo * 1e9,
                                 'C_eV': g_c * 1e9, 'K_coef_as_over_pi': math.pi * g_c / g_lo}
    out['cut_mode'] = CUT_MODE
    json.dump(out, open(os.path.join(HERE, 'hh_%s_%s_n%d%s%s.json' % (
        state, lepton, nodes, '' if boosted else '_rest',
        '_mhcut' if CUT_MODE == 'mH-m12' else '')), 'w'), indent=1)
    return out


if __name__ == '__main__':
    what, state, lepton = sys.argv[1:4]
    nodes = int(sys.argv[4]) if len(sys.argv) > 4 else 16
    if what == 'generate':
        print(generate(state, lepton))
    else:
        o = run(state, lepton, nodes, boosted=(what != 'run_rest'))
        print(json.dumps(o, indent=1))
