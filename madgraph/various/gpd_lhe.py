################################################################################
#
# Copyright (c) 2026 The MadGraph7 Development team and Contributors
#
# This file is a part of the MadGraph7 project, an application which
# automatically generates Feynman diagrams and matrix elements for arbitrary
# high-energy processes in the Standard Model and beyond.
#
# It is subject to the MadGraph7 license which should accompany this
# distribution.
#
################################################################################
"""The event record of GPD processes (madgraph/iolibs/gpd_madevent.py).

madevent generates gamma + pair -> X, with the GPD pair a massless incoming
leg (pdg 23) of momentum zeta P_beam and the hard kinematics at t = 0; the t
of the elastic proton, drawn with the matrix element, is written per GPD beam
in a '#gpd t1 t2' line of each event. Here the pair is replaced by the beam
proton p and the outgoing proton p' is added, with exact kinematics:
gamma(q) + p -> X + p', at the W^2 = (q + p)^2 and the invariant mass of X of
the event, the t of the event (moved into the physical range when the
massless t_min of the matrix element lies outside it) and a uniform azimuth;
X is boosted as a whole (its internal configuration is kept), and momentum is
conserved exactly. Only the standard library is used (it runs from
bin/internal of a madevent output).
"""
import gzip
import math
import os
import random

MPROTON = 0.93827208816
PAIR_PDG = 23


def _dot(a, b):
    return a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3]


def _boost(p, b):
    """p boosted by the velocity b (3-vector)."""
    b2 = b[0] ** 2 + b[1] ** 2 + b[2] ** 2
    if b2 <= 0:
        return list(p)
    g = 1 / math.sqrt(1 - b2)
    bp = b[0] * p[1] + b[1] * p[2] + b[2] * p[3]
    f = (g - 1) * bp / b2 + g * p[0]
    return [g * (p[0] + bp), p[1] + f * b[0], p[2] + f * b[1], p[3] + f * b[2]]


def _velocity(p):
    return [p[1] / p[0], p[2] / p[0], p[3] / p[0]]


def _rotate_to(v, n):
    """v given in a frame whose z axis is the unit vector n, in the lab."""
    nx, ny, nz = n
    if nz < -1 + 1e-15:
        return [-v[0], v[1], -v[2]]
    # rotation taking z to n (Rodrigues about z x n)
    c = nz
    k = 1 / (1 + c)
    rx = [1 - k * nx * nx, -k * nx * ny, nx]
    ry = [-k * nx * ny, 1 - k * ny * ny, ny]
    rz = [-nx, -ny, c]
    return [rx[0] * v[0] + rx[1] * v[1] + rx[2] * v[2],
            ry[0] * v[0] + ry[1] * v[1] + ry[2] * v[2],
            rz[0] * v[0] + rz[1] * v[1] + rz[2] * v[2]]


def recoil(q, pin, kx, t, phi, mp=MPROTON, mout=None):
    """gamma(q) + p(pin) -> X(kx') + p'(pout) at the mass of kx, with
    (pin - pout)^2 = t (moved into the physical range) and azimuth phi;
    pout of mass mout (default mp; the mass of pin is that of its momentum).
    Returns (kx', pout, t used)."""
    tot = [q[i] + pin[i] for i in range(4)]
    W2 = _dot(tot, tot)
    W = math.sqrt(W2)
    MX2 = max(_dot(kx, kx), 0.0)
    min2 = _dot(pin, pin)
    mp = mp if mout is None else mout
    # madevent keeps W >= M_X + m_p (massless beams): at the threshold the
    # massive beam can leave lam slightly negative
    lam = max((W2 - MX2 - mp * mp) ** 2 - 4 * MX2 * mp * mp, 0.0)
    bcm = _velocity(tot)
    pcm = _boost(pin, [-x for x in bcm])
    ppin = math.sqrt(pcm[1] ** 2 + pcm[2] ** 2 + pcm[3] ** 2)
    n = [pcm[1] / ppin, pcm[2] / ppin, pcm[3] / ppin]
    epin = pcm[0]
    pout_abs = math.sqrt(lam) / (2 * W)
    eout = (W2 + mp * mp - MX2) / (2 * W)
    cth = (t - min2 - mp * mp + 2 * epin * eout) / max(2 * ppin * pout_abs, 1e-300)
    cth = max(-1.0, min(1.0, cth))
    tused = min2 + mp * mp - 2 * (epin * eout - ppin * pout_abs * cth)
    sth = math.sqrt(max(0.0, 1 - cth * cth))
    v = _rotate_to([pout_abs * sth * math.cos(phi), pout_abs * sth * math.sin(phi), pout_abs * cth], n)
    pout = _boost([eout] + v, bcm)
    kx_new = [tot[i] - pout[i] for i in range(4)]
    return kx_new, pout, tused


def _fmt(pdg, st, m1, m2, c1, c2, p, m, vt, sp):
    return " %8d %2d %4d %4d %4d %4d %+.10e %+.10e %+.10e %.10e %.10e %.4e %.4e\n" % (
        pdg, st, m1, m2, c1, c2, p[1], p[2], p[3], p[0], m, vt, sp)


MNEUTRON = 0.93956542052


def ion_pdg(nbp, nbn):
    """PDG code of the nucleus (Z = nbp, A = nbp + nbn): 10LZZZAAAI."""
    if (nbp, nbn) == (1, 0):
        return 2212
    if (nbp, nbn) == (0, 1):
        return 2112
    return 1000000000 + 10000 * nbp + 10 * (nbp + nbn)


def rewrite_event(lines, ebeam, seed_rng, mp=MPROTON, beams=None, incoherent=False):
    """The lines of one event (between <event> and </event>) rewritten;
    returns (lines, t used, moved). beams: per beam a dict (pdg, mass, A, Z)
    of the proton or nucleus (default: protons). A nucleus probed coherently
    recoils as a whole (mass M_A); incoherently (incoherent=True), a nucleon
    of momentum p_A/A (p with probability Z/A, else n) recoils and the
    remnant (A - 1 nucleons, 4-momentum (A - 1)/A p_A) is written too, so
    that momentum is conserved exactly."""
    head = lines[0].split()
    nup = int(head[0])
    parts = [l.split() for l in lines[1:1 + nup]]
    rest = lines[1 + nup:]
    tvals = None
    keep = []
    for l in rest:
        if l.startswith('#gpd'):
            tvals = [float(x) for x in l.split()[1:3]]
        else:
            keep.append(l)
    ipair = [i for i, w in enumerate(parts) if int(w[0]) == PAIR_PDG and int(w[1]) == -1]
    if not ipair or tvals is None:
        return lines, None, False
    ipair = ipair[0]
    iq = [i for i, w in enumerate(parts) if int(w[1]) == -1 and i != ipair][0]
    mom = lambda w: [float(w[9]), float(w[6]), float(w[7]), float(w[8])]
    pair = mom(parts[ipair])
    q = mom(parts[iq])
    beam = 1 if pair[3] > 0 else 2
    eb = ebeam[beam - 1]
    info = (beams or [None, None])[beam - 1] or dict(pdg=2212, mass=mp, A=1, Z=1)
    mbeam = info['mass']
    pz = math.sqrt(eb * eb - mbeam * mbeam)
    pbeam = [eb, 0.0, 0.0, pz if beam == 1 else -pz]
    t = tvals[beam - 1]
    fin = [i for i, w in enumerate(parts) if int(w[1]) == 1]
    kx = [sum(mom(parts[i])[k] for i in fin) for k in range(4)]
    phi = 2 * math.pi * seed_rng.random()
    A, Z = info['A'], info['Z']
    incoh = incoherent and A > 1
    if incoh:
        # the struck nucleon carries p_A/A
        proton = seed_rng.random() < Z / A
        pdg_out, m_out = (2212, MPROTON) if proton else (2112, MNEUTRON)
        pin = [x / A for x in pbeam]
        remnant = [pbeam[k] - pin[k] for k in range(4)]
        pdg_rem = ion_pdg(Z - (1 if proton else 0), A - Z - (0 if proton else 1))
        kx_new, pout, tused = recoil(q, pin, kx, t, phi, mout=m_out)
    else:
        pdg_out, m_out = info['pdg'], mbeam
        pin = pbeam
        kx_new, pout, tused = recoil(q, pin, kx, t, phi, mp=mbeam)
    # X boosted as a whole: to its rest frame, then to kx_new
    b_old = _velocity(kx)
    b_new = _velocity(kx_new)
    out = []
    zeta = pair[0] / eb
    if incoh:
        zeta *= A
    xi = zeta / (2 - zeta)
    for i, w in enumerate(parts):
        p = mom(w)
        if i == ipair:
            out.append(_fmt(info['pdg'], -1, 0, 0, 0, 0, pbeam, mbeam, 0.0, 9.0))
            continue
        if int(w[1]) in (1, 2):
            p = _boost(_boost(p, [-x for x in b_old]), b_new)
            out.append(_fmt(int(w[0]), int(w[1]), int(w[2]), int(w[3]), int(w[4]), int(w[5]), p,
                            float(w[10]), float(w[11]), float(w[12])))
        else:
            out.append(lines[1 + i] if lines[1 + i].endswith('\n') else lines[1 + i] + '\n')
    out.append(_fmt(pdg_out, 1, 1, 2, 0, 0, pout, m_out, 0.0, 9.0))
    nadd = 1
    if incoh:
        out.append(_fmt(pdg_rem, 1, 1, 2, 0, 0, remnant, math.sqrt(max(_dot(remnant, remnant), 0.0)),
                        0.0, 9.0))
        nadd = 2
    head[0] = str(nup + nadd)
    newhead = " " + " ".join(head) + "\n"
    info = "#gpd t= %.10e xi= %.10e beam= %d" % (tused, xi, beam)
    if A > 1:
        info += " A= %d nucl= %s" % (A, 'incoherent' if incoh else 'coherent')
    info += "\n"
    moved = abs(tused - t) > 1e-9 * max(1.0, abs(t))
    return [newhead] + out + [info] + [l if l.endswith('\n') else l + '\n' for l in keep], tused, moved


def add_protons(path, ebeam, seed=0, logger=None, beams=None, incoherent=False):
    """Rewrite the event file path (.lhe or .lhe.gz) in place (beams,
    incoherent: see rewrite_event)."""
    op = gzip.open if path.endswith('.gz') else open
    tmp = path + '.gpdtmp' + ('.gz' if path.endswith('.gz') else '')
    rng = random.Random(seed)
    nev = nmoved = 0
    with op(path, 'rt') as fin, op(tmp, 'wt') as fout:
        buf = None
        for line in fin:
            if buf is None:
                if line.lstrip().startswith('<event'):
                    buf = []
                    fout.write(line)
                else:
                    fout.write(line)
                continue
            if line.lstrip().startswith('</event'):
                new, tused, moved = rewrite_event(buf, ebeam, rng, beams=beams, incoherent=incoherent)
                if tused is not None:
                    nev += 1
                    nmoved += moved
                fout.writelines(new)
                fout.write(line)
                buf = None
                continue
            buf.append(line)
    os.replace(tmp, path)
    if logger:
        logger.info("GPD: elastic %s added to %d events of %s (%d with t moved into the "
                    "physical range)" % ('protons' if not beams or all(b['A'] == 1 for b in beams)
                                         else 'protons and nuclei', nev, os.path.basename(path), nmoved))
    return nev, nmoved
