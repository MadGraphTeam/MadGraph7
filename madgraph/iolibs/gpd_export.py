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
"""Export of exclusive processes with generalised parton distributions (GPDs).

A GPD channel (base_objects.GPD_CHANNELS) is generated as two legs of the same
parton: the one leaving the probed proton with momentum k1 = (x+xi)P
('role': 'in', an incoming leg) and the one returning to it with
k2 = (x-xi)P ('role': 'out', an outgoing leg). In the matrix element

  * their wave functions are built on P (spinors u(P), ubar(P) or
    polarisations eps(P), eps*(P)), and only their momentum slots carry the
    flows: W%P is rescaled by (x+xi) and (x-xi) (patch_helas_calls);
  * the two helicities are summed coherently, lambda_in = lambda_out, without
    (vector: H, E) or with (axial: Ht, Et) the weight lambda;
  * the colour is projected on the singlet in the colour basis
    (color_amp.ColorBasis.GPDColorString), and normalised here: 1/N_c for
    quarks, 1/(N_c^2-1) for gluons; the spin normalisation is 1/2 for quarks
    (Tr[Gamma Pslash]/2) and 1/4 for gluons, whose light-cone gauge factor
    1/((x+xi-i0)(x-xi+i0)) is applied by the x integration
    (Source/GPD/gpd_xint.f).

The x integration needs the poles of the amplitude in x: a propagator carries
x iff exactly one of the two GPD legs is on one side of it (the analogue of
the loop line of an L-cut diagram). Its momentum is K + sigma x P with K a sum
of external momenta plus xi P, so its denominator is linear in x (P^2 = 0) and
vanishes at a = -(K^2-m^2)/(2 sigma P.K) (gpd_pole_lines).

The per-process Fortran (gpd_driver.f) provides the kernel at one x, the
poles, and the cross section SMATRIX_GPD; see formalism.typ of the report.
"""
import os
import re

import madgraph.core.base_objects as base_objects

pjoin = os.path.join

GPD_DECL = """
C     GPD pair: wave functions built on PGPD, momentum slots rescaled by
C     CGPD_IN = x+xi and CGPD_OUT = x-xi (madgraph/iolibs/gpd_export.py)
      REAL*8 PGPD(0:3), CGPD_IN, CGPD_OUT
      COMMON/GPD_KIN/PGPD, CGPD_IN, CGPD_OUT"""


def has_gpd(matrix_element):
    """True if the matrix element has a GPD pair."""
    procs = matrix_element.get('processes')
    return bool(procs) and any(leg.get('gpd') for leg in procs[0].get('legs'))


class GPDInfo(object):
    """Leg bookkeeping of a GPD matrix element.

    numbering:
      constituent  the external wave functions (1..nconst), onium constituents
                   and both GPD legs included (the legs of the diagrams);
      me           the legs of MATRIX/GET_AMP and of the NHEL rows: onia
                   collapsed (onia templates), GPD legs kept;
      phys         the legs of the GPD process: onia collapsed, the GPD 'in'
                   leg stands for the pair (momentum 2 xi P), the 'out' leg
                   dropped. This is what SMATRIX_GPD takes.
    """

    def __init__(self, matrix_element):
        self.me = matrix_element
        proc = matrix_element.get('processes')[0]
        self.model = proc.get('model')
        self.ext = matrix_element.get_external_wavefunctions()
        self.nonia = matrix_element.get_nonia()
        self.onia = self.nonia > 0
        gpd_wfs = [wf for wf in self.ext if wf.get('gpd')]
        if len(gpd_wfs) != 2:
            raise base_objects.PhysicsObject.PhysicsObjectError(
                "A GPD matrix element needs exactly two GPD legs")
        self.n_in = [wf.get('number_external') for wf in gpd_wfs
                     if wf.get('gpd').get('role') == 'in'][0]
        self.n_out = [wf.get('number_external') for wf in gpd_wfs
                      if wf.get('gpd').get('role') == 'out'][0]
        info = gpd_wfs[0].get('gpd')
        self.channel = info.get('name')
        self.parton = info.get('parton')
        self.is_gluon = (self.parton == 21)
        # constituent -> me numbering
        if self.onia:
            self.const_to_me = matrix_element.get_onia_leg_map()
        else:
            self.const_to_me = dict((wf.get('number_external'),
                                     wf.get('number_external')) for wf in self.ext)
        self.nme = max(self.const_to_me.values())
        self.me_in = self.const_to_me[self.n_in]
        self.me_out = self.const_to_me[self.n_out]
        # me -> phys numbering (the 'out' leg removed)
        self.me_to_phys = {}
        iphys = 0
        for ime in range(1, self.nme + 1):
            if ime == self.me_out:
                continue
            iphys += 1
            self.me_to_phys[ime] = iphys
        self.nphys = iphys
        self.ninitial = len([wf for wf in self.ext
                             if wf.get('leg_state') == False
                             and wf.get('number_external') != self.n_in]) + 1
        # onium constituents: their partner, for the momentum fractions
        self.onium_partner = {}
        if self.onia:
            for a, b in matrix_element.get_onia_pairs():
                self.onium_partner[a] = b
                self.onium_partner[b] = a

    # ------------------------------------------------------------ helicity --
    def helicity_map(self):
        """[(lambda=+1 IHEL, lambda=-1 IHEL)] for every helicity configuration
        of the non-GPD legs, in the order of the NHEL rows of matrix.f."""
        rows = list(self.me.get_helicity_matrix())
        index = {}
        for ihel, hel in enumerate(rows, start=1):
            if hel[self.me_in - 1] != hel[self.me_out - 1]:
                continue
            others = tuple(h for i, h in enumerate(hel)
                           if i + 1 not in (self.me_in, self.me_out))
            index.setdefault(others, {})[hel[self.me_in - 1]] = ihel
        out = []
        for others in sorted(index, key=lambda o: min(index[o].values())):
            pair = index[others]
            if 1 not in pair or -1 not in pair:
                raise base_objects.PhysicsObject.PhysicsObjectError(
                    "GPD legs without both helicities")
            out.append((pair[1], pair[-1]))
        return out

    # ------------------------------------------------------------- factors --
    def denominator(self):
        """Average over the initial non-GPD legs (spin x colour), the proton
        spin (2), and the identical-particle factor of the final state."""
        den = 2
        for wf in self.ext:
            if wf.get('leg_state') != False or wf.get('gpd'):
                continue
            part = self.model.get('particle_dict')[wf.get('pdg_code')]
            nhel = len(part.get_helicity_states())
            den *= nhel * abs(part.get('color'))
        proc = self.me.get('processes')[0]
        den *= proc.identical_particle_factor()
        return den

    def spin_colour_norm(self):
        """1/2 Tr[Gamma Pslash] x 1/N_c (quark), 1/4 x 1/(N_c^2-1) (gluon)."""
        if self.is_gluon:
            return "1D0/(4D0*8D0)"
        return "1D0/(2D0*3D0)"

    # ---------------------------------------------------------- momentum --
    def const_momentum_expr(self, j):
        """Fortran expression of the momentum (0:3) of constituent leg j,
        from the physical momenta PPHYS (onium constituents: their mass
        fraction of the bound state)."""
        ime = self.const_to_me[j]
        iphys = self.me_to_phys[ime]
        if j in self.onium_partner:
            mj = self.mass_expr(j)
            mk = self.mass_expr(self.onium_partner[j])
            return "(%s)/((%s)+(%s))*PPHYS(:,%d)" % (mj, mj, mk, iphys)
        return "PPHYS(:,%d)" % iphys

    def mass_expr(self, j):
        wf = [w for w in self.ext if w.get('number_external') == j][0]
        mass = self.model.get('particle_dict')[wf.get('pdg_code')].get('mass')
        return "0D0" if mass.lower() == 'zero' else "ABS(%s)" % mass

    # -------------------------------------------------------------- poles --
    def pole_list(self):
        """The x-dependent propagators: [(signed constituent legs, sigma,
        mass, width, name)], deduplicated."""

        def externals(wf):
            if not wf.get('mothers'):
                return frozenset([wf.get('number_external')])
            out = frozenset()
            for mother in wf.get('mothers'):
                out = out | externals(mother)
            return out

        seen = {}
        for diagram in self.me.get('diagrams'):
            for wf in diagram.get('wavefunctions'):
                if not wf.get('mothers'):
                    continue
                legs = externals(wf)
                has_in = self.n_in in legs
                has_out = self.n_out in legs
                if has_in == has_out:
                    continue
                part = wf.get('particle')
                mass = part.get('mass')
                width = part.get('width')
                key = (legs, mass.lower(), width.lower())
                if key in seen:
                    continue
                sigma = 1 if has_in else -1
                seen[key] = (sorted(legs), sigma, mass, width, part.get('name'))
        return list(seen.values())

    def leg_sign(self, j):
        """+1 for an incoming leg (momentum flows in), -1 outgoing."""
        wf = [w for w in self.ext if w.get('number_external') == j][0]
        return 1 if wf.get('leg_state') == False else -1

    def gpd_pole_lines(self):
        """Fortran lines of GPD_POLES: K_i and sigma_i of every x-carrying
        propagator, handed to GPD_POLE_ADD."""
        lines = []
        for legs, sigma, mass, width, name in self.pole_list():
            terms = []
            for j in legs:
                if j in (self.n_in, self.n_out):
                    continue
                sign = '+' if self.leg_sign(j) > 0 else '-'
                terms.append("%s%s" % (sign, self.const_momentum_expr(j)))
            terms.append("+XI*PGPD(:)")
            lines.append("C     propagator %s, legs %s" % (name, legs))
            lines.append("      K(:) = " + "".join(terms).lstrip('+'))
            m = "0D0" if mass.lower() == 'zero' else "ABS(%s)" % mass
            w = "0D0" if width.lower() == 'zero' else "ABS(%s)" % width
            lines.append("      CALL GPD_POLE_ADD(K, %dD0, %s, %s, PGPD, NP, AP, SP)"
                         % (sigma, m, w))
        return "\n".join(lines)

    # ---------------------------------------------------------- the kernel --
    def me_momenta_lines(self):
        """P(:,i) of the me legs at one x."""
        lines = []
        for ime in range(1, self.nme + 1):
            if ime == self.me_in:
                lines.append("      P(:,%d) = (X+XI)*PGPD(:)" % ime)
            elif ime == self.me_out:
                lines.append("      P(:,%d) = (X-XI)*PGPD(:)" % ime)
            else:
                lines.append("      P(:,%d) = PPHYS(:,%d)" % (ime, self.me_to_phys[ime]))
        return "\n".join(lines)


def patch_helas_calls(helas_calls, info):
    """The external wave functions of the GPD legs are built on PGPD and their
    momentum slots rescaled by x+xi (in) and x-xi (out)."""
    pattern = re.compile(r'^(\s*CALL\s+\w+\()P\(0,(\d+)\)(,.*,W\((\d+)\)\))\s*$')
    out = []
    found = set()
    for call in helas_calls:
        match = pattern.match(call)
        if match and int(match.group(2)) in (info.n_in, info.n_out):
            leg = int(match.group(2))
            out.append("%sPGPD(0)%s" % (match.group(1), match.group(3)))
            scale = 'CGPD_IN' if leg == info.n_in else 'CGPD_OUT'
            out.append("W(%s)%%P = W(%s)%%P*%s" % (match.group(4), match.group(4),
                                                  scale))
            found.add(leg)
        else:
            out.append(call)
    if found != set([info.n_in, info.n_out]):
        raise base_objects.PhysicsObject.PhysicsObjectError(
            "The external wave functions of the GPD legs were not found in "
            "the HELAS calls")
    return out


# ------------------------------------------------------------------------------
# the per-process driver
# ------------------------------------------------------------------------------
DRIVER_TEMPLATE = """C ======================================================================
C     GPD driver of %(process)s
C     (channel %(channel)s, parton %(parton)d), written by
C     madgraph/iolibs/gpd_export.py. Physical legs (SMATRIX_GPD): the GPD
C     pair is leg %(phys_in)d with momentum 2 xi P; NPHYS = %(nphys)d.
C ======================================================================
      SUBROUTINE %(prefix)sGPD_SET_POINT(PIN, XIIN, TIN, MUIN)
      IMPLICIT NONE
      INTEGER NPHYS
      PARAMETER (NPHYS=%(nphys)d)
      REAL*8 PIN(0:3,NPHYS), XIIN, TIN, MUIN
      REAL*8 PPHYS(0:3,NPHYS), XI, T, MU
      COMMON/%(prefix)sGPD_POINT/PPHYS, XI, T, MU
%(gpd_decl)s
      PPHYS = PIN
      XI = XIIN
      T = TIN
      MU = MUIN
      PGPD(:) = PPHYS(:,%(phys_in)d)/(2D0*XI)
      END

      SUBROUTINE %(prefix)sGPD_KERNEL(X, N, AV, AA)
C     The projected colour-flow amplitudes at X, for every helicity
C     configuration IH of the non-GPD legs: AV(IC + NCOLOR*(IH-1)) with the
C     vector, AA with the axial helicity weight of the GPD pair.
      USE ALOHA_OBJECT
      IMPLICIT NONE
      INTEGER NPHYS, NME, NCOLOR, NHPHYS, NCOMB, NGRAPHS
      PARAMETER (NPHYS=%(nphys)d, NME=%(nme)d, NCOLOR=%(ncolor)d)
      PARAMETER (NHPHYS=%(nhphys)d, NCOMB=%(ncomb)d, NGRAPHS=%(ngraphs)d)
      INTEGER N
      REAL*8 X
      COMPLEX*16 AV(N), AA(N)
      REAL*8 PPHYS(0:3,NPHYS), XI, T, MU
      COMMON/%(prefix)sGPD_POINT/PPHYS, XI, T, MU
%(gpd_decl)s
      INTEGER NHEL(NME,NCOMB)
      COMMON/%(prefix)sPROCESS_NHEL/NHEL
      INTEGER HELMAP(2,NHPHYS)
%(helmap_data)s
      REAL*8 P(0:3,NME), NORM
      INTEGER JC(NME), IH, L, I, IHEL
      COMPLEX*16 JAMP(NCOLOR), JL(NCOLOR,2)
%(kernel_decl)s
      NORM = %(norm)s
      JC = 1
      CGPD_IN = X + XI
      CGPD_OUT = X - XI
%(me_momenta)s
      DO IH = 1, NHPHYS
        DO L = 1, 2
          IHEL = HELMAP(L,IH)
%(jamp_call)s
          JL(:,L) = JAMP(:)
        ENDDO
        DO I = 1, NCOLOR
          AV(I+NCOLOR*(IH-1)) = (JL(I,1) + JL(I,2))*NORM
          AA(I+NCOLOR*(IH-1)) = (JL(I,1) - JL(I,2))*NORM
        ENDDO
      ENDDO
      END

      SUBROUTINE %(prefix)sGPD_POLES(NP, AP, SP)
C     The x-carrying propagators of every diagram: momentum K + sigma x P.
      IMPLICIT NONE
      INTEGER NPHYS
      PARAMETER (NPHYS=%(nphys)d)
      INTEGER NP, SP(*)
      COMPLEX*16 AP(*)
      REAL*8 PPHYS(0:3,NPHYS), XI, T, MU, K(0:3)
      COMMON/%(prefix)sGPD_POINT/PPHYS, XI, T, MU
%(gpd_decl)s
      INCLUDE 'coupl.inc'
      NP = 0
%(pole_lines)s
      END

      SUBROUTINE %(prefix)sSMATRIX_GPD(PIN, XIIN, TIN, MUIN, ANS)
C     |M|^2 of the GPD process, averaged over the initial spins (the proton
C     one included) and summed over the final ones, with the x integral of
C     the selected mode (gpd_x_integration: quadrature here, the Monte Carlo
C     replicas are set by GPD_SET_XMC), at the physical point PIN (NPHYS
C     legs, the pair being leg %(phys_in)d with momentum 2 xi P = zeta p),
C     skewness XIIN, momentum transfer TIN (GeV^2) and scale MUIN (GeV).
      IMPLICIT NONE
      INCLUDE 'gpd.inc'
      INTEGER NPHYS, NCOLOR, NHPHYS, NAMP, MAXP
      PARAMETER (NPHYS=%(nphys)d, NCOLOR=%(ncolor)d, NHPHYS=%(nhphys)d)
      PARAMETER (NAMP=NCOLOR*NHPHYS, MAXP=%(maxp)d)
      REAL*8 PIN(0:3,NPHYS), XIIN, TIN, MUIN, ANS
      EXTERNAL %(prefix)sGPD_KERNEL
      COMPLEX*16 AP(MAXP), C(NAMP,4), C2(NAMP,4)
      INTEGER SP(MAXP), NP, IERR
      COMPLEX*16 CLAST(NAMP,4)
      INTEGER IERRLAST
      COMMON/%(prefix)sGPD_LAST/CLAST, IERRLAST
      REAL*8 %(prefix)sGPD_SQUARE
      CALL GPD_INIT()
      CALL %(prefix)sGPD_SET_POINT(PIN, XIIN, TIN, MUIN)
      CALL %(prefix)sGPD_POLES(NP, AP, SP)
      CALL GPD_XSETUP(%(prefix)sGPD_KERNEL, NAMP, XIIN, TIN, MUIN, %(parton)d,
     &     NP, AP, SP, IERR)
      IERRLAST = IERR
      IF (IERR.EQ.1) THEN
        WRITE(*,*) 'GPD: %(process)s has a double pole at x = +-xi that does'
        WRITE(*,*) '     not cancel (end-point divergence): collinear'
        WRITE(*,*) '     factorisation fails at LO for this process. xi =', XIIN
        STOP 1
      ENDIF
      IF (USEMC.EQ.1) THEN
        CALL GPD_XMC(%(prefix)sGPD_KERNEL, RMC1, NAMP, C)
        CALL GPD_XMC(%(prefix)sGPD_KERNEL, RMC2, NAMP, C2)
      ELSE
        CALL GPD_XQUAD(%(prefix)sGPD_KERNEL, NAMP, C)
        C2 = C
      ENDIF
      CLAST = C
      ANS = %(prefix)sGPD_SQUARE(C, C2, XIIN, TIN)
      END

      REAL*8 FUNCTION %(prefix)sGPD_SQUARE(C, C2, XI, T)
C     Sum over helicities and colours of the proton-spin-summed bilinear
C     (formalism.typ, proton spin sums) of the CFFs C and C2 (two Monte
C     Carlo replicas, or the same quadrature), divided by the average.
      IMPLICIT NONE
      INTEGER NCOLOR, NHPHYS, NAMP
      PARAMETER (NCOLOR=%(ncolor)d, NHPHYS=%(nhphys)d, NAMP=NCOLOR*NHPHYS)
      COMPLEX*16 C(NAMP,4), C2(NAMP,4)
      REAL*8 XI, T, MP, TM, W1, W2, W3, W4, IDEN
      PARAMETER (MP=0.93827208816D0, IDEN=%(iden)d)
      INTEGER CF_INDEX, IH, I, J, OFF
      INTEGER %(prefix)sCF(NCOLOR*(NCOLOR+1)/2)
      INTEGER %(prefix)sDENOM
      COMMON/%(prefix)scolor_matrix/%(prefix)sCF, %(prefix)sDENOM
      COMPLEX*16 S, H1(NCOLOR), E1(NCOLOR), HT1(NCOLOR), ET1(NCOLOR)
      COMPLEX*16 H2(NCOLOR), E2(NCOLOR), HT2(NCOLOR), ET2(NCOLOR), Z
      CALL %(prefix)sINIT_CF()
      TM = T/(4D0*MP*MP)
      W1 = 1D0 - XI*XI
      W2 = -(XI*XI + TM)
      W3 = -XI*XI*TM
      W4 = -XI*XI
      S = (0D0,0D0)
      DO IH = 1, NHPHYS
        OFF = NCOLOR*(IH-1)
        H1 = C(OFF+1:OFF+NCOLOR,1)
        E1 = C(OFF+1:OFF+NCOLOR,2)
        HT1 = C(OFF+1:OFF+NCOLOR,3)
        ET1 = C(OFF+1:OFF+NCOLOR,4)
        H2 = C2(OFF+1:OFF+NCOLOR,1)
        E2 = C2(OFF+1:OFF+NCOLOR,2)
        HT2 = C2(OFF+1:OFF+NCOLOR,3)
        ET2 = C2(OFF+1:OFF+NCOLOR,4)
        CF_INDEX = 0
        DO I = 1, NCOLOR
          DO J = I, NCOLOR
            CF_INDEX = CF_INDEX + 1
            Z = W1*(H1(J)*DCONJG(H2(I)) + HT1(J)*DCONJG(HT2(I)))
     &        + W2*E1(J)*DCONJG(E2(I)) + W3*ET1(J)*DCONJG(ET2(I))
     &        + W4*(H1(J)*DCONJG(E2(I)) + E1(J)*DCONJG(H2(I))
     &            + HT1(J)*DCONJG(ET2(I)) + ET1(J)*DCONJG(HT2(I)))
            IF (I.NE.J) THEN
              Z = Z + W1*(H1(I)*DCONJG(H2(J)) + HT1(I)*DCONJG(HT2(J)))
     &          + W2*E1(I)*DCONJG(E2(J)) + W3*ET1(I)*DCONJG(ET2(J))
     &          + W4*(H1(I)*DCONJG(E2(J)) + E1(I)*DCONJG(H2(J))
     &              + HT1(I)*DCONJG(ET2(J)) + ET1(I)*DCONJG(HT2(J)))
              Z = Z/2D0
            ENDIF
            S = S + %(prefix)sCF(CF_INDEX)*Z
          ENDDO
        ENDDO
      ENDDO
      %(prefix)sGPD_SQUARE = 2D0*DBLE(S)/%(prefix)sDENOM/IDEN
      END
"""

JAMP_CALL_PLAIN = """          CALL %(prefix)sGET_AMP(P, NHEL(1,IHEL), JC, 1, AMP)
          CALL %(prefix)sGET_JAMP(AMP, JAMP)"""

JAMP_DECL_PLAIN = """      COMPLEX*16 AMP(%(namp_dim)s)"""

JAMP_CALL_ONIA = """          CALL %(prefix)sONIA_NEW_POINT()
          TMPM = %(prefix)sMATRIX(P, NHEL(1,IHEL), JC, 1)
          JAMP(:) = LAST_JAMP(:)*SQRT(LAST_LDME)"""

JAMP_DECL_ONIA = """      COMPLEX*16 LAST_JAMP(NCOLOR)
      REAL*8 LAST_LDME, TMPM, %(prefix)sMATRIX
      COMMON/%(prefix)sONIA_LAST_JAMP/LAST_JAMP, LAST_LDME"""


def write_driver(path, matrix_element, replace_dict, proc_prefix=''):
    """Write gpd_driver.f of the matrix element in the directory path."""
    info = GPDInfo(matrix_element)
    helmap = info.helicity_map()
    lines = []
    for ih, (lp, lm) in enumerate(helmap, start=1):
        lines.append("      DATA HELMAP(1,%d), HELMAP(2,%d) /%d,%d/" % (ih, ih, lp, lm))
    poles = info.pole_list()
    fmt = {'prefix': proc_prefix,
           'process': matrix_element.get('processes')[0].nice_string(prefix=False).replace("'", ""),
           'channel': info.channel, 'parton': info.parton,
           'nphys': info.nphys, 'nme': info.nme,
           'phys_in': info.me_to_phys[info.me_in],
           'ncolor': replace_dict['ncolor'], 'nhphys': len(helmap),
           'ncomb': replace_dict['ncomb'], 'ngraphs': replace_dict['ngraphs'],
           'namp_dim': replace_dict.get('namp_dim', 'NGRAPHS'),
           'helmap_data': "\n".join(lines),
           'gpd_decl': GPD_DECL, 'norm': info.spin_colour_norm(),
           'me_momenta': info.me_momenta_lines(),
           'pole_lines': info.gpd_pole_lines(),
           'maxp': max(1, len(poles)) + 2,
           'iden': info.denominator()}
    if info.onia:
        fmt['jamp_call'] = JAMP_CALL_ONIA % fmt
        fmt['kernel_decl'] = JAMP_DECL_ONIA % fmt
    else:
        fmt['jamp_call'] = JAMP_CALL_PLAIN % fmt
        fmt['kernel_decl'] = JAMP_DECL_PLAIN % fmt
    text = DRIVER_TEMPLATE % fmt
    with open(pjoin(path, 'gpd_driver.f'), 'w') as fsock:
        fsock.write(text)
    return info


CHECK_TEMPLATE = """      PROGRAM CHECK_GPD
C     Reads points from gpd_points.dat: per point a line 'xi t mu' and then
C     NPHYS lines 'E px py pz' (the pair as leg %(phys_in)d with 2 xi P);
C     writes |M|^2 and the CFFs of every helicity and colour flow.
      IMPLICIT NONE
      INTEGER NPHYS, NAMP
      PARAMETER (NPHYS=%(nphys)d, NAMP=%(namp)d)
      REAL*8 P(0:3,NPHYS), XI, T, MU, ANS
      COMPLEX*16 CLAST(NAMP,4)
      INTEGER IERRLAST, I, K, IOS
      COMMON/%(prefix)sGPD_LAST/CLAST, IERRLAST
      CALL SETPARA('param_card.dat')
      CALL SETONIA_IF_NEEDED()
      OPEN(11, FILE='gpd_points.dat', STATUS='OLD')
      DO
        READ(11, *, IOSTAT=IOS) XI, T, MU
        IF (IOS.NE.0) EXIT
        DO I = 1, NPHYS
          READ(11, *) P(0,I), P(1,I), P(2,I), P(3,I)
        ENDDO
        CALL %(prefix)sSMATRIX_GPD(P, XI, T, MU, ANS)
        WRITE(*,'(A,ES26.16E3)') 'ME2 ', ANS
        DO I = 1, NAMP
          WRITE(*,'(A,I4,8ES26.16E3)') 'CFF ', I, (CLAST(I,K), K=1,4)
        ENDDO
      ENDDO
      END
"""


def write_check(path, info, replace_dict, proc_prefix=''):
    nhphys = len(info.helicity_map())
    text = CHECK_TEMPLATE % {'prefix': proc_prefix, 'nphys': info.nphys,
                             'phys_in': info.me_to_phys[info.me_in],
                             'namp': replace_dict['ncolor'] * nhphys}
    if info.onia:
        text += """
      SUBROUTINE SETONIA_IF_NEEDED()
      CALL SETONIA('onia_card.dat')
      END
"""
    else:
        text += """
      SUBROUTINE SETONIA_IF_NEEDED()
      END
"""
    with open(pjoin(path, 'check_gpd.f'), 'w') as fsock:
        fsock.write(text)


GPD_CARD = """#*********************************************************************
#   GPD settings of a standalone output (madgraph/iolibs/gpd_export.py)
#   format: value = name    (as the run card)
#*********************************************************************
  toy      = gpd_model           ! toy | grid (dd needs the PDFs of madevent)
  ''       = gpd_grid            ! table of the grid model
  0        = gpd_tmodel          ! 0 none, 1 exp(B t), 2 Regge exp(B t)|x|^(-alpha' t)
  0.0      = gpd_tslope_g        ! B of gluons (GeV^-2), amplitude level
  0.0      = gpd_tslope_q        ! B of quarks (GeV^-2), amplitude level
  0.0      = gpd_alphap_g        ! alpha' of gluons (GeV^-2)
  0.0      = gpd_alphap_q        ! alpha' of quarks (GeV^-2)
  quadrature = gpd_x_integration ! quadrature | mc
  1e-8     = gpd_xtol            ! relative tolerance of the x quadrature
  0.05     = gpd_epsres          ! residue step (fraction of the pole spacing)
  1e-3     = gpd_dptol           ! tolerance of the double-pole detector
"""


def makefile_lines(base_text, proc_prefix=''):
    """The subprocess makefile of a GPD matrix element: gpd_driver.o in the
    process, libgpd linked, and the check_gpd program."""
    text = base_text.replace("PROCESS=  matrix.o", "PROCESS=  matrix.o gpd_driver.o")
    text = text.replace("LINKLIBS =  -L$(LIBDIR) -ldhelas -lmodel",
                        "LINKLIBS =  -L$(LIBDIR) -lgpd -ldhelas -lmodel")
    text = text.replace("F_INCLUDE = -I$(ROOT_DIR)/Source/DHELAS",
                        "F_INCLUDE = -I$(ROOT_DIR)/Source/GPD -I$(ROOT_DIR)/Source/DHELAS")
    text += """
check_gpd: $(LIBS) $(PROCESS) check_gpd.o makefile $(LIBDIR)/libgpd.$(libext)
	$(FC) $(FFLAGS) -o check_gpd $(PROCESS) check_gpd.o $(LINKLIBS)

$(LIBDIR)/libgpd.$(libext):
	$(MAKE) -C "$(LIBDIR)/../Source/GPD" standalone
"""
    return text
