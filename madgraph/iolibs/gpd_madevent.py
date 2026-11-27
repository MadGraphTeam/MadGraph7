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
"""GPD processes in 'output madevent'.

The physical process is gamma + pair(2 xi P) -> X: the GPD pair is one
massless incoming parton with light-cone fraction x2 = zeta = 2xi/(1+xi) of the
proton, the parton returning to the proton is not a particle of the event.
madevent's subprocess directory (phase space, channels, event record,
symmetries, mirror) is written for a *shadow* process with the same physical
legs, the pair replaced by a Z (pdg 23, its mass set to zero in pmass.inc):
'a z > X'. All the GPD channels (gpd_g, gpd_u, ...) with the same X share one
shadow and are summed coherently in amplitude (eq. 2.8 of the notes).

In that directory:
  * matrix<n>_gpdc<k>.f   the constituent matrix element of channel k (the
                          standalone template, prefix G<n>C<k>_) and its GPD
                          driver (kernel, poles, CFFs), self-contained;
  * matrix<n>_gpd.f       GPD<n>_ME: the CFFs of every channel summed, then
                          squared with the proton spin sums;
  * matrix<n>.f           the shadow's, with MATRIX<n> replaced: on the first
                          helicity slot it returns the GPD |M|^2 at
                          xi = zeta/(2-zeta) (zeta = xbk of the GPD beam), with
                          t drawn on [t_min, t_min - gpd_t_range] (map
                          exp(-gpd_t_bmap |t|)) and, for gpd_x_integration =
                          mc, the two x replicas drawn (unbiased estimate of
                          |int dx|^2), times the shadow's average IDEN.
The beam of the GPD (lpp = 5) gives zeta/(16 pi^2) for the pair (pdg 23) and
the gamma-UPC single-photon flux for a photon (Source/PDF/pdg2pdf.f).
"""
import collections
import os
import re
import shutil

import madgraph
import madgraph.core.base_objects as base_objects
import madgraph.core.diagram_generation as diagram_generation
import madgraph.core.helas_objects as helas_objects
import madgraph.iolibs.file_writers as writers
import madgraph.iolibs.gpd_export as gpd_export

pjoin = os.path.join
SHADOW_PDG = 23

# Cards/gpd_card.dat of a madevent output: the standalone card with the
# double-distribution model and the two-replica Monte Carlo x integration as
# defaults (the x integral then rides along with the phase space; the
# per-point quadrature of the dd model costs a DD integral at every x node)
GPD_CARD_MADEVENT = gpd_export.GPD_CARD.replace(
    "GPD settings of a standalone output (madgraph/iolibs/gpd_export.py)",
    "GPD settings of a madevent output (madgraph/iolibs/gpd_madevent.py)").replace(
    "toy      = gpd_model           ! toy | toy2 | toy3 | grid (dd needs the PDFs of madevent)",
    "dd       = gpd_model           ! dd | toy | toy2 | toy3 | grid").replace(
    "quadrature = gpd_x_integration ! quadrature | mc",
    "mc       = gpd_x_integration   ! mc (two Monte Carlo replicas in x) | quadrature (per point)")

# shadow process string -> the GPD channel amplitudes it stands for
SHADOWS = {}


def is_gpd_amplitude(amp):
    return any(leg.get('gpd') for leg in amp.get('process').get('legs'))


def physical_key(amp):
    proc = amp.get('process')
    legs = proc.get('legs')
    ini = []
    for leg in legs:
        if leg.get('state'):
            continue
        ini.append('pair' if leg.get('gpd') else leg.get('id'))
    fin = []
    for leg in legs:
        if not leg.get('state') or leg.get('gpd'):
            continue
        onium = leg.get('onium')
        fin.append((leg.get('id'), onium.get('id') if onium else 0,
                    onium.get('index') if onium else -1))
    return (tuple(ini), tuple(fin), proc.get('id'))


def make_shadow(channels, model):
    """The shadow amplitude of the GPD channels of one physical process."""
    proc0 = channels[0].get('process')
    legs = base_objects.LegList()
    for leg in proc0.get('legs'):
        if leg.get('gpd') and leg.get('gpd').get('role') == 'out':
            continue
        if leg.get('gpd'):
            new = base_objects.Leg({'id': SHADOW_PDG, 'state': False})
        else:
            new = base_objects.Leg({'id': leg.get('id'), 'state': leg.get('state'),
                                    'polarization': list(leg.get('polarization')),
                                    'onium': dict(leg.get('onium'))})
        legs.append(new)
    for i, leg in enumerate(legs, start=1):
        leg.set('number', i)
    process = base_objects.Process({'legs': legs, 'model': model,
                                    'id': proc0.get('id'), 'orders': {}})
    shadow = diagram_generation.Amplitude(process)
    if not shadow.get('diagrams'):
        raise madgraph.MadGraph5Error(
            "GPD: no shadow diagrams for %s; the madevent output of this GPD "
            "process is not supported" % process.nice_string())
    # both beams may radiate the photon (lpp1 = lpp2 = 5): madevent's mirror
    # orientation puts the pair on beam 1; MATRIX<n> keeps an orientation only
    # when its GPD beam has lpp = 5
    if len([l for l in legs if not l.get('state')]) == 2:
        shadow.set('has_mirror_process', True)
    SHADOWS[process.nice_string()] = list(channels)
    return shadow


def make_shadows(amps, model):
    """The amplitudes for madevent: the non-GPD ones unchanged, one shadow per
    physical GPD process."""
    groups = collections.OrderedDict()
    out = diagram_generation.AmplitudeList()
    for amp in amps:
        if is_gpd_amplitude(amp):
            groups.setdefault(physical_key(amp), []).append(amp)
        else:
            out.append(amp)
    for key, channels in groups.items():
        out.append(make_shadow(channels, model))
    return out


def flatten(items):
    out = []
    for item in items:
        if isinstance(item, (list, tuple, set)):
            out += flatten(item)
        else:
            out.append(item)
    return out


def shadow_channels(matrix_element):
    """The GPD channels of a shadow matrix element, or None."""
    for proc in matrix_element.get('processes'):
        chans = SHADOWS.get(proc.nice_string())
        if chans:
            return chans
    return None


def is_shadow_group(matrix_elements):
    return any(shadow_channels(me) for me in matrix_elements)


# ------------------------------------------------------------------------------
INCLUDE_RE = re.compile(r"^\s*INCLUDE\s+'([^']+)'\s*$", re.IGNORECASE | re.MULTILINE)


def inline_includes(text, files):
    """Replace INCLUDE 'name' by the content of files[name] (a path); the
    madevent coupl.inc needs vector.inc before it."""
    def repl(match):
        name = match.group(1)
        if name in files:
            return open(files[name]).read()
        if name.lower() == 'coupl.inc':
            return "      INCLUDE '../../Source/vector.inc'\n" + match.group(0)
        return match.group(0)
    text = INCLUDE_RE.sub(repl, text)
    # the madevent couplings are indexed by the event of the vector (IVEC)
    return text.replace("      COMMON/GPD_KIN/PGPD, CGPD_IN, CGPD_OUT",
                        "      COMMON/GPD_KIN/PGPD, CGPD_IN, CGPD_OUT\n"
                        "      INTEGER IVEC\n      COMMON/GPD_IVEC/IVEC")


def madevent_couplings(text, vector_couplings):
    """The standalone helas calls in a madevent directory: the couplings
    that madevent stores per event of the vector are indexed by IVEC, and
    the fake widths of madevent's t-channel propagators are the true ones."""
    text = re.sub(r',\s*FK_', ',', text, flags=re.IGNORECASE)
    for name in vector_couplings:
        text = re.sub(r'\b%s\b(?!\s*\()' % re.escape(name), '%s(IVEC)' % name, text,
                      flags=re.IGNORECASE)
    return text


COHERENT_TEMPLATE = """C ======================================================================
C     GPD process %(process)s in madevent: the CFFs of the GPD channels
C     %(channels)s, summed coherently (one amplitude per helicity and colour
C     flow), then squared with the proton spin sums
C     (madgraph/iolibs/gpd_madevent.py).
C ======================================================================
      SUBROUTINE GPD%(n)d_ME(P, XI, T, MU, ANS)
      IMPLICIT NONE
      INTEGER NPHYS, NAMP
      PARAMETER (NPHYS=%(nphys)d, NAMP=%(namp)d)
      REAL*8 P(0:3,NPHYS), XI, T, MU, ANS
      COMPLEX*16 C(NAMP,4), C2(NAMP,4), CS(NAMP,4), CS2(NAMP,4)
      REAL*8 %(prefix1)sGPD_SQUARE
      CS = (0D0,0D0)
      CS2 = (0D0,0D0)
%(channel_calls)s
      ANS = %(prefix1)sGPD_SQUARE(CS, CS2, XI, T)
      END

      REAL*8 FUNCTION MATRIX%(n)d(P,NHEL,IFLAV, IHEL,AMP2, JAMP2, IVEC)
C     The shadow's MATRIX%(n)d replaced: on the first helicity slot, the GPD
C     |M|^2 (its helicity sums are inside) with t and, in Monte Carlo mode,
C     the two x replicas drawn here; times the shadow's average %(iden)d,
C     which SMATRIX%(n)d divides out. The other slots are zero.
      IMPLICIT NONE
      INCLUDE '../../Source/vector.inc'
      INCLUDE 'genps.inc'
      INCLUDE 'nexternal.inc'
      INCLUDE 'maxamps.inc'
      INCLUDE 'run.inc'
%(gpd_inc)s
      REAL*8 P(0:3,NEXTERNAL)
      INTEGER IFLAV, NHEL(NEXTERNAL), IHEL, IVEC, I
      DOUBLE PRECISION AMP2(MAXAMPS), JAMP2(0:MAXFLOW)
      INTEGER SUBDIAG(MAXSPROC), IB(2)
      COMMON/TO_SUB_DIAG/SUBDIAG, IB
      INTEGER GIVEC
      COMMON/GPD_IVEC/GIVEC
C     t of the elastic proton per GPD beam, for the event record (unwgt.f
C     writes it as a '#gpd' line, madgraph/various/gpd_lhe.py adds the protons)
      DOUBLE PRECISION GPD_EVT_T(2, VECSIZE_MEMMAX)
      INTEGER GPD_EVT_SET
      COMMON/GPD_EVENT/GPD_EVT_T, GPD_EVT_SET
      DOUBLE PRECISION ZETA, XI, T, MU, TMIN, DELTA, U, R, WT, ANS, DSUM(0:3)
      DOUBLE PRECISION SHAT, MT, ZT, XIK, BMAP, FA, ANSN, R2, PEXP
      DOUBLE PRECISION GPD_FORMFACTOR, GPD_WS_R2, GPD_PAIR_NORM
      EXTERNAL GPD_PAIR_NORM
      INTEGER NA, NZ
      DOUBLE PRECISION PMASS(NEXTERNAL), ZERO
      LOGICAL OFFSH
      PARAMETER (ZERO=0D0)
      INCLUDE 'coupl.inc'
      INCLUDE 'ldme.inc'
      DOUBLE PRECISION MPROT
      PARAMETER (MPROT=0.93827208816D0)
      MATRIX%(n)d = 0D0
      IF (IHEL.NE.1) RETURN
C     the pair comes from the beam IB(2) of this orientation, a GPD beam
      IF (ABS(LPP(IB(2))).NE.5) RETURN
C     with a fixed beam (lpp = 0), madevent keeps the sampled momentum
C     fractions with the legs of the process, not with the beams: in the
C     mirror orientation the pair would get the fraction 1 of the fixed
C     beam. The fixed beam must carry the first leg, i.e. be beam 1.
      IF (IB(1).EQ.2 .AND. LPP(2).EQ.0) THEN
        WRITE(*,*) 'GPD: a fixed beam (lpp = 0) must be beam 1, with ',
     &    'the GPD target on beam 2 (e.g. lpp1 = 0, lpp2 = 5 for ',
     &    'e- gpd > e- X)'
        STOP 1
      ENDIF
C     multichannel and colour weights: one channel per shadow diagram, flat
      DO I = 1, %(ngraphs)d
        AMP2(I) = AMP2(I) + 1D0
      ENDDO
      DO I = 1, %(ncolor_flow)d
        JAMP2(I) = JAMP2(I) + 1D0
      ENDDO
C     the hard kinematics must conserve momentum with the final legs on
C     shell; madevent's 2 -> 1 phase space with one fixed beam does not (the
C     pair fraction is not fixed by the delta function there): refused
      DO I = 0, 3
        DSUM(I) = P(I,1) + P(I,2)
      ENDDO
      DO I = NINCOMING+1, NEXTERNAL
        DSUM(:) = DSUM(:) - P(:,I)
      ENDDO
      INCLUDE 'pmass.inc'
      OFFSH = MAXVAL(ABS(DSUM)).GT.1D-8*MAX(P(0,1)+P(0,2),1D0)
      DO I = NINCOMING+1, NEXTERNAL
        IF (ABS(P(0,I)**2-P(1,I)**2-P(2,I)**2-P(3,I)**2-PMASS(I)**2).GT.
     &      1D-6*MAX(P(0,I)**2,1D0)) OFFSH = .TRUE.
      ENDDO
      IF (OFFSH) THEN
        WRITE(*,*) 'GPD: a phase-space point off the mass shell or ',
     &    'not conserving momentum; for 2 -> 1 GPD processes use ',
     &    'lpp1 = lpp2 = 5 (one fixed beam is not supported)'
        STOP 1
      ENDIF
      GIVEC = IVEC
      CALL GPD_INIT()
      ZETA = XBK(IB(2))
      SHAT = (P(0,1)+P(0,2))**2 - (P(1,1)+P(1,2))**2
     &     - (P(2,1)+P(2,2))**2 - (P(3,1)+P(3,2))**2
      MU = SQRT(Q2FACT(IB(2)))
      GPD_BEAM = IB(2)
C     the target of the GPDs (formalism.typ, Heavy ions): a proton; a
C     nucleus probed coherently, with the variables of the nucleus (zeta =
C     the fraction of the ion, xi_A, M_A) and the nucleon kernels at
C     xi_N = A xi_A; or a nucleon of the nucleus (incoherent), with the
C     variables of a nucleon (zeta_N = A zeta, m_N)
      NA = NB_PROTON(IB(2)) + NB_NEUTRON(IB(2))
      NZ = NB_PROTON(IB(2))
      IF (NA.LE.1) THEN
        GPD_NUCL = 0
        CALL GPD_SET_TARGET(1, 1)
        MT = MPROT
        ZT = ZETA
        XI = ZT/(2D0-ZT)
        XIK = XI
        DELTA = GPD_TRANGE
        BMAP = GPD_BMAP
      ELSE
        CALL GPD_SET_TARGET(NA, NZ)
        IF (GPD_INCOH.EQ.0) THEN
          GPD_NUCL = 1
          GPD_ISO = 0
          MT = MASS_ION(IB(2))
          ZT = ZETA
          XI = ZT/(2D0-ZT)
          XIK = NA*XI
          DELTA = GPD_TRANGEA
          BMAP = GPD_BMAPA
          IF (BMAP.LE.0D0) BMAP = GPD_WS_R2()/3D0
        ELSE
          GPD_NUCL = 2
          MT = MPROT
          ZT = NA*ZETA
          IF (ZT.GE.1D0) RETURN
          XI = ZT/(2D0-ZT)
          XIK = XI
          DELTA = GPD_TRANGE
          BMAP = GPD_BMAP
        ENDIF
        IF (XIK.GE.1D0) RETURN
      ENDIF
C     the elastic target: W = sqrt((q + p)^2) above M_X + m_T, with
C     W^2 = shat/zeta + m_T^2 (massless pair kinematics)
      IF (SHAT/ZT + MT**2 .LT. (SQRT(SHAT) + MT)**2) RETURN
      TMIN = -4D0*XI*XI*MT**2/(1D0-XI*XI)
      CALL RANMAR(R)
      IF (GPD_NUCL.EQ.1) THEN
C       coherent nuclei: 0.9 of the map exp(-BMAP |t - t_min|) and 0.1
C       flat, which covers the diffraction maxima of F_A^2 beyond its
C       first zero (the exponential alone leaves their weights unbounded)
        CALL RANMAR(R2)
        IF (R2.LT.0.9D0) THEN
          U = -LOG(1D0 - R*(1D0-EXP(-BMAP*DELTA)))/BMAP
        ELSE
          U = R*DELTA
        ENDIF
        PEXP = BMAP*EXP(-BMAP*U)/(1D0-EXP(-BMAP*DELTA))
        WT = 1D0/(0.9D0*PEXP + 0.1D0/DELTA)
      ELSE
        U = -LOG(1D0 - R*(1D0-EXP(-BMAP*DELTA)))/BMAP
        WT = (1D0-EXP(-BMAP*DELTA))/BMAP*EXP(BMAP*U)
      ENDIF
      T = TMIN - U
      GPD_EVT_T(IB(2), IVEC) = T
      GPD_EVT_SET = 1
      IF (GPD_XMODE.EQ.1) THEN
        USEMC = 1
        CALL RANMAR(RMC1)
        CALL RANMAR(RMC2)
      ELSE
        USEMC = 0
      ENDIF
      IF (GPD_NUCL.EQ.2) THEN
C       incoherent: (1 - F_A^2) A^2 (Z |M_p|^2 + N |M_n|^2)
        GPD_ISO = 1
        CALL GPD%(n)d_ME(P, XIK, T, MU, ANS)
        ANSN = 0D0
        IF (NA.GT.NZ) THEN
          GPD_ISO = -1
          CALL GPD%(n)d_ME(P, XIK, T, MU, ANSN)
        ENDIF
        GPD_ISO = 0
        FA = GPD_FORMFACTOR(T)
        ANS = (NZ*ANS + (NA - NZ)*ANSN)*DBLE(NA)**2*(1D0 - FA*FA)
      ELSE
        CALL GPD%(n)d_ME(P, XIK, T, MU, ANS)
      ENDIF
      USEMC = 0
C     pdg2pdf multiplied the pair density by GPD_PAIR_NORM (PhotonFlux.f),
C     the expected scale of |M|^2 that steers madevent's choice of the
C     beam orientation: divided out here
      ANS = ANS/GPD_PAIR_NORM(ZETA, IB(2))
      MATRIX%(n)d = ANS*WT*GPD_SURVIVAL*%(iden)dD0
      END
"""

CHANNEL_CALL = """      CALL %(prefix)sGPD_CFF(P, XI, T, MU, C, C2)
      CS = CS + C
      CS2 = CS2 + C2"""


def finish_group(exporter, pdir, subproc_group, fortran_model):
    """Write the GPD files of the shadow matrix elements of one subprocess
    group into its directory pdir (called after the shadow export)."""
    import madgraph.iolibs.export_v4 as export_v4

    mgme = exporter.mgme_dir
    gpd_inc = open(pjoin(mgme, 'Template', 'LO', 'Source', 'GPD', 'gpd.inc')).read()
    if not hasattr(exporter, 'gpd_extra'):
        exporter.gpd_extra = {'lorentz': [], 'couplings': []}
    sa_opt = dict(exporter.opt)
    sa_opt['export_format'] = 'standalone'
    sa_opt['sa_symmetry'] = False
    sa = export_v4.ProcessExporterFortranSA(exporter.dir_path, sa_opt)
    sa.model = exporter.model
    sa.cmd_options = {}
    vector_couplings = []
    coupl = pjoin(exporter.dir_path, 'Source', 'MODEL', 'coupl.inc')
    if os.path.exists(coupl):
        vector_couplings = re.findall(r'::\s*(\w+)\(VECSIZE_MEMMAX\)', open(coupl).read(), re.I)
    tmp = pjoin(pdir, 'gpd_tmp')
    if not os.path.isdir(tmp):
        os.mkdir(tmp)
    for ime, me in enumerate(subproc_group.get('matrix_elements'), start=1):
        channels = shadow_channels(me)
        if not channels:
            continue
        infos, namps, prefixes = [], set(), []
        for k, amp in enumerate(channels, start=1):
            prefix = 'G%dC%d_' % (ime, k)
            prefixes.append(prefix)
            me_c = helas_objects.HelasMatrixElement(amp)
            files = {}
            for name, writer_fct in (('onia.inc', sa.write_onia_file if me_c.get_nonia() else None),
                                     ('pmassonia.inc', sa.write_pmassonia_file if me_c.get_nonia() else None),
                                     ('pmass.inc', sa.write_pmass_file)):
                if writer_fct is None:
                    continue
                path = pjoin(tmp, name)
                writer_fct(writers.FortranWriter(path), me_c)
                files[name] = path
            files['gpd.inc'] = pjoin(mgme, 'Template', 'LO', 'Source', 'GPD', 'gpd.inc')
            path = pjoin(tmp, 'nexternal.inc')
            sa.write_nexternal_file(writers.FortranWriter(path),
                                    *me_c.get_nexternal_ninitial())
            files['nexternal.inc'] = path
            mpath = pjoin(tmp, 'matrix.f')
            replace_dict = sa.write_matrix_element_v4(writers.FortranWriter(mpath), me_c,
                                                      fortran_model, proc_prefix=prefix,
                                                      return_replace_dict=True)
            info = gpd_export.write_driver(tmp, me_c, replace_dict, prefix)
            infos.append(info)
            namps.add(replace_dict['ncolor'] * len(info.helicity_map()))
            text = inline_includes(open(mpath).read(), files)
            text += "\n" + inline_includes(open(pjoin(tmp, 'gpd_driver.f')).read(), files)
            text = madevent_couplings(text, [])
            with open(pjoin(pdir, 'matrix%d_gpdc%d.f' % (ime, k)), 'w') as fsock:
                fsock.write(text)
            exporter.gpd_extra['lorentz'] += me_c.get_used_lorentz()
            exporter.gpd_extra['couplings'] += flatten(me_c.get_used_couplings())
        if len(namps) != 1:
            raise madgraph.MadGraph5Error(
                "GPD: the channels of %s have different colour or helicity layouts; "
                "their coherent sum is not supported yet" % me.get('processes')[0].nice_string())
        nflow = max(1, len(me.get('color_basis')))
        fmt = {'n': ime, 'process': me.get('processes')[0].nice_string(prefix=False),
               'channels': ", ".join(c.get('process').nice_string(prefix=False) for c in channels),
               'nphys': infos[0].nphys, 'namp': namps.pop(), 'prefix1': prefixes[0],
               'channel_calls': "\n".join(CHANNEL_CALL % {'prefix': p} for p in prefixes),
               'iden': me.get_denominator_factor(),
               'ngraphs': max(1, me.get_number_of_amplitudes()),
               'ncolor_flow': nflow, 'gpd_inc': gpd_inc}
        with open(pjoin(pdir, 'matrix%d_gpd.f' % ime), 'w') as fsock:
            fsock.write(COHERENT_TEMPLATE % fmt)
        # the shadow's MATRIX<n> is kept under another name
        mfile = pjoin(pdir, 'matrix%d.f' % ime)
        text = open(mfile).read()
        # inside the shadow's MATRIX<n> (up to the next program unit) the
        # function and its result are renamed MATRIX<n>_SHADOW
        start = re.search(r'REAL\*8\s+FUNCTION\s+MATRIX%d\(' % ime, text).start()
        nxt = re.search(r'\n\s+(SUBROUTINE|REAL\*8 FUNCTION|INTEGER FUNCTION|LOGICAL FUNCTION|DOUBLE PRECISION FUNCTION)\s',
                        text[start + 10:])
        end = start + 10 + nxt.start() if nxt else len(text)
        body = re.sub(r'\bMATRIX%d\b' % ime, 'MATRIX%d_SHADOW' % ime, text[start:end])
        text = text[:start] + body + text[end:]
        with open(mfile, 'w') as fsock:
            fsock.write(text)
    # the pair is massless
    pmass = pjoin(pdir, 'pmass.inc')
    text = open(pmass).read()
    text = re.sub(r'(PMASS\(2\)\s*=\s*)[^\n]*', r'\1ZERO', text, count=1)
    with open(pmass, 'w') as fsock:
        fsock.write(text)
    for f in os.listdir(tmp):
        os.remove(pjoin(tmp, f))
    os.rmdir(tmp)
    for name in os.listdir(pdir):
        if re.match(r'auto_dsig\d+\.f$', name):
            reset_fixed_beam_pdfs(pjoin(pdir, name))


LPP_IF = re.compile(r'^(\s*)IF \(ABS\(LPP\(IB\(\d\)\)\)\.GE\.1\) THEN\s*$', re.I)
PDF_SET = re.compile(r'^\s+(\w+(?:\(IVEC\))?)\s*=\s*PDG2PDF\(', re.I)


def reset_fixed_beam_pdfs(path):
    """madevent computes the parton densities of a beam only when its lpp is
    not 0 and otherwise keeps the last value of the variable (DATA 1D0). The
    shadow of a GPD process has a mirror orientation (the photon from either
    beam), whose densities of an lpp = 0 beam would then be those of the
    other orientation (zero for the electron of electroproduction asked of
    the lpp = 5 beam, for instance): reset them to 1 for lpp = 0."""
    lines = open(path).read().split('\n')
    out, i = [], 0
    while i < len(lines):
        m = LPP_IF.match(lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent, depth, names, j = m.group(1), 0, [], i
        while j < len(lines):
            line = lines[j]
            code = line.strip().upper() if line[:1] not in ('C', 'c', '!') else ''
            if code.startswith('IF') and code.endswith('THEN'):
                depth += 1
            elif code in ('ENDIF', 'END IF'):
                depth -= 1
                if depth == 0:
                    break
            elif depth == 1 and code == 'ELSE':
                names = []
                break
            pm = PDF_SET.match(line)
            if pm and depth == 1:
                names.append(pm.group(1))
            j += 1
        if j >= len(lines) or not names or depth != 0:
            out.append(lines[i])
            i += 1
            continue
        out.extend(lines[i:j])
        out.append(indent + 'ELSE')
        out.extend(indent + '  ' + name + '=1D0' for name in names)
        out.append(lines[j])
        i = j + 1
    with open(path, 'w') as fsock:
        fsock.write('\n'.join(out))


GPD_RUN_DEFAULTS = [(r'^(\s*)\S+(\s*=\s*lpp1\b)', r'\g<1>5\2'),
                    (r'^(\s*)\S+(\s*=\s*lpp2\b)', r'\g<1>5\2'),
                    (r'^(\s*)\S+(\s*=\s*pdlabel1\b)', r'\g<1>nn23lo1\2'),
                    (r'^(\s*)\S+(\s*=\s*pdlabel2\b)', r'\g<1>nn23lo1\2'),
                    (r'^(\s*)\S+(\s*=\s*pdlabel\b)', r'\g<1>nn23lo1\2'),
                    (r'^(\s*)\S+(\s*=\s*ebeam1\b)', r'\g<1>6800.0\2'),
                    (r'^(\s*)\S+(\s*=\s*ebeam2\b)', r'\g<1>6800.0\2')]


def finish_output(exporter, dir_path):
    """Once per madevent output with GPD processes: the GPD card, the GPD
    beams in the run card (lpp = 5 on both sides: either proton radiates the
    photon or is probed, with the proton PDFs nn23lo1 for the GPDs), the
    gamma-UPC library always built, and the GPD library in the link line of
    the subprocesses."""
    for name in ('run_card.dat', 'run_card_default.dat'):
        path = pjoin(dir_path, 'Cards', name)
        if not os.path.exists(path):
            continue
        text = open(path).read()
        for pattern, repl in GPD_RUN_DEFAULTS:
            text = re.sub(pattern, repl, text, count=1, flags=re.MULTILINE)
        with open(path, 'w') as fsock:
            fsock.write(text)
    # the couplings madevent stores per event (known once the model is
    # written): indexed by IVEC in the GPD channel files
    coupl = pjoin(dir_path, 'Source', 'MODEL', 'coupl.inc')
    vector_couplings = []
    if os.path.exists(coupl):
        vector_couplings = re.findall(r'::\s*(\w+)\(VECSIZE_MEMMAX\)', open(coupl).read(), re.I)
    sub = pjoin(dir_path, 'SubProcesses')
    for pdir in os.listdir(sub):
        if not pdir.startswith('P') or not os.path.isdir(pjoin(sub, pdir)):
            continue
        for fname in os.listdir(pjoin(sub, pdir)):
            if re.match(r'matrix\d+_gpdc\d+\.f$', fname):
                path = pjoin(sub, pdir, fname)
                text = madevent_couplings(open(path).read(), vector_couplings)
                with open(path, 'w') as fsock:
                    fsock.write(text)
    for mk in (pjoin(dir_path, 'Source', 'PDF', 'makefile'), pjoin(dir_path, 'Source', 'makefile')):
        if os.path.exists(mk):
            text = open(mk).read()
            text = text.replace("$(filter edff chff, $(pdlabel1) $(pdlabel2))",
                                "$(filter edff chff, $(pdlabel1) $(pdlabel2) edff)")
            with open(mk, 'w') as fsock:
                fsock.write(text)
    # the elastic protons of the event record (madevent_interface.py)
    shutil.copy(pjoin(madgraph.MG5DIR, 'madgraph', 'various', 'gpd_lhe.py'),
                pjoin(dir_path, 'bin', 'internal', 'gpd_lhe.py'))
    card = pjoin(dir_path, 'Cards', 'gpd_card.dat')
    if not os.path.exists(card):
        with open(card, 'w') as fsock:
            fsock.write(GPD_CARD_MADEVENT)
    mk = pjoin(dir_path, 'SubProcesses', 'makefile')
    text = open(mk).read()
    if '-lgpd' not in text:
        text = text.replace("-L../../lib/ -ldhelas", "-L../../lib/ -lgpd -ldhelas", 1)
        text = text.replace("LIBS = $(LIBDIR)libbias.$(libext)",
                            "LIBS = $(LIBDIR)libgpd.$(libext) $(LIBDIR)libbias.$(libext)", 1)
        text += """
$(LIBDIR)libgpd.$(libext):
	cd ../../Source/GPD && make
"""
        with open(mk, 'w') as fsock:
            fsock.write(text)
