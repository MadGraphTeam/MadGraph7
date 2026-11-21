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
"""One-loop amplitudes of processes with bound states (quarkonia, leptonia) in
MadLoop.

A bound state of momentum K and mass M = m1 + m2 is generated as its two
constituent legs (leg['onium']), and MadLoop computes the loop amplitudes of
that constituent process with the constituents at zero relative momentum,
p_i = (m_i/M) K. Its amplitude is projected onto the bound state as at tree
level (arXiv:2510.26773): the colour projector (delta_ij/sqrt(Nc) or
sqrt(2) T^a_ij) is in the colour bases, and the spin projector is the sum over
the constituent helicities h weighted by ONIA_PROJ(h, J_z).

MadLoop contracts the loop amplitude of each helicity configuration h with
the conjugate Born amplitude of the same h before the reduction. Since the
projection is linear,

  sum_{J_z} conj(B(J_z)) L(J_z) = sum_h conj(W(h)) L(h),
  B(J_z) = sum_h PROJ(h, J_z) A(h),   W(h) = sum_{J_z} conj(PROJ(h, J_z)) B(J_z),

so the projected interference is MadLoop's ordinary one with the Born
amplitudes A(h) of each constituent helicity configuration replaced by the
effective Born W(h). The configurations h sharing the helicities of the other
(elementary) legs are combined; their index is OTHERS(h).

A P-wave bound state (one per process) is the derivative of the projected
amplitude in the relative momentum q of its constituents, p_Q = K/2 + q and
p_Qbar = K/2 - q, along the polarisation vectors eps(L_z) of the orbital
angular momentum, combined with the spin projectors by the Clebsch-Gordan
coefficients <1 L_z; S S_z|J J_z> as at tree level. MadLoop computes the
constituent amplitudes in dual numbers (value, derivative along eps(L_z)):
the HELAS routines of a P-wave output are the dual ones of the tree-level
output, the loop polynomials carry their derivative (the updates are
linear in each factor), and the derivative of the loop propagators gives
the integrals with one propagator doubled, reduced by TIRust (raised
propagator powers). See onia_loop.inc for the projection of the
interference.

This module collects the data the generated Fortran needs.
"""

import fractions
import re
import itertools

import madgraph
from madgraph import MadGraph5Error


class OniaLoopError(MadGraph5Error):
    """A loop process with bound states outside of what MadLoop supports."""


def has_onia(matrix_element):
    """True if the (loop) matrix element has bound states."""
    try:
        return matrix_element.get_nonia() > 0
    except MadGraph5Error:
        return False


def onia_loop_info(matrix_element):
    """The bound-state data of a LoopHelasMatrixElement, as a dictionary:

      nexternal, ninitial       constituent legs (MadLoop's internal legs)
      nexternal_phys            legs with each bound state counted once
      mapping[i]                physical leg (1-based) of constituent leg i+1
      onia                      one dict per bound state: q1 (particle) and q2
                                (antiparticle) constituent legs, phys (its
                                physical leg), S, L, J, C, id, name
      helc                      the constituent helicity configurations, in the
                                order of get_helicity_matrix (HelConfigs.dat)
      others[h]                 index (1-based) of the helicities of the other
                                legs of configuration h
      nothers                   number of such signatures
      jz                        the J_z of each bound state, one tuple per
                                polarisation configuration of all bound states
      norm                      the colour normalisation (1/3 per singlet, 2
                                per octet), as a Fraction
    """

    ext = sorted(matrix_element.get_external_wavefunctions(),
                 key=lambda wf: wf.get('number_external'))
    nexternal, ninitial = matrix_element.get_nexternal_ninitial()
    if len(ext) != nexternal:
        raise OniaLoopError('Unexpected external wavefunctions for the bound states.')

    onia = []
    pending = {}
    for wf in ext:
        onium = wf.get('onium')
        if not onium:
            continue
        if not wf.get('leg_state'):
            raise OniaLoopError('A bound state in the initial state is not supported.')
        index = onium.get('index')
        if index in pending:
            first = pending.pop(index)
            if wf.get('number_external') != first.get('number_external') + 1:
                raise OniaLoopError('The constituents of %s are not adjacent legs.'
                                    % onium.get('name'))
            q1, q2 = (first, wf) if first.get('is_part') else (wf, first)
            onia.append(dict(q1=q1.get('number_external'), q2=q2.get('number_external'),
                             first=first.get('number_external'),
                             S=onium.get('S'), L=onium.get('L', 0), J=onium.get('J'),
                             C=onium.get('C'), id=abs(onium.get('id')),
                             name=onium.get('name'), index=index))
        else:
            pending[index] = wf
    if pending:
        raise OniaLoopError('A bound state without its second constituent.')
    if not onia:
        raise OniaLoopError('No bound state in this process.')
    pwave = [o for o in onia if o['L'] != 0]
    for o in onia:
        if o['L'] > 1:
            raise OniaLoopError('Bound states with L > 1 (%s) are not supported at one loop.'
                                % o['name'])
        if o['L'] == 0 and o['S'] != o['J']:
            raise OniaLoopError('Inconsistent S-wave bound state %s (S=%s, J=%s).'
                                % (o['name'], o['S'], o['J']))
        if o['L'] == 1 and not abs(o['S'] - 1) <= o['J'] <= o['S'] + 1:
            raise OniaLoopError('Inconsistent P-wave bound state %s (S=%s, J=%s).'
                                % (o['name'], o['S'], o['J']))
    if len(pwave) > 1:
        raise OniaLoopError('Processes with more than one P-wave bound state (%s) are '
                            'not supported at one loop.'
                            % ', '.join(o['name'] for o in pwave))

    # physical leg of each constituent leg: the second constituent of a bound
    # state takes the physical number of the first one
    second = set(max(o['q1'], o['q2']) for o in onia)
    mapping = []
    phys = 0
    for wf in ext:
        n = wf.get('number_external')
        if n not in second:
            phys += 1
        mapping.append(phys)
    nexternal_phys = phys
    for o in onia:
        o['phys'] = mapping[o['first'] - 1]

    # helicity configurations of the constituent legs, as in HelConfigs.dat
    helc = [tuple(h) for h in matrix_element.get_helicity_matrix()]
    constituents = set(n for o in onia for n in (o['q1'], o['q2']))
    other_legs = [i for i in range(nexternal) if i + 1 not in constituents]
    signatures = {}
    others = []
    for h in helc:
        key = tuple(h[i] for i in other_legs)
        if key not in signatures:
            signatures[key] = len(signatures) + 1
        others.append(signatures[key])

    jz = list(itertools.product(*[range(-o['J'], o['J'] + 1) for o in onia]))

    norm = fractions.Fraction(1)
    for o in onia:
        norm *= fractions.Fraction(1, 3) if o['C'] == 1 else fractions.Fraction(2)

    # the P-wave bound state (one at most): its index (1-based) in onia, 0 if
    # there is none
    ipwave = onia.index(pwave[0]) + 1 if pwave else 0

    return dict(nexternal=nexternal, ninitial=ninitial, nexternal_phys=nexternal_phys,
                mapping=mapping, onia=onia, helc=helc, others=others,
                nothers=len(signatures), jz=jz, norm=norm, ipwave=ipwave)


def has_pwave(matrix_element):
    """True if the (loop) matrix element has a P-wave bound state (its
    constituent legs carry onium['L'] = 1)."""
    if not has_onia(matrix_element):
        return False
    for leg in matrix_element.get('processes')[0].get('legs'):
        onium = leg.get('onium')
        if onium and onium.get('L', 0) >= 1:
            return True
    return False


def complex_fraction_product(a, b):
    """(re, im) of the product of two complex numbers given as (re, im)."""
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def born_born_color_factors(ampb_to_jampb, color_matrix_num, color_matrix_denom):
    """The colour factors CFB[i][j] = sum_ab conj(c_a(i)) M_ab c_b(j) between
    Born amplitudes i and j, with c_a(i) the coefficient of amplitude i in
    colour flow a and M the colour matrix of the (projected) Born colour basis,
    as (re, im) Fractions; |Born|^2 = sum_ij conj(A_i) CFB_ij A_j."""

    def as_complex(coeff):
        frac, is_imag = coeff
        return (fractions.Fraction(0), frac) if is_imag else (frac, fractions.Fraction(0))

    n = len(ampb_to_jampb)
    res = [[(fractions.Fraction(0), fractions.Fraction(0))] * n for _ in range(n)]
    for i, ilist in enumerate(ampb_to_jampb):
        for j, jlist in enumerate(ampb_to_jampb):
            re, im = fractions.Fraction(0), fractions.Fraction(0)
            for (a, ca), (b, cb) in itertools.product(ilist, jlist):
                m = fractions.Fraction(color_matrix_num[a][b]) / fractions.Fraction(color_matrix_denom[a])
                if m == 0:
                    continue
                cai = as_complex(ca)
                cai = (cai[0], -cai[1])
                prod = complex_fraction_product(cai, as_complex(cb))
                re += prod[0] * m
                im += prod[1] * m
            res[i][j] = (re, im)
    return res


def fortran_double(frac):
    """A Fraction as a Fortran double-precision literal."""
    return ('%.17e' % float(frac)).replace('e', 'D')


def fortran_quad(frac):
    """A Fraction as a Fortran quadruple-precision expression, exact for
    rational colour and normalisation factors."""
    frac = fractions.Fraction(frac)
    if frac.denominator == 1:
        return '%d.0E0_16' % frac.numerator
    return '(%d.0E0_16/%d.0E0_16)' % (frac.numerator, frac.denominator)


#===============================================================================
# Colour-linked Born amplitudes (IR poles of the virtual)
#===============================================================================
def constituent_color_legs(matrix_element):
    """FKS legs (number, state, colour, masslessness) of the constituent legs,
    as fks_common.legs_to_color_link_string needs them."""
    import madgraph.fks.fks_common as fks_common
    process = matrix_element.get('processes')[0]
    model = process.get('model')
    legs = []
    for leg in process.get('legs'):
        part = model.get_particle(leg.get('id'))
        legs.append(fks_common.FKSLeg({'id': leg.get('id'), 'number': leg.get('number'),
                                       'state': leg.get('state'),
                                       'color': part.get_color(),
                                       'massless': part.get('mass').lower() == 'zero'}))
    return legs


def color_linked_born_factors(matrix_element, amp_to_jamp):
    """The colour factors of the colour-linked Born amplitudes:
    {(j, k): CL} for each ordered pair of coloured constituent legs j, k
    (j == k for massive legs), with
       <M0| T_j.T_k |M0> = sum_ab conj(A_a) CL[a][b] A_b
    over the Born amplitudes A (unprojected constituent helicities), the
    colour projectors of the bound states included. The link is inserted in
    the colour strings of the Born diagrams before their projection
    (fks_common.legs_to_color_link_string, the normalisation of MadFKS) and
    the projected link basis is contracted with the projected Born basis.
    amp_to_jamp is the exporter's get_amp_to_jamp_map."""
    import madgraph.fks.fks_common as fks_common
    import madgraph.core.color_amp as color_amp
    import madgraph.loop.loop_color_amp as loop_color_amp

    process = matrix_element.get('processes')[0]
    model = process.get('model')
    born_diagrams = matrix_element.get('base_amplitude').get('born_diagrams')
    basis = loop_color_amp.LoopColorBasis()
    col_obj = [basis.colorize(d, model) for d in born_diagrams]
    onia_numbers = basis.get_onia_color_numbers(process)
    born_basis = matrix_element.get('born_color_basis')
    nborn = matrix_element.get_number_of_born_amplitudes()
    ampb = amp_to_jamp(matrix_element.get_born_color_amplitudes(), nborn)

    legs = [l for l in constituent_color_legs(matrix_element) if abs(l.get('color')) > 1]
    result = {}
    for leg1 in legs:
        for leg2 in legs:
            if leg1 is leg2 and leg1.get('massless'):
                continue
            link = fks_common.legs_to_color_link_string(leg1, leg2)
            linked = []
            for old in col_obj:
                new = dict(old)
                for key, string in new.items():
                    new[key] = string.create_copy()
                    for col in new[key]:
                        for ind in col:
                            for pair in link['replacements']:
                                if ind == pair[0]:
                                    col[col.index(ind)] = pair[1]
                    new[key].product(link['string'])
                if onia_numbers:
                    basis.OniaColorProjection(new, onia_numbers)
                linked.append(new)
            link_basis = color_amp.ColorBasis()
            for index, new in enumerate(linked):
                link_basis.update_color_basis(new, index)
            cm = color_amp.ColorMatrix(born_basis, link_basis)
            ampl = amp_to_jamp(matrix_element.generate_color_amplitudes(
                link_basis, matrix_element.get_born_diagrams()), nborn) if link_basis else \
                [[] for _ in range(nborn)]
            nrows = len(born_basis)
            cl = [[(fractions.Fraction(0), fractions.Fraction(0))] * nborn
                  for _ in range(nborn)]
            for a, alist in enumerate(ampb):
                for b, blist in enumerate(ampl):
                    re, im = fractions.Fraction(0), fractions.Fraction(0)
                    for (ja, ca), (jb, cb) in itertools.product(alist, blist):
                        m_val, m_imag = cm.col_matrix_fixed_Nc.get((ja, jb), (0, False))
                        if m_val == 0:
                            continue
                        m = (fractions.Fraction(0), fractions.Fraction(m_val)) if m_imag \
                            else (fractions.Fraction(m_val), fractions.Fraction(0))
                        x = (ca[0], fractions.Fraction(0)) if not ca[1] else \
                            (fractions.Fraction(0), ca[0])
                        x = (x[0], -x[1])
                        y = (cb[0], fractions.Fraction(0)) if not cb[1] else \
                            (fractions.Fraction(0), cb[0])
                        prod = complex_fraction_product(complex_fraction_product(x, m), y)
                        re += prod[0]
                        im += prod[1]
                    cl[a][b] = (re, im)
            result[(leg1.get('number'), leg2.get('number'))] = cl
    return result


def physical_color_links(info, constituent_links):
    """Sum the constituent colour links onto the physical legs: the colour
    charge of a bound state is T_Q + T_Qbar. Returns {(j, k): CL} for the
    ordered pairs of different physical legs j, k (the IR poles need the
    Casimir of a leg, not its self link)."""
    result = {}
    for (a, b), cl in constituent_links.items():
        key = (info['mapping'][a - 1], info['mapping'][b - 1])
        if key[0] == key[1]:
            continue
        if key in result:
            old = result[key]
            result[key] = [[(old[i][j][0] + cl[i][j][0], old[i][j][1] + cl[i][j][1])
                            for j in range(len(cl))] for i in range(len(cl))]
        else:
            result[key] = cl
    return result


#===============================================================================
# P-wave: the coefficient construction in dual numbers
#===============================================================================
_PW_UPDATE = re.compile(r'^CALL (?P<pre>\w*?)UPDATE_WL_(?P<r>\d+)_(?P<v>\d+)'
                        r'\(WL\((?P<a>[^()]*)\),(?P<n>[^,()]+),COEFS,(?P<m>[^,()]+),'
                        r'(?P<k>[^,()]+),WL\((?P<b>[^()]*)\)\)$')
_PW_CREATE = re.compile(r'^CALL (?P<pre>\w*?)CREATE_LOOP_COEFS'
                        r'\(WL\((?P<a>[^()]*)\),(?P<rest>[^()]*)\)$')


def pwave_coef_construction_calls(calls):
    """The double-precision coefficient construction of a P-wave output: the
    loop wavefunctions WL carry their derivative DWL in the relative
    momentum. The (dual) loop vertex routines are unchanged: their dual
    coefficients COEFS = C0 + d C1 update both polynomials,
       WL' = U(WL, C0),   DWL' = U(DWL, C0) + U(WL, C1),
    with the plain update routines U, and the closed loops are merged with
    the P-wave weights (PW_CREATE_LOOP_COEFS). calls are the statements
    of get_coef_construction_calls, with the process prefix substituted."""

    result = []
    for entry in calls:
        result.append('\n'.join(_pwave_coef_statements(entry.split('\n'))))
    return result


def _pwave_coef_statements(statements):
    """pwave_coef_construction_calls for the statements of one entry"""
    result = []
    for call in statements:
        text = call.strip()
        if not text.upper().startswith('CALL '):
            result.append(call)
            continue
        update = _PW_UPDATE.match(text)
        create = _PW_CREATE.match(text)
        if update:
            d = update.groupdict()
            d['rout'] = int(d['r']) + int(d['v'])
            result.append('\n'.join([
                'CALL {pre}PW_SPLIT_COEFS(COEFS,COEFS0,COEFS1)',
                'CALL {pre}UPDATE_WL_{r}_{v}(WL({a}),{n},COEFS0,{m},{k},WL({b}))',
                'CALL {pre}UPDATE_WL_{r}_{v}(DWL({a}),{n},COEFS0,{m},{k},DWL({b}))',
                'CALL {pre}UPDATE_WL_{r}_{v}(WL({a}),{n},COEFS1,{m},{k},TWL)',
                'CALL {pre}PW_ADD_WL(TWL,{rout},{n},{k},DWL({b}))']).format(**d))
        elif create:
            result.append('CALL {pre}PW_CREATE_LOOP_COEFS(WL({a}),DWL({a}),{rest})'
                          .format(**create.groupdict()))
        elif 'UPDATE_WL' in text.upper() or 'LOOP_COEFS' in text.upper() or \
                'TERMINAL_WL' in text.upper():
            raise OniaLoopError('Unexpected coefficient construction call for a P-wave '
                                'bound state: %s' % text)
        else:
            # a loop vertex (dual coefficients)
            result.append(call)
    return result


# The dual objects shared by the routines of a P-wave output are in the module
# <prefix>PW_DUAL (dual numbers cannot be in COMMON blocks). gfortran 15
# crashes on renamed module variables (USE ..., ONLY: W=>PWW), so the calls
# use the module names.
_PW_DUAL_NAMES = [(re.compile(r'\bW\('), 'PWW('), (re.compile(r'\bAMP\('), 'PWAMP('),
                  (re.compile(r'\bAMPL\('), 'PWAMPL('), (re.compile(r'\bPL\('), 'PWPL(')]


def pwave_dual_names(calls):
    """The HELAS calls of a P-wave output with the names of the dual
    wavefunctions (W), amplitudes (AMP, AMPL) and loop momenta (PL) of the
    module PW_DUAL."""
    result = []
    for call in calls:
        for pattern, name in _PW_DUAL_NAMES:
            call = pattern.sub(name, call)
        result.append(call)
    return result
