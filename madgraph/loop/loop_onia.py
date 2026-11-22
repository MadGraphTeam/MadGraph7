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
    if len(pwave) > 2:
        raise OniaLoopError('Processes with more than two P-wave bound states (%s) are '
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

    # the P-wave bound states (two at most): their indices (1-based) in onia,
    # in the order of their dual directions (component 2**(k-1) for the k-th);
    # ipwave is the first one, 0 if there is none
    ipwaves = [onia.index(o) + 1 for o in pwave]
    ipwave = ipwaves[0] if ipwaves else 0

    return dict(nexternal=nexternal, ninitial=ninitial, nexternal_phys=nexternal_phys,
                mapping=mapping, onia=onia, helc=helc, others=others,
                nothers=len(signatures), jz=jz, norm=norm, ipwave=ipwave,
                ipwaves=ipwaves, npwave=len(ipwaves))


def npwave(matrix_element):
    """The number of P-wave bound states of the (loop) matrix element."""
    if not has_onia(matrix_element):
        return 0
    seen = set()
    for wf in matrix_element.get_external_wavefunctions():
        onium = wf.get('onium')
        if onium and onium.get('L', 0) >= 1:
            seen.add(onium.get('index'))
    return len(seen)


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
# P-wave: the colour of the soft current (P -> S pole of the virtual)
#===============================================================================
# A soft gluon turns a P-wave quarkonium into the S-wave states of
# pwave_aux_states (arXiv:2402.19221, sect. 3.1 and appendix B), through the
# effective generators
#   Q_eff(18)^a = (2+r+1/r)/sqrt(2 N_c) Delta^a     (P[1] -> S[8], Delta^a_ob = delta_ab)
#   Q_eff(81)^a = (2+r+1/r)/sqrt(2 N_c) Delta^aT    (P[8] -> S[1])
#   Q_eff(88)^a = (2+r+1/r) D^a/2 + (1/r-r) T^a/2   (P[8] -> S[8], D^a_bc = d_abc,
#                                                     T^a_bc = -i f_abc)
# between normalised colour states, r = m_Q/m_Qbar; for equal masses
# Q_eff = 2 (T_Q - T_Qbar), from the derivative of the soft functions of the
# heavy quarks in their relative momentum. The colour charges are those of
# the onia FKS layer (fks_onia.colour_charge_string): t^a for a final quark
# or an initial antiquark, -t^aT for a final antiquark or an initial quark,
# T^a = -i f for an octet.

def pwave_aux_states(onium):
    """The S-wave states of the soft current of a P-wave quarkonium, as
    (colour C', label of Q_eff): P[1] -> S[8]; P[8] -> S[1] and S[8]."""
    if onium['C'] == 1:
        return [(8, '18')]
    return [(1, '81'), (8, '88')]


def _colour_charge_string(rep, num, a, fresh):
    """The colour charge Q^a of a leg of representation rep (3, -3, 8) and
    colour index num, acting on a colour string whose index num is renamed
    fresh."""
    import madgraph.core.color_algebra as color_algebra
    if rep == 3:
        return color_algebra.ColorString([color_algebra.T(a, num, fresh)])
    if rep == -3:
        string = color_algebra.ColorString([color_algebra.T(a, fresh, num)])
        string.coeff = fractions.Fraction(-1)
        return string
    if rep == 8:
        string = color_algebra.ColorString([color_algebra.f(a, num, fresh)])
        string.coeff = fractions.Fraction(-1)
        string.is_imaginary = True
        return string
    raise OniaLoopError('No colour charge for the representation %s.' % rep)


def _colour_contract(struct_p, struct_s, repl_p, repl_s, insertion, Nc=3):
    """sum over colours of conj(c_p) [insertion] c_s for two immutable colour
    strings, their indices renamed by repl_p and repl_s first, as (re, im)
    Fractions."""
    import madgraph.core.color_algebra as color_algebra
    import madgraph.core.color_amp as color_amp

    def rename(struct, repl):
        return tuple((elem[0], tuple(repl.get(i, i) for i in elem[1])) for elem in struct)

    linked = color_algebra.ColorString()
    linked.from_immutable(rename(struct_s, repl_s))
    if insertion is not None:
        linked.product(insertion.create_copy())
    other = color_algebra.ColorString()
    other.from_immutable(color_amp.ColorMatrix.fix_summed_indices(
        linked.to_immutable(), rename(struct_p, repl_p)))
    product = linked.create_copy()
    product.product(other.complex_conjugate())
    re, im = color_algebra.ColorFactor([product]).full_simplify().set_Nc(Nc)
    return fractions.Fraction(re), fractions.Fraction(im)


def _to_amplitude_space(mat, amp_p, amp_s, nborn):
    """M[a][b] = sum conj(c_ja(a)) mat[(ja, jb)] c_jb(b) over the colour
    flows of Born amplitudes a and b (amp_p, amp_s: get_amp_to_jamp_map of
    the two bases), as (re, im) Fractions; mat maps (ja, jb) to (re, im)."""

    def as_complex(coeff):
        frac, is_imag = coeff
        return (fractions.Fraction(0), frac) if is_imag else (frac, fractions.Fraction(0))

    zero = (fractions.Fraction(0), fractions.Fraction(0))
    res = [[zero] * nborn for _ in range(nborn)]
    for a, alist in enumerate(amp_p):
        for b, blist in enumerate(amp_s):
            re, im = fractions.Fraction(0), fractions.Fraction(0)
            for (ja, ca), (jb, cb) in itertools.product(alist, blist):
                m = mat.get((ja, jb))
                if m is None or m == zero:
                    continue
                x = as_complex(ca)
                prod = complex_fraction_product(complex_fraction_product(
                    (x[0], -x[1]), m), as_complex(cb))
                re += prod[0]
                im += prod[1]
            res[a][b] = (re, im)
    return res


def pwave_soft_current(matrix_element, info, amp_to_jamp):
    """The colour data of the soft current of the P-wave bound state of a
    loop matrix element, for its P -> S pole (ONIA_PW_SOFT_POLE): a list
    with one dict per S-wave state S' of pwave_aux_states that the colour
    allows, with
      colour, label   C' and the label of Q_eff
      cnorm           n_C'/n_C, the ratio of the squared normalisations of
                      the colour projectors (1/3 for a singlet, 2 for an
                      octet)
      cs              the colour matrix of S' between Born amplitudes,
                      CS[a][b] = sum_colours conj(c'_a) c'_b
      links           {(k, kind): X} for every coloured physical leg k
                      other than the P-wave state, kind 1 for Delta (18,
                      81) or D (88) and 2 for T (88), with
                      X[a][b] = sum_colours conj(c_a) Q_k^a Q_eff^a c'_b
                      without the mass factors of Q_eff
    c (c') are the colour strings of the Born amplitudes projected on the
    colour of the P-wave state (of S'), as (re, im) Fractions. amp_to_jamp
    is the exporter's get_amp_to_jamp_map."""
    import madgraph.core.color_algebra as color_algebra
    import madgraph.core.color_amp as color_amp
    import madgraph.loop.loop_color_amp as loop_color_amp

    pw = info['onia'][info['ipwave'] - 1]
    process = matrix_element.get('processes')[0]
    model = process.get('model')
    legs = process.get('legs')
    nborn = matrix_element.get_number_of_born_amplitudes()
    born_diagrams = matrix_element.get('base_amplitude').get('born_diagrams')
    basis = loop_color_amp.LoopColorBasis()
    col_obj = [basis.colorize(d, model) for d in born_diagrams]
    onia_numbers = basis.get_onia_color_numbers(process)
    born_basis = matrix_element.get('born_color_basis')
    keys_p = sorted(born_basis.keys())
    amp_p = amp_to_jamp(matrix_element.get_born_color_amplitudes(), nborn)
    # the octet index of a quarkonium is the number of its quark (colour 3)
    n_q = pw['q1']
    octet_index = {}
    for o in info['onia']:
        octet_index[o['q1']] = o
    a, fresh, c_new, m_new = -4001, -4002, -4003, -4004

    def qeff(kind, b, c):
        if kind == 'D':
            return color_algebra.ColorString([color_algebra.d(a, b, c)])
        return _colour_charge_string(8, b, a, c)

    # the coloured physical legs other than the P-wave state: (number of
    # the colour index, representation of the colour charge, physical leg)
    charged = []
    for leg in legs:
        num = leg.get('number')
        onium = leg.get('onium')
        if onium:
            o = octet_index.get(num)
            if o is None or o is pw or o['C'] != 8:
                continue
            if abs(model.get_particle(leg.get('id')).get_color()) != 3:
                continue
            charged.append((num, 8, info['mapping'][num - 1]))
            continue
        colour = model.get_particle(leg.get('id')).get_color()
        if colour in (1, 0):
            continue
        rep = 8 if colour == 8 else (colour if leg.get('state') else -colour)
        charged.append((num, rep, info['mapping'][num - 1]))

    norm = {1: fractions.Fraction(1, 3), 8: fractions.Fraction(2)}
    result = []
    for colour, label in pwave_aux_states(pw):
        numbers = [[charges, numbers_, (colour if n_q in numbers_ else charge)]
                   for charges, numbers_, charge in onia_numbers]
        projected = []
        for old in col_obj:
            new = dict((key, string.create_copy()) for key, string in old.items())
            basis.OniaColorProjection(new, numbers)
            projected.append(new)
        basis_s = color_amp.ColorBasis()
        for index, new in enumerate(projected):
            basis_s.update_color_basis(new, index)
        if not len(basis_s):
            # colour forbidden: this S-wave state has no Born
            continue
        keys_s = sorted(basis_s.keys())
        amp_s = amp_to_jamp(matrix_element.generate_color_amplitudes(
            basis_s, matrix_element.get_born_diagrams()), nborn)
        cm = color_amp.ColorMatrix(basis_s)
        mat = {}
        for (i, j), (m_val, m_imag) in cm.col_matrix_fixed_Nc.items():
            if m_val:
                mat[(i, j)] = (fractions.Fraction(0), fractions.Fraction(m_val)) if m_imag \
                    else (fractions.Fraction(m_val), fractions.Fraction(0))
        cs = _to_amplitude_space(mat, amp_s, amp_s, nborn)
        links = {}
        for num, rep, phys in charged:
            for kind in (['Delta'] if label in ('18', '81') else ['D', 'T']):
                repl_p, repl_s = {}, {num: fresh}
                insertion = _colour_charge_string(rep, num, a, fresh)
                if label == '18':
                    repl_s[n_q] = a
                elif label == '81':
                    repl_p[n_q] = a
                else:
                    repl_s[n_q] = c_new
                    insertion.product(qeff(kind, n_q, c_new))
                xmat = {}
                for ib, ks in enumerate(keys_s):
                    for ia, kp in enumerate(keys_p):
                        value = _colour_contract(kp, ks, repl_p, repl_s, insertion)
                        if value != (0, 0):
                            xmat[(ia, ib)] = value
                if xmat:
                    links[(phys, 2 if kind == 'T' else 1)] = \
                        _to_amplitude_space(xmat, amp_p, amp_s, nborn)
        result.append(dict(colour=colour, label=label, cnorm=norm[colour] / norm[pw['C']],
                           cs=cs, links=links))
    return result


#===============================================================================
# P-wave: the coefficient construction in dual numbers
#===============================================================================
_PW_UPDATE = re.compile(r'^CALL (?P<pre>\w*?)UPDATE_WL_(?P<r>\d+)_(?P<v>\d+)'
                        r'\(WL\((?P<a>[^()]*)\),(?P<n>[^,()]+),COEFS,(?P<m>[^,()]+),'
                        r'(?P<k>[^,()]+),WL\((?P<b>[^()]*)\)\)$')
_PW_CREATE = re.compile(r'^CALL (?P<pre>\w*?)CREATE_LOOP_COEFS'
                        r'\(WL\((?P<a>[^()]*)\),(?P<rest>[^()]*)\)$')


def pwave_coef_construction_calls(calls, npwave=1):
    """The double-precision coefficient construction of a P-wave output: the
    loop wavefunctions WL carry their derivative DWL in the relative
    momentum. The (dual) loop vertex routines are unchanged: their dual
    coefficients COEFS = C0 + d C1 update both polynomials,
       WL' = U(WL, C0),   DWL' = U(DWL, C0) + U(WL, C1),
    with the plain update routines U, and the closed loops are merged with
    the P-wave weights (PW_CREATE_LOOP_COEFS). calls are the statements
    of get_coef_construction_calls, with the process prefix substituted.

    Two P-wave bound states (npwave=2): the dual coefficients have the
    components C0 (value), C1 and C2 (derivatives along the directions of
    the first and of the second state) and C3 (the mixed derivative), and
    the loop wavefunctions WL, DWL (d1), DWL2 (d2) and D2WL (d1 d2) follow
    the Leibniz rule,
       WL' = U(WL,C0),  DWL' = U(DWL,C0) + U(WL,C1),
       DWL2' = U(DWL2,C0) + U(WL,C2),
       D2WL' = U(D2WL,C0) + U(DWL,C2) + U(DWL2,C1) + U(WL,C3);
    the closed loops are merged by PW_CREATE_LOOP_COEFS2."""

    result = []
    for entry in calls:
        result.append('\n'.join(_pwave_coef_statements(entry.split('\n'), npwave)))
    return result


def _pwave_coef_statements(statements, npwave=1):
    """pwave_coef_construction_calls for the statements of one entry"""
    result = []
    for call in statements:
        text = call.strip()
        if not text.upper().startswith('CALL '):
            result.append(call)
            continue
        update = _PW_UPDATE.match(text)
        create = _PW_CREATE.match(text)
        if update and npwave == 2:
            d = update.groupdict()
            d['rout'] = int(d['r']) + int(d['v'])
            u = 'CALL {pre}UPDATE_WL_{r}_{v}(%s({a}),{n},%s,{m},{k},%s)'
            add = 'CALL {pre}PW_ADD_WL(TWL,{rout},{n},{k},%s({b}))'
            result.append('\n'.join([
                'CALL {pre}PW_SPLIT_COEFS2(COEFS,COEFS0,COEFS1,COEFS2,COEFS3)',
                u % ('D2WL', 'COEFS0', 'D2WL({b})'),
                u % ('DWL', 'COEFS2', 'TWL'), add % 'D2WL',
                u % ('DWL2', 'COEFS1', 'TWL'), add % 'D2WL',
                u % ('WL', 'COEFS3', 'TWL'), add % 'D2WL',
                u % ('DWL', 'COEFS0', 'DWL({b})'),
                u % ('WL', 'COEFS1', 'TWL'), add % 'DWL',
                u % ('DWL2', 'COEFS0', 'DWL2({b})'),
                u % ('WL', 'COEFS2', 'TWL'), add % 'DWL2',
                u % ('WL', 'COEFS0', 'WL({b})')]).format(**d))
        elif create and npwave == 2:
            result.append('CALL {pre}PW_CREATE_LOOP_COEFS2(WL({a}),DWL({a}),DWL2({a}),'
                          'D2WL({a}),{rest})'.format(**create.groupdict()))
        elif update:
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


def pwave_dual_names(calls, mp=False):
    """The HELAS calls of a P-wave output with the names of the dual
    wavefunctions (W), amplitudes (AMP, AMPL) and loop momenta (PL) of the
    module PW_DUAL: PWW, ... in double precision, MPPWW, ... for the
    quadruple-precision calls (mp)."""
    prefix = 'MP' if mp else ''
    result = []
    for call in calls:
        for pattern, name in _PW_DUAL_NAMES:
            call = pattern.sub(prefix + name, call)
        result.append(call)
    return result
