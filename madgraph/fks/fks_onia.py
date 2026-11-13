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
"""FKS subtraction for processes with bound states (quarkonia, leptonia) at NLO
QCD, arXiv:2402.19221.

A bound state is generated as its two constituent legs, which carry the same
leg['onium'] dictionary (madgraph_interface.extract_process) and are projected
inside the matrix-element templates. For FKS the pair is ONE physical particle
of momentum K and mass m1+m2:
  - it is never the FKS parton i and never a collinear partner j;
  - it is a soft-only partner of the final-state gluons iff it is a quarkonium
    in a colour octet (it radiates as a massive octet) or in a P wave (the
    flavour-changing soft current of arXiv:2402.19221, sect. 3.1);
  - colour-singlet S-wave quarkonia and all leptonia (colourless constituents)
    are colourless massive particles.
In the Python FKS objects the constituent legs are kept (diagram generation
needs them), adjacent, with the FIRST constituent standing for the bound
state; the exporter maps constituent numbers onto physical ones
(physical_number) for everything written to Fortran.

Supported (arXiv:2402.19221): at most one P-wave quarkonium, any number of
S-wave quarkonia and of S- and P-wave leptonia, final-state onia only, QCD
corrections only.
"""

import fractions

import madgraph.various.misc as misc


class OniaNLOError(Exception):
    """A process with bound states outside the NLO (FKS) category."""
    pass


def onium_of(leg):
    """The onium dictionary of a leg ({} for an elementary particle)."""
    try:
        onium = leg.get('onium')
    except Exception:
        onium = dict.get(leg, 'onium', {})
    return onium or {}


def has_onia(legs):
    return any(onium_of(l) for l in legs)


def onium_pairs(legs):
    """[(first, second)] positions (0-based) of the constituents of each onium,
    in the order of the legs. Raises if the two constituents of an onium are
    not adjacent."""
    seen = {}
    pairs = []
    for pos, leg in enumerate(legs):
        onium = onium_of(leg)
        if not onium:
            continue
        key = (onium['index'], leg.get('state'))
        if key in seen:
            first = seen.pop(key)
            if pos != first + 1:
                raise OniaNLOError('the constituents of a bound state are not adjacent '
                                   '(positions %d and %d)' % (first + 1, pos + 1))
            pairs.append((first, pos))
        else:
            seen[key] = pos
    if seen:
        raise OniaNLOError('a bound state without its second constituent')
    return pairs


def is_second_constituent(legs, pos):
    """True for the second constituent of an onium (hidden for FKS)."""
    return any(pos == b for _, b in onium_pairs(legs))


def physical_number(legs, number):
    """The physical (1-based) number of the leg with constituent number `number`:
    the second constituent of every onium before it is removed. The first
    constituent of an onium gets the number of the bound state."""
    pairs = onium_pairs(legs)
    pos = number - 1
    for a, b in pairs:
        if pos == b:
            raise OniaNLOError('leg %d is the second constituent of a bound state' % number)
    return number - sum(1 for _, b in pairs if b < pos)


def physical_legs(legs):
    """The legs as FKS sees them: the second constituent of each onium dropped."""
    pairs = onium_pairs(legs)
    hidden = set(b for _, b in pairs)
    return [l for pos, l in enumerate(legs) if pos not in hidden]


def constituent_colour(leg, model):
    return model.get('particle_dict')[leg.get('id')].get_color()


def is_quarkonium(leg, model):
    """Coloured constituents (a leptonium has colourless ones)."""
    return abs(constituent_colour(leg, model)) == 3


def is_pwave(onium):
    return onium.get('L', 0) >= 1


def physical_colour(leg, model):
    """The QCD representation of the bound state for the colour links:
    8 for a colour octet, 1 otherwise (colour-singlet quarkonia, leptonia)."""
    onium = onium_of(leg)
    if not onium:
        return model.get('particle_dict')[leg.get('id')].get_color()
    if is_quarkonium(leg, model) and onium.get('C') == 8:
        return 8
    return 1


def onium_is_soft_partner(leg, model):
    """A final-state gluon becomes soft against the bound state iff it is a
    quarkonium in a colour octet or in a P wave (arXiv:2402.19221, sect. 3)."""
    onium = onium_of(leg)
    if not onium or not is_quarkonium(leg, model):
        return False
    return onium.get('C') == 8 or is_pwave(onium)


def aux_states(onium):
    """The S-wave states of the soft current of a P-wave quarkonium, as
    (colour of the S wave, effective generator): P[1] -> S[8] (Q_eff(18));
    P[8] -> S[1] (Q_eff(81)) and S[8] (Q_eff(88)). Empty for S waves."""
    if not is_pwave(onium):
        return []
    if onium.get('C') == 1:
        return [(8, '18')]
    return [(1, '81'), (8, '88')]


def check_nlo_onia(legs, perturbation_couplings, loop_option, model):
    """Refuse the processes with bound states outside the supported NLO category.

    legs: the MultiLegs of extract_process (constituents carry 'onium');
    perturbation_couplings: list of perturbed orders ([] at LO)."""
    onia = {}
    for leg in legs:
        onium = onium_of(leg)
        if onium:
            onia.setdefault((onium['index'], leg.get('state')), (leg, onium))
    if not onia or not perturbation_couplings:
        return
    if loop_option not in ('all', 'real', 'LOonly'):
        raise OniaNLOError(
            "Bound states at NLO are supported with [QCD], [real=QCD] and [LOonly=QCD] only, "
            "not with the '%s' mode." % loop_option)
    if [o for o in perturbation_couplings if o != 'QCD']:
        raise OniaNLOError(
            "Bound states at NLO are supported for QCD corrections only (arXiv:2402.19221), "
            "not for [%s]." % ' '.join(perturbation_couplings))
    for (index, state), (leg, onium) in onia.items():
        if not state:
            raise OniaNLOError(
                "A bound state in the initial state (%s) is not supported at NLO." % onium['name'])
    for leg in legs:
        pol = leg.get('polarization') if hasattr(leg, 'get') else None
        if pol:
            raise OniaNLOError(
                "Polarised particles are not supported at NLO in processes with bound states.")
    for leg in legs:
        if onium_of(leg):
            ids = leg.get('ids') if 'ids' in leg else [leg.get('id')]
            part = model.get('particle_dict')[ids[0]]
            if part.get('mass').lower() == 'zero':
                raise OniaNLOError(
                    "The constituents of %s are massless in this model (%s): at NLO a bound state "
                    "needs massive constituents (M = m1 + m2), e.g. 'import model "
                    "sm_onia-c_mass' for charmonia." % (onium_of(leg)['name'], part.get('name')))
    npwave_q = [onium for (leg, onium) in onia.values()
                if is_pwave(onium) and _multileg_is_quarkonium(leg, model)]
    if len(npwave_q) > 1:
        raise OniaNLOError(
            "The FKS counterterms of processes with more than one P-wave quarkonium (%s) are not "
            "implemented: their soft limit needs Born interferences with two Fock states changed at "
            "once (arXiv:2402.19221 covers one P-wave quarkonium). One P-wave quarkonium with any "
            "number of S-wave quarkonia and of leptonia is supported."
            % ', '.join(o['name'] for o in npwave_q))


def _multileg_is_quarkonium(leg, model):
    ids = leg.get('ids') if 'ids' in leg else [leg.get('id')]
    return abs(model.get('particle_dict')[ids[0]].get_color()) == 3


def fock_state_perturbation_tag(loop_option, perturbation_couplings):
    """The ' [..]' tag to append to a process line re-generated for another
    combination of Fock states, so that every combination keeps the NLO mode."""
    if not perturbation_couplings or loop_option == 'tree':
        return ''
    orders = ' '.join(perturbation_couplings)
    if loop_option == 'all':
        return ' [%s]' % orders
    return ' [%s=%s]' % (loop_option, orders)


# ------------------------------------------------------------- colour links --
def physical_fks_legs(legs, model):
    """FKSLegs of a Born with bound states as FKS sees them: the second
    constituent of every onium dropped, the first one standing for the bound
    state (massive, colour 8 for a colour octet, 1 otherwise). Its number stays
    the constituent number, which is the label of the octet index in the
    projected colour strings (color_amp.OniaColorProjection)."""
    import madgraph.fks.fks_common as fks_common
    fkslegs = fks_common.to_fks_legs(legs, model)
    out = fks_common.FKSLegList()
    hidden = set(b for _, b in onium_pairs(fkslegs))
    for pos, leg in enumerate(fkslegs):
        if pos in hidden:
            continue
        if onium_of(leg):
            leg = fks_common.FKSLeg(leg)
            leg['color'] = physical_colour(leg, model)
            leg['massless'] = False
            leg['spin'] = 2 * onium_of(leg).get('J', 0) + 1
            leg['self_antipart'] = True
            leg['is_part'] = True
        out.append(leg)
    return out


def born_color_links(born_me, perturbation='QCD'):
    """The colour links of a Born with bound states, as matrices in the Born
    colour basis (fks_common.born_basis_link_matrices), with the linked legs
    given by their PHYSICAL numbers (one per onium). Only coloured physical
    legs are linked: colour-singlet onia and leptonia never, octet onia as a
    massive octet (self-link included).

    The links are inserted in the simplified (projected) Born colour basis,
    where the octet label of an onium is the number of its first constituent
    and no fundamental index of its constituents survives."""
    import madgraph.fks.fks_common as fks_common
    process = born_me.get('base_amplitude').get('process')
    legs = process.get('legs')
    model = process.get('model')
    plegs = physical_fks_legs(legs, model)
    col_basis = born_me.get('color_basis')
    links_info = fks_common.find_color_links(plegs, symm=True, pert=perturbation)
    if not len(col_basis):
        # colourless, or colour forbidden (q q~ > chi_cJ, zero Born): zero links
        return [{'link': [physical_number(legs, l['number']) for l in link['legs']],
                 'matrix': {}, 'ncolor': 1} for link in links_info]
    result = fks_common.born_basis_link_matrices(col_basis, links_info)
    for link in result:
        link['link'] = [physical_number(legs, n) for n in link['link']]
    return result


# -------------------------------------------------------------- event record --
def leshouche_view(proc, color_basis, ninitial):
    """The physical particles of a process with bound states for the event
    record: (ids, flows), ids the PDG codes (one per onium, its Fock-state PDG),
    flows a list of colour flows, each a list of (colour, anticolour) per
    physical particle. An octet onium flows like a gluon (its index is the
    label of its first constituent in the projected colour basis)."""
    legs = proc.get_legs_with_decays()
    model = proc.get('model')
    hidden = set(b for _, b in onium_pairs(legs))
    plegs = [l for pos, l in enumerate(legs) if pos not in hidden]
    ids = [onium_of(l)['id'] if onium_of(l) else l.get('id') for l in plegs]
    if not color_basis:
        return ids, []
    repr_dict = {}
    for l in plegs:
        repr_dict[l.get('number')] = physical_colour(l, model) * (-1) ** (1 + l.get('state'))
    flows = color_basis.get_flow_basis().color_flow_decomposition(repr_dict, ninitial)
    return ids, [[tuple(flow[l.get('number')]) for l in plegs] for flow in flows]


# ------------------------------------------------ physical FKS bookkeeping --
def physical_pdgs(legs):
    """PDG codes of the physical particles (the Fock-state PDG for an onium)."""
    return [onium_of(l)['id'] if onium_of(l) else l.get('id') for l in physical_legs(legs)]


def to_physical_fks(real_legs, model, fks_infos, fks_j_from_i, colors, particle_tags, charges):
    """Translate the FKS information of a real-emission process with bound
    states from constituent numbering (the Python FKS objects) to physical
    numbering (everything written to Fortran, nexternal counting an onium once).
    Returns (fks_infos, fks_j_from_i, colors, particle_tags, charges)."""
    import copy
    legs = list(real_legs)
    pairs = onium_pairs(legs)
    hidden = set(b for _, b in pairs)
    constituents = hidden | set(a for a, _ in pairs)
    plegs = [l for pos, l in enumerate(legs) if pos not in hidden]
    new_infos = []
    for info in fks_infos:
        orig_i, orig_j = info['i'], info['j']
        if orig_i - 1 in constituents or orig_j - 1 in constituents:
            raise OniaNLOError('a constituent of a bound state as FKS parton i or j')
        info = copy.copy(info)
        info['i'] = physical_number(legs, orig_i)
        info['j'] = physical_number(legs, orig_j)
        # ij is a number of the Born (the real with i removed): drop the second
        # constituents of the Born before it
        born = [l for l in legs if l.get('number') != orig_i]
        bhidden = set(b for _, b in onium_pairs(born))
        info['ij'] = info['ij'] - sum(1 for b in bhidden if b < info['ij'] - 1)
        ub = info.get('underlying_born')
        if ub is not None and len(ub) == len(born) and all(isinstance(x, int) for x in ub):
            info['underlying_born'] = physical_pdgs(born)
        new_infos.append(info)
    new_j = {}
    for k, js in fks_j_from_i.items():
        if k - 1 in hidden:
            continue
        new_j[physical_number(legs, k)] = sorted(
            set(physical_number(legs, j) for j in js if j - 1 not in hidden))
    new_colors = [physical_colour(l, model) for l in plegs]
    new_tags = [particle_tags[pos] for pos in range(len(legs)) if pos not in hidden]
    # the charge of a bound state is that of its two constituents together
    second = dict(pairs)
    new_charges = [charges[pos] + (charges[second[pos]] if pos in second else 0)
                   for pos in range(len(legs)) if pos not in hidden]
    return new_infos, new_j, new_colors, new_tags, new_charges


def physical_process(proc):
    """A copy of a process with bound states whose legs are the physical
    particles: one leg per onium (its Fock-state PDG, its onium dictionary),
    numbered 1..n. For the bookkeeping of the exporter (identities,
    statistical factors), never for diagram generation."""
    import copy
    import madgraph.core.base_objects as MG
    legs = proc.get_legs_with_decays()
    if not has_onia(legs):
        return proc
    new = copy.copy(proc)
    plegs = MG.LegList()
    for n, leg in enumerate(physical_legs(legs), 1):
        leg = copy.copy(leg)
        if onium_of(leg):
            dict.__setitem__(leg, 'id', onium_of(leg)['id'])
        dict.__setitem__(leg, 'number', n)
        plegs.append(leg)
    dict.__setitem__(new, 'legs', plegs)
    dict.__setitem__(new, 'legs_with_decays', plegs)
    return new


# ---------------------------------------------------------- output record --
class OniaRecord(object):
    """The bound states of one P* directory of an NLO output, as recorded in
    its onia_fks.json, with the interface of a matrix element that the LO
    onia exporter (export_v4.export_onia_files, get_onia_info) reads:
    get_nonia(), get('processes') -> processes -> get('legs') -> legs with
    get('onium'), get('state'), get('id'), get('number')."""

    class _Item(dict):
        def get(self, key, default=None):
            return dict.get(self, key, default)

    def __init__(self, legs):
        self.legs = [self._Item(l, onium=self._Item(l['onium']) if l['onium'] else {})
                     for l in legs]

    @staticmethod
    def write(born_me, path):
        import json
        legs = born_me.get('processes')[0].get('legs')
        rows = [dict(id=l.get('id'), state=l.get('state'), number=l.get('number'),
                     onium=dict(l.get('onium')) if l.get('onium') else {}) for l in legs]
        with open(path, 'w') as f:
            json.dump(rows, f)

    @classmethod
    def load_all(cls, subprocesses):
        import glob
        import json
        import os
        return [cls(json.load(open(p))) for p in
                sorted(glob.glob(os.path.join(subprocesses, 'P*', 'onia_fks.json')))]

    def get_nonia(self):
        return len(set(l['onium']['index'] for l in self.legs if l['onium']))

    def get(self, key):
        if key == 'processes':
            return [self._Item(legs=self.legs)]
        raise KeyError(key)

    def constituents(self):
        """{onium pdg: [constituent pdgs]}"""
        out = {}
        for l in self.legs:
            if l['onium']:
                out.setdefault(abs(l['onium']['id']), []).append(l['id'])
        return out


# ------------------------------------------- P-wave soft colour transitions --
# The soft current of a P-wave quarkonium (arXiv:2402.19221, sect. 3.1) links
# the P-wave Born to the S-wave Borns of the same diagrams with the onium in
# the colour C' of aux_states, through the effective generators
#   Q_eff(18)^a = (2+r+1/r)/sqrt(2 N_c) Delta^a     (P[1] -> S[8], Delta^a_ob = delta_ab)
#   Q_eff(81)^a = (2+r+1/r)/sqrt(2 N_c) Delta^aT    (P[8] -> S[1])
#   Q_eff(88)^a = (2+r+1/r) D^a/2 + (1/r-r) T^a/2   (P[8] -> S[8], D^a_bc = d_abc,
#                                                     T^a_bc = -i f_abc)
# r = m_Q/m_Qbar. The colour part of <P| T_k^a Q_eff^a |S'> is exact
# (transition_matrices, without the mass factors and 1/sqrt(2N_c), applied at
# run time). Colour charges (eq. 3.2): t^a for a final quark or an initial
# antiquark, -t^aT for a final antiquark or an initial quark, T^a for an octet.

def aux_color_basis(amplitude, onium_index, colour):
    """The colour basis of the diagrams of amplitude with the onium of index
    onium_index projected onto colour (1 or 8) instead of its own."""
    import copy
    import madgraph.core.color_amp as color_amp
    import madgraph.core.diagram_generation as diagram_generation
    amp = diagram_generation.Amplitude()
    process = copy.copy(amplitude.get('process'))
    legs = copy.deepcopy(process.get('legs'))
    for leg in legs:
        onium = onium_of(leg)
        if onium and onium['index'] == onium_index:
            onium = dict(onium)
            onium['C'] = colour
            leg.set('onium', onium)
    process.set('legs', legs)
    amp.set('process', process)
    amp.set('diagrams', amplitude.get('diagrams'))
    basis = color_amp.ColorBasis()
    basis.create_color_dict_list(amp)
    basis.build()
    return basis


def leg_colour_rep(leg, model):
    """The representation of the colour charge of a leg: 3 (t^a), -3 (-t^aT),
    8 (T^a = -i f), 0 colourless. An initial quark counts as an antiquark."""
    colour = physical_colour(leg, model) if onium_of(leg) else \
        model.get('particle_dict')[leg.get('id')].get_color()
    if colour in (1, 0):
        return 0
    if colour == 8:
        return 8
    return colour if leg.get('state') else -colour


def colour_charge_string(rep, num, a, fresh):
    """The colour charge Q^a of a leg of representation rep and colour index
    num, acting on a colour string whose index num is renamed fresh."""
    import fractions
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
    raise OniaNLOError('no colour charge for representation %s' % rep)


def contract(struct_p, struct_s, repl_p, repl_s, insertion, Nc=3):
    """sum over colours of conj(c_p) [insertion] c_s, with the indices of the
    two immutable colour strings renamed by repl_p, repl_s first. Returns
    (re, im) as Fractions."""
    import fractions
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


def transition_matrices(born_amplitude, model, basis_p=None):
    """The colour transition matrices of a Born with one P-wave quarkonium, for
    its soft current: {'onium': physical number of the onium, 'basis_p',
    'keys_p', 'aux': [ {'colour': C', 'label': '18'|'81'|'88', 'basis',
    'keys', 'matrices': {(k, kind): {(alpha, beta): (re, im)}}} ]} with
        X^(k)_{alpha beta} = sum_colours conj(c_P_alpha) Q_k^a Q_eff^a c_S_beta
    for every coloured physical leg k (the onium too when it is an octet:
    then Q_k = T^a acts on the P-wave index, after Q_eff), kind 'Delta' for
    [18] and [81], 'D' (d_abc) and 'T' (-i f_abc) for [88]; the mass factors
    (2+r+1/r)/sqrt(2N_c), (2+r+1/r)/2 and (1/r-r)/2 are left out. alpha (beta)
    runs over the sorted colour basis of the P (S) wave."""
    import madgraph.core.color_amp as color_amp
    import madgraph.core.color_algebra as color_algebra
    process = born_amplitude.get('process')
    legs = process.get('legs')
    pwaves = [l for l in physical_legs(legs) if onium_of(l) and is_pwave(onium_of(l))
              and is_quarkonium(l, model)]
    if len(pwaves) != 1:
        raise OniaNLOError('transition_matrices needs exactly one P-wave quarkonium')
    pleg = pwaves[0]
    onium = onium_of(pleg)
    n_q = pleg.get('number')                 # the octet label of the onium
    if basis_p is None:
        basis_p = color_amp.ColorBasis()
        basis_p.create_color_dict_list(born_amplitude)
        basis_p.build()
    keys_p = sorted(basis_p.keys())
    a, fresh, c_new, m_new = -4001, -4002, -4003, -4004

    def qeff(kind, b, c):
        """(Q_eff^a)_{bc} of [88] without its mass factor."""
        if kind == 'D':
            return color_algebra.ColorString([color_algebra.d(a, b, c)])
        return colour_charge_string(8, b, a, c)      # -i f(a, b, c)

    result = {'onium': physical_number(legs, n_q), 'basis_p': basis_p, 'keys_p': keys_p,
              'aux': []}
    for colour, label in aux_states(onium):
        basis_s = aux_color_basis(born_amplitude, onium['index'], colour)
        keys_s = sorted(basis_s.keys())
        matrices = {}
        for leg in physical_legs(legs):
            rep = leg_colour_rep(leg, model)
            if rep == 0:
                continue
            num = leg.get('number')
            k = physical_number(legs, num)
            kinds = ['Delta'] if label in ('18', '81') else ['D', 'T']
            for kind in kinds:
                repl_p, repl_s = {}, {}
                if num != n_q:
                    # Q_k^a on the index of leg k (of the S wave)
                    repl_s[num] = fresh
                    insertion = colour_charge_string(rep, num, a, fresh)
                    if label == '18':
                        repl_s[n_q] = a          # Delta^a: S octet index = a
                    elif label == '81':
                        repl_p[n_q] = a          # Delta^aT: P octet index = a
                    else:
                        repl_s[n_q] = c_new
                        insertion.product(qeff(kind, n_q, c_new))
                else:
                    # the onium (a P-wave octet): T^a_{b' m} Q_eff^a_{m c}
                    if label == '81':
                        continue                 # T^a_{b' a} = -i f(a b' a) = 0
                    repl_s[n_q] = c_new
                    insertion = colour_charge_string(8, n_q, a, m_new)
                    insertion.product(qeff(kind, m_new, c_new))
                mat = {}
                for ib, ks in enumerate(keys_s):
                    for ia, kp in enumerate(keys_p):
                        value = contract(kp, ks, repl_p, repl_s, insertion)
                        if value != (0, 0):
                            mat[(ia, ib)] = value
                matrices[(k, kind)] = mat
        result['aux'].append({'colour': colour, 'label': label, 'basis': basis_s,
                              'keys': keys_s, 'matrices': matrices})
    return result


def colour_matrix(basis):
    """The colour matrix sum_colours conj(c_i) c_j of a colour basis, on its
    sorted keys, as {(i, j): (re, im)} Fractions (zeros left out)."""
    import fractions
    import madgraph.core.color_amp as color_amp
    out = {}
    if not len(basis):
        return out
    matrix = color_amp.ColorMatrix(basis)
    n = len(basis.keys())
    for i in range(n):
        for j in range(n):
            re, im = matrix.col_matrix_fixed_Nc[(i, j)]
            value = (fractions.Fraction(re), fractions.Fraction(im))
            if value != (0, 0):
                out[(i, j)] = value
    return out


def soft_current_data(born_me):
    """What the generated born_onia.f needs for the soft current of the
    P-wave quarkonium of a Born matrix element (None without one): the
    physical position of the onium, its S, J and index, the constituent
    numbers of the quark and the antiquark, and per S-wave state S' of
    aux_states its colour, label, colour basis, colour matrix and transition
    matrices (transition_matrices, P basis = the colour basis of born_me)."""
    amplitude = born_me.get('base_amplitude')
    process = amplitude.get('process')
    model = process.get('model')
    legs = process.get('legs')
    pwaves = [l for l in physical_legs(legs) if onium_of(l) and is_pwave(onium_of(l))
              and is_quarkonium(l, model)]
    if not pwaves:
        return None
    tm = transition_matrices(amplitude, model, basis_p=born_me.get('color_basis'))
    onium = onium_of(pwaves[0])
    constituents = [l for l in legs if onium_of(l) and onium_of(l)['index'] == onium['index']]
    quark = [l for l in constituents if l.get('id') > 0][0]
    antiquark = [l for l in constituents if l.get('id') < 0][0]
    for aux in tm['aux']:
        aux['cf'] = colour_matrix(aux['basis'])
    return dict(onium=tm['onium'], index=onium['index'], S=onium['S'], J=onium['J'],
                C=onium['C'], quark=quark.get('number'), antiquark=antiquark.get('number'),
                ncolor_p=len(tm['keys_p']), aux=tm['aux'])
