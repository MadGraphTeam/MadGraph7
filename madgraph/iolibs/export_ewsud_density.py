"""Export complex EW Sudakov amplitudes and their spin interferences.

Each auxiliary amplitude retains its own colour basis. Rectangular colour
metrics contract it with the physical Born; no projection onto the Born
amplitude, helicity averaging, or inversion of a colour metric is required.
"""

import os
import re

from madgraph import MadGraph5Error
from madgraph.core import color_amp

# Couplings, masses and widths in HELAS calls, e.g. GC_11 and MDL_MW.
_AMPLITUDE_INPUT = re.compile(r'\b(GC_\w+|MDL_\w+)\b', re.I)


def _real(value):
    return ('%.17e' % float(value)).replace('e', 'd')


def _complex(real, imag=0):
    return '(%s,%s)' % (_real(real), _real(imag))


def colour_matrix(cache, basis1, basis2):
    """ColorMatrix of two colour bases, computed once per pair in cache.

    The scalar Sudakov writer and the density kernel share one cache per
    process, so the colour algebra of each base/source pair is done once. The
    cache keeps the bases alive, so their ids cannot be reused while it lives.
    """
    key = (id(basis1), id(basis2))
    if key not in cache:
        cache[key] = (basis1, basis2, color_amp.ColorMatrix(basis1, basis2))
    return cache[key][2]


def amplitude_inputs(calls):
    """Couplings, masses and widths that HELAS calls use, in first-use order."""
    names = []
    for call in calls:
        for name in _AMPLITUDE_INPUT.findall(call):
            if name.upper() not in names:
                names.append(name.upper())
    return names


def sudakov_term_expressions(auxiliary, helicities):
    """(accumulator, Fortran expression) of each non-diagonal Sudakov term.

    auxiliary is a non-Goldstone entry of sudakov_matrix_elements and
    helicities the Fortran helicity array. The scalar Sudakov wrapper and the
    density kernel both take their calls from here.
    """
    legs = [l['number'] for l in auxiliary['legs']]
    old, new = auxiliary['pdgs']
    arguments = 'pdglist,%s,iflist,invariants' % helicities
    kind = auxiliary['type']
    if kind == 'cew':
        changed = '%d,%d,%d' % (legs[0], old[0], new[0])
        return [('LSC', 'get_lsc_nondiag(%s,%s)' % (arguments, changed)),
                ('XXC', 'get_xxc_nondiag(%s,%s)' % (arguments, changed))]
    if kind == 'iz1':
        return [('SSC', 'get_ssc_n_nondiag_1(%s,%d,%d,%d)' % (
            arguments, legs[0], old[0], new[0]))]
    if kind == 'iz2':
        return [('SSC', 'get_ssc_n_nondiag_2(%s,%d,%d,%d,%d,%d,%d)' % (
            arguments, legs[0], old[0], new[0], legs[1], old[1], new[1]))]
    if kind == 'ipm2':
        return [('SSC', 'get_ssc_c(%d,%d,pdglist,%d,%d,%s,iflist,invariants)' % (
            legs[0], legs[1], new[0], new[1], helicities))]
    raise MadGraph5Error('Unknown Sudakov term %s' % kind)


class DensityExporter:
    def __init__(self, exporter, matrix_element, fortran_model, color_matrices=None):
        self.exporter = exporter
        self.color_matrices = {} if color_matrices is None else color_matrices
        self.fks = matrix_element
        self.model = fortran_model
        self.born = matrix_element.born_me
        self.gold = [m for m in matrix_element.sudakov_matrix_elements
                     if m['type'] == 'goldstone']
        self.other = [m for m in matrix_element.sudakov_matrix_elements
                      if m['type'] != 'goldstone']
        self.bases = [self.born] + [m['matrix_element'] for m in self.gold]
        self.mes = self.bases + [m['matrix_element'] for m in self.other]
        self.nb = len(self.bases)
        self.ns = len(self.mes)
        # 1-based base amplitude of every source; each base is its own source.
        self.source_base = list(range(1, self.nb + 1)) + [
            self.base_index(m) for m in self.other]
        self.nc = max(max(1, len(m['color_basis'])) for m in self.mes)
        self.split_orders = self.born['processes'][0]['split_orders']
        self.orders = []
        for me in self.mes:
            for order, unused in me.get_split_orders_mapping()[1]:
                if order not in self.orders:
                    self.orders.append(order)
        self.no = len(self.orders)
        self.target = tuple(self.born['processes'][0]['born_sq_orders'][o]
                            for o in self.split_orders)
        self.constants = ('integer NC,NO,NS,NB\n'
                          'parameter (NC=%d,NO=%d,NS=%d,NB=%d)\n' %
                          (self.nc, self.no, self.ns, self.nb))

    def write(self, writer):
        path = os.path.join(os.path.dirname(__file__), 'template_files',
                            'density_interference.inc')
        with open(path) as stream:
            template = stream.read()
        legs = self.born['processes'][0]['legs']
        gmu = 'Gmu' in self.exporter.model.get('name')
        calls = [self.model.get_matrix_element_calls(me) for me in self.mes]
        inputs = [amplitude_inputs(me_calls) for me_calls in calls[:self.nb]]
        replace = dict(constants=self.constants,
                       ncomb=self.born.get_helicity_combinations(),
                       polarised=','.join('.true.' if l.get('polarization')
                                          else '.false.' for l in legs),
                       ninputs=max([1] + [len(names) for names in inputs]),
                       base_inputs=self.base_inputs(inputs),
                       helicity_lines=self.exporter.get_helicity_lines(self.born),
                       den_factor_lines='\n'.join(
                           self.exporter.get_den_factor_lines(self.fks)),
                       iflist=','.join('1' if l['state'] else '-1' for l in legs),
                       base_selection=self.base_selection(),
                       source_selection=self.source_selection(),
                       source_bases=','.join(str(b) for b in self.source_base),
                       coefficient_calls=self.coefficient_calls(),
                       metric_data=self.metric_data(),
                       order_mask=','.join(
                           '1' if tuple(a + b for a, b in zip(x, y)) == self.target
                           else '0' for y in self.orders for x in self.orders),
                       first_parameter='MDL_GF' if gmu else 'AEWM1',
                       par_ren='GET_PAR_REN_GMU' if gmu else 'GET_PAR_REN_ALPHAMZ')
        writer.writelines(template % replace)
        for i, (me, me_calls) in enumerate(zip(self.mes, calls), 1):
            writer.writelines(self.raw_amplitude(i, me, me_calls))

    def base_index(self, auxiliary):
        if not auxiliary['base_amp']:
            return 1
        base = self.fks.sudakov_matrix_elements[auxiliary['base_amp'] - 1]['matrix_element']
        return next(i for i, me in enumerate(self.bases, 1) if me is base)

    def base_selection(self):
        """Select the base amplitude of each helicity state.

        A state with longitudinal vectors uses the Goldstone amplitude with
        exactly those legs replaced. A missing Goldstone tree amplitude is a
        zero amplitude (ibase=0), not permission to use the transverse
        operator on a longitudinal state. The scalar wrapper differs here: it
        takes the first Goldstone amplitude whose legs are all longitudinal,
        or else the Born with its Born-based operators.
        """
        lines = ['ibase=1', 'phase=(1d0,0d0)']
        legs = self.born['processes'][0]['legs']
        vectors = [l for l in legs if abs(l['id']) in (23, 24)]
        for leg in vectors:
            pos = leg['number']
            # The Sudakov SM UFO uses M(V_L,out) = -i M(G,out).
            # Crossing the external vector conjugates this phase. Individual
            # W+h, W-h and Zh checks fix what WW's product alone cannot.
            phase = '(0d0,-1d0)' if leg['state'] else '(0d0,1d0)'
            lines += ['if (hels(%d).eq.0) then' % pos,
                      'ibase=0', 'phase=phase*%s' % phase, 'endif']
        for i, gold in enumerate(self.gold, 2):
            positions = {l['number'] for l in gold['legs']}
            condition = '.and.'.join('hels(%d).%s.0' % (
                l['number'], 'eq' if l['number'] in positions else 'ne')
                for l in vectors)
            lines += ['if (%s) ibase=%d' % (condition, i)]
        return '\n'.join(lines)

    def source_selection(self):
        lines = []
        for i in range(1, self.ns + 1):
            lines += [('if' if i == 1 else 'else if') +
                      ' (isource.eq.%d) then' % i,
                      'call sdk_density_raw_%d(p,hels,jamp)' % i]
        return '\n'.join(lines + ['endif'])

    def base_inputs(self, inputs):
        """Copy the couplings, masses and widths of each base into values."""
        lines = []
        for b, names in enumerate(inputs, 1):
            lines.append(('if' if b == 1 else 'else if') + ' (ibase.eq.%d) then' % b)
            lines += ['values(%d)=%s' % (k, name) for k, name in enumerate(names, 1)]
        return '\n'.join(lines + ['endif'])

    def coefficient_calls(self):
        lines = []
        for b, me in enumerate(self.bases, 1):
            lines += [('if' if b == 1 else 'else if') +
                      ' (ibase.eq.%d) then' % b,
                      'pdglist=(/%s/)' % ','.join(str(l['id']) for l in
                                                me['processes'][0]['legs']),
                      'coeff(%d)=get_lsc_diag(pdglist,hels,iflist,invariants)' % b,
                      'coeff(%d)=coeff(%d)+get_ssc_n_diag(pdglist,hels,iflist,invariants)' % (b, b),
                      'coeff(%d)=coeff(%d)+get_xxc_diag(pdglist,hels,iflist,invariants)' % (b, b)]
            lines += ['coeff(%d)=%s' % (i, '+'.join(
                          expression for unused, expression in
                          sudakov_term_expressions(aux, 'hels')))
                      for i, aux in enumerate(self.other, self.nb + 1)
                      if self.source_base[i - 1] == b]
        return '\n'.join(lines + ['endif'])

    def metric_data(self):
        """Colour metrics of the pairs sdk_density_inter is called with.

        The Born contracts with every source, and a Goldstone base with its
        own sources; the metrics of other pairs are never read.
        """
        lines = []
        for b, base in enumerate(self.bases, 1):
            for s, source in enumerate(self.mes, 1):
                if b != 1 and self.source_base[s - 1] != b:
                    continue
                matrix = colour_matrix(self.color_matrices, base['color_basis'],
                                       source['color_basis'])
                values = matrix.col_matrix_fixed_Nc
                if not matrix:
                    values = {(0, 0): (1, 0)}
                for j in range(self.nc):
                    row = [_complex(*values.get((i, j), (0, 0)))
                           for i in range(self.nc)]
                    lines.append('data (metric(i,%d,%d,%d),i=1,NC)/%s/' %
                                 (j + 1, s, b, ','.join(row)))
        return '\n'.join(lines)

    def raw_amplitude(self, index, me, calls):
        """Raw colour amplitudes, without spin/colour/symmetry divisors."""
        amp_orders = me.get_split_orders_mapping()[1]
        jamp, ntemp = self.exporter.get_JAMP_lines_split_order(
            me, amp_orders, split_order_names=self.split_orders)
        nlegs = len(me['processes'][0]['legs'])
        ncolor = max(1, len(me['color_basis']))
        lines = ['subroutine sdk_density_raw_%d(pin,nhel,result)' % index,
                 'use aloha_object', 'implicit none', "include 'nexternal.inc'",
                 "include 'coupl.inc'", self.constants,
                 'double precision pin(0:3,nexternal-1),p(0:3,nexternal-1)',
                 'double precision presh(0:3,nexternal-1)',
                 'integer nhel(nexternal-1),ic(nexternal-1)',
                 'data ic /%d*1/' % nlegs,
                 'integer flavor(nexternal-1)',
                 'data flavor /%d*1/' % nlegs,
                 'double precision zero', 'parameter (zero=0d0)',
                 'double complex imag1', 'parameter (imag1=(0d0,1d0))',
                 'double complex result(NC,NO)',
                 'double complex amp(%d),jamp(%d,%d),tmp_jamp(%d)' % (
                     me.get_number_of_amplitudes(), ncolor, len(amp_orders), ntemp),
                 'type(aloha) w(%d)' % me.get_number_of_wavefunctions(),
                 # JAMP lines skip colour flows without amplitudes at an order.
                 'logical ok', 'result=(0d0,0d0)', 'jamp=(0d0,0d0)', 'p=pin']
        if index > self.nb:
            aux = self.other[index - self.nb - 1]
            legs = [l['number'] for l in aux['legs']]
            old, new = [list(x) for x in aux['pdgs']]
            for vector in (legs, old, new):
                vector.extend([0] * (2 - len(vector)))
            lines += ['call reshuffle_momenta(p,presh,(/%s/),(/%s/),(/%s/),ok)' %
                      tuple(','.join(str(x) for x in v) for v in (legs, old, new)),
                      # Match the scalar Sudakov prescription: an auxiliary
                      # process below its reshuffling threshold contributes
                      # zero, while the physical Born remains available.
                      'if (.not.ok) return',
                      'p=presh']
        lines += calls + jamp
        for local, (order, unused) in enumerate(amp_orders, 1):
            lines.append('result(1:%d,%d)=jamp(:,%d)' %
                         (ncolor, self.orders.index(order) + 1, local))
        return '\n'.join(lines + ['end', ''])
