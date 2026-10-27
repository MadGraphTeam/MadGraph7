"""Export complex EW Sudakov amplitudes and their spin interferences.

Each auxiliary amplitude retains its own colour basis. Rectangular colour
metrics contract it with the physical Born; no projection onto the Born
amplitude, helicity averaging, or inversion of a colour metric is required.
"""

import os

from madgraph import MadGraph5Error
from madgraph.core import color_amp


def _real(value):
    return ('%.17e' % float(value)).replace('e', 'd')


def _complex(real, imag=0):
    return '(%s,%s)' % (_real(real), _real(imag))


class DensityExporter:
    def __init__(self, exporter, matrix_element, fortran_model,
                 aloha_objects=False):
        self.exporter = exporter
        self.fks = matrix_element
        self.model = fortran_model
        self.aloha_objects = aloha_objects
        self.born = matrix_element.born_me
        self.gold = [m for m in matrix_element.sudakov_matrix_elements
                     if m['type'] == 'goldstone']
        self.other = [m for m in matrix_element.sudakov_matrix_elements
                      if m['type'] != 'goldstone']
        self.bases = [self.born] + [m['matrix_element'] for m in self.gold]
        self.mes = self.bases + [m['matrix_element'] for m in self.other]
        self.nb = len(self.bases)
        self.ns = len(self.mes)
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
        replace = dict(constants=self.constants,
                       supported_initial=('.false.' if any(
                           not l['state'] and abs(l['id']) in (23, 24)
                           for l in self.born['processes'][0]['legs']) else '.true.'),
                       ncomb=self.born.get_helicity_combinations(),
                       helicity_lines=self.exporter.get_helicity_lines(self.born),
                       den_factor_lines='\n'.join(
                           self.exporter.get_den_factor_lines(self.fks)),
                       iflist=','.join('1' if l['state'] else '-1'
                                      for l in self.born['processes'][0]['legs']),
                       base_selection=self.base_selection(),
                       source_selection=self.source_selection(),
                       source_bases=','.join(str(i) for i in
                           list(range(1, self.nb + 1)) + [
                               self.base_index(m) for m in self.other]),
                       coefficient_calls=self.coefficient_calls(),
                       metric_data=self.metric_data(),
                       order_mask=','.join(
                           '1' if tuple(a + b for a, b in zip(x, y)) == self.target
                           else '0' for y in self.orders for x in self.orders),
                       first_parameter=('MDL_GF' if 'Gmu' in
                                        self.exporter.model.get('name') else 'AEWM1'),
                       par_ren=('GET_PAR_REN_GMU' if 'Gmu' in
                                self.exporter.model.get('name') else
                                'GET_PAR_REN_ALPHAMZ'))
        writer.writelines(template % replace)
        for i, me in enumerate(self.mes, 1):
            writer.writelines(self.raw_amplitude(i, me))

    def base_index(self, auxiliary):
        if not auxiliary['base_amp']:
            return 1
        base = self.fks.sudakov_matrix_elements[auxiliary['base_amp'] - 1]['matrix_element']
        return next(i for i, me in enumerate(self.bases, 1) if me is base)

    def base_selection(self):
        """Select precisely the set of longitudinal legs, as the scalar kernel.

        A missing Goldstone tree amplitude is a zero amplitude, not permission
        to use the transverse operator on a longitudinal state.
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
                      'call sdk_density_raw_%d(p,hels,jamp,ok)' % i]
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
            for i, aux in enumerate(self.other, self.nb + 1):
                # base_amp indexes the original Sudakov list, with 0 for Born.
                original_base = self.born if not aux['base_amp'] else (
                    self.fks.sudakov_matrix_elements[aux['base_amp'] - 1]['matrix_element'])
                if original_base is not me:
                    continue
                nums = [l['number'] for l in aux['legs']]
                old, new = aux['pdgs']
                if aux['type'] == 'cew':
                    args = '%d,%d,%d' % (nums[0], old[0], new[0])
                    expression = ('get_lsc_nondiag(pdglist,hels,iflist,invariants,%s)'
                                  '+get_xxc_nondiag(pdglist,hels,iflist,invariants,%s)' % (args, args))
                elif aux['type'] == 'iz1':
                    expression = 'get_ssc_n_nondiag_1(pdglist,hels,iflist,invariants,%d,%d,%d)' % (nums[0], old[0], new[0])
                elif aux['type'] == 'iz2':
                    expression = 'get_ssc_n_nondiag_2(pdglist,hels,iflist,invariants,%d,%d,%d,%d,%d,%d)' % (nums[0], old[0], new[0], nums[1], old[1], new[1])
                elif aux['type'] == 'ipm2':
                    expression = 'get_ssc_c(%d,%d,pdglist,%d,%d,hels,iflist,invariants)' % (nums[0], nums[1], new[0], new[1])
                else:
                    raise MadGraph5Error('Unknown density Sudakov term %s' % aux['type'])
                lines.append('coeff(%d)=%s' % (i, expression))
        return '\n'.join(lines + ['endif'])

    def metric_data(self):
        lines = []
        for b, base in enumerate(self.bases, 1):
            for s, source in enumerate(self.mes, 1):
                matrix = color_amp.ColorMatrix(base['color_basis'], source['color_basis'])
                values = matrix.col_matrix_fixed_Nc
                if not matrix:
                    values = {(0, 0): (1, 0)}
                for j in range(self.nc):
                    row = [_complex(*values.get((i, j), (0, 0)))
                           for i in range(self.nc)]
                    lines.append('data (metric(i,%d,%d,%d),i=1,NC)/%s/' %
                                 (j + 1, s, b, ','.join(row)))
        return '\n'.join(lines)

    def raw_amplitude(self, index, me):
        """Raw colour amplitudes, without spin/colour/symmetry divisors."""
        calls = self.model.get_matrix_element_calls(me)
        amp_orders = me.get_split_orders_mapping()[1]
        jamp, ntemp = self.exporter.get_JAMP_lines_split_order(
            me, amp_orders, split_order_names=self.split_orders)
        lines = ['subroutine sdk_density_raw_%d(pin,nhel,result,ok)' % index]
        if self.aloha_objects:
            lines.append('use aloha_object')
        lines += ['implicit none', "include 'nexternal.inc'",
                  "include 'coupl.inc'", self.constants,
                  'double precision pin(0:3,nexternal-1),p(0:3,nexternal-1)',
                  'double precision presh(0:3,nexternal-1)',
                  'integer nhel(nexternal-1),ic(nexternal-1)',
                  'data ic /%d*1/' % (len(me['processes'][0]['legs'])),
                  'integer flavor(nexternal-1)',
                  'data flavor /%d*1/' % (len(me['processes'][0]['legs'])),
                  'double precision zero', 'parameter (zero=0d0)',
                  'double complex imag1', 'parameter (imag1=(0d0,1d0))',
                  'double complex result(NC,NO)',
                  'double complex amp(%d),jamp(%d,%d),tmp_jamp(%d)' % (
                      me.get_number_of_amplitudes(), max(1, len(me['color_basis'])),
                      len(amp_orders), ntemp),
                  ('type(aloha) w(%d)' % me.get_number_of_wavefunctions()
                   if self.aloha_objects else 'double complex w(20,%d)' %
                   me.get_number_of_wavefunctions()),
                  'logical ok', 'result=(0d0,0d0)', 'ok=.true.', 'p=pin']
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
                      'if (.not.ok) then', 'ok=.true.', 'return', 'endif',
                      'p=presh']
        lines += calls + jamp
        for local, (order, unused) in enumerate(amp_orders, 1):
            global_order = self.orders.index(order) + 1
            lines.append('result(1:%d,%d)=jamp(:,%d)' %
                         (max(1, len(me['color_basis'])), global_order, local))
        return '\n'.join(lines + ['end', ''])
