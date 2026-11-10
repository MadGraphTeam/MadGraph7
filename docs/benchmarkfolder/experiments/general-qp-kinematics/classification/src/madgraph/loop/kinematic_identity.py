"""Private graph-derived kinematic identities; no invariant or shell arithmetic.

Class codes: 0 unknown, 1 literal single leg, 2 conservation complement,
3 composite, 4 closure-only.  A known identity is not an on-shell certificate.
The supported momentum convention is ordinary scalar-export SM HELAS.  An
unsupported graph produces explicit unknown rows, never a guessed shell.
"""
from types import SimpleNamespace
from aloha.aloha_writers import WriteALOHA


class UnsupportedIdentity(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise UnsupportedIdentity(message)


def plus(vectors, size):
    return tuple(sum(v[i] for v in vectors) for i in range(size))


def classify(vector):
    """Classify literal incidence, retaining the exact conservation witness.

    For a single/complement, vector = sign*e_leg + closure*(1,...,1).
    No floating-point coordinates or model mass values enter this function.
    """
    size = len(vector)
    for i in range(size):
        for sign in (-1, 1):
            if all(value == (sign if j == i else 0)
                   for j, value in enumerate(vector)):
                return {'class': 1, 'leg': i + 1, 'sign': sign, 'closure': 0}
    if vector and len(set(vector)) == 1:
        return {'class': 4, 'leg': 0, 'sign': 0, 'closure': vector[0]}
    for i in range(size):
        others = {value for j, value in enumerate(vector) if j != i}
        if len(others) == 1:
            closure = others.pop()
            if closure and vector[i] - closure in (-1, 1):
                return {'class': 2, 'leg': i + 1,
                        'sign': vector[i] - closure, 'closure': closure}
    return {'class': 3, 'leg': 0, 'sign': 0, 'closure': 0}


def direction(corners, reverse=False):
    size, n = len(corners[0]), len(corners)
    if reverse and n > 2:
        corners = [tuple(-v for v in row) for row in reversed(corners)]
        mass_order = [1] + list(range(n, 1, -1))
    else:
        corners = list(corners)
        mass_order = list(range(1, n + 1))
    # Adapter q1 is the literal sum, not an assumed zero closure vector.
    shifts = [plus(corners, size)] + [plus(corners[:k], size) for k in range(1, n)]
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            vector = tuple(a - b for a, b in zip(shifts[i], shifts[j]))
            pairs.append({'i': i + 1, 'j': j + 1, 'vector': list(vector),
                          **classify(vector)})
    return {'corners': [list(v) for v in corners],
            'shifts': [list(v) for v in shifts], 'mass_order': mass_order,
            'pairs': pairs}


class Incidence:
    def __init__(self, process):
        self.model = process.get('model')
        self.legs = list(process.get('legs'))
        self.size = len(self.legs)
        require([leg.get('number') for leg in self.legs] == list(range(1, self.size + 1)),
                'external leg numbering is not contiguous')
        require(not process.get('is_decay_chain'), 'decay-chain export is outside prototype scope')
        self.k_signs = tuple(-1 if leg.get('state') else 1 for leg in self.legs)
        self.memo, self.active = {}, set()

    def wavefunction(self, wf):
        key = id(wf)
        if key in self.memo:
            return self.memo[key]
        require(key not in self.active, 'cyclic HELAS ancestry')
        require(not wf.get('is_loop'), 'loop mother leaked into external corner')
        self.active.add(key)
        try:
            mothers = wf.get('mothers')
            if not mothers:
                leg = wf.get('number_external')
                require(1 <= leg <= self.size, 'external wavefunction has unknown leg')
                require(wf.get('spin') in (1, 2, 3), 'external spin outside scalar SM export scope')
                state_id = wf.get_external_helas_call_dict()['state_id']
                # ixxxxx writes -NSF*P; oxxxxx/sxxxxx/vxxxxx write +NS*P.
                sign = -state_id if wf.get_spin_state_number() == -2 else state_id
                require(sign == -self.k_signs[leg - 1], 'HELAS external momentum sign differs from crossing')
                vector = tuple(sign if i == leg - 1 else 0 for i in range(self.size))
            else:
                info = wf.get_aloha_info(optimized_output=False)
                require(info is not None, 'intermediate wavefunction lacks ALOHA routing')
                _, tags, outgoing = info
                require(not any(tag.startswith('P') and tag != 'P0' for tag in tags),
                        'custom/projected propagator momentum path outside prototype scope')
                interaction = self.model.get('interaction_dict')[wf.get('interaction_id')]
                spins = [particle.get('spin') for particle in interaction.get('particles')]
                require(len(spins) == len(mothers) + 1 and all(spin in (1, 2, 3) for spin in spins),
                        'unsupported intermediate spin/arity')
                writer = SimpleNamespace(particles=[WriteALOHA.type_to_variable[s] for s in spins],
                                         outgoing=outgoing, tag=tags)
                signs = WriteALOHA.get_momentum_conservation_sign(writer)
                require([value for i, value in enumerate(signs, 1) if i != outgoing]
                        == ['+'] * len(mothers), 'unsupported ALOHA mother momentum signs')
                vector = plus([self.wavefunction(mother) for mother in mothers], self.size)
            # Repeated legs are invalid even if get_struct_external_leg_ids deduplicates them.
            require(all(value in (0, -self.k_signs[i]) for i, value in enumerate(vector)),
                    'repeated leg or unexpected sign in external subtree')
            require({i + 1 for i, value in enumerate(vector) if value}
                    == wf.get_struct_external_leg_ids(), 'ancestry and structural leg sets disagree')
            self.memo[key] = vector
            return vector
        finally:
            self.active.remove(key)

    def amplitude(self, amplitude):
        n = len(amplitude.get('wavefunctions')) - 2
        mothers = list(amplitude.get('mothers'))
        pairing = list(amplitude.get('pairing'))
        require(n > 0 and len(pairing) == n and all(p > 0 for p in pairing)
                and sum(pairing) == len(mothers), 'invalid corner PAIRING')
        corner_vectors, cursor = [], 0
        for count in pairing:
            values = plus([self.wavefunction(wf) for wf in mothers[cursor:cursor + count]], self.size)
            # LOOP_n forms PL = -sum(W%P). Convert physical P to all-incoming k.
            corner_vectors.append(tuple(-value * self.k_signs[i] for i, value in enumerate(values)))
            cursor += count
        require(all(value in (0, 1) for row in corner_vectors for value in row),
                'corner incidence is not a disjoint positive partition')
        require(plus(corner_vectors, self.size) == (1,) * self.size,
                'corners do not partition all external legs')
        denominators = amplitude.get_denominators()
        require(len(denominators) == n, 'denominator/corner count mismatch')
        require([set(legs) for legs, _ in denominators]
                == [{i + 1 for i, value in enumerate(row) if value} for row in corner_vectors],
                'denominator ancestry order differs from emitted mothers/PAIRING')
        masses = list(amplitude.get_masses())
        require(len(masses) == n, 'mass/corner count mismatch')
        return {'n': n, 'corners': corner_vectors, 'mass_arguments': masses,
                'ordered_mass_arguments': masses[-1:] + masses[:-1],
                'mothers': [wf.get('number') for wf in mothers], 'pairing': pairing}


def build(matrix_element, group_loops=True):
    """Derive per-emitted-reduction metadata without editing the HELAS graph.

    Loop groups must already have their final IDs, as when CT calls are emitted.
    Unknown rows preserve an explicit diagnostic reason and enable legacy fallback.
    """
    process = matrix_element.get('processes')[0]
    grouped = bool(group_loops and process.get('has_born'))
    if grouped:
        selected = [(lamps[0].get('loop_group_id') + 1, list(lamps))
                    for _, lamps in matrix_element.get('loop_groups')]
    else:
        selected = [(lamp.get('number'), [lamp])
                    for diag in matrix_element.get_loop_diagrams()
                    for lamp in diag.get_loop_amplitudes()]
    ids = [item[0] for item in selected]
    require(len(ids) == len(set(ids)) and all(i > 0 for i in ids), 'ambiguous emitted reduction IDs')
    legs = list(process.get('legs'))
    external = [{'leg': leg.get('number'), 'pdg': leg.get('id'), 'outgoing': leg.get('state'),
                 'offshell': bool(leg.get('offshell')),
                 'mass_symbol': process.get('model').get_particle(leg.get('id')).get('mass')}
                for leg in legs]
    try:
        incidence = Incidence(process)
        process_reason = None
    except UnsupportedIdentity as exc:
        incidence, process_reason = None, str(exc)
    records = []
    for reduction_id, members in selected:
        ref = members[0]
        record = {'reduction_id': reduction_id, 'reference_amplitude': ref.get('number'),
                  'member_amplitudes': [amp.get('number') for amp in members],
                  'n': len(ref.get('wavefunctions')) - 2, 'known': False}
        try:
            require(incidence is not None, process_reason)
            first = incidence.amplitude(ref)
            for member in members[1:]:
                other = incidence.amplitude(member)
                require((first['n'], first['corners'], first['mass_arguments'])
                        == (other['n'], other['corners'], other['mass_arguments']),
                        'group members have inconsistent corner/mass routing')
            record.update(first)
            record['corners'] = [list(v) for v in first['corners']]
            record['known'] = True
            record['directions'] = {'primary': direction(first['corners']),
                                    'reverse': direction(first['corners'], True)}
            for mode in record['directions'].values():
                for pair in mode['pairs']:
                    pair['declared_onshell'] = bool(pair['leg'] and not external[pair['leg'] - 1]['offshell'])
        except UnsupportedIdentity as exc:
            record['reason'] = str(exc)
        records.append(record)
    return {'schema': 1, 'grouped': grouped, 'has_born': bool(process.get('has_born')),
            'external': external, 'basis': 'k_i = +P_i incoming, -P_i outgoing',
            'scope': 'classification only; no threshold, shell assignment or acceptance change',
            'reductions': records}


def render(table, prefix):
    """Emit an integer/logical-only fixed-form lookup, with safe unknown return."""
    require(prefix and all(c.isalnum() or c == '_' for c in prefix), 'invalid Fortran prefix')
    records = table['reductions']
    count = max([row['reduction_id'] for row in records] + [1])
    offsets, sizes, valid = [0] * count, [0] * count, [0] * count
    columns = {name: [] for name in ('classes', 'legs', 'signs', 'closures', 'shells')}
    for row in records:
        index = row['reduction_id'] - 1
        sizes[index] = row['n']
        if not row['known']:
            continue
        valid[index] = 1
        offsets[index] = len(columns['classes'])
        for mode in ('primary', 'reverse'):
            for pair in row['directions'][mode]['pairs']:
                for key, field in [('classes', 'class'), ('legs', 'leg'), ('signs', 'sign'),
                                   ('closures', 'closure'), ('shells', 'declared_onshell')]:
                    columns[key].append(int(pair[field]))
    length = max(len(columns['classes']), 1)
    arrays = {'GOFF': offsets, 'GNUM': sizes, 'GVALID': valid,
              **{name.upper(): (values or [0]) for name, values in columns.items()}}
    lines = [f'      SUBROUTINE {prefix}KINEMATIC_IDENTITY(G,CT,N,I,J,',
             '     $ CLASS,LEG,SGN,CLOSURE,DECLARED_ONSHELL,KNOWN)',
             '      IMPLICIT NONE',
             '      INTEGER G,CT,N,I,J,CLASS,LEG,SGN,CLOSURE',
             '      LOGICAL DECLARED_ONSHELL,KNOWN',
             '      INTEGER A,B,K,DIR,NP,ORI,L',
             f'      INTEGER GOFF({count}),GNUM({count}),GVALID({count})',
             f'      INTEGER CLASSES({length}),LEGS({length}),SIGNS({length})',
             f'      INTEGER CLOSURES({length}),SHELLS({length})',
             'C     Identity only: declared on-shell is not a numerical certificate.']
    for name, values in arrays.items():
        for start in range(0, len(values), 8):
            chunk = values[start:start + 8]
            lines += [f'      DATA ({name}(L),L={start + 1},{start + len(chunk)}) /',
                      '     $ ' + ','.join(map(str, chunk)) + '/']
    lines += ['      CLASS=0', '      LEG=0', '      SGN=0', '      CLOSURE=0',
              '      DECLARED_ONSHELL=.FALSE.', '      KNOWN=.FALSE.',
              f'      IF(G.LT.1.OR.G.GT.{count}) RETURN',
              '      IF(GVALID(G).NE.1.OR.N.NE.GNUM(G)) RETURN',
              '      IF(I.LT.1.OR.J.LT.1.OR.I.GT.N.OR.J.GT.N) RETURN',
              '      IF(I.EQ.J) RETURN', '      DIR=0',
              '      IF(CT.EQ.2.OR.CT.EQ.5) DIR=1',
              '      IF(CT.NE.1.AND.CT.NE.2.AND.CT.NE.4.AND.CT.NE.5)',
              '     $ RETURN', '      A=MIN(I,J)', '      B=MAX(I,J)',
              '      ORI=1', '      IF(I.GT.J) ORI=-1', '      NP=N*(N-1)/2',
              '      K=GOFF(G)+DIR*NP+(A-1)*N-(A-1)*A/2+B-A',
              '      CLASS=CLASSES(K)', '      LEG=LEGS(K)',
              '      SGN=ORI*SIGNS(K)', '      CLOSURE=ORI*CLOSURES(K)',
              '      DECLARED_ONSHELL=SHELLS(K).EQ.1', '      KNOWN=.TRUE.', '      END']
    require(all(len(line) <= 72 for line in lines if not line.startswith('C')), 'fixed-form line overflow')
    return '\n'.join(lines) + '\n'
