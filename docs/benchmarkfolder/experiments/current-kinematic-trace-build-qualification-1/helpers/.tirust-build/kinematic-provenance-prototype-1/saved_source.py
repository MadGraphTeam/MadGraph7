"""Narrow PRIVATE consumer of hash-bound saved optimized Fortran sources.

This is not a Fortran interpreter: only explicit external-wavefunction calls,
linear ALOHA momentum assignments and generated LOOP_N calls are admitted.
It executes no generated source, compiler, reducer or amplitude.
"""
import hashlib
import json
from pathlib import Path
import re

from routing import add, negate, route, subtract, conservation_equivalence, require


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Sources:
    def __init__(self, repo, binding_path, expected_binding_sha):
        self.repo = Path(repo).resolve()
        raw = Path(binding_path).read_bytes()
        require(sha(raw) == expected_binding_sha, "binding digest mismatch")
        self.binding = json.loads(raw)
        require(self.binding['schema'] == 'private-saved-routing-inputs-v1', "binding schema")
        self.raw = {}
        self.consumed = set()
        # Every source is hashed before ANY source decoding or parsing. Store
        # the verified bytes, so later changes cannot create a TOCTOU parse.
        for name, record in self.binding['files'].items():
            p = Path(name)
            require(not p.is_absolute() and '..' not in p.parts, 'unsafe input path')
            actual = self.repo / p
            require(actual.resolve().is_relative_to(self.repo), 'input escapes repository')
            data = actual.read_bytes()
            require(len(data) == record['bytes'] and sha(data) == record['sha256'],
                    'source digest mismatch: ' + name)
            self.raw[name] = data

    def text(self, name):
        require(name in self.raw, 'unbound input: ' + name)
        self.consumed.add(name)
        return self.raw[name].decode('ascii')


def statements(text):
    """Return fixed-form statements with original first line and comments.

    Generated call/assignment syntax has no relevant string literals; strings
    elsewhere are retained but never interpreted. Continuations are joined
    without an inserted separator, as fixed-form Fortran permits split names.
    """
    out, code, line_number, comments, pending_comments = [], '', 0, [], []
    for number, raw in enumerate(text.splitlines(), 1):
        if not raw.strip():
            continue
        if raw[0] in 'Cc*!':
            pending_comments.append(raw[6:] if len(raw) >= 6 else '')
            continue
        if raw[0] == '#':
            raise ValueError('preprocessor syntax not supported in saved fixture')
        require(len(raw) >= 6, 'short fixed-form line')
        part = re.sub(r'\s+', '', raw[6:]).upper()
        if raw[5] not in ' 0':
            require(bool(code), 'orphan continuation')
            code += part
        else:
            if code:
                out.append((line_number, code, comments))
            code, line_number, comments = part, number, pending_comments
            pending_comments = []
    if code:
        out.append((line_number, code, comments))
    return out


def args(text):
    if not text:
        return []
    out, level, start = [], 0, 0
    for i, ch in enumerate(text):
        if ch == '(':
            level += 1
        elif ch == ')':
            level -= 1
            require(level >= 0, 'unbalanced arguments')
        elif ch == ',' and level == 0:
            out.append(text[start:i])
            start = i + 1
    require(level == 0, 'unbalanced arguments')
    out.append(text[start:])
    require(all(out), 'empty argument')
    return out


def call(statement):
    m = re.fullmatch(r'CALL([A-Z][A-Z0-9_]*)\((.*)\)', statement)
    return (m[1], args(m[2])) if m else None


def blocks(text):
    result, current = {}, None
    for line, statement, comments in statements(text):
        m = re.fullmatch(r'SUBROUTINE([A-Z][A-Z0-9_]*)\((.*)\)', statement)
        if m:
            require(current is None and m[1] not in result, 'ambiguous subroutine')
            current = m[1]
            result[current] = {'arguments': args(m[2]), 'line': line, 'statements': []}
        elif statement in ('END', 'ENDSUBROUTINE'):
            current = None
        elif current is not None:
            result[current]['statements'].append((line, statement, comments))
    return result


def wf_number(token):
    m = re.fullmatch(r'W\(([1-9][0-9]*)\)', token)
    require(m is not None, 'unsupported wavefunction argument: ' + token)
    return int(m[1])


def external_rule(block, supplied):
    formal = block['arguments']
    require(len(formal) == len(supplied), 'external argument count')
    output, pname = formal[-1], formal[0]
    mapping = dict(zip(formal, supplied))
    matches = []
    for mu in range(4):
        prefix = f'{output}%P({mu})='
        assignments = [s[len(prefix):] for _, s, _ in block['statements'] if s.startswith(prefix)]
        require(len(assignments) == 1, 'external momentum assignment not unique')
        m = re.fullmatch(re.escape(pname) + rf'\({mu}\)\*\(?(-?)([A-Z]+)\)?', assignments[0])
        require(m is not None and mapping.get(m[2]) in ('+1', '-1', '1'),
                'unsupported external momentum sign')
        matches.append((-1 if m[1] else 1) * int(mapping[m[2]]))
    require(len(set(matches)) == 1, 'component-dependent external sign')
    leg = re.fullmatch(r'P\(0,([1-9][0-9]*)\)', supplied[0])
    require(leg is not None, 'unsupported external P argument')
    # SXXXXX has no explicit mass argument. Preserve that absence.
    mass = supplied[1] if len(formal) >= 5 else None
    return int(leg[1]), matches[0], mass


def internal_rule(block, supplied):
    formal = block['arguments']
    require(len(formal) == len(supplied), 'internal argument count')
    output = formal[-1]
    prefix = output + '%P(:)='
    found = [s[len(prefix):] for _, s, _ in block['statements'] if s.startswith(prefix)]
    require(len(found) == 1, 'unsupported/ambiguous internal momentum assignment')
    terms = re.findall(r'([+-])([A-Z][A-Z0-9_]*)%P\(:\)', found[0])
    require(terms and ''.join(s + n + '%P(:)' for s, n in terms) == found[0],
            'nonlinear or unsupported momentum assignment')
    mapping = dict(zip(formal, supplied))
    return [(1 if sign == '+' else -1, wf_number(mapping[name])) for sign, name in terms]


def source_call_order(sources, case, precision):
    p = case['subprocess']
    filename = p + ('/loop_matrix.f' if precision == 'dp' else '/mp_compute_loop_coefs.f')
    found = []
    for _, statement, _ in statements(sources.text(filename)):
        parsed = call(statement)
        if not parsed:
            continue
        name, _ = parsed
        m = re.fullmatch(r'ML5_[0-9]+_(MP_)?((?:HELAS_CALLS_(?:AMPB|UVCT)|COEF_CONSTRUCTION)_[0-9]+)', name)
        if m and bool(m[1]) == (precision == 'qp'):
            candidate = p + '/' + ((m[1] or '') + m[2]).lower() + '.f'
            require(candidate not in found, 'repeated generated source call')
            found.append(candidate)
    require(bool(found), 'no generated wavefunction source order')
    return found


def wavefunctions(sources, case, precision):
    width = case['nexternal']
    external_blocks = blocks(sources.text(case['dhelas'] + '/aloha_functions.f'))
    wfs, external, definitions = {}, {}, {}
    for filename in source_call_order(sources, case, precision):
        for line, statement, _ in statements(sources.text(filename)):
            parsed = call(statement)
            if not parsed or not parsed[1] or re.fullmatch(r'W\([1-9][0-9]*\)', parsed[1][-1]) is None:
                continue
            name, supplied = parsed
            output = wf_number(supplied[-1])
            if re.fullmatch(r'(?:MP_)?[IOVST]XXXXX', name):
                require(name in external_blocks, 'missing external routine')
                leg, sign, mass = external_rule(external_blocks[name], supplied)
                require(1 <= leg <= width, 'external leg range')
                vector = tuple(sign if i == leg else 0 for i in range(1, width + 1))
                item = {'leg': leg, 'stored_helas_sign': sign,
                        'wavefunction_slot': output,
                        'initial_from_momentum_sign': sign == -1,
                        'external_call': name, 'external_call_mass_expression': mass,
                        'model_pdg': None, 'onshell_decay_flag': None,
                        'offshell_object_flag': None,
                        'metadata_scope': 'unarchived Python model flags remain unknown'}
                require(leg not in external or external[leg] == item, 'external metadata conflict')
                external[leg] = item
            else:
                routine_path = case['dhelas'] + '/' + name + '.f'
                library = blocks(sources.text(routine_path))
                require(name in library, 'routine/file mismatch')
                terms = internal_rule(library[name], supplied)
                require(all(i in wfs for _, i in terms), 'use before wavefunction definition')
                vector = add((tuple(s*x for x in wfs[i]) for s, i in terms), width)
            require(output not in wfs or wfs[output] == vector,
                    'reused W slot changes provenance; unsupported fixture')
            wfs[output] = vector
            definitions.setdefault(output, []).append({'source': filename, 'line': line,
                                                      'statement': statement})
    require(sorted(external) == list(range(1, width + 1)), 'missing external legs')
    # Keep generated repair policy separate from missing original model flags.
    repair = {}
    for _, statement, _ in statements(sources.text(case['subprocess'] + '/loop_matrix.f')):
        m = re.fullmatch(r'KEEP_OFFSHELL_MASS\(([1-9][0-9]*)\)=\.(TRUE|FALSE)\.', statement)
        if m:
            leg, value = int(m[1]), m[2] == 'TRUE'
            require(leg not in repair or repair[leg] == value, 'conflicting repair flags')
            repair[leg] = value
    pmass = {}
    for _, statement, _ in statements(sources.text(case['subprocess'] + '/pmass.inc')):
        m = re.fullmatch(r'PMASS\(([1-9][0-9]*)\)=(.*)', statement)
        require(m is not None, 'unsupported PMASS statement')
        pmass[int(m[1])] = m[2]
    for leg, item in external.items():
        item['generated_keep_offshell_mass'] = repair.get(leg)
        item['pmass_expression'] = pmass.get(leg)
    return wfs, external, definitions


def loop_comment(comments):
    start = next((i for i, s in enumerate(comments) if 'CutTools call for loop' in s), None)
    require(start is not None, 'missing emitted loop identity comment')
    # Generated comments can wrap in the middle of a decimal ID: concatenate.
    joined = ''.join(s.strip() for s in comments[start:])
    m = re.fullmatch(r'CutTools call for loop (numbers|#)\s*([0-9,]+)', joined)
    require(m is not None, 'unsupported emitted loop identity comment: ' + joined)
    numbers = tuple(int(x) for x in m[2].split(','))
    require(len(numbers) == len(set(numbers)), 'duplicate source loop identity')
    # The grouped comment lists lamp.number; ungrouped '# N' comes from
    # ldiag.number. The final LOOP argument independently holds lamp.number.
    return ('amplitude' if m[1] == 'numbers' else 'diagram'), numbers


def reductions(sources, case, wfs, external):
    width = case['nexternal']
    conservation = tuple(-external[i]['stored_helas_sign'] for i in range(1, width + 1))
    wrappers = blocks(sources.text(case['subprocess'] + '/CT_interface.f'))
    out = []
    for filename in case['loop_call_sources']:
        for line, statement, comments in statements(sources.text(filename)):
            parsed = call(statement)
            if not parsed:
                continue
            name, supplied = parsed
            match = re.fullmatch(r'ML5_[0-9]+_LOOP_([1-6])(?:_([1-9][0-9]*))?', name)
            if not match:
                continue
            n = int(match[1]); nw = int(match[2] or match[1])
            cursor = n if match[2] else 0
            pairing = tuple(map(int, supplied[:n])) if match[2] else (1,) * n
            require(all(x > 0 for x in pairing) and sum(pairing) == nw, 'invalid corner pairing')
            require(len(supplied) == cursor + nw + n + 3, 'LOOP argument count')
            mothers = tuple(map(int, supplied[cursor:cursor+nw])); cursor += nw
            masses = tuple(supplied[cursor:cursor+n]); cursor += n
            rank, split_order, identifier = supplied[cursor:]
            require(split_order == 'I_SO', 'unsupported split-order selector')
            comment_kind, numbers = loop_comment(comments)
            identity_kind = 'group' if comment_kind == 'amplitude' else 'amplitude'
            require(name in wrappers, 'missing generated LOOP wrapper')
            wrapper = wrappers[name]
            require(len(wrapper['arguments']) == len(supplied), 'wrapper argument count')
            # Verify actual generated mass rotation rather than assume labels.
            formal_mass = wrapper['arguments'][len(wrapper['arguments'])-n-3:-3]
            expected = [f'M2L({i+1})={formal_mass[(i-1)%n]}**2' for i in range(n)]
            actual = [s for _, s, _ in wrapper['statements'] if re.match(r'M2L\([0-9]+\)=', s)]
            require(actual == expected, 'unsupported generated mass mapping')
            require(any(s == 'PL(I,J)=PL(I,J)-W(WE(K))%P(I)' for _, s, _ in wrapper['statements']),
                    'unsupported corner sign in wrapper')
            require(any(s == 'MP_PL(I,J)=MP_PL(I,J)-MP_W(WE(K))%P(I)' for _, s, _ in wrapper['statements']),
                    'unsupported QP corner sign in wrapper')
            corner_mothers, corners, offset = [], [], 0
            for count in pairing:
                indexes = mothers[offset:offset+count]; offset += count
                require(all(i in wfs for i in indexes), 'missing loop mother')
                corner_mothers.append(indexes)
                corners.append(negate(add((wfs[i] for i in indexes), width)))
            require(add(corners, width) == conservation, 'corners do not cover physical external signs')
            require(all(c[k] in (0, conservation[k]) for c in corners for k in range(width)),
                    'corner contains duplicate/cancelled leaves; not an admitted partition')
            modes = {str(mode): route(corners, masses, mode) for mode in (1, 2, 4, 5)}
            pairs = []
            q = modes['1']['shifts']
            for i in range(n):
                for j in range(i+1, n):
                    d = subtract(q[i], q[j])
                    equivalents = []
                    for leg in range(1, width+1):
                        for sign in (-1, 1):
                            unit = tuple(sign if k == leg else 0 for k in range(1, width+1))
                            k = conservation_equivalence(d, unit, conservation)
                            if k is not None:
                                equivalents.append({'leg': leg, 'sign': sign, 'residual_multiple': k,
                                                    'is_exact_symbolic_singleton': k == 0})
                    pairs.append({'pair': [i+1, j+1], 'difference_coefficients': d,
                                  'external_leg_equivalences': equivalents})
            out.append({'source': filename, 'line': line, 'statement': statement,
                        'reduction_identifier': int(identifier), 'identity_kind': identity_kind,
                        'source_comment_numbers': numbers,
                        'source_comment_number_kind': comment_kind, 'rank': int(rank),
                        'corner_wavefunctions': corner_mothers, 'raw_mass_expressions': masses,
                        'conservation_coefficients': conservation, 'modes': modes, 'primary_pairs': pairs})
    require(out and len({r['reduction_identifier'] for r in out}) == len(out), 'missing/duplicate reductions')
    return out


def consume(sources):
    cases = {}
    for key, case in sources.binding['cases'].items():
        dp_w, dp_ext, dp_def = wavefunctions(sources, case, 'dp')
        qp_w, qp_ext, qp_def = wavefunctions(sources, case, 'qp')
        require(dp_w == qp_w, 'DP/QP symbolic wavefunctions differ')
        for leg in dp_ext:
            for field in dp_ext[leg]:
                if field == 'external_call':
                    require(qp_ext[leg][field] == 'MP_' + dp_ext[leg][field], 'external precision mismatch')
                else:
                    require(dp_ext[leg][field] == qp_ext[leg][field], 'DP/QP metadata mismatch')
        records = reductions(sources, case, dp_w, dp_ext)
        cases[key] = {'scope': 'historical saved generated source; no native routing qualification',
                      'external_provenance': list(dp_ext.values()),
                      'wavefunction_coefficients': {str(k): v for k,v in dp_w.items()},
                      'dp_definition_sources': dp_def, 'qp_definition_sources': qp_def,
                      'dp_qp_symbolic_match': True, 'reductions': records}
    return {'schema': 'private-saved-routing-symbolic-v1', 'native_executions': 0,
            'production_changes': False, 'cases': cases,
            'consumed_sources': sorted(sources.consumed),
            'limits': ['Conservation equivalences do not assert zero numerical residual.',
                       'Singleton provenance is not a physical on-shell predicate.',
                       'Actual rounding/order and snapshots require native trace validation.',
                       'No unknown Python model metadata is inferred from generated repair flags.',
                       'Private prototype only; no threshold, mass, or guard change.']}
