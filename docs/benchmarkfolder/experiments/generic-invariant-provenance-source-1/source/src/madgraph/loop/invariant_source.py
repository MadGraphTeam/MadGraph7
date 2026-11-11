"""Narrow fixed-form momentum reader. Unsupported source is never certified."""
import re

def require(ok, reason):
    if not ok: raise ValueError(reason)

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


