#!/usr/bin/env python3
"""Source-only, explicitly versioned QP mass transport for generated MadLoop.

Original routines and reducer ABIs are retained. No compiler or reducer runs.
"""
import re


def logical_blocks(text):
    """Yield exact fixed-form line spans and their joined statement text."""
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines):
        start = i
        line = lines[i]
        if not line.strip() or line[0] in 'Cc*!':
            yield start, i+1, None
            i += 1
            continue
        if line[72:].strip():
            raise ValueError('Unqualified executable text beyond fixed-form column72')
        statement = line[6:72].strip()
        i += 1
        while i < len(lines) and len(lines[i]) > 5 and lines[i][0] not in 'Cc*!' and lines[i][5] not in ' 0\n':
            if lines[i][72:].strip():
                raise ValueError('Unqualified continuation text beyond fixed-form column72')
            statement += ' ' + lines[i][6:72].strip()
            i += 1
        yield start, i, statement


def split_args(text):
    level = 0
    start = 0
    result = []
    for i, char in enumerate(text):
        if char == '(':
            level += 1
        elif char == ')':
            level -= 1
            assert level >= 0
        elif char == ',' and level == 0:
            result.append(text[start:i].strip())
            start = i+1
    assert level == 0
    result.append(text[start:].strip())
    return result


def fixed(statement):
    result = []
    rest = statement.strip()
    while len(rest) > 66:
        cut = max(rest.rfind(',', 0, 66)+1, rest.rfind(' ', 0, 67))
        assert cut > 0, 'Cannot wrap fixed-form token: '+rest
        result.append(('      ' if not result else '     $ ')+rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    result.append(('      ' if not result else '     $ ')+rest)
    assert all(len(line) <= 73 for line in result)
    # Continuations have one extra separator; reserve one more column there.
    if any(len(line) > 72 for line in result):
        return fixed_narrow(statement)
    return '\n'.join(result)+'\n'


def fixed_narrow(statement):
    result = []
    rest = statement.strip()
    while len(rest) > 65:
        cut = max(rest.rfind(',', 0, 65)+1, rest.rfind(' ', 0, 66))
        assert cut > 0
        result.append(('      ' if not result else '     $ ')+rest[:cut].rstrip())
        rest = rest[cut:].lstrip()
    result.append(('      ' if not result else '     $ ')+rest)
    assert all(len(line) <= 72 for line in result)
    return '\n'.join(result)+'\n'


def replace_statements(text, callback):
    lines = text.splitlines(keepends=True)
    result = []
    for first, last, statement in logical_blocks(text):
        replacement = callback(statement) if statement is not None else None
        result.append(''.join(lines[first:last]) if replacement is None else replacement)
    return ''.join(result)


def routines(text):
    lines = text.splitlines(keepends=True)
    found = {}
    start = name = header = None
    for first, last, statement in logical_blocks(text):
        if statement is None:
            continue
        match = re.fullmatch(r'SUBROUTINE\s+(\w+)\s*\((.*)\)', statement, re.I)
        if match:
            assert name is None
            start, name, header = first, match[1].upper(), split_args(match[2])
        elif statement.upper() == 'END' and name is not None:
            assert name not in found
            found[name] = {'text': ''.join(lines[start:last]), 'args': header}
            start = name = header = None
    assert name is None
    return found


def qp_symbols(text, zero='MP__ZERO'):
    result = {zero: 'REAL*16 local zero'}
    for _, _, statement in logical_blocks(text):
        if statement is None:
            continue
        match = re.fullmatch(r'(REAL\*16|COMPLEX\*32)\s+(.+)', statement, re.I)
        if match:
            for token in split_args(match[2]):
                name = token.split('(')[0].strip().upper()
                result[name] = match[1].upper()
    return result


def mass_actual(dp_actual, declarations, mp_prefix):
    match = re.fullmatch(r'DCMPLX\(\s*([A-Z][A-Z0-9_]*)\s*\)', dp_actual, re.I)
    if not match:
        raise ValueError('Unqualified model mass expression: '+dp_actual)
    symbol = match[1].upper()
    mp_symbol = mp_prefix+symbol
    if mp_symbol not in declarations:
        raise ValueError('Missing genuine MP model symbol: '+mp_symbol)
    return 'CMPLX('+mp_symbol+',KIND=16)', symbol, mp_symbol


def loop_variants(ct_text, prefix='ML5_0_'):
    original = routines(ct_text)
    selected = {name: routine for name, routine in original.items()
                if re.fullmatch(re.escape(prefix)+r'LOOP_\d+(?:_\d+)?', name)}
    clones = []
    for name, routine in selected.items():
        n = int(name[len(prefix+'LOOP_'):].split('_')[0])
        args = routine['args']
        mass_formals = ['M'+str(i) for i in range(1,n+1)]
        assert all(args.count(formal) == 1 for formal in mass_formals)
        pairs = [a for old in args for a in ([old, 'MP_'+old] if old in mass_formals else [old])]
        text = routine['text']
        def change(statement):
            if statement.upper().startswith('SUBROUTINE '+name+'('):
                return fixed('SUBROUTINE '+name+'_QPMASS('+','.join(pairs)+')')
            if statement.upper().startswith('COMPLEX*16 '+mass_formals[0]+',') or statement.upper() == 'COMPLEX*16 '+mass_formals[0]:
                return fixed(statement)+fixed('COMPLEX*32 '+','.join('MP_'+a for a in mass_formals))
            if statement.upper() == 'COMPLEX*16 M2L(NLOOPLINE)':
                return fixed(statement)+fixed('COMPLEX*32 MP_M2L(NLOOPLINE)')
            if statement.upper() == 'DOING_QP = (CTMODE.GE.4)':
                # Preserve all original DP execution and legacy non8/9 paths.
                call = fixed('CALL '+name+'('+','.join(args)+')')
                return (fixed(statement)+fixed('IF (.NOT.DOING_QP) THEN')+call+
                        fixed('RETURN')+fixed('ENDIF')+
                        fixed('IF (MLREDUCTIONLIB(LIBINDEX).NE.8.AND.MLREDUCTIONLIB(LIBINDEX).NE.9) THEN')+
                        call+fixed('RETURN')+fixed('ENDIF'))
            match = re.fullmatch(r'M2L\((\d+)\)=M(\d+)\*\*2', statement, re.I)
            if match:
                i, source = map(int, match.groups())
                assert source == (n if i == 1 else i-1)
                # This code is reachable only after the QP-only delegation guard.
                return fixed(statement)+fixed('MP_M2L(%d)=MP_M%d**2'%(i,source))
            if 'DIMAG(M2L(I))' in statement.upper():
                return fixed(statement.replace('DIMAG(M2L(I))','AIMAG(MP_M2L(I))')
                             .replace('ABS(M2L(I))','ABS(MP_M2L(I))')
                             .replace('0D0','0.0E0_16').replace('1D-2','1.0E-2_16').replace('1D-15','1.0E-15_16'))
            for adapter in ('MP_NINJARS_LOOP','MP_TIRUST_LOOP'):
                if statement.upper().startswith('CALL '+prefix+adapter+'('):
                    assert ',MP_PL,M2L,RANK' in statement.replace(' ','')
                    return fixed(statement.replace(prefix+adapter+'(',prefix+adapter+'_QPMASS(')
                                 .replace(',MP_PL,M2L,RANK',',MP_PL,MP_M2L,RANK'))
            return None
        clone = replace_statements(text, change)
        assert clone.count('MP_M2L(') >= n+1
        clones.append(clone)
    assert selected
    return selected, '\n'.join(clones)


def qp_adapter_clone(text, name, inner=None):
    original = routines(text)[name]
    seen = {'header':0,'formal':0,'inner':0}
    def change(statement):
        if statement.upper().startswith('SUBROUTINE '+name+'('):
            seen['header'] += 1
            return fixed(statement.replace(name+'(', name+'_QPMASS(',1))
        if statement.upper() == 'COMPLEX*16 M2L(NLOOPLINE)':
            seen['formal'] += 1
            return fixed('COMPLEX*32 M2L(NLOOPLINE)')
        if inner and statement.upper().startswith('CALL '+inner+'('):
            seen['inner'] += 1
            return fixed(statement.replace(inner+'(',inner+'_QPMASS(',1))
        return None
    clone = replace_statements(original['text'], change)
    assert seen == {'header':1,'formal':1,'inner':int(inner is not None)}, seen
    return clone


def transform_calls(text, loop_routines, declarations, mp_prefix='MP__'):
    rows = []
    def change(statement):
        match = re.fullmatch(r'CALL\s+(\w+)\s*\((.*)\)',statement,re.I)
        if not match or match[1].upper() not in loop_routines:
            return None
        name = match[1].upper()
        actuals = split_args(match[2])
        formals = loop_routines[name]['args']
        assert len(actuals) == len(formals), (name,actuals,formals)
        result = []
        masses = []
        for formal, actual in zip(formals,actuals):
            result.append(actual)
            if re.fullmatch(r'M\d+',formal):
                qp, symbol, mp_symbol = mass_actual(actual,declarations,mp_prefix)
                result.append(qp)
                masses.append({'formal':formal,'original_dp_actual':actual,'model_symbol':symbol,
                               'mp_symbol':mp_symbol,'mp_declaration':declarations[mp_symbol],'new_qp_actual':qp})
        row = {'entrypoint':name,'new_entrypoint':name+'_QPMASS','loop_group':int(actuals[-1]),
               'original_actuals':actuals,'new_actuals':result,'mass_pairs':masses,
               'ordered_mp_square_symbols':[masses[-1]['mp_symbol']]+[m['mp_symbol'] for m in masses[:-1]]}
        rows.append(row)
        return fixed('CALL '+name+'_QPMASS('+','.join(result)+')')
    result = replace_statements(text,change)
    return result, rows
