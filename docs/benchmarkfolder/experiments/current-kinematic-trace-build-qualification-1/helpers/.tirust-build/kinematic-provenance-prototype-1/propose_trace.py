"""Build PRIVATE, UNCOMPILED insertion-only patches of two saved exports.

Never mutates source inputs. Removing explicitly marked insertions must
recover each source byte for byte. The patch adds integer-bit I/O only;
no arithmetic expression, threshold test, mass, or reducer guard is replaced.
"""
from pathlib import Path
import difflib
import hashlib
import json
import re
import sys

from saved_source import Sources, consume, statements, require

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BEGIN = 'C     BEGIN PRIVATE_KTRACE_V1\n'
END = 'C     END PRIVATE_KTRACE_V1\n'


def marked(code):
    require(all(len(line) <= 72 for line in code.splitlines()), 'trace fixed-form line too long')
    return BEGIN + code.rstrip('\n') + '\n' + END


def strip_trace(source):
    pattern = re.escape(BEGIN) + r'.*?' + re.escape(END)
    stripped = re.sub(pattern, '', source, flags=re.S)
    require(BEGIN not in stripped and END not in stripped, 'unbalanced trace markers')
    return stripped


def replace_once(source, anchor, replacement):
    require(source.count(anchor) == 1, 'trace anchor not unique: ' + anchor[:80])
    return source.replace(anchor, replacement, 1)


def declarations(prefix, matrix=False, loop_id=False):
    code = ('      INTEGER*8 KT_SERIAL\n'
            '      INTEGER KT_UNIT,KT_POINT\n'
            '      LOGICAL KT_ENABLED\n'
            f'      COMMON/{prefix}PRIVATE_KTRACE/KT_SERIAL,KT_UNIT,\n'
            '     $ KT_POINT,KT_ENABLED\n')
    if matrix or loop_id:
        code += ('      INTEGER KT_ID,KT_SQSO,KT_R\n'
                 f'      COMMON/{prefix}LOOP/KT_ID,KT_SQSO,KT_R\n')
    if matrix:
        code += ('      INTEGER KT_MODE\n'
                 '      REAL*8 KT_LSCALE\n'
                 f'      COMMON/{prefix}CT/KT_LSCALE,KT_MODE\n')
    return marked(code)


def write_bits(tag, payloads, identifier='KT_ID', mode='KT_MODE', rank='KT_R',
               i='0', j='0', branch='0'):
    # Header: point, reduction serial, emitted ID, mode, rank, N, i, j, branch.
    lines = ['      IF(KT_ENABLED) THEN',
             "        WRITE(KT_UNIT,'(A,9(1X,I0),*(1X,Z16.16))')",
             f"     $ '{tag}',KT_POINT,KT_SERIAL,{identifier},{mode},{rank},",
             f'     $ NLOOPLINE,{i},{j},{branch},']
    for n, value in enumerate(payloads):
        lines.append('     $ ' + value + (',' if n < len(payloads)-1 else ''))
    lines.append('      ENDIF')
    return marked('\n'.join(lines))


def routine_spans(source):
    """Locate only actual generated subroutine spans, retaining original bytes."""
    lines = source.splitlines(keepends=True)
    offsets, total = [], 0
    for line in lines:
        offsets.append(total); total += len(line)
    current, start, out = None, None, []
    for n, line in enumerate(lines):
        match = re.match(r'^      SUBROUTINE ([A-Z0-9_]+)\(', line)
        if match:
            require(current is None, 'nested subroutine')
            current, start = match[1], offsets[n]
        if current and line.strip() == 'END':
            out.append((current, start, offsets[n] + len(line)))
            current = None
    require(current is None, 'unterminated subroutine')
    return out


def insert_declarations(body, declaration):
    anchor = 'C     ----------\nC     BEGIN CODE\nC     ----------\n'
    if anchor in body:
        return replace_once(body, anchor, declaration + anchor)
    # TIRust starts execution directly after its parameter include.
    anchor = "      INCLUDE 'MadLoopParams.inc'\n"
    return replace_once(body, anchor, anchor + declaration)


def instrument_loop(body, prefix):
    body = insert_declarations(body, declarations(prefix) + marked('      INTEGER KT_EXT'))
    event = marked('      IF(KT_ENABLED) KT_SERIAL=KT_SERIAL+1_8')
    event += write_bits('KT_LOOP_DP',
                        ['TRANSFER(PL,0_8,4*NLOOPLINE)',
                         'TRANSFER(M2L,0_8,2*NLOOPLINE)'],
                        identifier='LOOPNUM', mode='CTMODE', rank='RANK')
    event += marked('      IF(KT_ENABLED) THEN\n      DO KT_EXT=1,NEXTERNAL')
    event += write_bits('KT_EXTERNAL_DP', ['TRANSFER(W(KT_EXT)%P,0_8,8)'],
                        identifier='LOOPNUM', mode='CTMODE', rank='RANK', i='KT_EXT')
    event += marked('      ENDDO\n      ENDIF')
    event += marked('      IF(DOING_QP) THEN')
    event += write_bits('KT_LOOP_QP', ['TRANSFER(MP_PL,0_8,8*NLOOPLINE)'],
                        identifier='LOOPNUM', mode='CTMODE', rank='RANK')
    event += marked('      IF(KT_ENABLED) THEN\n      DO KT_EXT=1,NEXTERNAL')
    event += write_bits('KT_EXTERNAL_QP', ['TRANSFER(MP_W(KT_EXT)%P,0_8,16)'],
                        identifier='LOOPNUM', mode='CTMODE', rank='RANK', i='KT_EXT')
    event += marked('      ENDDO\n      ENDIF')
    event += marked('      ENDIF')
    anchor = '        COMPLEX_MASS=.FALSE.\n'
    require(anchor in body, 'loop corner-construction boundary absent')
    return replace_once(body, anchor, event + anchor)


def instrument_matrix(body, prefix, qp):
    body = insert_declarations(body, declarations(prefix, matrix=True))
    chunks = 4 if qp else 2
    suffix = 'QP' if qp else 'DP'
    entry = write_bits('KT_MATRIX_' + suffix,
                       [f'TRANSFER(P_LOOP,0_8,{4*(2 if qp else 1)}*NLOOPLINE)',
                        f'TRANSFER(M2L,0_8,{chunks}*NLOOPLINE)'])
    anchor = '      DO I=1,NLOOPLINE\n'
    body = replace_once(body, anchor, entry + anchor)
    anchor = '            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)\n'
    body = replace_once(body, anchor, anchor + write_bits('KT_RAW_' + suffix,
        [f'TRANSFER(DIFFSQ,0_8,{chunks})', f'TRANSFER(S_MAT(I,J),0_8,{chunks})'], i='I', j='J'))
    for branch, label, assignment in (
            (1, 'MASSI', '                S_MAT(I,J)=-M2L(J)\n'),
            (2, 'MASSJ', '                S_MAT(I,J)=-M2L(I)\n'),
            (3, 'ZERO', '                S_MAT(I,J)=-(M2L(I)+M2L(J))\n')):
        before = write_bits('KT_' + label + '_BEFORE_' + suffix,
                            [f'TRANSFER(S_MAT(I,J),0_8,{chunks})'],
                            i='I', j='J', branch=str(branch))
        after = write_bits('KT_' + label + '_AFTER_' + suffix,
                           [f'TRANSFER(S_MAT(I,J),0_8,{chunks})'],
                           i='I', j='J', branch=str(branch))
        body = replace_once(body, assignment, before + assignment + after)
    anchor = '            REF_NORMALIZATION = (REF_NORMALIZATION/' + ('2.0E0_16' if qp else '2.0D0') + ')**2\n'
    body = replace_once(body, anchor, anchor + write_bits('KT_SCALE_' + suffix,
        [f'TRANSFER(REF_NORMALIZATION,0_8,{2 if qp else 1})',
         'TRANSFER(OSTHRES,0_8,1)'], i='I', j='J'))
    anchor = '          ENDIF\n\n        ENDDO\n'
    body = replace_once(body, anchor, '          ENDIF\n' +
                        write_bits('KT_FINAL_' + suffix,
                                   [f'TRANSFER(S_MAT(I,J),0_8,{chunks})'], i='I', j='J') +
                        '\n        ENDDO\n')
    return body


def instrument_tirust(body, prefix, qp):
    body = insert_declarations(body, declarations(prefix, loop_id=True))
    width = 2 if qp else 1
    snapshot = write_bits('KT_TIRUST_' + ('QP' if qp else 'DP'),
        [f'TRANSFER(SHIFTS,0_8,{4*width}*NLOOPLINE)',
         f'TRANSFER(MASSES,0_8,{2*width}*NLOOPLINE)',
         f'TRANSFER(INVARIANTS,0_8,{width}*NLOOPLINE*NLOOPLINE)',
         f'TRANSFER(SCALE,0_8,{width})'], mode='CTMODE', rank='RANK')
    anchor = '      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS\n'
    return replace_once(body, anchor, snapshot + anchor)


def control_routines(prefix):
    # Appended to CT_interface's object: no additional makefile or source file.
    # The future private driver must open the chosen unit and configure this
    # before each saved point. Disabled initialization makes other use a no-op.
    declarations_plain = declarations(prefix).removeprefix(BEGIN).removesuffix(END)
    code = (f'      SUBROUTINE {prefix}PRIVATE_KTRACE_SET(IUNIT,IPOINT,ON)\n'
            '      IMPLICIT NONE\n'
            '      INTEGER IUNIT,IPOINT\n'
            '      LOGICAL ON\n' + declarations_plain +
            '      KT_ENABLED=ON\n'
            '      KT_UNIT=IUNIT\n'
            '      KT_POINT=IPOINT\n'
            '      KT_SERIAL=0_8\n'
            '      IF(KT_ENABLED) THEN\n'
            "        WRITE(KT_UNIT,'(A,5(1X,I0))') 'KT_SCHEMA_V1',\n"
            '     $ KT_POINT,STORAGE_SIZE(0D0),STORAGE_SIZE(0.0_16),\n'
            '     $ BIT_SIZE(0_8),STORAGE_SIZE(.TRUE.)\n'
            '      ENDIF\n'
            '      END\n'
            f'      BLOCK DATA {prefix}PRIVATE_KTRACE_INIT\n' +
            declarations_plain +
            '      DATA KT_ENABLED/.FALSE./,KT_UNIT/0/,KT_POINT/0/\n'
            '      DATA KT_SERIAL/0_8/\n'
            '      END\n')
    return marked(code)


def instrument(source, filename):
    require(BEGIN not in source, 'source already instrumented')
    out, cursor, counts = [], 0, {'loops': 0, 'matrix_dp': 0, 'matrix_qp': 0,
                                'tensor_dp': 0, 'tensor_qp': 0}
    prefixes = set()
    for name, start, end in routine_spans(source):
        body = source[start:end]
        m = re.fullmatch(r'(ML5_[0-9]+_)(LOOP_[1-6](?:_[0-9]+)?|(?:MP_)?BUILD_KINEMATIC_MATRIX|(?:MP_)?TIRUST_TENSOR_DIAGNOSTIC)', name)
        if m:
            prefix, role = m.groups(); prefixes.add(prefix)
            if role.startswith('LOOP_'):
                body = instrument_loop(body, prefix); counts['loops'] += 1
            elif role.endswith('BUILD_KINEMATIC_MATRIX'):
                qp = role.startswith('MP_')
                body = instrument_matrix(body, prefix, qp)
                counts['matrix_qp' if qp else 'matrix_dp'] += 1
            else:
                qp = role.startswith('MP_')
                body = instrument_tirust(body, prefix, qp)
                counts['tensor_qp' if qp else 'tensor_dp'] += 1
        out.extend((source[cursor:start], body)); cursor = end
    out.append(source[cursor:])
    require(len(prefixes) == 1, 'ambiguous process prefix')
    prefix = prefixes.pop()
    if filename == 'CT_interface.f':
        require(counts['matrix_dp'] == counts['matrix_qp'] == 1 and counts['loops'] > 0,
                'missing CT trace boundaries')
        out.append(control_routines(prefix))
    elif filename == 'TIRUST_interface.f':
        require(counts['tensor_dp'] == counts['tensor_qp'] == 1, 'missing tensor trace boundaries')
    else:
        raise ValueError('unsupported trace source')
    result = ''.join(out)
    require(strip_trace(result) == source, 'trace changes original source bytes')
    return result, counts


def propose(expected_digest):
    sources = Sources(ROOT, HERE/'trace-bindings.json', expected_digest)
    routing_sources = Sources(ROOT, HERE/'bindings.json',
        'd6da968f024c6fa2f17e2d201d94ac914c57450174a08c5cb853a16f0b2a9c9c')
    routing = consume(routing_sources)
    for key in sources.binding['cases']:
        external = routing['cases'][key]['external_provenance']
        require(all(e['leg'] == e['wavefunction_slot'] for e in external),
                'trace external slot assumption fails for saved export')
        types = [s for _,s,_ in statements(routing_sources.text(
            routing_sources.binding['cases'][key]['dhelas']+'/aloha_functions.f'))]
        for typename, expected in (('ALOHA','DOUBLECOMPLEX::P(0:3)'),
                                   ('MP_ALOHA','COMPLEX*32::P(0:3)')):
            first, last = types.index('TYPE'+typename), types.index('ENDTYPE'+typename)
            require(expected in types[first:last],
                    'unqualified external momentum storage type')
    artifacts, records = {}, []
    for key, case in sources.binding['cases'].items():
        for filename in ('CT_interface.f', 'TIRUST_interface.f'):
            name = case['subprocess'] + '/' + filename
            original = sources.text(name)
            proposed, counts = instrument(original, filename)
            target = 'trace-proposal/' + key + '/' + filename
            patch = ''.join(difflib.unified_diff(original.splitlines(keepends=True),
                                               proposed.splitlines(keepends=True),
                                               fromfile='a/' + name, tofile='b/' + name))
            artifacts[target] = proposed.encode('ascii')
            artifacts[target + '.patch'] = patch.encode('ascii')
            records.append({'input': name, 'input_sha256': hashlib.sha256(original.encode('ascii')).hexdigest(),
                            'proposed_source': target, 'proposed_sha256': hashlib.sha256(artifacts[target]).hexdigest(),
                            'boundaries': counts, 'strip_insertions_recovers_original_bytes': True})
    report = {'schema': 'private-unbuilt-kinematic-trace-proposal-v1', 'records': records,
              'native_executions': 0, 'compiled': False, 'production_changes': False,
              'qualification': 'Unbuilt review proposal only; integer I/O may perturb execution and needs native differential validation.'}
    artifacts['trace-proposal/derivation.json'] = (json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
    return artifacts, report


if __name__ == '__main__':
    require(len(sys.argv) == 3 and sys.argv[1] in ('--write', '--check'),
            'usage: propose_trace.py --write|--check EXPECTED_TRACE_BINDING_SHA256')
    artifacts, report = propose(sys.argv[2])
    for path, raw in artifacts.items():
        destination = HERE/path
        if sys.argv[1] == '--write':
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(raw)
        else:
            require(destination.read_bytes() == raw, 'trace proposal drift: ' + path)
    print(json.dumps(report,sort_keys=True))
