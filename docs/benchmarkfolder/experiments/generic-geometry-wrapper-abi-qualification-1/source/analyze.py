"""Strict direct-wrapper ABI protocol, not SLOOPMATRIX lifetime qualification."""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('wrapper_reference_for_analysis', HERE / 'reference.py')
REF = importlib.util.module_from_spec(spec)
spec.loader.exec_module(REF)
MASS = [[0x3FF0000000000000, 0x8000000000000000],
        [0x4000000000000000, 0xBFF0000000000000],
        [0x4008000000000000, 0x3F50000000000000]]


def need(condition, message):
    if not condition:
        raise ValueError(message)


def parse(raw, cases, expected):
    text = raw.decode('ascii') if isinstance(raw, bytes) else raw
    lines = [line.split() for line in text.splitlines() if line.strip()]
    position = 0
    def row(tag, length):
        nonlocal position
        need(position < len(lines), 'missing ' + tag)
        got = lines[position]; position += 1
        need(len(got) == length + 1 and got[0] == tag, 'unexpected/malformed ' + tag)
        return got[1:]
    def ints(tag, length):
        try:
            return list(map(int, row(tag, length)))
        except (ValueError, TypeError) as exc:
            raise ValueError('noninteger ' + tag) from exc
    need(len(cases) == len(expected) == 240, 'frozen corpus size mismatch')
    need(ints('READY', 1) == [240], 'READY count mismatch')
    results = []
    for c, e in zip(cases, expected):
        need(c['id'] == e['id'], 'case/expected join mismatch')
        need(ints('CASE', 4) == [c['id'], c['mode'], c['route'], int(c['fresh'])], 'CASE mismatch')
        try:
            before, after, supplied = [int(x, 16) for x in row('TOKEN', 3)]
        except ValueError as exc:
            raise ValueError('bad TOKEN words') from exc
        need(before == after, 'actual token changed')
        selected = [before, 0, before ^ 1, before ^ (1 << 40)][c['token_mode']]
        need(supplied == selected, 'supplied token selection mismatch')
        need(REF.environment_ok(before) == (c['round_index'] == 0), 'actual rounding/token mismatch')
        pre = ints('PRE', 11); post = ints('POST', 11)
        state = e['flags'] + e['halts'] + [e['round_index']]
        need(pre == state, 'requested IEEE state not established')
        need(post == pre, 'wrapper changed observed IEEE state')
        need(ints('SLOTS', 4) == [int(e['slots_ok']), *e['slots']], 'slot query mismatch')
        for j in range(3):
            need(ints('BOUNDS', 6) == [j + 1, int(e['have'][j]), *e['errors'][j]], 'chord bounds mismatch')
        for j in range(3):
            for i in range(3):
                need(ints('CLASS', 3) == [i + 1, j + 1, e['classes'][i][j]], 'pair class mismatch')
        need(ints('FORWARD', 8) == [1, 1, *c['we'], 2, 0, 1], 'legacy forwarding mismatch')
        for i in range(3):
            got = row('MASS', 3)
            need(int(got[0]) == i + 1 and [int(x, 16) for x in got[1:]] == MASS[i], 'mass word mismatch')
        for tag, field in [('DPMOM', 'dp_words'), ('QPMOM', 'qp_words')]:
            for leg in range(3):
                for mu in range(4):
                    got = row(tag, 4)
                    need(list(map(int, got[:2])) == [leg + 1, mu], 'momentum coordinate mismatch')
                    need([int(x, 16) for x in got[2:]] == c[field][leg][mu], 'momentum storage changed')
        need(int(row('RECORDED_ENV', 1)[0], 16) == supplied, 'dispatch environment transport mismatch')
        need(ints('ENDCASE', 1) == [c['id']], 'ENDCASE mismatch')
        results.append(dict(id=c['id'], mode=c['mode'], label=c['label'],
                            source_bound_chords_available=all(e['have']),
                            token_before=before, token_after=after,
                            ieee_state_unchanged=True, legacy_forwarding=True,
                            momentum_words_unchanged=True))
    need(ints('DONE', 1) == [240], 'DONE count mismatch')
    need(position == len(lines), 'extra trailing protocol')
    have = sum(r['source_bound_chords_available'] for r in results)
    need(have == 180, 'source-bound availability count changed')
    return dict(schema='generic-emitted-wrapper-abi-result-v1', passed=True,
                cases=240, source_bound_available=180, conservative_rejections=60,
                actual_emitted_wrapper_abi=True, scientific_dispatch='recording legacy stub only',
                full_sloop_lifetime_coverage=False, physical_amplitude_qualified=False,
                tensor_or_scientific_evaluations=0, observations=results)


def golden(cases, expected):
    """Pure protocol fixture for parser corruption tests, never native evidence."""
    lines = ['READY 240']
    for c, e in zip(cases, expected):
        csr = 0x1F80; cw = 0x033F
        rounding = {0: 0, 1: 2, 2: 1, 3: 3}[c['round_index']]
        csr |= rounding << 13; cw |= rounding << 10
        if c['halt_flag']:
            flag = c['halt_flag'] - 1
            csr &= ~(1 << [7, 9, 10, 11, 12][flag])
            cw &= ~(1 << [0, 2, 3, 4, 5][flag])
        token = REF.MARK | csr << 16 | cw
        supplied = [token, 0, token ^ 1, token ^ (1 << 40)][c['token_mode']]
        lines.append('CASE ' + ' '.join(map(str, [c['id'], c['mode'], c['route'], int(c['fresh'])])))
        lines.append(f'TOKEN {token:016X} {token:016X} {supplied:016X}')
        state = e['flags'] + e['halts'] + [e['round_index']]
        for tag in ('PRE', 'POST'): lines.append(tag + ' ' + ' '.join(map(str, state)))
        lines.append('SLOTS ' + ' '.join(map(str, [int(e['slots_ok']), *e['slots']])))
        for j in range(3):
            lines.append('BOUNDS ' + ' '.join(map(str, [j + 1, int(e['have'][j]), *e['errors'][j]])))
        for j in range(3):
            for i in range(3):
                lines.append(f'CLASS {i + 1} {j + 1} {e["classes"][i][j]}')
        lines.append('FORWARD ' + ' '.join(map(str, [1, 1, *c['we'], 2, 0, 1])))
        for i, pair in enumerate(MASS, 1):
            lines.append(f'MASS {i} {pair[0]:016X} {pair[1]:016X}')
        for tag, field, digits in [('DPMOM', 'dp_words', 16), ('QPMOM', 'qp_words', 32)]:
            for leg in range(3):
                for mu in range(4):
                    pair = c[field][leg][mu]
                    lines.append(f'{tag} {leg + 1} {mu} {pair[0]:0{digits}X} {pair[1]:0{digits}X}')
        lines.append(f'RECORDED_ENV {supplied:016X}')
        lines.append(f'ENDCASE {c["id"]}')
    lines.append('DONE 240')
    return '\n'.join(lines) + '\n'
