#!/usr/bin/env python3
"""Read-only analysis of frozen text, plus append-only audit collection.

No compiler, subprocess, ctypes, numerical-library call, or tolerance change.
Default verification is portable; --live additionally checks original bindings.
"""
from decimal import Decimal as D, localcontext
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import sys

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.parent
ROOT = HERE.parents[6]
RUN = ROOT / '.tirust-build/generated-rank6-poles-1/common-input1/run1'
NOTE = Path('/shared/external/madloop_ninjars/madgraph7_madloop_ninjars/.ninjars-build/upstream-poles-20261023/ninjars/COLLIER_CACHE_BUG.md')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, obj):
    with (HERE / name).open('x') as stream:
        json.dump(obj, stream, indent=2, sort_keys=True)
        stream.write('\n')


def load(name):
    return json.loads((HERE / name).read_text())


def binary128(words):
    bits = int(words[0], 16) | (int(words[1], 16) << 64)
    sign, exponent, fraction = bits >> 127, (bits >> 112) & 32767, bits & ((1 << 112) - 1)
    if exponent == 32767:
        return D('NaN') if fraction else D('-Infinity' if sign else 'Infinity')
    with localcontext() as ctx:
        ctx.prec = 180
        return D(-1 if sign else 1) * D(fraction + ((1 << 112) if exponent else 0)) * D(2) ** ((exponent or 1) - 16383 - 112)


def number(row, index, slot, part):
    return binary128(row['values'][index][slot][2 * part:2 * part + 2])


def symmetric(x, y):
    return abs(x - y) / abs(x + y) if x + y else D(0) if x == y else D('Infinity')


def summarize(snapshot):
    by = {row['action']: row for row in snapshot['rows']}
    monomials = snapshot['monomials']
    left, right = by[3007], by[3013]
    ranks, suspects, poles = [], [], []
    for rank in range(7):
        indices = [i for i, powers in enumerate(monomials) if sum(powers) == rank]
        parts = []
        for part in range(2):
            values = [(symmetric(number(left, i, 0, part), number(right, i, 0, part)), i) for i in indices]
            worst, index = max(values)
            parts.append({'part': ['real', 'imaginary'][part], 'mismatched_at_3e6': sum(v >= D('3e-6') for v, _ in values),
                          'maximum_symmetric_residual': str(worst), 'worst_index': index,
                          'maximum_absolute_difference': str(max(abs(number(left, i, 0, part) - number(right, i, 0, part)) for i in indices))})
            for residual, index in values:
                if residual >= D('3e-6'):
                    suspects.append({'index': index, 'powers': monomials[index], 'rank': rank, 'part': ['real', 'imaginary'][part],
                        'tirust_action': 3007, 'collier_action': 3013, 'tirust_binary128_words': left['values'][index][0],
                        'collier_binary128_words': right['values'][index][0],
                        'tirust_value': str(number(left, index, 0, part)), 'collier_value': str(number(right, index, 0, part)),
                        'absolute_difference': str(abs(number(left, index, 0, part) - number(right, index, 0, part))),
                        'symmetric_residual': str(residual)})
        ranks.append({'rank': rank, 'coefficients': len(indices), 'parts': parts})
    for slot in (1, 2):
        finite_scaled = []
        residuals = []
        for i in range(210):
            for part in range(2):
                x, y = number(left, i, slot, part), number(right, i, slot, part)
                finite = max(D('1e-99'), abs(number(left, i, 0, part)), abs(number(right, i, 0, part)))
                finite_scaled.append(max(abs(x), abs(y)) / finite)
                residuals.append(symmetric(x, y))
        poles.append({'slot': ['finite', 'simple', 'double'][slot],
                      'maximum_symmetric_residual': str(max(residuals)),
                      'maximum_pole_to_finite_ratio': str(max(finite_scaled)),
                      'tirust_nonzero_complex_coefficients': sum(any(number(left, i, slot, part) != 0 for part in range(2)) for i in range(210)),
                      'collier_nonzero_complex_coefficients': sum(any(number(right, i, slot, part) != 0 for part in range(2)) for i in range(210))})
    selector_equal = {}
    for label, actions in [('tirust', (3007, 3009, 3011)), ('collier', (3013, 3014, 3015))]:
        first = by[actions[0]]
        selector_equal[label] = all(first['values'][i][0] == by[action]['values'][i][0] for action in actions[1:] for i in range(210))
    full_partial = all(by[full]['values'][i][slot] == by[partial]['values'][i][slot]
                       for full, partial in ((3007, 3008), (3009, 3010), (3011, 3012)) for i in range(210) for slot in (1, 2))
    maximum_finite = max(D(r['symmetric_residual']) for r in suspects)
    return {'schema': 1, 'status': 'open; neither finite truth nor implementation blame established',
            'fixture': 108, 'n': 6, 'rank': 6, 'coefficient_count': 210,
            'finite_mismatched_parts': len(suspects), 'finite_mismatched_real_parts': sum(s['part'] == 'real' for s in suspects),
            'finite_mismatched_imaginary_parts': sum(s['part'] == 'imaginary' for s in suspects),
            'maximum_finite_symmetric_residual': str(maximum_finite), 'rank_summaries': ranks,
            'same_finite_binary128_words_across_pole_selectors': selector_equal,
            'qp_full_partial_pole_binary128_words_equal': full_partial, 'pole_summaries': poles,
            'n6_row_states': [{'action': a, 'state': by[a]['state'], 'adapter_stable': by[a]['stable']} for a in range(3001, 3016)],
            'suspect_rows': suspects}


def elf_version(path):
    """Inspect an initialized ELF symbol without loading or executing the DSO."""
    b = path.read_bytes()
    assert b[:6] == b'\x7fELF\x02\x01'
    off = struct.unpack_from('<Q', b, 40)[0]
    size, count = struct.unpack_from('<HH', b, 58)
    sections = [struct.unpack_from('<IIQQQQIIQQ', b, off + i * size) for i in range(count)]
    found = []
    for section in sections:
        if section[1] != 11:
            continue
        strings_section = sections[section[6]]
        strings = b[strings_section[4]:strings_section[4] + strings_section[5]]
        for position in range(section[4], section[4] + section[5], section[9]):
            name, _, _, index, address, length = struct.unpack_from('<IBBHQQ', b, position)
            symbol = strings[name:].split(b'\0', 1)[0]
            if symbol == b'__collier_global_MOD_version_cll':
                storage = sections[index]
                file_offset = storage[4] + address - storage[3]
                found.append({'symbol': symbol.decode(), 'section_index': index, 'virtual_address_hex': hex(address),
                              'file_offset': file_offset, 'symbol_size': length, 'bytes_hex': b[file_offset:file_offset + length].hex(),
                              'version': b[file_offset:file_offset + length].decode('ascii'), 'dso_sha256': sha(path)})
    assert len(found) == 1
    return found[0]


def collect():
    spec = importlib.util.spec_from_file_location('frozen_common_input', PROPOSAL / 'analyze.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.analyze(RUN) == json.loads((RUN / 'results.json').read_text())
    parser = module.parser()
    text = (RUN / 'records.txt').read_text().splitlines()
    class Records:
        def read_text(self):
            return '\n'.join(text[:1] + text[3:]) + '\n'
        def __str__(self):
            return str(RUN / 'records.txt')
    _, rows = parser['parse_records'](Records(), 'ew', json.loads((PROPOSAL / 'schedule.json').read_text()))
    frozen = []
    for row in rows:
        frozen.append({**{key: value for key, value in row.items() if key != 'values'},
                       'values': [[component['words'] for component in coefficient] for coefficient in row['values']]})
    snapshot = {'schema': 1, 'scope': 'All thirty parsed original rows; numerical values retained as exact binary128 words.',
                'rows': frozen, 'monomials': json.loads((PROPOSAL.parent / 'materialized/ew/monomials.json').read_text())}
    raw = (json.dumps(snapshot, sort_keys=True, separators=(',', ':')) + '\n').encode()
    with (HERE / 'rows.json.gz').open('xb') as stream:
        stream.write(gzip.compress(raw, mtime=0))
    with localcontext() as ctx:
        ctx.prec = 100
        save('analysis.json', summarize(snapshot))
    dso = ROOT / 'HEPTools/collier/libcollier.so'
    identity = elf_version(dso)
    execution = json.loads((RUN / 'execution.json').read_text())
    assert identity['version'] == '1.2.9'
    assert sha(ROOT / 'HEPTools/collier/COLLIER-1.2.9/libcollier.so') == identity['dso_sha256']
    assert execution['mapping']['mapped_files_sha256_after_exit'][str(dso)] == identity['dso_sha256']
    save('collier-identity.json', identity)
    excerpts = []
    requested = {
        str(PROPOSAL.relative_to(ROOT) / 'driver.f'): [(75, 85), (145, 168)],
        str(PROPOSAL.parent.relative_to(ROOT) / 'materialized/ew/cap_helpers.f'): [(285, 289), (305, 324), (330, 345)],
        str(PROPOSAL.parent.relative_to(ROOT) / 'materialized/ew/cap_collier.f'): [(251, 259), (449, 456), (551, 588)],
        'Template/loop_material/StandAlone/SubProcesses/MadLoopParamReader.f': [(368, 377)],
        'HEPTools/collier/COLLIER-1.2.9/src/collier_global.F90': [(21, 27)],
        'HEPTools/collier/COLLIER-1.2.9/src/collier_init.F90': [(238, 243), (509, 533), (2248, 2260), (2507, 2539)],
        'HEPTools/collier/COLLIER-1.2.9/src/collier_tensors.F90': [(5990, 5999), (6053, 6077), (6185, 6190)],
        'HEPTools/collier/COLLIER-1.2.9/CMakeLists.txt': [(72, 76)],
        'HEPTools/collier/collier_install.log': [(1, 6), (85, 91)],
        'vendor/tirust/src/lib.rs': [(354, 362)],
        '.benchmark-work/tirust/campaigns/split-dp-terminal-1/outputs/candidate/ew_n4_epem_mupmumaa/SubProcesses/MadLoopParamReader.f': [(368, 377)],
    }
    for name, ranges in requested.items():
        path = ROOT / name
        lines = path.read_text().splitlines()
        for first, last in ranges:
            excerpts.append({'path': name, 'source_sha256': sha(path), 'first_line': first, 'last_line': last,
                             'text': '\n'.join(lines[first - 1:last]) + '\n'})
    save('source-excerpts.json', excerpts)
    bindings = {}
    names = [RUN / x for x in ('records.txt', 'observer.txt', 'execution.json', 'results.json')]
    names += [PROPOSAL / x for x in ('analyze.py', 'driver.f', 'fixtures.json', 'schedule.json', 'source-index.json', 'build1/build.json')]
    names += [PROPOSAL.parent / 'analyze.py', PROPOSAL.parent / 'materialize.py', PROPOSAL.parent / 'materialized/ew/monomials.json']
    names += [ROOT / name for name in requested]
    names += [dso, ROOT / 'HEPTools/collier/COLLIER-1.2.9/libcollier.so', NOTE]
    for path in names:
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        bindings[name] = {'bytes': path.stat().st_size, 'sha256': sha(path), 'external': not path.is_relative_to(ROOT)}
    retained_index = PROPOSAL / 'runtime1/index.json'
    save('bindings.json', {'schema': 1, 'source_inputs': bindings, 'live_verification_optional': True,
                           'retained_runtime_index': {'path': str(retained_index.relative_to(ROOT)), 'sha256': sha(retained_index)},
                           'known_cache_note_applicability': 'COLLIER1.2.10 cached N7R9 note; current linked1.2.9 standalone fixture explicitly uncached N6R6; cause not established.',
                           'user_instruction': 'Do not fix the known COLLIER cache implementation and do not contact upstream.'})
    print('Collected thirty rows; original strict analysis replay matched unchanged results.')


def verify(live=False):
    index = load('index.json')
    for name, expected in index['files'].items():
        assert sha(HERE / name) == expected, name
    snapshot = json.loads(gzip.decompress((HERE / 'rows.json.gz').read_bytes()))
    assert len(snapshot['rows']) == 30
    retained = PROPOSAL / 'runtime1'
    assert sha(retained / 'index.json') == index['retained_runtime_index_sha256']
    raw = gzip.decompress((retained / 'data/records.txt.gz').read_bytes())
    original = '.tirust-build/generated-rank6-poles-1/common-input1/run1/records.txt'
    assert hashlib.sha256(raw).hexdigest() == load('bindings.json')['source_inputs'][original]['sha256']
    by = {row['action']: row for row in snapshot['rows']}
    words = {}; states = {}; current = None
    for line in raw.decode().splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == 'CASE':
            current = int(tokens[1]); words[current] = []
        elif tokens[0] == 'STATE':
            states[current] = list(map(int, tokens[1:]))
        elif tokens[0] == 'V':
            index_value, slot = map(int, tokens[1:3])
            assert index_value * 3 + slot - 1 == len(words[current])
            words[current].append([word.upper() for word in tokens[3:7]])
    assert set(words) == set(states) == set(by)
    for action, row in by.items():
        assert words[action] == [component for coefficient in row['values'] for component in coefficient]
        assert states[action] == row['state']
    with localcontext() as ctx:
        ctx.prec = 100
        derived = summarize(snapshot)
    assert derived == load('analysis.json')
    assert derived['finite_mismatched_parts'] == 122
    assert derived['finite_mismatched_real_parts'] == 2
    assert derived['same_finite_binary128_words_across_pole_selectors'] == {'tirust': True, 'collier': True}
    identity = load('collier-identity.json')
    assert identity['version'] == bytes.fromhex(identity['bytes_hex']).decode() == '1.2.9'
    if live:
        for name, item in load('bindings.json')['source_inputs'].items():
            path = Path(name) if item['external'] else ROOT / name
            assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], name
        assert elf_version(ROOT / 'HEPTools/collier/libcollier.so') == identity
    print('PASS: owned hashes, thirty-row snapshot and exact finite/pole diagnostics replay; live bindings=' + str(live))


if __name__ == '__main__':
    if sys.argv[1:] == ['collect']:
        collect()
    elif sys.argv[1:] in ([], ['--live']):
        verify(bool(sys.argv[1:]))
    else:
        raise SystemExit('Usage: audit.py [collect | --live]')
