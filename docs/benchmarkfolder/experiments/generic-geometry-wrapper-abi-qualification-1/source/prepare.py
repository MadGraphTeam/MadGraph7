"""Pure one-shot snapshots/corpus. No compiler, native program or subprocess."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE3 = ROOT / '.tirust-build/generic-invariant-provenance-proposal-3'
SOURCE3_SHA = 'c99ce0ffb80a1dabe90a109cc958305efabe1af7968b97ec0adc65a16b009dc2'
GENERATED = SOURCE3 / 'generation-work/born-grouped'
P = GENERATED / 'SubProcesses/P0'
OLD = ROOT / 'docs/benchmarkfolder/experiments/caller-bound-native-qualification-1/proposal'
REFERENCE = ROOT / '.tirust-build/kinematic-invariant-caller-metadata-1/runtime_reference.py'


def identity(path):
    b = path.read_bytes()
    return dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def verify_source3():
    index = SOURCE3 / 'source-index.json'
    if identity(index)['sha256'] != SOURCE3_SHA:
        raise ValueError('source3 index mismatch')
    d = json.loads(index.read_bytes())
    for name, pin in d['files'].items():
        if identity(SOURCE3 / name) != pin:
            raise ValueError('source3 payload changed: ' + name)
    row = next(r for r in json.loads((SOURCE3 / 'generation-results.json').read_bytes())
               if r['case'] == 'born-grouped')
    for name, pin in row['generated'].items():
        if identity(GENERATED / name) != pin:
            raise ValueError('generated source changed: ' + name)
    return row


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def span(text, start, end):
    first = re.search(start, text, re.M | re.I)
    if first is None:
        raise ValueError('missing source start')
    last = re.search(end, text[first.start():], re.M | re.I)
    if last is None:
        raise ValueError('missing source end')
    selected = text[first.start():first.start() + last.end()]
    if not selected.endswith('\n'):
        selected += '\n'
    return selected, dict(start_line=text[:first.start()].count('\n') + 1,
                         end_line=text[:first.start() + last.end()].count('\n'))


def snapshots():
    verify_source3()
    sources = {}
    def save(name, origin, selected=None, selection=None):
        data = origin.read_bytes() if selected is None else selected.encode('utf8')
        dest = HERE / name
        if dest.exists() and dest.read_bytes() != data:
            raise ValueError('one-shot snapshot already differs: ' + name)
        dest.write_bytes(data)
        sources[name] = dict(origin=str(origin.relative_to(ROOT)),
                             origin_identity=identity(origin), selection=selection,
                             snapshot_identity=identity(dest))
    ct = (P / 'CT_interface.f').read_text()
    wrapper, selection = span(ct, r'^      SUBROUTINE ML5_0_LOOP_3_PROVENANCE\(',
                              r'^      END\s*$')
    if '_PROVENANCE_DISPATCH(' not in wrapper or 'SUBROUTINE ML5_0_LOOP_3_PROVENANCE_DISPATCH' in wrapper:
        raise ValueError('incorrect exact-wrapper extraction')
    save('full-CT_interface.f', P / 'CT_interface.f')
    save('wrapper.f', P / 'CT_interface.f', wrapper, selection)
    a = GENERATED / 'Source/DHELAS/aloha_functions.f'
    module, selection = span(a.read_text(), r'^      module ALOHA_OBJECT\s*$',
                            r'^      end module ALOHA_OBJECT\s*$')
    save('full-aloha_functions.f', a)
    save('aloha_object.f', a, module, selection)
    for name, origin in [
        ('invariant_bounds.f90', P / 'invariant_bounds.f90'),
        ('invariant_bounds_data.inc', P / 'invariant_bounds_data.inc'),
        ('environment_token.c', P / 'invariant_environment.c'),
        ('geometry.json', P / 'invariant_provenance.json'),
        ('MadLoopParams.inc', ROOT / 'Template/loop_material/StandAlone/SubProcesses/MadLoopParams.inc'),
        ('supervise.py', OLD / 'supervise.py'),
    ]:
        save(name, origin)
    # Only dimensional/schema adaptation; integer exponent/error arithmetic is
    # inherited from the previously compiled standalone factory's reference.
    original = REFERENCE.read_text()
    adapted = original.replace('1<=route<=26', "1<=route<=len(d['routes'])")
    adapted = adapted.replace("m['width']!=width", "(64 if mode in (1,2) else 128)!=width")
    adapted = adapted.replace('if len(mu)!=4:return None', "if len(mu)!=d['graph']['nexternal']:return None")
    adapted = adapted.replace("chord['expanded_add_sub_operations']", "chord['ops']")
    adapted = adapted.replace("chord['unsigned_leaf_multiplicities']", "chord['mult']")
    if adapted == original:
        raise ValueError('reference schema was not adapted')
    save('reference.py', REFERENCE, adapted,
         {'adaptation': 'route count, external count and source1 chord field names only'})
    return sources


def power_word(exponent, width, sign=False):
    fraction, bias = (52, 1023) if width == 64 else (112, 16383)
    return (int(sign) << (width - 1)) | ((bias + exponent) << fraction)


def base(mode, rotation=0):
    def words(width):
        return [[[power_word(8 * ((e + rotation) % 3) + mu - 7, width,
                             (e + mu) % 2 == 1), 0]
                 for mu in range(4)] for e in range(3)]
    return dict(mode=mode, route=1, we=[1, 2, 3], fresh=True, token_mode=0,
                sticky_mask=0, round_index=0, halt_flag=0,
                dp_words=words(64), qp_words=words(128),
                label='positive-slot-rotation-' + str(rotation))


def corpus():
    cases = []
    def add(case, label=None):
        c = copy.deepcopy(case)
        c['id'] = len(cases) + 1
        if label is not None:
            c['label'] = label
        cases.append(c)
    modes = (1, 2, 4, 5)
    for mode in modes:
        for rotation in range(3): add(base(mode, rotation))
    for mode in modes:
        for mask in range(1, 32):
            c = base(mode); c['sticky_mask'] = mask
            add(c, 'sticky-mask-' + str(mask))
    for mode in modes:
        for label, field, value in [
            ('direct-fresh-false', 'fresh', False),
            ('route-zero', 'route', 0), ('route-too-large', 'route', 2),
            ('we-mismatch', 'we', [3, 2, 1]),
            ('token-zero', 'token_mode', 1),
            ('token-control-mismatch', 'token_mode', 2),
            ('token-bad-marker', 'token_mode', 3),
        ]:
            c = base(mode); c[field] = value; add(c, label)
        width = 128 if mode >= 4 else 64
        fraction, ne = (112, 15) if width == 128 else (52, 11)
        for label, imag, word in [
            ('external-imaginary', True, power_word(0, width)),
            ('external-qnan', False, (((1 << ne) - 1) << fraction) | (1 << (fraction - 1))),
            ('external-snan-bits', False, (((1 << ne) - 1) << fraction) | 1),
            ('external-inf', False, ((1 << ne) - 1) << fraction),
        ]:
            c = base(mode); active = c['qp_words' if width == 128 else 'dp_words']
            active[0][0][int(imag)] = word; add(c, label)
    for mode in modes:
        for rounding in (1, 2, 3):
            c = base(mode); c['round_index'] = rounding
            add(c, 'actual-nonnearest-' + str(rounding))
    for mode in modes:
        for flag in range(1, 6):
            c = base(mode); c['halt_flag'] = flag
            add(c, 'actual-single-halting-' + str(flag))
    for mode in (0, 3, 6, 100): add(base(mode), 'invalid-mode-' + str(mode))
    for mode in modes:
        width = 128 if mode >= 4 else 64
        fraction = 112 if width == 128 else 52
        for label, word in [
            ('positive-zero', 0), ('negative-zero', 1 << (width - 1)),
            ('smallest-subnormal', 1), ('largest-subnormal', (1 << fraction) - 1),
            ('smallest-normal', 1 << fraction),
            ('last-significand-bit', power_word(0, width) | 1),
        ]:
            c = base(mode); active = c['qp_words' if width == 128 else 'dp_words']
            for e in range(3):
                for mu in range(4): active[e][mu][0] = word
            add(c, label)
    if len(cases) != 240:
        raise ValueError('frozen corpus count changed')
    return cases


def expected(cases):
    d = json.loads((HERE / 'geometry.json').read_bytes())
    ref = load('generic_wrapper_reference', HERE / 'reference.py')
    rows = []
    for c in cases:
        mode = c['mode']; width = 128 if mode >= 4 else 64
        mode_ok = mode in (1, 2, 4, 5)
        route_ok = ref.route_matches(d, c['route'], 3, c['we'], [1, 1, 1])
        slots_ok = mode_ok and c['route'] == 1
        env_ok = c['token_mode'] == 0 and c['round_index'] == 0
        admitted = route_ok and mode_ok and c['fresh'] and env_ok
        active = c['qp_words' if width == 128 else 'dp_words']
        external = [[active[e][mu] for e in range(3)] for mu in range(4)]
        bounds = ref.build(d, c['route'], mode, width, 3, external, admitted)
        rows.append(dict(id=c['id'], label=c['label'], mode=mode, width=width,
                         route=c['route'], fresh=c['fresh'], supplied_we=c['we'],
                         slots_ok=slots_ok, slots=[1, 2, 3] if slots_ok else [0, 0, 0],
                         have=[bounds is not None] * 3,
                         errors=bounds['chord_error'] if bounds else [[ref.ZERO] * 4 for _ in range(3)],
                         classes=bounds['pair_class'] if bounds else [[0] * 3 for _ in range(3)],
                         flags=[int(bool(c['sticky_mask'] & (1 << i))) for i in range(5)],
                         halts=[int(c['halt_flag'] == i) for i in range(1, 6)],
                         round_index=c['round_index'], token_mode=c['token_mode']))
    return rows


def render(cases):
    lines = [str(len(cases))]
    for c in cases:
        lines.append(' '.join(map(str, [c['id'], c['mode'], c['route'], *c['we'],
                                      int(c['fresh']), c['token_mode'], c['sticky_mask'],
                                      c['round_index'], c['halt_flag']])))
        for field, digits in [('dp_words', 16), ('qp_words', 32)]:
            for e in range(3):
                for mu in range(4):
                    lines.append(' '.join(f'{v:0{digits}X}' for v in c[field][e][mu]))
    return '\n'.join(lines) + '\n'


def main():
    sources = snapshots()
    cases = corpus(); answers = expected(cases)
    for name, value in [('source-bindings.json', sources), ('cases.json', cases),
                        ('expected.json', answers)]:
        (HERE / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    (HERE / 'inputs.txt').write_text(render(cases))
    print(json.dumps(dict(cases=len(cases), positive_bound_cases=sum(all(r['have']) for r in answers),
                          exact_emitted_wrapper=True, native_executions=0,
                          full_sloop_lifetime_coverage=False)))


if __name__ == '__main__':
    main()
