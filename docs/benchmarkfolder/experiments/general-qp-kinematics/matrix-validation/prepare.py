"""Prepare standalone Fortran matrix-policy validation; no compiler or native run."""
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

ROOT = Path.cwd()
HERE = ROOT / '.tirust-build/general-kinematics-consumer-validation'
BASE = ROOT / '.tirust-build/general-kinematics-consumer'
IDENTITY = ROOT / '.tirust-build/general-kinematics-classification'
TRACE = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-native/attempt1/topology-c/backend-9'
DECODER = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer-native/decode_trace.py'
ORACLE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer/routing-modes.json'
TEMPLATE = ROOT / 'madgraph/iolibs/template_files/loop_optimized/CT_interface.inc'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def decimal(value):
    with localcontext() as context:
        context.prec = 150
        return format(Decimal(value.numerator) / Decimal(value.denominator), '.145E')


def exact_record(value):
    return {'numerator_hex': hex(value.numerator), 'denominator_hex': hex(value.denominator)}


def subroutine(text, name):
    pattern = re.compile(r'^\s{6}SUBROUTINE\s+' + re.escape(name) + r'\(', re.M | re.I)
    match = pattern.search(text)
    assert match, name
    end = re.search(r'^\s{6}END\s*$', text[match.start():], re.M | re.I)
    assert end, name
    return text[match.start():match.start() + end.end()] + '\n'


def main():
    assert sha(BASE / 'source-index.json') == '7d002155a9d43f0e172570ab06e7e2e1a2ec29019fe20dd72c5cf0928ba27048'
    index = json.loads((BASE / 'source-index.json').read_text())
    protected = {str((BASE / 'source-index.json').relative_to(ROOT)): sha(BASE / 'source-index.json')}
    for name, entry in index['files'].items():
        assert sha(BASE / name) == entry['sha256'], name
        protected[str((BASE / name).relative_to(ROOT))] = entry['sha256']
    protected.update(index['protected_inputs'])
    source = (BASE / 'generated/consumer.f').read_text()
    names = ['KC_MP_BUILD_KINEMATIC_MATRIX', 'KC_KINEMATIC_IDENTITY',
             'KC_MP_EXTERNAL_SHELL_MASS2', 'KC_MP_BUILD_KINEMATIC_MATRIX_KIN']
    selected = '\n'.join(subroutine(source, name) for name in names)
    (HERE / 'matrix.f').write_text(selected)
    # This off-shell table is generated anew from a real process descriptor;
    # it does not patch the frozen lookup DATA array by hand.
    sys.path.insert(0, str(ROOT))
    graph = load('matrix_validation_graph', IDENTITY / 'test_generation.py')
    import madgraph.loop
    kc = load('madgraph.loop.kinematic_consumer', BASE / 'src/madgraph/loop/kinematic_consumer.py')
    madgraph.loop.kinematic_consumer = kc
    matrix = graph.generate([2, -2, 23, 21, 21])
    matrix.get('processes')[0].get('legs')[3]['offshell'] = True
    off = subroutine(selected, names[0]).replace('KC_', 'OFF_') + kc.render(matrix, True, 'OFF_')
    from madgraph.iolibs import file_writers
    with file_writers.FortranWriter(str(HERE / 'off_shell.f')) as writer:
        writer.writelines(off)
    oracle = json.loads(ORACLE.read_text())
    decoder = load('matrix_validation_decoder', DECODER)
    trace = decoder.load_trace(TRACE / 'trace.dat', TRACE / 'observed.log')
    calls = {(c['ctmode'], c['group']): c for c in trace['points'][1]['calls']}
    cases = []

    def add(label, g, ct, n, i, j, p, expected, which=1, massmode=0,
            internal=1, metadata=None, inject=0, golden=None):
        p = [[F(x) for x in row] for row in p]
        delta = [p[i - 1][k] - p[j - 1][k] for k in range(4)]
        exact = delta[0]**2 - sum(x*x for x in delta[1:])
        # Independent exact Minkowski oracle. The allowance only covers finite
        # input-operation rounding, not an invariant snap to a different target.
        work = sum((abs(p[i - 1][k]) + abs(p[j - 1][k]))**2 for k in range(4))
        bound = F(128, 2**112) * max(work, abs(exact), F(1, 2**1000))
        cases.append(dict(label=label, group=g, ctmode=ct, n=n, i=i, j=j,
            which=which, massmode=massmode, internal=internal, inject=inject,
            expected=expected, shifts=p, exact=exact, bound=bound, metadata=metadata,
            has_golden=int(golden is not None), golden=F(0) if golden is None else golden))

    for ct in (4, 5):
        for row in oracle['groups_by_mode'][str(ct)]:
            for pair in row['pairs']:
                if not pair['target_p1_minus_p4']:
                    continue
                call = calls[ct, row['group']]
                words = next(e['words'] for e in call['events'] if e['tag'] == 'SHIFTS')
                values = [decoder.finite_fraction(decoder.decode_binary128(words[k:k + 2], trace['byte_order']))
                          for k in range(0, len(words), 2)]
                p = [values[4*k:4*k + 4] for k in range(row['n'])]
                i, j = pair['propagators']
                diff = next(e['words'] for e in call['events']
                            if e['tag'] == 'PAIR_DIFFSQ' and e['i'] == i and e['j'] == j)
                golden = decoder.finite_fraction(decoder.decode_binary128(diff[:2], trace['byte_order']))
                assert golden != 0
                add('retained_composite', row['group'], ct, row['n'], i, j, p, 0,
                    internal=0, golden=golden, metadata={'trace_words_sha256': hashlib.sha256(''.join(words).encode()).hexdigest(),
                                          'classification': pair['classification']})
                assert cases[-1]['exact'] != 0

        def pick(classification, leg, no_first=False, minimum_n=2):
            for row in oracle['groups_by_mode'][str(ct)]:
                if row['n'] < minimum_n:
                    continue
                for pair in row['pairs']:
                    if pair['classification'] == classification and pair['single_leg'] == leg:
                        if no_first and 1 in pair['propagators']:
                            continue
                        return row, pair
            raise AssertionError((ct, classification, leg, no_first))

        def geometry(row, pair, delta, hard=0, closure=0):
            p = [[F(0)] * 4 for _ in range(row['n'])]
            i, j = pair['propagators']
            if i == 1:
                p[0][0] = F(closure)
                p[j - 1] = [p[0][k] - F(delta[k]) for k in range(4)]
            else:
                p[i - 1] = [F(hard) + F(delta[0]), *map(F, delta[1:])]
                p[j - 1][0] = F(hard)
                p[0][0] = F(closure)
            return p

        row, pair = pick('direct_single_leg', 4, no_first=True)
        g, n = row['group'], row['n']
        i, j = pair['propagators']
        near_null = [F(1), F(1) - F(1, 2**36), 0, 0]
        base = geometry(row, pair, near_null)
        add('massless_near_shell', g, ct, n, i, j, base, 1)
        add('off_shell_descriptor', g, ct, n, i, j, base, 0, which=2)
        add('invalid_group', 0, ct, n, i, j, base, 0)
        add('invalid_direction', g, 3, n, i, j, base, 0)
        add('invalid_line_count', g, ct, n + 1, i, j, base + [[F(0)] * 4], 0)
        add('invalid_tolerance', g, ct, n, i, j, base, 0, massmode=4)
        add('soft_timelike_hard_shift', g, ct, n, i, j,
            geometry(row, pair, [1, 0, 0, 0], hard=2**40), 0)
        add('true_null_hard_shift', g, ct, n, i, j,
            geometry(row, pair, [1, 1, 0, 0], hard=2**40), 1)
        add('finite_pair_extreme_common_shift', g, ct, n, i, j,
            geometry(row, pair, [2**8100, 0, 0, 0], hard=2**8200), 0)
        add('nonfinite_other_shift', g, ct, n, i, j, base, 0, inject=1)
        row, pair = pick('complement_of_single_leg', 4, minimum_n=3)
        g, n = row['group'], row['n']
        i, j = pair['propagators']
        add('complement_closed', g, ct, n, i, j, geometry(row, pair, near_null), 1)
        p = geometry(row, pair, near_null, closure=F(1, 2**10))
        free = next(k for k in range(n) if k + 1 not in (i, j))
        p[free][0] = F(2**40)
        add('complement_soft_closure_failure', g, ct, n, i, j, p, 0)
        row, pair = pick('direct_single_leg', 3, no_first=True)
        g, n = row['group'], row['n']
        i, j = pair['propagators']
        # External MZ=2, while internal real/CMS poles are distinct. Small
        # perturbations must target 4 from the physical external symbol.
        p = geometry(row, pair, [F(2) + F(1, 2**38), 0, 0, 0])
        add('massive_external_target', g, ct, n, i, j, p, 2)
        add('massive_reversed_pair', g, ct, n, j, i, p, 2)
        add('genuine_qp_external_mass', g, ct, n, i, j, p, 3, massmode=1, internal=0)
        add('negative_external_mass', g, ct, n, i, j, p, 0, massmode=2)
        add('nonfinite_external_mass', g, ct, n, i, j, p, 0, massmode=3)
    assert sum(c['label'] == 'retained_composite' for c in cases) == 38
    data = [str(len(cases))]
    report = []
    for number, case in enumerate(cases, 1):
        keys = ['which', 'group', 'ctmode', 'n', 'i', 'j', 'expected', 'massmode', 'internal', 'inject', 'has_golden']
        data.append(' '.join(str(case[k]) for k in keys))
        data.append(' '.join(decimal(case['shifts'][j][mu]) for mu in range(4) for j in range(case['n'])))
        data.append(decimal(case['exact']) + ' ' + decimal(case['bound']) + ' ' + decimal(case['golden']))
        report.append({k: v for k, v in case.items() if k not in ('shifts', 'exact', 'bound', 'golden')} |
                      {'case': number, 'exact_selected_invariant': exact_record(case['exact']),
                       'rounding_bound': exact_record(case['bound']),
                       'retained_computed_value': exact_record(case['golden'])})
    (HERE / 'cases.dat').write_text('\n'.join(data) + '\n')
    (HERE / 'case-manifest.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    model_paths = {
        'real': ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/Source/MODEL/mp_coupl.inc',
        'cms': ROOT / '.tirust-build/complex-stress-direct-qp-candidate/Source/MODEL/mp_coupl.inc'}
    params = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg/SubProcesses/MadLoopParams.inc'
    for flavor, path in model_paths.items():
        directory = HERE / flavor
        directory.mkdir(exist_ok=True)
        (directory / 'mp_coupl.inc').write_bytes(path.read_bytes())
        (directory / 'MadLoopParams.inc').write_bytes(params.read_bytes())
        body = (HERE / 'model_context.f.in').read_text().replace('@INTERNAL@',
            'MP__CMASS_MDL_MZ=CMPLX(2.0E0_16,-0.125E0_16,KIND=16)\n'
            '      INTERNAL=MP__CMASS_MDL_MZ**2' if flavor == 'cms' else 'INTERNAL=9.0E0_16')
        (directory / 'model_context.f').write_text(body)
    for path in [DECODER, ORACLE, TRACE / 'trace.dat', TRACE / 'observed.log', params, *model_paths.values()]:
        protected[str(path.relative_to(ROOT))] = sha(path)
    for name, digest in protected.items():
        assert sha(ROOT / name) == digest, name
    import os
    protocol = {'cases_per_flavor': len(cases), 'flavors': ['real', 'cms'],
        'compiler_executed': False, 'native_executed': False, 'source_only': True,
        'compiler': {'path': os.environ['FC'], 'sha256': sha(Path(os.environ['FC']))},
        'cpu': 13, 'total_deadline_seconds': 120, 'no_reducer_or_model_library': True,
        'protected_inputs': protected}
    (HERE / 'protocol.json').write_text(json.dumps(protocol, indent=2, sort_keys=True) + '\n')
    files = {str(p.relative_to(HERE)): sha(p) for p in sorted(HERE.rglob('*'))
             if p.is_file() and p.name != 'source-index.json'}
    (HERE / 'source-index.json').write_text(json.dumps({'files': files}, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'cases_per_flavor': len(cases), 'source_index_sha256': sha(HERE / 'source-index.json')}))


if __name__ == '__main__':
    main()
