#!/usr/bin/env python3
"""Offline source/bit audit of denominator and model masses in QCD refusals."""
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
import argparse
import importlib.util
import json
import re
import sys

sys.dont_write_bytecode = True
from common import ROOT, HERE, DESIGN, PROC, sha, rel, require, pins, controller_pins
from analyze_handoff import load_decoder, event_one, promote64, digest

ROUTING = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace/design/topology'
SP = PROC / 'SubProcesses/P0_uux_zgg'
MODEL = PROC / 'Source/MODEL'


def rational(value):
    return {'numerator_hex': hex(value.numerator), 'denominator_hex': hex(value.denominator)}


def normal_bits(value, width, exact=False):
    """Integer-only encoding, exact or nearest-even; zero and normal range only."""
    sign = int(value < 0)
    value = abs(value)
    if not value:
        return sign << (width - 1)
    precision, bias = (53, 1023) if width == 64 else (113, 16383)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    power = Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)
    if value < power:
        exponent -= 1
    shift = precision - 1 - exponent
    scaled = value * (1 << shift) if shift >= 0 else value / (1 << -shift)
    significand, remainder = divmod(scaled.numerator, scaled.denominator)
    if exact:
        require(remainder == 0, 'Derived mass square is not exactly representable at requested precision.')
    elif remainder * 2 > scaled.denominator or (remainder * 2 == scaled.denominator and significand % 2):
        significand += 1
    if significand == 1 << precision:
        significand >>= 1
        exponent += 1
    require(0 < exponent + bias < 2 * bias + 1, 'Mass outside normal encoding range.')
    return (sign << (width - 1)) | ((exponent + bias) << (precision - 1)) | (significand - (1 << (precision - 1)))


def quad_words(bits, byte_order):
    result = [format(bits & ((1 << 64) - 1), '016X'), format(bits >> 64, '016X')]
    return result if byte_order == 'little' else result[::-1]


class SourceAudit:
    def __init__(self):
        spec = importlib.util.spec_from_file_location('frozen_mass_routing_parser', ROUTING / 'audit_routing.py')
        self.parser = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.parser)
        self.sources = {}
        self.inputs = {rel(path): {'bytes': path.stat().st_size, 'sha256': sha(path)}
                       for path in (ROUTING / 'audit_routing.py', HERE / 'analyze_handoff.py')}

    def read(self, path):
        if path not in self.sources:
            self.inputs[rel(path)] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
            self.sources[path] = self.parser.statements(path.read_text())
        return self.sources[path]

    def anchor(self, path, statement):
        return {'path': rel(path), **statement}

    def selected(self, path, pattern):
        return [self.anchor(path, row) for row in self.read(path) if re.search(pattern, self.parser.compact(row['text']))]

    def groups(self):
        result = {}
        for row in self.read(SP / 'loop_CT_calls_1.f'):
            match = re.fullmatch(r'CALL(ML5_0_LOOP_(\d+)(?:_(\d+))?)\((.*)\)', self.parser.compact(row['text']))
            if not match:
                continue
            name, n = match[1], int(match[2])
            nw = int(match[3]) if match[3] else n
            args = self.parser.arguments(match[4])
            start = (n if match[3] else 0) + nw
            mass_args = args[start:start + n]
            symbols = []
            for argument in mass_args:
                mass = re.fullmatch(r'DCMPLX\((ZERO|MDL_MB|MDL_MT)\)', argument)
                require(mass is not None, 'Unexpected mass prescription: ' + argument)
                symbols.append(mass[1])
            group = int(args[-1])
            require(group not in result, 'Duplicate denominator group.')
            result[group] = {'group': group, 'n': n, 'rank': int(args[-3]), 'routine': name,
                             'call_mass_symbols': symbols, 'call': self.anchor(SP / 'loop_CT_calls_1.f', row),
                             'initial_m2_symbols': symbols[-1:] + symbols[:-1]}
        require(sorted(result) == list(range(1, 117)), 'Incomplete denominator group coverage.')
        return result

    def numerator_groups(self):
        nodes, groups = {0: {'mass_symbols': [], 'ancestors': []}}, defaultdict(list)
        path = SP / 'mp_coef_construction_1.f'
        for row in self.read(path):
            match = re.fullmatch(r'CALL(\w+)\((.*)\)', self.parser.compact(row['text']))
            if not match:
                continue
            name, args = match[1], self.parser.arguments(match[2])
            if name == 'MP_ML5_0_CREATE_LOOP_COEFS':
                node = re.fullmatch(r'WL\(1,0,1,(\d+)\)', args[0])
                require(node is not None and int(node[1]) in nodes, 'Unknown terminal numerator node.')
                groups[int(args[3])].append({'create_call': self.anchor(path, row),
                                             'mass_symbols': nodes[int(node[1])]['mass_symbols'],
                                             'producer_lines': nodes[int(node[1])]['ancestors']})
            elif args[-1] == 'COEFS' and re.fullmatch(r'PL\(0,\d+\)', args[-2]):
                parent = re.fullmatch(r'PL\(0,(\d+)\)', args[0])
                output = int(re.fullmatch(r'PL\(0,(\d+)\)', args[-2])[1])
                require(parent is not None and int(parent[1]) in nodes and output not in nodes, 'Unexpected numerator graph.')
                definition = PROC / 'Source/DHELAS' / (name + '.f')
                signature = next(s for s in self.read(definition) if self.parser.compact(s['text']).startswith('SUBROUTINE' + name + '('))
                formals = self.parser.arguments(self.parser.compact(signature['text']).split('(', 1)[1][:-1])
                require(len(formals) == len(args), 'Numerator kernel arity changed.')
                mass_symbols = [args[i] for i, formal in enumerate(formals) if re.fullmatch(r'M\d+', formal)]
                require(len(mass_symbols) == 1 and mass_symbols[0] in ('ZERO', 'MDL_MB', 'MDL_MT'), 'Unexpected numerator mass symbol.')
                prior = nodes[int(parent[1])]
                nodes[output] = {'mass_symbols': sorted(set(prior['mass_symbols'] + mass_symbols)),
                                 'ancestors': prior['ancestors'] + [row['line']]}
        require(sorted(groups) == list(range(1, 117)), 'Incomplete numerator group coverage.')
        return dict(groups)


def model_records(call, byte_order, decoder):
    result = {}
    for mass in ('MB', 'MT', 'MZ'):
        dp = event_one(call['events'], 'CONTEXT_' + mass + '_DP')['words']
        mp = event_one(call['events'], 'CONTEXT_' + mass + '_MP')['words']
        dv = decoder.finite_fraction(decoder.decode_binary64(dp[0]))
        mv = decoder.finite_fraction(decoder.decode_binary128(mp, byte_order))
        exact_square = mv * mv
        quad_square = normal_bits(exact_square, 128, exact=True)
        dp_square = format(normal_bits(dv * dv, 64), '016X')
        dp_square_value = decoder.finite_fraction(decoder.decode_binary64(dp_square))
        result[mass] = {'dp_mass_words_observed': dp, 'mp_mass_words_observed': mp,
                        'mp_mass_is_exact_dp_promotion': mp == promote64(dp[0], byte_order),
                        'dp_mass_exact': rational(dv), 'mp_mass_exact': rational(mv),
                        'derived_dp_square_nearest_even_word': dp_square,
                        'derived_mp_square_exact': rational(exact_square),
                        'derived_mp_square_exact_quad_words': quad_words(quad_square, byte_order),
                        'promoted_dp_square_equals_exact_mp_square': dp_square_value == exact_square,
                        'promoted_dp_square_minus_exact_mp_square': rational(dp_square_value - exact_square),
                        'square_evidence': 'Derived by exact integer arithmetic from observed mass bits, not a captured model squared-mass variable. QP product is exactly representable.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Refusing to replace mass audit output.')
    frozen = pins()
    frozen.update(controller_pins())
    audit = SourceAudit()
    groups, numerator = audit.groups(), audit.numerator_groups()
    for group, info in groups.items():
        info['numerator_sources'] = numerator[group]
        require(all(set(item['mass_symbols']) == set(info['call_mass_symbols']) for item in numerator[group]),
                'Denominator/numerator mass-symbol set differs.')
    anchors = {
        'dp_complex_mass_declarations_and_square_order': audit.selected(DESIGN / 'topology-design/CT_interface.f', r'COMPLEX\*16M1|M2L\(\d+\)=M\d+\*\*2'),
        'qp_denominator_promotion_and_order': audit.selected(DESIGN / 'topology-design/TIRUST_interface.f', r'MASSES=CMPLX\(M2L,KIND=KI_QTIR\)|CALLML5_0_MP_SWITCH_ORDER'),
        'mp_numerator_model_include': audit.selected(SP / 'mp_coef_construction_1.f', r"INCLUDE'MP_COUPL_SAME_NAME.INC'"),
        'mp_mass_alias': audit.selected(MODEL / 'mp_coupl_same_name.inc', r'REAL\*16MDL_MB|COMMON/MP_MASSES/'),
        'mp_mass_declaration': audit.selected(MODEL / 'mp_coupl.inc', r'REAL\*16MP__MDL_MB|COMMON/MP_MASSES/'),
        'mp_model_mass_squares': audit.selected(MODEL / 'mp_intparam_definition.inc', r'MP__MDL_M[ZBT]__EXP__2=MP__MDL_M[ZBT]\*\*2'),
        'model_parameter_reads': audit.selected(MODEL / 'param_read.inc', r"CALL(?:MP_)?LHA_GET_REAL\(.*'MDL_M[ZBT]'"),
        'mp_external_parameter_reader': audit.selected(MODEL / 'lha_read.f', r'SUBROUTINEMP_LHA_GET_REAL_SILENT|REAL\*16VAR,DEF_VALUE_NUM|REAL\*8BUFF|READ\(VALUE\(I\),\*\)BUFF|VAR=BUFF'),
        'external_mass_targets_and_shell': audit.selected(SP / 'improve_ps.f', r'MASSES\(3\)=MP__MDL_MZ|MASSES\(I\)\*\*2|KEEP_OFFSHELL_MASS|CALLML5_0_MP_ORIG_IMPROVE_PS_POINT_PRECISION'),
        'setter_offshell_flags': audit.selected(DESIGN / 'source-variants/born-c/loop_matrix.f', r'KEEP_OFFSHELL_MASS\([1-5]\)=.FALSE.'),
    }
    require(all(anchors.values()), 'Missing required mass prescription source anchors.')
    improve = audit.read(SP / 'improve_ps.f')
    begin = next(i for i, row in enumerate(improve) if audit.parser.compact(row['text']).startswith('SUBROUTINEML5_0_MP_IMPROVE_PS_POINT_PRECISION('))
    end = next(i for i in range(begin + 1, len(improve)) if audit.parser.compact(improve[i]['text']) == 'END')
    keep_offshell_mentions = [audit.anchor(SP / 'improve_ps.f', row) for row in improve[begin:end]
                             if 'KEEP_OFFSHELL_MASS' in audit.parser.compact(row['text'])]
    require(len(keep_offshell_mentions) == 2
            and audit.parser.compact(keep_offshell_mentions[1]['text']) == 'LOGICALKEEP_OFFSHELL_MASS(NEXTERNAL)',
            'KEEP_OFFSHELL_MASS is no longer confined to the formal argument/declaration.')
    decoder, runs, observed_models, inputs = load_decoder(), {}, {}, {}
    for arm in ('topology-b', 'topology-c'):
        directory = HERE / 'attempt1' / arm / 'backend-9'
        execution = json.loads((directory / 'execution.json').read_text())
        require(execution['complete'] and execution['protected_inputs_unchanged'], 'Incomplete replay.')
        for name, key in (('trace.dat', 'trace_sha256'), ('observed.log', 'log_sha256')):
            require(sha(directory / name) == execution[key], 'Replay trace/log differs.')
        for path in (directory / 'execution.json', directory / 'trace.dat', directory / 'observed.log'):
            inputs[rel(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
        trace = decoder.load_trace(directory / 'trace.dat', directory / 'observed.log')
        point = trace['points'][1]
        model = model_records(point['calls'][0], trace['byte_order'], decoder)
        observed_models[arm] = model
        rows, failure_types, clamps = [], Counter(), Counter()
        for call in point['calls']:
            require(call['backend'] == 9, 'Unexpected traced reducer.')
            require(model_records(call, trace['byte_order'], decoder) == model, 'Model masses changed between calls.')
            info = groups[call['group']]
            dp = event_one(call['events'], 'MASS2_DP')['words']
            qp = event_one(call['events'], 'MASS2_ORDERED')['words']
            symbols = info['initial_m2_symbols']
            ordered = symbols if call['ctmode'] == 4 or call['n'] <= 2 else symbols[:1] + symbols[:0:-1]
            expected_dp = []
            for symbol in symbols:
                real = '0000000000000000' if symbol == 'ZERO' else model[symbol.removeprefix('MDL_')]['derived_dp_square_nearest_even_word']
                expected_dp += [real, '0000000000000000']
            permutation = list(range(call['n'])) if call['ctmode'] == 4 or call['n'] <= 2 else [0, *range(call['n'] - 1, 0, -1)]
            expected_qp = [part for index in permutation for word in dp[2 * index:2 * index + 2]
                           for part in promote64(word, trace['byte_order'])]
            exact_model_qp = []
            for symbol in ordered:
                real = ['0000000000000000'] * 2 if symbol == 'ZERO' else model[symbol.removeprefix('MDL_')]['derived_mp_square_exact_quad_words']
                exact_model_qp += real + ['0000000000000000'] * 2
            mass_type = 'massless' if set(symbols) == {'ZERO'} else symbols[0].removeprefix('MDL_')
            if call['native_status'] != 0:
                failure_types[mass_type + '_n' + str(call['n'])] += 1
            for clamp in call['taken_clamps']:
                clamps.update(map(str, clamp['branches']))
            rows.append({'ctmode': call['ctmode'], 'group': call['group'], 'n': call['n'], 'rank': call['rank'],
                         'mass_type': mass_type, 'denominator_mass_symbols_ordered': ordered,
                         'native_status': call['native_status'], 'stable': call['stable'], 'fpe': call['fpe'],
                         'mass2_dp_observed_words': dp, 'mass2_ordered_observed_words': qp,
                         'dp_square_matches_observed_mass_rne_product': dp == expected_dp,
                         'qp_denominator_is_exact_ordered_dp_square_promotion': qp == expected_qp,
                         'qp_denominator_equals_exact_existing_mp_model_mass_squared': qp == exact_model_qp,
                         'mass_representation_difference': 'DP-square-before-promotion' if qp != exact_model_qp else None})
        c = event_one(point['events'], 'POINT_MP_AFTER')['words']
        z_components = [decoder.finite_fraction(decoder.decode_binary128(c[16 + 2 * mu:18 + 2 * mu], trace['byte_order'])) for mu in range(4)]
        z_invariant = z_components[0] ** 2 - sum(value ** 2 for value in z_components[1:])
        target = decoder.finite_fraction(decoder.decode_binary128(model['MZ']['derived_mp_square_exact_quad_words'], trace['byte_order']))
        runs[arm] = {'host_code': point['host_code'], 'total_calls': len(rows), 'refusal_count': sum(r['native_status'] != 0 for r in rows),
                     'refusals_by_mass_and_n': dict(failure_types), 'taken_clamp_branches': dict(clamps),
                     'all_dp_square_checks_pass': all(r['dp_square_matches_observed_mass_rne_product'] for r in rows),
                     'all_qp_promotion_checks_pass': all(r['qp_denominator_is_exact_ordered_dp_square_promotion'] for r in rows),
                     'mass_representation_difference_calls': sum(not r['qp_denominator_equals_exact_existing_mp_model_mass_squared'] for r in rows),
                     'mass_representation_difference_refusals': sum(r['native_status'] != 0 and not r['qp_denominator_equals_exact_existing_mp_model_mass_squared'] for r in rows),
                     'external_z_exact_c_invariant_minus_exact_mp_mass_squared': rational(z_invariant - target), 'calls': rows}
    report = {'schema': 1, 'offline_analysis_complete': True, 'native_executed': False, 'physics_accepted': False,
              'script': {'path': rel(Path(__file__)), 'sha256': sha(Path(__file__))},
              'inputs': inputs, 'source_inputs': audit.inputs, 'source_anchors': anchors,
              'frozen_pin_count': len(frozen), 'frozen_pins_sha256': digest(frozen),
              'groups': groups, 'observed_model_masses_and_derived_squares': observed_models, 'runs': runs,
              'keep_offshell_mass_argument_ignored_in_mp_improve': {'verified': True, 'only_occurrences': keep_offshell_mentions},
              'limitations': [
                  'Mass square values labeled derived are exact arithmetic consequences of observed model mass bits; no unobserved MP squared-mass variable is presented as measured.',
                  'This export has real zero-width denominator mass arguments; no complex-pole reconstruction or projection is performed by this audit.',
                  'MP numerator sources alias MP_MASSES. The external Z repair target is MP__MDL_MZ; KEEP_OFFSHELL_MASS is an unused policy argument in the inspected improvement routine.',
                  'The MP target mass square is representable exactly in binary128 for these observed promoted binary64 inputs; this does not imply exact floating-point momentum shell closure.',
                  'A denominator/numerator mass representation difference is evidenced for affected groups, but no causal attribution of a native refusal or amplitude accuracy claim follows.',
                  'Massless-group refusals and exactly representable top-mass groups remain outside the bottom square-before-promotion difference.'
              ]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'output': rel(args.output.resolve()), 'sha256': sha(args.output),
                      'runs': {arm: {key: value for key, value in run.items() if key != 'calls'} for arm, run in runs.items()}}))


if __name__ == '__main__':
    main()
