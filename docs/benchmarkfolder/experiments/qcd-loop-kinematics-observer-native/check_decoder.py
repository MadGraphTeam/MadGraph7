#!/usr/bin/env python3
"""Reproduce the frozen decoder checks using existing text traces only.

Run from the MadGraph repository root. With no options, recompute and compare
decoder-checks.json exactly as a JSON object. --report selects another reference
report; --output instead writes a freshly computed report to a new file using
exclusive creation. No compiler, subprocess, native library, floating-point
conversion, or new physics evaluation is used.
"""
import argparse
import copy
from fractions import Fraction
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


class CheckError(ValueError):
    pass


def ensure(condition, message):
    if not condition:
        raise CheckError(message)


def compute_report(base):
    """Recreate the 18 exact IEEE and 20 rejection checks in the original report."""
    spec = importlib.util.spec_from_file_location('qcd_observer_decoder', base / 'decode_trace.py')
    decoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(decoder)
    checks, exact = [], []

    def reject(name, operation, fragment):
        try:
            operation()
        except decoder.TraceError as error:
            ensure(fragment in str(error), f'{name}: wrong rejection: {error}')
            checks.append({'name': name, 'expected': 'TraceError', 'message_contains': fragment,
                           'observed': str(error), 'passed': True})
        else:
            raise CheckError(f'{name}: accepted malformed evidence')

    for width, fraction_bits, bias in ((64, 52, 1023), (128, 112, 16383)):
        exponent_bits = 11 if width == 64 else 15
        patterns = [
            (0, 'zero', Fraction(0)),
            (1 << (width - 1), 'zero', Fraction(0)),
            (bias << fraction_bits, 'normal', Fraction(1)),
            ((bias << fraction_bits) | (1 << (width - 1)), 'normal', Fraction(-1)),
            (1, 'subnormal', Fraction(1, 1 << (bias + fraction_bits - 1))),
            ((((1 << exponent_bits) - 2) << fraction_bits) | ((1 << fraction_bits) - 1),
             'normal', Fraction(((1 << (fraction_bits + 1)) - 1) << (bias - fraction_bits))),
        ]
        for bits, classification, value in patterns:
            component = decoder.ieee_component(bits, width)
            ensure(component['classification'] == classification
                   and decoder.finite_fraction(component) == value,
                   f'binary{width}: incorrect finite decoding for {bits:X}')
            if width == 128:
                low, high = bits & ((1 << 64) - 1), bits >> 64
                ensure(decoder.decode_binary128([low, high], 'little')
                       == decoder.decode_binary128([high, low], 'big') == component,
                       f'binary128: word-order mismatch for {bits:X}')
            exact.append({'width': width, 'bits_hex': component['bits_hex'],
                          'classification': classification, 'passed': True})
        for tail, classification in ((0, 'infinity'), (1, 'signaling_nan'),
                                      (1 << (fraction_bits - 1), 'quiet_nan')):
            bits = (((1 << exponent_bits) - 1) << fraction_bits) | tail
            component = decoder.ieee_component(bits, width)
            ensure(not component['finite'] and component['classification'] == classification,
                   f'binary{width}: incorrect nonfinite decoding for {bits:X}')
            exact.append({'width': width, 'bits_hex': component['bits_hex'],
                          'classification': classification, 'passed': True})
        reject(f'binary{width}_nonfinite_fraction',
               lambda component=component: decoder.finite_fraction(component), 'nonfinite IEEE')
    reject('binary128_word_overflow', lambda: decoder.decode_binary128([1 << 64, 0]), 'outside INT64')
    reject('binary128_bad_endian', lambda: decoder.decode_binary128([0, 0], 'middle'),
           'unknown native byte order')

    traces = {}
    for backend in (9, 8):
        directory = base / 'attempt1' / f'backend-{backend}'
        result = decoder.load_trace(directory / 'trace.dat', directory / 'observed.log')
        calls = [call for point in result['points'] for call in point['calls']]
        traces[str(backend)] = {
            'inputs': result['inputs'], 'decoder': result['decoder'], 'summary': result['summary'],
            'point_summaries': [
                {'seq': point['seq'], 'point': point['point'], 'host_code': point['host_code'],
                 'summary': point['summary'], 'health': point['health']}
                for point in result['points']
            ],
            'coverage': [{'ctmode': mode, 'groups': sorted(call['group'] for call in calls
                                                         if call['ctmode'] == mode)} for mode in (4, 5)],
            'finite_payload_counts': {
                tag: {
                    'present': sum(tag in call['numerical_summaries'] for call in calls),
                    'all_finite': sum(call['numerical_summaries'].get(tag, {}).get('all_finite', False)
                                      for call in calls),
                } for tag in ('NUM_ORIGINAL', 'NUM_NATIVE', 'NATIVE_MOMENTS', 'NATIVE_CONTRACTED')
            },
        }
        if backend == 9:
            baseline = result
            raw = (directory / 'trace.dat').read_text(encoding='ascii')
            stdout = (directory / 'observed.log').read_text(encoding='utf-8')

    # These validators receive the otherwise valid complete native trace.
    reject('missing_health', lambda: decoder.validate_stdout(
        stdout.replace('QO_HEALTH 1 0 0', ''), baseline['points']), 'health acknowledgement')
    reject('nonzero_health', lambda: decoder.validate_stdout(
        stdout.replace('QO_HEALTH 1 0 0', 'QO_HEALTH 1 0 1'), baseline['points']), 'health acknowledgement')
    reject('completion_before_points', lambda: decoder.validate_stdout(
        'AMPLITUDE_DIAGNOSTIC_COMPLETE\n' + stdout.replace('AMPLITUDE_DIAGNOSTIC_COMPLETE', ''),
        baseline['points']), 'chronology')
    rows = stdout.splitlines()
    first_value = next(index for index, line in enumerate(rows) if line.startswith('AMPLITUDE_VALUE'))
    for name, field, replacement, expected in (
        ('bad_stdout_numeric', 5, 'not_a_real', 'malformed Fortran real'),
        ('bad_stdout_logical', 10, 'Z', 'malformed amplitude IEEE flag'),
        ('stdout_flag_mismatch', 10, 'T', 'stdout/trace IEEE flags differ'),
    ):
        changed = rows[:]
        tokens = changed[first_value].split()
        tokens[field] = replacement
        changed[first_value] = ' '.join(tokens)
        reject(name, lambda text='\n'.join(changed): decoder.validate_stdout(text, baseline['points']), expected)

    # Protocol mutations go through the full public text parser.
    raw_lines = raw.splitlines()

    def edited_event(field, value):
        changed = raw_lines[:]
        index = next(index for index, line in enumerate(changed) if line.startswith('QO_EVENT 2 '))
        tokens = changed[index].split()
        tokens[2 + decoder.META.index(field)] = str(value)
        changed[index] = ' '.join(tokens)
        return '\n'.join(changed)

    first = next(index for index, line in enumerate(raw_lines) if line.startswith('QO_POINT '))
    changed = raw_lines[:]
    tokens = changed[first].split()
    tokens[7] = '1'
    changed[first] = ' '.join(tokens)
    reject('nonzero_fault_counter', lambda: decoder.parse_trace('\n'.join(changed), stdout), 'fault/drop counters')
    reject('word_offset_gap', lambda: decoder.parse_trace(edited_event('word_first', 2), stdout),
           'payload offset gap/overlap')
    reject('bad_tag_length', lambda: decoder.parse_trace(edited_event('tag_length', 1), stdout),
           'tag length/encoding mismatch')
    reject('payload_call_scope_forgery', lambda: decoder.parse_trace(edited_event('call', 1), stdout),
           'point-scoped event contains call context')
    reject('payload_unknown_byte_order', lambda: decoder.decode_payload(baseline['points'][0]['events'][1], 'middle'),
           'unknown native byte order')
    end_first = next(index for index, line in enumerate(raw_lines) if line.startswith('QO_POINT_END '))
    one_point = '\n'.join(raw_lines[:end_first + 1])
    reject('truncated_point_sequence', lambda: decoder.parse_trace(one_point, stdout), 'frozen three-point driver')
    reject('missing_final_footer', lambda: decoder.parse_trace('\n'.join(raw_lines[:-1]), stdout),
           'unexpected end of trace')
    changed = raw_lines[:]
    error = next(index for index, line in enumerate(changed) if line.startswith('QO_ERROR_BYTES '))
    changed[error] = 'QO_ERROR_BYTES'
    reject('empty_error_bytes_record', lambda: decoder.parse_trace('\n'.join(changed), stdout),
           'error bytes/count mismatch')

    call = copy.deepcopy(next(call for call in baseline['points'][1]['calls'] if call['stable']))
    call['stable'], call['fpe'] = 0, 0
    reject('unstable_without_fpe', lambda: decoder.validate_call(call), 'unstable call missing final FPE')
    call = copy.deepcopy(next(call for call in baseline['points'][1]['calls'] if call['stable'] and not call['fpe']))
    next(event for event in call['events'] if event['tag'] == 'FPE_ENTRY')['value'] = 1
    reject('cleared_sticky_fpe', lambda: decoder.validate_call(call), 'cleared the sticky wrapper FPE')

    ensure(len(exact) == 18 and len(checks) == 20, 'frozen check coverage changed')
    return {
        'schema': 1, 'kind': 'source-only_decoder_checks', 'native_execution': False,
        'traces': traces, 'exact_ieee_checks': exact, 'rejected_mutations': checks, 'passed': True,
        'limitations': [
            'These checks validate decoding and trace structure; they do not qualify the physical amplitude '
            'or repair any failed native evaluation.'
        ],
    }


def main():
    # Suppress import cache writes independently of the invoking environment.
    sys.dont_write_bytecode = True
    script = Path(__file__).resolve()
    repository_root = script.parents[4]
    base = script.parent.relative_to(repository_root)
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--report', type=Path, help='existing reference report (default: adjacent decoder-checks.json)')
    action.add_argument('--output', type=Path, help='write a fresh report; the destination must not exist')
    args = parser.parse_args()
    try:
        ensure(Path.cwd().resolve() == repository_root, 'run from the MadGraph repository root')
        if args.output is not None:
            ensure(not args.output.exists(), f'output already exists: {args.output}')
        result = compute_report(base)
        if args.output is None:
            reference = args.report or base / 'decoder-checks.json'
            ensure(json.loads(reference.read_text(encoding='utf-8')) == result,
                   f'recomputed decoder checks differ from {reference}')
            action_name, destination = 'verified_existing_report', reference
        else:
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(json.dumps(result, indent=2, sort_keys=True) + '\n')
            action_name, destination = 'wrote_new_report', args.output
    except (CheckError, OSError, ValueError, StopIteration) as error:
        parser.exit(1, f'decoder checks failed: {error}\n')
    print(json.dumps({
        'action': action_name, 'report': str(destination), 'native_execution': False,
        'trace_count': len(result['traces']), 'exact_ieee_checks': len(result['exact_ieee_checks']),
        'rejected_mutations': len(result['rejected_mutations']),
        'report_sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
        'driver_sha256': hashlib.sha256(script.read_bytes()).hexdigest(),
    }, sort_keys=True))


if __name__ == '__main__':
    main()
