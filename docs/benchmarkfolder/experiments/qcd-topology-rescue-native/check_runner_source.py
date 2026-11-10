#!/usr/bin/env python3
"""Check replay comparison guards using retained bytes only; launch no process.

Run from the MG checkout with Python -B. This imports run.py without invoking its
main function, parses frozen traces, and mutates Python objects/text in memory.
It does not test the native executable, timeout delivery, or live linker lookup.
"""
from pathlib import Path
import ast
import hashlib
import importlib.util
import json
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('private_replay_source_check', HERE / 'run.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def changed_row(text, tag, seq, token, value):
    lines = text.splitlines()
    for index, line in enumerate(lines):
        fields = line.split()
        if fields and fields[0] == tag and (seq is None or fields[1] == str(seq)):
            fields[token] = value
            lines[index] = ' '.join(fields)
            return '\n'.join(lines) + '\n'
    raise RuntimeError('Mutation anchor not found.')


def changed_trace(trace, seq, tag, field, value):
    result = dict(trace)
    result['points'] = list(trace['points'])
    index = next(i for i, point in enumerate(trace['points']) if point['seq'] == seq)
    point = result['points'][index] = dict(trace['points'][index])
    if tag is None:
        point[field] = value
    else:
        point['events'] = list(point['events'])
        item = next(i for i, event in enumerate(point['events']) if event['tag'] == tag)
        point['events'][item] = dict(point['events'][item], **{field: value})
    return result


def main():
    positive, negative = [], []

    def rejected(name, operation):
        try:
            operation()
        except (RuntimeError, decoder.TraceError):
            negative.append(name)
        else:
            raise RuntimeError('Guard accepted mutation: ' + name)

    ast.parse((HERE / 'run.py').read_text())
    decoder = runner.load_decoder()
    for backend in (8, 9):
        old = runner.OLD / 'attempt1' / ('backend-' + str(backend))
        text = (old / 'observed.log').read_text()
        trace = decoder.load_trace(old / 'trace.dat', old / 'observed.log')
        runner.compare_rows(text, text)
        runner.compare_traces(trace, trace, backend)
        runtime, _ = runner.runtime_files((old / 'ldd.txt').read_text())
        runner.require(runtime == json.loads((old / 'execution.json').read_text())['runtime_sha256_before'],
                       'Historical runtime map changed.')
        positive.append(str(backend) + ': historical rows/raw words/runtime files')
        # Status, numeric and IEEE changes at the stress point are retained, not
        # interpreted as a successful amplitude. Both split-order rows may vary.
        stress = text
        lines = stress.splitlines()
        for i, line in enumerate(lines):
            fields = line.split()
            if fields[:2] == ['AMPLITUDE_VALUE', '2']:
                fields[4], fields[5], fields[10:13] = '323', 'NaN', ['F', 'F', 'T']
                lines[i] = ' '.join(fields)
        stress = '\n'.join(lines) + '\n'
        stress = changed_row(stress, 'BORN_RESCUE_COUNTS', 2, 3, '1')
        stress = changed_row(stress, 'BORN_RESCUE_COUNTS', 3, 3, '1')
        result = runner.compare_rows(stress, text)
        runner.require(not result['stress_rows_interpreted_as_physics_acceptance'], 'Stress change was accepted as physics.')
        positive.append(str(backend) + ': changed stress rows/flags and cumulative failure count retained')
        for name, tag, seq, token, value in (
                ('ordinary number', 'AMPLITUDE_VALUE', 1, 5, '0.00000000000000000E+000'),
                ('ordinary status', 'AMPLITUDE_VALUE', 1, 4, '420'),
                ('ordinary flag', 'AMPLITUDE_VALUE', 1, 12, 'T'),
                ('repeated ordinary number', 'AMPLITUDE_VALUE', 3, 5, '0.00000000000000000E+000'),
                ('warmup status', 'AMPLITUDE_WARMUP', 1, 2, '420'),
                ('normalization', 'LOOP_NORMALIZATION', 3, 2, '2.00000000000000000E+003'),
                ('ordinary first rescue', 'BORN_RESCUE_COUNTS', 1, 2, '1'),
                ('ordinary repeated rescue', 'BORN_RESCUE_COUNTS', 3, 2, '2'),
                ('row input identity', 'AMPLITUDE_VALUE', 2, 2, '1')):
            mutation = changed_row(text, tag, seq, token, value)
            rejected(str(backend) + ': ' + name, lambda mutation=mutation: runner.compare_rows(mutation, text))
        rejected(str(backend) + ': missing completion',
                 lambda: runner.compare_rows(text.replace('AMPLITUDE_DIAGNOSTIC_COMPLETE', ''), text))
        for tag in ('POINT_P_USER', 'POINT_HOST_ANS', 'POINT_HOST_ACC', 'POINT_MU_DP'):
            event = next(e for e in trace['points'][0]['events'] if e['tag'] == tag)
            words = list(event['words'])
            words[0] = format(int(words[0], 16) ^ 1, '016X')
            mutation = changed_trace(trace, 1, tag, 'words', words)
            rejected(str(backend) + ': raw bit ' + tag,
                     lambda mutation=mutation: runner.compare_traces(mutation, trace, backend))
        mutation = changed_trace(trace, 3, None, 'host_code', 420)
        rejected(str(backend) + ': raw ordinary code', lambda: runner.compare_traces(mutation, trace, backend))
        rejected(str(backend) + ': wrong reducer backend', lambda: runner.compare_traces(trace, trace, 17 - backend))
        # Feed a real log with nonzero independent health to the frozen parser,
        # not merely a fabricated normalized dictionary.
        rejected(str(backend) + ': nonzero observer health',
                 lambda: decoder.parse_trace((old / 'trace.dat').read_text(), text.replace('QO_HEALTH 2 0 0', 'QO_HEALTH 2 1 0')))
    rejected('conflicting protected pin', lambda: runner.merge_pins({'x': 'a'}, {'x': 'b'}))
    rejected('missing runtime dependency', lambda: runner.runtime_files('libmissing.so => not found\n'))
    print(json.dumps({'source_only': True, 'native_executed': False,
                      'runner_sha256': hashlib.sha256((HERE / 'run.py').read_bytes()).hexdigest(),
                      'checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'positive_checks': positive, 'rejected_mutations': negative,
                      'limitations': ['No compiler, native process, timeout, or live ldd execution.']},
                     indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
