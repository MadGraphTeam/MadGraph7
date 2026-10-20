#!/usr/bin/env python3
"""Explicitly coordinated trace/timing commands, preserving every attempt."""
import argparse
import collections
import datetime
from decimal import Decimal
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-performance-v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def binding(path):
    return {'path': str(path.relative_to(ROOT)), 'sha256': sha(path), 'bytes': path.stat().st_size}


def pack_trace(trace, destination):
    with destination.open('xb') as output, gzip.GzipFile(fileobj=output, mode='wb', mtime=0, filename='') as packed:
        with trace.open('rb') as stream:
            while data := stream.read(1024 * 1024):
                packed.write(data)
    return binding(destination)


def verify_build(path):
    build = json.loads(path.read_text())
    assert build['status'] == 'compiled-not-executed' and build['exit_code'] == 0
    assert build['plan_identity'] == json.loads((HERE / 'plan.json').read_text())['identity']
    for name, expected in build['sources'].items():
        assert sha(WORK / 'crate' / name) == expected, f'compiled source changed: {name}'
        assert sha(path.parent / 'source' / name) == expected, f'retained source changed: {name}'
    for executable in build['executables'].values():
        assert sha(ROOT / executable['path']) == executable['sha256']
    return build


def check_float_metadata(value):
    assert set(value) == {'precision', 'negative', 'decimal'}
    assert type(value['precision']) is int and 1 <= value['precision'] <= 4096
    assert isinstance(value['negative'], bool) and isinstance(value['decimal'], str)
    # Rug's LowerExp prints zero as "0"/"-0", even with a requested precision.
    assert re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', value['decimal'])
    decimal = Decimal(value['decimal'])
    assert decimal.is_finite() and decimal.is_signed() == value['negative']


def check_trace(path, original):
    expected = [json.loads(line) for line in original.read_text().splitlines()]
    counts = collections.Counter()
    shapes = collections.Counter()
    fields = 0
    floats = 0
    precision = collections.Counter()

    def visit(value):
        nonlocal floats
        if isinstance(value, dict):
            if set(value) == {'precision', 'negative', 'decimal'}:
                check_float_metadata(value)
                floats += 1
                precision[str(value['precision'])] += 1
            else:
                for child in value.values():
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    records = 0
    with path.open() as stream:
        for i, line in enumerate(stream):
            record = json.loads(line)
            old = expected[i]
            assert record['bridge_record'] == i
            assert (record['n'], record['rank']) == (old['n'], old['rank'])
            assert record['final_ascii'] == old['output'], f'final ASCII mismatch at record {i}'
            fields += len(record['final_ascii'])
            shapes[f"{record['n']},{record['rank']}"] += 1
            for call in record['calls']:
                counts[call['kind']['name']] += len(call['output'])
                width = 1 if call['kind']['name'] == 'value' else len(call['kind']['powers'])
                assert len(call['output']) == len(call['requests']) * width
            visit(record)
            records += 1
    assert records == len(expected) == 1392 and fields == 11136
    assert dict(counts) == {'value': 2184, 'Mu': 108, 'T2': 47304, 'T3': 36840}
    assert len(shapes) == 9
    return {'bridge_records': records, 'callback_complex_outputs': dict(sorted(counts.items())),
            'final_ascii_fields': fields, 'exact_ascii_observer_parity': True,
            'shapes': dict(sorted(shapes.items())), 'roundtrip_checked_floats': floats,
            'tracked_precision_counts': dict(sorted(precision.items())),
            'roundtrip_validation': 'Every serialized Float was reparsed and compared for exact stored rational value, precision and sign by the capture executable before writing.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['capture', 'kernel'])
    parser.add_argument('--build', required=True, type=Path)
    parser.add_argument('--authorized-run', required=True)
    parser.add_argument('--trace-result', type=Path)
    args = parser.parse_args()
    assert Path.cwd() == ROOT
    assert args.authorized_run.strip(), 'retain the explicit coordination/quiet-window description'
    assert Path(os.environ['TMPDIR']).resolve().is_relative_to(ROOT)
    os.sched_setaffinity(0, {212})
    build_path = args.build.resolve()
    build = verify_build(build_path)
    plan = json.loads((HERE / 'plan.json').read_text())
    attempts = HERE / ('traces' if args.action == 'capture' else 'timings')
    attempts.mkdir(exist_ok=True)
    attempt = attempts / f'attempt-{len(list(attempts.iterdir())) + 1}'
    attempt.mkdir()
    work = WORK / attempts.name / attempt.name
    work.mkdir(parents=True)
    executable = ROOT / build['executables']['tensor-' + args.action]['path']
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', RAYON_NUM_THREADS='1')
    source = ROOT / plan['input_capture']['path']
    assert sha(source) == plan['input_capture']['sha256']
    original = work / 'original-input.jsonl'
    with gzip.open(source, 'rb') as compressed, original.open('xb') as stream:
        while data := compressed.read(1024 * 1024):
            stream.write(data)
    assert sha(original) == plan['input_capture']['uncompressed_sha256']
    if args.action == 'capture':
        trace = work / 'callbacks.jsonl'
        env.update(NINJARS_TENSOR_REPLAY=str(original), NINJARS_CALLBACK_TRACE=str(trace))
        command = [str(executable)]
        trace_binding = None
    else:
        assert args.trace_result is not None
        trace_result = json.loads(args.trace_result.read_text())
        assert trace_result['status'] == 'passed' and trace_result['action'] == 'capture'
        assert trace_result['build'] == binding(build_path)
        compressed = ROOT / trace_result['retained_trace']['path']
        assert binding(compressed) == trace_result['retained_trace']
        trace = work / 'callbacks.jsonl'
        with gzip.open(compressed, 'rb') as packed, trace.open('xb') as stream:
            while data := packed.read(1024 * 1024):
                stream.write(data)
        assert sha(trace) == trace_result['trace_sha256']
        check_trace(trace, original)
        trace_binding = binding(args.trace_result.resolve())
        command = [str(executable), str(trace), str(attempt / 'blocks.jsonl')]
    record = {'schema': 'pure-tensor-execution-v1', 'action': args.action, 'started_utc': now(),
              'status': 'running', 'coordination': args.authorized_run, 'command': command,
              'cpu_affinity': sorted(os.sched_getaffinity(0)),
              'thread_siblings': Path('/sys/devices/system/cpu/cpu212/topology/thread_siblings_list').read_text().strip(),
              'build': binding(build_path), 'executable': binding(executable), 'trace_preparation': trace_binding,
              'plan': binding(HERE / 'plan.json'), 'controller': binding(Path(__file__).resolve()),
              'frozen_input': binding(source), 'production_changed': False, 'acceptance_evidence': False}
    record['host_before'] = {name: Path('/proc' + '/' + name).read_text() for name in ('stat', 'loadavg')}
    record['processes_before'] = subprocess.check_output(['ps', '-eo', 'pid,ppid,psr,pcpu,etime,args'], text=True)
    write(attempt / 'started.json', record)
    start = time.monotonic()
    try:
        with (attempt / 'stdout.log').open('x') as stdout, (attempt / 'stderr.log').open('x') as stderr:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr).returncode
        record['exit_code'] = code
        verify_build(build_path)
        assert sha(source) == plan['input_capture']['sha256']
        assert code == 0, f'{args.action} exited {code}'
        record['trace_sha256'] = sha(trace)
        if args.action == 'capture':
            record['observer'] = check_trace(trace, original)
            compressed = attempt / 'callbacks.jsonl.gz'
            record['retained_trace'] = pack_trace(trace, compressed)
        else:
            assert record['trace_sha256'] == trace_result['trace_sha256']
            rows = [json.loads(line) for line in (attempt / 'blocks.jsonl').read_text().splitlines()]
            assert rows[-1] == {'event': 'complete', 'pairs': 21, 'exact': True}
            assert len([v for v in rows if v['event'] == 'block']) == 42
            record['blocks'] = binding(attempt / 'blocks.jsonl')
        record['status'] = 'passed'
    except BaseException as exc:
        record['status'] = 'failed'
        record['failure'] = repr(exc)
        if trace.exists():
            record['partial_or_failed_trace'] = binding(trace)
            record['retained_failed_trace'] = pack_trace(trace, attempt / 'failed-callbacks.jsonl.gz')
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - start
        record['finished_utc'] = now()
        record['host_after'] = {name: Path('/proc' + '/' + name).read_text() for name in ('stat', 'loadavg')}
        record['all_numerical_children_exited'] = True
        write(attempt / 'result.json', record)
    print(json.dumps({'status': record['status'], 'result': str((attempt / 'result.json').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
