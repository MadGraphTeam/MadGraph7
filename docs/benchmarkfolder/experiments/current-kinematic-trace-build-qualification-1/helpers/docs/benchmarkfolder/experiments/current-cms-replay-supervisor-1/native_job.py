#!/usr/bin/env python3
"""One CMS request, launched only by the separately admitted owned supervisor.

The outer owner supplies the immutable request specification and owns this
controller and its descendants. This adapter preserves the Worker protocol;
warmup diagnostics never become a qualified final traversal.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.with_name('current-cms-replay-proposal-1')
PLAN = json.loads((PROPOSAL / 'proposal.json').read_bytes())
sys.path.insert(0, str(HERE / 'protocol'))
from tirust_benchmark_worker import (Worker, parse_outputs, RAW_OUTPUTS,
                                    ProtocolError, MeasurementError, validate_status)

POINT_COUNT = 21
PHASE_TIMEOUT_SECONDS = 300.
ACCURACY_LIMIT = 1e-3
EXPECTED_DIMENSIONS = {'allocated': 1, 'loop_orders': 1, 'born_orders': 1}
EXPECTED_ROWS = [(point, order) for point in range(1, 22) for order in (0, 1)]
assert PLAN['point_count'] == POINT_COUNT
assert PLAN['strict_host_accuracy_limit'] == ACCURACY_LIMIT
assert PLAN['caps']['startup_seconds'] == PLAN['caps']['traversal_seconds'] == PHASE_TIMEOUT_SECONDS
assert PLAN['expected_observation']['answer_dimensions'] == EXPECTED_DIMENSIONS
assert PLAN['expected_observation']['ordered_row_keys'] == [list(key) for key in EXPECTED_ROWS]


def safe_json(value):
    """Retain nonfinite diagnostic values explicitly without emitting JSON NaN."""
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')}
    if isinstance(value, dict):
        return {str(key): safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(item) for item in value]
    return value


def save(path, value):
    Path(path).write_text(json.dumps(safe_json(value), indent=2, sort_keys=True,
                                   allow_nan=False) + '\n')


def qualified(parsed, failure, warnings, maps_ok, unexpected, cleanup_failure=None,
              expected_backend=None):
    """Require a complete checked CMS traversal independently of warmup rows."""
    if (failure is not None or parsed is None or warnings or not maps_ok
            or unexpected or cleanup_failure is not None):
        return False
    try:
        rows = parsed['rows']
        if (parsed['answer_dimensions'] != EXPECTED_DIMENSIONS
                or [(row['point'], row['order']) for row in rows] != EXPECTED_ROWS):
            return False
        if any(rows[index]['status'] != rows[index + 1]['status']
               for index in range(0, len(rows), 2)):
            return False
        counts = {}
        for row in rows:
            backend = expected_backend if expected_backend is not None else row['actual_backend']
            status = validate_status(row['status'], backend)
            if (status['backend'] != row['actual_backend'] or status['precision'] != row['precision']
                    or len(row['coefficients']) != 4
                    or not all(math.isfinite(value) for value in row['coefficients'])
                    or not math.isfinite(row['accuracy'])
                    or not 0 <= row['accuracy'] <= ACCURACY_LIMIT):
                return False
            if row['order'] == 0:
                key = str(row['status'])
                counts[key] = counts.get(key, 0) + 1
        histogram = parsed['timed_status_counts']
        if (not all(isinstance(count, int) and not isinstance(count, bool) and count > 0
                    for count in histogram.values())
                or sum(histogram.values()) != POINT_COUNT or histogram != counts
                or parsed['status_counts'] != counts):
            return False
        return True
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def classify_maps(raw, expected):
    """All absolute mappings must be bound ELF or immutable read-only locale data."""
    elf = {}; data = {}
    for line in raw.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/'): continue
        spelling = fields[5]
        if spelling.endswith(' (deleted)'): raise ValueError('Deleted runtime mapping')
        path = Path(spelling); stat = path.stat()
        if int(fields[4]) != stat.st_ino or tuple(int(x, 16) for x in fields[3].split(':')) != (os.major(stat.st_dev), os.minor(stat.st_dev)):
            raise ValueError('Mapped inode/device mismatch: ' + spelling)
        with path.open('rb') as stream: header = stream.read(64)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        name = str(path.resolve())
        if header[:4] == b'\x7fELF':
            if expected.get(name) != checksum: raise ValueError('Unexpected ELF bytes/path: ' + name)
            elf[name] = checksum
        else:
            if not (spelling.startswith('/nix/store/') and spelling.endswith('/lib/locale/locale-archive') and fields[1] == 'r--p'):
                raise ValueError('Unclassified non-ELF mapping: ' + spelling)
            data[name] = {'sha256': checksum, 'size': stat.st_size, 'inode': stat.st_ino,
                          'map_device': fields[3], 'header64_hex': header.hex(), 'permissions': fields[1]}
    return {'elf_files': elf, 'data_files': data, 'exact_expected': elf == expected}


def _error(error):
    return {'class': type(error).__name__, 'message': str(error)}


def _diagnostic_row(fields, backend, phase, line_number, raw_line):
    offset = 1 if phase == 'warmup' else 0
    expected = 9 if offset else 8
    if len(fields) != expected:
        raise ValueError('Expected %d fields, received %d' % (expected, len(fields)))
    row = {'phase': phase, 'line_number': line_number, 'raw_line': raw_line,
           'point': int(fields[offset]), 'order': int(fields[offset + 1]),
           'status': int(fields[offset + 2]),
           'coefficients': [float(value.replace('D', 'E').replace('d', 'e')) for value in fields[offset + 3:offset + 7]],
           'accuracy': float(fields[offset + 7].replace('D', 'E').replace('d', 'e'))}
    if offset:
        row['call_index'] = int(fields[0])
        if row['call_index'] < 0: raise ValueError('Negative warmup call index')
    if (row['point'], row['order']) not in EXPECTED_ROWS:
        raise ValueError('Unexpected point/order')
    reasons = []
    try:
        status = validate_status(row['status'], backend)
        row.update(actual_backend=status['backend'], precision=status['precision'])
    except ValueError as error:
        reasons.append(str(error))
    if not all(math.isfinite(value) for value in row['coefficients'] + [row['accuracy']]):
        reasons.append('Nonfinite coefficient or accuracy')
    elif not 0 <= row['accuracy'] <= ACCURACY_LIMIT:
        reasons.append('Accuracy outside [0, 1e-3]')
    if offset and row['call_index'] == 0:
        reasons.append('Bootstrap only; never a final observation')
    row.update(eligible=not reasons, reason='; '.join(reasons) if reasons else None)
    return row


def diagnostics(raw, backend):
    """Keep every diagnostic row and explicit outcomes for both execution phases."""
    result = {'bootstrap_rows': [], 'warmup_rows': [], 'traversal_rows': [],
              'phase_outcomes': {}, 'parse_issues': []}
    for phase, filename in (('warmup', 'warmup.dat'), ('traversal', 'values.dat')):
        rows = result[phase + '_rows']
        identities = set()
        for line_number, line in enumerate(raw.get(filename, '').splitlines(), 1):
            if not line.strip() or line.lstrip().startswith('#'): continue
            try:
                row = _diagnostic_row(line.split(), backend, phase, line_number, line)
                identity = (row.get('call_index'), row['point'], row['order'])
                if identity in identities:
                    row.update(eligible=False, reason='Duplicate phase row identity')
                    result['parse_issues'].append({'file': filename, 'line_number': line_number,
                                                 'message': 'Duplicate phase row identity', 'raw_line': line})
                identities.add(identity)
                if phase == 'warmup' and row['call_index'] == 0:
                    result['bootstrap_rows'].append(row)
                else:
                    rows.append(row)
            except (ValueError, OverflowError) as error:
                result['parse_issues'].append({'file': filename, 'line_number': line_number,
                                             'message': str(error), 'raw_line': line})
        outcomes = []
        for point, order in EXPECTED_ROWS:
            matching = [row for row in rows if (row['point'], row['order']) == (point, order)]
            latest = matching[-1] if matching else None
            outcomes.append({'point': point, 'order': order,
                             'state': ('accepted' if latest['eligible'] else 'rejected') if latest else 'unexecuted',
                             'reason': latest['reason'] if latest else 'No row produced in this phase',
                             'observations': len(matching),
                             'last_call_index': latest.get('call_index') if latest else None})
        result['phase_outcomes'][phase] = outcomes
    return safe_json(result)


def warning_context(console):
    lines = console.splitlines()
    return [{'line_number': index + 1, 'line': line,
             'context_start_line': max(0, index - 3) + 1,
             'context': '\n'.join(lines[max(0, index - 3):min(len(lines), index + 25)])}
            for index, line in enumerate(lines) if '##W03' in line]


def run_request(spec_path, worker_class=None, parser=None, maps_classifier=None):
    """Retain a result for numerical failure and controller/cleanup failure alike."""
    spec_path = Path(spec_path)
    evidence = spec_path.parent
    base_worker = Worker if worker_class is None else worker_class
    parse = parse_outputs if parser is None else parser
    classify = classify_maps if maps_classifier is None else maps_classifier
    start = time.monotonic()
    spec = {}; captures = []; mapping_errors = []; protocol = []; instances = []
    close_receipts = []; collection_errors = []; raw = {}; raw_bytes = {}; parsed = None
    worker = None; directory = None; failure = None; cleanup_failure = None
    fatal = False; run_complete = False; phase = 'startup'; failure_phase = None

    class Observed(base_worker):
        def __init__(self, *args, **kwargs):
            instances.append(self)
            super().__init__(*args, **kwargs)

        def _message(self):
            try:
                message = super()._message()
                protocol.append({'message': message})
                return message
            except BaseException as error:
                protocol.append({'error': _error(error)})
                raise
            finally:
                process = getattr(self, 'process', None)
                if process is not None:
                    try:
                        try: maps = Path('/proc/%d/maps' % process.pid).read_text()
                        except FileNotFoundError: maps = None
                        if maps is not None:
                            target = evidence / ('maps-%03d.txt' % len(captures))
                            target.write_text(maps)
                            capture = classify(maps, spec['expected_runtime'])
                            captures.append({'path': target.name, 'sha256': hashlib.sha256(maps.encode()).hexdigest(), **capture})
                    except BaseException as error:
                        mapping_errors.append(_error(error))
                        raise

        def _send(self, command):
            protocol.append({'sent': command})
            return super()._send(command)

        def close(self, graceful=True):
            process = getattr(self, 'process', None)
            receipt = {'graceful_requested': graceful, 'pid': getattr(process, 'pid', None)}
            try:
                super().close(graceful=graceful)
                receipt['complete'] = True
            except BaseException as error:
                receipt.update(complete=False, failure=_error(error))
                raise
            finally:
                receipt['worker_closed'] = getattr(self, 'closed', None)
                receipt['returncode'] = getattr(process, 'returncode', None)
                close_receipts.append(receipt)

    try:
        spec = json.loads(spec_path.read_bytes())
        if isinstance(spec['backend'], bool) or spec['backend'] not in (1, 6, 7, 8, 9):
            raise ValueError('Unsupported CMS backend')
        directory = Path(spec['cwd'])
        if not directory.is_absolute(): raise ValueError('Worker cwd must be absolute')
        if not isinstance(spec['argv_without_fifo'], list) or not spec['argv_without_fifo']:
            raise ValueError('Missing worker argv')
        if not isinstance(spec['expected_runtime'], dict): raise ValueError('Missing expected runtime map')
        worker = Observed(spec['argv_without_fifo'], directory, evidence / 'stdout.txt',
                          env=os.environ.copy(), timeout=PHASE_TIMEOUT_SECONDS)
        phase = 'traversal'
        worker.run(1)
        run_complete = True
        phase = 'parsing'
        parsed = parse(directory, [spec['backend']], POINT_COUNT, 1)
    except BaseException as error:
        failure = _error(error)
        failure_phase = phase
        fatal = not isinstance(error, (ProtocolError, MeasurementError))
    finally:
        if worker is not None:
            try: worker.close(graceful=run_complete)
            except BaseException as error: cleanup_failure = _error(error)
        # Worker.__init__ can fail after performing its own cleanup, before
        # assignment to worker. Observed.close records that failure too.
        bad_closes = [receipt['failure'] for receipt in close_receipts if not receipt['complete']]
        if bad_closes and cleanup_failure is None: cleanup_failure = bad_closes[0]
        if directory is not None:
            for name in RAW_OUTPUTS:
                try:
                    path = directory / name
                    if path.is_file():
                        raw_bytes[name] = path.read_bytes()
                        raw[name] = raw_bytes[name].decode(errors='replace')
                        (evidence / name).write_bytes(raw_bytes[name])
                except BaseException as error:
                    collection_errors.append({'file': name, **_error(error)})
    try:
        console_path = evidence / 'stdout.txt'
        console_bytes = console_path.read_bytes() if console_path.is_file() else b''
        console = console_bytes.decode(errors='replace')
        warnings = warning_context(console)
        diagnostic = diagnostics(raw, spec.get('backend', 9))
    except BaseException as error:
        collection_errors.append(_error(error))
        console_bytes = b''; console = ''; warnings = []; diagnostic = None
    maps_ok = bool(captures) and any(item['exact_expected'] for item in captures)
    fatal = bool(fatal or cleanup_failure or mapping_errors or collection_errors)
    passed = qualified(parsed, failure, warnings, maps_ok,
                       mapping_errors or collection_errors, cleanup_failure,
                       expected_backend=spec.get('backend'))
    if diagnostic is None or diagnostic['parse_issues']: passed = False
    result = {'schema_version': 1, 'request': spec.get('request'), 'orchestration_complete': True,
              'numerically_qualified': passed, 'fatal_controller_failure': fatal,
              'failure': failure, 'failure_phase': failure_phase, 'cleanup_failure': cleanup_failure,
              'cleanup_receipts': close_receipts, 'incomplete_warnings': warnings,
              'actual_map_captures': captures, 'runtime_identity_failure': bool(mapping_errors),
              'mapping_errors': mapping_errors, 'complete_runtime_map': maps_ok,
              'collection_errors': collection_errors, 'protocol_messages': protocol,
              'observation': parsed, 'diagnostics': diagnostic,
              'raw_outputs_sha256': {name: hashlib.sha256(value).hexdigest() for name, value in raw_bytes.items()},
              'stdout_sha256': hashlib.sha256(console_bytes).hexdigest(),
              'elapsed_seconds': time.monotonic() - start,
              'phase_timeout_seconds': PHASE_TIMEOUT_SECONDS, 'point_count': POINT_COUNT,
              'accuracy_limit': ACCURACY_LIMIT, 'performance_eligible': False,
              'independent_finite_term_oracle': False}
    result = safe_json(result)
    save(evidence / 'result.json', result)
    return result


def main():
    run_request(Path(sys.argv[1]))


if __name__ == '__main__':
    main()
