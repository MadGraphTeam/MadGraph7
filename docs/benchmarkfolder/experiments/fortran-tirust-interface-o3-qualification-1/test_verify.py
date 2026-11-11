"""Pure O3 capsule replay, relocation and negative controls; no native execution."""
import builtins
from contextlib import contextmanager
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import struct
import subprocess
import tempfile
import types

import pytest

HERE = Path(__file__).resolve().parent
INDEX = 'evidence-index.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob('*') if p.is_file()}


def reader(files, calls=None):
    def read(name):
        if calls is not None:
            calls.append(name)
        path = PurePosixPath(name)
        if (not isinstance(name, str) or not name or '\\' in name or path.is_absolute()
                or '..' in path.parts or str(path) != name or name not in files):
            raise AssertionError('Outside or unindexed capsule read: ' + str(name))
        return files[name]
    read.paths = set(files)
    return read


def forbidden(*args, **kwargs):
    raise AssertionError('Native/process API is forbidden during portable replay')


@contextmanager
def no_process_api():
    with pytest.MonkeyPatch.context() as patch:
        for name in ('Popen', 'run', 'call', 'check_call', 'check_output'):
            patch.setattr(subprocess, name, forbidden)
        for name in ('CDLL', 'PyDLL', 'WinDLL', 'OleDLL'):
            if hasattr(ctypes, name):
                patch.setattr(ctypes, name, forbidden)
        for name in ('system', 'fork', 'forkpty', 'posix_spawn', 'posix_spawnp',
                     'spawnv', 'spawnve', 'spawnvp', 'spawnvpe', 'execv', 'execve',
                     'execvp', 'execvpe', 'execl', 'execle', 'execlp', 'execlpe'):
            if hasattr(os, name):
                patch.setattr(os, name, forbidden)
        yield


@contextmanager
def no_live_reads():
    def blocked(*args, **kwargs):
        raise AssertionError('Callback replay attempted a live filesystem read')
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, 'open', blocked)
        patch.setattr(io, 'open', blocked)
        patch.setattr(Path, 'read_bytes', blocked)
        patch.setattr(Path, 'read_text', blocked)
        yield


def logical(files, suffix):
    payloads = json.loads(files[INDEX])['payloads']
    names = [name for name in payloads if name.endswith(suffix)]
    assert len(names) == 1, (suffix, names)
    return names[0]


def mutate_payload(files, name, transform):
    files = dict(files)
    index = json.loads(files[INDEX])
    row = index['payloads'][name]
    raw = files[row['path']]
    raw = gzip.decompress(raw) if row['encoding'] == 'gzip+utf8' else raw
    raw = transform(raw)
    stored = gzip.compress(raw, mtime=0) if row['encoding'] == 'gzip+utf8' else raw
    files[row['path']] = stored
    row.update(bytes=len(raw), sha256=sha(raw))
    index['files'][row['path']].update(bytes=len(stored), sha256=sha(stored))
    files[INDEX] = encode(index)
    return files


def change_json(raw, update):
    value = json.loads(raw)
    update(value)
    return encode(value)


@pytest.fixture(scope='module')
def packet():
    files = snapshot(HERE)
    module = types.ModuleType('independent_o3_portable_profile_verifier')
    module.__file__ = str(HERE / 'verify_evidence.py')
    exec(compile(files['verify_evidence.py'], module.__file__, 'exec'), module.__dict__)
    calls = []
    with no_process_api(), no_live_reads():
        result = module.verify(read_bytes=reader(files, calls), expected_index_sha256=sha(files[INDEX]))
    assert result['status'] == 'PASS' and isinstance(result['summary'], dict)
    assert set(calls) <= set(files)
    yield types.SimpleNamespace(files=files, module=module, verify=module.verify, result=result,
                                index_sha=sha(files[INDEX]))
    assert snapshot(HERE) == files, 'Original capsule changed during tests'


def replay(packet, files, **kwargs):
    with no_process_api(), no_live_reads():
        return packet.verify(read_bytes=reader(files), derive_summary=True, **kwargs)


def test_complete_callback_replay_has_no_live_reads_or_process_apis(packet):
    assert packet.result['status'] == 'PASS'


def test_wrong_expected_index_fails_after_one_read(packet):
    calls = []
    with no_process_api(), no_live_reads(), pytest.raises(ValueError):
        packet.verify(read_bytes=reader(packet.files, calls), expected_index_sha256='0' * 64)
    assert calls == [INDEX]


def test_stored_byte_mutation_and_missing_payload_fail(packet):
    name = next(iter(json.loads(packet.files[INDEX])['files']))
    files = dict(packet.files)
    files[name] += b'changed'
    with pytest.raises((ValueError, AssertionError)):
        replay(packet, files)
    files = dict(packet.files)
    del files[name]
    with pytest.raises((ValueError, AssertionError, FileNotFoundError, KeyError)):
        replay(packet, files)


@pytest.mark.parametrize('unsafe', ['/tmp/outside', '../outside', 'a/../outside',
                                    'a//outside', 'a\\outside', INDEX])
def test_unsafe_physical_names_fail_before_payload_reads(packet, unsafe):
    files = dict(packet.files)
    index = json.loads(files[INDEX])
    original = next(iter(index['files']))
    index['files'][unsafe] = index['files'].pop(original)
    files[INDEX] = encode(index)
    calls = []
    with no_process_api(), no_live_reads(), pytest.raises((ValueError, AssertionError)):
        packet.verify(read_bytes=reader(files, calls), derive_summary=True)
    assert calls == [INDEX]


def test_duplicate_logical_physical_alias_rejected_before_payload_reads(packet):
    files = dict(packet.files)
    index = json.loads(files[INDEX])
    original = next(iter(index['payloads']))
    index['payloads']['unexpected-alias.txt'] = dict(index['payloads'][original])
    files[INDEX] = encode(index)
    calls = []
    with no_process_api(), no_live_reads(), pytest.raises((ValueError, AssertionError)):
        packet.verify(read_bytes=reader(files, calls), derive_summary=True)
    assert calls == [INDEX]


def test_extra_callback_file_is_rejected(packet):
    files = {**packet.files, 'unindexed-extra.txt': b'extra\n'}
    with pytest.raises((ValueError, AssertionError)):
        replay(packet, files)


@pytest.mark.parametrize('native', [b'\x7fELF', b'!<arch>\n', b'valid\0nul'])
def test_reindexed_native_or_nul_payload_rejected(packet, native):
    name = next(iter(json.loads(packet.files[INDEX])['payloads']))
    files = mutate_payload(packet.files, name, lambda raw: native)
    with pytest.raises((ValueError, AssertionError, UnicodeError)):
        replay(packet, files)


def test_reindexed_gzip_decoded_identity_still_required(packet):
    files = dict(packet.files)
    index = json.loads(files[INDEX])
    name = next(n for n, row in index['payloads'].items() if row['encoding'] == 'gzip+utf8')
    row = index['payloads'][name]
    raw = gzip.decompress(files[row['path']]) + b'changed\n'
    stored = gzip.compress(raw, mtime=0)
    files[row['path']] = stored
    index['files'][row['path']].update(bytes=len(stored), sha256=sha(stored))
    files[INDEX] = encode(index)
    with pytest.raises((ValueError, AssertionError)):
        replay(packet, files)


@pytest.mark.parametrize('change', ['own_word', 'QP_status', 'accuracy_decimal',
                                    'accuracy_report', 'median', 'IEEE', 'learning',
                                    'review_join', 'prospective_patch',
                                    'prospective_metadata', 'predecessor_unadmitted'])
def test_reindexed_scientific_or_ancestry_mutation_fails_semantic_replay(packet, change):
    if change == 'own_word':
        name = logical(packet.files, 'n1timing/candidate/full100/pair-1/timed-exact-words.dat')
        def transform(raw):
            lines = raw.decode().splitlines()
            fields = lines[0].split()
            fields[3] = f'{int(fields[3], 16) ^ 1:016X}'
            lines[0] = ' '.join(fields)
            return ('\n'.join(lines) + '\n').encode()
    elif change == 'QP_status':
        name = logical(packet.files, 'n3parity/candidate/exact/exact-words.dat')
        def transform(raw):
            lines = raw.decode().splitlines()
            changed = False
            for index, line in enumerate(lines):
                fields = line.split()
                if fields[0] == '16':
                    assert fields[2] == '323'
                    #328 is an accepted QP alias for a different reducer.
                    fields[2] = '328'
                    lines[index] = ' '.join(fields)
                    changed = True
                    break
            assert changed
            return ('\n'.join(lines) + '\n').encode()
    elif change == 'accuracy_decimal':
        name = logical(packet.files, 'n1timing/candidate/full100/pair-1/values.dat')
        def transform(raw):
            lines = raw.decode().splitlines()
            fields = lines[0].split()
            assert len(fields) == 8
            #Finite, nonnegative drift across events may be diagnostic;
            #all five same-event decimal/raw words must still agree.
            fields[-1] = '1.00000000000000000E+000'
            lines[0] = ' '.join(fields)
            return ('\n'.join(lines) + '\n').encode()
    elif change == 'accuracy_report':
        name = logical(packet.files, 'n1timing/candidate/full100/pair-1/own-comparisons.json')
        def transform(raw):
            def update(value):
                value['qualified_parity']['accuracy_drift_count'] += 1
            return change_json(raw, update)
    elif change in ('median', 'review_join'):
        name = logical(packet.files, 'n1timing/result.json')
        def transform(raw):
            def update(value):
                if change == 'median':
                    value['scopes']['full100']['summary']['median_ratio'] += 0.01
                else:
                    value['reviewed']['build'] = '0' * 64
            return change_json(raw, update)
    elif change == 'IEEE':
        name = logical(packet.files, 'n1timing/candidate/full100/pair-1/timed-ieee.dat')
        def transform(raw):
            lines = raw.decode().splitlines()
            fields = lines[0].split()
            fields[3] = '1' if fields[3] == '0' else '0'
            lines[0] = ' '.join(fields)
            return ('\n'.join(lines) + '\n').encode()
    elif change == 'learning':
        name = logical(packet.files, 'n1timing/candidate/warmup.json')
        def transform(raw):
            return change_json(raw, lambda value: value['rows'][0].update(status=323))
    elif change == 'prospective_patch':
        name = logical(packet.files, 'prospective/delivery/backend9-wrapper-opt-in.patch')
        def transform(raw):
            return raw + b'\n# prospective patch mutation\n'
    elif change == 'prospective_metadata':
        name = logical(packet.files, 'prospective/delivery/index.json')
        def transform(raw):
            return change_json(raw, lambda value: value.update(production_changes=True, patches_applied=1))
    else:
        name = logical(packet.files, 'source/source2/predecessor.json')
        def transform(raw):
            return change_json(raw, lambda value: value.update(predecessor_native_admitted=True))
    files = mutate_payload(packet.files, name, transform)
    with pytest.raises((ValueError, AssertionError, KeyError)):
        replay(packet, files)


def test_relocated_full_replay_reads_only_copied_capsule(packet):
    with tempfile.TemporaryDirectory(prefix='o3-wrapper-relocated-', dir='/tmp') as temporary:
        copy = Path(temporary) / 'packet'
        shutil.copytree(HERE, copy)
        before = snapshot(copy)
        path_bytes, path_text = Path.read_bytes, Path.read_text
        builtin_open, io_open = builtins.open, io.open
        def allowed(path):
            if isinstance(path, int):
                raise AssertionError('Unexpected raw file descriptor')
            target = Path(path).absolute()
            if not target.is_relative_to(copy) or target.is_symlink():
                raise AssertionError('Recorded live path read: ' + str(target))
        def read_bytes(path, *args, **kwargs):
            allowed(path)
            return path_bytes(path, *args, **kwargs)
        def read_text(path, *args, **kwargs):
            allowed(path)
            return path_text(path, *args, **kwargs)
        def open_builtin(path, *args, **kwargs):
            allowed(path)
            return builtin_open(path, *args, **kwargs)
        def open_io(path, *args, **kwargs):
            allowed(path)
            return io_open(path, *args, **kwargs)
        with no_process_api(), pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, 'read_bytes', read_bytes)
            patch.setattr(Path, 'read_text', read_text)
            patch.setattr(builtins, 'open', open_builtin)
            patch.setattr(io, 'open', open_io)
            result = packet.verify(copy, expected_index_sha256=packet.index_sha)
        assert result == packet.result
        assert snapshot(copy) == before


def test_local_extra_and_symlink_roots_are_rejected(packet):
    with tempfile.TemporaryDirectory(prefix='o3-wrapper-membership-', dir='/tmp') as temporary:
        root = Path(temporary)
        alias = root / 'alias'
        alias.symlink_to(HERE, target_is_directory=True)
        with no_process_api(), pytest.raises(ValueError):
            packet.verify(alias, derive_summary=True)
        copy = root / 'copy'
        shutil.copytree(HERE, copy)
        (copy / 'unindexed-extra.txt').write_bytes(b'extra\n')
        with no_process_api(), pytest.raises(ValueError):
            packet.verify(copy, derive_summary=True)


def indexed_bytes(files, suffix):
    name = logical(files, suffix)
    row = json.loads(files[INDEX])['payloads'][name]
    raw = files[row['path']]
    return gzip.decompress(raw) if row['encoding'] == 'gzip+utf8' else raw


@pytest.fixture(scope='module')
def analysis(packet):
    raw = indexed_bytes(packet.files, 'source/source2/analysis.py')
    module = types.ModuleType('indexed_source2_o3_analysis')
    with no_process_api(), no_live_reads():
        exec(compile(raw, '<indexed:source/source2/analysis.py>', 'exec'), module.__dict__)
    return module


def row_text(rows, decimal=False):
    lines = []
    for row in rows:
        values = row['words']
        if decimal:
            values = [format(struct.unpack('>d', bytes.fromhex(word))[0], '.17e')
                      for word in values]
        lines.append(' '.join(map(str, (row['point'], row['order'], row['status']))) +
                     ' ' + ' '.join(values))
    return '\n'.join(lines) + '\n'


def history_rows(packet, analysis, case):
    if case == 'n1':
        text = indexed_bytes(packet.files,
                             'n1timing/candidate/full100/pair-1/timed-exact-words.dat').decode()
        return analysis.exact(text, analysis.N1_CODES), analysis.N1_CODES, None
    text = indexed_bytes(packet.files,
                         'n3anchortiming/candidate/QP/pair-1/anchor-exact.dat').decode()
    return analysis.exact(text, analysis.N3_CODES, point=16), analysis.N3_CODES, 16


def own_reports(analysis, reference, previous, actual):
    return {'qualified_parity': analysis.compare_own(reference, actual),
            'previous_batch': analysis.compare_own(previous, actual)}


@pytest.mark.parametrize('case', ['n1', 'QP_anchor'])
def test_legal_accuracy_only_history_drift_is_retained_as_diagnostic(packet, analysis, case):
    reference, codes, point = history_rows(packet, analysis, case)
    previous = json.loads(json.dumps(reference))
    actual = json.loads(json.dumps(reference))
    actual[0]['words'][4] = struct.pack('>d', 0.5).hex().upper()
    with no_process_api(), no_live_reads():
        if point is None:
            parsed = analysis.exact(row_text(actual), codes)
            assert analysis.decimal_exact(row_text(actual, decimal=True), codes) == parsed
        else:
            raw = {name: indexed_bytes(packet.files,
                                      'n3anchortiming/candidate/QP/pair-1/' + name).decode()
                   for name in ('anchor-timing.dat', 'anchor-status.dat', 'anchor-exact.dat',
                                'anchor-ieee.dat', 'anchor-values.dat')}
            raw['anchor-exact.dat'] = row_text(actual)
            raw['anchor-values.dat'] = row_text(actual, decimal=True)
            parsed = analysis.anchor_outputs(raw, codes, point=16, loops=1)['rows']
        stored = own_reports(analysis, reference, previous, parsed)
        replayed = packet.module.replay_own_reports(reference, previous, parsed, stored, analysis)
    assert replayed == stored
    assert all(report['passed'] and report['coeff_exact'] and report['status_equal']
               and report['accuracy_drift_count'] > 0 and report['accuracy_differences']
               for report in replayed.values())


@pytest.mark.parametrize('change', ['coefficient_ULP', 'status', 'nonfinite_accuracy', 'negative_accuracy'])
def test_own_history_gate_still_rejects_scientific_or_invalid_accuracy_changes(packet, analysis, change):
    reference, _, _ = history_rows(packet, analysis, 'QP_anchor')
    actual = json.loads(json.dumps(reference))
    if change == 'coefficient_ULP':
        actual[0]['words'][0] = f"{int(actual[0]['words'][0], 16) ^ 1:016X}"
    elif change == 'status':
        for row in actual:
            row['status'] = 219
    elif change == 'nonfinite_accuracy':
        actual[0]['words'][4] = '7FF0000000000000'
    else:
        actual[0]['words'][4] = 'BFF0000000000000'
    with no_process_api(), no_live_reads(), pytest.raises(ValueError):
        stored = own_reports(analysis, reference, reference, actual)
        packet.module.replay_own_reports(reference, reference, actual, stored, analysis)


def test_recomputed_diagnostic_accuracy_report_must_match_stored_report(packet, analysis):
    reference, _, _ = history_rows(packet, analysis, 'QP_anchor')
    actual = json.loads(json.dumps(reference))
    actual[0]['words'][4] = struct.pack('>d', 0.5).hex().upper()
    stored = own_reports(analysis, reference, reference, actual)
    stored['qualified_parity']['accuracy_drift_count'] += 1
    with no_process_api(), no_live_reads(), pytest.raises(ValueError):
        packet.module.replay_own_reports(reference, reference, actual, stored, analysis)


def test_same_event_decimal_raw_accuracy_join_remains_exact(packet, analysis):
    raw = {name: indexed_bytes(packet.files,
                              'n3anchortiming/candidate/QP/pair-1/' + name).decode()
           for name in ('anchor-timing.dat', 'anchor-status.dat', 'anchor-exact.dat',
                        'anchor-ieee.dat', 'anchor-values.dat')}
    lines = raw['anchor-values.dat'].splitlines()
    fields = lines[0].split()
    fields[-1] = '5.00000000000000000E-001'
    lines[0] = ' '.join(fields)
    raw['anchor-values.dat'] = '\n'.join(lines) + '\n'
    with no_process_api(), no_live_reads(), pytest.raises(ValueError, match='decimal/raw'):
        analysis.anchor_outputs(raw, analysis.N3_CODES, point=16, loops=1)
