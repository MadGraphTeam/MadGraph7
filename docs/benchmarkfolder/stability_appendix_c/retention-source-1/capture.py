"""Retain one explicitly closed raw pilot; requires root-supplied terminal SHA.

This command never runs a compiler, loader or amplitude. It opens only exact
closed receipt/raw/point paths selected from the frozen twenty-request plan.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT / '.tirust-build/stability-appendix-c-pilot-5'
SOURCE = HERE.parent / 'source5'
SOURCE_SHA = 'ffbc57948b062e6380f71e22c9f303b0e652bcd4060952b7bc0a4fe216191271'
PREPARATION_SHA = '1588341143201f03700361de79b9bfa43e2dd4485d5cb0fd9887cccddbbef682'
LARGE_TEXT = 40 * 1024 * 1024
RAW_BUDGET = 512 * 1024 * 1024


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def module(path, name):
    result = types.ModuleType(name)
    result.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), result.__dict__)
    return result


def card_observation(path):
    """Describe a canonical post-run card endpoint without following aliases."""
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return {'status': 'missing'}
    if stat.S_ISLNK(mode):
        return {'status': 'symlink', 'readlink': str(path.readlink())}
    return {'status': 'regular' if stat.S_ISREG(mode) else 'nonregular'}


def capture(destination, *, expected_terminal_sha256, root_reported_exit_code):
    destination = Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError('A new capture namespace is required')
    source_index_blob = (SOURCE / 'source-index.json').read_bytes()
    if sha(source_index_blob) != SOURCE_SHA:
        raise ValueError('Frozen source5 seal differs')
    source_index = json.loads(source_index_blob)
    for name, item in source_index['files'].items():
        blob = (SOURCE / name).read_bytes()
        if len(blob) != item['bytes'] or sha(blob) != item['sha256']:
            raise ValueError('Frozen source5 payload differs: ' + name)
    pilot = module(SOURCE / 'pilot.py', 'capture_frozen_pilot')
    metrics = module(SOURCE / 'plot_metrics.py', 'capture_frozen_metrics')
    codec = module(HERE / 'journal_codec.py', 'capture_journal_codec')
    verifier = module(HERE / 'verify.py', 'capture_verifier')
    value = pilot.protocol()
    report = WORK / 'evidence'
    terminal_path = report / 'run-terminal.json'
    terminal_blob = terminal_path.read_bytes()
    if sha(terminal_blob) != expected_terminal_sha256:
        raise ValueError('The root-admitted closed run terminal differs')
    terminal = json.loads(terminal_blob)
    prepared_blob = (report / 'prepare-terminal.json').read_bytes()
    prepared = json.loads(prepared_blob)
    if (terminal['source_index_sha256'] != SOURCE_SHA or terminal['request_count'] != 20
            or terminal['preparation_sha256'] != PREPARATION_SHA or sha(prepared_blob) != PREPARATION_SHA
            or prepared['source_index_sha256'] != SOURCE_SHA or prepared['status'] != 'passed'
            or [row['request'] for row in terminal['requests']] != pilot.requests(value)):
        raise ValueError('Closed run/preparation/source/request identities differ')
    destination.mkdir(parents=True)
    payloads = {}; originals = {}

    def retain(path, logical, *, journal=False):
        path = Path(path)
        if not path.is_file() or path.is_symlink():
            raise ValueError('Only exact regular closed inputs are captured: ' + str(path))
        size = path.stat().st_size
        if size > RAW_BUDGET:
            raise ValueError('Closed text input exceeds the declared decompression budget')
        if journal or size > LARGE_TEXT:
            prefix = 'compressed/' + logical.replace('/', '__')
            descriptor = codec.encode_journal(path, destination / prefix, total_raw_budget=RAW_BUDGET)
            for chunk in descriptor['chunks']:
                chunk['path'] = prefix + '/' + chunk['path']
            item = {'kind': 'gzip-chunks', 'descriptor': descriptor,
                    'bytes': descriptor['raw_bytes'], 'sha256': descriptor['raw_sha256']}
        else:
            blob = path.read_bytes()
            target = destination / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(blob)
            item = {'kind': 'file', 'path': logical, 'bytes': len(blob), 'sha256': sha(blob)}
        payloads[logical] = item
        originals[str(path)] = {'logical_payload': logical, 'bytes': item['bytes'], 'sha256': item['sha256']}
        return item

    for filename in ('inherit-started.json', 'inherit-terminal.json', 'prepare-terminal.json',
                     'run-started.json', 'run-terminal.json'):
        retain(report / filename, 'receipts/' + filename)
    retain(report / 'point-metrics.json', 'point-metrics.json')
    for case in pilot.CASE_IDS:
        case_record = prepared['cases'][case]; donor = case_record['donor']
        physical = Path(case_record['master_snapshot']['root']) / 'Cards/param_card.dat'
        item = retain(physical, 'cards/' + case + '/param_card.dat')
        if item['sha256'] != donor['physical_card_sha256']:
            raise ValueError('Frozen physical parameter-card bytes differ')
        prefix = ROOT / '.tirust-build/stability-appendix-c-pilot-3/points' / case
        if Path(donor['points']) != prefix.with_suffix('.dat'):
            raise ValueError('The inherited physical point path differs')
        for suffix in ('.dat', '.json'):
            retain(prefix.with_suffix(suffix), 'points/' + case + suffix)
    card_observations = {}
    for record in terminal['requests']:
        rid = record['request']['id']; evidence = report / 'requests' / rid
        receipt = evidence / 'result.json'
        if receipt.is_file():
            if json.loads(receipt.read_bytes()) != record:
                raise ValueError('Closed request/terminal receipt differs')
            retain(receipt, 'requests/' + rid + '/result.json')
        elif record['state'] != 'unexecuted' or record['native_started']:
            raise ValueError('An executed request lacks its closed receipt')
        if 'effective_card' in record:
            path = WORK / 'workers' / rid / 'Cards/MadLoopParams.dat'
            observation = card_observation(path)
            card_observations[rid] = observation
            if observation['status'] == 'regular':
                card = retain(path, 'cards/' + rid + '/MadLoopParams.dat')
                observation.update(sha256=card['sha256'], bytes=card['bytes'])
                if (record.get('readonly_guard_passed') is True
                        and any(checksum != card['sha256'] for checksum in record['effective_card']['files'].values())):
                    raise ValueError('Closed effective numerical-card bytes differ')
            elif record.get('readonly_guard_passed') is True:
                raise ValueError('Guard-passed request lost its canonical numerical card')
        for filename, expected in record.get('raw_files', {}).items():
            if filename not in ('stability-values.dat', 'stability-dimensions.dat', 'stability-groups.dat'):
                raise ValueError('Unexpected collected raw filename')
            item = retain(evidence / filename, 'requests/' + rid + '/' + filename,
                          journal=filename == 'stability-groups.dat')
            if item['sha256'] != expected['sha256'] or item['bytes'] != expected['size']:
                raise ValueError('Collected raw input differs from its closed receipt')
        if record['native_started']:
            native = evidence / 'native'; prefix = 'requests/' + rid + '/native/'
            command = json.loads((native / 'command.json').read_bytes())
            if command != record['native']:
                raise ValueError('Native command receipt differs')
            retain(native / 'command.json', prefix + 'command.json')
            stdout = retain(native / 'stdout.txt', prefix + 'stdout.txt')
            if stdout['sha256'] != command['stdout_sha256']:
                raise ValueError('Closed native console differs')
            for sample in command['runtime_map_samples']:
                if Path(sample['path']).name != sample['path']:
                    raise ValueError('Unsafe map sample name')
                item = retain(native / sample['path'], prefix + sample['path'])
                if item['sha256'] != sample['sha256']:
                    raise ValueError('Closed native map differs')
    # Scientific source packet and parser are local to the capsule. External
    # source5 input paths remain historical pins and are never followed by replay.
    for name in (*source_index['files'], 'source-index.json'):
        target = destination / 'source5' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE / name, target)
    for name in ('verify.py', 'journal_codec.py'):
        shutil.copyfile(HERE / name, destination / name)
    validator = ROOT / 'tools/tirust_benchmark_points.py'
    if sha(validator.read_bytes()) != source_index['inputs']['tools/tirust_benchmark_points.py']:
        raise ValueError('Frozen physical-point validator differs')
    shutil.copyfile(validator, destination / 'point_validator.py')
    dataset = json.loads((report / 'point-metrics.json').read_bytes())
    summary = verifier.summarize_records(dataset, terminal['requests'], metrics)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    for spelling, item in originals.items():
        path = Path(spelling); hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        if path.stat().st_size != item['bytes'] or hasher.hexdigest() != item['sha256']:
            raise ValueError('A closed original changed during capture')
    retention = {'schema': 'madloop-raw-stability-retention-v1', 'root_reported_run_terminal':
        {'session_id': 46098, 'exit_code': root_reported_exit_code},
        'original_allowlist': originals, 'original_hashes_rechecked_after_capture': True,
        'scope': 'exact closed paths only; no worker-tree discovery or binaries; lossless bounded gzip for group journals and text larger than40MiB',
        'source5_index_sha256': SOURCE_SHA, 'run_terminal_sha256': expected_terminal_sha256,
        'host_precision_caption': 'host forced-DP/card-selected backend; reducer-internal precision and actual backend unobserved'}
    (destination / 'retention.json').write_text(json.dumps(retention, indent=2, sort_keys=True) + '\n')
    (destination / 'card-observations.json').write_text(json.dumps(card_observations, indent=2, sort_keys=True) + '\n')
    shutil.copyfile(HERE / 'README.md', destination / 'README.md')
    files = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            blob = path.read_bytes()
            files[str(path.relative_to(destination))] = {'sha256': sha(blob), 'bytes': len(blob)}
    index = {'schema': 'madloop-raw-stability-capsule-v1', 'files': files, 'payloads': payloads,
        'source5_index_sha256': SOURCE_SHA, 'root_admitted_run_terminal_sha256': expected_terminal_sha256,
        'interpretation': summary['interpretation']}
    (destination / 'index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    return {'index_sha256': sha((destination / 'index.json').read_bytes()), 'files': len(files),
            'physical_bytes': sum(item['bytes'] for item in files.values())}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--expected-terminal-sha256', required=True)
    parser.add_argument('--root-reported-exit-code', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(capture(args.destination, expected_terminal_sha256=args.expected_terminal_sha256,
        root_reported_exit_code=args.root_reported_exit_code), indent=2, sort_keys=True))
