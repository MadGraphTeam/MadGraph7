#!/usr/bin/env python3
"""Copy the explicit closed first-failure diagnostic inventory; no native work."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
EXPERIMENT = HERE.parent
WORK1 = ROOT / '.tirust-build/tirust-first-failure-capture-1'
WORK2 = ROOT / '.tirust-build/tirust-first-failure-replay-2'
SOURCE1 = EXPERIMENT / 'source-proposal-1'
SOURCE2 = EXPERIMENT / 'source-proposal-2'
PREPARE_SHA = '41ddbc70654dac71134bfc11a1892e8795786d869b603abe76f5bcf735b51309'
CAPTURE_SHA = '7f21e3bd9e2b8bf0458c0c6008c500bc941786337523f863c5d64dec9c72f05e'
REPLAY_SHA = 'ba0d8d043c54572fdc1f465129a9b3fdb899786422fdd863af74d3d37f140c01'


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def closed_inventory():
    """Declare files from exact terminal records; never walk a work directory."""
    originals = {}
    def add(path, logical, checksum=None):
        path = Path(path)
        require(path.is_absolute() and path.is_file(), 'Missing declared closed file: ' + str(path))
        blob = path.read_bytes()
        require(not blob.startswith((b'\x7fELF', b'!<arch>\n')) and len(blob) <= 40 * 1024 * 1024,
                'Unexpected binary or oversized text payload')
        require(checksum is None or sha(blob) == checksum, 'Closed file identity differs: ' + str(path))
        require(str(path) not in originals or originals[str(path)]['logical'] == logical, 'Duplicate original mapping')
        originals[str(path)] = {'logical': logical, 'sha256': sha(blob), 'bytes': len(blob)}
    def source(directory, prefix):
        index = read(directory / 'source-index.json')
        add(directory / 'source-index.json', prefix + '/source-index.json')
        for name, expected in index['files'].items():
            add(directory / name, prefix + '/' + name, expected['sha256'])
    def command(evidence, prefix, receipt):
        require(read(evidence / 'command.json') == receipt, 'Command differs from its closed terminal')
        add(evidence / 'command.json', prefix + '/command.json')
        add(evidence / 'stdout.txt', prefix + '/stdout.txt', receipt['stdout_sha256'])
        require(not receipt['cleanup']['survivors'], 'Native owned process survived cleanup')
        for sample in receipt['runtime_map_samples']:
            require(Path(sample['path']).name == sample['path'], 'Unsafe declared runtime-map name')
            add(evidence / sample['path'], prefix + '/' + sample['path'], sample['sha256'])
    prep_path = WORK1 / 'evidence/prepare-terminal.json'
    capture_path = WORK1 / 'evidence/capture-terminal.json'
    replay_path = WORK2 / 'evidence/replay-terminal.json'
    add(prep_path, 'receipts/prepare-terminal.json', PREPARE_SHA)
    add(capture_path, 'receipts/capture-terminal.json', CAPTURE_SHA)
    add(replay_path, 'receipts/replay-terminal.json', REPLAY_SHA)
    add(WORK1 / 'evidence/capture-started.json', 'receipts/capture-started.json')
    add(WORK2 / 'evidence/replay-started.json', 'receipts/replay-started.json')
    add(WORK2 / 'evidence/additive-admission.json', 'receipts/additive-admission.json')
    add(ROOT / '.tirust-build/tirust-first-failure-readiness-1/proof.json',
        'receipts/readiness-1.json', '56b4117f13eecee52b07da1858d940edd055e7952111399ddfa6092aeacfce7b')
    add(ROOT / '.tirust-build/tirust-first-failure-replay-readiness-2/proof.json',
        'receipts/replay-readiness-2.json', '18cb676644201fd52e3808ea3fc1d57742d6a5a11c6e0e523aefe97ec98e518d')
    add(ROOT / '.tirust-build/tirust-first-failure-closed-validation-1/helfilter-proof.json',
        'receipts/helfilter-proof.json', '230d1fc54362032ebcc47befdc2ee7f33165a1d9d76942a68d4c9fa5caa0c18d')
    source(SOURCE1, 'source')
    source(SOURCE2, 'source2')
    prepare, capture, replay = read(prep_path), read(capture_path), read(replay_path)
    for result in capture['results']:
        case = result['case']
        prepared = prepare['cases'][case]
        command(WORK1 / 'evidence/prepare' / case, 'prepare/' + case, prepared['command'])
        for label, runtime in prepared['capture_runtimes'].items():
            command(WORK1 / 'evidence/prepare-loader' / case / label,
                    'prepare-loader/' + case + '/' + label, runtime['command'])
        command(WORK1 / 'evidence/capture' / case, 'capture/' + case, result['command'])
        directory = Path(result['command']['cwd'])
        for name in ('stability-values.dat', 'stability-dimensions.dat', 'stability-groups.dat', 'tirust-capture-events.jsonl'):
            path = directory / name
            if path.is_file():
                add(path, 'capture/' + case + '/outputs/' + name)
        for row in result['inputs']:
            if row['state'] == 'captured':
                add(Path(row['path']), 'capture/' + case + '/outputs/' + row['name'], row['sha256'])
        worker = WORK1 / 'capture' / case
        for name in ('MadLoopParams.dat', 'param_card.dat', 'ident_card.dat'):
            add(worker / 'Cards' / name, 'capture/' + case + '/cards/' + name)
        master_dir = Path(prepared['master_snapshot']['root']) / prepared['donor']['subprocess']
        for name in ('TIRUST_interface.f', 'TIR_interface.f', 'check_sa.f', 'observer.c', 'observer.f90',
                     'replay.c', 'tirust.h', 'capture.mk', 'capture-link.map'):
            add(master_dir / name, 'private-sources/' + case + '/' + name)
        points = Path(prepared['donor']['points'])
        add(points, 'points/' + case + '.dat')
        add(points.with_suffix('.json'), 'points/' + case + '.json')
    add(WORK1 / 'capture/qcd_n3_uux_zgg/SubProcesses/MadLoop5_resources/ML5_0_HelFilter.dat',
        'capture/qcd_n3_uux_zgg/outputs/ML5_0_HelFilter.dat',
        '7fe9630e61964b5a90af92e2a76b48b14d75fd4925e95fefa40fa1ce1a644ee6')
    for result in replay['results']:
        case = result['case']
        command(WORK2 / 'evidence/replay' / case, 'replay/' + case, result['command'])
        directory = Path(result['command']['cwd'])
        for name in ('request-manifest.txt', 'tirust-replay.jsonl'):
            add(directory / name, 'replay/' + case + '/outputs/' + name)
    return originals


def capture(destination):
    require(not destination.exists() and not destination.is_symlink(), 'Fresh retention destination required')
    source_index = read(HERE / 'source-index.json')
    for name, expected in source_index['files'].items():
        require(Path(name).name == name, 'Unsafe retention source path')
        blob = (HERE / name).read_bytes()
        require(expected == {'sha256': sha(blob), 'bytes': len(blob)}, 'Retention source differs: ' + name)
    originals = closed_inventory()
    prior = read(SOURCE2 / 'prior-attempt.json')
    prior.update(logical='source2/prior-attempt.txt')
    registry = {'schema': 'madloop-tirust-first-failure-retention-registry-v1', 'originals': originals,
                'capture_terminal_confirmation': {'observed_terminal': True, 'session_id': 62191,
                    'exit_code': 1, 'sha256': CAPTURE_SHA},
                'prepare_terminal_confirmation': {'observed_terminal': True, 'session_id': 86009,
                    'exit_code': 0, 'sha256': PREPARE_SHA},
                'replay_status': 'closed', 'replay_admission_confirmation': prior,
                'replay_terminal_confirmation': {'observed_terminal': True, 'session_id': 94497,
                    'exit_code': 0, 'sha256': REPLAY_SHA},
                'original_capture_failure_is_not_overridden': True,
                'missing_inputs_are_not_reconstructed': True, 'native_retries': 0}
    payloads = {item['logical']: Path(original).read_bytes() for original, item in originals.items()}
    payloads['retention.json'] = encode(registry)
    spec = importlib.util.spec_from_file_location('closed_capture_verifier', HERE / 'verify.py')
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    payloads['summary.json'] = encode(verifier.derive_closed(payloads, registry))
    payloads['README.md'] = (HERE / 'CLOSED_README.md').read_bytes()
    for name in ('verify.py', 'test_verify.py', 'capture.py', 'CLOSED_README.md', 'source-index.json'):
        payloads['retention-source/' + name] = (HERE / name).read_bytes()
    require(len(payloads) == len(set(payloads)), 'Duplicate retention logical paths')
    for original, item in originals.items():
        require(sha(Path(original).read_bytes()) == item['sha256'], 'Original changed during retention')
    index = {'schema': 'madloop-tirust-first-failure-capture-closed-v1',
             'files': {name: {'sha256': sha(blob), 'bytes': len(blob)} for name, blob in sorted(payloads.items())},
             'inputs': {'source1_index_sha256': '2b4bac95022fe550902164a0636699f46798bf4733185610d8f9829f6cd3880d',
                        'source2_index_sha256': 'bb990d419677675095d5c230b189a3986d428283e5c3df74edb2907e071a9734',
                        'original_raw_pilot_index_sha256': 'ebd8f1faf5228f205b1038a490329037066988dc03e7e774210d648b38dbbb6e'},
             'no_native_execution_during_retention': True}
    destination.mkdir(parents=True)
    for name, blob in payloads.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(blob)
    with (destination / 'index.json').open('xb') as stream:
        stream.write(encode(index))
    verified = verifier.verify(destination)
    print(json.dumps({'verified': verified['verified'], 'index_sha256': verified['index_sha256'],
                      'payloads': len(payloads), 'bytes': sum(map(len, payloads.values()))}, sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    capture(args.destination)
