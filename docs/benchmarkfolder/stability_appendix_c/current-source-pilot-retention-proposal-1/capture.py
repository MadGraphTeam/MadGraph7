"""Capture one already closed, root-admitted current-source raw pilot.

Filesystem evidence only: no compiler, loader, numerical executable or lock.
The fixed allowlist is derived from the admitted five requests and exact names.
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
BASE = HERE.parent
WORK = ROOT / '.tirust-build/current-source-qcd3-raw-pilot-1'
REPORT = WORK / 'evidence'
PILOT = BASE / 'current-source-pilot-proposal-1'
SOURCE5 = BASE / 'source5'
CLASSIFIER = BASE / 'current-source-classifier-proposal-1'
CURRENT_SHA = 'c071ec1c87ce6d5034e8a2daf07f093e1c71581ba79bcaa1c9176ca0c79e05ef'
SOURCE5_SHA = 'ffbc57948b062e6380f71e22c9f303b0e652bcd4060952b7bc0a4fe216191271'
CLASSIFIER_SHA = '7bc7a08c2a96483100ea47c2d652694a34a4f426af2b5ba9a8317798d6003032'
CODEC_SHA = '33505d7db5186df76c974cf419b2333a8efbc1ce90ecee67a19c9edfd9de7a7a'
PREP_SHA = '9c1b45cdbf2783d2da63df8c55a744c2cb976fe4316ed715973dfbdae0e34d85'
DRY_SHA = '5bac8cc40c16eda6504764bbcb0a7675cd1fe6a819019bcb95843a595c4b50a5'
RUN_SHA = '4e6cc378cf14b9f9bab387efb02c63307a3f1d6dc2a1cbff03f4905a4d022fac'
CASE = 'qcd_n3_uux_zgg'
BACKENDS = (1, 6, 7, 8, 9)
RAW_NAMES = ('stability-dimensions.dat', 'stability-values.dat', 'stability-groups.dat')
MAP_NAMES = tuple('runtime-maps-%02d.txt' % n for n in range(17))
MAX_RAW = 512 * 1024 * 1024
LARGE_TEXT = 40 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def module(path, name):
    value = types.ModuleType(name); value.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), value.__dict__)
    return value


def regular(path):
    path = Path(path)
    require(path.parent.resolve() == path.parent and stat.S_ISREG(path.lstat().st_mode),
            'Exact regular, unaliased evidence required: ' + str(path))
    return path


def packet(path, expected):
    blob = regular(path / 'source-index.json').read_bytes()
    require(sha(blob) == expected, 'Source packet seal differs')
    value = json.loads(blob)
    for name, item in value['files'].items():
        require(name and not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe source name')
        data = regular(path / name).read_bytes()
        require(len(data) == item['bytes'] and sha(data) == item['sha256'], 'Source packet payload differs')
    return value


def capture(destination, *, expected_terminal_sha256, root_reported_exit_code):
    destination = Path(destination).absolute()
    require(destination.parent == BASE and destination.parent.resolve() == BASE,
            'Capture destination must be a fresh sibling in stability_appendix_c')
    require(not destination.exists() and not destination.is_symlink(), 'Fresh capture destination required')
    require(expected_terminal_sha256 == RUN_SHA and type(root_reported_exit_code) is int
            and root_reported_exit_code == 0, 'Exact root-admitted closed run required')
    packets = [(PILOT, CURRENT_SHA, 'pilot'), (SOURCE5, SOURCE5_SHA, 'source5'),
               (CLASSIFIER, CLASSIFIER_SHA, 'classifier')]
    source_indexes = [(path, packet(path, expected), prefix) for path, expected, prefix in packets]
    prepared_blob = regular(REPORT / 'prepare-terminal.json').read_bytes()
    dry_blob = regular(REPORT / 'dryrun-terminal.json').read_bytes()
    terminal_blob = regular(REPORT / 'run-terminal.json').read_bytes()
    require((sha(prepared_blob), sha(dry_blob), sha(terminal_blob)) == (PREP_SHA, DRY_SHA, RUN_SHA),
            'Admitted preparation/dryrun/run bytes differ')
    prepared, dry, terminal = map(json.loads, (prepared_blob, dry_blob, terminal_blob))
    expected_requests = [{'id': CASE + '-backend' + str(backend), 'case': CASE, 'backend': backend,
                          'lock_required': backend in (8, 9), 'primary_calls_expected': 300, 'bootstrap_calls_maximum': 16}
                         for backend in BACKENDS]
    require(terminal['status'] == dry['status'] == prepared['status'] == 'passed'
            and terminal['source_index_sha256'] == dry['source_index_sha256'] == prepared['source_index_sha256'] == CURRENT_SHA
            and [row['request'] for row in terminal['results']] == expected_requests
            and [row['request'] for row in dry['results']] == expected_requests,
            'Closed source/status/fixed-five schedule differs')
    require(sha(regular(HERE / 'journal_codec.py').read_bytes()) == CODEC_SHA, 'Frozen codec differs')
    codec = module(HERE / 'journal_codec.py', 'current_capture_codec')
    destination.mkdir(); payloads = {}; originals = {}

    def retain(path, logical, *, journal=False):
        path = regular(path)
        require(logical not in payloads, 'Duplicate logical payload')
        size = path.stat().st_size
        require(size <= MAX_RAW, 'Original exceeds bounded decompression budget')
        if journal or size > LARGE_TEXT:
            prefix = 'compressed/' + logical.replace('/', '__')
            descriptor = codec.encode_journal(path, destination / prefix, total_raw_budget=MAX_RAW)
            for chunk in descriptor['chunks']:
                chunk['path'] = prefix + '/' + chunk['path']
            item = {'kind': 'gzip-chunks', 'descriptor': descriptor,
                    'bytes': descriptor['raw_bytes'], 'sha256': descriptor['raw_sha256']}
        else:
            blob = path.read_bytes(); target = destination / logical
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(blob)
            item = {'kind': 'file', 'path': logical, 'bytes': len(blob), 'sha256': sha(blob)}
        payloads[logical] = item
        originals[str(path)] = {'logical_payload': logical, 'bytes': item['bytes'], 'sha256': item['sha256']}
        return item

    for name in ('prepare-started.json', 'prepare-inputs.json', 'prepare-terminal.json',
                 'dryrun-started.json', 'dryrun-terminal.json', 'run-started.json', 'run-terminal.json'):
        retain(REPORT / name, 'receipts/' + name)
    retain(REPORT / 'point-metrics.json', 'point-metrics.json')
    case = prepared['case']; donor = case['donor']; master = Path(case['master_snapshot']['root'])
    directory = master / donor['subprocess']
    retain(master / 'Cards/param_card.dat', 'cards/param_card.dat')
    for suffix in ('.dat', '.json'):
        retain(Path(donor['points']).with_suffix(suffix), 'points/' + CASE + suffix)
    closed = json.loads((PILOT / 'closed-input.json').read_bytes())
    qualification = {'manifest.json': ROOT / 'docs/benchmarkfolder/campaigns/entry-latch-cache-guard-1/manifest.json',
        'generation.json': Path(donor['export']) / 'generation.json',
        'compilation.json': Path(donor['export']) / 'compilation.json',
        'loop_matrix.f': directory / 'loop_matrix.f', 'worker_driver.f': directory / 'check_sa.f',
        'entry-template.inc': ROOT / 'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc',
        'case-checkpoint.json': ROOT / closed['checkpoint']['path'],
        'numerical-inputs.json': ROOT / closed['numerical_inputs']['path']}
    for name, path in qualification.items():
        retain(path, 'qualification/' + name)
    for key, subdirectory in (('command', 'prepare'), ('runtime_command', 'loader')):
        command = case[key]; prefix = 'build/' + subdirectory + '/'
        require(json.loads((REPORT / subdirectory / 'command.json').read_bytes()) == command,
                'Build/query command receipt differs')
        retain(REPORT / subdirectory / 'command.json', prefix + 'command.json')
        console = retain(REPORT / subdirectory / 'stdout.txt', prefix + 'stdout.txt')
        require(console['sha256'] == command['stdout_sha256'], 'Build/query stdout differs')
        for sample in command['runtime_map_samples']:
            require(sample['path'] in MAP_NAMES, 'Unsafe map sample name')
            item = retain(REPORT / subdirectory / sample['path'], prefix + sample['path'])
            require(item['sha256'] == sample['sha256'], 'Build/query map differs')
    observations = {}
    for record in terminal['results']:
        rid = record['request']['id']; worker = WORK / 'run' / rid
        canonical_card = worker / 'Cards/MadLoopParams.dat'
        mode = canonical_card.lstat().st_mode
        require(stat.S_ISREG(mode), 'This closed run requires retained canonical regular cards')
        item = retain(canonical_card, 'requests/' + rid + '/MadLoopParams.dat')
        observations[rid] = {'status': 'regular', 'bytes': item['bytes'], 'sha256': item['sha256']}
        for name, expected in record.get('raw_diagnostics', {}).items():
            require(name in RAW_NAMES, 'Unknown raw diagnostic name')
            path = worker / donor['subprocess'] / name
            require(str(path) == expected['path'], 'Raw diagnostic original path differs')
            item = retain(path, 'requests/' + rid + '/' + name, journal=name == 'stability-groups.dat')
            require((item['bytes'], item['sha256']) == (expected['bytes'], expected['sha256']), 'Raw diagnostic bytes differ')
        command = record['command']; evidence = REPORT / 'run' / rid; prefix = 'requests/' + rid + '/'
        require(json.loads((evidence / 'command.json').read_bytes()) == command, 'Native command receipt differs')
        retain(evidence / 'command.json', prefix + 'command.json')
        console = retain(evidence / 'stdout.txt', prefix + 'stdout.txt')
        require(console['sha256'] == command['stdout_sha256'], 'Native stdout differs')
        for sample in command['runtime_map_samples']:
            require(sample['path'] in MAP_NAMES, 'Unsafe native map name')
            item = retain(evidence / sample['path'], prefix + sample['path'])
            require(item['sha256'] == sample['sha256'], 'Native map differs')
    for path, index, prefix in source_indexes:
        for name in (*index['files'], 'source-index.json'):
            target = destination / prefix / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(regular(path / name), target)
    for name in ('verify.py', 'journal_codec.py'):
        shutil.copyfile(regular(HERE / name), destination / name)
    shutil.copyfile(regular(ROOT / 'tools/tirust_benchmark_points.py'), destination / 'point_validator.py')
    retention = {'schema': 'madloop-current-source-raw-stability-retention-v1',
        'source_index_sha256': CURRENT_SHA, 'preparation_sha256': PREP_SHA, 'dryrun_sha256': DRY_SHA,
        'run_terminal_sha256': RUN_SHA, 'root_reported_run_terminal': {'session_id': 29683, 'exit_code': root_reported_exit_code},
        'original_allowlist': originals, 'original_hashes_rechecked_after_capture': True,
        'scope': 'exact closed inputs only; no binary packaging or original tree walk; lossless bounded deterministic gzip',
        'host_precision_caption': 'host forced-DP/card-selected backend; reducer-internal precision and actual backend unobserved'}
    for spelling, item in originals.items():
        path = regular(spelling); hasher = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                hasher.update(block)
        require(path.stat().st_size == item['bytes'] and hasher.hexdigest() == item['sha256'], 'Closed original changed during capture')
    (destination / 'retention.json').write_text(json.dumps(retention, indent=2, sort_keys=True) + '\n')
    (destination / 'card-observations.json').write_text(json.dumps(observations, indent=2, sort_keys=True) + '\n')
    shutil.copyfile(HERE / 'README.md', destination / 'README.md')
    verifier = module(HERE / 'verify.py', 'current_capture_verifier')
    metrics = module(SOURCE5 / 'plot_metrics.py', 'current_capture_metrics')
    summary = verifier.summarize(json.loads((REPORT / 'point-metrics.json').read_bytes()), terminal['results'], metrics)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + '\n')
    files = {}
    for path in sorted(destination.rglob('*')):
        if path.is_file():
            blob = path.read_bytes(); files[str(path.relative_to(destination))] = {'bytes': len(blob), 'sha256': sha(blob)}
    index = {'schema': 'madloop-current-source-raw-stability-capsule-v1', 'files': files, 'payloads': payloads,
             'source_index_sha256': CURRENT_SHA, 'root_admitted_run_terminal_sha256': RUN_SHA}
    (destination / 'index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    result = verifier.verify(destination)
    return {'index_sha256': result['index_sha256'], 'files': len(index['files']),
            'physical_bytes': sum(item['bytes'] for item in index['files'].values()), 'verified': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--expected-terminal-sha256', required=True)
    parser.add_argument('--root-reported-exit-code', required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(capture(args.destination, expected_terminal_sha256=args.expected_terminal_sha256,
                            root_reported_exit_code=args.root_reported_exit_code), indent=2, sort_keys=True))
