"""Copy only the root-reviewed, closed replay receipts and source manifest."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/ew-refusal-replay-executor-1'
SOURCE = ROOT / '.tirust-build/ew-refusal-replay-proposal-1'
SOURCE_SHA = '9384471349f462e7f028161c5b9a1673027698d930e82ac7da0aa6910c08ad61'
DEPENDENCIES = {
    'current-kinematic-trace-build-qualification-1': '89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033',
    'current-kinematic-trace-runtime-qualification-1': '730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4',
}
PINS = {
    'results/build-closure.json': 'c195e4de9aa301fb3f17b00e3a89a06fca948352f359bbd212b1d6ac2dad75f3',
    'results/build-terminal-review.json': 'dc5e27e4a1fc13e8033ce90b8e20f0778efd58467d8797d91122457294443a87',
    'results/root-build-review.json': '70a5f4052a5b784bc21d87fc84c845d13974b88049c778ae5f92a72cc3075cd8',
    'results/run-terminal-closure.json': 'ec0dcaf70e5921fca8b32041bfde688430cf07c347b5ed31275ecb5845f1ce04',
    'results/root-run-review.json': '344db4e1c506486cba50faf4883c5aaf7ad877ef25a554708a0b6804676f67c4',
}

BUILD_COMMANDS = ('compile-driver', 'link-driver', 'query-build-0001')


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    if sys.argv[1:] != ['--execute'] or Path.cwd() != ROOT or (HERE / 'evidence-index.json').exists():
        raise ValueError('Fresh explicit capture from repository cwd required')
    for sibling, checksum in DEPENDENCIES.items():
        if pin((HERE.parent / sibling / 'evidence-index.json').read_bytes())['sha256'] != checksum:
            raise ValueError('Declared sibling changed')
    index = {'schema': 'ew-original-refusal-retention-v1', 'files': {}, 'origins': {},
             'compressed_text': {}, 'dependencies': DEPENDENCIES}

    def copy(path, logical, expected=None, compressed=True):
        raw = path.read_bytes()
        identity = pin(raw)
        if expected is not None and (identity['sha256'] != (expected if isinstance(expected, str) else expected['sha256'])
                or isinstance(expected, dict) and identity['bytes'] != expected['bytes']):
            raise ValueError('Closed origin changed: ' + str(path))
        raw.decode('utf-8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Only text may be retained')
        if logical in index['origins']:
            if index['origins'][logical]['sha256'] != identity['sha256']:
                raise ValueError('Conflicting repeated origin')
            return raw
        stored_name = 'compressed/' + logical + '.gz' if compressed else logical
        stored = gzip.compress(raw, compresslevel=6, mtime=0) if compressed else raw
        destination = HERE / stored_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(stored)
        index['files'][stored_name] = pin(stored)
        index['origins'][logical] = {'path': str(path.relative_to(ROOT)), **identity}
        if compressed:
            index['compressed_text'][logical] = {'path': stored_name, 'raw_bytes': len(raw), 'raw_sha256': identity['sha256']}
        return raw

    source = json.loads(copy(SOURCE / 'index.json', 'proposal/index.json', SOURCE_SHA, False))
    for name, identity in source['files'].items():
        copy(SOURCE / name, 'proposal/' + name, identity, Path(name).suffix not in {'.py', '.f90', '.words', '.md'})
    copy(ROOT / '.tirust-build/ew-refusal-replay-independent-review-1.json', 'reviews/source-review.json',
         '4e0e7beb6cd4b3f47f204a4fc3c784545f18e36988597637802d52c254bb7d0d')
    for name, checksum in PINS.items():
        copy(WORK / name, name, checksum)
    closed = json.loads((WORK / 'results/run-terminal-closure.json').read_bytes())
    build = json.loads((WORK / 'results/build-closure.json').read_bytes())
    for key, name in [('before', 'build-before.json'), ('after', 'build-after.json'), ('build_receipt', 'build.json')]:
        copy(WORK / 'results' / name, 'results/' + name, build[key])
    # Explicit named build artifacts written by the sealed controller and reviewed
    # with its three commands. The old closure is copied, never extended in place.
    build_terminal = json.loads((WORK / 'results/build-terminal-review.json').read_bytes())
    for name, key in [('prelink.json', 'prelink_receipt'), ('build-runtime.json', 'runtime_receipt')]:
        copy(WORK / 'results' / name, 'results/' + name, build_terminal[key])
    for label in BUILD_COMMANDS:
        name = 'results/execution/' + label
        expected = build_terminal['commands'][str((WORK / (name + '.json')).relative_to(ROOT))]
        receipt = json.loads(copy(WORK / (name + '.json'), name + '.json', expected['receipt']))
        copy(WORK / (name + '.txt'), name + '.txt', receipt['log_sha256'])
    for original, identities in closed['commands'].items():
        origin = ROOT / original
        logical = str(origin.relative_to(WORK))
        copy(origin, logical, identities['receipt'])
        copy(origin.with_suffix('.txt'), str(Path(logical).with_suffix('.txt')), identities['log'])
    for original, identity in closed['closed_receipts'].items():
        origin = ROOT / original
        copy(origin, str(origin.relative_to(WORK)), identity)
    for worker in closed['workers'].values():
        for original, identity in worker['files'].items():
            origin = ROOT / original
            copy(origin, str(origin.relative_to(WORK)), identity)
    for name in ['README.md', 'capture.py', 'verify.py', 'input_checks.py']:
        index['files'][name] = pin((HERE / name).read_bytes())
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'payloads': len(index['files']), 'stored_bytes': sum(x['bytes'] for x in index['files'].values()),
                      'raw_origin_bytes': sum(x['bytes'] for x in index['origins'].values()),
                      'native_executions': 0, 'external_binary_reads': 0}, indent=2))


if __name__ == '__main__':
    main()
