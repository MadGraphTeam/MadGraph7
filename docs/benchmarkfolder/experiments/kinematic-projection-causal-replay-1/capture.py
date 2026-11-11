"""Copy only the root-reviewed, closed replay receipts and source manifest."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/kinematic-projection-replay-executor-1'
SOURCE = ROOT / '.tirust-build/kinematic-projection-replay-proposal-1'
SOURCE_SHA = 'ba130c540e466550ad2e4fa560b0b8df75a93912c4d1b271ef4d60c5fd4571a9'
DEPENDENCIES = {
    'current-kinematic-trace-build-qualification-1': '89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033',
    'current-kinematic-trace-runtime-qualification-1': '730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4',
}
PINS = {
    'results/build-closure.json': '7b829cd965971060f21bf0938c8f1aeb42516d85ac3e36f851df9907bf47283f',
    'results/run-terminal-closure.json': '3a8e13486d4cd26425efefaceb7d6f2099be0a867a80e22fdb00fc3ff1e678d8',
    'results/root-build-review.json': '80522e0b9b94f921693e2b92273dee54711df1a8083ebb6d1b01245cff291642',
    'results/root-run-review.json': '01e2f35915a151cd5569da7f1b6c2f53d7672065ca8dcf46b8df69732cacfe85',
    'results/root-exact-gram-analysis.json': 'df69bf17225f942b8939558843da57b7156d652914890fa931b48af24f2222d3',
    'results/root-error-subset-gram-analysis.json': 'a18d54cc7b40c0a263bcf1a4ea61c88ce6f94cf3caf818203c6f5d0e382c40f8',
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
    index = {'schema': 'kinematic-projection-causal-retention-v1', 'files': {}, 'origins': {},
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
    for name, checksum in PINS.items():
        copy(WORK / name, name, checksum)
    closed = json.loads((WORK / 'results/run-terminal-closure.json').read_bytes())
    build = json.loads((WORK / 'results/build-closure.json').read_bytes())
    for key, name in [('before', 'build-before.json'), ('after', 'build-after.json'), ('build_receipt', 'build.json')]:
        copy(WORK / 'results' / name, 'results/' + name, build[key])
    # Explicit named build artifacts written by the sealed controller and reviewed
    # with its three commands. The old closure is copied, never extended in place.
    for name in ['prelink.json', 'build-runtime.json']:
        copy(WORK / 'results' / name, 'results/' + name)
    for label in BUILD_COMMANDS:
        name = 'results/execution/' + label
        receipt = json.loads(copy(WORK / (name + '.json'), name + '.json'))
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
    for name in ['README.md', 'capture.py', 'verify.py', 'math_checks.py']:
        index['files'][name] = pin((HERE / name).read_bytes())
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'payloads': len(index['files']), 'stored_bytes': sum(x['bytes'] for x in index['files'].values()),
                      'raw_origin_bytes': sum(x['bytes'] for x in index['origins'].values()),
                      'native_executions': 0, 'external_binary_reads': 0}, indent=2))


if __name__ == '__main__':
    main()
