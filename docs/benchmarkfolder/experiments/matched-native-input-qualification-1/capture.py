"""Text-only retention from explicit immutable sources and CLOSED matched-run lists."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/kinematic-matched-native-input-executor-2'
OLD_WORK = ROOT / '.tirust-build/kinematic-matched-native-input-executor-1'
SOURCES = {
    'science': ('.tirust-build/kinematic-matched-native-input-proposal-1', '357e116b51879cda1e1145c9d040194732c21ffd9fcd2557d9f68451901b262b'),
    'controller': ('.tirust-build/kinematic-matched-native-input-controller-2', '9076a51a0be56c57fb7dda32cfec3ef9b038bec2e73dc5e751bf31dc5ac3bc41'),
    'predecessor/controller': ('.tirust-build/kinematic-matched-native-input-controller-1', '9ff20966e4d24a3062a730a6037e9c9e50ab22d0007f343fdf6b4cee84cfaae7'),
}
DEPENDENCIES = {
    'kinematic-invariant-amplitude-qualification-1': 'c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4',
    'kinematic-legacy-reference-adjudication-1': '248bd61e987ac97c81e5397489eb4322ecfbf96cf85a42c84607aef2c013500e',
}
FAILED = 'd0343e3ec8f8b4db046a5fc3ac7c0dd2cace9c5a271fffb00394e1e7ca16af41'
CLOSURES = {
    'build': 'c09358a689b2b856b81580a7d0371b73a3d13edb3a173e9b8aed4b8f143ad00d',
    'run': '52faa470d8be22b81f8636edba21f5fba6c2d13be1f0241a8212a7b2055839a7',
}
OLD_EXTRAS = {
    'results/source-admission.json': '4228043d50fac1b355d90c3a1e1cf3493f398fd8ad014adc6a8f617525087ca7',
    'results/build-terminal-review.json': '8e6c66cce304c7496fc341cfc8f9167f9a6a75e548d148b5fa972a84e2cf041b',
}
EXTRAS = {
    'results/source-admission.json': '58b236112d91a369f8451cb7df475fb2e190f451310ca88bfe30162eba33a84e',
    'results/run-admission.json': '7eae41eaa58ab76a42c0e9a8b9eec35df81ffd1bee6c49455659fe0d82db60b7',
    'results/build-terminal-review.json': '09c5423d9416269d71bc64d2f4d3679b6caa8f798633ec95b9da341cd4c97a6c',
    'results/independent-build-review.json': 'e00970bbb5ce07ccb13e0a9d792a4281237bee9f1dacae67474a1e9bdea75b0d',
    'results/root-build-review.json': '9fdc7d3b456e5a2653539a996673a8921312de70ca4813b59428de445f795b96',
    'results/run-terminal-review.json': 'b71d3c19381b905416a5880022f51d7fe49866159f02235cd3cb61c82e114bce',
    'results/independent-run-review.json': 'e645593e6f8080595a17d95aeefeb7607db2fbc36c64d9cdd2c56605528a3778',
    'results/root-run-review.json': '9ca6601ab14a58d25eee4d77159805731e6c3d9ecd7264c364e636ed9501bf66',
    'results/prior-candidate-word-parity.json': '5f9f4cc6c8d2ad35944158cd7ad9fe30c8c0a91729b69456cc9528cf7d07c8e9',
}


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    if sys.argv[1:] != ['--execute'] or Path.cwd() != ROOT or (HERE / 'evidence-index.json').exists():
        raise ValueError('Fresh explicit capture from repository cwd required')
    if any('PENDING' in value for value in EXTRAS.values()):
        raise ValueError('All extra receipts must be closed and sealed')
    for name, seal in DEPENDENCIES.items():
        if pin((HERE.parent / name / 'evidence-index.json').read_bytes())['sha256'] != seal:
            raise ValueError('Public dependency changed')
    index = {'schema': 'matched-native-input-qualification-retention-v1', 'dependencies': DEPENDENCIES,
             'source_indices': {prefix: seal for prefix, (_, seal) in SOURCES.items()},
             'closed_stages': CLOSURES, 'failed_predecessor_build': FAILED,
             'files': {}, 'compressed_text': {}, 'origins': {}}

    def copy(path, logical, expected, compress=True):
        if Path(logical).is_absolute() or '..' in Path(logical).parts:
            raise ValueError('Unsafe closed logical path')
        raw = path.read_bytes(); identity = pin(raw)
        wanted = {'sha256': expected} if isinstance(expected, str) else expected
        if any(identity[key] != value for key, value in wanted.items()):
            raise ValueError('Closed payload changed: ' + str(path))
        raw.decode('utf-8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Only text may be retained')
        if logical in index['origins']:
            if identity != {k: index['origins'][logical][k] for k in ('bytes', 'sha256')}:
                raise ValueError('Repeated closed origin differs')
            return raw
        name = 'compressed/' + logical + '.gz' if compress else logical
        stored = gzip.compress(raw, compresslevel=6, mtime=0) if compress else raw
        destination = HERE / name; destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(stored)
        index['files'][name] = pin(stored)
        index['origins'][logical] = {'path': str(path.relative_to(ROOT)), **identity}
        if compress:
            index['compressed_text'][logical] = {'path': name, 'raw_bytes': len(raw), 'raw_sha256': identity['sha256']}
        return raw

    for prefix, (name, seal) in SOURCES.items():
        directory = ROOT / name
        source = json.loads(copy(directory / 'index.json', prefix + '/index.json', seal, False))
        for member, identity in source['files'].items():
            copy(directory / member, prefix + '/' + member, identity)
    failed = json.loads(copy(OLD_WORK / 'results/build-closure.json', 'predecessor/results/build-closure.json', FAILED))
    for name, identity in failed['payloads'].items():
        copy(OLD_WORK / 'results' / name, 'predecessor/results/' + name, identity)
    for name, seal in OLD_EXTRAS.items():
        copy(OLD_WORK / name, 'predecessor/' + name, seal)
    for stage, seal in CLOSURES.items():
        logical = 'results/' + stage + '-closure.json'
        closed = json.loads(copy(WORK / logical, logical, seal))
        for name, identity in closed['payloads'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe closed payload')
            copy(WORK / 'results' / name, 'results/' + name, identity)
    for name, seal in EXTRAS.items():
        copy(WORK / name, name, seal)
    for name in ('README.md', 'capture.py', 'verify.py', 'build_checks.py', 'worker_checks.py'):
        index['files'][name] = pin((HERE / name).read_bytes())
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'payloads': len(index['files']),
        'stored_bytes': sum(p['bytes'] for p in index['files'].values()),
        'original_text_bytes': sum(p['bytes'] for p in index['origins'].values()),
        'native_executions': 0, 'external_binary_reads': 0}, indent=2))


if __name__ == '__main__':
    main()
