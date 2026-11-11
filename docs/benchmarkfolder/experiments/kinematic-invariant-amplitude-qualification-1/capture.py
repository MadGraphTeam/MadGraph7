"""Retain sealed sources and explicit CLOSED stage lists; no native/binary reads."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/kinematic-invariant-amplitude-executor-2'
OLD_WORK = ROOT / '.tirust-build/kinematic-invariant-amplitude-executor-1'
SOURCE = ROOT / '.tirust-build/kinematic-invariant-amplitude-proposal-2'
OLD_SOURCE = ROOT / '.tirust-build/kinematic-invariant-amplitude-proposal-1'
SOURCES = {
    'proposal': (SOURCE, 'ccbb3dff88f673043251dac5d5099d184f709fdffdba1e5f2c3c1dc0aca37a74'),
    'predecessor/proposal': (OLD_SOURCE, '286ecfe5f5fab199dee34e4401cb5154ba1212870eaf2b921d3f44536c251b0e'),
}
DEPENDENCIES = {
    'current-kinematic-trace-build-qualification-1': '89697913341ecc5fe765fb023cc09bced783eb3eedb85436afe1ca7d4dc6c033',
    'current-kinematic-trace-runtime-qualification-1': '730f8f220c17676287dda092a19fb24fbca2073d1ecb0e7ceb44a3a16b5402e4',
    'caller-bound-native-qualification-1': '51e1ba911d1bbe19c83141777f76fc967b0c876630fa3cef60f818adda205174',
}
CLOSURES = {
    'build': '8cab371e9affd5297c0f48c30184361959d20f3d3a48c6dbd6109b86bf49bacb',
    'ordinary': 'b364bd8535d0f5850cb5a21390c5678cbb1c1dd8fddfae6f0174cf50d261f7a7',
    'stress': 'c696421eb4948b4d2de5993cc84455b1ac04d0f659b245341729714f91ae8482',
}
REVIEWS = {
    'results/build-terminal-review.json': '9191a4513b79c5f55c0a34bde328c71516ad82b64ce0f81d098af2d41965b44d',
    'results/root-build-review.json': 'f517efdb36ab976bcebdc1612a45d4e1b5ad77c08d9835e276355be6e0c0e70c',
    'results/ordinary-terminal-review.json': 'a0ae797b55a2c9d620246fdafd74f4120ff9a132101d016be926724c1203289f',
    'results/root-ordinary-review.json': 'bc2b479c61cbec2b1f8a7df743dab3653dc5388f93870825bd5f1d3c4c91eed3',
    'results/stress-terminal-review.json': '84e365f3a5338acc3b35457a0d7a841e0e53743debc4a5710ae03bc9e948bfff',
    'results/stress-active-owned-sample.json': '22440ae66d2fc64e8a2318ce8e03629667a146c48dccc4141c48aa146484dce3',
    'results/root-stress-review.json': '725af61aa460a1f1f02f98032be989c2daa707664ef8c408043b40685cd4f579',
    'results/root-point20-comparison.json': '7b2390b98193dd11c66ae8eae64ec7b36f5497accd9e2e355de03ccfacda5d6d',
}
COMPARISON_SOURCES = {
    'tools/tirust_benchmark.py': {'bytes': 96661, 'sha256': 'bc405b44d8d794cc4396ef3da3d48af238ac4dfb619c58901d7bab910a7e1370'},
    'tests/parallel_tests/loop_me_comparator.py': {'bytes': 74750, 'sha256': 'b6da773dadbe1e202639a865cbe3f29f65af8147a250a8bb05e756d537548354'},
}


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    if sys.argv[1:] != ['--execute'] or Path.cwd() != ROOT or (HERE / 'evidence-index.json').exists():
        raise ValueError('Fresh explicit retention from repository cwd required')
    if any(value.startswith('PENDING') for value in [*CLOSURES.values(), *REVIEWS.values()]):
        raise ValueError('All stages and reviews must be CLOSED and sealed before retention')
    for sibling, checksum in DEPENDENCIES.items():
        if pin((HERE.parent / sibling / 'evidence-index.json').read_bytes())['sha256'] != checksum:
            raise ValueError('Declared sibling changed')
    index = {'schema': 'kinematic-invariant-amplitude-retention-v1', 'files': {}, 'compressed_text': {},
             'origins': {}, 'dependencies': DEPENDENCIES, 'closed_stages': CLOSURES}

    def copy(origin, logical, expected, compress=True):
        raw = origin.read_bytes()
        identity = pin(raw)
        if identity['sha256'] != (expected if isinstance(expected, str) else expected['sha256']) or (
                isinstance(expected, dict) and identity['bytes'] != expected['bytes']):
            raise ValueError('Closed origin changed: ' + str(origin))
        raw.decode('utf-8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Only text may be retained')
        if logical in index['origins']:
            if {k: index['origins'][logical][k] for k in ('bytes', 'sha256')} != identity:
                raise ValueError('Conflicting repeated closed origin')
            return raw
        stored_name = 'compressed/' + logical + '.gz' if compress else logical
        stored = gzip.compress(raw, compresslevel=6, mtime=0) if compress else raw
        destination = HERE / stored_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(stored)
        index['files'][stored_name] = pin(stored)
        index['origins'][logical] = {'path': str(origin.relative_to(ROOT)), **identity}
        if compress:
            index['compressed_text'][logical] = {'path': stored_name, 'raw_bytes': len(raw), 'raw_sha256': identity['sha256']}
        return raw

    for prefix, (directory, checksum) in SOURCES.items():
        source = json.loads(copy(directory / 'index.json', prefix + '/index.json', checksum, False))
        for name, identity in source['files'].items():
            copy(directory / name, prefix + '/' + name, identity)
    failed = json.loads(copy(OLD_WORK / 'results/build-closure.json', 'predecessor/results/build-closure.json',
        '99cbb544826b741c030708ce4ab5bc01db83eac0ec49c4dce02a6be6858d76aa'))
    copy(OLD_WORK / 'results/build-terminal-review.json', 'predecessor/results/build-terminal-review.json',
         'b504c91987b6939bea4b2ebf2e4d9ab122f88ecfe1cf76c0a83628cdf1901564')
    for name, identity in failed['payloads'].items():
        copy(OLD_WORK / 'results' / name, 'predecessor/results/' + name, identity)
    for stage, checksum in CLOSURES.items():
        name = 'results/' + stage + '-closure.json'
        closed = json.loads(copy(WORK / name, name, checksum))
        for relative, identity in closed['payloads'].items():
            path = Path(relative)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Unsafe closed payload')
            copy(WORK / 'results' / path, 'results/' + relative, identity)
    for name, checksum in REVIEWS.items():
        copy(WORK / name, name, checksum)
    for original, identity in COMPARISON_SOURCES.items():
        copy(ROOT / original, 'comparison-sources/' + original, identity)
    for name in ['README.md', 'capture.py', 'verify.py', 'build_checks.py', 'worker_checks.py']:
        index['files'][name] = pin((HERE / name).read_bytes())
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'payloads': len(index['files']), 'stored_bytes': sum(p['bytes'] for p in index['files'].values()),
                      'original_bytes': sum(p['bytes'] for p in index['origins'].values()),
                      'native_executions': 0, 'external_binary_reads': 0}, indent=2))


if __name__ == '__main__':
    main()
