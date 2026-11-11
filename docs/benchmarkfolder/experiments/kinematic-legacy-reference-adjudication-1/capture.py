"""Retain explicit CLOSED legacy-reference source/build/run text; no native reads."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / '.tirust-build/kinematic-invariant-reference-adjudication-proposal-1'
WORK = ROOT / '.tirust-build/kinematic-invariant-reference-adjudication-executor-1'
SOURCE_SEAL = '0a0dba9baf12ff5a988e4d459141a48140f285a396658ee10f7f09099848dbd8'
PARENT = 'kinematic-invariant-amplitude-qualification-1'
PARENT_SEAL = 'c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4'
CLOSURES = {
    'build': '08a5b485954e20dc25fcffc2c26edddd73520b982e9daa380e1d5eca06709f22',
    'run': 'b94dca1a42b7acd2c835ce6b1bf04fe9036b0d64755ec1348240235a76b06462',
}
REVIEWS = {
    'results/build-terminal-review.json': '3f2ecb8eb90ac3b3e6b8730a4fbf17e0bba76cff57d624ae37666b4b6230bf72',
    'results/independent-build-review.json': '2e59e4146439991a9ceb1d7b53b6c36824bc82a51b341174befbbb051d5ebea9',
    'results/run-terminal-review.json': 'ae4d531856143c775222efc3464acd8da9efc43c456c41cec091e60a572da5a9',
    'results/independent-run-review.json': '7fd4c73c86880e95ecebe5dc8a1171db317e6d6803be36d863dcd6af1f75a710',
    'results/root-review.json': '1ec1a16b7deed7f80e81ad32cc17ac6405d536efffdf78a25d6222798e0e50bb',
}


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    if sys.argv[1:] != ['--execute'] or Path.cwd() != ROOT or (HERE / 'evidence-index.json').exists():
        raise ValueError('Fresh explicit capture from repository cwd required')
    if any('PENDING' in name or 'PENDING' in seal for name, seal in REVIEWS.items()):
        raise ValueError('All reviews must be closed and sealed')
    if pin((HERE.parent / PARENT / 'evidence-index.json').read_bytes())['sha256'] != PARENT_SEAL:
        raise ValueError('Public parent seal differs')
    index = {'schema': 'legacy-reference-adjudication-retention-v1', 'source_index_sha256': SOURCE_SEAL,
             'dependencies': {PARENT: PARENT_SEAL}, 'closed_stages': CLOSURES,
             'files': {}, 'compressed_text': {}, 'origins': {}}

    def copy(path, logical, expected, compress=True):
        relative = Path(logical)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe closed logical path')
        raw = path.read_bytes()
        identity = pin(raw)
        wanted = {'sha256': expected} if isinstance(expected, str) else expected
        if any(identity[key] != value for key, value in wanted.items()):
            raise ValueError('Closed payload changed: ' + str(path))
        raw.decode('utf-8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Only text may be retained')
        if logical in index['origins']:
            if identity != {k: index['origins'][logical][k] for k in ('bytes', 'sha256')}:
                raise ValueError('Repeated closed identity differs')
            return raw
        name = 'compressed/' + logical + '.gz' if compress else logical
        stored = gzip.compress(raw, compresslevel=6, mtime=0) if compress else raw
        destination = HERE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(stored)
        index['files'][name] = pin(stored)
        index['origins'][logical] = {'path': str(path.relative_to(ROOT)), **identity}
        if compress:
            index['compressed_text'][logical] = {'path': name, 'raw_bytes': len(raw), 'raw_sha256': identity['sha256']}
        return raw

    source = json.loads(copy(SOURCE / 'index.json', 'source/index.json', SOURCE_SEAL, False))
    for name, identity in source['files'].items():
        copy(SOURCE / name, 'source/' + name, identity)
    for stage, seal in CLOSURES.items():
        logical = 'results/' + stage + '-closure.json'
        closed = json.loads(copy(WORK / logical, logical, seal))
        for name, identity in closed['payloads'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe closed payload')
            copy(WORK / 'results' / name, 'results/' + name, identity)
    for name, seal in REVIEWS.items():
        copy(WORK / name, name, seal)
    for name in ('README.md', 'capture.py', 'verify.py', 'build_checks.py', 'worker_checks.py'):
        index['files'][name] = pin((HERE / name).read_bytes())
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'payloads': len(index['files']),
                      'stored_bytes': sum(p['bytes'] for p in index['files'].values()),
                      'original_text_bytes': sum(p['bytes'] for p in index['origins'].values()),
                      'native_executions': 0, 'external_binary_reads': 0}, indent=2))


if __name__ == '__main__':
    main()
