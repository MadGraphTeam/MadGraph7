"""Retain only the explicit reviewed synthetic text; never read native binaries."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.tirust-build/spacelike-kernel-native-qualification-1'
SOURCE = ROOT / '.tirust-build/kinematic-invariant-general-proposal-1'
REVIEW_SHA = '8b887fc6d9f24f2585cdcbe658401f0bae8f4e4343e323da0a55e68b6e63f281'
SOURCE_SHA = '1665eff3188c947f35673b958cd0f02525fd031dc814b708dfdda69160b57964'


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    if sys.argv[1:] != ['--execute'] or Path.cwd() != ROOT or (HERE / 'evidence-index.json').exists():
        raise ValueError('Fresh explicit retention from repository cwd required')
    review_raw = (WORK / 'independent-review.json').read_bytes()
    if pin(review_raw)['sha256'] != REVIEW_SHA:
        raise ValueError('Reviewed closed attempt changed')
    review = json.loads(review_raw)
    index = {'schema': 'spacelike-kernel-synthetic-retention-v1', 'files': {}, 'origins': {},
             'compressed_text': {}, 'source_proposal_sha256': SOURCE_SHA, 'review_sha256': REVIEW_SHA}

    def copy(origin, logical, identity):
        raw = origin.read_bytes()
        if pin(raw) != identity:
            raise ValueError('Reviewed origin changed: ' + logical)
        raw.decode('utf-8')
        if b'\0' in raw or raw.startswith((b'\x7fELF', b'!<arch>')):
            raise ValueError('Only text may be retained')
        compressed = logical in {'inputs.txt', 'run.stdout'}
        stored_name = 'compressed/' + logical + '.gz' if compressed else logical
        stored = gzip.compress(raw, compresslevel=6, mtime=0) if compressed else raw
        path = HERE / stored_name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(stored)
        index['files'][stored_name] = pin(stored)
        index['origins'][logical] = {'path': str(origin.relative_to(ROOT)), **identity}
        if compressed:
            index['compressed_text'][logical] = {'path': stored_name, 'raw_bytes': len(raw), 'raw_sha256': identity['sha256']}

    copy(WORK / 'independent-review.json', 'independent-review.json', pin(review_raw))
    for name, identity in review['reviewed_files'].items():
        copy(WORK / name, name, identity)
    for name, identity in review['source_provenance'].items():
        copy(SOURCE / name, 'source/' + name, identity)
    for name in ('README.md', 'capture.py', 'verify.py'):
        index['files'][name] = pin((HERE / name).read_bytes())
    with (HERE / 'evidence-index.json').open('x') as stream:
        json.dump(index, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'payloads': len(index['files']), 'stored_bytes': sum(p['bytes'] for p in index['files'].values()),
                      'original_bytes': sum(p['bytes'] for p in index['origins'].values()),
                      'native_executions': 0, 'external_binary_reads': 0}, indent=2))


if __name__ == '__main__':
    main()
