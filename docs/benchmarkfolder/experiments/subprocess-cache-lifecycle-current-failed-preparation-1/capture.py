#!/usr/bin/env python3
"""Retain the closed failed preparation as portable UTF-8; binaries are hashes only."""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
REPORT = ROOT / 'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-qualification-1'
PROTOCOL = ROOT / 'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-source-proposal-1'
OUTPUT = ROOT / '.tirust-build/subprocess-cache-lifecycle-current-1/export'
SOURCE_INDEX = '8fc12d8160810016ed302d28fafe936830a3bd3c12cb00d1d5afdc049af2ebff'
STAGES = ('freeze', 'materialize', 'generate', 'prepare-build', 'build')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    assert Path.cwd() == ROOT
    assert not (HERE / 'payload.json.gz').exists()
    records, blobs, inventory = {}, {}, {}

    def retain(name, raw, origin):
        raw.decode('utf-8')
        assert b'\0' not in raw
        digest = sha(raw)
        records[name] = {'sha256': digest, 'bytes': len(raw), 'origin': origin}
        blobs[digest] = raw.decode('utf-8')

    stages = {name: json.loads((REPORT / ('stage-' + name + '.json')).read_text()) for name in STAGES}
    assert [s['status'] for s in stages.values()] == ['passed'] * 4 + ['failed']
    assert all(s['source_index_sha256'] == SOURCE_INDEX and not s['native_execution'] for s in stages.values())
    assert not (REPORT / 'stage-smoke.json').exists() and not (REPORT / 'stage-run.json').exists()
    assert sha((PROTOCOL / 'source-index.json').read_bytes()) == SOURCE_INDEX
    before = {}
    for label, base in (('protocol', PROTOCOL), ('report', REPORT), ('generated', OUTPUT)):
        for path in sorted(base.rglob('*')):
            if not path.is_file() and not path.is_symlink():
                continue
            name = label + '/' + str(path.relative_to(base))
            item = {'kind': 'link' if path.is_symlink() else 'file'}
            if path.is_symlink():
                item['target'] = str(path.readlink())
            if path.is_file():
                raw = path.read_bytes()
                item.update(sha256=sha(raw), bytes=len(raw), mode=oct(path.stat().st_mode & 0o777))
                before[str(path)] = item['sha256']
                try:
                    assert b'\0' not in raw and path.suffix not in ('.pkl', '.pyc', '.o', '.a', '.mod')
                    retain(name, raw, str(path))
                    item['retained_utf8'] = True
                except (UnicodeDecodeError, AssertionError):
                    assert label == 'generated', 'Non-text protocol/report file'
                    item['retained_utf8'] = False
            inventory[name] = item
    # Exact wrapper/search chain implicated in the failure; read-only evidence.
    paths = (
        '/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/nix-support/cc-ldflags',
        '/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/nix-support/add-flags.sh',
        '/nix/store/xc0ga87wdclrx54qjaryahkkmkmqi9qz-gcc-15.2.0-lib/lib/libgcc_s.so',
        '/nix/store/wy205ik03yp1vl55il5i332nbc285n0i-gfortran-15.2.0-libgcc/lib/libgcc_s.so')
    for n, spelling in enumerate(paths):
        path = Path(spelling)
        retain('linker-chain/' + str(n) + '-' + path.name, path.read_bytes(), spelling)
    for path, expected in before.items():
        assert sha(Path(path).read_bytes()) == expected, 'Closed source changed during capture'
    package = {'schema': 1, 'kind': 'closed-failed-current-lifecycle-preparation',
               'records': records, 'blobs': blobs, 'generated_inventory': inventory}
    packed = gzip.compress(json.dumps(package, sort_keys=True, separators=(',', ':')).encode(), mtime=0)
    (HERE / 'payload.json.gz').write_bytes(packed)
    summary = {'schema': 1, 'source_index_sha256': SOURCE_INDEX, 'stages': stages,
               'stage_elapsed_sum_seconds': sum(s['elapsed_seconds'] for s in stages.values()),
               'generation_succeeded': True, 'build_qualification_succeeded': False,
               'native_execution_performed': False, 'payload_sha256': sha(packed),
               'retained_paths': len(records), 'unique_utf8_contents': len(blobs),
               'binary_contents_retained': False,
               'cause': 'Pinned Nix compiler wrapper injects additional GCC library search directories; libgcc_s.so and its transitive members were absent from pre-link inventory. Strict actual LOAD check rejected the linked plain executable before loader/native execution.'}
    (HERE / 'summary.json').write_text(json.dumps(summary, sort_keys=True, indent=2) + '\n')
    print(json.dumps({k: summary[k] for k in ('retained_paths', 'unique_utf8_contents', 'stage_elapsed_sum_seconds', 'payload_sha256')}, indent=2))


if __name__ == '__main__':
    main()
