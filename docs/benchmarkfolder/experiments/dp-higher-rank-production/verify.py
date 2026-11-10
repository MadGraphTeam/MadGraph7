#!/usr/bin/env python3
"""Verify portable DP-terminal evidence; optionally compare staged Git bytes."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import struct
import runpy


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracked', action='store_true')
    parser.add_argument('--live-sources', action='store_true')
    args = parser.parse_args()
    folder = Path(__file__).resolve().parent
    root = folder.parents[3]
    index = folder / 'evidence-index.json'
    records = json.loads(index.read_text())['files']
    if any(p.is_symlink() for p in folder.rglob('*')):
        raise ValueError('Evidence tree contains a symlink')
    expected = set(records) | {index.name}
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    if actual != expected:
        raise ValueError('Unexpected or missing evidence: ' + repr(actual ^ expected))
    hashes = {}
    for name, record in records.items():
        path = folder / name
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe evidence path: ' + name)
        if any(part.is_symlink() for part in [path, *path.parents] if part != root):
            raise ValueError('Symlink evidence: ' + name)
        data = path.read_bytes()
        if b'\0' in data or data.startswith(b'\x7fELF'):
            raise ValueError('Binary evidence: ' + name)
        if len(data) != record['bytes'] or digest(data) != record['sha256']:
            raise ValueError('Evidence hash/size mismatch: ' + name)
        hashes[path.relative_to(root).as_posix()] = digest(data)
    hashes[index.relative_to(root).as_posix()] = digest(index.read_bytes())
    summary = json.loads((folder / 'validation-summary.json').read_text())
    for name, expected_sha in summary['source_and_test_sha256'].items():
        if digest((folder / 'source-snapshot' / name).read_bytes()) != expected_sha:
            raise ValueError('Source snapshot mismatch: ' + name)
        if args.live_sources and digest((root / name).read_bytes()) != expected_sha:
            raise ValueError('Live source mismatch: ' + name)
    compiled = json.loads((folder / 'compiled/qualification-summary.json').read_text())
    if not compiled['passed'] or not compiled['sources_unchanged']:
        raise ValueError('Final compiled test did not pass on unchanged sources')
    for name, checksum in summary['source_and_test_sha256'].items():
        if name in compiled['source_sha256_before'] and compiled['source_sha256_before'][name] != checksum:
            raise ValueError('Compiled-test/source-summary binding mismatch: '+name)
    if compiled['source_sha256_after'] != compiled['source_sha256_before']:
        raise ValueError('Compiled-test before/after sources differ')
    control = json.loads((folder / 'cuttools/control/raw/observation.json').read_text())
    candidate = json.loads((folder / 'cuttools/candidate/raw/observation.json').read_text())
    protocol = Path(summary['observation_protocol'])
    if protocol.is_absolute() or '..' in protocol.parts:
        raise ValueError('Unsafe parser path')
    parse_outputs = runpy.run_path(str(folder / protocol))['parse_outputs']
    for name, observation in (('control', control), ('candidate', candidate)):
        if parse_outputs(folder / 'cuttools' / name / 'raw', [1], 100, 1) != observation:
            raise ValueError('Stored observation differs from reparsed raw files: '+name)
    if len(control['rows']) != len(candidate['rows']) or not control['rows']:
        raise ValueError('Numerical observation row counts differ or are empty')
    for left, right in zip(control['rows'], candidate['rows'], strict=True):
        for name in ('point', 'order', 'status', 'actual_backend', 'precision'):
            if left[name] != right[name]:
                raise ValueError('Numerical observation identities/statuses differ')
        if len(left['coefficients']) != 4 or len(right['coefficients']) != 4:
            raise ValueError('Unexpected coefficient shape')
        if any(struct.pack('!d', a) != struct.pack('!d', b)
               for a, b in zip(left['coefficients'], right['coefficients'], strict=True)):
            raise ValueError('Numerical coefficient bits differ')
    comparison = json.loads((folder / 'cuttools/comparison.json').read_text())
    if (not comparison['bitwise_equal'] or not comparison['status_equal'] or comparison['point_count'] != 100
            or not comparison['retained_committed_control_bit_and_status_equal']):
        raise ValueError('Numerical comparison did not pass')
    retained = json.loads((folder / 'source-order-attempt/cuttools/control/raw/observation.json').read_text())
    for left, right in zip(retained['rows'], control['rows'], strict=True):
        if any(left[key] != right[key] for key in ('point', 'order', 'status', 'actual_backend', 'precision')):
            raise ValueError('Retained committed control status differs')
        if any(struct.pack('!d', a) != struct.pack('!d', b)
               for a, b in zip(left['coefficients'], right['coefficients'], strict=True)):
            raise ValueError('Retained committed control coefficient differs')
    if args.tracked:
        result = subprocess.run(['git', 'ls-files', '--stage', '-z', '--', *hashes],
            cwd=root, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        staged = {}
        for entry in result.stdout.split(b'\0'):
            if not entry:
                continue
            header, name = entry.split(b'\t', 1)
            mode, oid, stage = header.decode().split()
            if stage != '0' or mode not in ('100644', '100755'):
                raise ValueError('Nonregular/unmerged staged evidence')
            staged[name.decode()] = oid
        if set(staged) != set(hashes):
            raise ValueError('Evidence not staged: ' + repr(set(hashes) - set(staged)))
        oids = sorted(set(staged.values()))
        result = subprocess.run(['git', 'cat-file', '--batch'], cwd=root, check=True,
            input=('\n'.join(oids) + '\n').encode(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        stream = io.BytesIO(result.stdout)
        blobs = {}
        for oid in oids:
            returned, kind, size = stream.readline().decode().split()
            if returned != oid or kind != 'blob':
                raise ValueError('Unexpected Git object')
            data = stream.read(int(size))
            if len(data) != int(size) or stream.read(1) != b'\n':
                raise ValueError('Truncated Git object')
            blobs[oid] = digest(data)
        for name, expected_sha in hashes.items():
            if blobs[staged[name]] != expected_sha:
                raise ValueError('Staged bytes differ: ' + name)
    print(json.dumps({'status': 'passed', 'portable_files': len(records),
        'staged_bytes_checked': args.tracked, 'live_sources_checked': args.live_sources,
        'scope': 'Artifact integrity and source/test bindings; no numerical rerun.'},
        indent=2))


if __name__ == '__main__':
    main()
