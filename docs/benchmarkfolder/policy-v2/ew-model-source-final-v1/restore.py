#!/usr/bin/env python3
"""Restore the checksum-bound benchmark EW model without downloads or imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_payload():
    inventory = json.loads((HERE / 'publication-inventory.json').read_text())
    if inventory['schema'] != 'benchmark-ew-model-source-publication-v1':
        raise ValueError('Unexpected model publication inventory')
    record = inventory['actual_source_archive']
    filename = Path(record['path']).name
    if filename != 'loop_qcd_qed_sm_Gmu-actual-source.tar.gz':
        raise ValueError('Unexpected actual-source archive name')
    archive = HERE / filename
    data = archive.read_bytes()
    if len(data) != record['bytes'] or digest(data) != record['sha256']:
        raise ValueError('Actual-source archive checksum differs from inventory')
    source_files = inventory['source_files']
    model = inventory['model']
    if model != 'loop_qcd_qed_sm_Gmu' or len(source_files) != 176:
        raise ValueError('Unexpected complete model source inventory')
    expected_files = {model + '/' + name for name in source_files}
    directories = {model}
    for name in expected_files:
        directories.update(str(p) for p in PurePosixPath(name).parents if str(p) != '.')
    payload, observed = {}, set()
    with tarfile.open(archive, 'r:gz') as stream:
        for member in stream.getmembers():
            name = member.name
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or name in observed:
                raise ValueError('Unsafe or duplicate model archive member: ' + name)
            observed.add(name)
            if member.isdir():
                if name not in directories:
                    raise ValueError('Unexpected model archive directory: ' + name)
                continue
            if not member.isfile() or name not in expected_files:
                raise ValueError('Unexpected non-source archive member: ' + name)
            relative = name[len(model) + 1:]
            record = source_files[relative]
            extracted = stream.extractfile(member)
            if extracted is None:
                raise ValueError('Unreadable source archive member: ' + name)
            contents = extracted.read()
            if (len(contents) != record['bytes'] or digest(contents) != record['sha256']
                    or stat.S_IMODE(member.mode) != record['archive_mode']):
                raise ValueError('Source member checksum or mode differs: ' + name)
            payload[relative] = contents
    if observed != expected_files | directories or set(payload) != set(source_files):
        raise ValueError('Incomplete or additional model archive members')
    names = sorted(name for name in payload if '/' not in name and name.endswith('.py'))
    if names != inventory['original_python_checksum']['hashed_files']:
        raise ValueError('Top-level UFO Python inventory differs')
    aggregate = hashlib.sha256()
    for name in names:
        aggregate.update(name.encode('utf-8'))
        aggregate.update(b'\0')
        aggregate.update(payload[name])
    if aggregate.hexdigest() != inventory['original_python_checksum']['sha256']:
        raise ValueError('Original UFO Python aggregate checksum differs')
    return inventory, payload, aggregate.hexdigest()


def restore(target):
    # Verify all archive bytes before creating files or directories at target.
    inventory, payload, aggregate = checked_payload()
    target = Path(target)
    if not target.is_absolute():
        target = ROOT / target
    if target.is_symlink():
        raise ValueError('Refuse to overwrite a symlink target')
    target = target.resolve()
    if not target.is_relative_to(ROOT):
        raise ValueError('Restore target must remain inside this repository')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError('Restore target must be absent or an empty directory')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.ew-model-restore-', dir=target.parent))
    try:
        for name, contents in sorted(payload.items()):
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(contents)
            path.chmod(inventory['source_files'][name]['archive_mode'])
            os.utime(path, (0, 0))
        stage.chmod(0o755)
        # POSIX rename also rejects a directory populated after the check.
        os.replace(stage, target)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return dict(target=str(target.relative_to(ROOT)), files=len(payload),
                archive_sha256=inventory['actual_source_archive']['sha256'],
                python_sources_sha256=aggregate, network_calls=0, model_imports=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', required=True,
                        help='Repository-relative model directory; must be absent or empty')
    args = parser.parse_args()
    print(json.dumps(restore(args.target), sort_keys=True))
