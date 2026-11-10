#!/usr/bin/env python3
"""Retain a closed run4 without following its library symlinks or executing code."""
from pathlib import Path
import gzip
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = HERE.parent
ATTEMPT = ROOT / '.tirust-build/qcd-n3-genuine-qp-diagnostic/run4'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    if (HERE / 'evidence-index.json').exists():
        raise FileExistsError('Sealed retention is immutable')
    result = json.loads((ATTEMPT / 'result.json').read_bytes())
    if result.get('passed') is not False or result.get('failure') != {
            'type': 'ValueError', 'message': 'malformed decimal transport'}:
        raise ValueError('Unexpected closed failure')
    originals, links, directories = {}, {}, []

    def capture(original, logical, *,generated_text=False):
        if original.is_symlink() and not generated_text:
            raise ValueError('Capture must not follow a symlink')
        raw = original.read_bytes()
        if raw.startswith(b'\x7fELF') or raw.startswith(b'!<arch>\n'):
            raise ValueError('No binary or archive retention: ' + str(original))
        compressed = len(raw) > 1000000
        member = logical + ('.gz' if compressed else '')
        output = gzip.compress(raw, compresslevel=9, mtime=0) if compressed else raw
        target = HERE / member
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes()!=output:
                raise ValueError('Previously captured bytes differ: '+member)
        else:
            with target.open('xb') as stream:
                stream.write(output)
        originals[logical] = {'original_path': str(original.relative_to(ROOT)),
                              'member': member, 'codec': 'gzip' if compressed else 'identity',
                              'raw_bytes': len(raw), 'raw_sha256': digest(raw),
                              'stored_bytes': len(output), 'stored_sha256': digest(output)}
        if original.is_symlink():
            originals[logical]['original_text_symlink_target']=str(original.readlink())

    for path in sorted(ATTEMPT.rglob('*')):
        relative = path.relative_to(ATTEMPT).as_posix()
        if path.is_symlink():
            links[relative] = str(path.readlink())
        elif path.is_dir():
            directories.append(relative)
        elif path.is_file():
            capture(path, 'actual-run4/' + relative)
        else:
            raise ValueError('Unknown attempt file type')
    source = BASE / 'source-proposal4'
    source_index = json.loads((source / 'source-index.json').read_bytes())
    for name in [*source_index['files'], 'source-index.json']:
        capture(source / name, 'controller/' + name)
    binding = json.loads((source / 'bindings.json').read_bytes())
    process = Path(binding['process'])
    for name in ('loop_matrix.f', 'born_matrix.f', 'improve_ps.f', 'coupl.inc', 'mp_coupl.inc',
                 'mp_compute_loop_coefs.f', 'loop_max_coefs.inc'):
        capture(process / name, 'generated-source/' + name,generated_text=True)
    capture(BASE / 'build2-static/evidence-index.json', 'inherited-build2-static-index.json')
    dump('retention-inventory.json', {
        'schema': 'closed-qcd-n3-run4-retention-inventory-v1', 'original_files': originals,
        'actual_attempt_symlinks': links, 'actual_attempt_directories': directories,
        'symlink_targets_retained_as_metadata_only': True, 'no_dso_or_executable_copies': True,
        'original_paths_are_metadata_not_replay_inputs': True})
    import verify
    summary = verify.replay({name: (gzip.decompress((HERE / item['member']).read_bytes())
                                    if item['codec'] == 'gzip' else (HERE / item['member']).read_bytes())
                             for name, item in originals.items()},
                            json.loads((HERE / 'retention-inventory.json').read_bytes()))
    dump('summary.json', summary)
    files = {}
    for path in sorted(HERE.rglob('*')):
        if path.is_file():
            raw = path.read_bytes()
            files[path.relative_to(HERE).as_posix()] = {'bytes': len(raw), 'sha256': digest(raw)}
    dump('evidence-index.json', {'schema': 'qcd-n3-genuine-qp-failed-run4-retention-v1',
                                'scope': 'complete closed failed native attempt; not adjudication acceptance',
                                'files': files})
    print(json.dumps(verify.verify(), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
