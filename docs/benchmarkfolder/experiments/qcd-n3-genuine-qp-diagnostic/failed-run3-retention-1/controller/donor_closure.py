"""Create/verify private SONAME aliases; never load or execute a library."""
from pathlib import Path
import hashlib


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_manifest(manifest, runtime):
    require(manifest['schema'] == 'exact-donor-soname-aliases-v1', 'Unknown closure manifest')
    entries = manifest['entries']
    require(len(entries) == len(runtime) == 15, 'Exactly fifteen donor libraries required')
    require(len({x['soname'] for x in entries}) == 15, 'Duplicate SONAME alias')
    require({x['source_path']: x['sha256'] for x in entries} == runtime, 'Donor source identity differs')
    require(len({x['resolved_path'] for x in entries}) == 15, 'Duplicate resolved provider')
    for entry in entries:
        name = entry['soname']
        require(name and '/' not in name and name not in ('.', '..'), 'Unsafe SONAME')
        require(Path(entry['resolved_path']).is_absolute() and Path(entry['source_path']).is_absolute(),
                'Provider path must be absolute')


def verify(directory, manifest, runtime):
    directory = Path(directory)
    validate_manifest(manifest, runtime)
    require(not directory.is_symlink() and directory.is_dir(), 'Private runtime directory is not a directory')
    entries = {x['soname']: x for x in manifest['entries']}
    require({p.name for p in directory.iterdir()} == set(entries), 'Missing or extra runtime alias')
    checked = []
    for name, entry in sorted(entries.items()):
        target = directory / name
        source = Path(entry['source_path'])
        expected = entry['resolved_path']
        require(target.is_symlink() and str(target.readlink()) == expected, 'Alias target changed: ' + name)
        require(str(source.resolve(strict=True)) == expected and str(target.resolve(strict=True)) == expected,
                'Resolved donor provider changed: ' + name)
        require(digest(source) == entry['sha256'] and digest(target) == entry['sha256'],
                'Donor provider bytes changed: ' + name)
        checked.append({'soname': name, 'alias_target': str(target.readlink()),
                        'resolved_path': expected, 'sha256': entry['sha256']})
    return {'passed': True, 'libraries': checked, 'library_count': 15,
            'alias_directory': str(directory.resolve()),
            'scope': 'static alias/hash verification, not loader or amplitude execution'}


def create(directory, manifest, runtime):
    directory = Path(directory)
    validate_manifest(manifest, runtime)
    # Check every original provider before creating any private symlink.
    for entry in manifest['entries']:
        source = Path(entry['source_path'])
        require(str(source.resolve(strict=True)) == entry['resolved_path'] and
                digest(source) == entry['sha256'], 'Original provider changed')
    directory.mkdir(exist_ok=False)
    for entry in manifest['entries']:
        (directory / entry['soname']).symlink_to(entry['resolved_path'])
    return verify(directory, manifest, runtime)
