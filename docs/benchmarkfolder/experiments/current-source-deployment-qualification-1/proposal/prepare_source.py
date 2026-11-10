"""One-use source/prefix copy for the bounded deployment smoke; no native execution."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import shutil
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
BIND = json.loads((HERE / 'bindings.json').read_text())
ROOT = Path(BIND['root'])
WORK = Path(BIND['work'])
REVISION = '0fcd2cd873d61d134837e1d9e87afb13a47824fd'
ROOTS = ['Template', 'aloha', 'bin', 'input', 'madgraph', 'models', 'vendor',
         'VERSION', 'LICENSE', 'HELAS', 'MadSpin', 'PLUGIN', 'madmatrix',
         'madspace', 'mg5decay']


def require(value, message):
    if not value:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for data in iter(lambda: stream.read(1048576), b''):
            h.update(data)
    return h.hexdigest()


def write(name, value):
    with (WORK / name).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def protected():
    for name, row in BIND['tools'].items():
        if name.startswith('env:'):
            continue
        path=Path(row['path'])
        require(str(path.resolve())==row['resolved'] and digest(path)==row['sha256'],
                'Tool drift: '+name)
    for name, row in BIND['protected_before'].items():
        path = Path(name)
        if row['kind'] == 'absent':
            require(not path.exists() and not path.is_symlink(), name)
        elif row['kind'] == 'symlink':
            require(path.is_symlink() and os.readlink(path) == row['target'], name)
            require((digest(path) if path.is_file() else None) == row['target_sha256'], name)
        else:
            require(path.is_file() and not path.is_symlink() and digest(path) == row['sha256'], name)


def main():
    require(Path.cwd().resolve() == ROOT, 'Run from the required repository root')
    require(not (WORK / 'mg').exists() and not (WORK / 'source-prefixes').exists(), 'Already prepared')
    protected()
    names = BIND['candidate']['exporter_identity']['files']
    for name, expected in names.items():
        path = ROOT / name
        require((digest(path) if path.is_file() else None) == expected, 'Live source drift: ' + name)
    command = [BIND['tools']['git']['path'], 'archive', REVISION, '--', *ROOTS]
    archive = subprocess.check_output(command, cwd=ROOT, timeout=120)
    with (WORK / 'source.tar').open('xb') as stream:
        stream.write(archive)
    copied, excluded, symlinks, dangling = {}, {}, {}, {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        members = {item.name: item for item in tar.getmembers()}

        def content(name, seen=()):
            require(name not in seen, 'Archive symlink cycle')
            require(not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts,
                    'Unsafe archive path')
            item = members[name]
            if item.issym():
                target = posixpath.normpath(posixpath.join(posixpath.dirname(name), item.linkname))
                return content(target, seen + (name,))
            require(item.isfile(), 'Unsupported archive member')
            return tar.extractfile(item).read()

        for name, item in sorted(members.items()):
            if item.isdir():
                continue
            require(item.isfile() or item.issym(), 'Unexpected archive object')
            try:
                data = content(name)
            except KeyError:
                require(item.issym() and names.get(name) is None, 'Unregistered missing archive target')
                dangling[name] = item.linkname
                continue
            if name.startswith('vendor/IREGI/'):
                destination = WORK / 'omitted-optional' / name.removeprefix('vendor/')
                excluded[name] = hashlib.sha256(data).hexdigest()
            else:
                destination = WORK / 'mg' / name
                copied[name] = hashlib.sha256(data).hexdigest()
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open('xb') as stream:
                stream.write(data)
            destination.chmod(item.mode if item.isfile() else 0o755)
            if item.issym():
                symlinks[name] = item.linkname
    for name, expected in names.items():
        require((copied | excluded).get(name) == expected, 'Archive differs from registered exporter: ' + name)
    prefixes = WORK / 'source-prefixes'
    prefixes.mkdir()
    shutil.copytree(Path(BIND['candidate']['native']['prefix']), prefixes / 'tirust', symlinks=True)
    shutil.copytree(ROOT / 'HEPTools/ninjars', prefixes / 'ninjars', symlinks=True)
    for directory in ['xdg-config', 'xdg-state', 'xdg-cache', 'matplotlib', 'logs', 'results']:
        (WORK / directory).mkdir()
    config = WORK / 'mg/input/mg7_configuration.txt'
    require(not config.exists(), 'Archived configuration unexpectedly exists')
    config.write_text((WORK / 'mg/input/.mg7_configuration_default.txt').read_text()
                      + '\nauto_update = 0\nautomatic_html_opening = False\n')
    write('preparation.json', {'revision': REVISION, 'archive_command': command,
        'archive_sha256': hashlib.sha256(archive).hexdigest(), 'copied_files': copied,
        'excluded_optional_iregi': excluded, 'materialized_symlinks': symlinks,
        'dangling_generated_links': dangling, 'private_configuration_sha256': digest(config),
        'source_prefix_files': {str(path.relative_to(prefixes)): {'sha256': digest(path),
            'symlink': os.readlink(path) if path.is_symlink() else None}
            for path in sorted(prefixes.rglob('*')) if path.is_file()},
        'native_or_compiler_executed': False})
    protected()
    print(json.dumps({'prepared': True, 'copied': len(copied), 'excluded_iregi': len(excluded),
                      'dangling': dangling}, sort_keys=True))


if __name__ == '__main__':
    main()
