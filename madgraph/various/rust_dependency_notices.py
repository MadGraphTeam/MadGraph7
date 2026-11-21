"""Preserve the notices and declarations of the sources resolved by Cargo.

This module records upstream documents; it neither interprets their terms nor
changes dependency selection.  The resulting regular files can accompany both
an installation and a self-contained MadLoop export.
"""

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile

from madgraph import MadGraph5Error


_DOCUMENT = re.compile(r'^(?:licen[cs]e|notice|copying|copyright|authors|readme)s?(?:$|[._-])',
                       re.IGNORECASE)
_SKIP_DIRS = frozenset(('.git', '.hg', '.svn', 'target', '__pycache__', '.direnv'))


def _filenames(backend):
    if not isinstance(backend, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', backend):
        raise MadGraph5Error('Invalid Rust dependency notice backend name: %r' % backend)
    return [backend + '-dependency-notices.tar.gz',
            backend + '-dependency-provenance.json', backend + '-NOTICES.txt']


def export_files(libdir, backend, expected=None):
    """Return notice basenames, rejecting an incomplete existing notice bundle.

    Older installations without this bundle remain usable.  Once any notice
    artifact exists all three are required, so exports cannot silently omit it.
    """
    names = _filenames(backend)
    paths = [Path(libdir) / name for name in names]
    present = [path.exists() or path.is_symlink() for path in paths]
    if not any(present) and expected is None:
        return []
    if not all(present) or any(not path.is_file() or not path.stat().st_size
                               for path in paths):
        raise MadGraph5Error('Incomplete %s dependency notices in %s; reinstall the backend.'
                             % (backend, libdir))
    if expected is not None:
        if (not isinstance(expected, dict) or expected.get('files') != names or
                not isinstance(expected.get('sha256'), dict) or
                set(expected['sha256']) != set(names) or
                any(_sha256(path.read_bytes()) != expected['sha256'][name]
                    for name, path in zip(names, paths))):
            raise MadGraph5Error('Invalid or modified %s dependency notices in %s; reinstall the backend.'
                                 % (backend, libdir))
    return names


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _source_root(directory):
    """Find the nearest Rust source boundary without treating MG as a crate."""
    import tomllib

    for ancestor in (directory,) + tuple(directory.parents):
        if (ancestor / '.cargo-checksum.json').is_file():
            return ancestor
        manifest = ancestor / 'Cargo.toml'
        if manifest.is_file():
            if (ancestor / '.git').exists():
                return ancestor
            try:
                if 'workspace' in tomllib.loads(manifest.read_text(encoding='utf-8')):
                    return ancestor
            except (OSError, UnicodeError, tomllib.TOMLDecodeError) as error:
                raise MadGraph5Error('Cannot inspect Cargo source boundary %s: %s'
                                     % (manifest, error))
    return directory


def _within(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _read_original(path, root):
    """Dereference only in-source regular files; never archive symlink escapes."""
    resolved = path.resolve(strict=True)
    if not _within(resolved, root) or not resolved.is_file():
        raise MadGraph5Error('Dependency notice is not a regular file inside its source: %s'
                             % path)
    return resolved.read_bytes()


def _documents(directory, root, recursive):
    """Find original notice documents, including dedicated LICENSES directories."""
    result = set()
    for current, directories, files in os.walk(directory, followlinks=False):
        current = Path(current)
        in_notice_dir = any(_DOCUMENT.match(part) for part in current.relative_to(directory).parts)
        for name in sorted(files):
            if in_notice_dir or _DOCUMENT.match(name):
                result.add(current / name)
        kept = []
        for name in sorted(directories):
            path = current / name
            if name in _SKIP_DIRS:
                continue
            if path.is_symlink():
                if _DOCUMENT.match(name):
                    raise MadGraph5Error('Dependency notice directory is a symlink: %s' % path)
                continue
            if recursive or in_notice_dir or _DOCUMENT.match(name):
                kept.append(name)
        directories[:] = kept
    return result


def _metadata(bridge, environment):
    command = ['cargo', 'metadata', '--format-version=1', '--locked']
    try:
        result = subprocess.run(command, cwd=str(bridge), env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                check=True, text=True)
        metadata = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        detail = getattr(error, 'stderr', '') or str(error)
        raise MadGraph5Error('Cannot resolve Rust dependency notices with Cargo: %s'
                             % detail[-4000:]) from error
    if not isinstance(metadata, dict) or not isinstance(metadata.get('packages'), list) \
            or not metadata['packages']:
        raise MadGraph5Error('Cargo returned no resolved packages for dependency notices.')
    return metadata


def package(bridge, libdir, backend, environment=None):
    """Package exact upstream notice bytes for the actual locked Cargo graph.

    Return ``files`` (three basenames), ``sha256`` (their SHA256 hashes), and
    ``package_count``.  ``environment`` is passed directly to Cargo, with None
    inheriting the current process environment.  Sources are never modified.
    """
    names = _filenames(backend)
    bridge, libdir = Path(bridge), Path(libdir)
    metadata = _metadata(bridge, environment)
    members, packages = {}, []
    try:
        resolved = sorted(metadata['packages'], key=lambda item: item['id'])
        seen_ids = set()
        for item in resolved:
            identifier = item['id']
            if identifier in seen_ids:
                raise MadGraph5Error('Cargo returned duplicate package identity: %s' % identifier)
            seen_ids.add(identifier)
            manifest = Path(item['manifest_path']).resolve(strict=True)
            directory = manifest.parent
            root = _source_root(directory)
            key = '%s-%s-%s' % (re.sub(r'[^a-zA-Z0-9_.-]', '_', item['name']),
                                re.sub(r'[^a-zA-Z0-9_.-]', '_', item['version']),
                                _sha256(identifier.encode('utf-8')))
            documents = _documents(directory, root, recursive=True)
            ancestor = directory
            while ancestor != root:
                ancestor = ancestor.parent
                documents.update(_documents(ancestor, root, recursive=False))
            documents.add(manifest)
            if (directory / 'Cargo.toml.orig').is_file():
                documents.add(directory / 'Cargo.toml.orig')
            license_file = item.get('license_file')
            if license_file:
                declared = Path(license_file)
                documents.add(Path(os.path.abspath(str(
                    declared if declared.is_absolute() else directory / declared))))
            files = []
            for path in sorted(documents):
                data = _read_original(path, root)
                # Use lexical source-relative names but archive regular bytes.
                relative = path.relative_to(root).as_posix()
                archive_path = key + '/source/' + relative
                if '..' in Path(archive_path).parts or archive_path in members:
                    raise MadGraph5Error('Ambiguous dependency notice archive path: %s' % archive_path)
                members[archive_path] = data
                files.append({'source_path': str(path), 'source_relative_path': relative,
                              'archive_path': archive_path, 'sha256': _sha256(data),
                              'size': len(data)})
            packages.append({'id': identifier, 'name': item['name'], 'version': item['version'],
                             'source': item.get('source'), 'license': item.get('license'),
                             'license_file': license_file, 'authors': item.get('authors', []),
                             'repository': item.get('repository'), 'homepage': item.get('homepage'),
                             'manifest_path': str(manifest), 'source_root': str(root),
                             'manifest_sha256': _sha256(manifest.read_bytes()), 'files': files})
        provenance = {'schema_version': 1, 'backend': backend,
                      'metadata_command': ['cargo', 'metadata', '--format-version=1', '--locked'],
                      'package_count': len(packages), 'packages': packages,
                      'cargo_lock_sha256': _sha256((bridge / 'Cargo.lock').read_bytes())}
        data = io.BytesIO()
        with gzip.GzipFile(fileobj=data, mode='wb', filename='', mtime=0) as zipped:
            with tarfile.open(fileobj=zipped, mode='w', format=tarfile.PAX_FORMAT) as archive:
                for name in sorted(members):
                    info = tarfile.TarInfo(name)
                    info.size, info.mode, info.mtime = len(members[name]), 0o644, 0
                    archive.addfile(info, io.BytesIO(members[name]))
        notice = (
            '%s Rust dependency notices\n\n'
            'Original upstream notice, license, copyright, author and README documents are\n'
            'preserved byte for byte in %s.\n'
            '%s records the actual locked Cargo package identities,\n'
            'source declarations, source document paths and SHA256 hashes.\n\n'
            'These records preserve upstream statements without interpreting their terms.\n'
            'Symbolica runtime rights are not included or transferred by this notice bundle absent\n'
            'a separate written agreement. Packaging these documents does not grant any\n'
            'additional license or change any upstream terms. Consult the complete original\n'
            'documents, including Symbolica License.md and the oneloop README where present.\n'
            % (backend, names[0], names[1])).encode('utf-8')
        payloads = {names[0]: data.getvalue(), names[1]:
                    (json.dumps(provenance, indent=2, sort_keys=True) + '\n').encode('utf-8'),
                    names[2]: notice}
        libdir.mkdir(parents=True, exist_ok=True)
        # Each output is a regular file, including when replacing legacy aliases.
        with tempfile.TemporaryDirectory(prefix='.%s-notices-' % backend, dir=str(libdir)) as staging:
            for name, payload in payloads.items():
                (Path(staging) / name).write_bytes(payload)
            for name in names:
                os.replace(str(Path(staging) / name), str(libdir / name))
        return {'files': names, 'sha256': {name: _sha256(payloads[name]) for name in names},
                'package_count': len(packages)}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise MadGraph5Error('Cannot package %s Rust dependency notices: %s' % (backend, error)) from error
