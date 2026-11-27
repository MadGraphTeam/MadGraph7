#!/usr/bin/env python3
"""Read-only upstream checkpoint. Exit 0=current, 1=update, 2=invalid/unavailable.

Requires Python 3.11+. This never fetches, changes a pin, modifies a reference
clone, or creates a PR. An explicit --output creates a new JSON snapshot only.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
REPOSITORIES = {
    'ninjars': 'https://github.com/alphal00p/ninjars',
    'oneloopmaster': 'https://github.com/alphal00p/oneloopmaster',
}
SHA = re.compile(r'[0-9a-f]{40}\Z')


def canonical_url(url):
    if not isinstance(url, str):
        raise ValueError('Git URL must be a string')
    return url.removesuffix('.git')


def revision(value, label):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError('%s must pin a full 40-character commit' % label)
    return value


def lock_source(package):
    source = package.get('source', '')
    if not isinstance(source, str) or not source.startswith('git+'):
        raise ValueError('%s has no pinned Git lock source' % package.get('name'))
    parsed = urlsplit(source[4:])
    query = parse_qs(parsed.query, strict_parsing=True)
    if set(query) != {'rev'} or len(query['rev']) != 1:
        raise ValueError('Lock Git source must use one explicit rev')
    pin = revision(query['rev'][0], 'lock rev')
    if revision(parsed.fragment, 'resolved lock revision') != pin:
        raise ValueError('Lock rev differs from resolved commit')
    return canonical_url(parsed._replace(query='', fragment='').geturl()), pin


def git_dependencies(packages):
    """Inventory and validate every transitive Git source, including overrides."""
    sources = {}
    for package in packages:
        source = package.get('source', '') if isinstance(package, dict) else ''
        if isinstance(source, str) and source.startswith('git+'):
            url, pin = lock_source(package)
            sources.setdefault((url, pin), []).append(package['name'])
    return [dict(repository=url, revision=pin, packages=sorted(names))
            for (url, pin), names in sorted(sources.items())]


def read_pins(manifest, lockfile):
    import tomllib

    manifest = Path(manifest)
    lockfile = Path(lockfile)
    cargo = tomllib.loads(manifest.read_text())
    lock = tomllib.loads(lockfile.read_text())
    dependencies = cargo.get('dependencies', {})
    explicit = []
    for name in ('ninjars', 'ninjars-symbolica'):
        dependency = dependencies.get(name)
        if not isinstance(dependency, dict):
            raise ValueError('Missing explicit dependency %s' % name)
        if canonical_url(dependency.get('git')) != REPOSITORIES['ninjars']:
            raise ValueError('Unexpected %s repository' % name)
        explicit.append(revision(dependency.get('rev'), name))
    if explicit[0] != explicit[1]:
        raise ValueError('ninjars and ninjars-symbolica pins disagree')
    packages = lock.get('package')
    if not isinstance(packages, list):
        raise ValueError('Cargo.lock must contain a package array')
    git_dependencies(packages)  # No unnoticed floating transitive Git source.
    found = {}
    for name in ('ninjars', 'ninjars-symbolica', 'oneloop'):
        matches = [p for p in packages if isinstance(p, dict) and p.get('name') == name]
        if len(matches) != 1:
            raise ValueError('Cargo.lock must contain exactly one %s package' % name)
        found[name] = lock_source(matches[0])
    for name in ('ninjars', 'ninjars-symbolica'):
        if found[name] != (REPOSITORIES['ninjars'], explicit[0]):
            raise ValueError('Cargo.toml and Cargo.lock disagree for %s' % name)
    # Other crates from the same NinjaRS repository must not silently diverge.
    for package in packages:
        source = package.get('source', '') if isinstance(package, dict) else ''
        if isinstance(source, str) and source.startswith('git+'):
            url = canonical_url(urlsplit(source[4:])._replace(query='', fragment='').geturl())
            if url == REPOSITORIES['ninjars'] and lock_source(package)[1] != explicit[0]:
                raise ValueError('A locked NinjaRS workspace package has a different pin')
    if found['oneloop'][0] != REPOSITORIES['oneloopmaster']:
        raise ValueError('Unexpected oneloopmaster repository')
    return {'ninjars': explicit[0], 'oneloopmaster': found['oneloop'][1]}


def parse_remote(text):
    refs, default = {}, None
    for line in text.splitlines():
        fields = line.split()
        if fields[:1] == ['ref:']:
            if len(fields) != 3 or fields[2] != 'HEAD' or not fields[1].startswith('refs/heads/') or default is not None:
                raise ValueError('Malformed or duplicate symbolic HEAD')
            default = fields[1]
            continue
        if len(fields) != 2:
            raise ValueError('Malformed ls-remote output')
        commit, name = fields
        revision(commit, 'remote ref')
        if name not in ('HEAD', 'refs/heads/main') and not name.startswith('refs/tags/'):
            raise ValueError('Unexpected remote ref %s' % name)
        if name in refs:
            raise ValueError('Duplicate remote ref %s' % name)
        refs[name] = commit
    if default is None or 'HEAD' not in refs or 'refs/heads/main' not in refs:
        raise ValueError('Upstream must advertise symbolic HEAD and main')
    if default == 'refs/heads/main' and refs['HEAD'] != refs['refs/heads/main']:
        raise ValueError('Inconsistent main/HEAD advertisement')
    tags = {}
    for name, object_id in refs.items():
        if name.startswith('refs/tags/') and not name.endswith('^{}'):
            tags[name[10:]] = {'object': object_id, 'commit': refs.get(name+'^{}', object_id)}
    if any(name.endswith('^{}') and name[:-3] not in refs for name in refs):
        raise ValueError('Annotated tag peeled ref has no tag object')
    return {'default_branch': default, 'default_head': refs['HEAD'],
            'main': refs['refs/heads/main'], 'tags': dict(sorted(tags.items()))}


def query(url, timeout=30):
    command = ['git', 'ls-remote', '--symref', url, 'HEAD', 'refs/heads/main', 'refs/tags/*']
    process = subprocess.run(command, check=True, text=True, capture_output=True, timeout=timeout)
    return parse_remote(process.stdout)


def check(manifest, lockfile, query_remote=query):
    import tomllib

    pins = read_pins(manifest, lockfile)  # Reject local inconsistency before network.
    results = {}
    for name, url in REPOSITORIES.items():
        remote = query_remote(url)
        results[name] = dict(repository=url, pinned=pins[name], **remote,
            update_pending=remote['main'] != pins[name],
            default_head_differs_from_pin=remote['default_head'] != pins[name])
    return {'schema':'ninjars-upstream-check-v1', 'checked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'complete':True, 'update_pending':any(r['update_pending'] for r in results.values()),
        'inputs':{str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (manifest,lockfile)},
        'git_dependencies':git_dependencies(tomllib.loads(Path(lockfile).read_text())['package']),
        'repositories':results,
        'policy':'main commits determine pending updates; tags are a complete observed inventory, not ordered releases. No pin or reference clone is changed.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'vendor/ninjars/Cargo.toml')
    parser.add_argument('--lockfile',type=Path,default=ROOT/'vendor/ninjars/Cargo.lock')
    parser.add_argument('--output',type=Path,help='Create a new snapshot; existing files are never overwritten.')
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise ValueError('Snapshot already exists: %s' % args.output)
        result = check(args.manifest,args.lockfile)
        code = int(result['update_pending'])
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as error:
        result = {'schema':'ninjars-upstream-check-v1','complete':False,'error':str(error)}
        code = 2
    serialized = json.dumps(result,indent=2,sort_keys=True)+'\n'
    if args.output is not None and not args.output.exists():
        try:
            with args.output.open('x') as stream:
                stream.write(serialized)
        except OSError as error:
            result = {'schema':'ninjars-upstream-check-v1','complete':False,'error':str(error)}
            serialized = json.dumps(result,indent=2,sort_keys=True)+'\n'
            code = 2
    print(serialized,end='')
    return code


if __name__ == '__main__':
    raise SystemExit(main())
