"""Finite read-only link admission and private worker copies for CMS replay 2.

``tree_snapshot`` uses the sealed audit inventory schema and never descends
through a directory symlink. ``verify_external_graph`` accepts the sealed
cms-replay-link-closure-audit-v1 tables; the small ``external_roots`` plus
``readonly_files`` form is useful for pure temporary fixtures. Directory
permission comes from complete pinned inventories, never from an ancestor.

``prepare_worker(master, worker, expected_master, audit)`` checks those pins,
copies only the private master tree, rebases absolute internal links, and
returns a context. It must precede configuration. Every write target should
pass ``assert_private_path``. ``assert_worker_inputs`` checks the context
afterwards; permitted local file changes and additions are exact path lists.
It never accepts a new link, deleted original entry, or inherited write path.
All functions use filesystem data only and launch no executable.
"""
import copy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat


class LinkClosureError(ValueError):
    pass


def _require(value, message):
    if not value:
        raise LinkClosureError(message)


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _absolute(path):
    path = Path(path)
    return path if path.is_absolute() else Path.cwd() / path


def _inside(path, root):
    return Path(path).is_relative_to(Path(root))


def _kind(mode):
    if stat.S_ISREG(mode): return 'file'
    if stat.S_ISDIR(mode): return 'directory'
    if stat.S_ISLNK(mode): return 'symlink'
    raise LinkClosureError('Special filesystem node is not admitted')


def _resolve(path, *, missing=False):
    """Resolve components without losing an ancestor alias or lexical '..'."""
    path = _absolute(path)
    current = Path(path.anchor)
    todo = list(path.parts[1:])
    seen = set(); chain = []
    while todo:
        name = todo.pop(0)
        if name == '.': continue
        if name == '..':
            current = current.parent
            continue
        candidate = current / name
        try:
            info = candidate.lstat()
        except FileNotFoundError:
            _require(missing, 'Dangling/missing link target: ' + str(candidate))
            current = candidate
            continue
        except OSError as error:
            raise LinkClosureError('Cannot resolve target: ' + str(candidate)) from error
        kind = _kind(info.st_mode)
        if kind == 'symlink':
            spelling = str(candidate)
            _require(spelling not in seen, 'Symlink cycle: ' + spelling)
            seen.add(spelling)
            target = os.readlink(candidate)
            chain.append({'logical_path': spelling, 'readlink': target})
            target_path = Path(target)
            if target_path.is_absolute():
                current = Path(target_path.anchor)
                todo = list(target_path.parts[1:]) + todo
            else:
                todo = list(target_path.parts) + todo
        else:
            _require(kind == 'directory' or not todo, 'Non-directory ancestor: ' + str(candidate))
            current = candidate
    try:
        final_kind = _kind(current.lstat().st_mode)
    except FileNotFoundError:
        _require(missing, 'Dangling/missing final target: ' + str(current))
        final_kind = 'missing'
    for item in chain:
        node = Path(item['logical_path'])
        _require(node.is_symlink() and os.readlink(node) == item['readlink'],
                 'Alias changed during resolution: ' + str(node))
    return current, final_kind, chain


def _file_hash(path):
    before = path.stat()
    checksum = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    after = path.stat()
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    _require(all(getattr(before, key) == getattr(after, key) for key in fields),
             'File changed while hashing: ' + str(path))
    return before.st_size, checksum.hexdigest()


def _describe(path):
    path = _absolute(path)
    try:
        info = path.lstat()
    except OSError as error:
        raise LinkClosureError('Missing graph node: ' + str(path)) from error
    kind = _kind(info.st_mode)
    resolved, target_type, chain = _resolve(path)
    result = {'kind': 'regular_file' if kind == 'file' else kind,
              'mode': oct(stat.S_IMODE(info.st_mode)), 'resolved': str(resolved),
              'resolved_type': target_type}
    if kind == 'symlink':
        result.update(readlink=os.readlink(path), symlink_chain=chain)
    if target_type == 'file':
        result['size'], result['sha256'] = _file_hash(resolved)
    return result


def _directory_cycles(inventories):
    edges = {}; directories = set()
    for inventory in inventories:
        root = inventory['root_resolved']; directories.add(root)
        for name, item in inventory['entries'].items():
            if item['kind'] == 'directory': directories.add(item['resolved'])
    for inventory in inventories:
        root = Path(inventory['root']); canonical_root = inventory['root_resolved']
        directory_names = {'.': canonical_root}
        directory_names.update({name: item['resolved'] for name, item in inventory['entries'].items()
                                if item['kind'] == 'directory'})
        for name, item in inventory['entries'].items():
            if item['resolved_type'] != 'directory': continue
            parent = str(PurePosixPath(name).parent)
            _require(parent in directory_names, 'Missing directory in inventory: ' + str(root / name))
            edges.setdefault(directory_names[parent], set()).add(item['resolved'])
    visiting = set(); complete = set()

    def visit(node):
        _require(node not in visiting, 'Directory alias cycle: ' + node)
        if node in complete: return
        visiting.add(node)
        for child in edges.get(node, ()):
            if child in directories: visit(child)
        visiting.remove(node); complete.add(node)
    for directory in directories: visit(directory)


def tree_snapshot(root, follow_external=False):
    """Snapshot a real tree in the audit schema, preserving every link spelling.

    External file endpoints are hashed; external directory membership is only
    admitted through separately declared inventories, never by following links.
    ``follow_external=True`` is rejected to prevent accidental ambient admission.
    """
    _require(follow_external is False, 'External traversal requires explicit pinned envelopes')
    root = _absolute(root)
    _require(root.is_dir() and not root.is_symlink(), 'Snapshot root must be a real directory')
    resolved, kind, unused_chain = _resolve(root)
    _require(kind == 'directory', 'Snapshot root is not a directory')
    entries = {}

    def walk(directory):
        names = sorted(os.listdir(directory))
        for name in names:
            path = directory / name
            relative = path.relative_to(root).as_posix()
            description = _describe(path)
            entries[relative] = description
            if description['kind'] == 'directory': walk(path)
        _require(names == sorted(os.listdir(directory)), 'Directory membership changed during snapshot')
    walk(root)
    result = {'root': str(root), 'root_resolved': str(resolved), 'entry_count': len(entries),
              'entries': entries, 'canonical_entries_sha256': _canonical(entries)}
    _directory_cycles([result])
    return result


def _inventory(expected):
    _require(isinstance(expected, dict) and isinstance(expected.get('entries'), dict), 'Missing pinned inventory')
    _require(expected['entry_count'] == len(expected['entries'])
             and expected['canonical_entries_sha256'] == _canonical(expected['entries']),
             'Pinned inventory count/hash differs')
    return {key: expected[key] for key in ('root', 'root_resolved', 'entry_count', 'entries', 'canonical_entries_sha256')}


def verify_tree(root, expected):
    expected = _inventory(expected)
    observed = tree_snapshot(root)
    _require(observed == expected, 'Tree membership/hash/mode/link/target differs: ' + str(root))
    return observed


def verify_snapshot(snapshot):
    return verify_tree(snapshot['root'], snapshot)


def _external_parts(audit):
    if 'external_roots' in audit:
        roots = audit['external_roots']
        leaves = audit.get('readonly_files', {})
        observations = {}; transitive = {}
    else:
        _require(audit.get('schema') == 'cms-replay-link-closure-audit-v1', 'Unknown external audit schema')
        roots = dict(audit['external_directory_envelopes'])
        roots.update(audit['external_directory_alias_target_envelopes'])
        leaves = {}
        for path, record in audit['historical_leaf_bindings'].items():
            _require(record['all_recorded_bindings_match'] is True
                     and all(item['sha256'] == record['observed_sha256'] for item in record['references']),
                     'Historical leaf binding disagrees: ' + path)
            leaves[path] = record['observed_sha256']
        observations = audit['logical_bound_leaf_observations']
        transitive = audit['transitive_symlink_nodes']
    _require(all(name == item['root'] for name, item in roots.items()), 'Explicit envelope key/root differs')
    return [_inventory(item) for item in roots.values()], leaves, observations, transitive


def _allowed_graph(inventories, leaves, observations, transitive):
    files = set(leaves); directories = set(); nodes = set(observations) | set(transitive) | set(leaves)
    for inventory in inventories:
        directories.add(inventory['root_resolved']); nodes.add(inventory['root'])
        for name, item in inventory['entries'].items():
            nodes.add(str(Path(inventory['root']) / name))
            if item['kind'] == 'regular_file': files.add(item['resolved'])
            elif item['kind'] == 'directory': directories.add(item['resolved'])
    return files, directories, nodes


def _admit_links(inventories, files, directories, nodes):
    for inventory in inventories:
        for name, item in inventory['entries'].items():
            if item['kind'] != 'symlink': continue
            allowed = files if item['resolved_type'] == 'file' else directories
            _require(item['resolved'] in allowed, 'Unbound ambient link target: ' + str(Path(inventory['root']) / name))
            for hop in item['symlink_chain']:
                _require(hop['logical_path'] in nodes, 'Unbound ambient alias hop: ' + hop['logical_path'])
    _directory_cycles(inventories)


def verify_external_graph(audit):
    """Verify finite envelope membership, aliases, leaves and historical pins."""
    inventories, leaves, observations, transitive = _external_parts(audit)
    for inventory in inventories: verify_snapshot(inventory)
    for path, checksum in leaves.items():
        description = _describe(path)
        _require(description['kind'] == 'regular_file' and description['resolved_type'] == 'file'
                 and description['resolved'] == path and description['sha256'] == checksum,
                 'Read-only canonical leaf changed: ' + path)
    for path, expected in observations.items():
        _require(_describe(path) == expected, 'Read-only logical leaf/alias changed: ' + path)
    for path, expected in transitive.items():
        observed = _describe(path)
        fields = {key: observed[key] for key in expected if key != 'logical_path'}
        _require(expected.get('logical_path', path) == path
                 and fields == {key: value for key, value in expected.items() if key != 'logical_path'},
                 'Transitive alias changed: ' + path)
    files, directories, nodes = _allowed_graph(inventories, leaves, observations, transitive)
    _admit_links(inventories, files, directories, nodes)
    return {'inventories': len(inventories), 'readonly_leaves': len(leaves),
            'canonical_audit_sha256': _canonical(audit)}


def assert_private_path(worker, path):
    """Return a private write endpoint, including a not-yet-created FIFO/file."""
    worker = _absolute(worker)
    _require(worker.is_dir() and not worker.is_symlink(), 'Worker root must be a real directory')
    canonical_worker, kind, root_chain = _resolve(worker)
    path = Path(path); path = path if path.is_absolute() else worker / path
    _require(_inside(path, worker), 'Write path is lexically outside worker: ' + str(path))
    resolved, target_type, chain = _resolve(path, missing=True)
    _require(_inside(resolved, canonical_worker), 'Write path resolves outside worker: ' + str(path))
    _require(all(_inside(hop['logical_path'], worker) for hop in chain),
             'Write path traverses an inherited alias: ' + str(path))
    _require(not root_chain, 'Worker root traverses an ancestor alias')
    if target_type == 'file':
        _require(resolved.stat().st_nlink == 1, 'Write path has an unproven hard-link alias: ' + str(path))
    return resolved


def _sensitive(name):
    path = PurePosixPath(name)
    return (path.parts[0] == 'Cards' or 'MadLoop5_resources' in path.parts
            or path.suffix.lower() in ('.f', '.inc')
            or path.name in ('MadLoopParams.dat', 'param_card.dat', 'ident_card.dat', 'worker-protocol.fifo')
            or path.name.endswith(('HelFilter.dat', 'LoopFilter.dat')))


def _guard_writes(root, snapshot):
    for name in snapshot['entries']:
        if _sensitive(name): assert_private_path(root, Path(root) / name)


def _rebase(value, old, new):
    return new + value[len(old):] if value == old or value.startswith(old + '/') else value


def _worker_expected(expected, worker):
    old = expected['root']; new = str(_absolute(worker))
    result = copy.deepcopy(_inventory(expected))
    result.update(root=new, root_resolved=new)
    for item in result['entries'].values():
        item['resolved'] = _rebase(item['resolved'], old, new)
        if item['kind'] == 'symlink':
            item['readlink'] = _rebase(item['readlink'], old, new)
            for hop in item['symlink_chain']:
                hop['logical_path'] = _rebase(hop['logical_path'], old, new)
                hop['readlink'] = _rebase(hop['readlink'], old, new)
    result['canonical_entries_sha256'] = _canonical(result['entries'])
    return result


def _private_regular_files(master, worker, expected):
    for name, item in expected['entries'].items():
        if item['kind'] != 'regular_file': continue
        origin = (master / name).lstat(); copied = (worker / name).lstat()
        _require((origin.st_dev, origin.st_ino) != (copied.st_dev, copied.st_ino),
                 'Worker regular file shares inherited inode: ' + name)


def prepare_worker(master, worker, expected_master, audit):
    """Validate pinned inputs, copy privately, then rebase before any writes."""
    master = _absolute(master); worker = _absolute(worker)
    expected_master = copy.deepcopy(_inventory(expected_master))
    verify_tree(master, expected_master); verify_external_graph(audit)
    _guard_writes(master, expected_master)
    inventories, leaves, observations, transitive = _external_parts(audit)
    files, directories, nodes = _allowed_graph(inventories + [expected_master], leaves, observations, transitive)
    _admit_links(inventories + [expected_master], files, directories, nodes)
    _require(not worker.exists() and not worker.is_symlink(), 'Fresh worker destination required')
    destination, kind, chain = _resolve(worker, missing=True)
    _require(not chain and destination == worker, 'Worker destination traverses ancestor alias')
    _require(not _inside(worker, master) and not _inside(master, worker)
             and not any(_inside(worker, item['root_resolved']) for item in inventories),
             'Worker destination overlaps inherited directory')
    worker.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(master, worker, symlinks=True, copy_function=shutil.copy2)
    for name, item in expected_master['entries'].items():
        if item['kind'] != 'symlink': continue
        path = worker / name
        rebased = _rebase(item['readlink'], str(master), str(worker))
        if rebased != item['readlink']:
            path.unlink(); path.symlink_to(rebased)
    expected_worker = _worker_expected(expected_master, worker)
    verify_tree(worker, expected_worker)
    _guard_writes(worker, expected_worker)
    _private_regular_files(master, worker, expected_master)
    verify_tree(master, expected_master); verify_external_graph(audit)
    return {'master': str(master), 'worker': str(worker), 'expected_master': expected_master,
            'expected_worker': expected_worker, 'audit': audit, 'audit_sha256': _canonical(audit)}


def assert_worker_inputs(worker, context, *, allow_file_changes=(), allow_new_paths=()):
    """Recheck inherited inputs and the exact worker link set after local work."""
    worker = _absolute(worker); master = Path(context['master'])
    _require(str(worker) == context['worker'] and _canonical(context['audit']) == context['audit_sha256'],
             'Worker context/audit changed')
    verify_tree(master, context['expected_master']); verify_external_graph(context['audit'])
    observed = tree_snapshot(worker); expected = context['expected_worker']
    changes = {str(assert_private_path(worker, path)) for path in allow_file_changes}
    new_types = allow_new_paths if isinstance(allow_new_paths, dict) else {path: 'regular_file' for path in allow_new_paths}
    additions = {str(_absolute(path) if Path(path).is_absolute() else worker / path): kind
                 for path, kind in new_types.items()}
    for path in additions: assert_private_path(worker, path)
    _require(set(expected['entries']).issubset(observed['entries']), 'Original worker member deleted')
    extra = set(observed['entries']) - set(expected['entries'])
    _require({str(worker / name) for name in extra} == set(additions), 'Unbound worker addition')
    for name in extra:
        kind = additions[str(worker / name)]
        _require(kind in ('regular_file', 'directory') and observed['entries'][name]['kind'] == kind,
                 'New worker node type is not admitted')
    for name, original in expected['entries'].items():
        actual = observed['entries'][name]
        if original == actual: continue
        before = dict(original); after = dict(actual)
        if actual['resolved'] in changes:
            for key in ('sha256', 'size'):
                before.pop(key, None); after.pop(key, None)
        _require(before == after, 'Worker input/link changed outside explicit local file permission: ' + name)
    _guard_writes(worker, observed)
    _private_regular_files(master, worker, context['expected_master'])
    return observed


copy_worker = prepare_worker
verify_worker = assert_worker_inputs
