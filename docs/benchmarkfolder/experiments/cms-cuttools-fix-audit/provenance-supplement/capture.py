#!/usr/bin/env python3
"""Recover an audited dirty source from its retained Git base and exact patch."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
AUDIT = HERE.parent / 'audit.json'
REL = 'madgraph/loop/loop_exporters.py'
IDENTITY = ROOT / 'docs/benchmarkfolder/experiments/dp-terminal-production/initial-proof-hardening/inputs/source-identities/c1d6ea3d354aab9e63056a5700724d32f25a02423af10049d83cac956838bdc7.json'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def strict_apply(base, patch):
    original = base.decode().splitlines(True)
    lines = patch.decode().splitlines(True)
    result, cursor, i = [], 0, 0
    while i < len(lines):
        match = re.match(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@', lines[i])
        if not match:
            assert not lines[i].startswith('@@'), lines[i]
            i += 1
            continue
        start, old_count, new_start, new_count = map(int, (match[1], match[2] or '1', match[3], match[4] or '1'))
        assert start - 1 >= cursor
        result.extend(original[cursor:start - 1])
        cursor = start - 1
        assert len(result) == new_start - 1
        old_seen = new_seen = 0
        i += 1
        while i < len(lines) and not lines[i].startswith('@@'):
            line = lines[i]
            assert line[0] in ' +-', repr(line)
            if line[0] in ' -':
                assert original[cursor] == line[1:], (cursor, original[cursor], line)
                cursor += 1
                old_seen += 1
            if line[0] in ' +':
                result.append(line[1:])
                new_seen += 1
            i += 1
        assert (old_seen, new_seen) == (old_count, new_count)
    result.extend(original[cursor:])
    return ''.join(result).encode()


def main():
    assert Path.cwd().resolve() == ROOT
    original_names = ('README.md', 'audit.json', 'audit_sources.py', 'namespace-evidence.txt')
    before = {name: digest((HERE.parent / name).read_bytes()) for name in original_names}
    audit = json.loads(AUDIT.read_bytes())
    identity = json.loads(IDENTITY.read_bytes())
    whole_patch = IDENTITY.with_suffix('.patch').read_bytes()
    assert digest(whole_patch) == identity['dirty_patch_sha256']
    assert identity['dirty'] and identity['revision'] == audit['git_head']
    target = audit['source_sha256'][REL]
    assert identity['files'][REL] == target
    sections = whole_patch.split(b'diff --git ')
    matches = [b'diff --git ' + p for p in sections if p.startswith(('a/' + REL + ' b/' + REL + '\n').encode())]
    assert len(matches) == 1
    patch = matches[0]
    base = subprocess.check_output(['git', 'show', audit['git_head'] + ':' + REL], cwd=ROOT)
    recovered = strict_apply(base, patch)
    assert digest(recovered) == target
    blob = subprocess.check_output(['git', 'rev-parse', audit['git_head'] + ':' + REL], cwd=ROOT, text=True).strip()
    outputs = {'audited-loop_exporters.py': recovered, 'recorded-head-loop_exporters.py': base,
        'audited-working-tree.patch': patch}
    for filename, data in outputs.items():
        path = HERE / filename
        if path.exists():
            assert path.read_bytes() == data, filename
        else:
            path.write_bytes(data)
    assert before == {name: digest((HERE.parent / name).read_bytes()) for name in original_names}
    manifest = {
        'schema': 1,
        'scope': 'Recovery of the exact dirty working-tree source already bound by the original read-only CMS audit; not a current-source re-audit or numerical qualification.',
        'original_audit_sha256': digest(AUDIT.read_bytes()),
        'original_audit_files_unchanged': before,
        'recorded_repository_HEAD': audit['git_head'],
        'recorded_HEAD_blob_oid': blob,
        'recorded_HEAD_file_sha256': digest(base),
        'audited_working_tree_file_sha256': target,
        'audited_file_differs_from_recorded_HEAD': digest(base) != target,
        'recovery_source_identity': str(IDENTITY.relative_to(ROOT)),
        'recovery_source_identity_sha256': digest(IDENTITY.read_bytes()),
        'recovery_whole_patch_path': str(IDENTITY.with_suffix('.patch').relative_to(ROOT)),
        'recovery_whole_patch_sha256': digest(whole_patch),
        'recovery_method': 'Strict in-memory unified-diff application to git show recorded_HEAD:path; every context/deletion and hunk count checked; exact original audited SHA-256 required.',
        'snapshot_sha256': {name: digest(data) for name, data in outputs.items()},
        'capture_script_sha256': digest(Path(__file__).read_bytes()),
        'current_source_observation': {
            'repository_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'working_tree_file_sha256': digest((ROOT / REL).read_bytes()),
        },
        'interpretation': 'The original git_head field names the repository pointer. Its source_sha256 field independently identifies audited working-tree bytes, which included an uncommitted DP-terminal exporter change. The original audit did not identify those bytes with the Git commit.',
        'production_changes_or_compiler_or_reducer_jobs': False,
    }
    path = HERE / 'manifest.json'
    if path.exists():
        old = json.loads(path.read_text())
        # A later repository pointer is not a change to the recovered snapshot.
        manifest['current_source_observation'] = old['current_source_observation']
        assert old == manifest
    else:
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    print('PASS: exact audited source recovered; original audit unchanged; SHA256 ' + target)


if __name__ == '__main__':
    main()
