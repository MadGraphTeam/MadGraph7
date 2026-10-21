"""Verify retained watch evidence; optionally compare existing Git objects."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], cwd=Path.cwd())


def entries(repo, revision):
    result = []
    for record in git(repo, 'ls-tree', '-r', '-z', '--full-tree', revision).split(b'\0'):
        if not record:
            continue
        metadata, name = record.split(b'\t', 1)
        path = name.decode()
        if path == 'TODO_NEXT_PLAN.md' or path.startswith(('docs/', 'validation/')):
            continue
        mode, kind, object_id = metadata.decode().split()
        result.append(dict(path=path, mode=mode, kind=kind, git_object=object_id))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, help='Existing NinjaRS clone; no fetch is attempted')
    options = parser.parse_args()
    manifest = json.loads((HERE/'manifest.json').read_text())
    for item in manifest['files']:
        path = Path(item['path'])
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid manifest path')
        data = (HERE/path).read_bytes()
        assert len(data) == item['bytes'] and sha(data) == item['sha256'], str(path)
    evidence = json.loads((HERE/'evidence.json').read_text())
    assert sha((HERE.parent/'manifest.json').read_bytes()) == evidence['prior_dependency_capsule_manifest_sha256']
    assert evidence['all_582_non_report_tree_entries_equal'] and evidence['changed_paths'] == 65
    assert evidence['cargo_files']['old'] == evidence['cargo_files']['new']
    for observation in evidence['official_ref_observations'].values():
        assert observation['exit_code'] == 0 and not observation['stderr']
        assert observation['symbolic_head'] == 'refs/heads/main'
        assert observation['refs']['HEAD'] == observation['refs']['refs/heads/main']
        for stream in ('stdout', 'stderr'):
            assert sha(observation[stream].encode()) == observation[stream+'_sha256']
    for flag in ('fresh_numerical_validation_claimed', 'new_binary_identity_claimed', 'new_performance_claimed'):
        assert evidence[flag] is False
    if options.repo:
        changed = git(options.repo, 'diff', '--name-only', evidence['from_revision'], evidence['to_revision']).decode().splitlines()
        assert len(changed) == 65 and all(path == 'TODO_NEXT_PLAN.md' or path.startswith(('docs/', 'validation/')) for path in changed)
        before = entries(options.repo, evidence['from_revision'])
        after = entries(options.repo, evidence['to_revision'])
        assert before == after and len(before) == evidence['non_report_source_entries'] == 582
        assert sha(json.dumps(before, sort_keys=True, separators=(',', ':')).encode()) == evidence['canonical_non_report_inventory_sha256']
        for path, pair in evidence['equal_git_objects'].items():
            assert pair['before'] == pair['after']
            for revision in (evidence['from_revision'], evidence['to_revision']):
                assert git(options.repo, 'rev-parse', revision+':'+path).decode().strip() == pair['before']
        for path, item in evidence['key_source_sha256'].items():
            for revision in (evidence['from_revision'], evidence['to_revision']):
                data = git(options.repo, 'show', revision+':'+path)
                assert len(data) == item['bytes'] and sha(data) == item['sha256']
    print(json.dumps(dict(passed=True, existing_git_objects_checked=bool(options.repo),
                         identical_non_report_entries=582, changed_report_paths=65,
                         compiler_calls=0, engine_calls=0, pin_changes=0), sort_keys=True))


if __name__ == '__main__':
    main()
