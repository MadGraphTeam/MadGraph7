#!/usr/bin/env python3
"""Bind existing MG7 regressions to one exact candidate commit; no tests run."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import posixpath
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT/'docs/benchmarkfolder/policy-v2/final-regressions-v1'
FINAL_REVISION = 'c2b48e988fe6e3c6ba9403b67ec4d26fa5fa9ccb'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def git_file(revision, relative, chain=()):
    if relative in chain or len(chain) > 16:
        raise RuntimeError('Invalid candidate symlink chain: '+relative)
    item = git('ls-tree', '-z', revision, '--', relative)
    entries = [entry for entry in item.split(b'\0') if entry]
    if len(entries) != 1:
        raise RuntimeError('Candidate does not contain exact path: '+relative)
    metadata, filename = entries[0].split(b'\t', 1)
    mode, kind, object_id = metadata.decode().split()
    if filename.decode() != relative or kind != 'blob':
        raise RuntimeError('Candidate path is not a file/blob: '+relative)
    raw = git('cat-file', 'blob', object_id)
    if mode == '120000':
        target = posixpath.normpath(posixpath.join(posixpath.dirname(relative), raw.decode()))
        if target.startswith('../') or target.startswith('/'):
            raise RuntimeError('Candidate symlink leaves repository: '+relative)
        resolved, target_record = git_file(revision, target, (*chain, relative))
        record = {'path': relative, 'mode': mode, 'git_blob': object_id,
                  'link_text': raw.decode(), 'resolved_path': target_record.get('resolved_path', target),
                  'content_sha256': sha(resolved)}
        return resolved, record
    if mode not in ('100644', '100755'):
        raise RuntimeError('Unsupported candidate file mode: '+relative)
    return raw, {'path': relative, 'mode': mode, 'git_blob': object_id,
                 'content_sha256': sha(raw)}


def load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def bind():
    if git('rev-parse', FINAL_REVISION).decode().strip() != FINAL_REVISION:
        raise RuntimeError('Final candidate revision is unavailable')
    validation_path = OLD/'validation.json'
    validation = json.loads(validation_path.read_text())
    seal = load('ninjars_old_regression_seal', OLD/'verify_seal.py').verify()
    old = load('ninjars_old_regression_validation', OLD/'verify_validation.py')
    records, files = [], {}
    for relative, expected in sorted(validation['production_source_sha256'].items()):
        raw, record = git_file(FINAL_REVISION, relative)
        if sha(raw) != expected:
            raise RuntimeError('Final candidate differs from tested production source: '+relative)
        record['tested_sha256'] = expected
        records.append(record)
        files[relative] = sha(raw)
    if len(records) != 102:
        raise RuntimeError('Expected exactly 102 reused production source paths')
    evidence = {}
    report = old.report_snapshot(validation_path, files, evidence)
    # Match all verified source/test/fixture/log bytes to the immutable final Git
    # objects as well; the old report adapter independently checks their live bytes.
    evidence_records = []
    for relative, expected in sorted(evidence.items()):
        raw, record = git_file(FINAL_REVISION, relative)
        if sha(raw) != expected:
            raise RuntimeError('Final commit differs from retained tested/evidence file: '+relative)
        evidence_records.append(record)
    return {'schema': 'final-mg7-regression-binding-v1', 'status': 'passed',
            'candidate_revision': FINAL_REVISION,
            'candidate_tree': git('rev-parse', FINAL_REVISION+'^{tree}').decode().strip(),
            'reused_proof': str(validation_path.relative_to(ROOT)),
            'reused_proof_sha256': sha(validation_path.read_bytes()),
            'original_seal': seal, 'production_source_records': records,
            'production_paths_checked': len(records),
            'final_commit_evidence_files_checked': len(evidence_records),
            'final_commit_evidence_records': evidence_records,
            'reused_report_summary': report,
            'tests_rerun': 0, 'compiler_invocations': 0, 'numerical_engine_calls': 0,
            'scope': 'Reuses 90 source-bound MG7 cases (89 passed and one existing GoSam Python3 skip); generator, compiled Fortran, HELAS and exporter coverage. Rust reduction/scalar-integral engine coverage is explicitly excluded.',
            'historical_source_revision': validation['source_revision'],
            'method': 'Exact final-commit Git blobs with symlink resolution plus original sealed read-only fixture/test/log validators; old evidence is unchanged'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = bind()
    serialized = json.dumps(result, indent=2, sort_keys=True)+'\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(serialized)
    print(json.dumps({key: result[key] for key in ('status', 'candidate_revision', 'production_paths_checked',
          'final_commit_evidence_files_checked', 'tests_rerun', 'numerical_engine_calls')}, indent=2))
