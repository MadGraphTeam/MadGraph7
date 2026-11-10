#!/usr/bin/env python3
"""Finalize retention-only checks after closed scientific records were captured."""
import argparse
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
REPORT = HERE.with_name('subprocess-cache-lifecycle-current-qualification-2')
PROVISIONAL = 'b3eea1677e6b6c988ce4fbb0daf448e2176cccc88bff7fe15794e29bdbce74c6'

def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(value): return (json.dumps(value, indent=2, sort_keys=True) + '\n').encode()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'native_execution': False, 'report': str(REPORT), 'requires_provisional_index': PROVISIONAL}))
        return
    index_raw = (REPORT / 'evidence-index.json').read_bytes()
    assert sha(index_raw) == PROVISIONAL
    index = json.loads(index_raw)
    tests = (HERE / 'tests.log').read_text()
    assert 'Ran 14 tests' in tests and tests.rstrip().endswith('OK')
    review = json.loads((HERE / 'verifier-review.json').read_bytes())
    assert review['status'] == 'PASS' and review['verifier_sha256'] == sha((HERE / 'verify_evidence.py').read_bytes())
    # Existing scientific inputs and results cannot change during packaging.
    original = json.loads((REPORT / 'retention-binding.json').read_bytes())['closed_original_files']
    for name, checksum in original.items():
        assert sha((REPORT / name).read_bytes()) == checksum
    def add(name, raw):
        path = REPORT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(raw)
        index['files'][name] = {'kind': 'file', 'size': len(raw), 'sha256': sha(raw)}
    for name in ('verify_evidence.py', 'evidence-index.json', 'qualification-summary.json'):
        before = (HERE / 'provisional-first-capture' / name).read_bytes()
        assert before == (REPORT / name).read_bytes()
        add('retention-history/provisional-first-capture/' + name, before)
    before = (REPORT / 'verify_evidence.py').read_text()
    after = (HERE / 'verify_evidence.py').read_text()
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                        'provisional/verify_evidence.py', 'final/verify_evidence.py')).encode()
    add('retention-history/verifier-hardening.patch', patch)
    for name in ('test_verify.py', 'test_verify.attempt1.py', 'tests.log', 'tests-attempt1.log', 'verifier-review.json'):
        add('retention-validation/' + name, (HERE / name).read_bytes())
    add('retention-finalize.py', Path(__file__).read_bytes())
    add('retention-finalization.json', encoded({'schema_version': 1,
        'provisional_index_sha256': PROVISIONAL, 'scope': 'Retention-only checks; no change to original scientific records.',
        'provisional_verifier_sha256': sha(before.encode()), 'final_verifier_sha256': sha(after.encode()),
        'patch_sha256': sha(patch), 'tests_sha256': sha((HERE / 'tests.log').read_bytes()),
        'independent_review_sha256': sha((HERE / 'verifier-review.json').read_bytes()),
        'original_closed_files_rechecked': len(original), 'native_execution': False,
        'first_test_failure': 'One test attempted to delete compressed results.json as a direct index key; parser test fixed to remove its actual gzip index entry. Original failed test/log retained.'}))
    # Replace only provisional packaging outputs, preserving their exact originals above.
    (REPORT / 'verify_evidence.py').write_text(after)
    index['files']['verify_evidence.py'] = {'kind': 'file', 'size': len(after.encode()), 'sha256': sha(after.encode())}
    spec = importlib.util.spec_from_file_location('final_retained_lifecycle', REPORT / 'verify_evidence.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def reader(name):
        return encoded(index) if name == 'evidence-index.json' else (REPORT / name).read_bytes()
    proof = module.verify(read_bytes=reader, derive_summary=True)
    raw = encoded(proof['summary']); (REPORT / 'qualification-summary.json').write_bytes(raw)
    index['files']['qualification-summary.json'] = {'kind': 'file', 'size': len(raw), 'sha256': sha(raw)}
    module.verify(read_bytes=reader)
    for name, checksum in original.items(): assert sha((REPORT / name).read_bytes()) == checksum
    (REPORT / 'evidence-index.json').write_bytes(encoded(index))
    result = module.verify(REPORT)
    print(json.dumps({name: value for name, value in result.items() if name != 'summary'}, indent=2, sort_keys=True))

if __name__ == '__main__': main()
