#!/usr/bin/env python3
"""Read-only verification and report adapter for the bounded MG7 regressions."""
import argparse
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re

if not __debug__:
    raise RuntimeError('regression verification requires Python assertions enabled')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _artifact(relative):
    path = ROOT / relative
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f'Artifact leaves repository: {relative}')
    return path


def _read_bound(item, evidence):
    path = _artifact(item['path'])
    data = path.read_bytes()
    assert _sha(data) == item['sha256'], item['path']
    evidence[item['path']] = item['sha256']
    return data


def _footer(text):
    runs = re.findall(r'^Ran (\d+) tests? in ([0-9.]+)s$', text, re.M)
    assert len(runs) == 1, 'Exactly one completed unittest footer is required'
    ok = re.findall(r'^OK(?: \(skipped=(\d+)\))?$', text, re.M)
    assert len(ok) == 1 and not re.search(r'^FAILED ', text, re.M)
    return int(runs[0][0]), float(runs[0][1]), int(ok[0] or 0)


def verify(path=None):
    """Verify current tested files, raw logs, fixture scope and historical failures."""
    path = Path(path) if path is not None else HERE / 'validation.json'
    if not path.is_absolute():
        path = ROOT / path
    data = path.read_bytes()
    spec = json.loads(data)
    assert spec['schema'] == 1 and len(spec['suites']) == 3
    evidence = {str(path.relative_to(ROOT)): _sha(data)}
    summaries = []
    for suite in spec['suites']:
        assert suite['exit_code'] == suite['failures'] == suite['errors'] == 0
        text = _read_bound(suite['logbound'], evidence).decode()
        count, seconds, skipped = _footer(text)
        assert count == suite['tests_run'] and skipped == suite['skipped'], suite['name']
        assert abs(seconds-suite['seconds']) < 1e-6, suite['name']
        assert len(suite['skip_details']) == skipped
        for name, reason in suite['skip_details']:
            assert name in text and reason in text
        for relative, expected in suite['source_sha256'].items():
            assert _sha(_artifact(relative).read_bytes()) == expected, relative
            evidence[relative] = expected
        summaries.append({key: suite[key] for key in
                          ('name', 'tests_run', 'failures', 'errors', 'skipped',
                           'skip_details', 'scope', 'source_revision')})
    for relative, expected in spec['production_source_sha256'].items():
        assert _sha(_artifact(relative).read_bytes()) == expected, relative
        evidence[relative] = expected
    for relative, link in spec['symlink_targets'].items():
        item = ROOT / relative
        assert item.is_symlink() and item.readlink().as_posix() == link['link_text']
        assert item.resolve() == (ROOT / link['target']).resolve()
    for attempt in spec['historical_attempts']:
        raw = _read_bound(attempt['logbound'], evidence)
        if 'uncompressed_sha256' in attempt:
            raw = gzip.decompress(raw)
            assert len(raw) == attempt['uncompressed_bytes']
            assert _sha(raw) == attempt['uncompressed_sha256']
            assert not re.search(rb'^Ran \d+ tests? in ', raw, re.M)
        else:
            text = raw.decode()
            assert re.search(rf"^Ran {attempt['tests_run']} tests? in ", text, re.M)
            assert 'FAILED (' in text
            assert f"failures={attempt['failures']}" in text
            if attempt['errors']:
                assert f"errors={attempt['errors']}" in text
    fixture_path = _artifact(spec['fixture_review'])
    evidence[spec['fixture_review']] = _sha(fixture_path.read_bytes())
    module_path = HERE / 'verify_fixture_review.py'
    module_spec = importlib.util.spec_from_file_location('ninjars_fixture_review_v1', module_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    fixture_result = module.verify()
    for script in ('verify_fixture_review.py', 'verify_validation.py'):
        relative = str((HERE / script).relative_to(ROOT))
        evidence[relative] = _sha((HERE / script).read_bytes())
    total = sum(suite['tests_run'] for suite in summaries)
    skipped = sum(suite['skipped'] for suite in summaries)
    assert total == spec['tests_run'] == 90
    assert skipped == spec['total_skipped'] == 1
    assert total-skipped == spec['tests_passed'] == 89
    return {'status': 'passed-with-explicit-existing-python3-bypass',
            'suites': summaries, 'tests_run': total, 'tests_passed': total-skipped,
            'total_skipped': skipped, 'failures': 0, 'errors': 0,
            'production_files_bound': len(spec['production_source_sha256']),
            'fixture_review': fixture_result,
            'scope': spec['scope'],
            'source_scope_note': spec['source_scope_note'],
            'historical_attempts_retained': len(spec['historical_attempts']),
            'evidence_sha256': evidence}


def report_snapshot(path, candidate_generator_files, inputs=None):
    """Require exact tested-production hashes in the final timing candidate.

    ``candidate_generator_files`` is the timing target's
    ``generation_source_identity.tree_state['tracked']`` path-to-SHA256 map.
    When provided, ``inputs`` receives every checked local file and its digest.
    Test/fixture/log files are checked independently from the live repository.
    Rust reduction/scalar-integral engines are explicitly outside this campaign.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    result = verify(path)
    spec = json.loads(path.read_text())
    assert isinstance(candidate_generator_files, dict)
    for relative, digest in spec['production_source_sha256'].items():
        assert relative in candidate_generator_files, f'Candidate omits tested production source: {relative}'
        assert candidate_generator_files[relative] == digest, f'Tested/candidate source mismatch: {relative}'
    result['candidate_generation_match'] = 'all tested production source hashes match'
    result['candidate_generation_files_checked'] = len(spec['production_source_sha256'])
    result['qualification'] = result['status']
    result['status'] = 'passed'
    if inputs is not None:
        if not isinstance(inputs, dict):
            raise TypeError('inputs must be a dictionary or None')
        inputs.update(result['evidence_sha256'])
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', default=str(HERE / 'validation.json'))
    args = parser.parse_args()
    print(json.dumps(verify(args.spec), indent=2, sort_keys=True))
