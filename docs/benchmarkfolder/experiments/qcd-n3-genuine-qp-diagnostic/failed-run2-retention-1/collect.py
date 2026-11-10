#!/usr/bin/env python3
"""One-shot read-only capture of closed run2; no target/process execution."""
from pathlib import Path
import hashlib
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BASE = HERE.parent
ATTEMPT = ROOT / '.tirust-build/qcd-n3-genuine-qp-diagnostic/run2'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(name, value):
    (HERE / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def main():
    if (HERE / 'evidence-index.json').exists():
        raise FileExistsError('Closed retention is immutable')
    actual = sorted(p.relative_to(ATTEMPT).as_posix() for p in ATTEMPT.rglob('*') if p.is_file())
    if actual != ['loader.json', 'loader.log', 'result.json']:
        raise ValueError('Unexpected closed run2 membership')
    source = BASE / 'source-proposal2'
    copies = {}
    inputs = {**{'actual-run2/' + n: ATTEMPT / n for n in actual},
              'inherited-source2-index.json': source / 'source-index.json',
              'inherited-source2-bindings.json': source / 'bindings.json',
              'inherited-build2.json': ROOT / '.tirust-build/qcd-n3-genuine-qp-diagnostic/build2/build.json',
              'inherited-build2-static-index.json': BASE / 'build2-static/evidence-index.json'}
    for member, original in inputs.items():
        raw = original.read_bytes()
        target = HERE / member
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as out:
            out.write(raw)
        copies[str(original.relative_to(ROOT))] = {'member': member, 'sha256': digest(raw), 'bytes': len(raw)}
    binding = json.loads(inputs['inherited-source2-bindings.json'].read_bytes())
    traced = set(re.findall(r'/[^\s()]+', inputs['actual-run2/loader.log'].read_text()))
    aliases = {p: str(Path(p).resolve(strict=True)) for p in traced | set(binding['runtime'])}
    expected = {aliases[p] for p in binding['runtime']}
    actual_set = {aliases[p] for p in traced}
    dump('retention-inventory.json', {'actual_attempt_files': actual, 'original_copies': copies,
                                   'resolved_aliases_at_retention': aliases,
                                   'original_paths_are_metadata_not_replay_inputs': True})
    dump('summary.json', {'passed': True, 'scope': 'retention verification, not numerical qualification',
                         'loader_trace_executed': True, 'loader_trace_returncode': 0,
                         'strict_loader_validation_passed': False, 'attempted_workers': 0,
                         'diagnostic_calls': 0, 'bootstrap_calls': 0, 'scientific_results': False,
                         'actual_runtime_count': 15, 'compiler_support_substitutions': 3,
                         'unexpected_actual_paths': sorted(actual_set - expected),
                         'missing_expected_paths': sorted(expected - actual_set)})
    files = {}
    for p in sorted(HERE.rglob('*')):
        if p.is_file():
            raw = p.read_bytes()
            files[p.relative_to(HERE).as_posix()] = {'bytes': len(raw), 'sha256': digest(raw)}
    dump('evidence-index.json', {'schema': 'qcd-n3-genuine-qp-failed-run2-retention-v1', 'files': files,
                                'scope': 'complete closed loader-only failure; zero amplitudes'})
    import verify
    print(json.dumps(verify.verify(), sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
