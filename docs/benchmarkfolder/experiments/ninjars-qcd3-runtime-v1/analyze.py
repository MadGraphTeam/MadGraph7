#!/usr/bin/env python3
"""Partition retained warmed-run samples and archive the original perf bytes."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0, str(ROOT/'tools'))
import ninjars_profile_v2 as profiler


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_json(path, value):
    if path.exists():
        assert json.loads(path.read_text()) == value, path
    else:
        path.write_text(json.dumps(value, indent=2)+'\n')


def partition(path):
    counts = collections.Counter()
    leaves = collections.Counter()
    by = collections.defaultdict(collections.Counter)
    inclusive = collections.Counter()
    examples = {}
    for block in path.read_text().split('\n\n'):
        lines = [line for line in block.splitlines()
                 if line.strip() and not line.startswith('#')]
        if not lines:
            continue
        category = profiler.category(lines)
        counts[category] += 1
        leaf = re.sub(r'^\s*[0-9a-f]+ ', '', lines[1]) if len(lines)>1 else '[nochain]'
        leaves[leaf] += 1
        by[category][leaf] += 1
        examples.setdefault(category, lines)
        stack = '\n'.join(lines)
        for name, pattern in [('FortranQP', 'mp_coef_construction_'),
                ('MPReduction', 'madloop_ninjars::MpState::evaluate'),
                ('Compensated', 'madloop_ninjars::guarded::'),
                ('DPFast', 'ninjars::fast::'), ('OLMNative', 'oneloop::'),
                ('MPFR', 'mpfr_')]:
            if pattern in stack:
                inclusive[name] += 1
    total = sum(counts.values())
    assert total > 0
    return dict(samples=total,
        categories={k:dict(samples=v, percent=100*v/total) for k,v in counts.most_common()},
        top_leaves=leaves.most_common(35),
        top_category_leaves={k:v.most_common(12) for k,v in by.items()},
        inclusive_diagnostics=dict(inclusive), examples=examples)


def main():
    report = HERE/'full-profile-v2'
    record = json.loads((report/'profile.json').read_text())
    assert record['complete'] and record['all_printed_outputs_exact']
    assert not record['acceptance_eligible']
    for name, digest in record['evidence_sha256'].items():
        assert sha(report/name) == digest, name
    actual = partition(report/'perf-script.txt')
    # JSON normalization turns most_common's tuple pairs into array pairs.
    stable_json(report/'partition.json', json.loads(json.dumps(actual)))
    raw = Path(record['perf_data_path'])
    assert sha(raw) == record['perf_data_sha256']
    compressed = report/'perf.data.gz'
    if not compressed.exists():
        with raw.open('rb') as source, compressed.open('wb') as output:
            with gzip.GzipFile(filename='', mode='wb', fileobj=output, mtime=0) as archive:
                shutil.copyfileobj(source, archive)
    with gzip.open(compressed, 'rb') as stream:
        recovered = hashlib.file_digest(stream, 'sha256').hexdigest()
    assert recovered == record['perf_data_sha256']
    stable_json(report/'perf-archive.json', dict(
        raw_sha256=recovered, raw_bytes=raw.stat().st_size,
        compressed_file=compressed.name, compressed_sha256=sha(compressed),
        compressed_bytes=compressed.stat().st_size,
        classification_source_sha256=sha(Path(profiler.__file__)),
        classification_source='tools/ninjars_profile_v2.py',
        scope='Lossless original completed perf.data stream; no repaired header or filtered samples.'))
    print(json.dumps(dict(samples=actual['samples'], categories=actual['categories']), indent=2))


if __name__ == '__main__':
    main()
