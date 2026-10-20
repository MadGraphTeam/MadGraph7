#!/usr/bin/env python3
"""Read-only numerical, source, archive and sample verification."""
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0, str(HERE))
import analyze


def load(path):
    return json.loads(path.read_text())


def main():
    seal = load(HERE/'SHA256SUMS.json')
    for name, expected in seal['files'].items():
        path = HERE/name
        assert path.stat().st_size == expected['bytes'], name
        assert analyze.sha(path) == expected['sha256'], name
    report = HERE/'full-profile-v2'
    record = load(report/'profile.json')
    assert record['complete'] and not record['acceptance_eligible']
    assert record['all_printed_outputs_exact'] and record['loops'] == 3
    for name, checksum in record['evidence_sha256'].items():
        assert analyze.sha(report/name) == checksum, name
    archive = load(report/'perf-archive.json')
    compressed = report/archive['compressed_file']
    assert analyze.sha(compressed) == archive['compressed_sha256']
    with gzip.open(compressed, 'rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == archive['raw_sha256']
    assert archive['raw_sha256'] == record['perf_data_sha256']
    partition = json.loads(json.dumps(analyze.partition(report/'perf-script.txt')))
    assert partition == load(report/'partition.json')
    assert partition['samples'] == 7363
    original = ROOT/'docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-qcd3-v1'
    assert (report/'values.dat').read_bytes() == (original/'last-backend8-raw/values.dat').read_bytes()
    counts = dict(map(int, line.split()) for line in (report/'timed_status.dat').read_text().splitlines())
    assert counts == {218: 294, 322: 6}
    before = load(original/'inputs/candidate/qcd-3/compiled-sources.json')
    after = load(report/'generated-sources.json')
    assert before['files'].keys() == after['files'].keys()
    changes = sorted(k for k in before['files'] if before['files'][k] != after['files'][k])
    assert changes == record['changed_sources'] == ['SubProcesses/P0_uux_zgg/check_sa.f']
    print(json.dumps(dict(verified_files=len(seal['files']), samples=7363,
        original_stream_lossless=True, unchanged_printed_rows=True,
        timed_status_counts=counts, acceptance_eligible=False), indent=2))


if __name__ == '__main__':
    main()
