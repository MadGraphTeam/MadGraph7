#!/usr/bin/env python3
"""Create or verify the portable inventory of this saved-data supplement."""
import argparse
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
INDEX = HERE / 'evidence-index.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    assert Path.cwd().resolve() == ROOT
    if not args.check:
        summary = json.loads((HERE / 'summary.json').read_text())
        pinch = json.loads((HERE / 'pinch-summary.json').read_text())
        for records in (summary['sources'], pinch['sources_sha256']):
            for name, expected in records.items():
                assert sha(ROOT / name) == expected, name
        source_ranges = {
            '.tirust-build/pole-development/crates/iregi-rust/src/direct/exec.rs':
                [(1534, 1547), (1623, 1632), (1696, 1705), (3338, 3358), (3619, 3632)],
            '.tirust-build/pole-development/crates/iregi-rust/src/direct/plan.rs': [(1116, 1153)],
            'docs/benchmarkfolder/experiments/ew-threshold-adjudication/tensor-replay/replay.f90': [(97, 103)],
        }
        text = ['Numbered source excerpts; full source-file SHA-256 precedes each selection.\n']
        for name, ranges in source_ranges.items():
            source = ROOT / name
            lines = source.read_text().splitlines()
            text.append(name + '\nSHA256 ' + sha(source) + '\n')
            for start, end in ranges:
                text.extend(f'{i:5d}: {lines[i-1]}' for i in range(start, end + 1))
                text.append('')
        content = '\n'.join(text) + '\n'
        excerpts = HERE / 'source-excerpts.txt'
        if excerpts.exists():
            assert excerpts.read_text() == content
        else:
            excerpts.write_text(content)
    files = {str(path.relative_to(HERE)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
        for path in sorted(HERE.rglob('*')) if path.is_file() and path != INDEX}
    expected = {'schema': 1, 'scope': 'Saved-data rank-one geometry supplement; no native execution or production change.',
        'file_count': len(files), 'files': files}
    if INDEX.exists():
        assert json.loads(INDEX.read_text()) == expected
    else:
        assert not args.check
        INDEX.write_text(json.dumps(expected, indent=2, sort_keys=True) + '\n')
    print(f'PASS: {len(files)} supplement files verified; index SHA256 {sha(INDEX)}')


if __name__ == '__main__':
    main()
