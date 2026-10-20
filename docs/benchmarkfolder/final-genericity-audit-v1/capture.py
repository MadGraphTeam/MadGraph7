#!/usr/bin/env python3
"""Bounded source census; never generate/build/evaluate a process or run tests."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / 'madgraph/__init__.py').is_file())
BASELINE = '27147b3300f556a75661a886e92bdbe00558ee25'
SCOPE = ['madgraph', 'aloha', 'Template', 'bin', 'input', 'models',
         'vendor/ninjars/src', 'vendor/ninjars/mninjars.f90', 'vendor/ninjars/ninjars.h',
         'vendor/ninjars/smoke.f90']
PATTERNS = {
    'benchmark_case_or_process': r'\b(?:qcd[-_][1-4]|ew[-_][1-4]|ggh|hgg|gg_h|ttx|uux_z|loop_sm|loop_qcd_qed_sm_Gmu)\b',
    'benchmark_paths': r'benchmarkfolder|\.benchmark-work|ninjars[-_]benchmark',
    'fixed_seed_or_point': r'\bseed\b|\bpoint[ _-]?(?:44|67|80)\b',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    names = subprocess.check_output(['git', 'diff', '--name-only', BASELINE, '--', *SCOPE],
                                    cwd=ROOT, text=True).splitlines()
    patch = subprocess.check_output(['git', 'diff', '--unified=0', BASELINE, '--', *SCOPE],
                                   cwd=ROOT, text=True)
    added, filename, line_number = {}, None, None
    for line in patch.splitlines():
        if line.startswith('+++ b/'):
            filename = line[6:]
            added.setdefault(filename, set())
        elif line.startswith('@@'):
            line_number = int(re.search(r'\+(\d+)', line)[1])
        elif line_number is not None and line.startswith('+') and not line.startswith('+++'):
            added[filename].add(line_number)
            line_number += 1
        elif line_number is not None and not line.startswith(('-', '\\', 'diff ', 'index ')):
            line_number += 1
    records, matches = {}, []
    for name in names:
        path = ROOT / name
        if not path.is_file():
            records[name] = {'deleted': True}
            continue
        raw = path.read_bytes()
        text = raw.decode()
        region = text
        kind = 'production'
        if name.startswith('vendor/ninjars/src/') and (name.endswith('_tests.rs')):
            kind, region = 'cfg(test)-only module', ''
        elif name == 'vendor/ninjars/src/lib.rs':
            marker = '#[cfg(test)]\nmod tests {'
            assert marker in text
            region = text.split(marker, 1)[0]
            kind = 'production prefix; embedded cfg(test) module excluded'
        elif name == 'vendor/ninjars/smoke.f90':
            kind, region = 'installer smoke test, not production amplitude code', ''
        records[name] = {'sha256': sha(raw), 'bytes': len(raw), 'classification': kind,
                         'production_lines_searched': len(region.splitlines())}
        for number, line in enumerate(region.splitlines(), 1):
            for label, pattern in PATTERNS.items():
                if re.search(pattern, line, re.I):
                    matches.append({'file': name, 'line': number, 'pattern': label, 'text': line.strip(),
                                    'added_or_modified_line': number in added.get(name, set())})
    result = {'schema': 'bounded-production-genericity-source-census-v1',
              'baseline_revision': BASELINE,
              'observed_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'scope': SCOPE, 'files': records, 'count': len(records), 'patterns': PATTERNS,
              'matches_for_manual_review': matches,
              'new_production_matches': [item for item in matches if item['added_or_modified_line']],
              'source_diff_sha256': sha(patch.encode()),
              'interpretation': 'Text census plus separate manual dispatch/fallback review; no proof over all models or phase space',
              'builds_started': 0, 'tests_started': 0, 'reduction_engine_calls': 0}
    (HERE / 'source-census.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'files': len(records), 'historical_matches': len(matches),
                      'new_production_matches': result['new_production_matches'],
                      'builds_started': 0, 'tests_started': 0, 'reduction_engine_calls': 0}, indent=2))


if __name__ == '__main__':
    main()
