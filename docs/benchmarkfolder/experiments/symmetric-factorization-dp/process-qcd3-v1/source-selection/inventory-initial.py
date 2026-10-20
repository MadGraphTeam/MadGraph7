#!/usr/bin/env python3
"""Read-only inventory of adjacent supported DP vertex/updater statements."""
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
SOURCE = ROOT/'.benchmark-work/experiments/symmetric-factorization-production/dev-qcd3-v1/outputs/candidate/qcd-3'
RETAINED = ROOT/'docs/benchmarkfolder/experiments/symmetric-factorization-production/dev-qcd3-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def statements(text):
    pending = ''
    start = 0
    for index, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line[0] in 'cC*!':
            continue
        assert len(line) >= 6
        if line[5] in ' 0':
            if pending:
                yield start, pending
            start, pending = index, re.sub(r'\s+', '', line[6:])
        else:
            pending += re.sub(r'\s+', '', line[6:])
    if pending:
        yield start, pending


def main():
    directory = next(SOURCE.glob('SubProcesses/P*_*'))
    pattern = re.compile(r'CALL(\w*UPDATE_WL_([2-4])_1)\(WL\(1,0,1,(-?\d+)\),(\d+),COEFS,(\d+),(\d+),WL\(1,0,1,(-?\d+)\)\)')
    entries, bound = [], {}
    for path in sorted(directory.glob('coef_construction_*.f')):
        bound[str(path.relative_to(ROOT))] = sha(path)
        previous = None
        for line, statement in statements(path.read_text()):
            match = pattern.fullmatch(statement)
            vertex = re.fullmatch(r'CALL(FFV1L1_2|FFV1L2_1)\(.*?,COEFS\)', previous[1]) if previous else None
            if match and vertex and match[5] == match[6] == '4':
                assert match[3] != match[7] and 0 <= int(match[4]) <= 4
                entries.append(dict(file=path.name, line=line, rank=int(match[2]),
                                    vertex=vertex[1], vertex_line=previous[0], updater=match[1],
                                    input_wavefunction=int(match[3]), output_wavefunction=int(match[7]),
                                    lcut_size=int(match[4]), original_statement=statement))
            previous = line, statement
    for name in ('generation.json', 'compilation.json', 'compiled-sources.json',
                 'Cards/param_card.dat', 'Cards/MadLoopParams.dat', 'Source/make_opts'):
        path = SOURCE/name
        bound[str(path.relative_to(ROOT))] = sha(path)
    for name in ('polynomial.f', 'coef_specs.inc', 'loop_max_coefs.inc', 'check_sa.f'):
        path = directory/name
        bound[str(path.relative_to(ROOT))] = sha(path)
    for vertex in ('FFV1L1_2', 'FFV1L2_1'):
        path = SOURCE/'Source/DHELAS'/f'{vertex}.f'
        bound[str(path.relative_to(ROOT))] = sha(path)
        stored = HERE.parent/'oracle/proof-inputs'/f'{vertex}.f'
        assert sha(path) == sha(stored), vertex
    for name in ('data/candidate-qcd-3-6-0.json', 'data/candidate-qcd-3-8-0.json',
                 'inputs/candidate/qcd-3/compiled-sources.json'):
        path = RETAINED/name
        bound[str(path.relative_to(ROOT))] = sha(path)
    assert entries
    print(json.dumps(dict(schema='dp-factor-process-source-inventory-v1', complete=True,
        source=str(SOURCE.relative_to(ROOT)), directory=str(directory.relative_to(ROOT)),
        source_files=bound, script_sha256=sha(Path(__file__)), calls=entries,
        counts=dict(Counter(f'{e["vertex"]}/rank{e["rank"]}' for e in entries)),
        maxcoefs_include=(directory/'loop_max_coefs.inc').read_text(),
        scope='Read-only plan inventory; no transformation, compilation or amplitude evaluation'),
        indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
