#!/usr/bin/env python3
"""Read-only inventory of adjacent supported DP vertex/updater statements."""
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import importlib.util
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
    parser_path = HERE.parent.parent/'symmetric-factorization/analyze.py'
    spec = importlib.util.spec_from_file_location('frozen_vertex_proof', parser_path)
    exact = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exact)
    source_parity = {}
    for vertex in ('FFV1L1_2', 'FFV1L2_1'):
        path = SOURCE/'Source/DHELAS'/f'{vertex}.f'
        bound[str(path.relative_to(ROOT))] = sha(path)
        stored = HERE.parent/'oracle/proof-inputs'/f'{vertex}.f'
        old_rows, new_rows = list(statements(stored.read_text())), list(statements(path.read_text()))
        old_executable = next(i for i, (_, row) in enumerate(old_rows) if row.startswith('P2(0)='))
        new_executable = next(i for i, (_, row) in enumerate(new_rows) if row.startswith('P2(0)='))
        assert [r for _, r in old_rows[old_executable:]] == [r for _, r in new_rows[new_executable:]]
        assert sorted(r for _, r in old_rows[:old_executable]) == sorted(r for _, r in new_rows[:new_executable])
        old_coefficients, _ = exact.assignments(stored)
        new_coefficients, _ = exact.assignments(path)
        assert len(old_coefficients) == len(new_coefficients) == 80
        assert old_coefficients == new_coefficients
        proof = exact.prove(path)
        assert proof['passed'] and proof['coefficient_identities_checked'] == 64
        old_proof = exact.prove(stored)
        for key in ('q_rows', 'terminal_required_u_rows', 'b0_nonzero_by_row', 'h_nonzero_by_row'):
            assert proof[key] == old_proof[key]
        source_parity[vertex] = dict(passed=True, current_source_sha256=sha(path),
            retained_source_sha256=sha(stored), exact_coefficient_expressions=80,
            derivative_row_identities=64, executable_statements_equal=True,
            declaration_multisets_equal=True, current_proof=proof)
    for name in ('data/candidate-qcd-3-6-0.json', 'data/candidate-qcd-3-8-0.json',
                 'inputs/candidate/qcd-3/compiled-sources.json'):
        path = RETAINED/name
        bound[str(path.relative_to(ROOT))] = sha(path)
    assert entries
    print(json.dumps(dict(schema='dp-factor-process-source-inventory-v1', complete=True,
        source=str(SOURCE.relative_to(ROOT)), directory=str(directory.relative_to(ROOT)),
        source_files=bound, script_sha256=sha(Path(__file__)), calls=entries,
        source_parity=source_parity, exact_parser_sha256=sha(parser_path),
        counts=dict(Counter(f'{e["vertex"]}/rank{e["rank"]}' for e in entries)),
        maxcoefs_include=(directory/'loop_max_coefs.inc').read_text(),
        scope='Read-only plan inventory; no transformation, compilation or amplitude evaluation'),
        indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
