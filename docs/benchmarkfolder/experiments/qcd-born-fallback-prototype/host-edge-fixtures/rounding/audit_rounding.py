#!/usr/bin/env python3
"""Hash/reparse this fixture's portable evidence; never compile or execute."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
INDEX = HERE / 'evidence-index.json'
FINAL_SHA = 'ff1be878409d741a7491d97dd1db418a8ce6103d1e01775aac36080be9bbec14'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    require(path.is_absolute() and path.is_relative_to(HERE) and '..' not in path.parts and
            path.resolve().is_relative_to(HERE), 'Unsafe evidence path')
    require(not any(item.is_symlink() for item in (path, *path.parents) if item.is_relative_to(HERE)),
            'Symlink evidence path')
    return path.read_bytes()


def digest(path):
    return hashlib.sha256(read(path)).hexdigest()


def identity(path):
    content = read(path)
    return dict(path=path.relative_to(HERE).as_posix(), bytes=len(content), sha256=hashlib.sha256(content).hexdigest())


def same(left, right, label):
    require(json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True), label)


def inventory():
    rows = []
    for path in sorted(HERE.rglob('*')):
        if path.is_file():
            require(path.suffix != '.pyc' and not path.read_bytes().startswith(b'\x7fELF'),
                    'Generated binary/bytecode inside portable report')
        if path == INDEX or 'work' in path.relative_to(HERE).parts or '__pycache__' in path.parts:
            continue
        require(not path.is_symlink(), 'Symlink namespace entry')
        if path.is_file():
            rows.append(identity(path))
        else:
            require(path.is_dir(), 'Nonregular namespace entry')
    return rows


def reparse():
    final = json.loads(read(HERE / 'run-3/execution.json'))
    require(digest(HERE / 'run-3/execution.json') == FINAL_SHA, 'Final execution identity changed')
    require(final['complete'] and final['protected_inputs_unchanged'], 'Incomplete final fixture')
    require(final['cpu'] == 11 and not any(final[key] for key in
            ('physical_amplitudes', 'native_reducers', 'performance_eligible')), 'Unexpected qualification scope')
    parser_source = HERE / 'run-3/source-inputs/run_rounding.py'
    require(digest(parser_source) == final['source_inputs']['run_rounding.py']['sha256'], 'Retained parser identity')
    tree = ast.parse(read(parser_source))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('require', 'parse_output')]
    require({node.name for node in nodes} == {'require', 'parse_output'}, 'Missing retained pure parser')
    namespace = {'re': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(parser_source), 'exec'), namespace)
    parse = namespace['parse_output']
    for name, row in final['source_inputs'].items():
        require(row['sha256'] == digest(HERE / 'run-3/source-inputs' / name),
                'Historical source/snapshot mismatch: ' + name)
        if name == 'run_rounding.py':
            relocation = json.loads(read(HERE / 'work-relocation.json'))
            require(row['sha256'] == relocation['runner_before_sha256'] and
                    digest(HERE / name) == relocation['runner_after_sha256'],
                    'Work-directory-only runner migration mismatch')
            before = read(HERE / 'run-3/source-inputs' / name).decode()
            after = before.replace("work = HERE / 'work' / args.run_name",
                "work = ROOT / '.tirust-build' / 'born-host-edge-fixtures' / 'rounding' / args.run_name")
            after = after.replace('Executables/module files remain in ignored work/;',
                'Executables/module files remain under ignored .tirust-build/born-host-edge-fixtures/rounding;')
            require(read(HERE / name).decode() == after, 'Unexpected runner change beyond artifact paths')
        else:
            require(digest(HERE / name) == row['sha256'], 'Source/snapshot mismatch: ' + name)
    same(final['source_inputs'], final['source_inputs_after'], 'Source mutation during final run')
    for row in final['raw_files']:
        path = ROOT / row['path']
        require(digest(path) == row['sha256'] and len(read(path)) == row['bytes'], 'Final raw/snapshot mismatch')
    expected = [('original-O0', 0), ('original-O3', 0), ('missing-qp-flag-merge', 48), ('conversion-rounding', 16)]
    same([row['name'] for row in final['runs']], [name for name, _ in expected], 'Final run inventory')
    for row, (name, failures) in zip(final['runs'], expected):
        require(row['compile_returncode'] == 0 and row['returncode'] == (91 if failures else 0), 'Unexpected process status')
        for key in ('compile_command', 'execute_command'):
            require(row[key][1:3] == ['-c', '11'] and Path(row[key][0]).name == 'taskset', 'Missing CPU binding')
        parsed = parse(read(HERE / 'run-3' / (name + '.stdout')).decode(), bool(failures))
        same(parsed, row['summary'], 'Final raw/summary mismatch: ' + name)
        require(parsed['cases'] == 128 and parsed['failing_cases'] == failures, 'Unexpected final coverage')
    first = json.loads(read(HERE / 'run-1/execution.json'))
    require(not first['complete'] and first['runs'][0]['compile_returncode'] != 0, 'First attempt failure missing')
    for name, filename in [('rounding_driver.f90', 'rounding_driver-used.f90'), ('run_rounding.py', 'run_rounding-used.py')]:
        require(digest(HERE / 'run-1' / filename) == first['source_inputs'][name]['sha256'], 'First attempt source snapshot')
    second = json.loads(read(HERE / 'run-2/execution.json'))
    require(not second['complete'] and len(second['runs']) == 3, 'Second attempt failure missing')
    for row in second['runs']:
        require(row['compile_returncode'] == 0 and row['returncode'] == 0, 'Second attempt observed outcome')
        parsed = parse(read(HERE / 'run-2' / (row['name'] + '.stdout')).decode(), False)
        require(parsed['cases'] == 128 and parsed['failing_cases'] == 0, 'Second attempt outcome altered')
        if 'summary' in row:
            same(parsed, row['summary'], 'Second attempt summary mismatch')
    return dict(original_cases_per_build=128, original_optimizations=['O0', 'O3'],
                supported_modes=['nearest', 'toward-zero', 'upward', 'downward'], ties_away_supported=False,
                detected_mutations={'missing-qp-flag-merge': 48, 'conversion-rounding': 16},
                undetected_mutation='run-2 missing-final-restore: 128 cases passed; retained as a negative-control limitation',
                historical_compile_failure='run-1 IEEE_FLAG_TYPE equality in the fixture driver',
                performance_eligible=False, physical_amplitudes=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    if not args.write:
        expected = json.loads(read(INDEX))
        same(inventory(), expected['files'], 'Inventory/hash mismatch')
    value = dict(schema=1, namespace=HERE.relative_to(ROOT).as_posix(), files=inventory(),
                 evidence=reparse(), no_native_execution_during_audit=True)
    if args.write:
        with INDEX.open('x') as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write('\n')
    else:
        same(value, expected, 'Reparsed evidence mismatch')
    print(json.dumps(dict(verified=True, files=len(value['files']), index_sha256=digest(INDEX)), indent=2))


if __name__ == '__main__':
    main()
