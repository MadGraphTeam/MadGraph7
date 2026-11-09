#!/usr/bin/env python3
"""Prepare one exact denominator reversal; never compile or evaluate a reducer."""
import argparse
import copy
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
REPORT = Path(__file__).resolve().parent
ROOT = REPORT.parents[5]
FORWARD = REPORT.parent
TRACE = REPORT.parents[1]
WORK = ROOT / '.tirust-build/ew-threshold-tensor-replay-reversal'
OLD_WORK = ROOT / '.tirust-build/ew-threshold-tensor-replay'
RUNTIME = ROOT / '.tirust-build/ew-threshold-tensor-replay-runtime'
spec = importlib.util.spec_from_file_location('frozen_forward_preparation', FORWARD / 'prepare.py')
forward = importlib.util.module_from_spec(spec)
spec.loader.exec_module(forward)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def extract(point, group):
    assert point['index'] == 2 and group == 5
    result = copy.deepcopy(forward.extract(point, group))
    assert result['n'] == 4 and result['rank'] == 3 and result['historical_status'] == 1
    n = result['n']
    shifts = {row[0]: row[1:] for row in result['shifts']}
    masses = {row[0]: row[1:] for row in result['masses']}
    invariants = {(row[0], row[1]): row[2] for row in result['invariants']}
    result['shifts'] = [[i, *shifts[n+1-i]] for i in range(1, n+1)]
    result['masses'] = [[i, *masses[n+1-i]] for i in range(1, n+1)]
    result['invariants'] = [[i, j, invariants[n+1-i, n+1-j]]
                            for i in range(1, n+1) for j in range(1, n+1)]
    result['permutation_new_to_original'] = list(range(n, 0, -1))
    return result


def sources():
    original = forward.sources()
    checks = dict(original['sources_and_link_inputs'])
    # Bind the complete frozen forward package, including original NRS units.
    for path in sorted(FORWARD.iterdir()):
        if path.is_file():
            checks[str(path.relative_to(ROOT))] = digest(path)
    for file in ('prepare.py', 'run.py', 'analyze.py', 'compare.py', 'replay.f90', 'README.md', 'derivation.patch'):
        path = REPORT / file
        checks[str(path.relative_to(ROOT))] = digest(path)
    data = dict(original, sources_and_link_inputs=checks,
        scope='One reversed historical group; source-only preparation and unchanged-executable reuse.',
        permutation_new_to_original=[4, 3, 2, 1],
        shift_policy='Permute exact vectors without translating the loop-momentum origin.')
    path = REPORT / 'source-manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == data, 'Reversal source manifest changed'
    else:
        dump(path, data)
    return data


def prepare():
    if WORK.exists() or (REPORT / 'selected-inputs.json').exists():
        raise ValueError('Refusing to overwrite reversal evidence')
    patch = ''
    for file in ('run.py', 'analyze.py'):
        patch += ''.join(difflib.unified_diff((FORWARD / file).read_text().splitlines(True),
            (REPORT / file).read_text().splitlines(True), fromfile='forward/' + file, tofile='reversal/' + file))
    (REPORT / 'derivation.patch').write_text(patch)
    sources()
    old = json.loads((FORWARD / 'selected-inputs.json').read_text())
    for item in ('trace', 'summary', 'execution'):
        assert digest(TRACE / (item + '.json')) == old['parent_' + item + '_sha256']
    trace = json.loads((TRACE / 'trace.json').read_text())
    record = extract(trace['2'], 5)
    lines = ['EW_TENSOR_REPLAY_V1', '1', '2 5 4 3 1', record['scale']]
    lines.extend(' '.join(row[1:]) for row in record['shifts'])
    lines.extend(' '.join(row[1:]) for row in record['masses'])
    inv = {(row[0], row[1]): row[2] for row in record['invariants']}
    lines.extend(' '.join(inv[i, j] for j in range(1, 5)) for i in range(1, 5))
    (REPORT / 'selected-inputs.dat').write_text('\n'.join(lines) + '\n')
    WORK.mkdir()
    for file in ('replay.f90', 'selected-inputs.dat'):
        shutil.copy2(REPORT / file, WORK / file)
    compiled = json.loads((FORWARD / 'compilation.json').read_text())
    for file in ('replay', 'replay.o'):
        origin = OLD_WORK / file
        assert digest(origin) == compiled['artifacts_sha256'][str(origin.relative_to(ROOT))]
        shutil.copy2(origin, WORK / file)
        assert digest(WORK / file) == digest(origin)
    assert digest(WORK / 'replay.f90') == digest(OLD_WORK / 'replay.f90')
    selected = {key: old[key] for key in ('parent_trace_sha256', 'parent_summary_sha256', 'parent_execution_sha256')}
    selected.update(source_manifest_sha256=digest(REPORT / 'source-manifest.json'),
        input_stream_sha256=digest(REPORT / 'selected-inputs.dat'), records=[record],
        expected_NinjaRS_calls=35, scope='Exact reversal only; prepared without native execution.',
        common_scale_policy=old['common_scale_policy'], unit_numerator_policy=old['unit_numerator_policy'])
    dump(REPORT / 'selected-inputs.json', selected)
    # Reuse the identical executable; preserve its original compiler transcript.
    shutil.copy2(FORWARD / 'build.log', REPORT / 'build.log')
    reused = dict(compiled, reused_existing_executable=True,
        inherited_compilation_sha256=digest(FORWARD / 'compilation.json'),
        original_work=str(OLD_WORK.relative_to(ROOT)), run_script_sha256=digest(REPORT / 'run.py'),
        scope='No new compiler invocation. Exact executable/object copies from the frozen forward replay.',
        artifacts_sha256={str((WORK / f).relative_to(ROOT)): digest(WORK / f)
            for f in ('replay.f90', 'replay.o', 'replay', 'selected-inputs.dat')})
    dump(REPORT / 'compilation.json', reused)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'sources'))
    args = parser.parse_args()
    prepare() if args.action == 'prepare' else sources()
