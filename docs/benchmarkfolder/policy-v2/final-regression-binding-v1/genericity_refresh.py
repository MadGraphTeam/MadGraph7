#!/usr/bin/env python3
"""Bounded refresh of the existing 42-path genericity review at the final commit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
OLD = ROOT/'docs/benchmarkfolder/final-genericity-audit-v1'
FINAL = 'c2b48e988fe6e3c6ba9403b67ec4d26fa5fa9ccb'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*arguments):
    return subprocess.check_output(['git', *arguments], cwd=ROOT)


def require(value, reason):
    if not value:
        raise RuntimeError(reason)


def production_region(name, text):
    if name == 'vendor/ninjars/src/tensor_dense_reference.rs' or (
            name.startswith('vendor/ninjars/src/') and name.endswith('_tests.rs')):
        return '', 'cfg(test)-only module'
    if name == 'vendor/ninjars/src/lib.rs':
        marker = '#[cfg(test)]\nmod tests {'
        require(marker in text, 'Missing ordinary cfg(test) boundary')
        return text.split(marker, 1)[0], 'production prefix; embedded cfg(test) module excluded'
    if name == 'vendor/ninjars/src/tensor.rs':
        return text.split('#[cfg(test)]', 1)[0], 'production tensor prefix; three cfg(test) modules excluded'
    if name == 'vendor/ninjars/smoke.f90':
        return '', 'installer smoke, not production amplitude code'
    return text, 'production'


def refresh():
    original_manifest = json.loads((OLD/'manifest.json').read_text())
    for name, item in original_manifest['files'].items():
        raw = (OLD/name).read_bytes()
        require(len(raw) == item['bytes'] and sha(raw) == item['sha256'], 'Old genericity seal changed: '+name)
    original = json.loads((OLD/'source-census.json').read_text())
    require(original['count'] == len(original['files']) == 42, 'Old 42-source census changed')
    require(original['new_production_matches'] == [], 'Old census no longer has zero new dispatch matches')
    names = git('diff', '--name-only', original['baseline_revision'], FINAL, '--', *original['scope']).decode().splitlines()
    added_names = sorted(set(names)-set(original['files']))
    expected_added = ['vendor/ninjars/src/'+name for name in (
        'final_precision_tests.rs', 'panic_contract_tests.rs', 'tensor_allocation_tests.rs',
        'tensor_capture_tests.rs', 'tensor_dense_reference.rs')]
    require(added_names == expected_added, 'Unexpected new production-scope path')
    require(set(original['files']).issubset(names) and len(names) == 47, 'Changed census path inventory')
    patch = git('diff', '--unified=0', original['baseline_revision'], FINAL, '--', *original['scope']).decode()
    additions, filename, line_number = {}, None, None
    for line in patch.splitlines():
        if line.startswith('+++ b/'):
            filename = line[6:]
            additions.setdefault(filename, set())
        elif line.startswith('@@'):
            line_number = int(re.search(r'\+(\d+)', line)[1])
        elif line_number is not None and line.startswith('+') and not line.startswith('+++'):
            additions[filename].add(line_number)
            line_number += 1
        elif line_number is not None and not line.startswith(('-', '\\', 'diff ', 'index ')):
            line_number += 1
    files, matches, changed = {}, [], []
    for name in names:
        raw = git('show', FINAL+':'+name)
        require((ROOT/name).read_bytes() == raw, 'Live reviewed source differs from exact candidate: '+name)
        text = raw.decode()
        region, classification = production_region(name, text)
        files[name] = {'sha256': sha(raw), 'bytes': len(raw), 'classification': classification,
                       'production_lines_searched': len(region.splitlines())}
        if name in original['files'] and sha(raw) != original['files'][name]['sha256']:
            changed.append(name)
        for number, line in enumerate(region.splitlines(), 1):
            for label, pattern in original['patterns'].items():
                if re.search(pattern, line, re.I):
                    matches.append({'file': name, 'line': number, 'pattern': label, 'text': line.strip(),
                                    'added_or_modified_line': number in additions.get(name, set())})
    require(changed == ['madgraph/various/ninjars_install.py', 'vendor/ninjars/src/lib.rs', 'vendor/ninjars/src/tensor.rs'],
            'Unexpected changed path among old 42-source review')
    new_matches = [row for row in matches if row['added_or_modified_line']]
    require(new_matches == [], 'New production dispatch-pattern match requires review')
    lib = git('show', FINAL+':vendor/ninjars/src/lib.rs').decode()
    tensor = git('show', FINAL+':vendor/ninjars/src/tensor.rs').decode()
    require('#[cfg(test)]\nmod final_precision_tests;' in lib and '#[cfg(test)]\nmod panic_contract_tests;' in lib,
            'Final precision/panic tests are not cfg(test)-only')
    for filename, module in [('tensor_dense_reference.rs', 'dense_reference'),
                             ('tensor_allocation_tests.rs', 'allocation_tests'),
                             ('tensor_capture_tests.rs', 'capture_tests')]:
        require(f'#[cfg(test)]\n#[path = "{filename}"]\nmod {module};' in tensor,
                'Tensor helper not cfg(test)-only: '+filename)
    old_tensor = git('show', original['observed_head']+':vendor/ninjars/src/tensor.rs').decode()
    old_region, _ = production_region('vendor/ninjars/src/tensor.rs', old_tensor)
    new_region, _ = production_region('vendor/ninjars/src/tensor.rs', tensor)
    # The allocation delta adds no production selector: all existing conditional
    # and match-selector source lines remain unchanged, in order.
    selectors = lambda value: [line.strip() for line in value.splitlines() if re.match(
        r'\s*(?:if\b|\}\s*else\s+if\b|let\s+\(shape,|let\s+powers:|let\s+\(xs,|let\s+mut\s+g_powers|match\b)', line)]
    require(selectors(old_region) == selectors(new_region), 'Tensor dispatch/conditional source lines changed')
    require(not any(token in new_region for token in ('std::env', 'std::fs', 'include_str!', 'include_bytes!', 'std::process')),
            'Production tensor unexpectedly depends on external input/file/process state')
    formula = 'a.re.clone() * &b.re - a.im.clone() * &b.im,\n        a.re.clone() * &b.im + a.im.clone() * &b.re,'
    require(formula in new_region, 'Reviewed borrowed complex-product expression changed')
    installer_proof = ROOT/'docs/benchmarkfolder/experiments/ninjars-installer-transaction-v1/validation.json'
    installer = json.loads(installer_proof.read_text())
    require(installer['final_green']['tests'] == 37 and installer['final_green']['passed'], 'Installer proof not passed')
    for name, expected in installer['source_files'].items():
        require(sha(git('show', FINAL+':'+name)) == expected, 'Final installer source/test differs from 37-test proof')
    return {'schema': 'final-genericity-refresh-v1', 'candidate_revision': FINAL,
            'original_review_revision': original['observed_head'], 'baseline_revision': original['baseline_revision'],
            'original_census_sha256': sha((OLD/'source-census.json').read_bytes()),
            'original_manifest_sha256': sha((OLD/'manifest.json').read_bytes()),
            'old_source_paths': 42, 'source_paths': 47, 'new_cfg_test_only_paths': added_names,
            'changed_old_paths': changed, 'files': files, 'patterns': original['patterns'],
            'matches_for_manual_review': matches, 'new_production_matches': new_matches,
            'source_diff_sha256': sha(patch.encode()),
            'tensor_conditional_lines_unchanged': len(selectors(new_region)),
            'tensor_delta': 'Generic borrowed complex products, lazy scratch growth and finite zero/one set_from resets; recurrence/traversal/selector lines retained; cfg(test)-only oracle/capture modules.',
            'installer_delta': {'proof': str(installer_proof.relative_to(ROOT)), 'sha256': sha(installer_proof.read_bytes()),
                                'tests': 37, 'source_sha256': installer['source_files'],
                                'scope': 'Recoverable filesystem publication/cleanup only, outside amplitude dispatch'},
            'assessment': 'No new process, model, fixed-point, expected-output or benchmark dispatch found in the bounded exact-source review; regex census is supporting evidence, not arbitrary-model correctness proof.',
            'builds_started': 0, 'tests_started': 0, 'reduction_engine_calls': 0}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = refresh()
    if args.output:
        with args.output.open('x') as stream:
            stream.write(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps({'candidate_revision': result['candidate_revision'], 'old_source_paths': 42,
                     'source_paths': 47, 'new_cfg_test_only_paths': len(result['new_cfg_test_only_paths']),
                     'new_production_matches': result['new_production_matches'],
                     'tensor_conditional_lines_unchanged': result['tensor_conditional_lines_unchanged'],
                     'tests_started': 0, 'reduction_engine_calls': 0}, indent=2))
