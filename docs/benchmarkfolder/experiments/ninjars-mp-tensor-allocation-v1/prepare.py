#!/usr/bin/env python3
"""Prepare an isolated, allocation-only tensor experiment; never edit production."""
from pathlib import Path
import difflib
import gzip
import hashlib
import json
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
WORK = ROOT / '.benchmark-work/experiments/ninjars-mp-tensor-allocation-v1'
ORIGINAL = ROOT / 'vendor/ninjars'
DEPENDENCY = Path('/home/external/.cargo/git/checkouts/symbolica-cc465e8cf60c7b8e/e679e8c/lib/numerica/src/domains/float')
RUG_TRAITS = Path('/home/external/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/rug-1.30.0/src/float/traits.rs')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert ROOT == Path.cwd(), 'Run from the MadGraph repository root'
    assert subprocess.check_output(['git', 'branch', '--show-current'], text=True).strip() == 'madloop_ninjars'
    WORK.mkdir(parents=True, exist_ok=False)
    bridge = WORK / 'bridge'
    excluded = {'libninjars.a', 'smoke', 'mninjars.mod', 'mninjars.o', 'build.log'}
    shutil.copytree(ORIGINAL, bridge, ignore=shutil.ignore_patterns('target', '__pycache__', *excluded))
    (HERE / 'original').mkdir()
    original_files = {}
    for path in sorted(ORIGINAL.rglob('*')):
        if path.is_file() and path.name not in excluded and 'target' not in path.parts and '__pycache__' not in path.parts:
            rel = path.relative_to(ORIGINAL)
            target = HERE / 'original' / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            original_files[str(rel)] = sha(path)
    (HERE / 'dependency').mkdir()
    dependencies = {}
    for name in ['complex.rs', 'multiprecision.rs']:
        shutil.copyfile(DEPENDENCY / name, HERE / 'dependency' / name)
        dependencies[name] = {'source': str(DEPENDENCY / name), 'sha256': sha(DEPENDENCY / name)}
    shutil.copyfile(RUG_TRAITS, HERE / 'dependency/rug-float-traits.rs')
    dependencies['rug-float-traits.rs'] = {'source': str(RUG_TRAITS), 'sha256': sha(RUG_TRAITS)}
    tensor = (ORIGINAL / 'src/tensor.rs').read_text()
    (bridge / 'src/tensor_dense_reference.rs').write_text(tensor)
    new = tensor
    helper = '''// Allocation-only expansion of the pinned Complex<T>::mul expression.
// The four scalar products, operand order and two additions are unchanged.
#[inline]
fn borrowed_product<T: FloatLike>(a: &Complex<T>, b: &Complex<T>) -> Complex<T> {
    Complex::new(
        a.re.clone() * &b.re - a.im.clone() * &b.im,
        a.re.clone() * &b.im + a.im.clone() * &b.re,
    )
}

'''
    new = new.replace('fn minkowski<T: Real>', helper + 'fn minkowski<T: Real>')
    replacements = {
        'a[0].clone() * &b[0] - a[1].clone() * &b[1] - a[2].clone() * &b[2] - a[3].clone() * &b[3]':
        'borrowed_product(&a[0], &b[0]) - borrowed_product(&a[1], &b[1]) - borrowed_product(&a[2], &b[2]) - borrowed_product(&a[3], &b[3])',
        'values.resize(count, self.zero.clone());': 'values.resize_with(count, || self.zero.clone());',
        'values[0] = self.one.clone();': 'values[0].set_from(&self.one);',
        'values[parent].clone() * &q[c]': 'borrowed_product(&values[parent], &q[c])',
        'c.clone() * v': 'borrowed_product(c, v)',
        'series.resize(count * size, self.zero.clone());': 'series.resize_with(count * size, || self.zero.clone());',
        '*value = self.zero.clone();': 'value.set_from(&self.zero);',
        'series[at(0, 0, 0)] = self.one.clone();': 'series[at(0, 0, 0)].set_from(&self.one);',
        'g0.clone() * &previous[j]': 'borrowed_product(&g0, &previous[j])',
        'beta.clone() * &previous[j - 2]': 'borrowed_product(&beta, &previous[j - 2])',
        's[at(j1, 0, 0)].clone() * &g[j - j1]': 'borrowed_product(&s[at(j1, 0, 0)], &g[j - j1])',
        'series[i * size + at(j as usize, a, l)].clone() * c': 'borrowed_product(&series[i * size + at(j as usize, a, l)], c)',
    }
    for i in range(3):
        replacements[f'request.parameters[{i}].clone() * e4'] = f'borrowed_product(&request.parameters[{i}], e4)'
    for i, args in enumerate(['j, a, l', 'j - 1, a, l', 'j - 1, a - 1, l', 'j - 2, a, l', 'j - 2, a - 1, l', 'j - 2, a - 2, l', 'j - 2, a, l - 1']):
        replacements[f'f[{i}].clone() * &source[at({args})]'] = f'borrowed_product(&f[{i}], &source[at({args})])'
    for before, after in replacements.items():
        assert before in new, before
        new = new.replace(before, after)
    new += '\n#[cfg(test)]\n#[path = "tensor_dense_reference.rs"]\nmod dense_reference;\n'
    for name in ['allocation', 'capture']:
        new += f'\n#[cfg(test)]\n#[path = "tensor_{name}_tests.rs"]\nmod {name}_tests;\n'
    (bridge / 'src/tensor.rs').write_text(new)
    for name in ['tensor_allocation_tests.rs', 'tensor_capture_tests.rs']:
        shutil.copyfile(HERE / name, bridge / 'src' / name)
    (HERE / 'tensor.patch').write_text(''.join(difflib.unified_diff(tensor.splitlines(True), new.splitlines(True), fromfile='original/src/tensor.rs', tofile='prototype/src/tensor.rs')))
    shutil.copyfile(bridge / 'src/tensor.rs', HERE / 'tensor.rs')
    capture = ROOT / 'docs/benchmarkfolder/experiments/generic-tensor-plan/capture/actual-mp-requests.jsonl.gz'
    with gzip.open(capture, 'rb') as src, (WORK / 'actual-mp-requests.jsonl').open('wb') as dst:
        shutil.copyfileobj(src, dst)
    provenance = {
        'schema': 'isolated-mp-tensor-allocation-v1',
        'acceptance_evidence': False,
        'production_changed': False,
        'original_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'original_files': original_files, 'dependency_implementations': dependencies,
        'excluded_build_products': {name: sha(ORIGINAL / name) for name in sorted(excluded) if (ORIGINAL / name).is_file()},
        'prototype_tensor_sha256': sha(bridge / 'src/tensor.rs'),
        'capture': {'path': str(capture.relative_to(ROOT)), 'sha256': sha(capture), 'uncompressed_sha256': sha(WORK / 'actual-mp-requests.jsonl')},
        'precision_policy': 'Retain all original Float precision tracking, exact scalar expression/operand order, recurrence, contraction and structural-zero operations. Do not reset intermediate precision to 256.',
        'scope': ['Borrowed Complex products remove only the two redundant outer scalar clones.', 'Reuse existing zero/one scratch allocation with set_from; lazy Vec growth.'],
    }
    (HERE / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps({'bridge': str(bridge), 'provenance': str(HERE / 'provenance.json')}))


if __name__ == '__main__':
    main()
