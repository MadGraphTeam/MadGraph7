#!/usr/bin/env python3
"""Verify retained text and reparse measurements without native execution."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify():
    index = json.loads((HERE / 'source-index.json').read_text())
    for name, record in index['files'].items():
        path = HERE / name
        assert path.resolve().is_relative_to(HERE)
        assert path.is_file() and not path.is_symlink()
        assert sha(path) == record['sha256']
        assert path.stat().st_size == record['bytes']
    actual = {str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()
              and p.name != 'source-index.json' and '__pycache__' not in p.parts}
    assert actual == set(index['files'])
    spec = json.loads((HERE / 'spec.json').read_text())
    original = (HERE / 'source/gram.rs.baseline').read_text()
    candidate = (HERE / 'source/gram.rs.candidate').read_text()
    assert sha(HERE / 'source/gram.rs.baseline') == spec['upstream_gram_sha256']
    old = '    let ratio = terms / size(&det_x);\n    ratio.is_finite().then_some(ratio)\n'
    new = ('    let denominator = size(&det_x);\n'
           '    if denominator.to_bits() & 0x7fff_ffff_ffff_ffff == 0 {\n'
           '        return None;\n    }\n'
           '    let ratio = terms / denominator;\n'
           '    ratio.is_finite().then_some(ratio)\n')
    assert original.count(old) == 1 and candidate == original.replace(old, new)
    expected = ''.join(difflib.unified_diff(original.splitlines(True), candidate.splitlines(True),
                    fromfile='a/crates/iregi-rust/src/direct/exec/gram.rs',
                    tofile='b/crates/iregi-rust/src/direct/exec/gram.rs'))
    assert expected == (HERE / 'source/0001-guard-zero-cayley-diagnostic-denominator.patch').read_text()
    quotient = '// Extracted final quotient/filter only; size is an identity of a precomputed binary64 denominator.\n'
    for name, tail in [('baseline', old), ('candidate', new)]:
        quotient += ('#[inline(never)]\nfn ' + name + '(terms: f64, denominator: f64) -> Option<f64> {\n'
                     '    let det_x = denominator;\n    let size = |x: &f64| *x;\n' + tail + '}\n\n')
    assert quotient == (HERE / 'source/quotient.rs').read_text()
    path = HERE / 'analyze.py'
    module_spec = importlib.util.spec_from_file_location('quotient_analysis', path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    evidence = HERE / 'evidence/tiny-1'
    result = module.analyze(evidence / 'raw')
    assert result == json.loads((evidence / 'results.json').read_text())
    execution = json.loads((evidence / 'execution.json').read_text())
    assert execution['complete'] and execution['passed']
    assert execution['upstream_gram_before'] == execution['upstream_gram_after'] == spec['upstream_gram_sha256']
    assert len(execution['commands']) == 69
    assert all(c['reaped'] and c['returncode'] in (0, -8) for c in execution['commands'])
    for name, recorded in execution['sources'].items():
        assert sha(HERE / name) == recorded
    assert sha(evidence / 'results.json') == execution['results_sha256']
    return {'passed': True, 'payload_count': len(index['files']),
            'masked_pairs': sum(x['masked_pairs'] for x in result['optimization_levels']),
            'trap_workers': sum(x['trap_workers'] for x in result['optimization_levels']),
            'native_reducer_or_amplitude_execution': False}

if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
