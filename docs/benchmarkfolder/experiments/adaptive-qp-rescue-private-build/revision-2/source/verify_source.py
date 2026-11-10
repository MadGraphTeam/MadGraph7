#!/usr/bin/env python3
"""Pure source/provenance checks, not Rust type checking or numerical tests."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=Path.cwd().resolve()


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def function(text, name):
    start=text.index('fn '+name+'(')
    first=text.index('{',start); depth=0
    for pos in range(first,len(text)):
        if text[pos]=='{': depth+=1
        elif text[pos]=='}':
            depth-=1
            if depth==0: return text[start:pos+1]
    raise AssertionError(name)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--sources',action='store_true'); args=parser.parse_args()
    index=json.loads((HERE/'source-index.json').read_text())['files']
    paths=list(HERE.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert {str(p.relative_to(HERE)) for p in paths if p.is_file()}==set(index)|{'source-index.json'}
    for name,entry in index.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        path=HERE/name
        assert path.stat().st_size==entry['bytes'] and sha(path)==entry['sha256'],name
    bindings=json.loads((HERE/'source-bindings.json').read_text())['files']
    if args.sources:
        for name,digest in bindings.items(): assert sha(ROOT/name)==digest,name
    old=(HERE/'original-lib.rs').read_text();new=(HERE/'src/lib.rs').read_text()
    unchanged=['register','dp','dp_checked','dp_poles','settings','dimensions','dp_batch','pack_dp',
               'parse_records','mp_batch','pack_mp','ml_tirust_evaluate_dp','ml_tirust_evaluate_mp',
               'ml_tirust_clear_cache','ml_tirust_set_cache_budget','ml_tirust_context_destroy']
    for name in unchanged: assert function(old,name)==function(new,name),name
    patch=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                      fromfile='a/vendor/tirust/src/lib.rs',tofile='b/vendor/tirust/src/lib.rs'))
    assert patch==(HERE/'implementation.patch').read_text()
    assert new.count('self.mp_records_with(key, records, mode, Self::mp_at)')==1
    assert new.count('workspace.set_input_precision(113);')==3 # two production tier constructors plus retained old test
    assert new.count('pow(u64::from(bits - 120)).inv()')==2
    assert 256-120==136 and 512-120==392
    tests=(HERE/'src/adaptive_tests.rs').read_text()
    assert tests.count('#[test]')==9
    protocol=json.loads((HERE/'protocol.json').read_text())
    assert not protocol['compile_authorized'] and not protocol['native_authorized']
    assert protocol['cpu']==11 and protocol['automatic_retry_count']==0
    assert protocol['compile_seconds']==900 and protocol['native_seconds']==180
    fixture=json.loads((HERE/'fixtures/manifest.json').read_text())
    assert fixture['exact_binary128_roundtrip_checked'] and not fixture['native_executed']
    for name,entry in fixture['fields'].items():
        path=HERE/f'fixtures/group10-{name}.dat'
        assert sha(path)==entry['sha256'] and path.stat().st_size==entry['bytes']
    print(json.dumps({'source_files':len(index),'external_pins':len(bindings),
                      'unchanged_functions':len(unchanged),'new_rust_tests_not_run':9,
                      'compiler_executed':False,'native_executed':False},sort_keys=True))


if __name__=='__main__':main()
