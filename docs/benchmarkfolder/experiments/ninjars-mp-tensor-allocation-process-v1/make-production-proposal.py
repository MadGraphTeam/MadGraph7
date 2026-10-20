#!/usr/bin/env python3
"""Prepare, but never apply, the exact validated tensor and durable oracle tests."""
import difflib
import json
from pathlib import Path
from build import HERE, ROOT, PRIOR, binding, sha, write

assert Path.cwd() == ROOT
source = ROOT / 'vendor/ninjars/src'
assert sha(source / 'tensor.rs') == 'f5b153e179ac56e4a833f8d83b011a667e61cff1bcb567c2f1508a1d94013e55'
files = {
    'tensor.rs':PRIOR / 'reused/tensor.rs',
    'tensor_dense_reference.rs':PRIOR / 'original/src/tensor.rs',
    'tensor_allocation_tests.rs':PRIOR / 'reused/tensor_allocation_tests.rs',
    'tensor_capture_tests.rs':PRIOR / 'reused/tensor_capture_tests.rs',
}
patch = []
for name,path in files.items():
    before = (source / name).read_text() if (source / name).is_file() else ''
    if name != 'tensor.rs':
        assert not before
    after = path.read_text()
    patch.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),
        fromfile='a/vendor/ninjars/src/'+name if before else '/dev/null',
        tofile='b/vendor/ninjars/src/'+name))
out = HERE / 'production-with-tests.patch'
with out.open('x') as stream:
    stream.write(''.join(patch))
write(HERE / 'production-proposal.json', {'schema':'allocation-final-production-test-proposal-v1',
    'status':'HELD-pending-complete-process-decision-and-root-review',
    'controller':binding(Path(__file__)),'patch':binding(out),
    'sources':{name:binding(path) for name,path in files.items()},
    'release_tensor_body':binding(PRIOR / 'production-tensor.rs'),
    'validated_correctness_manifest':binding(PRIOR / 'manifest.json'),
    'production_changed':False,'new_tests':4,'ignored_integration_tests':2,
    'test_oracle':'Previous production implementation, preserved verbatim as cfg(test)-only module',
    'scope':'Same22-test/21golden/1392callback/1392state validated source; release body exactly matches process experiment'})
print(json.dumps({'proposal':binding(out),'production_changed':False}),flush=True)
