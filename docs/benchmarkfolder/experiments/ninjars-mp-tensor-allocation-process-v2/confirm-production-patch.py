#!/usr/bin/env python3
"""Confirm the held source patch exactly matches the measured release body."""
import json
import subprocess
from build import HERE, ROOT, OLD, binding, sha, write

assert sha(OLD / 'production-with-tests.patch') == '60f3bdb4c3bcf0e9369aebf6342c8c4f485b71099ea94eb5c646bf0c75693f24'
subprocess.run(['git','apply','--check',str(OLD / 'production-with-tests.patch')],check=True)
prior = OLD.parent / 'ninjars-mp-tensor-allocation-transport-v1'
assert sha(prior / 'production-tensor.rs') == '240bbc1f9c8937393227c16a6ce4887ee68524a3f021a8a93656151159445d70'
build = json.loads((OLD / 'exact-build/result.json').read_text())
assert build['allocation_sources']['src/tensor.rs'] == sha(prior / 'production-tensor.rs')
full = (prior / 'reused/tensor.rs').read_text()
body = (prior / 'production-tensor.rs').read_text()
assert full.startswith(body)
assert full[len(body):] == '''
#[cfg(test)]
#[path = "tensor_dense_reference.rs"]
mod dense_reference;

#[cfg(test)]
#[path = "tensor_allocation_tests.rs"]
mod allocation_tests;

#[cfg(test)]
#[path = "tensor_capture_tests.rs"]
mod capture_tests;
'''
ordinary = (prior / 'reused/tensor_allocation_tests.rs').read_text()
capture = (prior / 'reused/tensor_capture_tests.rs').read_text()
assert ordinary.count('#[test]') == 4 and capture.count('#[test]') == 2
assert capture.count('#[ignore = ') == 2
record = {'schema':'allocation-final-production-patch-confirmation-v1',
    'controller':binding(__import__('pathlib').Path(__file__)),
    'patch':binding(OLD / 'production-with-tests.patch'),
    'prior_proposal':binding(OLD / 'production-proposal.json'),
    'measured_release_body':binding(prior / 'production-tensor.rs'),
    'validated_source_with_tests':binding(prior / 'reused/tensor.rs'),
    'release_body_identical_to_measured':True,'new_ordinary_oracle_tests':4,
    'ignored_capture_integration_tests':2,
    'test_only_dense_reference_sha256':sha(prior / 'original/src/tensor.rs'),
    'source_files_touched':['vendor/ninjars/src/tensor.rs','vendor/ninjars/src/tensor_dense_reference.rs',
        'vendor/ninjars/src/tensor_allocation_tests.rs','vendor/ninjars/src/tensor_capture_tests.rs'],
    'lib_rs_untouched':True,'git_apply_check_passed':True,'production_changed':False,
    'preparation_diagnostic':'Initial ad-hoc confirmation counted #[ignore] literally and stopped on attributes written #[ignore = reason]; source/patch checks before that passed, no numerical execution. This source-backed check uses the actual Rust attribute spelling.',
    'required_final_followup':'Root applies after independent verification, runs final combined Rust/MG7 suites and rebuilds/installs final runtime; original-Ninja performance gates remain outstanding.'}
write(HERE / 'production-patch-confirmation.json',record)
print(json.dumps(record,indent=2),flush=True)
