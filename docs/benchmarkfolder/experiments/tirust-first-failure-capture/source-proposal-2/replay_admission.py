#!/usr/bin/env python3
"""Additive admission for six already captured standalone tensor requests.

This imports unchanged source1 and changes only its source-inventory admission
for two exact subsequently approved template hashes. The standalone executable,
library/runtime, ABI, original inputs and failed capture remain unchanged.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
SOURCE1 = HERE.with_name('source-proposal-1')
SOURCE1_SHA = '2b4bac95022fe550902164a0636699f46798bf4733185610d8f9829f6cd3880d'
CONTROLLER_SHA = 'fb121299e65adba6d49ece5577d8586c32f0cd89b5d998b7dae599a391833645'
WORK1 = ROOT / '.tirust-build/tirust-first-failure-capture-1'
WORK = ROOT / '.tirust-build/tirust-first-failure-replay-2'
PREPARE_SHA = '41ddbc70654dac71134bfc11a1892e8795786d869b603abe76f5bcf735b51309'
CAPTURE_SHA = '7f21e3bd9e2b8bf0458c0c6008c500bc941786337523f863c5d64dec9c72f05e'
GUARD_PROOF = ROOT / '.tirust-build/tirust-first-failure-closed-validation-1/helfilter-proof.json'
GUARD_PROOF_SHA = '230d1fc54362032ebcc47befdc2ee7f33165a1d9d76942a68d4c9fa5caa0c18d'
CHANGES = {
    'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc': {
        'old': 'fdcd4433366cd558b29e1cd4cad5aee0c8a10c669dd6c9311a642740628893a8',
        'new': '3223f38d2e61243f2b2849d08982d17eadf2a93ae1dc2edaccaa44515467494b'},
    'madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc': {
        'old': '4ccd4456d0b28f39185e10645e34b6ed2423cfbe2897c2620712e5f9432f461b',
        'new': 'ece7c4e855b50f13822f441d7be227631293acbc9ed345a2f1a154f0329757a2'},
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def digest(path):
    return sha(Path(path).read_bytes())


def seal():
    blob = (HERE / 'source-index.json').read_bytes()
    index = json.loads(blob)
    for name, item in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,
                'Unsafe additive source path')
        data = (HERE / name).read_bytes()
        require(item == {'sha256': sha(data), 'bytes': len(data)}, 'Additive source differs: ' + name)
    for name, checksum in index['inputs'].items():
        require(digest(ROOT / name) == checksum, 'Additive input differs: ' + name)
    return sha(blob)


def adjusted_inventory(root, files, original):
    """Keep the original complete verifier; substitute exactly two hashes."""
    require(Path(root).resolve() == ROOT.resolve(), 'Unexpected production source root')
    adjusted = dict(files)
    for name, change in CHANGES.items():
        require(files.get(name) == change['old'], 'Frozen template identity differs: ' + name)
        require(digest(Path(root) / name) == change['new'], 'Approved template successor differs: ' + name)
        adjusted[name] = change['new']
    return original(root, adjusted)


def controller():
    require(digest(SOURCE1 / 'source-index.json') == SOURCE1_SHA, 'Source1 seal differs')
    path = SOURCE1 / 'controller.py'
    require(digest(path) == CONTROLLER_SHA, 'Frozen controller differs')
    spec = importlib.util.spec_from_file_location('first_failure_source1', path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    original_load = result.module_from_bytes

    def load(path, checksum):
        module = original_load(path, checksum)
        if Path(path).resolve() == (result.PILOT / 'pilot.py').resolve():
            original = module.verify_source_inventory
            module.verify_source_inventory = lambda root, files: adjusted_inventory(root, files, original)
        return module

    result.module_from_bytes = load
    result.WORK = WORK
    result.REPORT = WORK / 'evidence'
    return result


def readonly_inputs():
    source = seal()
    c = controller()
    preparation = WORK1 / 'evidence/prepare-terminal.json'
    capture = WORK1 / 'evidence/capture-terminal.json'
    require(digest(preparation) == PREPARE_SHA and digest(capture) == CAPTURE_SHA,
            'Original preparation or failed capture differs')
    require(digest(GUARD_PROOF) == GUARD_PROOF_SHA, 'Additive filter validation differs')
    rows = c.capture_rows(c.read(capture))
    available = [row for row in rows if row['state'] == 'captured']
    require(len(rows) == 12 and len(available) == 6 and
            all(row['case_id'] == 1 for row in available), 'Only six captured Zgg requests are admitted')
    for row in available:
        path = Path(row['path'])
        require(path.is_relative_to(WORK1 / 'capture/qcd_n3_uux_zgg') and
                digest(path) == row['sha256'] and path.stat().st_size == row['size'],
                'Captured tensor input differs')
    original = c.readonly_inputs()
    return {'source2_index_sha256': source, 'source1_index_sha256': SOURCE1_SHA,
            'prepare_terminal_sha256': PREPARE_SHA, 'failed_capture_terminal_sha256': CAPTURE_SHA,
            'additive_guard_proof_sha256': GUARD_PROOF_SHA, 'approved_template_changes': CHANGES,
            'inputs': available, 'missing_input_count': 6, 'maximum_tensor_evaluations': 12,
            'original_readonly_preflight': original, 'compiler_invocations': 0,
            'loader_invocations': 0, 'amplitude_evaluations': 0, 'standalone_evaluations': 0}


def execute():
    admitted = readonly_inputs()
    require(not WORK.exists(), 'Additive standalone replay cannot be retried in place')
    c = controller()
    c.save(c.REPORT / 'additive-admission.json', admitted)
    for name in ('prepare-terminal.json', 'capture-terminal.json'):
        blob = (WORK1 / 'evidence' / name).read_bytes()
        with (c.REPORT / name).open('xb') as stream:
            stream.write(blob)
    c.native('replay')
    terminal = c.read(c.REPORT / 'replay-terminal.json')
    require(sum(row.get('replay_summary', {}).get('evaluate_count', 0)
                for row in terminal['results']) <= 12, 'Concrete replay evaluation bound exceeded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        execute()
    else:
        print(json.dumps(readonly_inputs(), indent=2, sort_keys=True, allow_nan=False))
