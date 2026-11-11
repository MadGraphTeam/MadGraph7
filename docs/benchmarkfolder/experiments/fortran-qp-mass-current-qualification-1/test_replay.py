"""Disposable-copy portability/integrity tests; no numerical program runs."""
import ctypes
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mass_retention_replay_tests', HERE/'replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def rejected(call):
    try:
        call()
    except (ValueError, RuntimeError):
        return True
    raise AssertionError('Corrupted or forbidden operation was accepted')


def main():
    draft = (HERE/'draft-inputs.json').exists()
    index_name = 'draft-inputs.json' if draft else 'index.json'
    clean = replay.verify(HERE, draft=draft)
    replay.require(clean['verified'] and clean['forbidden_API_attempts'] == 0, 'Clean replay failed')
    replay.require(clean['numerical']['candidate_promoted'] is False, 'Unqualified promotion claim')
    replay.require(clean['numerical']['independent_physical_finite_truth_qualified'] is False,
                   'Unqualified finite-truth claim')
    with tempfile.TemporaryDirectory(prefix='qp-mass-retention-relocated-') as temporary:
        destination = Path(temporary)/'capsule'
        shutil.copytree(HERE, destination)
        relocated = replay.verify(destination, draft=draft)
        replay.require(relocated == clean, 'Relocated replay differs')
        manifest = json.loads((destination/index_name).read_text())
        payload = destination/next(iter(manifest['files']))
        payload.write_bytes(payload.read_bytes()+b'corruption')
        rejected(lambda: replay.verify(destination, draft=draft))
    with tempfile.TemporaryDirectory(prefix='qp-mass-retention-claims-') as temporary:
        destination = Path(temporary)/'capsule'
        shutil.copytree(HERE, destination)
        manifest = json.loads((destination/index_name).read_text())
        manifest['candidate_promoted'] = True
        (destination/index_name).write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        rejected(lambda: replay.verify(destination, draft=draft))
    with tempfile.TemporaryDirectory(prefix='qp-mass-retention-pinned-input-') as temporary:
        destination = Path(temporary)/'capsule'
        shutil.copytree(HERE, destination)
        manifest = json.loads((destination/index_name).read_text())
        logical = next(name for name in manifest['logical_files']
                       if name.startswith('inputs/') and name.endswith('/generation.json'))
        row = manifest['logical_files'][logical]
        path = destination/row['path']
        payload = path.read_bytes()
        raw = gzip.decompress(payload) if row['compression'] == 'gzip' else payload
        raw += b'\n'  # Same JSON value, different frozen exact input.
        payload = gzip.compress(raw, mtime=0) if row['compression'] == 'gzip' else raw
        path.write_bytes(payload)
        row.update(raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest())
        manifest['files'][row['path']] = {'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        (destination/index_name).write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        rejected(lambda: replay.verify(destination, draft=draft))
    with tempfile.TemporaryDirectory(prefix='qp-mass-retention-handoff-binding-') as temporary:
        destination = Path(temporary)/'capsule'
        shutil.copytree(HERE, destination)
        manifest = json.loads((destination/index_name).read_text())
        path = destination/'followup/source-binding.json'
        binding = json.loads(path.read_text())
        binding['observer_bindings'] = {}
        payload = (json.dumps(binding, indent=2, sort_keys=True)+'\n').encode()
        path.write_bytes(payload)
        manifest['files']['followup/source-binding.json'] = {
            'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        (destination/index_name).write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
        rejected(lambda: replay.verify(destination, draft=draft))
    with replay.blocked_apis() as attempts:
        operations = [lambda: Path('/not-capsule').read_bytes(),
                      lambda: Path('/not-capsule').stat(),
                      lambda: Path('/not-capsule').write_text('forbidden'),
                      lambda: open('/not-capsule'),
                      lambda: os.scandir('/not-capsule'),
                      lambda: os.system('never executed'),
                      lambda: subprocess.run(['never-executed']),
                      lambda: ctypes.CDLL('/never-loaded'),
                      lambda: Path('/not-capsule').resolve()]
        for operation in operations:
            rejected(operation)
        replay.require(len(attempts) == len(operations), 'Guard denial coverage differs')
    print(json.dumps({'tests_passed': 7, 'in_place': True, 'clean_relocation': True,
                      'corrupt_payload_rejected': True, 'false_claim_rejected': True,
                      'resigned_frozen_input_rejected': True, 'empty_handoff_bindings_rejected': True,
                      'intentional_guard_denials': 9, 'forbidden_effects': 0,
                      'new_native_executions': 0, 'sealed': not draft}, sort_keys=True))


if __name__ == '__main__':
    main()
