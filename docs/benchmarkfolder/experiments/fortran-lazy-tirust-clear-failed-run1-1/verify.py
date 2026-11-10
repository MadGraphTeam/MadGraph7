"""Portable, read-only verification of the closed preparation failure."""
from pathlib import Path
import hashlib
import json
import re


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(root):
    index = json.loads((root / 'index.json').read_text())
    for name, item in index['files'].items():
        path = root / name
        if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
            raise ValueError('Payload identity mismatch: ' + name)
    capture = json.loads((root / 'capture.json').read_text())
    for name, item in capture['complete_work_regular_inventory'].items():
        if name not in capture['excluded_build_artifact_identities']:
            if sha(root / 'retained' / name) != item['sha256']:
                raise ValueError('Capture inventory mismatch: ' + name)
    if capture['native_controller_observed_exit_status'] != 1 or capture['candidate_source_created']:
        raise ValueError('Incorrect terminal or candidate-stage claim')
    commands = sorted((root / 'retained/commands').glob('*'))
    if len(commands) != 4:
        raise ValueError('Unexpected command scope')
    for path in commands:
        if json.loads((path / 'terminal.json').read_text())['returncode'] != 0:
            raise ValueError('Unexpected compilation/trace failure')
    profile = root / 'retained/workers/profile-original-1'
    native = json.loads((profile / 'ffi-profile.json').read_text())
    if any(native.values()):
        raise ValueError('Unexpected native TIRust lifecycle entry')
    text = (profile / 'stdout.txt').read_text()
    if '/ident_card.dat' not in text or 'not found in the parent directories' not in text:
        raise ValueError('Missing actual model-layout diagnostic')
    if any((profile / name).exists() for name in ('timing.dat', 'values.dat', 'warmup.dat')):
        raise ValueError('Unexpected numerical/timing outputs')
    terminal = json.loads((root / 'retained/terminal.json').read_text())
    if terminal['returncode'] != 1 or 'Worker exited (0) before reply' not in terminal['error']:
        raise ValueError('Failure reason differs')
    trace = (root / 'retained/commands/004-trace-original-observed/stdout.txt').read_text()
    if len(re.findall(r'(?:=>\s+)?/[^\s()]+', trace)) != 15:
        raise ValueError('Unexpected qualified loader trace size')
    return {'verified': True, 'command_count': 4, 'candidate_stage_entered': False,
        'native_tirust_lifecycle_calls': 0, 'numerical_results': 0, 'performance_results': 0,
        'failure': 'isolated MODEL parameter/identification-card layout before READY'}


if __name__ == '__main__':
    print(json.dumps(verify(Path(__file__).resolve().parent), sort_keys=True))
