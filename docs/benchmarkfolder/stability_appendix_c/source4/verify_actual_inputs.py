"""Check all twenty closed cards and inherited preparation without mutation."""
import ctypes
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PILOT_SHA = 'd021e31a36e0bf822929ef10cdf1719f50df31800d4196ed751c5f6eaa67ae27'
CAPSULE = HERE.parent / 'closed-preparation-and-failed-run-3'
CAPSULE_SHA = 'fd228945474bab62ae800fc1ebbab82588f8aa13de20bf2ebd7265cda3101838'


def load_exact(path, checksum, name):
    blob = path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == checksum, str(path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(blob, str(path), 'exec'), module.__dict__)
    return module


def main():
    pilot = load_exact(HERE / 'pilot.py', PILOT_SHA, 'checked_source4')
    relative, checksum = pilot.HELPERS['links']
    links = load_exact(ROOT / relative, checksum, 'checked_source4_links')
    index_blob = (CAPSULE / 'index.json').read_bytes()
    assert hashlib.sha256(index_blob).hexdigest() == CAPSULE_SHA
    index = json.loads(index_blob)
    verifier = load_exact(CAPSULE / 'verify.py', index['files']['verify.py']['sha256'], 'checked_closed3')
    value = pilot.protocol()
    state = {'active': False, 'denied': [], 'reads': set(), 'read_events': 0}
    forbidden = {'subprocess.Popen', 'ctypes.dlopen', 'os.system', 'os.exec',
        'os.posix_spawn', 'os.posix_spawnp', 'os.fork', 'os.forkpty', 'os.mkdir',
        'os.remove', 'os.rename', 'os.rmdir', 'os.link', 'os.symlink',
        'os.truncate', 'os.chmod', 'os.chown', 'os.utime', 'os.putenv', 'os.unsetenv'}

    def audit(event, args):
        if not state['active']:
            return
        deny = event in forbidden
        if event == 'open':
            path, mode, flags = args
            deny = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                or (isinstance(flags, int) and flags &
                    (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if not deny:
                state['read_events'] += 1
                state['reads'].add(str(path))
        if deny:
            state['denied'].append({'event': event, 'args': repr(args)})
            raise RuntimeError('Actual-input preflight forbids ' + event)

    sys.addaudithook(audit)
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.monotonic()
    assert not pilot.WORK.exists() and not pilot.WORK.is_symlink()
    state['active'] = True
    try:
        capsule_result = verifier.verify(CAPSULE)
        cards = []
        for case in pilot.CASE_IDS:
            for backend in pilot.BACKENDS:
                name = case + '-backend' + str(backend) + '.dat'
                blob = (CAPSULE / 'written-cards' / name).read_bytes()
                observed = pilot.verify_card_tokens(blob.decode(), pilot.card_settings(backend))
                cards.append({'case': case, 'backend': backend, 'sha256': hashlib.sha256(blob).hexdigest(),
                    'controls': observed, 'status': 'passed'})
        inherited = pilot.inherited_preparation(value, {'links': links})
        schedule = pilot.requests(value)
    finally:
        state['active'] = False
    assert not state['denied'], state['denied']
    assert not pilot.WORK.exists() and not pilot.WORK.is_symlink()
    checkpoint = inherited['checkpoint']
    result = {'schema': 'madloop-source4-actual-input-preflight-v1', 'status': 'passed',
        'controller_sha256': PILOT_SHA, 'protocol_sha256': pilot.digest(HERE / 'protocol.json'),
        'capsule_index_sha256': CAPSULE_SHA, 'closed_capsule_verification': capsule_result,
        'closed_card_readbacks': cards, 'inheritance_binding': inherited['binding'],
        'inherited_cases': {case: {'executable_sha256': item['executable_sha256'],
            'driver_sha256': item['driver_sha256'], 'groups': item['group_layout']['nloopgroups'],
            'frozen_points_sha256': item['donor']['points_sha256'],
            'frozen_point_metadata_sha256': item['donor']['point_metadata_sha256'],
            'physical_card_sha256': item['donor']['physical_card_sha256']}
            for case, item in checkpoint['cases'].items()},
        'schedule': schedule, 'request_count': len(schedule),
        'execution_proof': {'started_utc': start_utc, 'elapsed_seconds': time.monotonic()-started,
            'guard': 'audit hook denies writes, child processes, exec/fork and dynamic loading throughout actual-input checks',
            'denied_attempts': state['denied'], 'read_open_events': state['read_events'],
            'distinct_read_path_count': len(state['reads']),
            'sorted_read_paths_sha256': hashlib.sha256(json.dumps(sorted(state['reads']), separators=(',', ':')).encode()).hexdigest(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'work4_absent_after': True, 'compiler_calls': 0, 'loader_calls': 0,
            'point_generation_calls': 0, 'amplitude_calls': 0}}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
