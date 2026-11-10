"""Read-only actual-donor preflight; never starts a native stage.

Run with the pinned development Python and -B from the repository directory.
The JSON printed on stdout is retained as actual-readonly-preflight.json.
"""
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
PILOT_SHA = '6cf5ff8e296b023ce8640facb1c730437600c95b3a48344369dc48844c37c555'
LINK_PATH = 'docs/benchmarkfolder/experiments/current-cms-replay-supervisor-2/link_closure.py'
LINK_SHA = 'e6ac52e09b699e8929a567a9c963e7f8e52da6e18b0220f8ef8b1ec5a5ac92a7'


def load_exact(path, checksum, name):
    blob = path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == checksum, str(path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(blob, str(path), 'exec'), module.__dict__)
    return module


def main():
    # All Python imports finish before guarding: no module is permitted to
    # create a bytecode cache or load an extension during the checked call.
    pilot = load_exact(HERE / 'pilot.py', PILOT_SHA, 'checked_stability_pilot')
    links = load_exact(ROOT / LINK_PATH, LINK_SHA, 'checked_stability_links')
    protocol_blob = (HERE / 'protocol.json').read_bytes()
    protocol_sha = hashlib.sha256(protocol_blob).hexdigest()
    value = pilot.protocol()
    state = {'active': False, 'denied': [], 'reads': set(), 'read_events': 0}
    mutation_events = {'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir',
        'os.link', 'os.symlink', 'os.truncate', 'os.chmod', 'os.chown',
        'os.utime', 'os.putenv', 'os.unsetenv'}
    native_events = {'subprocess.Popen', 'ctypes.dlopen', 'os.system',
        'os.exec', 'os.posix_spawn', 'os.posix_spawnp', 'os.fork', 'os.forkpty'}

    def audit(event, args):
        if not state['active']:
            return
        deny = event in mutation_events or event in native_events
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
            raise RuntimeError('Read-only preflight forbids ' + event)

    sys.addaudithook(audit)
    start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.monotonic()
    assert not pilot.WORK.exists() and not pilot.WORK.is_symlink()
    state['active'] = True
    try:
        result = pilot.readonly_preflight(value, {'links': links})
    finally:
        state['active'] = False
    elapsed = time.monotonic() - started
    assert not state['denied'], state['denied']
    assert not pilot.WORK.exists() and not pilot.WORK.is_symlink()
    result['execution_proof'] = {
        'controller_sha256': PILOT_SHA, 'protocol_sha256': protocol_sha,
        'links_helper_path': LINK_PATH, 'links_helper_sha256': LINK_SHA,
        'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'started_utc': start_utc, 'elapsed_seconds': elapsed,
        'guard': 'Python audit hook denies writes, child processes, exec/fork and dynamic loading during the entire preflight call',
        'denied_attempts': state['denied'], 'read_open_events': state['read_events'],
        'distinct_read_path_count': len(state['reads']),
        'sorted_read_paths_sha256': hashlib.sha256(json.dumps(sorted(state['reads']),
            separators=(',', ':')).encode()).hexdigest(),
        'work_directory_absent_after': True,
        'native_evaluation_or_timing_qualification': False,
    }
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
