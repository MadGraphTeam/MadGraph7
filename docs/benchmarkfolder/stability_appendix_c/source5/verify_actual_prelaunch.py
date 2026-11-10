"""Exercise the exact prelaunch and postguard on twenty fresh private copies.

Writes are confined to the separately named ignored proof namespace. No lock,
subprocess, loader query, native library or amplitude is invoked. The actual
work5 namespace is not created. Retain stdout as actual-prelaunch-proof.json.
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
PROOF = ROOT / '.tirust-build/stability-appendix-c-prelaunch-proof-5-3'
PILOT_SHA = '22b06cda2303d9d2fd42aed418aa4518db0d9e834678862330ffa0d6600e9eef'


def load_exact(path, checksum, name):
    blob = path.read_bytes()
    assert hashlib.sha256(blob).hexdigest() == checksum, str(path)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(blob, str(path), 'exec'), module.__dict__)
    return module


def main():
    pilot = load_exact(HERE / 'pilot.py', PILOT_SHA, 'checked_source5')
    relative, checksum = pilot.HELPERS['links']
    links = load_exact(ROOT / relative, checksum, 'checked_source5_links')
    # Finish pure Python imports before the launch/load audit guard. The
    # actual MadLoopParam writer is used below, without a mock replacement.
    sys.path.insert(0, str(ROOT))
    from madgraph.various.banner import MadLoopParam
    assert MadLoopParam is not None
    value = pilot.protocol()
    assert not PROOF.exists() and not PROOF.is_symlink()
    assert not pilot.WORK.exists() and not pilot.WORK.is_symlink()
    state = {'active': False, 'denied': [], 'writes': set(), 'reads': set(), 'nofollow_metadata': None}
    forbidden = {'subprocess.Popen', 'ctypes.dlopen', 'os.system', 'os.exec',
        'os.posix_spawn', 'os.posix_spawnp', 'os.fork', 'os.forkpty',
        'os.putenv', 'os.unsetenv', 'os.chdir'}
    mutation_events = {'os.mkdir', 'os.remove', 'os.rename', 'os.rmdir',
        'os.link', 'os.symlink', 'os.truncate', 'os.chmod', 'os.chown', 'os.utime'}

    def private(path):
        if isinstance(path, int):
            path = os.readlink('/proc/self/fd/' + str(path))
        if not isinstance(path, (str, bytes, os.PathLike)):
            return False
        return Path(os.fsdecode(path)).resolve().is_relative_to(PROOF)

    # Python's metadata audit events omit follow_symlinks. Preserve the exact
    # OS call and expose that flag to the guard, so copy2 may update a newly
    # copied symlink node without granting permission to its external target.
    metadata_originals = {}; metadata_proxies = []
    for name in ('utime', 'chmod', 'chown'):
        original = getattr(os, name)
        metadata_originals[name] = original
        def proxy(*args, _original=original, _event='os.' + name, **kwargs):
            previous = state['nofollow_metadata']
            path = args[0] if args else kwargs.get('path')
            state['nofollow_metadata'] = (_event, str(path)) if kwargs.get('follow_symlinks', True) is False else None
            try:
                return _original(*args, **kwargs)
            finally:
                state['nofollow_metadata'] = previous
        setattr(os, name, proxy)
        if original in os.supports_follow_symlinks:
            os.supports_follow_symlinks.add(proxy)
            metadata_proxies.append(proxy)

    def audit(event, args):
        if not state['active']:
            return
        deny = event in forbidden
        destinations = []
        if event == 'open':
            path, mode, flags = args
            writing = ((isinstance(mode, str) and any(c in mode for c in 'wax+'))
                or (isinstance(flags, int) and flags &
                    (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)))
            if writing:
                destinations.append(path)
            else:
                state['reads'].add(str(path))
        elif event in mutation_events:
            destinations = list(args[:2]) if event == 'os.rename' else [args[1] if event in ('os.link', 'os.symlink') else args[0]]
        for destination in destinations:
            # unlink/rename operate on the final directory entry itself;
            # resolving that final symlink would check the wrong endpoint.
            node_operation = event in ('os.remove', 'os.rename', 'os.rmdir', 'os.symlink')
            private_node = (node_operation and isinstance(destination, (str, bytes, os.PathLike))
                and (Path(os.fsdecode(destination)).parent.resolve() / Path(os.fsdecode(destination)).name).is_relative_to(PROOF))
            nofollow_node = (state['nofollow_metadata'] == (event, str(destination))
                and isinstance(destination, (str, bytes, os.PathLike))
                and Path(os.fsdecode(destination)).is_symlink()
                and Path(os.fsdecode(destination)).parent.resolve().is_relative_to(PROOF))
            if not private(destination) and not nofollow_node and not private_node:
                deny = True
            else:
                state['writes'].add(str(destination))
        if deny:
            state['denied'].append({'event': event, 'args': repr(args)})
            raise RuntimeError('Prelaunch proof forbids ' + event + ' outside its private scope')

    sys.addaudithook(audit)
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    started = time.monotonic()
    helpers = {'links': links}
    rows = []
    state['active'] = True
    try:
        admitted = pilot.inherited_preparation(value, helpers)
        PROOF.mkdir()
        (PROOF / 'evidence').mkdir()
        for request in pilot.requests(value):
            case = admitted['checkpoint']['cases'][request['case']]
            master = Path(case['master_snapshot']['root'])
            output = PROOF / 'workers' / request['id']
            record = {'request': request, 'state': 'unexecuted', 'native_started': False,
                'host_accuracy_qualified': False, 'primary_calls_expected': 300}
            context = None
            try:
                # This is the same sequence as run before numerical_lock.
                pilot.verify_production(value)
                links.verify_tree(Path(case['donor_context']['master']), case['donor_context']['expected_master'])
                context = links.prepare_worker(master, output, case['master_snapshot'], case['donor_context']['audit'])
                pilot.prelaunch_worker(value, request, case, output, helpers, record)
                record['state'] = 'prelaunch-passed'
            except BaseException as error:
                record.update(state='failed', error={'class': type(error).__name__, 'message': str(error)})
            finally:
                if context is not None:
                    try:
                        record['readonly_guard'] = pilot.worker_postguard(value, case, output, context, helpers, record)
                        record['readonly_guard_passed'] = True
                    except BaseException as error:
                        record.update(state='failed', readonly_guard_error={'class': type(error).__name__, 'message': str(error)})
                pilot.save(PROOF / 'evidence' / (request['id'] + '.json'), record)
                rows.append(record)
                print(request['id'] + ': ' + record['state'], file=sys.stderr, flush=True)
        final = pilot.inherited_preparation(value, helpers)
        assert final == admitted
        assert not pilot.WORK.exists() and not pilot.WORK.is_symlink()
    finally:
        state['active'] = False
        for name, original in metadata_originals.items():
            setattr(os, name, original)
        for proxy in metadata_proxies:
            os.supports_follow_symlinks.remove(proxy)
    status = 'passed' if len(rows) == 20 and all(row['state'] == 'prelaunch-passed'
        and row.get('readonly_guard_passed') is True for row in rows) and not state['denied'] else 'failed'
    result = {'schema': 'madloop-source5-actual-prelaunch-proof-v1', 'status': status,
        'controller_sha256': PILOT_SHA, 'protocol_sha256': pilot.digest(HERE / 'protocol.json'),
        'proof_private_directory': str(PROOF.relative_to(ROOT)), 'request_count': len(rows), 'requests': rows,
        'inheritance_binding': admitted['binding'],
        'execution_proof': {'started_utc': started_utc, 'elapsed_seconds': time.monotonic()-started,
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'guard': 'audit hook denies child processes, exec/fork, dynamic loading and every write endpoint outside the fresh ignored proof namespace',
            'denied_attempts': state['denied'], 'distinct_read_path_count': len(state['reads']),
            'distinct_private_mutation_paths': len(state['writes']),
            'sorted_read_paths_sha256': hashlib.sha256(json.dumps(sorted(state['reads']), separators=(',', ':')).encode()).hexdigest(),
            'sorted_write_paths_sha256': hashlib.sha256(json.dumps(sorted(state['writes']), separators=(',', ':')).encode()).hexdigest(),
            'actual_work5_absent_after': not pilot.WORK.exists() and not pilot.WORK.is_symlink(),
            'compiler_calls': 0, 'loader_calls': 0, 'native_calls': 0, 'amplitude_calls': 0,
            'point_generation_calls': 0, 'numerical_lock_calls': 0}}
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if status == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
