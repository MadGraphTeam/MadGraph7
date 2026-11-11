"""Compact portable-replay tests. No compiler, reducer, or native library runs.

Only standard-library Python is needed. Temporary copies are intentional test
fixtures; forbidden-effect tests use outer deadman stubs as well as the guard.
"""
from contextlib import ExitStack
from pathlib import Path
from unittest import mock
import builtins
import ctypes
import _ctypes
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_replay(root, name):
    spec = importlib.util.spec_from_file_location(name, root / 'replay.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def rejected(call, label):
    try:
        call()
    except (RuntimeError, ValueError, AssertionError):
        return
    raise AssertionError(label + ' was accepted')


def verified(module, root, expected):
    facts = module.verify(root, expected)
    require(isinstance(facts, dict) and facts.get('verified') is True,
            'Portable verifier did not report authenticated success')
    return facts


class DeadmanReached(BaseException):
    """The tested guard failed to intercept an operation; no effect is allowed."""


def deadman(*args, **kwargs):
    raise DeadmanReached('Forbidden effect escaped the replay guard')


def check_guard(module, root, outside):
    memory = {str(p.resolve()): p.read_bytes()
              for p in root.rglob('*') if p.is_file()}
    owned = root / 'README.md'
    if not owned.is_file():
        owned = root / 'replay.py'
    owned = owned.resolve()
    wanted = memory[str(owned)]
    forbidden = outside / 'must-never-be-created-or-read'
    # A broken replay guard cannot create files/processes or load a library:
    # outer stubs intercept those attempts and make this test fail explicitly.
    targets = [(builtins, 'open'), (io, 'open'), (Path, 'open'),
               (Path, 'read_bytes'), (Path, 'read_text'), (Path, 'exists'),
               (Path, 'write_bytes'), (Path, 'write_text'),
               (os, 'open'), (os, 'mkdir'), (subprocess, 'Popen'),
               (ctypes, 'CDLL'), (ctypes, 'PyDLL'),
               (ctypes, '_dlopen'), (_ctypes, 'dlopen')]
    with ExitStack() as stack:
        for owner, name in targets:
            stack.enter_context(mock.patch.object(owner, name, deadman))
        with module.blocked_apis(memory) as attempts:
            require(isinstance(attempts, list), 'Guard must retain denied attempts')
            require(owned.read_bytes() == wanted, 'Owned memory bytes differ')
            require(owned.read_text(encoding='utf-8') == wanted.decode('utf-8'),
                    'Owned memory text differs')
            require(not attempts, 'Owned memory reads attempted external effects')
            probes = [
                ('external-read', lambda: forbidden.read_bytes()),
                ('builtin-open', lambda: builtins.open(forbidden, 'rb')),
                ('io-open', lambda: io.open(forbidden, 'rb')),
                ('metadata', lambda: forbidden.exists()),
                ('process', lambda: subprocess.Popen([sys.executable, '-c', 'pass'])),
                ('native-CDLL', lambda: ctypes.CDLL('must-not-be-loaded')),
                ('native-dlopen', lambda: _ctypes.dlopen('must-not-be-loaded', 0)),
                ('mutation-write', lambda: forbidden.write_bytes(b'must-not-be-written')),
                ('mutation-mkdir', lambda: os.mkdir(forbidden)),
            ]
            for label, call in probes:
                before = len(attempts)
                try:
                    call()
                except DeadmanReached as error:
                    raise AssertionError(label + ': guard failed before deadman') from error
                except RuntimeError:
                    pass
                else:
                    raise AssertionError(label + ': guard allowed a forbidden operation')
                require(len(attempts) > before, label + ': denied attempt was not recorded')
            count = len(attempts)
    require(not forbidden.exists(), 'Forbidden test path was created')
    return count


def main():
    root = HERE
    expected = digest(root / 'index.json')
    module = load_replay(root, 'portable_exact_geometry_replay_test_original')
    verified(module, root, expected)
    with tempfile.TemporaryDirectory(prefix='exact-geometry-portable-test-', dir='/tmp') as tmp:
        scratch = Path(tmp)
        relocated = scratch / 'relocated'
        shutil.copytree(root, relocated)
        moved = load_replay(relocated, 'portable_exact_geometry_replay_test_relocated')
        verified(moved, relocated, expected)

        damaged = scratch / 'corrupt-payload'
        shutil.copytree(root, damaged)
        index = json.loads((damaged / 'index.json').read_text())
        relative = next(iter(index['files']))
        payload = damaged / relative
        payload.write_bytes(payload.read_bytes() + b'\nPORTABLE_TEST_CORRUPTION\n')
        rejected(lambda: module.verify(damaged, expected), 'Corrupted owned payload')

        changed = scratch / 'corrupt-claims'
        shutil.copytree(root, changed)
        index = json.loads((changed / 'index.json').read_text())
        index['claims']['fixture_cases'] += 1
        (changed / 'index.json').write_text(json.dumps(index, sort_keys=True, indent=2) + '\n')
        # Use the modified index's correct hash: semantic recomputation must
        # reject the false claim independently of the external index seal.
        changed_hash = digest(changed / 'index.json')
        rejected(lambda: module.verify(changed, changed_hash), 'Modified index claims')
        attempts = check_guard(module, root, scratch)
    print(json.dumps({'tests_passed': 5, 'in_place': True, 'clean_relocation': True,
                      'corrupt_payload_rejected': True, 'false_claim_rejected': True,
                      'intentional_guard_denials': attempts,
                      'new_compiler_or_reducer_executions': 0,
                      'forbidden_effects': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
