#!/usr/bin/env python3
"""Replay a fresh relocated copy, denying original-workspace and native access."""
import argparse
import ctypes
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import types

HERE = Path(__file__).resolve().parent
ORIGINAL = Path('/shared/external/madloop_tirust')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    if destination.exists() or ORIGINAL in destination.parents:
        raise ValueError('Proof output must be fresh and outside the original workspace')
    base = Path(tempfile.mkdtemp(prefix='madloop-closed-main-plots-portable-'))
    shutil.copy2(HERE / 'adapter.py', base / 'adapter.py')
    shutil.copytree(HERE / 'figures', base / 'figures')
    shutil.copytree(HERE.parents[1] / 'campaigns/invariant-symmetry-1/numerical-retention-1/sealed-1', base / 'capsule')
    shutil.copytree(HERE.parent / 'plotter-source-5', base / 'plotter')
    before = {p.relative_to(base).as_posix(): digest(p) for p in base.rglob('*') if p.is_file()}
    # Preload the retainer's standard-library modules before installing guards.
    from contextlib import contextmanager
    from pathlib import PurePosixPath
    import re
    from types import SimpleNamespace
    os.chdir(base)
    denials = []
    original_stat, original_lstat = Path.stat, Path.lstat

    def original_path(value):
        if not isinstance(value, (str, bytes, os.PathLike)):
            return False
        path = os.fsdecode(value)
        # Relative paths are resolved against this deliberately relocated CWD;
        # tempfile cleanup uses dirfd-relative names, which must not be confused
        # with the process's original CWD.
        absolute = os.path.abspath(path)
        return absolute == str(ORIGINAL) or absolute.startswith(str(ORIGINAL) + os.sep)

    def denied(event, target):
        denials.append({'event': event, 'target': str(target)})
        raise PermissionError('Portable proof blocked: ' + event)

    def audit(event, values):
        if event in ('open','os.scandir','os.listdir','os.chdir') and values and original_path(values[0]):
            denied(event, values[0])
        if event in ('subprocess.Popen','ctypes.dlopen','os.system','os.exec','os.posix_spawn'):
            denied(event, values[0] if values else '')

    def checked_stat(path, *arguments, **keywords):
        if original_path(path):
            denied('Path.stat', path)
        return original_stat(path, *arguments, **keywords)

    def checked_lstat(path, *arguments, **keywords):
        if original_path(path):
            denied('Path.lstat', path)
        return original_lstat(path, *arguments, **keywords)

    Path.stat, Path.lstat = checked_stat, checked_lstat
    sys.addaudithook(audit)
    controls = [lambda: (HERE / 'adapter.py').read_bytes(), lambda: list(os.scandir(HERE)),
                lambda: (HERE / 'adapter.py').stat(), lambda: subprocess.Popen(['false']),
                lambda: ctypes.CDLL(None)]
    for probe in controls:
        try:
            probe()
        except PermissionError:
            pass
        else:
            raise AssertionError('Negative control escaped a portable replay guard')
    expected_denials = len(denials)
    spec = importlib.util.spec_from_file_location('relocated_closed_main_adapter', base / 'adapter.py')
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    result = adapter.verify(base / 'figures', base / 'capsule', base / 'plotter')
    if len(denials) != expected_denials:
        raise AssertionError('Replay attempted forbidden original/native access')
    after = {p.relative_to(base).as_posix(): digest(p) for p in base.rglob('*') if p.is_file()}
    if after != before:
        raise AssertionError('Relocated evidence changed during pure replay')
    proof = {'schema': 'madloop-stability-retrospective-main-portable-proof-v1', 'passed': True,
             'adapter_sha256': before['adapter.py'],
             'closed_capsule_index_sha256': before['capsule/index.json'],
             'plotter_source_index_sha256': before['plotter/index.json'],
             'figures_index_sha256': before['figures/index.json'],
             'negative_controls': denials, 'negative_controls_passed': expected_denials,
             'unexpected_original_or_native_accesses': 0, 'relocated_files_unchanged': True,
             'relocated_file_count': len(before), 'new_native_evaluations': 0,
             'verification': result, 'relocated_directory': str(base)}
    destination.write_text(json.dumps(proof, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'passed': True, 'negative_controls': expected_denials,
                      'figures': result['figures'], 'proof': str(destination)}))


if __name__ == '__main__':
    main()
