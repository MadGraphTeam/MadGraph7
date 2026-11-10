#!/usr/bin/env python3
"""Relocate this closed text package and prohibit native/process/network use."""
# Preload the standard modules used by frozen helpers before restricting opens.
import argparse
import ast
import contextlib
import datetime
import fcntl
import gzip
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent

def main():
    with tempfile.TemporaryDirectory(prefix='guarded-portable-relocation-') as directory:
        target = Path(directory) / 'qualification'
        shutil.copytree(HERE, target)
        permitted = [target]
        opens = []
        forbidden = ('subprocess.Popen','os.fork','os.forkpty','os.posix_spawn',
                     'os.exec','os.system','ctypes.dlopen','ctypes.dlsym',
                     'socket.connect','socket.bind')
        def audit(event, args):
            if event in forbidden:
                raise AssertionError('Forbidden offline operation: ' + event)
            if event == 'tempfile.mkdtemp':
                path = Path(args[0]).resolve()
                assert path.parent == Path(tempfile.gettempdir()).resolve()
                assert path.name.startswith('guarded-physical-offline-')
                permitted.append(path)
            if event == 'open' and not isinstance(args[0], int):
                path = Path(os.fsdecode(args[0])).resolve()
                assert any(path.is_relative_to(root) for root in permitted), str(path)
                assert path.suffix not in {'.so','.o','.a','.mod','.pyc','.rlib'}
                assert '.so.' not in path.name
                opens.append(str(path))
        sys.addaudithook(audit)
        spec = importlib.util.spec_from_file_location('relocated_guarded_verify',target / 'verify.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.verify()
        print(json.dumps({'relocated_verifier_passed':result['verified'],
                          'captured_texts':result['text_inputs'],
                          'confined_open_events':len(opens),
                          'forbidden_native_process_or_network_operations':0,
                          'historical_native_files_opened':False,
                          'execution_complete_as_recorded':result['execution_complete_as_recorded'],
                          'checked_trap_gate_qualified':result['offline_result']['checked_trap_gate_qualified'],
                          'forced_qp_trap_gate_qualified':result['offline_result']['forced_qp_trap_gate_qualified']},indent=2))

if __name__ == '__main__':
    main()
