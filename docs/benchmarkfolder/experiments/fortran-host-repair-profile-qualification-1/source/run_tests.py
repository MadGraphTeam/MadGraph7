#!/usr/bin/env python3
"""Guarded pure authoring checks; prints a receipt without writing source files."""
import io
import json
import os
from pathlib import Path
import sys
import unittest
sys.dont_write_bytecode=True
attempts=[]


def audit(event,args):
    bad=event in {'subprocess.Popen','os.system','os.posix_spawn','os.exec','os.fork','os.forkpty','ctypes.dlopen','os.kill','os.killpg',
                  'os.mkdir','os.remove','os.rmdir','os.rename','os.link','os.symlink','os.truncate','os.chmod'} or event.startswith('socket.')
    if event=='open':
        mode=args[1];flags=args[2]
        bad=(isinstance(mode,str) and any(x in mode for x in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
    if bad:
        attempts.append(event);raise RuntimeError('Forbidden pure-test operation:'+event)


sys.addaudithook(audit)
suite=unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent),pattern='test_*.py')
stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
receipt={'status':'passed' if result.wasSuccessful() and not attempts else 'failed','tests':result.testsRun,
         'failures':len(result.failures),'errors':len(result.errors),'forbidden_attempts':attempts,
         'optimized_python':not __debug__,'native_execution':False,'details':stream.getvalue()}
print(json.dumps(receipt,indent=2,sort_keys=True))
raise SystemExit(0 if receipt['status']=='passed' else 1)
