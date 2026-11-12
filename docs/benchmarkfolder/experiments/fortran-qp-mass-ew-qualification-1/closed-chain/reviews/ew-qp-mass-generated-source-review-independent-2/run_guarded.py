"""Independent launcher for pure helper2 tests/probes; historical GEN never runs."""
import builtins
from contextlib import ExitStack
import ctypes
import fcntl
import os
from pathlib import Path
import resource
import runpy
import signal
import socket
import subprocess
import sys
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'ew-qp-mass-generated-source-review-proposal-2'
def main():
    target=sys.argv[1];effects=[];original=builtins.__import__
    def denied(*a,**kw):effects.append(repr(a[:1]));raise AssertionError('Independent pure review forbids effects')
    def imports(name,*args,**kwargs):
        if name.split('.')[0] in {'madgraph','mg5decay','aloha','MadSpin','madmatrix','PLUGIN'}:return denied(name)
        return original(name,*args,**kwargs)
    with ExitStack() as stack:
        for module,name in ((subprocess,'Popen'),(subprocess,'run'),(subprocess,'check_output'),(ctypes,'CDLL'),(os,'system'),(os,'fork'),(os,'execv'),(os,'execve'),(resource,'setrlimit'),(signal,'signal'),(signal,'setitimer'),(fcntl,'flock'),(socket,'socket'),(socket,'create_connection')):stack.enter_context(patch.object(module,name,side_effect=denied))
        stack.enter_context(patch.object(builtins,'__import__',side_effect=imports));sys.argv=[str(SOURCE/target)]
        runpy.run_path(str(SOURCE/target),run_name='__main__')
    if effects:raise AssertionError(effects)
if __name__=='__main__':main()
