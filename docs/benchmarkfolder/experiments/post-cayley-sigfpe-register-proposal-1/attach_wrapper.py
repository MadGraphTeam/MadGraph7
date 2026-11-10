"""Run only after root admission: fork a stopped target, then become its GDB."""
import json
import os
from pathlib import Path
import signal
import sys

binding = json.loads(Path('launch.json').read_text())
assert os.getpid() == os.getpgrp() == os.getsid(0)
child = os.fork()
if child == 0:
    os.kill(os.getpid(), signal.SIGSTOP)
    os.execve(binding['executable'], binding['target_argv'], binding['target_environment'])
    os._exit(127)
Path('fork.json').write_text(json.dumps({'debugger_pid': os.getpid(),
    'debugger_pgid': os.getpgrp(), 'inferior_pid': child,
    'fork_inherits_owned_group': True}, indent=2) + '\n')
script = repr(binding['capture_script'])
python_command = ('python exec(compile(open(' + script + ',"rb").read(),' + script +
    ',"exec"),{"__file__":' + script + ',"__name__":"__main__"})')
os.execve(binding['debugger'], [binding['debugger'], '--batch', '--nx', '--nh', '--quiet',
    '-ex', python_command], dict(os.environ))
