"""Record real CLI execution; gate compilation and serialize only native smoke."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import time

_original_run = subprocess.run
_work = Path(os.environ['TIRUST_CLEAN_WORK'])


def _record(value):
    value['utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (_work / 'execution-observer.jsonl').open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True) + '\n')


def _run(command, *args, **kwargs):
    actual = command
    if isinstance(command, (list, tuple)):
        argv = [str(value) for value in command]
        if argv and argv[0] == 'make' and 'install' in argv:
            _record({'event': 'waiting-for-compile-window', 'argv': argv})
            while not (_work / 'allow-compilation').is_file():
                time.sleep(2)
            _record({'event': 'compile-window-released', 'argv': argv})
        if argv and Path(argv[0]).name == 'smoke' and '--require-poles' in argv:
            actual = ['flock', os.environ['TIRUST_NUMERICAL_LOCK'], *argv]
            _record({'event': 'serialized-smoke', 'argv': actual})
        _record({'event': 'run', 'argv': argv, 'cwd': str(kwargs.get('cwd', Path.cwd()))})
    return _original_run(actual, *args, **kwargs)


subprocess.run = _run
