"""Execute the reviewed compile/link arrays only; never load the outputs."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import signal
import sys
import time

ROOT = Path.cwd()
PACKAGE = ROOT / 'docs/benchmarkfolder/experiments/generated-rank6-poles-proposal/attempt1'
WORK = ROOT / '.tirust-build/generated-rank6-poles-1'
OUT = WORK / 'build-attempt1'
SUPERVISOR = ROOT / 'docs/benchmarkfolder/experiments/qcd-topology-rescue-higher-precision-runner2/supervise.py'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify():
    assert sha(PACKAGE / 'review-index.json') == '38b30cda616b7ef9d589b8c83a0d4e1eaa8699397d567d97c8517695a80f72fb'
    old = sys.argv
    try:
        sys.argv = [str(PACKAGE / 'verify.py'), '--sources']
        runpy.run_path(str(PACKAGE / 'verify.py'), run_name='__main__')
    finally:
        sys.argv = old

def main():
    assert ROOT == Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
    verify()
    assert sha(SUPERVISOR) == 'ed25a9fc96e2faf2b545a9bf70a91b34315c01e534a6ebe68d7520ec2171aa42'
    supervisor = runpy.run_path(str(SUPERVISOR))
    plan = json.loads((PACKAGE / 'build-plan.json').read_text())
    tools = json.loads((PACKAGE / 'tools.json').read_text())
    for name in ('gcc', 'gfortran'):
        assert Path(shutil.which(name)).resolve() == Path(tools[name]['path']).resolve()
        assert sha(shutil.which(name)) == tools[name]['sha256']
    for model, spec in plan['models'].items():
        for argv in spec['commands']:
            assert argv[0] in ('gfortran', 'gcc')
            output = Path(argv[argv.index('-o') + 1])
            assert output.parent == WORK / model and not output.exists()
    OUT.mkdir(exist_ok=False)
    os.sched_setaffinity(0, {12})
    record = {'complete': False, 'scope': 'compile/link only; no executable, loader or reducer execution',
              'review_index_sha256': sha(PACKAGE / 'review-index.json'),
              'build_plan_sha256': sha(PACKAGE / 'build-plan.json'),
              'runner_sha256': sha(__file__), 'supervisor_sha256': sha(SUPERVISOR),
              'cpu': 12, 'jobs': 1, 'compile_cap_seconds': 240, 'commands': [],
              'compiler_environment': {key: os.environ.get(key) for key in
                  ('PATH', 'LD_LIBRARY_PATH', 'LD_PRELOAD', 'LD_AUDIT', 'LIBRARY_PATH',
                   'CPATH', 'C_INCLUDE_PATH', 'CPLUS_INCLUDE_PATH', 'FC', 'CC', 'FFLAGS',
                   'LDFLAGS', 'NIX_CFLAGS_COMPILE', 'NIX_LDFLAGS')}}
    start = time.monotonic()
    previous = {}
    def interrupt(signum, frame):
        raise InterruptedError('Compile interrupted by signal %d' % signum)
    def save():
        (OUT / 'build.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.signal(signum, interrupt)
        for model, spec in plan['models'].items():
            for number, argv in enumerate(spec['commands']):
                remaining = 240 - (time.monotonic() - start)
                if remaining <= 0:
                    raise TimeoutError('Shared 240-second compile cap exhausted')
                item = {'model': model, 'step': number, 'argv': argv, 'cwd': spec['cwd'],
                        'compiler_path': shutil.which(argv[0]), 'timeout_seconds': remaining,
                        'progress': {}, 'log': '%s-%02d.log' % (model, number)}
                record['commands'].append(item)
                save()
                with (OUT / item['log']).open('xb') as log:
                    item['exit_code'] = supervisor['run_plain_capped'](
                        argv, log, cwd=spec['cwd'], env=dict(os.environ),
                        progress=item['progress'], seconds=remaining)
                item['log_sha256'] = sha(OUT / item['log'])
                save()
                print(model, number, 'exit', item['exit_code'], flush=True)
                if item['exit_code']:
                    raise RuntimeError('Frozen compile/link command failed: %s %d' % (model, number))
        if time.monotonic() - start > 240:
            raise TimeoutError('Shared 240-second compile cap exceeded')
        verify()
        record['sources_and_dependencies_unchanged'] = True
        record['complete'] = True
    except BaseException as error:
        record['failure'] = repr(error)
        raise
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
        record['elapsed_seconds'] = time.monotonic() - start
        record['artifacts'] = {str(path.relative_to(WORK)): {'sha256': sha(path), 'size': path.stat().st_size}
                               for model in plan['models'] for path in sorted((WORK / model).rglob('*'))
                               if path.is_file()}
        save()

if __name__ == '__main__':
    main()
