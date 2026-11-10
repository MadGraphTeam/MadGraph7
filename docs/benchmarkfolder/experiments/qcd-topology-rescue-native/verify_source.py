#!/usr/bin/env python3
"""Pure source/manifest verification; no compiler, reducer, or output files."""
import ast
import json
import subprocess
import sys

from common import HERE, ARMS, require, pins, controller_pins, frozen_verify
from build import plan


def main():
    historical = pins()
    controller = controller_pins()
    entries = json.loads((HERE / 'source-index.json').read_text())['files']
    require(set(entries) == {'README.md', 'common.py', 'build.py', 'run.py', 'plans.json', 'verify_source.py',
                             'check_runner_source.py', 'source-checks.json'},
            'Unexpected controller source inventory.')
    for name in entries:
        if name.endswith('.py'):
            ast.parse((HERE / name).read_text(), filename=name)
    require({arm: plan(arm) for arm in ARMS} == json.loads((HERE / 'plans.json').read_text()),
            'Reviewed pure source build plans differ.')
    checks = subprocess.run([sys.executable, '-I', '-B', str(HERE / 'check_runner_source.py')],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    require(checks.returncode == 0, 'Pure runner comparison checks failed:\n' + checks.stdout)
    require(json.loads(checks.stdout) == json.loads((HERE / 'source-checks.json').read_text()),
            'Pure runner comparison results changed.')
    print(frozen_verify().strip())
    print(json.dumps({'status': 'PASS', 'source_files': len(entries),
                      'historical_protected_paths': len(historical),
                      'controller_pins': len(controller), 'arms': list(ARMS),
                      'positive_runner_checks': 4, 'rejected_runner_mutations': 36,
                      'compiler_executed': False, 'native_executed': False}, sort_keys=True))


if __name__ == '__main__':
    main()
