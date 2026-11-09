#!/usr/bin/env python3
"""Run the production bridge's built unit-test artifact under the numeric lock."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[5]
EVIDENCE = Path(__file__).resolve().parent
assert Path.cwd().resolve() == ROOT


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


messages = [json.loads(line) for line in (EVIDENCE / 'native-test-build.jsonl').read_text().splitlines()]
artifacts = [m for m in messages if m.get('reason') == 'compiler-artifact'
             and m['target']['name'] == 'madloop_tirust'
             and m['profile']['test'] and m.get('executable')]
assert len(artifacts) == 1
executable = Path(artifacts[0]['executable'])
source = ROOT / 'vendor/tirust/src/lib.rs'
staged = ROOT / '.tirust-build/bridge-split-production/src/lib.rs'
assert source.read_bytes() == staged.read_bytes()
record = {'command': [str(executable), '--test-threads=1'],
          'source_sha256': digest(source), 'test_executable_sha256': digest(executable),
          'build_messages_sha256': digest(EVIDENCE / 'native-test-build.jsonl'),
          'runner_sha256': digest(Path(__file__)),
          'scope': 'All nine pre-existing native bridge tests plus two direction-workspace lifecycle tests.'}
with (ROOT / '.tirust-build/numerical-tests.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    with (EVIDENCE / 'native-tests.log').open('w') as log:
        result = subprocess.run(record['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
record['exit_code'] = result.returncode
record['log_sha256'] = digest(EVIDENCE / 'native-tests.log')
record['source_unchanged'] = digest(source) == record['source_sha256']
(EVIDENCE / 'native-tests.json').write_text(json.dumps(record, indent=2) + '\n')
print((EVIDENCE / 'native-tests.log').read_text())
assert record['source_unchanged']
raise SystemExit(result.returncode)
