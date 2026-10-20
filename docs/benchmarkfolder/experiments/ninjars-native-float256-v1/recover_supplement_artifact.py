#!/usr/bin/env python3
"""Retain the successful relink, recover from its disclosed artifact collision."""
import sys
sys.dont_write_bytecode = True
import datetime
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
WORK = ROOT/'.benchmark-work/experiments/ninjars-native-float256-v1'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert Path.cwd() == ROOT
    destination = HERE/'supplement'
    failed = json.loads((destination/'build-result.json').read_text())
    assert failed['complete'] is False and failed['returncode'] == 0
    assert failed['failure'] == 'AssertionError()'
    prepared = json.loads((HERE/'prepared.json').read_text())
    original = json.loads((HERE/'build.json').read_text())
    original_exe = ROOT/original['executables'][0]['path']
    preserved = WORK/'supplement/original-executable-preserved'
    assert sha(preserved) == original['executables'][0]['sha256']
    rows = [json.loads(line) for line in (destination/'build.stdout').read_text().splitlines()]
    assert [r for r in rows if r.get('reason') == 'build-finished'] == [{'reason':'build-finished','success':True}]
    artifacts = [r for r in rows if r.get('reason') == 'compiler-artifact' and r.get('executable')]
    assert len(artifacts) == 1 and Path(artifacts[0]['executable']) == original_exe
    new_digest = sha(original_exe)
    assert new_digest != original['executables'][0]['sha256']
    new_exe = WORK/'supplement/native-supplement-test-binary'
    assert not new_exe.exists()
    shutil.copy2(original_exe,new_exe)
    assert sha(new_exe) == new_digest
    shutil.copy2(preserved,original_exe)
    assert sha(original_exe) == original['executables'][0]['sha256']
    for name,digest in prepared['study_files'].items(): assert sha(WORK/'bridge'/name) == digest,name
    for name,digest in failed['source_hashes'].items(): assert sha(WORK/'supplement/bridge'/name) == digest,name
    completed = dict(failed,complete=True,executed=False,controller_complete=False,
        build_controller_collision_assertion_failed=True,
        failure_retained='build-result.json',
        recovery_reason='Cargo used identical local-package executable disambiguator despite separate manifest path. Original exact executable was copied before final link, successful new artifact copied here, original restored exactly.',
        cargo_reported_artifact=str(original_exe.relative_to(ROOT)),
        executables=[{'path':str(new_exe.relative_to(ROOT)),'sha256':new_digest}],
        original_executable_restored=True,original_executable_sha256=sha(original_exe),
        recovery_script_sha256=sha(Path(__file__)),
        failed_controller_result_sha256=sha(destination/'build-result.json'),
        recovered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    with (destination/'build-completed.json').open('x') as stream:
        json.dump(completed,stream,indent=2)
        stream.write('\n')
    print(json.dumps({'complete':True,'new_executable_sha256':new_digest,
        'original_restored_sha256':sha(original_exe),'failed_controller_preserved':True}))


if __name__ == '__main__': main()
