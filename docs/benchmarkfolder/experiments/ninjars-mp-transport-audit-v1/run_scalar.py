#!/usr/bin/env python3
"""Held scalar red/green diagnostic; no NinjaRS context or amplitude is created."""
import hashlib
import json
from pathlib import Path
import subprocess
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert Path.cwd()==ROOT
    build=json.loads((HERE/'scalar-build.json').read_text())
    assert build['compiled'] and not build['executed']
    for name,record in build['executables'].items():
        assert sha(ROOT/record['path'])==record['sha256']
    for name,digest in build['source_sha256'].items():assert sha(HERE/name)==digest
    for name,digest in build['input_libraries'].items():assert sha(ROOT/name)==digest
    result={'complete':False,'engine_calls':0,'production_modified':False,
            'build_sha256':sha(HERE/'scalar-build.json'),'runner_sha256':sha(Path(__file__)),
            'runs':[]}
    try:
        for arm in ('old','proposed'):
            folder=HERE/('scalar-'+arm)
            folder.mkdir()
            records=folder/'output-records.txt'
            commands=[['taskset','-c','2',str(ROOT/build['executables']['scalar']['path']),arm,str(records)],
                      ['taskset','-c','2',str(ROOT/build['executables']['real16-read']['path']),str(records)]]
            for index,command in enumerate(commands,1):
                start=time.monotonic()
                with (folder/f'{index}.stdout').open('x') as out,(folder/f'{index}.stderr').open('x') as err:
                    run=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err,timeout=60)
                entry={'arm':arm,'step':index,'command':command,'returncode':run.returncode,
                       'elapsed_seconds':time.monotonic()-start,
                       'stdout_sha256':sha(folder/f'{index}.stdout'),'stderr_sha256':sha(folder/f'{index}.stderr')}
                result['runs'].append(entry)
                assert run.returncode==(1 if arm=='old' else 0),entry
            summary=(folder/'1.stdout').read_text().splitlines()[-1]
            expected=('SUMMARY finite_cases=28 value_fail=0 sign_fail=2 precision_fail=0 format_fail=2 branch_fail=2 reject_fail=0'
                      if arm=='old' else
                      'SUMMARY finite_cases=28 value_fail=0 sign_fail=0 precision_fail=0 format_fail=0 branch_fail=0 reject_fail=0')
            assert summary==expected,summary
            if arm=='proposed':
                assert (folder/'2.stdout').read_text().splitlines()[-1].split()==['COMPLETE','14']
            else:
                assert 'MISMATCH 2' in (folder/'2.stdout').read_text()
        result['complete']=True
    except BaseException as exc:
        result['failure']=repr(exc)
        raise
    finally:
        with (HERE/'scalar-result.json').open('x') as stream:
            json.dump(result,stream,indent=2);stream.write('\n')

if __name__=='__main__':main()
