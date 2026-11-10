#!/usr/bin/env python3
"""Validate prepared source identities and source-only fixture contracts."""
from pathlib import Path
import hashlib
import json
import re

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest=json.loads((HERE/'manifest.json').read_text())
    assert manifest['prepared_only']
    assert sha(HERE/'prepare.py')==manifest['generator_sha256']
    for name,expected in manifest['inputs'].items():
        assert sha(ROOT/name)==expected,name
    for name,expected in manifest['files'].items():
        assert sha(HERE/name)==expected,name
    assert sha(HERE/'expected.json')==manifest['expected_sha256']
    assert sha(HERE/'compiler-jobs.json')==manifest['compiler_jobs_sha256']
    expected=json.loads((HERE/'expected.json').read_text())
    jobs=json.loads((HERE/'compiler-jobs.json').read_text())
    assert len(jobs['jobs'])==5
    assert set(j['name'] for j in jobs['jobs'])==set(expected['cases'])
    case_count=sum(map(len,expected['cases'].values()))
    assert case_count==51
    for job in jobs['jobs']:
        assert job['working_directory']==str(ROOT)
        assert job['compile_argv'][:6]==['timeout','30s','taskset','-c','14','gfortran']
        assert job['run_argv'][:5]==['timeout','15s','taskset','-c','14']
        assert '-fno-fast-math' in job['compile_argv']
        assert not any('tirust' in arg.lower() or 'ninja' in arg.lower() for arg in job['compile_argv'] if arg.startswith('-l'))
        for arg in job['compile_argv']:
            if arg.endswith(('.f','.f90')):
                assert Path(arg).is_file(),arg
    for proof in manifest['proof']:
        variant=proof['variant']
        filename='wrapper' if variant=='repair' else 'aggregate'
        raw=(HERE/'sources'/variant/(filename+'.raw.f')).read_text()
        formatted=(HERE/'sources'/variant/(filename+'.f')).read_text()
        assert '%(' not in raw+formatted
        assert '@' not in (HERE/'sources'/variant/'driver.f90').read_text()
        for row in formatted.splitlines():
            if re.match(r'^\s*\d+',row):
                assert row[:5].strip().isdigit() and row[5]==' ',row
        if variant=='repair':
            assert raw.count('SUBROUTINE RX_PREPARE_MP_PS_CHECKED(')==1
            assert raw.count('SUBROUTINE RX_MP_IMPROVE_PS_POINT_PRECISION_CHECKED(')==1
        else:
            assert raw.count('ENTRY TX_OBSERVE_LEARNING(')==1
            assert raw.count('ENTRY LEG_OBSERVE_LEARNING(')==1
            for line in raw.splitlines():
                if 'FX_OBS_' in line and '=' in line:
                    assert line.strip().startswith('FX_OBS_'),line
            stripped=re.sub(r'(?m)^C RUNTIME-OBSERVER-BEGIN ([A-Z_]+)\n.*?^C RUNTIME-OBSERVER-END \1\n','',raw,flags=re.S)
            # Capture complete first checked aggregate including final END.
            first=re.search(r'(?ms)^      SUBROUTINE TX_SMATRIX(?:_SPLITORDERS)?\(P, FLAV_IDX, ANS\)\n.*?^      END\n',stripped)
            old=re.search(r'(?ms)^      SUBROUTINE LEG_SMATRIX(?:_SPLITORDERS)?\(P, FLAV_IDX, ANS\)\n.*?^      END\n',stripped)
            assert first and old
            checked_bytes=stripped[first.start():].encode()[:proof['checked_before_observer_bytes']]
            old_bytes=stripped[old.start():].encode()[:proof['legacy_before_observer_bytes']]
            assert hashlib.sha256(checked_bytes).hexdigest()==proof['checked_before_observer_sha256']
            assert hashlib.sha256(old_bytes).hexdigest()==proof['legacy_before_observer_sha256']
    print('PASS prepared source identities, exact observer projection, fixed-form labels, five jobs and 51 planned cases. No compiler/native execution.')


if __name__=='__main__':main()
