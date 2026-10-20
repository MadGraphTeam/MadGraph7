#!/usr/bin/env python3
"""Read-only rehash/replay of the separate binary64 experimental evidence."""
import sys
sys.dont_write_bytecode = True
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import bind_sources
import oracle

HERE=Path(__file__).resolve().parent
BASE=HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(recompute=False):
    proof_path=HERE/'source-parity.json'
    proof=json.loads(proof_path.read_text())
    # Exact proof histograms have integer keys in memory and JSON string keys.
    assert json.loads(json.dumps(bind_sources.bind()))==proof
    summary=dict(schema='binary64-factor-independent-validation-v1',complete=False,
                 passed=False,production_change=False,performance_claim=False,
                 source_parity_sha256=sha(proof_path),verifier_sha256=sha(Path(__file__)),
                 exact_recomputed=recompute,streams=[],safety=[],zero_work_snan=[],retained_failures=[])
    for name in ('oracle-nlo-rank2-v1','oracle-nlo-rank34-v1','oracle-sa-rank234-v1','oracle-endpoints-all-v1'):
        folder=BASE/name
        run=json.loads((folder/'run.json').read_text())
        assert run['complete']
        for key,file in [('factor_sha256','factor.py'),('oracle_driver_sha256','oracle-driver.py'),
                         ('runner_sha256','run_oracle.py'),('support_sha256','support.py'),
                         ('source_parity_sha256','source-parity.json')]:
            assert run[key]==sha(folder/file), (name,key)
        expected_count={'oracle-nlo-rank2-v1':2,'oracle-nlo-rank34-v1':4,
                        'oracle-sa-rank234-v1':6,'oracle-endpoints-all-v1':12}[name]
        assert len(run['results'])==expected_count
        assert len({r['key'] for r in run['results']})==expected_count
        for record in run['results']:
            work=folder/record['key']
            assert record['returncode']==0 and record['build']['returncode']==0
            for file,value in record['sources'].items():
                assert sha(work/file)==value,(work,file)
            assert sha(work/'dump.txt.gz')==record['gzip_sha256']
            raw=gzip.decompress((work/'dump.txt.gz').read_bytes())
            assert hashlib.sha256(raw).hexdigest()==record['raw_dump_sha256']
            result=json.loads((work/'exact-oracle.json').read_text())
            assert result['complete'] and result['passed']
            assert result['dump_sha256']==sha(work/'dump.txt.gz')
            assert result['proof_sha256']==sha(proof_path)
            assert result['oracle_sha256']==sha(HERE/'oracle.py')
            assert result['vertex']==record['vertex'] and len(result['cases'])==10
            assert [r['case'] for r in result['cases']]==list(range(1,11))
            assert all(r['rank']==record['rank'] and not r['terminal'] and r['passed'] for r in result['cases'])
            cases=None
            if recompute or run.get('endpoint_fixtures'):
                cases=oracle.read_cases(work/'dump.txt.gz')
            if recompute:
                replay=[oracle.check_case(case,proof['vertices'][record['vertex']]) for case in cases]
                assert json.loads(json.dumps(replay))==result['cases'],work
            endpoint_check=False
            if run.get('endpoint_fixtures'):
                assert sha(folder/'endpoint_driver.py')==sha(BASE/'endpoint_driver.py')
                for number,exponent in ((4,1223),(5,823)):
                    case=cases[number-1]
                    for tag in ('A','B'):
                        active=[v for value in case['bits'][tag].values() for v in value if v&((1<<63)-1)]
                        assert active and all((v>>52)&2047==exponent and v&((1<<52)-1)==(1<<52)-1 for v in active)
                mixed=[v for value in cases[5]['bits']['A'].values() for v in value]
                assert {(v>>52)&2047 for v in mixed}=={823,1223}
                assert all(v&((1<<52)-1)==0 for v in mixed)
                endpoint_check=True
            summary['streams'].append(dict(path=str(work.relative_to(BASE)),
                run_sha256=sha(folder/'run.json'),result_sha256=sha(work/'exact-oracle.json'),
                fixtures=len(result['cases']),outputs=sum(c['output_coefficients'] for c in result['cases']),
                fast=sum(c['fast'] for c in result['cases']),fallback=sum(c['fallback'] for c in result['cases']),
                severe_cancellation=sum(c['severely_cancelled_coefficients'] for c in result['cases']),
                maximum_bound_fraction=max(c['max_fraction_of_forward_bound']['NEW'] for c in result['cases']),
                maximum_normalized_error=max(c['max_error_over_epsilon_product_scale']['NEW'] for c in result['cases']),
                exact_endpoint_bits_checked=endpoint_check))
    for name,combined in [('safety-all-ranks-flags-v1',False),('safety-combined-sticky-traps-v2',True)]:
        folder=BASE/name
        run=json.loads((folder/'results.json').read_text())
        assert run['complete'] and run['passed'] and len(run['results'])==12
        assert run['runner_sha256']==sha(folder/'check_safety.py')
        assert run['factor_sha256']==sha(folder/'factor.py')
        assert len({r['key'] for r in run['results']})==12
        for record in run['results']:
            work=folder/record['key']
            for file,value in record['sources'].items():
                assert sha(work/file)==value
            tests=record['tests']
            if not combined:
                for key,count in [('guards',20),('sticky',32)]:
                    assert tests[key]['returncode']==0
                    assert (work/(key+'.log')).read_text()==tests[key]['output']
                    rows=[row.split() for row in tests[key]['output'].splitlines()]
                    assert len(rows)==count and [int(row[1]) for row in rows]==list(range(1,count+1))
            subprocesses=tests['trap_subprocesses']
            assert len(subprocesses)==(160 if combined else 20)
            ids=set()
            for trap in subprocesses:
                key=(trap['backend'],trap['scenario'],trap['flag'],trap.get('initial_mask',0))
                assert key not in ids
                ids.add(key)
                assert trap['returncode']==(0 if trap['scenario']==0 else -8)
                if combined:
                    assert trap['scenario']==0 and not trap['initial_mask']&(1<<(trap['flag']-1))
                filename=f"trap-{trap['backend']}-{trap['scenario']}-{trap['flag']}"
                if 'initial_mask' in trap:
                    filename+=f"-{trap['initial_mask']}"
                stdout=(work/(filename+'.stdout')).read_text().splitlines()
                if combined:
                    observed=stdout.pop(0).split()
                    assert observed[0]=='PRECALL'
                    assert observed[1:]==trap['measured_precall_state']
                    assert len(observed)==11 and set(observed[1:])<=set(('T','F'))
                    assert observed[1:6]==['T' if i==trap['flag'] else 'F' for i in range(1,6)]
                    assert observed[6:]==['T' if trap['initial_mask']&(1<<i) else 'F' for i in range(5)]
                assert stdout[0].split()==['ENTER',str(trap['backend']),str(trap['scenario']),str(trap['flag'])]
                if trap['scenario']==0:
                    assert stdout[-1].split()==['PASS',str(trap['backend']),str(trap['scenario']),str(trap['flag'])]
            summary['safety'].append(dict(path=str(work.relative_to(BASE)),run_sha256=sha(folder/'results.json'),
                guards=0 if combined else 20,sticky_masks=0 if combined else 32,
                compatible_sticky_trap_pairs=80 if combined else 0,trap_pairs=0 if combined else 10))
    folder=BASE/'safety-zero-work-snan-v1'
    run=json.loads((folder/'results.json').read_text())
    assert run['complete'] and run['passed'] and len(run['results'])==12
    assert run['runner_sha256']==sha(folder/'check_zero_work_snan.py')
    assert len({r['key'] for r in run['results']})==12
    for record in run['results']:
        work=folder/record['key']
        assert record['build']['returncode']==record['run']['returncode']==0
        assert record['run']['output']==(work/'run.log').read_text()
        assert [row.split() for row in record['run']['output'].splitlines()]==[['PASS','1','0','1'],['PASS','2','0','1']]
        for file,value in record['sources'].items():
            assert sha(work/file)==value
        summary['zero_work_snan'].append(dict(path=str(work.relative_to(BASE)),
            run_sha256=sha(folder/'results.json'),true_entry_no_new_invalid_cases=2))
    prior_path=HERE/'verification-v2.json'
    prior=json.loads(prior_path.read_text())
    assert prior['passed'] and prior['exact_recomputed']
    assert prior['verifier_sha256']==sha(HERE/'verification-v2-source.py')
    assert prior['streams']==summary['streams']
    summary['retained_full_exact_replay_sha256']=sha(prior_path)
    failed=BASE/'safety-initial-v1'
    failure=json.loads((failed/'results.json').read_text())
    assert not failure['complete'] and 'Result of CMPLX is NaN' in failure['failure']
    summary['retained_failures'].append(dict(path=str(failed.relative_to(BASE)),
        results_sha256=sha(failed/'results.json'),classification='test-driver compile rejection of constant sNaN CMPLX; raw TRANSFER correction only'))
    failed=BASE/'safety-combined-sticky-traps-v1'
    failure=json.loads((failed/'results.json').read_text())
    assert not failure['complete'] and "'returncode': 5" in failure['failure']
    diagnosis=BASE/'combined-sticky-diagnosis-v1/results.json'
    observed=json.loads(diagnosis.read_text())
    assert len(observed)==2 and all(r['returncode']==5 for r in observed)
    for result in observed:
        rows=[row.split() for row in result['stdout'].splitlines()]
        assert rows[0]==['AFTER_SET_FLAGS','F','T','F','F','F','F','T','F','F','F']
        assert rows[1]==['AFTER_ENABLE','F','T','F','F','F','F','F','F','F','F']
    summary['retained_failures'].append(dict(path=str(failed.relative_to(BASE)),
        results_sha256=sha(failed/'results.json'),diagnosis_sha256=sha(diagnosis),
        classification='GNU halting-mode setup cleared requested sticky flags before both updater calls; corrected setup separately measures actual state'))
    summary.update(complete=True,passed=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--recompute',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.recompute),indent=2,sort_keys=True))
