#!/usr/bin/env python3
"""Read-only validation of all retained development NLO attempts."""
import argparse,hashlib,json,re,sys
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=HERE.parent
ROOT=next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def verify(live=False):
    seal=read(HERE/'nlo-validation.json')
    assert digest(seal['files'])==seal['files_sha256']
    actual={str(p.relative_to(BASE)) for name in ('dev-nlo-v1','dev-nlo-v2','dev-nlo-v3') for p in (BASE/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p!=HERE/'nlo-validation.json'}
    assert actual==set(seal['files'])
    for name,expected in seal['files'].items():assert sha(BASE/name)==expected,name
    for name,error in (('dev-nlo-v1','Broken generated-source link'),('dev-nlo-v2','IndexError')):
        failed=read(BASE/name/'result.json')
        assert not failed['complete'] and error in failed['error']
        assert failed['generation']['returncode']==0 and failed['selected_calls']==9
        assert 'compile_and_controls' not in failed
    record=read(HERE/'result.json');assert record['complete'] and not record['acceptance_eligible']
    assert record['backend']==6 and record['selected_calls']==9
    for name in ('generation','compile_and_controls'):
        assert record[name]['returncode']==0
    assert record['compile_and_controls']['log_sha256']==sha(HERE/'compile-and-controls.log')
    assert 'Poles successfully cancel for 20 points over 20' in (HERE/'compile-and-controls.log').read_text()
    assert 'Fraction of failures: 0.00' in (HERE/'P0_uux_zg-test_ME.log').read_text()
    for batch in record['factored_batches'].values():
        assert batch['empty_mp_prefix'] and batch['fresh_flag_before_every_call'] and batch['ieee_import_and_declaration']
        retained=HERE/'generated-factor-sources/V0_uux_zg'
        assert sha(retained/'mp_coef_construction_1.f')==batch['sha256']
        assert sha(retained/'polynomial.f')==batch['polynomial_sha256']
    for name in ('generated-sources.json','compiled-sources.json'):
        manifest=read(HERE/name);assert digest(manifest['files'])==manifest['files_sha256']
        assert all(x['reason'] for x in manifest['missing_template_links'].values())
        if name=='compiled-sources.json':assert all(not n.endswith('run_card.inc') for n in manifest['missing_template_links'])
    failed=read(HERE/'forced-qp/result.json')
    assert not failed['complete'] and failed['original_cards_restored'] and failed['source_and_executable_unchanged']
    assert failed['runs']['6']['returncode']==-15 and '8' not in failed['runs']
    diagnostic=read(HERE/'forced-qp-driver-v2/result.json')
    assert diagnostic['complete'] and not diagnostic['acceptance_eligible'] and diagnostic['parent_source_unchanged']
    assert set(diagnostic['changed_sources'])=={'SubProcesses/check_poles.f','SubProcesses/P0_uux_zg/check_poles.f'}
    for backend in (6,8):
        result=diagnostic['runs'][str(backend)]
        assert result['returncode']==0 and result['return_codes']==[140]+[120]*20
        assert result['counted_qp_return_codes']==[120]*20
        log=HERE/f'forced-qp-driver-v2/backend-{backend}.log'
        assert sha(log)==result['log_sha256']
        assert re.search(r'NUMBER OF POINTS PASSING THE CHECK\s+20\b',log.read_text())
        assert re.search(r'NUMBER OF POINTS FAILING THE CHECK\s+0\b',log.read_text())
    if live:
        sys.path[:0]=[str(HERE),str(ROOT/'tools'),str(ROOT)]
        import run
        import ninjars_benchmark_policy_v2 as policy
        assert run.nlo_source_manifest('after-build')==read(HERE/'compiled-sources.json')
        assert policy.library_hashes(run.OUTPUT)==read(HERE/'compiled-inputs.json')['libraries']
        assert sha(run.OUTPUT/'SubProcesses/P0_uux_zg/check_poles')==failed['executable_sha256']
        clone=run.WORK/'forced-qp-driver-v2'
        assert sha(clone/'SubProcesses/P0_uux_zg/check_poles')==diagnostic['executable_sha256']
    print(json.dumps({'passed':True,'files':len(seal['files']),'live':live,'normal_poles':20,'forced_qp_poles_per_backend':20,'backends':[6,8],'acceptance_eligible':False},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--live',action='store_true')
    verify(parser.parse_args().live)
