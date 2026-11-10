#!/usr/bin/env python3
"""Portable source reproduction checks; does not compile or run any reducer."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
from transform import loop_variants, mass_actual, qp_adapter_clone, qp_symbols, routines, transform_calls

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--working-root',type=Path,help='Also verify the isolated generated source files beneath the repository root.')
    args = parser.parse_args()
    index = json.loads((HERE/'evidence-index.json').read_text())
    for relative, expected in index['files'].items():
        assert digest(HERE/relative) == expected, relative
    record = json.loads((HERE/'prepared.json').read_text())
    for reference in record['reference_sources'].values():
        assert digest(HERE/reference['snapshot']) == reference['sha256']
    original = {name:(HERE/'sources/original'/name).read_text() for name in record['changed_generated_sources']}
    candidate = {name:(HERE/'sources/candidate'/name).read_text() for name in original}
    ct = original['CT_interface.f']; tir = original['TIRUST_interface.f']
    groups, helpers = loop_variants(ct)
    model_key = record['origin']+'/Source/MODEL/mp_coupl.inc'
    symbols = qp_symbols((HERE/record['reference_sources'][model_key]['snapshot']).read_text())
    calls,mapping = transform_calls(original['loop_CT_calls_1.f'],groups,symbols)
    divider = '\nC     PRIVATE SOURCE-ONLY QP MODEL MASS TRANSPORT PROTOTYPE\n'
    assert ct+divider+helpers+'\n'+qp_adapter_clone(ct,'ML5_0_MP_NINJARS_LOOP') == candidate['CT_interface.f']
    assert tir+divider+qp_adapter_clone(tir,'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC')+'\n'+qp_adapter_clone(tir,'ML5_0_MP_TIRUST_LOOP','ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC') == candidate['TIRUST_interface.f']
    assert calls == candidate['loop_CT_calls_1.f']
    assert mapping == json.loads((HERE/'group-mapping.json').read_text())['groups']
    assert sorted(row['loop_group'] for row in mapping) == list(range(1,201))
    for name in ('CT_interface.f','TIRUST_interface.f'):
        old = routines(original[name]); new = routines(candidate[name])
        assert all(new[symbol] == value for symbol,value in old.items())
    for name,text in candidate.items():
        assert all(not line or line[0] in 'Cc*!' or not line[72:].strip() for line in text.splitlines()), name
    # Reject unqualified expression reconstruction and absent/wrong-kind MP symbols.
    for actual,decl in [('DCMPLX(SQRT(M2))',symbols),('DCMPLX(UNKNOWN)',symbols),('DCMPLX(MDL_MT)',{})]:
        try: mass_actual(actual,decl,'MP__')
        except ValueError: pass
        else: raise AssertionError('Unsupported mass source admitted')
    assert qp_symbols('      REAL*8 MP__M\n') == {'MP__ZERO':'REAL*16 local zero'}
    cms = '.tirust-build/complex-stress-cuttools-qp-transport/'
    refs = record['reference_sources']
    cms_ct = (HERE/refs[cms+'SubProcesses/P0_epem_mupmum/CT_interface.f']['snapshot']).read_text()
    cms_calls = (HERE/refs[cms+'SubProcesses/P0_epem_mupmum/loop_CT_calls_1.f']['snapshot']).read_text()
    cms_symbols = qp_symbols((HERE/refs[cms+'Source/MODEL/mp_coupl.inc']['snapshot']).read_text())
    cms_groups,_ = loop_variants(cms_ct)
    _,cms_mapping = transform_calls(cms_calls,cms_groups,cms_symbols)
    complex_pairs = [pair for row in cms_mapping for pair in row['mass_pairs']
                     if pair['model_symbol'].startswith('CMASS_')]
    policy = json.loads((HERE/'model-policy.json').read_text())
    assert len(cms_mapping) == policy['cms_source_call_count'] == 26
    assert len(complex_pairs) == policy['cms_complex_mass_argument_count'] == 30
    assert dict(Counter(pair['mp_symbol'] for pair in complex_pairs)) == policy['cms_symbol_counts']
    assert all(pair['mp_declaration']=='COMPLEX*32' for pair in complex_pairs)
    if args.working_root:
        for name,identities in record['changed_generated_sources'].items():
            path=args.working_root/record['work']/'candidate/SubProcesses/P0_epem_ttxa'/name
            assert digest(path) == identities['candidate_sha256'], name
    print(json.dumps({'evidence_files':len(index['files']),'paired_groups':len(mapping),
                      'family_counts':dict(Counter(row['entrypoint'] for row in mapping)),
                      'private_loop_helpers':len(groups),'private_qp_adapters':3,
                      'original_ct_tir_routines_byte_identical':True,
                      'reproduced_source_bytes':True,'compiler_invocations':0,'native_reducer_calls':0},indent=2))


if __name__ == '__main__':main()
