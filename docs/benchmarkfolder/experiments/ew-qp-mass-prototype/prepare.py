#!/usr/bin/env python3
"""Create isolated generated-source prototype, with no build/native operation."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil
from collections import Counter

from transform import (loop_variants, mass_actual, qp_adapter_clone, qp_symbols,
                       routines, transform_calls)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
ORIGIN = ROOT/'.tirust-build/ew-threshold-adjudication'
WORK = ROOT/'.tirust-build/ew-qp-mass-prototype'
PROC = Path('SubProcesses/P0_epem_ttxa')
PROPOSAL = ROOT/'docs/benchmarkfolder/experiments/ew-qp-input-policy'
PREFIX = 'ML5_0_'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n')


def source_inventory(base):
    return {str(p.relative_to(base)): digest(p) for p in sorted(base.rglob('*'))
            if p.is_file() and p.suffix.lower() in ('.f','.f90','.inc','.dat')}


def main():
    assert Path.cwd().resolve() == ROOT
    assert not WORK.exists(), 'Refusing to overwrite a prepared prototype'
    assert not (HERE/'prepared.json').exists()
    assert digest(PROPOSAL/'evidence-index.json') == '2f90c77b68edf80df9a9e54da2fff6f9afba570ae24b474f2f049cc2d55e9ba1'
    prior = json.loads((PROPOSAL/'source-routing.json').read_text())
    identities = {item['path']: item['sha256'] for item in prior['sources']}
    identities.update(json.loads((PROPOSAL/'summary.json').read_text())['source_sha256'])
    traced = json.loads((ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication/prepared.json').read_text())
    identities[str((ORIGIN/PROC/'TIRUST_interface.f').relative_to(ROOT))] = traced['patched_source_hashes']['TIRUST_interface.f']
    for name in ('CT_interface.f','TIRUST_interface.f'):
        path = ORIGIN/PROC/name
        assert digest(path) == identities[str(path.relative_to(ROOT))], name
    originals = {name: (ORIGIN/PROC/name).read_text()
                 for name in ('CT_interface.f','TIRUST_interface.f','loop_CT_calls_1.f')}
    ct = originals['CT_interface.f']
    tir = originals['TIRUST_interface.f']
    groups, helpers = loop_variants(ct)
    assert len(groups) == 7
    declarations = qp_symbols((ORIGIN/'Source/MODEL/mp_coupl.inc').read_text())
    calls, mapping = transform_calls(originals['loop_CT_calls_1.f'], groups, declarations)
    assert len(mapping) == 200
    assert sorted(item['loop_group'] for item in mapping) == list(range(1,201))
    assert mapping[4]['ordered_mp_square_symbols'] == ['MP__MDL_MT','MP__ZERO','MP__MDL_MT','MP__MDL_MT']
    nrs_name = PREFIX+'MP_NINJARS_LOOP'
    diag_name = PREFIX+'MP_TIRUST_TENSOR_DIAGNOSTIC'
    tir_name = PREFIX+'MP_TIRUST_LOOP'
    divider = '\nC     PRIVATE SOURCE-ONLY QP MODEL MASS TRANSPORT PROTOTYPE\n'
    candidate = {
        'CT_interface.f': ct+divider+helpers+'\n'+qp_adapter_clone(ct,nrs_name),
        'TIRUST_interface.f': tir+divider+qp_adapter_clone(tir,diag_name)+'\n'+qp_adapter_clone(tir,tir_name,diag_name),
        'loop_CT_calls_1.f': calls}
    # All original routines retain exact bytes and signatures. New names make
    # the changed QP formal kind explicit to direct diagnostics and callers.
    for name in ('CT_interface.f','TIRUST_interface.f'):
        assert candidate[name].startswith(originals[name])
        old = routines(originals[name]); new = routines(candidate[name])
        for symbol, routine in old.items():
            assert new[symbol] == routine, symbol
    for row in mapping:
        actuals = []
        new = iter(row['new_actuals'])
        for formal, old in zip(groups[row['entrypoint']]['args'],row['original_actuals']):
            assert next(new) == old
            actuals.append(old)
            if formal.startswith('M') and formal[1:].isdigit(): next(new)
        assert list(new) == []
        assert actuals == row['original_actuals']
    # An actual preserved CMS export exercises the same generic symbol policy,
    # without compiling it or claiming CMS amplitude qualification.
    cms = ROOT/'.tirust-build/complex-stress-cuttools-qp-transport'
    cms_proc = cms/'SubProcesses/P0_epem_mupmum'
    cms_groups, _ = loop_variants((cms_proc/'CT_interface.f').read_text())
    _, cms_mapping = transform_calls((cms_proc/'loop_CT_calls_1.f').read_text(),cms_groups,
                                     qp_symbols((cms/'Source/MODEL/mp_coupl.inc').read_text()))
    complex_pairs = [pair for row in cms_mapping for pair in row['mass_pairs']
                     if pair['model_symbol'].startswith('CMASS_')]
    assert complex_pairs and all(pair['mp_declaration']=='COMPLEX*32' for pair in complex_pairs)
    derived = ROOT/'.benchmark-work/tirust/outputs/candidate/qcd_n1_uux_z/Source/MODEL'
    derived_symbols = qp_symbols((derived/'mp_coupl.inc').read_text())
    derived_actual = mass_actual('DCMPLX(MDL_MW)',derived_symbols,'MP__')
    derived_text = (derived/'mp_intparam_definition.inc').read_text()
    assert 'MP__MDL_MW =' in derived_text or 'MP__MDL_MW=' in derived_text
    before = source_inventory(ORIGIN)
    WORK.mkdir()
    for variant in ('original','candidate'):
        shutil.copytree(ORIGIN,WORK/variant,symlinks=True,
                        ignore=shutil.ignore_patterns('worker-raw','worker-protocol.fifo','__pycache__'))
    source_dir = HERE/'sources'
    (source_dir/'original').mkdir(parents=True)
    (source_dir/'candidate').mkdir()
    for name, text in candidate.items():
        target = WORK/'candidate'/PROC/name
        target.unlink()
        target.write_text(text)
        (source_dir/'original'/name).write_text(originals[name])
        (source_dir/'candidate'/name).write_text(text)
        (HERE/(name+'.patch')).write_text(''.join(difflib.unified_diff(
            originals[name].splitlines(keepends=True), text.splitlines(keepends=True),
            fromfile='original/'+name,tofile='candidate/'+name)))
    # A copied historical executable is not a built prototype.
    old_check = WORK/'candidate'/PROC/'check'
    old_check.rename(old_check.with_name('check.historical-unmodified'))
    assert source_inventory(ORIGIN) == before
    write_json(HERE/'group-mapping.json',{'scope':'All200 generated groups, no numerical execution','groups':mapping})
    write_json(HERE/'model-policy.json',{
        'mp_prefix':'MP__','policy':'Prefix existing model-selected mass symbol; require a genuine REAL*16/COMPLEX*32 declaration; do not reparse card or infer from squared mass.',
        'fixture_symbols':dict(Counter(pair['mp_symbol'] for row in mapping for pair in row['mass_pairs'])),
        'cms_source_call_count':len(cms_mapping),'cms_complex_mass_argument_count':len(complex_pairs),
        'cms_symbol_counts':dict(Counter(pair['mp_symbol'] for pair in complex_pairs)),
        'cms_examples':complex_pairs[:8],
        'derived_real_example':{'input':'DCMPLX(MDL_MW)','new_actual':derived_actual[0],
            'declaration':derived_symbols[derived_actual[2]],'expression_source':str((derived/'mp_intparam_definition.inc').relative_to(ROOT)),
            'expression_source_sha256':digest(derived/'mp_intparam_definition.inc')},
        'qualification':'Source-routing cases only. Real-derived and CMS numerical coverage remains unqualified.'})
    refs = [ORIGIN/'Source/MODEL'/name for name in ('mp_coupl.inc','coupl.inc','param_read.inc','lha_read.f','mp_intparam_definition.inc','actualize_mp_ext_params.inc','couplings.f')]
    refs += [ORIGIN/PROC/name for name in ('improve_ps.f','mp_compute_loop_coefs.f','loop_matrix.f')]
    refs += [cms/'Source/MODEL/mp_coupl.inc',cms/'Source/MODEL/mp_intparam_definition.inc',cms_proc/'CT_interface.f',cms_proc/'loop_CT_calls_1.f']
    refs += [derived/'mp_coupl.inc',derived/'mp_intparam_definition.inc']
    for i,path in enumerate(refs):
        shutil.copyfile(path,source_dir/('reference-%02d-%s'%(i,path.name)))
    write_json(HERE/'prepared.json',{
        'state':'source_only_uncompiled','compiler_invocations':0,'native_reducer_calls':0,
        'origin':str(ORIGIN.relative_to(ROOT)),'work':str(WORK.relative_to(ROOT)),
        'proposal_index_sha256':digest(PROPOSAL/'evidence-index.json'),
        'original_source_inventory':before,
        'changed_generated_sources':{name:{'original_sha256':digest(source_dir/'original'/name),
            'candidate_sha256':digest(source_dir/'candidate'/name)} for name in candidate},
        'reference_sources':{str(path.relative_to(ROOT)):{'sha256':digest(path),'snapshot':'sources/reference-%02d-%s'%(i,path.name)} for i,path in enumerate(refs)},
        'all_original_ct_tir_routines_byte_identical':True,'paired_group_count':200,
        'private_helper_count':7,'private_qp_adapter_count':3,
        'dp_delegates_before_qp_squaring':True,'scope_backends':[8,9],
        'native_identities_unchanged':{'tirust_sha256':digest(ORIGIN/'private-libraries/libmadloop_tirust.so')},
        'future_build_required_sources':list(candidate),'candidate_historical_executable_renamed':True,
        'scope_exclusions':['No production generator/template edits','No new card reader or model expression','No scale/repair/on-shell threshold/MPFR guard changes','No CutTools/Ninja/COLLIER QP ABI claim','No CMS or derived-mass numerical qualification','Mixed rescue chains unqualified: private guard examines requested LIBINDEX before actual I_LIB selection','Fixture uses optimized interference reduction only; amplitude-reduction generator qualification remains open']})
    print('Prepared source-only200-group prototype at '+str(WORK))


if __name__ == '__main__':
    main()
