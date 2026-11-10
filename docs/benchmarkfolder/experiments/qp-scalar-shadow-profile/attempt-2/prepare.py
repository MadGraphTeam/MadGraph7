#!/usr/bin/env python3
"""Prepare isolated source and exact saved-token inputs; never compile/run native."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / '.tirust-build/qp-scalar-shadow-profile-attempt-2'
UPSTREAM = ROOT / '.tirust-build/pole-development'
ONELOOP = ROOT / '.tirust-build/clean-install/cargo-home/git/checkouts/oneloopmaster-1e046f4c0b16c2f9/bd703d0'
TRACE = ROOT / 'docs/benchmarkfolder/experiments/ew-threshold-adjudication'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new)

def transform_exec(code):
    # These hooks contain no alternative result path. Both original kernels
    # still execute once with exactly their original input/output slices.
    for field, output in [('pole_masters', '&mut self.native_output[1..3]'),
                          ('native', '&mut self.native_output')]:
        expectation = 'pole master prepared' if field == 'pole_masters' else 'full master prepared'
        old = f'''                            self.{field}[family].as_mut().expect("{expectation}")
                                .evaluate(&input[..arity + 1], {output})?;'''
        new = f'''                            let shadow = crate::qp_scalar_shadow::scalar_begin(family, POLES, &input[..arity + 1]);
                            let scalar_result = self.{field}[family].as_mut().expect("{expectation}")
                                .evaluate(&input[..arity + 1], {output});
                            crate::qp_scalar_shadow::scalar_end(shadow, scalar_result.is_ok(), &self.native_output);
                            scalar_result?;'''
        code = replace(code, old, new)
    return code

def transform_bridge(code):
    code = replace(code, 'pub const ABI_VERSION: u32 = 1;',
                   'use iregi_rust::qp_scalar_shadow as shadow;\npub const ABI_VERSION: u32 = 1;')
    start = code.index('pub unsafe extern "C" fn ml_tirust_evaluate_mp(')
    end = code.index('\n}\n', start) + 3
    part = code[start:end]
    part = replace(part, '    guard(|| {', '''    shadow::parent_begin(handle as usize, n, rank, poles, finite, check, cache);
    let final_status = guard(|| {''')
    assert part.endswith('    })\n}\n')
    part = part[:-len('    })\n}\n')] + '''    });
    shadow::parent_end(final_status);
    final_status
}
'''
    code = code[:start] + part + code[end:]
    code = replace(code,
        '    fn mp_records(&mut self, key: Key, records: [&[u8]; 4], mode: Mode) -> Result<(i32, Vec<u8>)> {',
        '''    fn mp_records(&mut self, key: Key, records: [&[u8]; 4], mode: Mode) -> Result<(i32, Vec<u8>)> {
        #[allow(unused_mut)]
        let mut pole_bytes = 0;
        #[cfg(feature = "poles-only")]
        { pole_bytes = self.pole_store.statistics().bytes; }
        shadow::store_state(self.budget, self.qp_cache.bytes, self.store.statistics().bytes, pole_bytes);''')
    code = replace(code, '        state.qp_cache.clear();',
                   '        state.qp_cache.clear();\n        shadow::event(handle as usize, "clear");')
    code = replace(code, '            state.budget = bytes;',
                   '            state.budget = bytes;\n            shadow::event(handle as usize, "budget_changed");')
    code = replace(code, '                drop(Box::from_raw(handle));',
                   '                drop(Box::from_raw(handle));\n                shadow::event(handle as usize, "destroy");')
    code = replace(code, '                self.qp_cache.hits += 1;',
                   '                self.qp_cache.hits += 1;\n                shadow::parent_note("whole_tensor_cache_hit");')
    # Temporary ABI prefix stays inside the production export namespace.
    code += '''
// Private observation ABI, not part of the installed production interface.
// Exact C ABI from the pinned MPFR header; keep this outside the safe core.
unsafe extern "C" { fn mpfr_flags_save() -> std::ffi::c_uint; }
fn read_shadow_mpfr_flags() -> u32 { unsafe { mpfr_flags_save() } }
#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_shadow_control(enabled: i32) -> i32 {
    if !(0..=1).contains(&enabled) { return ERROR; }
    if !shadow::install_mpfr_reader(read_shadow_mpfr_flags) { return ERROR; }
    shadow::control(enabled == 1); SUCCESS
}
#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_shadow_label(point: i32, group: i32, synthetic: i32) {
    shadow::label(point, group, synthetic != 0);
}
#[unsafe(no_mangle)]
pub extern "C" fn ml_tirust_shadow_mpfr_flags() -> u32 { read_shadow_mpfr_flags() }
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ml_tirust_shadow_json(out: *mut u8, capacity: usize) -> usize {
    let data = shadow::snapshot();
    if !out.is_null() && capacity >= data.len() + 1 {
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), out, data.len()); out.add(data.len()).write(0); }
    }
    data.len() + 1
}
'''
    return code

def write_patch(original, altered, name):
    (HERE / name).write_text(''.join(difflib.unified_diff(
        original.splitlines(keepends=True), altered.splitlines(keepends=True),
        fromfile='original/' + name.removesuffix('.patch'),
        tofile='isolated/' + name.removesuffix('.patch'))))

def inputs():
    spec = importlib.util.spec_from_file_location('frozen_ew_replay_prepare', TRACE/'tensor-replay/prepare.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    trace = json.loads((TRACE/'trace.json').read_text())
    assert sha(TRACE/'trace.json') == '03fbfd3795a468661162c0e574a7de6878065efbab79ca9d0f80e743f29e787f'
    records = [helper.extract(trace[str(point)], group) for point in (1,2) for group in range(1,13)]
    lines = ['QP_SCALAR_SHADOW_V1', str(len(records))]
    for r in records:
        lines.append(' '.join(str(r[k]) for k in ('point','group','n','rank','historical_status')))
        lines.append(r['scale'])
        lines.extend(' '.join(row[1:]) for row in r['shifts'])
        lines.extend(' '.join(row[1:]) for row in r['masses'])
        inv = {(i,j):v for i,j,v in r['invariants']}
        lines.extend(' '.join(inv[i,j] for j in range(1,r['n']+1)) for i in range(1,r['n']+1))
    (HERE/'selected-inputs.dat').write_text('\n'.join(lines)+'\n')
    (HERE/'selected-inputs.json').write_text(json.dumps({
        'scope':'24 saved physical parents in original per-point group order; no new point generation',
        'trace_sha256':sha(TRACE/'trace.json'),
        'extraction_helper_sha256':sha(TRACE/'tensor-replay/prepare.py'),
        'input_stream_sha256':sha(HERE/'selected-inputs.dat'),
        'transport':'Read decimal tokens with GNU Fortran REAL16, then frozen mtirust ES64.39E4 packer. No Python float conversion.',
        'original_direction':'CTMODE4 only; synthetic check controls are explicitly separate',
        'records':records},indent=2)+'\n')

def main():
    assert Path.cwd().resolve() == ROOT
    if WORK.exists():
        raise SystemExit('Refusing to overwrite isolated preparation')
    exec_source = UPSTREAM/'crates/iregi-rust/src/direct/exec.rs'
    assert sha(exec_source) == '255b250f6997cfd08f017a9962ac79e6a4a87421f1f2d9f7ddb6a8bc1129eace'
    bridge_source = ROOT/'vendor/tirust/src/lib.rs'
    assert sha(bridge_source) == '455529b929b725e8fe2d7337ab4a41fa3de9dee1b3a953afe6dc032cf99e5a4c'
    tree = WORK/'tirust'
    tree.mkdir(parents=True)
    for name in ('crates/iregi-rust','crates/iregi-masters','madloop-pole-kernels'):
        shutil.copytree(UPSTREAM/name, tree/name, ignore=shutil.ignore_patterns('target','__pycache__','.git'))
    for name in ('Cargo.toml','Cargo.lock','LICENSE'):
        shutil.copy2(UPSTREAM/name, tree/name)
    assert sha(ONELOOP/'src/native/mod.rs') == 'a666d5acb3fe577de1b33f8907b4cf16a845c319d46de65da0897f44c9e13780'
    shutil.copytree(ONELOOP,WORK/'oneloop',ignore=shutil.ignore_patterns('target','__pycache__','.git'))
    cargo = (tree/'Cargo.toml').read_text()
    cargo = re.sub(r'^members = .*$', 'members = ["crates/iregi-rust", "crates/iregi-masters"]', cargo, flags=re.M)
    cargo = re.sub(r'^default-members = .*$', 'default-members = ["crates/iregi-rust"]', cargo, flags=re.M)
    (tree/'Cargo.toml').write_text(cargo)
    original = exec_source.read_text(); changed = transform_exec(original)
    (tree/'crates/iregi-rust/src/direct/exec.rs').write_text(changed)
    write_patch(original, changed, 'exec.rs.patch')
    core_lib = tree/'crates/iregi-rust/src/lib.rs'
    original = core_lib.read_text(); changed = original + '\n#[doc(hidden)]\npub mod qp_scalar_shadow;\n'
    core_lib.write_text(changed); write_patch(original, changed, 'iregi-lib.rs.patch')
    shutil.copy2(HERE/'qp_scalar_shadow.rs', tree/'crates/iregi-rust/src/qp_scalar_shadow.rs')
    bridge = WORK/'bridge'; (bridge/'src').mkdir(parents=True)
    original = bridge_source.read_text(); changed = transform_bridge(original)
    (bridge/'src/lib.rs').write_text(changed); write_patch(original, changed, 'bridge-lib.rs.patch')
    for name in ('Cargo.toml','mtirust.f90','tirust.h'):
        shutil.copy2(ROOT/'vendor/tirust'/name, bridge/name)
    shutil.copy2(ROOT/'.tirust-build/tirust-split-install/Cargo.lock', bridge/'Cargo.lock')
    cargo=(bridge/'Cargo.toml').read_text()
    cargo=re.sub(r'^iregi-rust = .*$', 'iregi-rust = { path = "../tirust/crates/iregi-rust" }',cargo,flags=re.M)
    cargo=replace(cargo, 'name = "madloop_tirust"', 'name = "madloop_tirust_qp_shadow_attempt2"')
    cargo=replace(cargo, 'crate-type = ["cdylib", "rlib"]', 'crate-type = ["cdylib"]')
    # Reuse the immutable attempt1 dependency identity; only the observer/core
    # and bridge change. The identical attempt2 copy remains in the inventory.
    original_oneloop=ROOT/'.tirust-build/qp-scalar-shadow-profile/oneloop'
    expected={str(p.relative_to(WORK/'oneloop')):sha(p) for p in (WORK/'oneloop').rglob('*') if p.is_file()}
    assert expected=={str(p.relative_to(original_oneloop)):sha(p) for p in original_oneloop.rglob('*') if p.is_file()}
    cargo += '\n[patch."https://github.com/alphal00p/oneloopmaster.git"]\noneloop = { path = '+json.dumps(str(original_oneloop))+' }\n'
    (bridge/'Cargo.toml').write_text(cargo)
    inputs()
    files=[p for p in WORK.rglob('*') if p.is_file()]
    runtime=ROOT/'.tirust-build/ew-threshold-tensor-replay-runtime'
    runtime_inputs={str((runtime/name).relative_to(ROOT)):sha(runtime/name)
                    for name in ('libmadloop_tirust.so','libtirust.a','mtirust.mod','mtirust.f90')}
    assert runtime_inputs[str((runtime/'libmadloop_tirust.so').relative_to(ROOT))]=='49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
    (HERE/'prepared.json').write_text(json.dumps({
        'state':'source_prepared_not_compiled', 'no_native_execution':True,
        'root':str(ROOT),'isolated_root':str(WORK),
        'source_before':{str(p.relative_to(ROOT)):sha(p) for p in [exec_source,bridge_source,UPSTREAM/'crates/iregi-rust/src/lib.rs']},
        'isolated_source_sha256':{str(p.relative_to(WORK)):sha(p) for p in sorted(files)},
        'frozen_fortran_runtime_sha256':runtime_inputs,
        'actual_oneloop_source':str(original_oneloop),
        'actual_oneloop_source_sha256':expected,
        'changed_numerical_operations':False,
        'private_manifest_changes':'Limit workspace members; bridge uses private path dependency; no dependency revision change.',
        'caps':{'build_seconds':900,'build_jobs':2,'replay_seconds_each':180,'replay_processes':3,'scalar_observation_limit':4096,'json_payload_limit_bytes':8388608},
    },indent=2)+'\n')
    print('Prepared isolated source and 24 saved parents; no compiler/native process launched.')

if __name__ == '__main__':
    main()
