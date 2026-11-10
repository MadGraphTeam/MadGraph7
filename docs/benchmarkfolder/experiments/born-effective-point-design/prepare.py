#!/usr/bin/env python3
"""Freeze the source-only hook specification; never invoke MG or native tools."""
import base64
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
HOST = 'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc'
MP = 'madgraph/iolibs/template_files/loop_optimized/mp_compute_loop_coefs.inc'
REPAIR = 'madgraph/iolibs/template_files/loop/improve_ps.inc'
EXPORT = 'madgraph/loop/loop_exporters.py'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(name, value):
    text = json.dumps(value, indent=2, sort_keys=True)+'\n'
    path = HERE / name
    if path.exists():
        assert path.read_text() == text or not (HERE/'evidence-index.json').exists(), f'Refusing to change frozen {name}'
    path.write_text(text)


def main():
    assert Path.cwd().resolve() == ROOT
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    hooks = [
        ('original_born_boundary', HOST,
         'CALL %(proc_prefix)sSMATRIXHEL_SPLITORDERS(P_USER,USERHEL,IC,BORNBUFF(0))', 1,
         'Retain ordinary call; snapshot IEEE entry for nonfinite fallback, reset call-local certificates before early exits. Nonfinite Born branches to staged C/first-QP preparation before REF-dependent work.'),
        ('reference_update', HOST, 'REF=REF+ANS(0,I)', 1,
         'Publish a checked contiguous Born buffer into the strided ANS row; keep REF=sum(Born(1:NSQSO_BORN)). Refresh for finite-Born QP rescue too.'),
        ('ordinary_mp_resets', HOST, 'MP_DONE=.FALSE.\nMP_DONE_ONCE=.FALSE.\nMP_PS_SET=.FALSE.', 1,
         'Preserve any current-call staged C across these resets; set MP_PS_SET only after successful adoption. Never set MP_DONE or MP_DONE_ONCE for preparation/Born work.'),
        ('qp_policy_entry', HOST, '205 CONTINUE', 1,
         'For every selected QP library, resolve the integrand policy outside the MP_PS_SET guard and before any MP_DONE/reuse test. First adoption/policy change refreshes Born and starts a fresh QP stability cohort (result slots, EVAL_DONE, STAB_INDEX); historical cohorts are diagnostic only.'),
        ('qp_coordinate_entry', HOST, 'IF (.NOT.MP_PS_SET.AND.(CTMODE.EQ.0.OR.CTMODE.GE.4)) THEN', 1,
         'Prepare/adopt C only once. Keep CTMode0 and unsupported formats on explicit legacy setter path. Failure uses initialized-stage cleanup and rejected output.'),
        ('qp_backend_reuse_shortcut', HOST, 'I_LIB=INDEX_QP_TOOLS(I_QP_LIB)', 1,
         'Recheck the selected policy before the zero-rotation GOTO300 shortcut (which bypasses label205 and numerator generation). If the policy changed, route through staging and recomputation with MP_DONE false; preserve same-policy direction/reuse shortcuts.'),
        ('host_initialization_commit', HOST, '  NTRY=NTRY+1', 1,
         'Nonfinite-Born preparation failures occur before this counter/filter publication. Use validity flags for threshold/bypass/rotation snapshots; do not jump blindly to final cleanup with uninitialized state/backend identity.'),
        ('actual_numerator_completion', HOST, 'CALL %(proc_prefix)sMP_COMPUTE_LOOP_COEFS(MP_P,BUFFR_BIS)', 2,
         'Existing calls remain actual completion boundaries. Invalidate MP_DONE and coefficient-reuse state on policy/view change; do not fabricate completion.'),
        ('rotation', HOST, 'IF (DOING_QP_EVALS) CALL %(proc_prefix)sMP_ROTATE_PS', 3,
         'Use fixed MP_PS=C for every rotation, preserve current MP_P view and cache resets. No second phase-space repair.'),
        ('failure_cleanup', HOST, 'C Make sure that we finish by emptying caches', 1,
         'Restore only initialized threshold/rotation/bypass snapshots and relevant caches, then finalize rejected accuracy1/invalid output. Do not let CHECKPHASE overwrite a preparation-failure rejection or access an uninitialized backend identity.'),
        ('integrand_policy', MP, 'COMPUTE_INTEGRAND_IN_QP = ((MLREDUCTIONLIB(LIBINDEX)', 1,
         'One generated policy predicate shared with effective-Born staging; preserve all existing UseQPIntegrand flags and actual selected-library identity.'),
        ('model_sync', MP, 'CALL MP_UPDATE_AS_PARAM()', 1,
         'The true Born evaluator uses the existing MP model after the same synchronization contract. No decimal reparse or blanket promotion of DP parameters.'),
        ('rounded_view', MP, 'W(I)%P = DPW(I)%P', 1,
         'False integrand option retains rounded DP coordinate/numerator view; published Born/reference must match it. QP tensor and mass policies remain independently reported.'),
        ('born_graph', EXPORT, 'bornME.optimization = True', 1,
         'Reuse the copied Born graph/color/split-order data to emit a stateless precision-specific aggregate with generated HELAS calls, flavor, polarization, averaging and selections. No fixture dimensions or shared waveform scratch.'),
        ('split_born_learning', 'madgraph/iolibs/template_files/matrix_standalone_splitOrders_v4.inc', 'IF (BUFF .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN', 1,
         'Generate a narrowly supported transaction for private saved Born learning state: successful finite calls retain exact existing updates; failed nonfinite attempts restore entry NTRY/GOODHEL/HELRESET/selection state. Host cleanup cannot access these saved locals.'),
        ('unsplit_born_learning', 'madgraph/iolibs/template_files/matrix_standalone_v4.inc', 'IF (T .NE. 0D0 .AND. .NOT.    GOODHEL(IHEL,FLAV_IDX)) THEN', 1,
         'Apply the same generated transaction to the ordinary Born template without changing successful finite calls or unsupported exports. Validate selection/learning equivalence through source fixtures before native tests.'),
        ('repair_outcome', REPAIR, 'C Report to the user or update the PS point.', 1,
         'Render a checked sibling preserving the existing numerical body and explicit primary/alternate errors. Copy output only on actual success; a finite retained input is not success.'),
    ]
    source_paths = {HOST, MP, REPAIR, EXPORT,
        'madgraph/iolibs/template_files/matrix_standalone_v4.inc',
        'madgraph/iolibs/template_files/matrix_standalone_splitOrders_v4.inc',
        'madgraph/iolibs/helas_call_writers.py'}
    records = {}
    for relative in sorted(source_paths):
        data = (ROOT / relative).read_bytes()
        blob = subprocess.check_output(['git', 'rev-parse', head+':'+relative], text=True).strip()
        committed = subprocess.check_output(['git', 'cat-file', 'blob', blob])
        record = {'sha256': sha(data), 'bytes': len(data), 'git_blob_at_recorded_head': blob,
            'git_blob_sha256': sha(committed), 'working_differs_from_head': data != committed}
        if data != committed:
            record['working_source_gzip_base64'] = base64.b64encode(gzip.compress(data, mtime=0)).decode()
        records[relative] = record
    hook_records = []
    for name, relative, anchor, count, action in hooks:
        text = (ROOT / relative).read_text()
        assert text.count(anchor) == count, (name, text.count(anchor), count)
        hook_records.append({'name': name, 'source': relative, 'anchor': anchor,
            'expected_occurrences': count, 'proposed_action': action})
    evidence_paths = [
        'docs/benchmarkfolder/experiments/general-kinematics-rescue-design/evidence-index.json',
        'docs/benchmarkfolder/experiments/qcd-topology-rescue-native/evidence-index.json',
        'docs/benchmarkfolder/experiments/qcd-topology-rescue-native/RESULTS.md',
        'docs/benchmarkfolder/experiments/qcd-topology-rescue-prototype/born-c-design/born-c-handoff.patch',
        'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype/revision-1/born_fallback.f',
        'docs/benchmarkfolder/experiments/ew-qp-input-policy/evidence-index.json']
    evidence = {name: sha((ROOT/name).read_bytes()) for name in evidence_paths}
    save('source-bindings.json', {'schema': 1, 'repository_head_context': head,
        'identity_contract': 'Source bytes are independently hashed. A Git HEAD is not an assertion that dirty source belongs to it; dirty bytes are snapshotted.',
        'source_sha256': records, 'evidence_sha256': evidence,
        'production_changed': False, 'compiler_or_reducer_executed': False})
    save('hook-plan.json', {'schema': 1, 'apply_ready': False, 'compile_ready': False,
        'reason': 'Reviewable implementation plan. Checked-repair and generated stateless Born bodies must be implemented together; these hooks alone are not executable changes.',
        'supported': 'scalar optimized on-shell SM Born with direct interference reduction',
        'legacy_preserved': ['LoopInduced', 'AmplitudeReduction/color flow', 'CTMode0', 'off-shell/spacelike', 'vector'],
        'interfaces': {
            'checked_repair': 'PREPARE_EFFECTIVE_POINT(P_USER, C128, C8, ERR_PRIMARY, ERR_ALTERNATE, OK); same model and repair arithmetic, explicit success.',
            'born': 'EFFECTIVE_BORN(C128, C8, INTEGRAND_POLICY, USERHEL, FLAV_IDX, BORN(0:NSQSO_BORN), OK); stateless generated aggregate, no loop scratch mutation.',
            'adoption': 'ADOPT_EFFECTIVE_CONTEXT(..., CONTEXT_CHANGED, OK); call-local readiness distinct from numerator completion; update host arrays/REF atomically.',
            'cleanup': 'Current-call rejected state survives normal cleanup; precision return digits describe actual loop work only.'},
        'hooks': hook_records,
        'no_accuracy_claim': 'Coordinate/normalization alignment does not resolve scalar/tensor accuracy or the separate default stability estimator limitation.'})
    for relative, record in records.items():
        assert sha((ROOT/relative).read_bytes()) == record['sha256']
    print(json.dumps({'source_files': len(records), 'hook_anchors': len(hook_records),
        'native_execution': False, 'apply_ready': False}))


if __name__ == '__main__':
    main()
