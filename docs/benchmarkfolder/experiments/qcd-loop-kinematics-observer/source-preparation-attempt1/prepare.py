#!/usr/bin/env python3
"""Prepare insertion-only QP observer sources. Never compile or run native code."""
from pathlib import Path
import difflib
import hashlib
import json
import re

ROOT = Path.cwd().resolve()
HERE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer'
PRIOR = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace'
BORN = ROOT / 'docs/benchmarkfolder/experiments/qcd-born-fallback-prototype'
PROC = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
SP = PROC / 'SubProcesses/P0_uux_zgg'
block_id = 0


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def block(lines):
    global block_id
    block_id += 1
    return ('C QOBS_BEGIN ' + str(block_id) + '\n' + lines.rstrip() + '\n'
            + 'C QOBS_END ' + str(block_id) + '\n')


def unique(text, needle):
    assert text.count(needle) == 1, (needle, text.count(needle))


def before(text, needle, lines):
    unique(text, needle)
    return text.replace(needle, block(lines) + needle)


def after(text, needle, lines):
    unique(text, needle)
    return text.replace(needle, needle + block(lines))


def body(text, routine, transform):
    start = text.index('      SUBROUTINE ' + routine)
    end = text.index('\n      END\n', start) + len('\n      END\n')
    old = text[start:end]
    return text[:start] + transform(old) + text[end:]


def use(text):
    # The first USE, when present, starts after the complete continued signature.
    anchor = next(line + '\n' for line in text.splitlines()
                  if re.match(r'^      (USE |IMPLICIT |INTEGER |REAL\*|COMPLEX\*|LOGICAL )', line))
    return before(text, anchor, '      USE QCD_KINEMATICS_OBSERVER')


def put(tag, value, count, i='0', j='0'):
    return (f"      IF(QO_IS_ACTIVE()) CALL QO_PUT('{tag}',{i},{j},\n"
            f'     $ TRANSFER({value},(/0_8/),{count}))')


def mark(tag, value='0', i='0', j='0'):
    return f"      CALL QO_MARK('{tag}',{i},{j},{value})"


def returns(text):
    lines = text.splitlines(True)
    out = []
    for line in lines:
        if line.strip() == 'RETURN':
            out.append(block('      CALL QO_END_CALL(STABLE,FPE_IN_QP_REDUCTION)'))
        if line == '      END\n':
            out.append(block('      CALL QO_END_CALL(STABLE,FPE_IN_QP_REDUCTION)'))
        out.append(line)
    return ''.join(out)


def nrs(text):
    text = use(text)
    text = before(text, '      STABLE=.FALSE.\n', '\n'.join([
        '      CALL QO_BEGIN_CALL(8,CTMODE,ID,SQSOINDEX,NLOOPLINE,RANK)',
        mark('FPE_ENTRY', 'MERGE(1,0,FPE_IN_QP_REDUCTION)'),
        put('EDGE_INITIAL', 'PL', '8*NLOOPLINE'),
        put('MASS2_DP', 'M2L', '2*NLOOPLINE'),
        put('NUM_ORIGINAL', 'LOOPCOEFS(0:NCOEF_R(RANK)-1,SQSOINDEX,ID)', '4*NCOEF_R(RANK)'),
        '      CALL QO_CAPTURE_CONTEXT()',
    ]))
    anchor = '      CALL ML5_0_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)\n'
    text = after(text, anchor, '\n'.join([
        put('EDGE_ORDERED', 'PL', '8*NLOOPLINE'),
        put('MASS2_ORDERED', 'MASSES', '4*NLOOPLINE'), mark('ORDER_DONE')]))
    text = after(text, '      REF_P=MAXVAL(ABSP_TMP)\n', '\n'.join([
        put('CLOSURE_SUM', 'P_TMP(:,0)', '8'),
        put('CLOSURE_ABS', 'ABSP_TMP', '8'), put('CLOSURE_REF', 'REF_P', '2'),
        mark('CLOSURE_CHECK')]))
    anchor = '      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES\n     $ ,S_MAT)\n'
    text = before(text, anchor, put('SHIFTS', 'P_NRS', '8*NLOOPLINE'))
    text = after(text, anchor, put('S_COMPLEX', 'S_MAT', '4*NLOOPLINE*NLOOPLINE'))
    text = before(text, '      CURR_MAXCOEF=NCOEF_R(RANK)\n',
                  put('DISTANCES', 'REAL_S_MAT', '2*NLOOPLINE*NLOOPLINE'))
    anchor = '      CALL NINJARS_TENSOR_EVALUATE(TENSORCOEFS(0:CURR_MAXCOEF-1)\n'
    text = before(text, anchor, '\n'.join([
        put('NUM_NATIVE', 'TENSORCOEFS(0:CURR_MAXCOEF-1)', '4*CURR_MAXCOEF'),
        put('NATIVE_SCALE', 'NRS_SCALE', '2'), mark('NATIVE_BEGIN')]))
    anchor = '     $ ,NRS_STATUS,MAX(0,LOOPPOLEMODE))\n'
    text = after(text, anchor, '\n'.join([
        '      CALL QO_NATIVE_STATUS(NRS_STATUS)',
        '      IF(NRS_STATUS.NE.NINJARS_SUCCESS) CALL QO_ERROR_NRS()',
        '      IF(NRS_STATUS.EQ.NINJARS_SUCCESS) THEN',
        put('NATIVE_CONTRACTED', 'NRS_RES', '12'),
        '      ENDIF', mark('NATIVE_RETURN')]))
    anchor = '      DO I=1,3\n        STABLE=STABLE.AND.IEEE_IS_FINITE(DBLE(RES(I)))\n'
    text = before(text, anchor, put('DP_CONVERTED', 'RES', '6'))
    return returns(text)


def tir_wrapper(text):
    text = use(text)
    text = before(text, '      RES=0\n', '\n'.join([
        '      CALL QO_BEGIN_CALL(9,CTMODE,ID,SQSOINDEX,NLOOPLINE,RANK)',
        mark('FPE_ENTRY', 'MERGE(1,0,FPE_IN_QP_REDUCTION)'),
        put('EDGE_INITIAL', 'PL', '8*NLOOPLINE'),
        put('MASS2_DP', 'M2L', '2*NLOOPLINE'),
        put('NUM_ORIGINAL', 'LOOPCOEFS(0:NCOEF_R(RANK)-1,SQSOINDEX,ID)', '4*NCOEF_R(RANK)'),
        '      CALL QO_CAPTURE_CONTEXT()',
    ]))
    anchor = '      RESULT=0\n'
    text = before(text, anchor, put('MAPPED_MOMENTS', 'COEFS(0:NCOEF_R(RANK)-1,1:3)', '12*NCOEF_R(RANK)'))
    anchor = '        RESULT(I)=2.0E0_16*NORMALIZATION*REAL(RESULT(I),KIND=KI_QTIR)\n'
    text = before(text, anchor, put('CONTRACTION_RAW', 'RESULT(I)', '4', i='I'))
    text = after(text, anchor, put('CONTRACTION_PROJECTED', 'RESULT(I)', '4', i='I'))
    anchor = '        RES(I)=CMPLX(RESULT(I),KIND=KIND(0D0))\n'
    text = after(text, anchor, put('DP_CONVERTED', 'RES(I)', '2', i='I'))
    return returns(text)


def tir_diagnostic(text):
    text = use(text)
    text = before(text, '      IF(RANK.LT.0.OR.RANK.GT.4) RETURN\n', mark('RANK_SUPPORT_CHECK', 'RANK'))
    text = after(text, '      CALL ML5_0_INITIALIZE_TIRUST(STATUS)\n', mark('INIT_STATUS', 'STATUS'))
    anchor = '     $ ,STATUS)\n'
    text = after(text, anchor, mark('BUDGET_STATUS', 'STATUS'))
    anchor = '      CALL ML5_0_MP_SWITCH_ORDER(CTMODE,NLOOPLINE,PL,PDEN_DUMMY,MASSES)\n'
    text = after(text, anchor, '\n'.join([
        put('EDGE_ORDERED', 'PL', '8*NLOOPLINE'),
        put('MASS2_ORDERED', 'MASSES', '4*NLOOPLINE'), mark('ORDER_DONE')]))
    text = after(text, '      REF_P=MAXVAL(ABSP)\n', '\n'.join([
        put('CLOSURE_SUM', 'P_TMP(:,0)', '8'), put('CLOSURE_ABS', 'ABSP', '8'),
        put('CLOSURE_REF', 'REF_P', '2'), mark('CLOSURE_CHECK')]))
    anchor = '      CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(NLOOPLINE,P_S_MAT,MASSES\n     $ ,S_MAT)\n'
    text = before(text, anchor, put('SHIFTS', 'SHIFTS', '8*NLOOPLINE'))
    text = after(text, anchor, put('S_COMPLEX', 'S_MAT', '4*NLOOPLINE*NLOOPLINE'))
    text = before(text, '      SCALE=REAL(MU_R,KIND=KI_QTIR)**2\n',
                  put('DISTANCES', 'INVARIANTS', '2*NLOOPLINE*NLOOPLINE'))
    anchor = '      CALL TIRUST_TENSOR_EVALUATE(NLOOPLINE,RANK,SHIFTS,INVARIANTS\n'
    text = before(text, anchor, '\n'.join([put('NATIVE_SCALE', 'SCALE', '2'), mark('NATIVE_BEGIN')]))
    anchor = '     $ ,TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED)\n'
    text = after(text, anchor, '\n'.join([
        '      CALL QO_NATIVE_STATUS(STATUS)',
        '      IF(STATUS.NE.TIRUST_SUCCESS) CALL QO_ERROR_TIRUST()',
        '      IF(STATUS.EQ.TIRUST_SUCCESS) THEN',
        put('NATIVE_MOMENTS', 'RAW(1:NC,1:4)', '16*NC'),
        '      ENDIF', mark('NATIVE_RETURN')]))
    return text


def matrix(text):
    text = use(text)
    anchor = '          IF(I.EQ.J)THEN\n            S_MAT(I,J)=-(M2L(I)+M2L(J))\n'
    text = after(text, anchor, mark('PAIR_DIAGONAL', i='I', j='J'))
    anchor = '            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)\n'
    text = after(text, anchor, '\n'.join([
        put('PAIR_DIFFSQ', 'DIFFSQ', '4', i='I', j='J'),
        put('PAIR_S_INITIAL', 'S_MAT(I,J)', '4', i='I', j='J')]))
    for index, anchor in [(1, '                S_MAT(I,J)=-M2L(J)\n'),
                          (2, '                S_MAT(I,J)=-M2L(I)\n'),
                          (3, '                S_MAT(I,J)=-(M2L(I)+M2L(J))\n')]:
        text = after(text, anchor, '\n'.join([
            mark('PAIR_CLAMP', str(index), 'I', 'J'),
            put('PAIR_S_CLAMPED', 'S_MAT(I,J)', '4', i='I', j='J')]))
    anchor = '            REF_NORMALIZATION = (REF_NORMALIZATION/2.0E0_16)**2\n'
    text = after(text, anchor, put('PAIR_E_REFERENCE', 'REF_NORMALIZATION', '2', i='I', j='J'))
    anchor = '          ENDIF\n\n        ENDDO\n'
    unique(text, anchor)
    text = text.replace(anchor, '          ENDIF\n\n' + block(
        put('PAIR_S_FINAL', 'S_MAT(I,J)', '4', i='I', j='J')) + '        ENDDO\n')
    return text


def set_mp(text):
    text = before(text, '      INTEGER    NEXTERNAL\n', '      USE QCD_KINEMATICS_OBSERVER')
    anchor = '      CALL ML5_0_MP_IMPROVE_PS_POINT_PRECISION(KEEP_OFFSHELL_MASS\n'
    text = before(text, anchor, put('POINT_MP_BEFORE', 'MP_PS', '40'))
    text = before(text, '      END\n', '\n'.join([
        put('POINT_MP_AFTER', 'MP_PS', '40'), put('POINT_MP_P', 'MP_P', '40')]))
    return text


def driver(text):
    text = use(text)
    text = before(text, '      COMMON/AUDIT_BORN_RESCUE_COUNTS/RESCUES,FAILS\n',
                  '      CHARACTER*1024 OBS_PATH')
    anchor = '      CALL GET_COMMAND_ARGUMENT(3,RESOURCES)\n'
    text = after(text, anchor, '      CALL GET_COMMAND_ARGUMENT(4,OBS_PATH)\n      CALL QO_OPEN(OBS_PATH)')
    anchor = "        WRITE(*,'(A,2(1X,I4))') 'AMPLITUDE_BEGIN',I,K\n        FLUSH(6)\n"
    text = after(text, anchor, '\n'.join([
        '        CALL QO_BEGIN_POINT(I,K)', put('POINT_P_USER', 'P', '20')]))
    anchor = '        CALL IEEE_GET_FLAG(IEEE_OVERFLOW,OVERFLOW)\n'
    text = after(text, anchor, '\n'.join([
        put('POINT_HOST_ANS', 'ANS', '8'), put('POINT_HOST_ACC', 'ACC', '2'),
        put('POINT_G_DP', 'G', '1'), put('POINT_MU_DP', 'MU_R', '1'),
        mark('POINT_INVALID', 'MERGE(1,0,INVALID)'),
        mark('POINT_DIVZERO', 'MERGE(1,0,DIVZERO)'),
        mark('POINT_OVERFLOW', 'MERGE(1,0,OVERFLOW)'),
        '        CALL QO_END_POINT(CODE)']))
    return text


def strip_insertions(text):
    return re.sub(r'(?ms)^C QOBS_BEGIN (\d+)\n.*?^C QOBS_END \1\n', '', text)


def main():
    assert not (HERE / 'prepared.json').exists(), 'Refusing to replace prepared evidence'
    for directory in (PRIOR, BORN):
        inventory = json.loads((directory / 'evidence-index.json').read_text())
        for name, entry in inventory['files'].items():
            assert sha(directory / name) == entry['sha256'], name
    historical = json.loads((BORN / 'amplitude-native-1/build.json').read_text())
    assert historical['complete'] and historical['protected_inputs_unchanged']
    for path, expected in historical['protected_inputs_before'].items():
        assert sha(ROOT / path) == expected, path
    source = HERE / 'sources'
    source.mkdir()
    specs = [
        (SP / 'CT_interface.f', lambda text: body(body(text, 'ML5_0_MP_BUILD_KINEMATIC_MATRIX', matrix), 'ML5_0_MP_NINJARS_LOOP', nrs)),
        (SP / 'TIRUST_interface.f', lambda text: body(body(text, 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC', tir_diagnostic), 'ML5_0_MP_TIRUST_LOOP', tir_wrapper)),
        (BORN / 'loop_matrix.f', lambda text: body(text, 'ML5_0_SET_MP_PS(P)', set_mp)),
        (BORN / 'amplitude_driver.f', driver),
    ]
    records = []
    for path, transform in specs:
        old = path.read_text()
        assert 'QOBS_BEGIN' not in old and 'QO_' not in old
        new = transform(old)
        assert strip_insertions(new) == old, 'Original statement bytes changed: ' + str(path)
        for number, line in enumerate(new.splitlines(), 1):
            if len(line) > 132:
                assert line in old.splitlines(), (path.name, number, len(line))
        (source / (path.name + '.original')).write_text(old)
        (source / path.name).write_text(new)
        patch = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True),
                                           fromfile=path.name + '.original', tofile=path.name))
        (source / (path.name + '.patch')).write_text(patch)
        records.append({'original_path': str(path.relative_to(ROOT)), 'original_sha256': sha(path),
                        'observed_path': str((source / path.name).relative_to(ROOT)),
                        'observed_sha256': sha(source / path.name), 'strip_insertions_exact': True})
    report = {
        'schema': 1, 'scope': 'Insertion-only private QP observer preparation; no compile or native evaluation.',
        'native_executed': False, 'production_modified': False,
        'source_transformations': records, 'insertion_blocks': block_id,
        'protected_historical_inputs': historical['protected_inputs_before'],
        'frozen_evidence_indices': {str((p / 'evidence-index.json').relative_to(ROOT)): sha(p / 'evidence-index.json')
                                   for p in (PRIOR, BORN)},
        'prepare_sha256': sha(HERE / 'prepare.py'),
        'limits': ['Source equivalence is not native parity; observer compile/output/IEEE validation remains required.',
                   'Fortran public status may include decimal transport failure, not exclusively raw Rust ABI status.',
                   'All group visits are retained; source-target pairs guide analysis without filtering out other failures.'],
    }
    (HERE / 'prepared.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'sources_prepared': len(records), 'insertion_blocks': block_id,
                      'original_statement_bytes_preserved': True, 'native_executed': False}))


if __name__ == '__main__':
    main()
