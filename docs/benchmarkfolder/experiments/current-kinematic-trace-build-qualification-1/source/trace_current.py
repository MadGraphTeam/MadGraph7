"""Re-derive insertion-only traces from hash-bound CURRENT generated sources.

This module only reads/writes text. It never generates a process, compiles,
loads a library or evaluates an amplitude. The historical emitter supplies
fixed-form insertion primitives; all anchors are checked against current bytes.
"""
import difflib
import json
from pathlib import Path
import re

from admission import HERE, ROOT, historical_modules, require, sha, save
from bypass_contract import validate as validate_bypass


def driver_common_contract(saved, source):
    """Read-only driver observations require these initialized current COMMONs."""
    routines=saved.blocks(source)
    candidates=[body for name,body in routines.items() if name.endswith('_SLOOPMATRIX_RETRY_IMPL')]
    require(len(candidates)==1,'Unique full-amplitude matrix routine required')
    statements=[s for _,s,_ in candidates[0]['statements']]
    required=('DATACHECKPHASE/.TRUE./','DATAHELDOUBLECHECKED/.FALSE./',
              'COMMON/ML5_0_INIT/CHECKPHASE,HELDOUBLECHECKED',
              'REAL*8LSCALE','INTEGERCTMODE','COMMON/ML5_0_CT/LSCALE,CTMODE',
              'DATAFPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION/.FALSE.,.FALSE./',
              'COMMON/ML5_0_FPE_IN_REDUCTION/FPE_IN_DP_REDUCTION,FPE_IN_QP_REDUCTION')
    for value in required:
        require(value in statements,'Driver COMMON/DATA contract differs: '+value)
    return {'scope':'source declaration/DATA only; native storage parity pending',
            'required_statements':list(required)}


def current_emitter(groups):
    require(groups and all(type(g) is int and g > 0 for g in groups),
            'Positive, reviewed group IDs required')
    saved, emitter = historical_modules()
    old_loop = emitter.instrument_loop
    old_tensor = emitter.instrument_tirust

    def declarations(prefix, matrix=False, loop_id=False):
        code = ('      INTEGER*8 KT_SERIAL\n'
                '      INTEGER KT_UNIT,KT_POINT\n'
                '      LOGICAL KT_ENABLED,KT_ACTIVE\n'
                f'      COMMON/{prefix}PRIVATE_KTRACE/KT_SERIAL,KT_UNIT,\n'
                '     $ KT_POINT,KT_ENABLED,KT_ACTIVE\n')
        if matrix or loop_id:
            code += ('      INTEGER KT_ID,KT_SQSO,KT_R\n'
                     f'      COMMON/{prefix}LOOP/KT_ID,KT_SQSO,KT_R\n')
        if matrix:
            code += ('      INTEGER KT_MODE\n      REAL*8 KT_LSCALE\n'
                     f'      COMMON/{prefix}CT/KT_LSCALE,KT_MODE\n')
        return emitter.marked(code)

    def write_bits(tag, payloads, identifier='KT_ID', mode='KT_MODE', rank='KT_R',
                   i='0', j='0', branch='0'):
        sqso = 'SQUAREDSOINDEX' if identifier == 'LOOPNUM' else 'KT_SQSO'
        lines = ['      IF(KT_ACTIVE) THEN',
                 "        WRITE(KT_UNIT,'(A,10(1X,I0),*(1X,Z16.16))')",
                 f"     $ '{tag}',KT_POINT,KT_SERIAL,{identifier},{mode},{rank},",
                 f'     $ NLOOPLINE,{i},{j},{branch},{sqso},']
        lines += ['     $ '+v+(',' if n < len(payloads)-1 else '')
                  for n, v in enumerate(payloads)]
        lines.append('      ENDIF')
        return emitter.marked('\n'.join(lines))

    def loop(body, prefix):
        result = old_loop(body, prefix)
        old = emitter.marked('      IF(KT_ENABLED) KT_SERIAL=KT_SERIAL+1_8')
        select = ['      IF(KT_ENABLED) KT_SERIAL=KT_SERIAL+1_8',
                  '      KT_ACTIVE=KT_ENABLED.AND.(']
        select += ['     $ '+(' .OR.' if i else '')+'LOOPNUM.EQ.'+str(g)
                   for i, g in enumerate(groups)]
        select[-1] += ')'
        result = emitter.replace_once(result, old, emitter.marked('\n'.join(select)))
        # Only change inserted private blocks; never substitute original code.
        def active(match):
            return match.group(0).replace('IF(KT_ENABLED) THEN', 'IF(KT_ACTIVE) THEN')
        result = re.sub(re.escape(emitter.BEGIN)+r'.*?'+re.escape(emitter.END),
                        active, result, flags=re.S)
        return result

    def tensor(body, prefix, qp):
        result = old_tensor(body, prefix, qp)
        suffix = 'QP' if qp else 'DP'
        width = 2 if qp else 1
        # Existing zero-seeded residual is observed before its original guard.
        anchor = '      REF_P=MAXVAL(ABSP)\n'
        result = emitter.replace_once(result, anchor, anchor+write_bits(
            'KT_RESIDUAL_'+suffix,
            [f'TRANSFER(P_TMP(:,0),0_8,{4*width})',
             f'TRANSFER(PL,0_8,{4*width}*NLOOPLINE)'],
            mode='CTMODE', rank='RANK'))
        anchor = '     $ ,TIRUSTCOMPUTEFINITE,CHECK_PASS,CACHE_ENABLED)\n'
        status = emitter.marked(
            '      IF(KT_ACTIVE) THEN\n'
            "        WRITE(KT_UNIT,'(A,11(1X,I0))')\n"
            f"     $ 'KT_STATUS_{suffix}',KT_POINT,KT_SERIAL,KT_ID,CTMODE,\n"
            '     $ RANK,NLOOPLINE,0,0,0,KT_SQSO,STATUS\n'
            '      ENDIF')
        return emitter.replace_once(result, anchor, anchor+status)

    def control(prefix):
        declaration = declarations(prefix).removeprefix(emitter.BEGIN).removesuffix(emitter.END)
        return emitter.marked(
            f'      SUBROUTINE {prefix}PRIVATE_KTRACE_SET(IUNIT,IPOINT,ON)\n'
            '      IMPLICIT NONE\n      INTEGER IUNIT,IPOINT\n'
            '      LOGICAL ON\n'+declaration+
            '      KT_ENABLED=ON\n      KT_ACTIVE=.FALSE.\n'
            '      KT_UNIT=IUNIT\n      KT_POINT=IPOINT\n'
            '      KT_SERIAL=0_8\n      IF(KT_ENABLED) THEN\n'
            "        WRITE(KT_UNIT,'(A,5(1X,I0))') 'KT_SCHEMA_V2',\n"
            '     $ KT_POINT,STORAGE_SIZE(0D0),STORAGE_SIZE(0.0_16),\n'
            '     $ BIT_SIZE(0_8),STORAGE_SIZE(.TRUE.)\n'
            '      ENDIF\n      END\n'
            f'      BLOCK DATA {prefix}PRIVATE_KTRACE_INIT\n'+declaration+
            '      DATA KT_ENABLED/.FALSE./,KT_ACTIVE/.FALSE./\n'
            '      DATA KT_UNIT/0/,KT_POINT/0/,KT_SERIAL/0_8/\n      END\n')

    emitter.declarations = declarations
    emitter.write_bits = write_bits
    emitter.instrument_loop = loop
    emitter.instrument_tirust = tensor
    emitter.control_routines = control
    return saved, emitter


def derive(binding_path, expected_sha, groups, destination):
    """The binding must already exist and be reviewed/pinned before parsing."""
    saved, emitter = current_emitter(groups)
    sources = saved.Sources(ROOT, binding_path, expected_sha)
    routing = saved.consume(sources)
    require(len(routing['cases']) == 1, 'One generated process per derivation')
    key, route = next(iter(routing['cases'].items()))
    case = sources.binding['cases'][key]
    common_contract=driver_common_contract(saved,sources.text(case['subprocess']+'/loop_matrix.f'))
    commons_name=case['subprocess']+'/MadLoopCommons.f'
    if commons_name not in sources.raw:
        commons_name=str(Path(case['subprocess']).parent/'MadLoopCommons.f')
    bypass=validate_bypass(saved,sources.text(case['subprocess']+'/loop_matrix.f'),
                           sources.text(commons_name))
    ids = {r['reduction_identifier']: r for r in route['reductions']}
    require(all(g in ids and ids[g]['identity_kind'] == 'group' for g in groups),
            'Requested target does not resolve to an actual current group')
    expected_corners={
        'cms_ew_n2_epem_mupmum':((1,0,0,0),(0,1,0,0),(0,0,0,-1),(0,0,-1,0)),
        'ew_n3_epem_ttxa':((0,0,-1,0,0),(0,0,0,-1,0),(0,0,0,0,-1),(1,1,0,0,0))}
    require(key in expected_corners,'Unexpected diagnostic process')
    for group in groups:
        require(tuple(map(tuple,ids[group]['modes']['1']['corners']))==expected_corners[key],
                'Current target group has changed signed corner provenance')
        require(ids[group]['modes']['1']['shifts'][0]==ids[group]['conservation_coefficients'],
                'First shift residual was simplified or changed')
    require(all(e['leg'] == e['wavefunction_slot'] for e in route['external_provenance']),
            'External wavefunction slots differ from physical leg indexes')
    types = [s for _, s, _ in saved.statements(sources.text(case['dhelas']+'/aloha_functions.f'))]
    for name, expected in (('ALOHA', 'DOUBLECOMPLEX::P(0:3)'),
                           ('MP_ALOHA', 'COMPLEX*32::P(0:3)')):
        require(expected in types[types.index('TYPE'+name):types.index('ENDTYPE'+name)],
                'External complex storage type differs')
    destination = Path(destination)
    require(not destination.exists(), 'Fresh trace destination required')
    destination.mkdir(parents=True)
    records = []
    for name in ('CT_interface.f', 'TIRUST_interface.f'):
        source_name = case['subprocess']+'/'+name
        original = sources.text(source_name)
        proposed, counts = emitter.instrument(original, name)
        require(emitter.strip_trace(proposed) == original, 'Original bytes changed')
        (destination/name).write_text(proposed, encoding='ascii')
        diff = ''.join(difflib.unified_diff(original.splitlines(True), proposed.splitlines(True),
                                           fromfile='a/'+source_name, tofile='b/'+source_name))
        (destination/(name+'.patch')).write_text(diff, encoding='ascii')
        records.append({'input': source_name, 'input_sha256': sha(original.encode()),
                        'proposed_sha256': sha(proposed.encode()), 'boundaries': counts,
                        'strip_recovers_original_bytes': True})
    route['scope'] = 'Current saved generated source; no native routing qualification'
    save(destination/'routing.json', routing)
    report = {'schema': 'current-kinematic-trace-derivation-v2',
              'input_binding_sha256': expected_sha, 'target_groups': groups,
              'target_identities': [ids[g] for g in groups], 'records': records,
              'header_fields': ['point','serial','id','mode','rank','n','i','j','branch','split_order'],
              'initial_native_modes': [1,4], 'deferred_native_modes': [2,5],
              'driver_common_contract':common_contract,
              'explicit_full_pole_bypass_contract':bypass,
              'compiled': False, 'native_executions': 0,
              'unknown_model_metadata_remains_unknown': True}
    save(destination/'derivation.json', report)
    return report
