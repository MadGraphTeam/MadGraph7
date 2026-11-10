"""Additive, source-bound classification for future current-source raw pilots.

No files, native libraries or external paths are opened. Supplied receipt bytes
must also be authenticated by the caller's existing worker/source/runtime
admission. Historical source5 parsing remains the default and is unchanged.
"""
import ast
from collections import namedtuple
import hashlib
import json
import math
import re

PILOT_SHA = '22b06cda2303d9d2fd42aed418aa4518db0d9e834678862330ffa0d6600e9eef'
WORKER_SHA = '371f8ccb72a5ee3dc1bad628cc855e239c7d0c26d5f30b5ff652f15afc564cc9'
MANIFEST_SHA = '4a27dda7a0977cd1b0ed0d779b30c403da50968bf494aa96d31cf18fc9fe2076'
TEMPLATE_SHA = '3223f38d2e61243f2b2849d08982d17eadf2a93ae1dc2edaccaa44515467494b'
TEMPLATE_KEY = 'madgraph/iolibs/template_files/loop_optimized/loop_matrix_standalone.inc'
LATCH_REASON = 'Sticky or newly set MadLoop floating-point failure latch'
_Qualification = namedtuple('_Qualification', ('manifest_sha256', 'template_sha256',
    'generation_sha256', 'compilation_sha256', 'generated_matrix_sha256',
    'worker_driver_sha256', 'private_build_sha256', 'private_executable_sha256'))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(blob):
    return hashlib.sha256(blob).hexdigest()


def fixed_statements(text):
    statements = []
    for line in text.splitlines():
        if not line.strip() or line[0] in 'Cc*!':
            continue
        require(len(line) >= 6, 'Malformed generated fixed-format source')
        body = re.sub(r'\s+', '', line[6:].split('!', 1)[0].upper())
        if line[5] not in (' ', '0'):
            require(statements, 'Generated continuation without statement')
            statements[-1] += body
        else:
            statements.append(body)
    return statements


def generated_entry_reset(matrix, prefix, nexternal, template):
    """Require the actual Born matrix's unconditional qualified reset block."""
    require(sha(template) == TEMPLATE_SHA, 'Unqualified reset semantics template')
    statements = fixed_statements(matrix.decode('utf-8'))
    start = 'SUBROUTINE' + prefix.upper() + 'SLOOPMATRIX(P_USER,ANS)'
    require(statements.count(start) == 1, 'Qualified public matrix body is missing or ambiguous')
    body = statements[statements.index(start) + 1:]
    end = next((index for index, statement in enumerate(body) if statement in ('END', 'ENDSUBROUTINE')), None)
    require(end is not None, 'Unterminated generated public matrix body')
    public = body[:end + 1]
    implementation = 'SUBROUTINE' + prefix.upper() + 'SLOOPMATRIX_RETRY_IMPL(P_USER,ANS)'
    if implementation in statements:
        # NinjaRS-enabled exports retain an outer bookkeeping wrapper. Admit
        # only its exact qualified template expansion, with one implementation
        # call and no FPE reset/conditional/early-return additions.
        wrapper = template.decode('utf-8').split('## if(ninjars_available) {', 1)[1]
        wrapper = wrapper.split('\n      SUBROUTINE %(proc_prefix)sSLOOPMATRIX_RETRY_IMPL', 1)[0]
        wrapper = wrapper % {'proc_prefix': prefix, 'nexternal': nexternal, 'real_dp_format': 'REAL*8'}
        require([start] + public == fixed_statements(wrapper), 'Generated public retry wrapper differs')
        require(statements.count(implementation) == 1, 'Ambiguous generated implementation')
        body = statements[statements.index(implementation) + 1:]
        end = next((index for index, statement in enumerate(body) if statement in ('END', 'ENDSUBROUTINE')), None)
        require(end is not None, 'Unterminated generated implementation')
    body = body[:end]
    fragment = ('REF=0.0D0', 'DOI=1,NSQSO_BORN', 'REF=REF+ANS(0,I)', 'ENDDO',
                'FPE_IN_DP_REDUCTION=.FALSE.', 'FPE_IN_QP_REDUCTION=.FALSE.',
                'MP_DONE=.FALSE.', 'MP_DONE_ONCE=.FALSE.', 'MP_PS_SET=.FALSE.',
                'STAB_INDEX=0', 'DOING_QP_EVALS=.FALSE.')
    positions = [index for index in range(len(body) - len(fragment) + 1)
                 if tuple(body[index:index + len(fragment)]) == fragment]
    require(len(positions) == 1, 'Actual generated matrix lacks the qualified entry-reset block')
    stack = []
    for statement in body[:positions[0]]:
        require(statement not in ('END', 'ENDSUBROUTINE'), 'Reset is outside the public matrix body')
        require(not statement.startswith(('SELECT', 'ENDSELECT')) and not re.match(r'^DO[0-9]', statement),
                'Unsupported generated control-flow layout')
        if statement in ('ENDIF', 'ENDDO'):
            kind = 'IF' if statement == 'ENDIF' else 'DO'
            require(stack and stack.pop() == kind, 'Unsupported generated control-flow layout')
        elif statement.startswith('IF(') and statement.endswith('THEN'):
            stack.append('IF')
        elif re.match(r'^DO[A-Z_][A-Z0-9_]*=[^,]+,', statement) or statement.startswith('DOWHILE('):
            stack.append('DO')
    require(not stack, 'Generated entry-reset block is conditional')


def qualify_sources(*, manifest, template, generation, compilation, matrix,
                    worker_template, driver, private_build, matrix_source_key,
                    worker_parameters):
    """Join actual generated/compiled source receipts; no live-template shortcut.

    All arguments containing sources or receipts are byte strings. The private
    build uses the existing source5 prepared-case shape. Its executable digest
    is returned for the caller to join to its actual worker/runtime receipt.
    This does not replace that admission or claim observed execution.
    """
    require(sha(manifest) == MANIFEST_SHA and sha(template) == TEMPLATE_SHA,
            'Only the qualified entry-reset campaign/template is supported')
    require(sha(worker_template) == WORKER_SHA, 'Only the sealed positive-mode source5 worker is supported')
    manifest_value, generated, compiled, private = map(json.loads,
        (manifest, generation, compilation, private_build))
    require(manifest_value['id'] == 'entry-latch-cache-guard-1', 'Campaign identity differs')
    identity = manifest_value['candidate']['exporter_identity']
    require(identity['files'][TEMPLATE_KEY] == TEMPLATE_SHA, 'Manifest does not bind qualified entry resets')
    observed = generated['source_identity']
    require(generated['source'] == 'candidate' and observed['files'] == identity['files']
            and observed['files_sha256'] == identity['files_sha256'], 'Generated source inventory differs')
    campaign = {'id': manifest_value['id'], 'manifest_sha256': MANIFEST_SHA}
    require(generated['campaign_binding'] == compiled['campaign_binding'] == campaign,
            'Generated/compiled campaign ancestry differs')
    require(compiled['full_numerical_rebuild_this_invocation'] is True
            and compiled['numerical_build_from_sources'] is True, 'Current numerical source build is unqualified')
    require(isinstance(matrix_source_key, str) and
            re.fullmatch(r'SubProcesses/P[0-9]+_[A-Za-z0-9_]+/loop_matrix\.f', matrix_source_key),
            'Expected canonical generated loop matrix source key')
    matrix_sha = sha(matrix)
    require(compiled['sources'][matrix_source_key] == matrix_sha,
            'Compiled generated loop matrix differs')
    require(set(worker_parameters) == {'prefix', 'nexternal', 'nborn'} and
            re.fullmatch(r'ML[0-9]+_[0-9]+_', worker_parameters['prefix']) and
            type(worker_parameters['nexternal']) is int and worker_parameters['nexternal'] >= 3 and
            type(worker_parameters['nborn']) is int and worker_parameters['nborn'] > 0,
            'Unqualified worker render parameters')
    require(compiled['worker_render_parameters'] == worker_parameters and
            private['group_layout']['prefix'] == worker_parameters['prefix'],
            'Private worker dimensions/prefix differ from compiled process')
    require(private['donor']['compiled'] == compiled and
            private['donor']['subprocess'] == matrix_source_key.rsplit('/', 1)[0],
            'Private worker belongs to a different compiled subprocess')
    generated_entry_reset(matrix, worker_parameters['prefix'], worker_parameters['nexternal'], template)
    rendered = worker_template.decode('utf-8')
    for token, name in (('@PREFIX@', 'prefix'), ('@NEXTERNAL@', 'nexternal'), ('@NBORN@', 'nborn')):
        rendered = rendered.replace(token, str(worker_parameters[name]))
    require(rendered.encode('utf-8') == driver, 'Actual private driver differs from sealed forced-DP worker')
    require(private['driver_sha256'] == sha(driver) and
            private['group_layout']['layout_source_hashes']['loop_matrix.f'] == matrix_sha,
            'Private driver/matrix source ancestry differs')
    command = private['command']
    require(command['process_started'] is True and command['returncode'] == 0
            and command['error'] is None and command['cleanup']['survivors'] == [],
            'Private driver build did not complete successfully')
    executable = private['executable_sha256']
    require(isinstance(executable, str) and re.fullmatch('[0-9a-f]{64}', executable),
            'Private executable digest is missing')
    return _Qualification(MANIFEST_SHA, TEMPLATE_SHA, sha(generation), sha(compilation),
                          matrix_sha, sha(driver), sha(private_build), executable)


def adjust_group_calls(result, qualification):
    """Ignore only incoming software flags cleared at a qualified public entry."""
    require(type(qualification) is _Qualification and
            qualification.manifest_sha256 == MANIFEST_SHA and
            qualification.template_sha256 == TEMPLATE_SHA, 'Explicit qualified generated-source binding required')
    for call in result['calls'].values():
        before = call['fpe_dp_before'] or call['fpe_qp_before']
        after = call['fpe_dp'] or call['fpe_qp']
        call.update(legacy_source5_eligible=call['eligible'], legacy_source5_reason=call['reason'],
                    entry_reset_source_qualified=True,
                    incoming_software_latch_cleared=False,
                    fpe_dp_set_during_current_call=call['fpe_dp'],
                    fpe_qp_set_during_current_call=call['fpe_qp'],
                    hardware_ieee_state_observed=False)
        if before and not after and not result['parse_issues']:
            reasons = [] if call['reason'] is None else call['reason'].split('; ')
            require(reasons.count(LATCH_REASON) == 1, 'Legacy sticky-latch reason contract differs')
            reasons.remove(LATCH_REASON)
            call.update(reason='; '.join(reasons) or None, eligible=not reasons,
                        incoming_software_latch_cleared=True)
    return result


def parsers(pilot_blob, *, qualification=None):
    """Extract only exact frozen pure functions; never import controller code.

    Without qualification these functions reproduce historical source5 output.
    With qualification the one group wrapper runs BEFORE diagnostics derives
    row eligibility, split-order rejection, fixed outcomes and completeness.
    """
    require(sha(pilot_blob) == PILOT_SHA, 'Historical parser source differs')
    names = ('require', 'safe_json', 'parse_groups', 'parse_diagnostics')
    tree = ast.parse(pilot_blob, filename='frozen-source5/pilot.py')
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    require({node.name for node in nodes} == set(names), 'Frozen pure parser functions missing')
    namespace = {'math': math, 're': re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'frozen-source5/pilot.py', 'exec'), namespace)
    if qualification is not None:
        original = namespace['parse_groups']
        def parse_groups(raw, layout):
            return adjust_group_calls(original(raw, layout), qualification)
        namespace['parse_groups'] = parse_groups
    return {'parse_groups': namespace['parse_groups'], 'parse_diagnostics': namespace['parse_diagnostics']}
