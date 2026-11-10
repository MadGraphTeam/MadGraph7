#!/usr/bin/env python3
"""Prepare one private QCD QP-classification source variant; never build/run."""
from collections import Counter
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
DESIGN = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer'
NATIVE = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-observer-native'
ROUTING = ROOT / 'docs/benchmarkfolder/experiments/qcd-loop-kinematics-trace/design/topology'
PROCESS = ROOT / '.tirust-build/physical-stress/outputs/candidate/qcd_n3_uux_zgg'
OUT = HERE / 'topology-design'
INDEX_PINS = {
    DESIGN: '03dcf0b638a32edbd8362fa8cf6d12148250f276f095a53b7579e392aae4ad90',
    NATIVE: '6a2e48f62ee48ebd2a892da18f797249e686be660225e9d05c66415f98e8f5ac',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replace_once(source, old, new):
    require(source.count(old) == 1, ('unique replacement', old, source.count(old)))
    return source.replace(old, new)


def routine(source, name):
    begin = source.index('      SUBROUTINE ' + name + '(')
    end = source.index('\n      END\n', begin) + len('\n      END\n')
    return source[begin:end]


def fixed(statement):
    """Format a fixed-form statement; these statements contain no string literals."""
    rows = []
    prefix = '      '
    while len(prefix + statement) > 120:
        cut = statement.rfind(',', 0, 114)
        require(cut >= 0, ('unwrappable statement', statement))
        rows.append(prefix + statement[:cut + 1])
        statement = statement[cut + 1:]
        prefix = '     $ '
    rows.append(prefix + statement)
    return '\n'.join(rows) + '\n'


def replace_statement(source, prefix, transform, expected=1):
    pattern = re.compile(r'^ {6,}' + re.escape(prefix) + r'[^\n]*\n(?:^     \$[^\n]*\n)*', re.M)
    matches = list(pattern.finditer(source))
    require(len(matches) == expected, ('statement count', prefix, len(matches), expected))
    for match in reversed(matches):
        statement = ''.join(row[6:].strip() for row in match[0].splitlines())
        source = source[:match.start()] + fixed(transform(statement)) + source[match.end():]
    return source


def append_context(statement, old_name, new_name, context):
    require(statement.count(old_name + '(') == 1 and statement.endswith(')'), statement)
    return statement.replace(old_name + '(', new_name + '(', 1)[:-1] + ',' + context + ')'


def table_source(modes):
    counts = {}
    nlines = [g['n'] for g in modes['4']]
    lines = ['C     Generated once from the pinned HELAS producer graph. No real arithmetic.',
             'C     0=composite/unknown, +leg=direct, -leg=conservation complement.',
             'C     All five external legs in this retained QCD process are on shell.',
             '      INTEGER FUNCTION QK_PAIR_CLASS(GROUP,MODE,N,IP,JP)',
             '      IMPLICIT NONE', '      INTEGER GROUP,MODE,N,IP,JP,I,J,D',
             '      INTEGER N_LINES(116),CLASSES(6,6,116,2)',
             fixed('DATA N_LINES/' + ','.join(map(str, nlines)) + '/').rstrip()]
    for direction, mode in enumerate(('4', '5'), 1):
        tally = Counter()
        for group in modes[mode]:
            matrix = [[0] * 6 for _ in range(6)]
            for pair in group['pairs']:
                kind, leg = pair['classification'], pair['single_leg']
                if kind == 'direct_single_leg':
                    value = leg
                elif kind == 'complement_of_single_leg':
                    value = -leg
                else:
                    require(kind == 'composite' and leg is None, kind)
                    value = 0
                i, j = pair['propagators']
                matrix[i-1][j-1] = matrix[j-1][i-1] = value
                tally[kind] += 1
            values = [matrix[i][j] for j in range(6) for i in range(6)]
            lines.append(fixed('DATA ((CLASSES(I,J,%d,%d),I=1,6),J=1,6)/%s/' %
                               (group['group'], direction, ','.join(map(str, values)))).rstrip())
        counts[mode] = dict(tally)
    lines += ['      QK_PAIR_CLASS=0', '      IF(GROUP.LT.1.OR.GROUP.GT.116) RETURN',
              '      IF(N.NE.N_LINES(GROUP)) RETURN',
              '      IF(IP.LT.1.OR.IP.GT.N.OR.JP.LT.1.OR.JP.GT.N) RETURN',
              '      IF(IP.EQ.JP) RETURN', '      IF(MODE.EQ.4) THEN', '        D=1',
              '      ELSEIF(MODE.EQ.5) THEN', '        D=2', '      ELSE', '        RETURN',
              '      ENDIF', '      QK_PAIR_CLASS=CLASSES(IP,JP,GROUP,D)', '      END', '']
    return '\n'.join(lines), counts


def classified_builder(original):
    source = routine(original, 'ML5_0_MP_BUILD_KINEMATIC_MATRIX')
    source = replace_statement(source, 'SUBROUTINE ML5_0_MP_BUILD_KINEMATIC_MATRIX(',
        lambda s: append_context(s, 'ML5_0_MP_BUILD_KINEMATIC_MATRIX', 'QK_MP_BUILD_CLASSIFIED',
                                 'GROUP_CONTEXT,MODE_CONTEXT'))
    source = replace_once(source, '      INTEGER NLOOPLINE\n',
        '      INTEGER NLOOPLINE,GROUP_CONTEXT,MODE_CONTEXT\n'
        '      INTEGER QK_PAIR_CLASS\n      LOGICAL SHELL_ELIGIBLE\n'
        '      EXTERNAL QK_PAIR_CLASS\n')
    source = replace_once(source, '            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)\n',
        '            S_MAT(I,J)=DIFFSQ-M2L(I)-M2L(J)\n'
        '            SHELL_ELIGIBLE=QK_PAIR_CLASS(GROUP_CONTEXT,\n'
        '     $       MODE_CONTEXT,NLOOPLINE,I,J).NE.0\n')
    # Wrap each old threshold branch without moving the energy reference or
    # its observer. The old body, thresholds and assignment order stay exact.
    for start in ('            IF(ABS(M2L(I)).NE.0.0E0_16)THEN\n',
                  '            IF(ABS(M2L(J)).NE.0.0E0_16)THEN\n',
                  '            IF(REF_NORMALIZATION.NE.0.0E0_16)THEN\n'):
        begin = source.index(start)
        closing = re.search(r'^            ENDIF\n', source[begin:], re.M)
        require(closing is not None, 'Outer threshold block ending missing')
        end = begin + closing.end()
        source = source[:begin] + '            IF(SHELL_ELIGIBLE) THEN\n' + source[begin:end] + \
                 '            ENDIF\n' + source[end:]
    require(source.count('            IF(SHELL_ELIGIBLE) THEN\n') == 3, 'three branch guards')
    return source


def classified_nrs(original):
    source = routine(original, 'ML5_0_MP_NINJARS_LOOP')
    source = replace_statement(source, 'SUBROUTINE ML5_0_MP_NINJARS_LOOP(',
        lambda s: append_context(s, 'ML5_0_MP_NINJARS_LOOP', 'QK_MP_NINJARS_CLASSIFIED',
                                 'GROUP_CONTEXT,MODE_CONTEXT'))
    source = replace_once(source, '      INTEGER NLOOPLINE,RANK\n',
                          '      INTEGER NLOOPLINE,RANK,GROUP_CONTEXT,MODE_CONTEXT\n')
    source = replace_statement(source, 'CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(',
        lambda s: append_context(s, 'ML5_0_MP_BUILD_KINEMATIC_MATRIX', 'QK_MP_BUILD_CLASSIFIED',
                                 'GROUP_CONTEXT,MODE_CONTEXT'))
    return source


def classified_tirust(original):
    diagnostic = routine(original, 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC')
    diagnostic = replace_statement(diagnostic, 'SUBROUTINE ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC(',
        lambda s: append_context(s, 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC', 'QK_MP_TIRUST_DIAGNOSTIC',
                                 'GROUP_CONTEXT'))
    diagnostic = replace_once(diagnostic, '      INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS\n',
        '      INTEGER CTMODE,NLOOPLINE,RANK,I,J,K,NC,STATUS,GROUP_CONTEXT\n')
    diagnostic = replace_statement(diagnostic, 'CALL ML5_0_MP_BUILD_KINEMATIC_MATRIX(',
        lambda s: append_context(s, 'ML5_0_MP_BUILD_KINEMATIC_MATRIX', 'QK_MP_BUILD_CLASSIFIED',
                                 'GROUP_CONTEXT,CTMODE'))
    wrapper = routine(original, 'ML5_0_MP_TIRUST_LOOP')
    wrapper = replace_statement(wrapper, 'SUBROUTINE ML5_0_MP_TIRUST_LOOP(',
        lambda s: append_context(s, 'ML5_0_MP_TIRUST_LOOP', 'QK_MP_TIRUST_CLASSIFIED',
                                 'GROUP_CONTEXT,MODE_CONTEXT'))
    wrapper = replace_once(wrapper, '      INTEGER NLOOPLINE,RANK,CTMODE,I,J\n',
        '      INTEGER NLOOPLINE,RANK,CTMODE,I,J,GROUP_CONTEXT,MODE_CONTEXT\n')
    wrapper = replace_statement(wrapper, 'CALL ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC(',
        lambda s: append_context(s.replace('(CTMODE,', '(MODE_CONTEXT,', 1),
                                 'ML5_0_MP_TIRUST_TENSOR_DIAGNOSTIC', 'QK_MP_TIRUST_DIAGNOSTIC',
                                 'GROUP_CONTEXT'))
    return diagnostic + '\n' + wrapper


def main():
    require(ROOT == HERE.parents[3], 'Run from MadGraph root')
    require(not OUT.exists(), 'One-shot preparation; preserve prior attempts')
    inputs = {}
    for directory, digest in INDEX_PINS.items():
        require(sha(directory / 'evidence-index.json') == digest, 'Frozen inventory changed')
        inputs[str((directory / 'evidence-index.json').relative_to(ROOT))] = digest
        for name, entry in json.loads((directory / 'evidence-index.json').read_text())['files'].items():
            path = directory / name
            require(sha(path) == entry['sha256'] and path.stat().st_size == entry['bytes'], str(path))
    graph = load_module(ROUTING / 'audit_routing.py', 'frozen_qcd_graph')
    graph_result = graph.audit(PROCESS)
    require(graph_result == json.loads((ROUTING / 'routing.json').read_text()), 'Source graph proof changed')
    modes_module = load_module(DESIGN / 'derive_routing.py', 'frozen_qcd_directions')
    modes = modes_module.derive()
    require(modes == json.loads((DESIGN / 'routing-modes.json').read_text()), 'Direction proof changed')
    card = PROCESS / 'Cards/proc_card_mg5.dat'
    generation = [s for s in card.read_text().splitlines() if s.startswith('generate ')]
    require(generation == ['generate u u~ > z g g [virt=QCD]'], 'Wrong retained process/off-shell policy')
    require('set complex_mass_scheme False\n' in card.read_text(), 'Wrong mass scheme')
    for path in (ROUTING / 'audit_routing.py', ROUTING / 'routing.json', DESIGN / 'derive_routing.py',
                 DESIGN / 'routing-modes.json', card):
        inputs[str(path.relative_to(ROOT))] = sha(path)
    ct = (DESIGN / 'sources/CT_interface.f').read_text()
    tir = (DESIGN / 'sources/TIRUST_interface.f').read_text()
    changed_ct = ct
    for old, new in (('ML5_0_MP_NINJARS_LOOP', 'QK_MP_NINJARS_CLASSIFIED'),
                     ('ML5_0_MP_TIRUST_LOOP', 'QK_MP_TIRUST_CLASSIFIED')):
        changed_ct = replace_statement(changed_ct, 'CALL ' + old + '(',
            lambda s, old=old, new=new: append_context(s, old, new, 'LOOPNUM,CTMODE'), expected=7)
    # All old entry points and DP routines remain byte-identical; only the
    # actual generated QP8/9 call sites select the new private routines.
    ct_appendix = classified_builder(ct) + '\n' + classified_nrs(ct)
    tir_appendix = classified_tirust(tir)
    table, counts = table_source(modes['groups_by_mode'])
    outputs = {
        'CT_interface.f': changed_ct + '\nC     BEGIN PRIVATE CLASSIFIED APPENDIX\n' + ct_appendix,
        'TIRUST_interface.f': tir + '\nC     BEGIN PRIVATE CLASSIFIED APPENDIX\n' + tir_appendix,
        'shell_classes.f': table,
    }
    for name, text in outputs.items():
        require(max(map(len, text.splitlines())) <= 132, ('fixed form overflow', name))
    OUT.mkdir(parents=True)
    for name, text in outputs.items():
        (OUT / name).write_text(text)
    for name, original in (('CT_interface.f', ct), ('TIRUST_interface.f', tir)):
        (OUT / (name + '.patch')).write_text(''.join(difflib.unified_diff(original.splitlines(True),
            outputs[name].splitlines(True), fromfile='frozen-observer/' + name, tofile='private-classified/' + name)))
    table_metadata = {'scope': 'Retained QCD only; static producer-DAG identities, not floating matching.',
                      'external_legs': [{'leg': i, 'offshell': False, 'mass_kind': 'Z' if i == 3 else 'massless'}
                                        for i in range(1, 6)],
                      'counts': counts, 'groups_by_mode': modes['groups_by_mode'],
                      'unknown_policy': 'QK_PAIR_CLASS returns 0; keep DIFFSQ. Any unknown actual replay context disqualifies observation.'}
    (OUT / 'table-metadata.json').write_text(json.dumps(table_metadata, indent=2, sort_keys=True) + '\n')
    record = {'schema': 1, 'compiler_executed': False, 'native_executed': False,
              'inputs': inputs, 'producer_graph_sources': graph_result['sources'],
              'graph_summary': graph_result['summary'], 'direction_summary': modes['summary'],
              'changes': ['14 generated QP call sites pass explicit LOOPNUM/CTMODE to private routines',
                          'Private helper gates the three original threshold branches by static eligibility',
                          'No DP entry point or standalone original diagnostic is changed',
                          'REF_NORMALIZATION and unconditional observer events remain outside eligibility guards; clamp events remain inside their branches'],
              'private_entry_points': ['QK_MP_NINJARS_CLASSIFIED', 'QK_MP_TIRUST_CLASSIFIED',
                                       'QK_MP_TIRUST_DIAGNOSTIC', 'QK_MP_BUILD_CLASSIFIED', 'QK_PAIR_CLASS'],
              'outputs': {name: {'sha256': sha(OUT / name), 'bytes': (OUT / name).stat().st_size}
                          for name in outputs}, 'script_sha256': sha(Path(__file__))}
    (OUT / 'prepared.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'source_prepared': True, 'compiler_executed': False, 'native_executed': False,
                      'groups': 116, 'pairs_per_direction': 477, 'explicit_qp_call_sites': 14}, sort_keys=True))


if __name__ == '__main__':
    main()
