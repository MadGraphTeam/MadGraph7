#!/usr/bin/env python3
"""Pure-Python, in-memory mutations of the initial source-scope audit inputs.

No compiler, native executable, reducer, or generated source writer is invoked.
Only the requested JSON evidence file is written. Bytecode writes are disabled.
"""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import re
import sys


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
AUDIT = HERE / 'source-audit.py'
DEFAULT_CONTROL = ROOT / '.tirust-build/dp-terminal-production/outputs/candidate/qcd_n3_uux_zgg'
DEFAULT_CANDIDATE = ROOT / '.tirust-build/dp-higher-rank-production-initial-reference-census/outputs/candidate/qcd_n3_uux_zgg'


def digest(data):
    return sha256(data).hexdigest()


def run(control, candidate):
    audit_sha = digest(AUDIT.read_bytes())
    spec = importlib.util.spec_from_file_location('source_audit_mutation_target', AUDIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = module.audit(control, candidate)
    result = dict(
        schema=1,
        scope='In-memory source-classifier checks only; no numerical or compiler qualification.',
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        runner_sha256=digest(Path(__file__).read_bytes()),
        audit_script=dict(path=str(AUDIT.relative_to(ROOT)), sha256=audit_sha),
        python=dict(executable=sys.executable, version=sys.version),
        baseline=dict(passed=baseline['passed'], errors=baseline['errors'],
                      unchanged_selected_files=baseline['unchanged_selected_files'],
                      classification_counts=baseline['classification_counts']),
        input_snapshots={side: dict(
            root=str(root), manifest_sha256=baseline[side+'_manifest_sha256'],
            selected_file_count=len(baseline[side+'_inventory']))
            for side, root in (('control', control), ('candidate', candidate))},
        build_configuration_selection={name: module.selected(Path(name))
                                       for name in ('GNUmakefile', 'flags.mk')},
        selected_inputs={}, checks=[], errors=[], passed=False)
    if not baseline['passed']:
        result['errors'].append('Unmodified initial source pair failed the audit')
        return result

    def selected_input(classification):
        rel = next(item['path'] for item in baseline['changed']
                   if item['classification'] == classification)
        left, right = (control/rel).read_bytes(), (candidate/rel).read_bytes()
        result['selected_inputs'][rel] = dict(control_sha256=digest(left),
                                             candidate_sha256=digest(right))
        return rel, left, right

    def check(name, function, left, right, expected, **extra):
        item = dict(name=name, expected=expected, control_sha256=digest(left),
                    tested_candidate_sha256=digest(right), **extra)
        try:
            function(left, right)
            item['observed'] = 'accepted'
        except module.Refusal as error:
            item.update(observed='rejected', refusal=str(error))
        except Exception as error:
            item.update(observed='unexpected_error', error=type(error).__name__+': '+str(error))
        item['passed'] = item['observed'] == expected
        result['checks'].append(item)

    rel, left, right = selected_input('declaration_line_permutation_only')
    # Also establish that a byte-identical original is accepted by this classifier.
    module.compare_dhelas(left, left)
    check('unmodified DHELAS pair and identical original', module.compare_dhelas,
          left, right, 'accepted', path=rel, identical_original_accepted=True)
    _, declarations, body, start, end = module.declaration_parts(right)
    line = next(value for value in declarations if value.startswith(b'      REAL*'))
    replacement = (line.replace(b'REAL*8', b'REAL*16') if b'REAL*8' in line
                   else line.replace(b'REAL*16', b'REAL*8'))
    changed = right[:start] + b''.join(
        replacement if value == line else value for value in declarations) + right[end:]
    check('declaration type change', module.compare_dhelas, left, changed, 'rejected', path=rel)
    assert b' = +' in body
    changed = right[:end] + body.replace(b' = +', b' = -', 1)
    check('executable byte change', module.compare_dhelas, left, changed, 'rejected', path=rel)
    include = declarations.index(b"      INCLUDE 'coef_specs.inc'\n")
    changed_declarations = list(declarations)
    changed_declarations[include], changed_declarations[include+1] = (
        changed_declarations[include+1], changed_declarations[include])
    changed = right[:start] + b''.join(changed_declarations) + right[end:]
    check('include/COEFF dependency reorder', module.compare_dhelas,
          left, changed, 'rejected', path=rel)

    rel, left, right = selected_input('two_private_helpers_added_only')
    prefix = module.compare_polynomial(left, right)['prefix'].encode()
    check('unmodified polynomial pair', module.compare_polynomial,
          left, right, 'accepted', path=rel)
    for suffix, old, new, label in (
        (b'DP_TERMINAL_WL_1_1', b'IF(.NOT.WL_ENTRY_INEXACT)', b'IF(WL_ENTRY_INEXACT)',
         'retained 1x1 helper guard change'),
        (b'UPDATE_WL_2_1', b'      DO J=1,OUT_SIZE\n', b'      DO J=1,OUT_SIZE-1\n',
         'public compact loop bound change')):
        block = module.routine(right, prefix+suffix)
        assert old in block.group()
        changed = right[:block.start()] + block.group().replace(old, new, 1) + right[block.end():]
        check(label, module.compare_polynomial, left, changed, 'rejected', path=rel)

    rel, left, right = selected_input('immediate_query_and_private_call_substitutions_only')
    check('unmodified caller pair', module.compare_caller, left, right, 'accepted', path=rel)
    blocks = module.fixed_blocks(right)
    position = next(index for index, block in enumerate(blocks)
                    if block == module.GET_FLAG and index+1 < len(blocks)
                    and re.search(rb'DP_TERMINAL_WL_[23]_1', blocks[index+1]))
    changed_blocks = list(blocks)
    assert b'WL_ENTRY_INEXACT)' in changed_blocks[position+1]
    changed_blocks[position+1] = changed_blocks[position+1].replace(
        b'WL_ENTRY_INEXACT)', b'WL_ENTRY_INEXACT,.TRUE.)', 1)
    check('extra private caller argument', module.compare_caller,
          left, b''.join(changed_blocks), 'rejected', path=rel)
    changed_blocks = list(blocks)
    changed_blocks[position] = changed_blocks[position].replace(b'IEEE_INEXACT', b'IEEE_INVALID')
    check('query changes selected IEEE flag', module.compare_caller,
          left, b''.join(changed_blocks), 'rejected', path=rel)
    changed_blocks = blocks[:position+1] + [b'      CONTINUE\n'] + blocks[position+1:]
    check('statement inserted between query and call', module.compare_caller,
          left, b''.join(changed_blocks), 'rejected', path=rel)

    # Recheck the complete selected inventories and directory-link targets.
    # The mutations above existed solely in immutable Python byte strings.
    stable = digest(AUDIT.read_bytes()) == audit_sha
    for side, root in (('control', control), ('candidate', candidate)):
        current, _, links = module.inventory(root)
        stable = stable and current == baseline[side+'_inventory']
        stable = stable and links == baseline['nontraversed_directory_links'][side]
    result['input_snapshots_and_audit_script_stable'] = stable
    if not stable:
        result['errors'].append('Audit script or selected input snapshots changed during checks')
    if not all(result['build_configuration_selection'].values()):
        result['errors'].append('GNUmakefile and .mk files must enter the selected inventory')
    if len(result['checks']) != 11:
        result['errors'].append('Expected exactly 11 retained mutation/control checks')
    result['passed'] = stable and not result['errors'] and all(
        item['passed'] for item in result['checks'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', type=Path, default=DEFAULT_CONTROL)
    parser.add_argument('--candidate', type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument('--output', type=Path, default=HERE/'source-audit-mutations.json')
    args = parser.parse_args()
    result = run(args.control.resolve(), args.candidate.resolve())
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(passed=result['passed'], checks=len(result['checks']),
                         errors=result['errors'], output=str(args.output))))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
