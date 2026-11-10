#!/usr/bin/env python3
"""Render and compare source only; write outputs solely in this private folder."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines

TEMPLATE = 'madgraph/iolibs/template_files/loop_optimized/TIRUST_interface.inc'
TEST = 'tests/unit_tests/loop/test_tirust_fortran.py'
PINS = {
    TEMPLATE: 'ae18cc526313dac260be88c40d3ac92210687a576f94f32f1bf42a5311afa9ec',
    TEST: 'f278c9609724165591f92d22a83284a2e1a9203b79ac07f78f24a91e076859ee',
    str((HERE.parent / 'TIRUST_interface.f').relative_to(ROOT)):
        '177a7973ea9182978097d5256fab0f0c93c1d4c799919203f7855631bee294ad',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(source):
    """Join fixed-form continuations; ignore comments and unquoted whitespace."""
    statements = []
    for line in source.splitlines():
        if not line.strip() or line[0] in 'Cc*!':
            continue
        if '\t' in line or len(line) > 132 or len(line) < 6:
            raise ValueError('Unsupported fixed-form layout')
        if line[5] not in ' 0':
            if not statements:
                raise ValueError('Orphan continuation')
            statements[-1] += line[6:]
        else:
            statements.append(line[:5].strip() + line[6:])
    result = []
    for statement in statements:
        quote = None
        compact = []
        for char in statement:
            if quote:
                compact.append(char)
                if char == quote:
                    quote = None
            elif char in "'\"":
                quote = char
                compact.append(char)
            elif char == '!':
                break
            elif not char.isspace():
                compact.append(char.upper())
        if quote:
            raise ValueError('Unterminated string')
        result.append(''.join(compact))
    return result


def main():
    originals = {}
    for name, expected in PINS.items():
        data = (ROOT / name).read_bytes()
        if sha(data) != expected:
            raise ValueError('Frozen input changed: ' + name)
        originals[name] = data.decode('ascii')
    qualified = originals[next(reversed(PINS))]
    proposed = (HERE / 'TIRUST_interface.inc').read_text()
    restored = proposed
    blocks = []
    for size in (8, 16):
        start = qualified.index('C     Private v2: integer exponent')
        if size == 16:
            start = qualified.index('C     Private v2: integer exponent', start + 1)
        end = qualified.index(f'      REAL*{size} PL(', start)
        decl = qualified[start:end]
        start = qualified.index('C     Private v2: the ABI', end)
        body = qualified[start:qualified.index('      SCALE=', start)]
        for block in (decl, body):
            cleaned = block.replace('Private v2: ', '').replace('this private source', 'this source')
            if cleaned not in restored:
                raise ValueError('Qualified insertion absent')
            restored = restored.replace(cleaned, '', 1)
            blocks.append({'bytes': len(cleaned.encode()), 'sha256': sha(cleaned.encode())})
    if restored != originals[TEMPLATE] or 'Private v2' in proposed or 'this private source' in proposed:
        raise ValueError('Proposal changed original code or retained private wording')
    test = (HERE / 'test_tirust_fortran.py').read_text()
    ast.parse(test)
    registrations = qualified[qualified.index('      CALL TIRUST_REGISTER_PLAN('):
                              qualified.index('      LAST_GENERATION=TIRUST_CONTEXT_GENERATION()')]
    if registrations.count('CALL TIRUST_REGISTER_PLAN(') != 13:
        raise ValueError('Unexpected registration plan')
    permutation = FortranPolynomialRoutines(3).get_TIRUST_mapping()
    if len(permutation) != 35:
        raise ValueError('Unexpected coefficient mapping size')
    mapping = '\n'.join('DATA (TIRUST_MAP(I),I=%d,%d)/%s/' % (
        start, min(start + 9, 34), ','.join(map(str, permutation[start:start + 10])))
        for start in range(0, 35, 10))
    replacements = dict(proc_prefix='ML5_0_', nloop_groups=200, nSquaredSO=1, maxrank=3,
        real_dp_format='REAL*8', complex_dp_format='COMPLEX*16', real_mp_format='REAL*16',
        complex_mp_format='COMPLEX*32', mass_dp_format='COMPLEX*16', include_vector='',
        tirust_coefmap=mapping, tirust_plan_registration=registrations)
    rendered_path = HERE / 'rendered-TIRUST_interface.f'
    with FortranWriter(str(rendered_path), 'x') as writer:
        writer.writelines(proposed % replacements, context={
            'AmplitudeReduction': True, 'tirust_available': True,
            'tirust_supports_quad_prec': True})
    rendered = rendered_path.read_text()
    expected, actual = canonical(qualified), canonical(rendered)
    if actual != expected:
        raise ValueError('Rendered source differs from the qualified adapter')
    patch = ''.join(''.join(difflib.unified_diff(originals[name].splitlines(True), text.splitlines(True),
                     fromfile='a/' + name, tofile='b/' + name))
                    for name, text in ((TEMPLATE, proposed), (TEST, test)))
    with (HERE / 'promotion.patch').open('x') as output:
        output.write(patch)
    receipt = {
        'schema': 'ew-invariant-v2-production-source-comparison-v1',
        'live_before_sha256': {name: PINS[name] for name in (TEMPLATE, TEST)},
        'qualified_source_sha256': sha(qualified.encode()),
        'proposed_template_sha256': sha(proposed.encode()),
        'proposed_tests_sha256': sha(test.encode()),
        'rendered_sha256': sha(rendered.encode()),
        'patch_sha256': sha(patch.encode()),
        'comparison_script_sha256': sha(Path(__file__).read_bytes()),
        'inserted_blocks': blocks,
        'original_template_recovered_by_removing_four_exact_blocks': True,
        'canonical_statement_count': len(actual),
        'canonical_statements_sha256': sha(('\n'.join(actual) + '\n').encode()),
        'qualified_canonical_statements_equal': True,
        'registration_call_count': 13,
        'mapping_entries': len(permutation),
        'render_replacements': replacements,
        'python_test_source_parsed': True,
        'compiled': False, 'tests_executed': False, 'native_executed': False,
        'live_files_modified': False,
    }
    for name in (TEMPLATE, TEST):
        if sha((ROOT / name).read_bytes()) != PINS[name]:
            raise ValueError('Live source changed during comparison')
    with (HERE / 'source-comparison.json').open('x') as output:
        json.dump(receipt, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps({key: receipt[key] for key in (
        'canonical_statement_count', 'qualified_canonical_statements_equal',
        'proposed_template_sha256', 'proposed_tests_sha256', 'patch_sha256')}, sort_keys=True))


if __name__ == '__main__':
    main()
