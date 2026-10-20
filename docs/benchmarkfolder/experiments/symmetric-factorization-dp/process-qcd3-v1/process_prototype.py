#!/usr/bin/env python3
"""Private DP transformation of inventoried adjacent full-updater calls only."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path
import re
import tempfile

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = next(p for p in HERE.parents if (p/'madgraph/__init__.py').is_file())
sys.path[:0] = [str(BASE), str(ROOT)]
import factor
from madgraph.iolibs.file_writers import FortranWriter
from madgraph.various.q_polynomial import FortranPolynomialRoutines


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render(source):
    with tempfile.TemporaryDirectory(prefix='dp-factor-render-', dir=ROOT/'.ninjars-build/tmp') as tmp:
        path = Path(tmp)/'body.f'
        with FortranWriter(str(path)) as writer:
            writer.writelines(source)
        return path.read_text()


def logical(text):
    result = []
    for index, line in enumerate(text.splitlines(True)):
        if not line.strip() or line[0] in 'cC*!':
            continue
        body = re.sub(r'\s+', '', line[6:]).upper()
        if line[5] not in ' 0':
            assert result
            result[-1][2] += body
        else:
            result.append([index, line[:6], body])
    return result


def transform(directory, variant):
    assert variant in ('capture-only', 'factor-trace')
    inventory = json.loads((BASE/'process-plan-v1/inventory-v3.json').read_text())
    directory = Path(directory)
    original = (directory/'polynomial.f').read_text()
    prefix = re.search(r'(?m)^      MODULE (\w+)POLYNOMIAL_CONSTANTS$', original)[1]
    gen = FortranPolynomialRoutines(4, updater_max_rank=1, proc_prefix=prefix,
                                   sub_prefix=prefix, coef_format='complex*16')
    helpers = sorted({(e['vertex'], e['rank']) for e in inventory['calls']})
    assert helpers == [('FFV1L1_2', 2), ('FFV1L1_2', 3), ('FFV1L2_1', 2), ('FFV1L2_1', 3)]
    replacements, counts, files = [], {}, {}
    for path in sorted(directory.glob('coef_construction_*.f')):
        entries = [e for e in inventory['calls'] if e['file'] == path.name]
        if not entries:
            continue
        text = path.read_text()
        assert 'IEEE_' not in text
        lines = text.splitlines(True)
        rows = {index + 1: (label, row) for index, label, row in logical(text)}
        edits = []
        for entry in entries:
            index = entry['line'] - 1
            label, row = rows[entry['line']]
            assert row == entry['original_statement']
            end = index + 1
            while end < len(lines) and lines[end].strip() and lines[end][0] not in 'cC*!':
                if len(lines[end]) < 6 or lines[end][5] in ' 0':
                    break
                end += 1
            helper = helpers.index((entry['vertex'], entry['rank'])) + 1
            query = 'CALL IEEE_GET_FLAG(IEEE_INEXACT,DP_ENTRY_INEXACT)\n'
            if variant == 'factor-trace':
                new_name = entry['updater'] + '_DP_FACTOR_' + entry['vertex']
                assert len(new_name) <= 63
                call = 'CALL ' + row[4:].replace(entry['updater'], new_name, 1)
                call = call[:-1] + ',DP_ENTRY_INEXACT)'
            else:
                new_name = entry['updater']
                query += (f'DP_CAPTURE_COUNTS(1,{helper})=DP_CAPTURE_COUNTS(1,{helper})+1\n'
                          f'IF(DP_ENTRY_INEXACT) DP_CAPTURE_COUNTS(2,{helper})=DP_CAPTURE_COUNTS(2,{helper})+1\n')
                call = 'CALL ' + row[4:]
            replacement = render(query + call).splitlines(True)
            replacement[0] = label[:5] + replacement[0][5:]
            edits.append((index, end, ''.join(replacement)))
            replacements.append({**entry, 'replacement': new_name, 'helper': helper})
        for start, end, replacement in reversed(edits):
            lines[start:end] = [replacement]
        updated = ''.join(lines)
        assert updated.count('      IMPLICIT NONE') == 1
        updated = updated.replace('      IMPLICIT NONE',
            '      USE, INTRINSIC :: IEEE_EXCEPTIONS\n      IMPLICIT NONE\n'
            '      LOGICAL DP_ENTRY_INEXACT\n' +
            ('      INTEGER*8 DP_CAPTURE_COUNTS(2,4)\n'
             '      COMMON/DP_FACTOR_CAPTURE/DP_CAPTURE_COUNTS\n' if variant == 'capture-only' else ''), 1)
        files[path.name] = updated
    assert len(replacements) == 136
    if variant == 'factor-trace':
        appended = []
        for index, (vertex, rank) in enumerate(helpers, 1):
            public = render(gen.write_compact_wl_updater(rank, 1)).strip()
            assert public in original, (rank, 'public DP fallback differs')
            body = factor.generate(gen, rank, vertex, suffix='_DP_FACTOR_' + vertex)
            body = body.replace('  implicit none', '  implicit none\n'
                '  INTEGER*8 DP_FACTOR_COUNTS(3,4)\n'
                '  COMMON/DP_FACTOR_TRACE/DP_FACTOR_COUNTS', 1)
            marker = 'C Experimental binary64 derivative-row factorization'
            assert body.count(marker) == 1
            body = body.replace(marker, marker +
                f'\nDP_FACTOR_COUNTS(1,{index})=DP_FACTOR_COUNTS(1,{index})+1', 1)
            body = body.replace('C FACTOR_FAST_PATH', 'C FACTOR_FAST_PATH' +
                f'\nDP_FACTOR_COUNTS(2,{index})=DP_FACTOR_COUNTS(2,{index})+1', 1)
            body, count = re.subn(r'(?m)^(\s*CALL ' + re.escape(prefix) + r'UPDATE_WL_\d_1\(.*)$',
                rf'  DP_FACTOR_COUNTS(3,{index})=DP_FACTOR_COUNTS(3,{index})+1\n\1', body)
            assert count >= 5
            appended.append(render(body))
        files['polynomial.f'] = original + '\n' + '\n'.join(appended)
        assert files['polynomial.f'].startswith(original)
    driver = (directory/'check_sa.f').read_text()
    name, common, extent = (('DP_FACTOR_COUNTS', 'DP_FACTOR_TRACE', 3)
                           if variant == 'factor-trace' else ('DP_CAPTURE_COUNTS', 'DP_FACTOR_CAPTURE', 2))
    driver = driver.replace('      IMPLICIT NONE', '      IMPLICIT NONE\n'
        f'      INTEGER*8 {name}({extent},4)\n'
        f'      COMMON/{common}/{name}\n'
        f'      DATA {name}/{extent*4}*0/\n', 1)
    marker = "      OPEN(6,FILE='/dev/null',STATUS='OLD')"
    assert driver.count(marker) == 1
    driver = driver.replace(marker, marker + '\n      ' + name + '=0', 1)
    marker = '      ELAPSED=DBLE(T1-T0)/DBLE(RATE)'
    assert driver.count(marker) == 1
    driver = driver.replace(marker, marker +
        "\n      OPEN(74,FILE='dp_factor_counts.dat',STATUS='REPLACE')\n"
        '      DO I=1,4\n' + f"         WRITE(74,'({extent+1}I20)') I,{name}(:,I)\n" +
        '      ENDDO\n      CLOSE(74)', 1)
    files['check_sa.f'] = driver
    return files, dict(variant=variant, helpers=[dict(index=i+1, vertex=v, rank=r) for i,(v,r) in enumerate(helpers)],
        calls=replacements, helper_factor_sha256=sha(BASE/'factor.py'),
        inventory_sha256=sha(BASE/'process-plan-v1/inventory-v3.json'),
        public_polynomial_preserved_as_prefix=True, changed_association=variant == 'factor-trace',
        new_batch_ieee_boundary=True, original_qp_sources_unchanged=True)
