#!/usr/bin/env python3
"""Recheck the narrowly reviewed fixture updates against their checkpoint."""
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess

if not __debug__:
    raise RuntimeError('fixture review requires Python assertions enabled')

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def statements(text):
    result = []
    for line in text.splitlines():
        if not line.strip() or line[0] in 'Cc*!':
            continue
        if len(line) > 5 and line[5] not in ' 0':
            result[-1] += line[6:].replace(' ', '')
        else:
            result.append(line[6:].replace(' ', ''))
    return result


def verify():
    review = json.loads((HERE / 'fixture-review.json').read_text())
    assert review['updated_fixtures'] == 22
    assert review['unchanged_whitespace_only_fixtures'] == 4
    payloads = {}
    for row in review['rows']:
        path = ROOT / row['fixture']
        before = subprocess.check_output(
            ['git', 'show', f"{review['checkpoint']}:{row['fixture']}"], cwd=ROOT)
        after = path.read_bytes()
        assert digest(before) == row['old_sha256'], row['fixture']
        expected = row['actual_sha256'] if row['applied'] else row['old_sha256']
        assert digest(after) == expected, row['fixture']
        if not row['applied']:
            continue
        old_lines, new_lines = before.decode().splitlines(True), after.decode().splitlines(True)
        opcodes = [(tag, i2-i1, j2-j1) for tag, i1, i2, j1, j2 in
                   difflib.SequenceMatcher(a=old_lines, b=new_lines, autojunk=False).get_opcodes()
                   if tag != 'equal']
        assert [list(item) for item in opcodes] == row['opcodes'], row['fixture']
        payload = '\n'.join(line.rstrip('\n') for line in difflib.unified_diff(old_lines, new_lines)
                            if line[:1] in '+-' and not line.startswith(('+++', '---')))
        assert digest(payload.encode()) == row['changed_payload_sha256'], row['fixture']
        payloads.setdefault(row['diagnostic'], []).append(row)
        if row['diagnostic'] == 'fks-absent' and path.name.endswith('%loop_matrix.f'):
            text = after.decode()
            assert 'DATA LOOPLIBS_QPAVAILABLE(8)/.FALSE./' in text
            assert 'NRS_RETRY_MODE' not in text and 'NINJARS_FINALIZE' not in text
    assert len(review['rows']) == 26
    assert len(payloads['loop']) == 10 and len(payloads['fks-absent']) == 12
    for name in ('TIR_interface.f', 'loop_matrix.f', 'mp_compute_loop_coefs.f'):
        family = [row for row in payloads['fks-absent'] if row['fixture'].endswith('%'+name)]
        assert len(family) == 4 and len({row['changed_payload_sha256'] for row in family}) == 1
    relative = 'tests/input_files/IOTestsComparison/short_ML_SMQCD_optimized/gg_ttx/mp_coef_construction_1.f'
    old = statements(subprocess.check_output(
        ['git', 'show', f"{review['checkpoint']}:{relative}"], cwd=ROOT, text=True))
    new = statements((ROOT / relative).read_text())
    restored, factors, capture = [], 0, 0
    for index, line in enumerate(new):
        factors += '_FACTOR_FFV' in line
        if (line == 'CALLIEEE_GET_FLAG(IEEE_INEXACT,WL_ENTRY_INEXACT)' and index+1 < len(new)
                and new[index+1].startswith('CALLMP_ML5_0_UPDATE_WL_2_1_FACTOR_')):
            capture += 1
            continue
        line = re.sub(r'_FACTOR_FFV1L[12]_[12]', '', line)
        if line.startswith('CALLMP_ML5_0_UPDATE_WL_2_1('):
            line = re.sub(r',WL_ENTRY_INEXACT\)$', ')', line)
        restored.append(line)
    assert factors == 7 and capture == 1 and restored == old
    return {'updated_fixtures': 22, 'unchanged_blank_line_only': 4,
            'all_hashes_and_difference_families': 'pass',
            'coefficient_construction_legacy_statements': 'exact except reviewed substitutions'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
