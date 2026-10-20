#!/usr/bin/env python3
"""Read-only adjacent temporary-use census; not a fusion implementation."""
import sys
sys.dont_write_bytecode = True
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import analyze

HERE, ROOT = analyze.HERE, analyze.ROOT


def arguments(row):
    match = re.fullmatch(r'CALL(\w+)\((.*)\)', row)
    if not match:
        return None
    result, start, level = [], 0, 0
    for i, char in enumerate(match[2]):
        if char == '(':
            level += 1
        elif char == ')':
            level -= 1
        elif char == ',' and level == 0:
            result.append(match[2][start:i])
            start = i+1
    assert level == 0
    result.append(match[2][start:])
    return match[1], result


def compute():
    region = json.loads((HERE / 'analysis.json').read_text())
    results = {}
    for case, bound in region['cases'].items():
        records = []
        for batch in bound['batches']:
            source = ROOT / batch['file']
            assert analyze.sha(source) == bound['source_files'][batch['file']]
            rows = analyze.logical(source.read_text())
            targets = {entry['line']: entry for group in batch['policies']['two_vertices']['regions'] for entry in group}
            for i, row in enumerate(rows):
                if row['line'] not in targets:
                    continue
                vertex, vertex_args = arguments(rows[i-1]['text'])
                updater, updater_args = arguments(row['text'])
                assert vertex == targets[row['line']]['vertex']
                assert vertex_args[-1] == updater_args[2] == 'COEFS'
                assert updater_args[1] == updater_args[3] == updater_args[4] == '4'
                # The established generated layout uses separate local COEFS,
                # PL and WL arrays. Check distinct WL entry slots explicitly.
                read_wl = re.fullmatch(r'WL\(1,0,1,(\d+)\)', updater_args[0])
                write_wl = re.fullmatch(r'WL\(1,0,1,(\d+)\)', updater_args[5])
                assert read_wl and write_wl and read_wl[1] != write_wl[1]
                outcome, consumers = None, [row['line']]
                for later in rows[i+1:]:
                    call = arguments(later['text'])
                    if call:
                        name, args = call
                        uses = [arg for arg in args if 'COEFS' in arg]
                        if not uses:
                            continue
                        if name in bound['aloha_source_screen'] and bound['aloha_source_screen'][name]['screen_passed'] and args[-1] == 'COEFS':
                            outcome = 'next_generated_vertex_overwrite'
                            break
                        if name in bound['public_updater_bodies'] and args[2] == 'COEFS':
                            consumers.append(later['line'])
                        else:
                            outcome = 'unknown_temporary_use'
                            break
                    elif later['text'] == 'END':
                        outcome = 'routine_end'
                        break
                    elif later['text'].startswith(('GOTO', 'RETURN', 'IF', 'ELSE', 'ENDIF', 'DO', 'ENDDO')) or later['label']:
                        outcome = 'control_flow_barrier'
                        break
                    elif 'COEFS' in later['text']:
                        outcome = 'unknown_temporary_statement'
                        break
                records.append({'file': batch['file'], 'line': row['line'], 'vertex': vertex,
                    'rank': targets[row['line']]['rank'], 'vertex_arguments': vertex_args,
                    'updater_arguments': updater_args, 'consumer_lines': consumers, 'end_reason': outcome,
                    'single_adjacent_consumer_before_overwrite': len(consumers) == 1 and outcome == 'next_generated_vertex_overwrite'})
        stores = {}
        for vertex in sorted(analyze.TARGETS):
            path = analyze.SOURCE / case / 'Source/DHELAS' / (vertex + '.f')
            rows = analyze.logical(path.read_text())
            assignments = []
            for row in rows:
                match = re.fullmatch(r'COEFF\((\d+),(\d+),(\d+)\)=(.*)', row['text'])
                if match:
                    assignments.append((int(match[2]), match[4]))
            assert len(assignments) == 80
            stores[vertex] = {'source_sha256': analyze.sha(path), 'total_complex_assignments': 80,
                'b0_complex_assignments': sum(m == 0 for m, _ in assignments),
                'q0_complex_assignments': sum(m == 1 for m, _ in assignments),
                'q1_q2_q3_complex_assignments': sum(m >= 2 for m, _ in assignments),
                'q1_q2_q3_literal_zero_assignments': sum(m >= 2 and expr == '0D0' for m, expr in assignments),
                'whole_array_clear_statements': sum(bool(re.match(r'COEFF\([^)]*:[^)]*\)=', row['text'])) for row in rows)}
        results[case] = {'targets': len(records), 'single_adjacent_before_overwrite': sum(r['single_adjacent_consumer_before_overwrite'] for r in records),
            'end_reason_counts': dict(Counter(r['end_reason'] for r in records)),
            'extra_consumer_targets': sum(len(r['consumer_lines']) > 1 for r in records),
            'records': records, 'source_assignment_counts': stores}
    return {'schema': 'dp-coefficient-temporary-use-census-v1', 'scope': 'Syntactic generated-source use/slot census. A stored-arithmetic DAG proof and compiler/IEEE safeguards are still required before fusion.',
        'script_sha256': analyze.sha(Path(__file__)), 'region_analysis_sha256': analyze.sha(HERE / 'analysis.json'), 'cases': results}


if __name__ == '__main__':
    value = compute()
    path = HERE / 'consumer-census.json'
    assert not path.exists()
    path.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({case: {k: v for k, v in result.items() if k not in ('records', 'source_assignment_counts')} for case, result in value['cases'].items()}, indent=2))
