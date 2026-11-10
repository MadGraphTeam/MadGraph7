#!/usr/bin/env python3
"""Read retained text only; no worker calls and no output-file mutation."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def need(value, message):
    if not value:
        raise ValueError(message)


def public(path):
    lines = Path(path).read_text().splitlines()
    need(len(lines) == 13 and lines[0] == 'OCC_PUBLIC 1', 'Public record shape')
    result = []
    for call in range(2):
        block = [line.split() for line in lines[1+6*call:7+6*call]]
        need(block[0][:2] == ['CALL', str(call)] and len(block[0]) == 3, 'Public call order')
        for row, label, size in zip(block[1:], ('INPUT', 'INPUT_AFTER', 'ANS', 'ACC', 'STATE'), (25,25,8,2,22)):
            need(row[0] == label and len(row) == size+1, 'Public field shape')
            if label != 'STATE':
                need(all(len(x) == 16 and all(c in '0123456789ABCDEF' for c in x) for x in row[1:]),
                     'Public bit-word encoding')
        result.append({'call': call, 'status': int(block[0][2]),
                       'inputs': block[1][1:], 'inputs_after': block[2][1:],
                       'answers': block[3][1:], 'accuracies': block[4][1:],
                       'state': list(map(int, block[5][1:]))})
    return result


def occupancy(path, enabled):
    lines = Path(path).read_text().splitlines()
    header = lines[0].split()
    need(header[:3] == ['OCCUPANCY', '1', str(int(enabled))] and len(header) == 6,
         'Observer header')
    count, dropped, faults = map(int, header[3:])
    need(dropped == 0 and faults == 0, 'Incomplete or state-changing observer')
    need(count == len(lines)-1 and (enabled or count == 0), 'Observer row count')
    rows = [list(map(int, line.split())) for line in lines[1:]]
    for i, row in enumerate(rows, 1):
        need(len(row) == 53 and row[0] == i, 'Observer width/sequence')
        need(row[1] in (0, 1, 2) and row[2] in (10, 20, 21), 'Observer call/boundary')
        need(row[31:42] == row[42:53], 'Observer changed IEEE state')
        query = row[13]
        need(query in (0, 1), 'Query marker')
        if query:
            need(row[12] > 0 and row[14:16] == [0, 0], 'Statistics query failed/created context')
            need(row[2] != 10 or row[7] == 0, 'Queried failed evaluation')
            need(all(x >= 0 for x in row[16:31]), 'Negative cache counter')
        else:
            need(row[14:31] == [-999, -999]+[0]*15, 'Unqueried counter fields')
            need(row[12] <= 0 or (row[2] == 10 and row[7] != 0), 'Missed safe boundary')
        if row[2] == 10:
            need((row[3],row[4]) in ((53,1),(53,2),(113,4),(113,5)), 'Precision/direction')
            need(row[8:10] == [0, 1], 'Ordinary full/both-pole mode changed')
            need(row[10] == int(row[4] in (2, 5)), 'Independent check marker')
            need(row[11] == 1, 'Default global cache changed')
        else:
            need(row[3:12] == [0,0,0,0,-999,-999,0,0,0], 'Clear sentinel metadata')
    return rows


def summarize(root):
    root = Path(root)
    records = {arm: public(root/arm/'public.txt') for arm in ('original', 'disabled', 'enabled')}
    need(records['original'] == records['disabled'] == records['enabled'], 'Original/off/on public mismatch')
    occupancy(root/'original/occupancy.txt', False)
    occupancy(root/'disabled/occupancy.txt', False)
    rows = occupancy(root/'enabled/occupancy.txt', True)
    need(rows and any(row[2] == 10 for row in rows), 'No native evaluations observed')
    need([row[1] for row in rows] == sorted(row[1] for row in rows), 'Reordered public-call epochs')
    for call in (0,1):
        epoch = [row for row in rows if row[1] == call]
        need(epoch and epoch[0][2] == 20 and epoch[-1][2] == 21, 'Missing entry/exit clear boundary')
        need(sum(row[2] == 20 for row in epoch) >= 2, 'Missing entry/exit clear pair')
        need(any(row[2] == 10 for row in epoch), 'Public call has no native visits')
    need(all(row[2] != 10 for row in rows if row[1] == 2), 'Native evaluation during exit')
    for i,row in enumerate(rows):
        if row[2] == 20:
            need(i+1 < len(rows) and rows[i+1][2] == 21 and rows[i+1][1] == row[1],
                 'Clear boundaries are not adjacent')
        if row[2] == 21:
            need(i > 0 and rows[i-1][2] == 20 and rows[i-1][1] == row[1],
                 'Clear boundaries are not adjacent')
    queried = [row for row in rows if row[13]]
    # For this full-only run the pole store is unused; QP contributions are
    # subtracted from aggregate slots5..9. These are boundary samples, not RSS.
    dp = []
    for row in queried:
        a, q = row[16:25], row[25:31]
        value = a[:4]+[a[4+i]-q[i] for i in range(5)]
        need(all(x >= 0 for x in value), 'Invalid aggregate-minus-QP counters')
        dp.append(value)
    visits = Counter((row[1], row[3], row[4], row[5], row[6], row[7])
                     for row in rows if row[2] == 10)
    failures = sum(row[2] == 10 and row[7] != 0 for row in rows)
    clears = [row for row in rows if row[2] in (20, 21)]
    need(len(clears) % 2 == 0, 'Unpaired clear boundaries')
    for before, after in zip(clears[::2], clears[1::2]):
        need(before[2] == 20 and after[2] == 21 and before[1] == after[1], 'Clear ordering')
        need(after[13] == 1 and after[16:31] == [0]*15, 'Clear did not expose empty counters')
    return {
        'schema': 1, 'scope': 'QCD n4 ordinary-0000 bootstrap plus one public call; diagnostic only',
        'exact_original_disabled_enabled_parity': True,
        'public_statuses': [row['status'] for row in records['original']],
        'second_call_accepted': records['original'][1]['status'] in (219, 323),
        'boundary_samples': len(rows), 'statistics_samples': len(queried),
        'failed_evaluations_without_statistics': failures, 'existing_clear_calls': len(clears)//2,
        'full_dp_observed_peak': dict(zip(('subsets_reused', 'subsets_computed', 'masters_reused',
          'masters_computed', 'moments_reused', 'moments_computed', 'entries', 'estimated_bytes',
          'not_stored'), [max((row[i] for row in dp), default=0) for i in range(9)])),
        'qp_observed_peak': [max((row[25+i] for row in queried), default=0) for i in range(6)],
        'native_visits': [{'public_call': k[0], 'interface_significand_bits': k[1], 'ctmode': k[2],
          'denominators': k[3], 'rank': k[4], 'status': k[5], 'count': v}
          for k,v in sorted(visits.items())],
        'limitations': ['Boundary peaks are lower bounds; a failed evaluation can clear unseen activity.',
          'Counter maxima are per-clear epoch values, not total call counts or bytes of resident memory.',
          'The full-DP-interface store can include internal DD work; its counters do not identify internal precision.',
          'Clear status/error text is unobserved; a post-clear query can replace an unconsumed clear error.',
          'Insertion refusal is not a numerical failure or proof of avoidable runtime.',
          'No cache policy, numerical tolerance, or precision was changed; no performance acceptance.'],
        'inputs': {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
          for arm in ('original','disabled','enabled') for path in
          (root/arm/'public.txt',root/arm/'occupancy.txt')},
    }


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('runs', type=Path)
    print(json.dumps(summarize(parser.parse_args().runs), indent=2, sort_keys=True))
