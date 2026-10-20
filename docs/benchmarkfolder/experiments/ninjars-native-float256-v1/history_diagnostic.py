#!/usr/bin/env python3
"""Retain exact changed-history cases and input-order identities; no engines."""
import sys
sys.dont_write_bytecode = True
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def main():
    replay_path = HERE/'run-replay/raw.jsonl'
    convergence_path = HERE/'run-convergence/raw.jsonl'
    replay = [json.loads(line) for line in replay_path.read_text().splitlines()]
    convergence = [json.loads(line) for line in convergence_path.read_text().splitlines()]
    assert replay[-1]['kind'] == convergence[-1]['kind'] == 'summary'
    replay = replay[:-1]
    convergence = convergence[:-1]
    selected = [row for row in convergence if row['working_bits'] == 256]
    assert len(replay) == 1392 and len(selected) == 959
    changed = []
    for row in selected:
        sequence = row['sequence']
        if not isinstance(sequence, int): continue
        baseline = replay[sequence]
        assert baseline['input'] == row['input']
        if any(a['raw'] != b['raw'] for a,b in zip(baseline['arms'],row['arms'])):
            high = next(r for r in convergence if r['sequence'] == sequence and r['working_bits'] == 384)
            changed.append({'sequence':sequence,
                'input_canonical_sha256':hashlib.sha256(canonical(row['input'])).hexdigest(),
                'full_history':baseline,'selected_history_256':row,'selected_history_384':high})
    assert len(changed) == 20
    print(json.dumps({'complete':True,'engine_calls':0,'performance_claim':False,
        'script_sha256':sha(Path(__file__)),
        'replay_raw_sha256':sha(replay_path),'convergence_raw_sha256':sha(convergence_path),
        'full_history_order':'all1392 requests in original capture order, one reused state per policy at256bits',
        'full_history_input_order_canonical_sha256':hashlib.sha256(canonical([r['input'] for r in replay])).hexdigest(),
        'selected_history_order':'all958original replay differences in ascending original sequence order, then physicalcomplexmassfixture; fresh state per policy/precision, retaining state through that selected stream',
        'selected_history_input_order_canonical_sha256':hashlib.sha256(canonical([r['input'] for r in selected])).hexdigest(),
        'selected_sequence_order':[r['sequence'] for r in selected],
        'raw_component_order':['finite.re','finite.im','single.re','single.im','double.re','double.im','CC.re','CC.im','rational.re','rational.im'],
        'changed_sequences':[r['sequence'] for r in changed],
        'assessment':'Unexplained tiny rawfinite/CC history dependence common to all policies. ASCII/status/trackedprecision/sign exact; do not claim exact256bitcorrectrounding or historyindependence.',
        'cases':changed},indent=2))


if __name__ == '__main__': main()
