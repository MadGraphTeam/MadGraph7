#!/usr/bin/env python3
"""Summarize the private512 experiment and its order/precision controls."""
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPLAY = HERE.parents[1]
ROOT = HERE.parents[6]
D = Decimal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distance(a, b):
    ar, ai, br, bi = map(D, [*a, *b])
    absolute = ((ar - br) ** 2 + (ai - bi) ** 2).sqrt()
    scale = max((ar ** 2 + ai ** 2).sqrt(), (br ** 2 + bi ** 2).sqrt())
    return {'absolute': str(absolute), 'relative': str(absolute / scale if scale else D(0))}


def compare(left, right, engine):
    values = []
    for unit in range(35):
        for slot in range(3):
            if engine == 'NRS':
                a = left['NRS_units'][str(unit)]
                b = right['NRS_units'][str(unit)]
                if a['status'] != 0 or b['status'] != 0:
                    continue
                av, bv = a['results'][str(slot)], b['results'][str(slot)]
            else:
                if left['tir_status'] != 0 or right['tir_status'] != 0:
                    continue
                av = next(r['tir'] for r in left['comparisons'] if r['unit'] == unit and r['slot'] == slot)
                bv = next(r['tir'] for r in right['comparisons'] if r['unit'] == unit and r['slot'] == slot)
            values.append({'unit': unit, 'slot': slot, 'left': av, 'right': bv,
                'identical_printed_tokens': av == bv, 'difference': distance(av, bv)})
    slots = {}
    for slot, name in enumerate(('finite', 'single', 'double')):
        subset = [r for r in values if r['slot'] == slot]
        if subset:
            slots[name] = {metric: max(subset, key=lambda r: D(r['difference'][metric]))
                for metric in ('absolute', 'relative')}
    return {'engine': engine, 'comparison_count': len(values),
        'nonidentical_printed_count': sum(not r['identical_printed_tokens'] for r in values),
        'slot_maxima': slots, 'comparisons': values}


def main():
    assert Path.cwd().resolve() == ROOT
    spec = importlib.util.spec_from_file_location('frozen_tensor_summarizer', REPLAY / 'summarize.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    paths = {'historical_forward': REPLAY, 'historical_reversal': REPLAY / 'reversal',
        'private512_forward': HERE.parent / 'replay-forward',
        'private512_reversal': HERE.parent / 'replay-reversal'}
    sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__)),
        str((REPLAY / 'summarize.py').relative_to(ROOT)): sha(REPLAY / 'summarize.py')}
    data = {}
    metrics = {}
    for label, folder in paths.items():
        for filename in ('summary.json', 'execution.json', 'output.log'):
            path = folder / filename
            sources[str(path.relative_to(ROOT))] = sha(path)
        summary = json.loads((folder / 'summary.json').read_text())
        execution = json.loads((folder / 'execution.json').read_text())
        assert execution['complete'] and execution['returncode'] == 0
        assert execution['protected_inputs_unchanged']
        assert execution['output_sha256'] == sha(folder / 'output.log')
        assert summary['raw_output_sha256'] == sha(folder / 'output.log')
        assert summary['execution_manifest_sha256'] == sha(folder / 'execution.json')
        data[label] = {(r['point'], r['group']): r for r in summary['records']}
        if label.startswith('private512'):
            helper.HERE = folder
            metrics[label] = helper.summarize()
    with localcontext() as context:
        context.prec = 100
        checks = {}
        key = (2, 5)
        checks['TIR512_forward_vs_reversal'] = compare(data['private512_forward'][key], data['private512_reversal'][key], 'TIR')
        checks['NRS256_forward_vs_reversal_in_private512_process'] = compare(data['private512_forward'][key], data['private512_reversal'][key], 'NRS')
        for label in ('forward', 'reversal'):
            for key in data['historical_' + label]:
                check = compare(data['historical_' + label][key], data['private512_' + label][key], 'NRS')
                checks[f'unchanged_NRS_{label}_point{key[0]}_group{key[1]}'] = check
    result = {'schema': 1, 'sources': sources, 'metrics': metrics, 'controls': checks,
        'scope': 'Private precision+guard diagnostic only: TIRust512 with2^-392 guard, NinjaRS256, unchanged113-bit inputs/output and original lightlike policy. No production change or full-amplitude replay.',
        'metric': 'Complex abs(a-b)/max(abs(a),abs(b)); zero-zero=0. Absolute differences also retained. Not the campaign symmetric real metric.',
        'limits': ['Native success and a looser guard are not accuracy certificates.',
            'Supplied invariants and shifts retain their original on-shell convention mismatch.',
            'Different tensor degrees have different physical dimensions; cross-degree absolute maxima are descriptive only.',
            'Both reducers share oneloopmaster ancestry; agreement would not alone certify the scalar integrals.',
            'Failed tensors never contribute zero-valued reference rows.'],
        'full_amplitude_qualified': False}
    (HERE / 'summary.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps({name: {slot: row['relative']['difference']['relative']
        for slot, row in value['slot_maxima'].items()} for name, value in checks.items()}, indent=2))


if __name__ == '__main__':
    main()
