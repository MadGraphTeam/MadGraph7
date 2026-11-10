#!/usr/bin/env python3
"""Reproduce the compact summary from the unchanged full analysis."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();here = Path(__file__).resolve().parent
    result = json.loads((here / 'analysis.json').read_text())
    out = {'envelope_verified': result['runner_envelope']['verified'],
           'transport_interpretation_eligible': result['transport_interpretation_eligible'],
           'cases': {name: {'refusals': [point['native_refusals'] for point in case['points']],
                            'fpe_qp': [point['fpe_end']['FPE_QP_END'][0][1] for point in case['points']]}
                     for name, case in result['cases'].items()}, 'comparisons': {}}
    for name, comparison in result['comparisons'].items():
        out['comparisons'][name] = {
            'nonmass_input_bits_match': comparison['nonmass_input_bits_match'],
            'failures': comparison['failures'],
            'difference_counts': dict(Counter(record['label'] for record in comparison['differences'])),
            'mass_square_change_records': len(comparison['mass_square_changes']),
            'amplitude_first_split': [record for record in comparison['amplitude_comparison'] if record['flat_index'] <= 4]}
    with args.output.open('x') as stream:
        json.dump(out, stream, indent=2);stream.write('\n')


if __name__ == '__main__':
    main()
