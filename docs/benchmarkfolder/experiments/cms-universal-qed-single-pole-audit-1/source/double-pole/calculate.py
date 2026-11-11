"""Pure source/card reconstruction of the four-leg universal QED double pole.

No native library, compiler, model initializer or amplitude is executed.
This is a pole consistency test, not a finite/single-pole truth assignment.
"""
from decimal import Decimal as D, localcontext
import argparse
import hashlib
import json
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def card_values(text):
    # Match the external REAL*8 parsing boundary before the existing MP reader
    # widens it. Decimal.from_float preserves that resulting binary64 value.
    values = {}
    block = ''
    for line in text.splitlines():
        fields = line.split('#')[0].split()
        if not fields:
            continue
        if fields[0].upper() == 'BLOCK':
            block = fields[1].upper()
        elif fields[0].upper() == 'DECAY':
            values['WIDTH', int(fields[1])] = D.from_float(float(fields[2]))
        elif fields[0].isdigit() and len(fields) >= 2:
            values[block, int(fields[0])] = D.from_float(float(fields[1]))
    return values


def binary64(word):
    require(len(word) == 16, 'Unexpected binary64 word')
    value = D.from_float(struct.unpack('>d', bytes.fromhex(word))[0])
    require(value.is_finite(), 'Nonfinite retained amplitude component')
    return value


def calculate():
    pins = json.loads((HERE/'source-pins.json').read_bytes())
    excerpts = json.loads((HERE/'source-excerpts.json').read_bytes())
    for entry in excerpts.values():
        require(entry['path'] in pins, 'Unpinned excerpt source')
        for section in entry['ranges']:
            require(hashlib.sha256(section['text'].encode()).hexdigest() == section['sha256'],
                    'Retained excerpt changed')
    require('MDL_GFREDEFINITIONCHOICE = 1.000000D+00' in excerpts['model_dp']['ranges'][0]['text'],
            'Source does not select the modulus Gmu definition')
    require('MDL_CMSPARAM = 1.000000D+00' in excerpts['model_dp']['ranges'][0]['text'],
            'Source does not select complex masses')
    observed = json.loads((HERE/'observations.json').read_bytes())
    require(observed['source_sha256'] == pins[observed['source']]['sha256'], 'Observation ancestry differs')
    card = (HERE/'param_card.dat').read_bytes()
    card_origin = next(name for name in pins if name.endswith('/Cards/param_card.dat'))
    require(digest(card) == pins[card_origin], 'Retained card differs')
    values = card_values(card.decode())
    with localcontext() as context:
        context.prec = 70
        pi = D('3.1415926535897932384626433832795')
        mw, mz = values['MASS', 24], values['MASS', 23]
        ww, wz = values['WIDTH', 24], values['WIDTH', 23]
        gf = values['SMINPUTS', 2]
        # The selected model sets Gfbar=Gf*|A|/A, so alpha is real:
        # A=muW^2*(1-muW^2/muZ^2), muV^2=MV^2-i*MV*GammaV.
        w, z = (mw*mw, -mw*ww), (mz*mz, -mz*wz)
        denominator = z[0]**2 + z[1]**2
        ratio_real = (w[0]*z[0]+w[1]*z[1])/denominator
        ratio_imag = (w[1]*z[0]-w[0]*z[1])/denominator
        a = (w[0]*(1-ratio_real)+w[1]*ratio_imag,
             w[1]*(1-ratio_real)-w[0]*ratio_imag)
        alpha = D(2).sqrt()*gf*(a[0]**2+a[1]**2).sqrt()/pi
        comparisons = []
        require(len(observed['rows']) == 8, 'Retained row count differs')
        for row in observed['rows']:
            require(row['eligible'] and row['code'] == 120 and row['order'] in (0,1),
                    'This audit requires the eligible fixed-QP rows')
            values_row = list(map(binary64, row['words']))
            born, _, _, double = values_row[:4]
            expected = -4*born*alpha/(2*pi)
            comparisons.append({
                'worker': row['worker'], 'phase': row['phase'], 'point': row['point'],
                'order': row['order'], 'Born': str(born), 'observed_double_pole': str(double),
                'expected_four_massless_unit_charge_legs': str(expected),
                'relative_residual': str(abs((double-expected)/expected)),
                'effective_sum_charge_squared': str(-double/(born*alpha/(2*pi))),
                'finite_and_single_pole_adjudicated': False,
            })
        return {
            'schema': 'private-cms-four-leg-qed-double-pole-source-audit-v1',
            'native_executions': 0, 'decimal_working_digits': 70,
            'model': 'Existing complex-mass Gmu scheme, GFRedefinitionChoice=1',
            'source_math': 'alpha=sqrt(2)*Gf*abs(muW^2*(1-muW^2/muZ^2))/pi',
            'pole_formula': 'A[-2]=-sum_massless_charged_legs(Q_i^2)*Born*alpha/(2*pi)',
            'massless_legs': 4, 'sum_charge_squared': 4,
            'alpha_reconstructed': str(alpha), 'inverse_alpha_reconstructed': str(1/alpha),
            'external_values_after_DP_parse': {str(k): str(v) for k,v in values.items()
                                              if k in [('MASS',24),('MASS',23),('WIDTH',24),('WIDTH',23),('SMINPUTS',2)]},
            'comparisons': comparisons,
            'model_runtime_coupling_bits_observed': False,
            'complete_Fortran_rounding_emulated': False,
            'normalization_ordinary_crosscheck': 'Both ordinary QP rows obey the same four-leg formula',
            'independent_full_amplitude_accuracy_qualified': False,
            'finite_or_single_pole_truth_qualified': False,
            'checked_stability_qualified': False,
            'performance_qualified': False,
            'cuttools_specific_scalar_cause_proven': False,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-live-sources', action='store_true')
    args = parser.parse_args()
    if args.verify_live_sources:
        repo = HERE.parents[1]
        for name, identity in json.loads((HERE/'source-pins.json').read_bytes()).items():
            require(digest((repo/name).read_bytes()) == identity, 'Live source changed: '+name)
    print(json.dumps(calculate(), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
