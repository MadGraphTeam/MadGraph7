#!/usr/bin/env python3
"""Plot explicitly supplied, closed point metrics; never discover native results.

Input schema: madloop-stability-point-metrics-v1, with metric, provenance,
and cases=[{id,label,point_ids:[...],lanes:[{backend,values:[number|null,...]}]}].
Values align exactly with point_ids. Null means unavailable, including failed,
timed-out, unexecuted, warning-invalidated or inadmissible-precision points.
The caller must preserve the reason for each unavailable value in provenance.
This module does not validate a numerical campaign or certify true errors.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

BACKENDS = {1: 'CutTools', 6: 'Ninja', 7: 'COLLIER', 8: 'NinjaRS', 9: 'TIRust'}
COLORS = {1: '#c74943', 6: '#3f6ea3', 7: '#8356a3', 8: '#ce8b20', 9: '#188469'}
METRICS = {
    'raw-dp-finite-consistency': 'Raw forced-DP finite consistency estimate',
    'accepted-composite-internal-estimate': 'Accepted MadLoop composite stability estimate',
    'accepted-reducer-disagreement': 'Accepted finite coefficient disagreement',
}


def relative_spread(values):
    """Legacy three-probe diagnostic, with explicit zero and overflow behavior."""
    if len(values) != 3 or any(not math.isfinite(x) for x in values):
        raise ValueError('Exactly three finite probe values are required')
    # Dividing subnormal values by three can erase a nonzero mean. Restrict
    # this rescaling to that range so ordinary legacy arithmetic stays intact.
    scale = max(abs(x) for x in values)
    if 0 < scale < 4 * sys.float_info.min:
        values = [x / scale for x in values]
    largest, smallest = max(values), min(values)
    spread = largest - smallest
    mean = math.fsum(x / 3.0 for x in values)
    if not (math.isfinite(spread) and math.isfinite(mean)):
        scale = max(abs(x) for x in values)
        normalized = [x / scale for x in values]
        spread = max(normalized) - min(normalized)
        mean = math.fsum(x / 3.0 for x in normalized)
    if mean == 0.0:
        return 0.0 if spread == 0.0 else math.inf
    return spread / abs(mean)


def symmetric_disagreement(a, b):
    """Existing regression metric; separate from the three-probe diagnostic."""
    if not (math.isfinite(a) and math.isfinite(b)):
        raise ValueError('Finite coefficients are required')
    numerator, denominator = abs(a - b), abs(a + b)
    if not (math.isfinite(numerator) and math.isfinite(denominator)):
        scale = max(abs(a), abs(b))
        a, b = a / scale, b / scale
        numerator, denominator = abs(a - b), abs(a + b)
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def metric_value(value):
    if value is None:
        return None
    # JSON stores +inf as an explicit string, never the nonstandard Infinity token.
    if value == 'inf':
        return math.inf
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Metric must be a nonnegative number, "inf", or null')
    if not math.isfinite(value) or value < 0:
        raise ValueError('Metric must be nonnegative; use "inf" for infinity')
    return float(value)


def tail_bounds(values, delta):
    """All intended points remain in the denominator; equality does not exceed."""
    if not values or not math.isfinite(delta) or delta <= 0:
        raise ValueError('Nonempty population and finite positive threshold required')
    parsed = [metric_value(v) for v in values]
    known = sum(v is not None and v > delta for v in parsed)
    unavailable = sum(v is None for v in parsed)
    n = len(parsed)
    return {'n': n, 'known_exceedances': known, 'unavailable': unavailable,
            'lower': known / n, 'upper': (known + unavailable) / n,
            'fraction': known / n if unavailable == 0 else None,
            'exact_zero_metrics': sum(v == 0 for v in parsed if v is not None)}


def tail_ratio(values, ninja_values, delta):
    """No pseudocount, denominator floor, or ratio from incomplete populations."""
    if len(values) != len(ninja_values):
        raise ValueError('Paired populations have different sizes')
    numerator, denominator = tail_bounds(values, delta), tail_bounds(ninja_values, delta)
    if numerator['unavailable'] or denominator['unavailable']:
        return None
    if denominator['known_exceedances'] == 0:
        return None
    return numerator['known_exceedances'] / denominator['known_exceedances']


def validate_dataset(data):
    if data.get('schema') != 'madloop-stability-point-metrics-v1':
        raise ValueError('Unexpected point-metrics schema')
    if data.get('metric') not in METRICS:
        raise ValueError('Metric type must be explicit')
    if not isinstance(data.get('provenance'), dict) or not data['provenance']:
        raise ValueError('Explicit closed-input provenance is required')
    if not isinstance(data.get('cases'), list) or not data['cases']:
        raise ValueError('At least one case is required')
    seen = set()
    for case in data['cases']:
        if not isinstance(case.get('id'), str) or case['id'] in seen:
            raise ValueError('Case IDs must be unique strings')
        seen.add(case['id'])
        points = case.get('point_ids')
        if not isinstance(points, list) or not points:
            raise ValueError('A physical point population is required')
        if any(isinstance(p, bool) or not isinstance(p, (str, int)) for p in points):
            raise ValueError('Point IDs must be strings or integers')
        if len(set(points)) != len(points):
            raise ValueError('Unique physical point IDs are required; no duplicate orders')
        backends = []
        for lane in case.get('lanes', []):
            if lane.get('backend') not in BACKENDS or lane['backend'] in backends:
                raise ValueError('Unknown or repeated reduction backend')
            backends.append(lane['backend'])
            if len(lane.get('values', [])) != len(points):
                raise ValueError('Every intended physical point needs a value or null')
            for value in lane['values']:
                metric_value(value)
        if set(backends) != set(BACKENDS):
            raise ValueError('Every case must retain all five intended backends')
    return data


def summarize(data, thresholds):
    validate_dataset(data)
    cases = []
    for case in data['cases']:
        lanes = {lane['backend']: lane['values'] for lane in case['lanes']}
        cases.append({'id': case['id'], 'n': len(case['point_ids']),
                      'resolution': 1 / len(case['point_ids']),
                      'lanes': [{'backend': backend, 'label': BACKENDS[backend],
                                 'tail': [tail_bounds(lanes[backend], d) for d in thresholds],
                                 'ratio_over_Ninja': [tail_ratio(lanes[backend], lanes[6], d)
                                                      for d in thresholds]}
                                for backend in BACKENDS]})
    return {'schema': 'madloop-stability-tail-summary-v1', 'metric': data['metric'],
            'metric_label': METRICS[data['metric']], 'thresholds': thresholds,
            'interpretation': 'internal consistency or reducer disagreement; not certified true error',
            'confidence': 'finite fixture frequencies; no population confidence claim', 'cases': cases}


def render(data, output_prefix, *, input_sha256):
    """Only explicit input is read; matplotlib is imported solely for rendering."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    thresholds = [10 ** (-16 + i / 8) for i in range(129)]
    summary = summarize(data, thresholds)
    summary['input_sha256'] = input_sha256
    fig = plt.figure(figsize=(5.1 * len(data['cases']), 5.5), constrained_layout=True)
    grid = fig.add_gridspec(2, len(data['cases']), height_ratios=(3.2, 1.25))
    for column, (case, case_summary) in enumerate(zip(data['cases'], summary['cases'])):
        top, ratio = fig.add_subplot(grid[0, column]), fig.add_subplot(grid[1, column])
        n = case_summary['n']
        for lane in case_summary['lanes']:
            backend, tails = lane['backend'], lane['tail']
            complete = [row['fraction'] if row['fraction'] and row['fraction'] > 0 else math.nan
                        for row in tails]
            top.step(thresholds, complete, where='post', color=COLORS[backend],
                     label=f"{BACKENDS[backend]} ({tails[0]['exact_zero_metrics']} zeros)")
            if any(row['unavailable'] for row in tails):
                lower = [row['lower'] if row['lower'] > 0 else math.nan for row in tails]
                upper = [row['upper'] if row['upper'] > 0 else math.nan for row in tails]
                top.fill_between(thresholds, lower, upper, step='post', alpha=.12,
                                 color=COLORS[backend], hatch='//')
                # Bounds with an exact-zero lower edge still need a visible upper bound.
                top.step(thresholds, upper, where='post', color=COLORS[backend], linestyle=':')
            ratios = [x if x is not None else math.nan for x in lane['ratio_over_Ninja']]
            ratio.step(thresholds, ratios, where='post', color=COLORS[backend])
        for axis in (top, ratio):
            axis.set_xscale('log'); axis.grid(True, which='major', alpha=.18)
            axis.tick_params(labelsize=8)
            axis.spines[['top', 'right']].set_visible(False)
        top.set_yscale('log'); top.set_ylim(.8 / n, 1.1)
        top.set_title(case.get('label', case['id']), color='#173451', fontsize=11)
        top.set_ylabel('Fraction with metric > target Δ')
        top.axhline(1 / n, color='#aaa', linewidth=.6, linestyle='--')
        top.text(.02, .02, f"N={n}; resolution={1/n:g}\nZeros omitted on log y; missing values remain bounded",
                 transform=top.transAxes, fontsize=7, va='bottom')
        for target, style in ((1e-3, '--'), (3e-6, ':'), (1e-8, '-.')):
            top.axvline(target, color='#777', linewidth=.65, linestyle=style)
        top.legend(fontsize=7, loc='upper right')
        # A linear ratio axis preserves valid exact-zero ratios. The upper
        # tail panel stays logarithmic; no zero is assigned a fabricated floor.
        ratio.set_ylim(bottom=0); ratio.axhline(1, color='#777', linewidth=.65)
        ratio.set_ylabel('Ratio over Ninja'); ratio.set_xlabel('Target relative metric Δ')
        ratio.text(.02, .03, 'Masked: missing population or zero Ninja tail',
                   transform=ratio.transAxes, fontsize=7)
    fig.suptitle(METRICS[data['metric']] + '\n100-point pilot where N=100; not an independent true-error bound',
                 color='#173451', fontsize=12)
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.pdf', '.png', '.json'):
        if prefix.with_suffix(suffix).exists():
            raise FileExistsError('Refusing to overwrite an existing stability artifact')
    fig.savefig(prefix.with_suffix('.pdf'))
    fig.savefig(prefix.with_suffix('.png'), dpi=180)
    plt.close(fig)
    prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output-prefix', type=Path, required=True)
    args = parser.parse_args()
    raw = args.input.read_bytes()
    data = json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    render(data, args.output_prefix, input_sha256=hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    main()
