#!/usr/bin/env python3
"""Compare frozen printed amplitudes to the separately derived QED poles."""
from decimal import Decimal, localcontext
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
TRACE = HERE.parents[1]
STRESS = ROOT / 'docs/benchmarkfolder/experiments/physical-stress'
D = Decimal


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert Path.cwd().resolve() == ROOT
    prediction = json.loads((HERE / 'summary.json').read_text())
    for value in prediction['sources'].values():
        assert sha(ROOT / value['path']) == value['sha256'], value['path']
    sources = {}

    def bind(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return path

    bind(HERE / 'summary.json')
    bind(Path(__file__))
    points_sha = sha(bind(TRACE / 'selected.dat'))
    card_sha = sha(bind(TRACE / 'param_card.dat'))
    rows = []
    with localcontext() as context:
        context.prec = 90

        def consume(path, name, status, backend, recorded=None):
            values = [line.split() for line in bind(path).read_text().splitlines() if line.strip()]
            assert {(int(x[0]), int(x[1])) for x in values} == {(p, o) for p in (1, 2) for o in (0, 1)}
            assert len(values) == 4
            if recorded is not None:
                saved = [r for category in ('accepted', 'rejected', 'unchecked')
                    for r in recorded[category]]
                assert len(saved) == len(values)
                for raw in values:
                    matches = [r for r in saved if (r['point'], r['order']) ==
                        (int(raw[0]), int(raw[1]))]
                    assert len(matches) == 1
                    assert matches[0]['status'] == int(raw[2])
                    assert matches[0]['raw_numbers'] == raw[3:]
            for point in (1, 2):
                a, b = [x for x in values if int(x[0]) == point]
                assert a[2:] == b[2:]  # total and sole split order must agree
                reference = prediction['records'][point - 1]
                born, finite, single, double, accuracy = map(D, a[3:])
                assert all(x.is_finite() for x in (born, finite, single, double, accuracy))
                assert abs(born - D(reference['born_used'])) / abs(born) < D('1e-17')
                expected_single = D(reference['single_pole_prediction'])
                expected_double = D(reference['double_pole_prediction'])
                expected_ratio = D(reference['single_over_double_ratio'])
                ratio = single / double
                retcode = int(a[2])
                incomplete = backend == 9 and point == 2
                rows.append({'run': name, 'point': point, 'backend': backend,
                    'retcode': retcode, 'reported_accuracy': str(accuracy),
                    'run_status': status, 'historical_TIR_incomplete': incomplete,
                    'ordinary_acceptance_claim': False,
                    'host_classification': 'unchecked fixed precision' if retcode // 100 == 1 else
                        ('accepted by historical host stability check' if retcode // 100 in (2, 3) else 'rejected by historical host stability check'),
                    'printed_born': str(born), 'printed_finite': str(finite),
                    'printed_single': str(single), 'printed_double': str(double),
                    'single_relative_to_prediction': str(abs(single - expected_single) / abs(expected_single)),
                    'double_relative_to_prediction': str(abs(double - expected_double) / abs(expected_double)),
                    'single_over_double': str(ratio),
                    'ratio_relative_to_prediction': str(abs(ratio - expected_ratio) / abs(expected_ratio)),
                    'ratio_uses_no_Born_or_coupling_normalization': True})

        for section in ('runs', 'diagnostics'):
            for folder in sorted((STRESS / section / 'ew_n3_epem_ttxa').iterdir()):
                if section == 'diagnostics' and not folder.name.endswith('-qp-init-dp'):
                    continue
                if section == 'runs' and not folder.name.endswith('-auto'):
                    continue
                spec = json.loads(bind(folder / 'specification.json').read_text())
                result = json.loads(bind(folder / 'result.json').read_text())
                inputs = json.loads(bind(folder / (spec['source'] + '-inputs.json')).read_text())
                assert spec['points_sha256'] == inputs['points_sha256'] == points_sha
                assert result['points_sha256'] == points_sha and result['backend'] == spec['backend']
                assert inputs['param_card_sha256'] == card_sha
                assert inputs['generation']['case']['process'] == 'e+ e- > t t~ a [virt=QED]'
                assert spec['backend'] in (1, 6, 7, 8, 9)
                settings = inputs['settings']
                assert str(settings['mlreductionlib']) == str(spec['backend'])
                assert settings['looppolemode'] == 0 and settings['osthres'] == 1e-8
                assert settings['mlstabthres'] == 1e-6 and settings['improvepspoint'] == 2
                if section == 'diagnostics':
                    assert settings['ctmoderun'] == 4 and settings['ctmodeinit'] == 1
                    assert spec['force_stability_checks'] is False and spec['stability_checked'] is False
                consume(folder / 'raw/values.dat', section + '/' + folder.name, result['status'], spec['backend'], result['rows'])

        private = TRACE / 'ct-qp-transport'
        execution = json.loads(bind(private / 'execution.json').read_text())
        assert execution['complete'] and execution['original_inputs_unchanged']
        assert execution['policy'] == {'QP_numerator': True, 'backend': 1, 'init_mode': 1,
            'run_mode': 4, 'scalar_cutoff': 1e-6, 'stability_checks': False}
        assert sha(bind(private / 'selected.dat')) == points_sha
        assert sha(bind(private / 'param_card.dat')) == card_sha
        assert sha(private / 'values.dat') == execution['values_sha256']
        consume(private / 'values.dat', 'private-CT-QP-denominators', 'unchecked private diagnostic', 1)

    output = {'schema': 1, 'sources': sources, 'point_sha256': points_sha,
        'physical_card_sha256': card_sha,
        'scope': 'Source-derived pole comparisons only; printed raw coefficients are retained, including host-rejected and unchecked values. No finite/whole-amplitude certificate.',
        'metric': 'abs(observed-predicted)/abs(predicted), including for the single/double ratio. Not the campaign symmetric regression metric.',
        'ratio_limit': 'The ratio cancels Born and coupling normalization, but still tests the independently specified repaired-QP eikonal kinematics and scale.',
        'incomplete_output_policy': 'Historical TIRust threshold outputs omit37 refused groups; residuals only illustrate invalidity and cannot adjudicate a served TIRust tensor.',
        'rows': rows}
    (HERE / 'comparisons.json').write_text(json.dumps(output, indent=2, sort_keys=True) + '\n')
    print(json.dumps([{'run': r['run'], 'single_relative': r['single_relative_to_prediction'],
        'ratio_relative': r['ratio_relative_to_prediction']} for r in rows if r['point'] == 2]))


if __name__ == '__main__':
    main()
