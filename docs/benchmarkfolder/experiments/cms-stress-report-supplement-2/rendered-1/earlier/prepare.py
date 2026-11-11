#!/usr/bin/env python3
"""Prepare source-only report data from four explicitly closed public packets."""
import argparse
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKETS = {
    'cms-universal-qed-pole-audit-1': '1565ac55fe6aeeefbd4f660455c422c4b954737a434d27138cbe8433d9534b77',
    'cms-universal-qed-single-pole-audit-1': '99603d41d662e90f071b8052eb4303bcba8af451c5c96654febc276e620ad11e',
    'kinematic-invariant-amplitude-qualification-1': 'c04a73c86f6aa3db59e6d5cafd2fb9edf92dd2dbd6d3e418cab462ebbb5327d4',
    'kinematic-legacy-reference-adjudication-1': '248bd61e987ac97c81e5397489eb4322ecfbf96cf85a42c84607aef2c013500e',
}
PRESERVED_PDF = '2ecf83e01bd6367a7a39ecd7ea23dec40024d2ef5072dc7a583e26ebf9a37cc8'
WORKERS = {'cms-p20-mode4-lib1-original': 'CutTools QP control',
           'cms-p20-mode4-lib9-candidate': 'Private candidate TIRust QP'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n'


def parsed(raw):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'Repeated JSON key')
            out[key] = value
        return out
    def invalid(value):
        raise ValueError('Nonfinite JSON token: ' + value)
    return json.loads(raw, object_pairs_hook=unique, parse_constant=invalid)


def safe(name):
    path = PurePosixPath(name)
    require(isinstance(name, str) and name and not path.is_absolute() and '..' not in path.parts
            and str(path) == name and '\\' not in name, 'Unsafe evidence path')
    return name


class Packet:
    def __init__(self, directory, expected):
        self.directory = Path(directory)
        require(not any(path.is_symlink() for path in (self.directory, *self.directory.parents)),
                'Symlink packet root')
        raw = (self.directory / 'evidence-index.json').read_bytes()
        require(sha(raw) == expected, 'Closed public index changed')
        self.index, self.files = parsed(raw), {}
        paths = list(self.directory.rglob('*'))
        require(not any(path.is_symlink() for path in paths), 'Symlink packet member')
        actual = {p.relative_to(self.directory).as_posix() for p in paths if p.is_file()}
        require(actual == set(self.index['files']) | {'evidence-index.json'}, 'Closed inventory differs')
        for name, identity in self.index['files'].items():
            raw = (self.directory / safe(name)).read_bytes()
            require({'bytes': len(raw), 'sha256': sha(raw)} == identity, 'Closed payload differs: ' + name)
            self.files[name] = raw
        self.expected = expected
        self.used = {}

    def raw(self, name):
        name = safe(name)
        item = self.index.get('compressed_text', {}).get(name)
        raw = gzip.decompress(self.files[item['path']]) if item else self.files[name]
        if item:
            require(len(raw) == item['raw_bytes'] and sha(raw) == item['raw_sha256'], 'Expanded payload differs')
        self.used[name] = {'bytes': len(raw), 'sha256': sha(raw)}
        return raw

    def json(self, name):
        return parsed(self.raw(name))


def duplicate_order_pairs(rows, fields):
    selected = []
    require(len(rows) == 8, 'Expected two workers, two phases and duplicate orders')
    for worker in WORKERS:
        for phase, point in ((0,1),(1,20)):
            pair = [r for r in rows if r['worker'] == worker and r['phase'] == phase and r['point'] == point]
            require(len(pair) == 2 and {r['order'] for r in pair} == {0,1}, 'Order population differs')
            zero, one = sorted(pair, key=lambda r:r['order'])
            require(all(zero[k] == one[k] for k in fields), 'Duplicate-order values differ')
            selected.append(zero)
    return selected


def derive(packets_root):
    packets = {name:Packet(Path(packets_root)/name, expected) for name,expected in PACKETS.items()}
    double = packets['cms-universal-qed-pole-audit-1']
    single = packets['cms-universal-qed-single-pole-audit-1']
    trial = packets['kinematic-invariant-amplitude-qualification-1']
    legacy = packets['kinematic-legacy-reference-adjudication-1']
    dr, sr = double.json('source/result.json'), single.json('source/result.json')
    parent = trial.json('results/root-point20-comparison.json')
    observations = double.json('source/observations.json')
    require(sha(trial.raw('results/root-point20-comparison.json')) == observations['source_sha256'],
            'Pole observations belong to another closed trial')
    observed = duplicate_order_pairs(observations['rows'], ('eligible','code','words'))
    for row in observations['rows']:
        parent_obs = parent['observations'][row['worker']]
        require(row['eligible'] is True and parent_obs['eligible'] is True and row['code'] == 120,
                'Only eligible unchecked fixed-QP observations admitted')
        require({k:v for k,v in row.items() if k not in ('worker','eligible')} in parent_obs['raw_rows'],
                'Pole row differs from retained parent words')
    drows = duplicate_order_pairs(dr['comparisons'], ('Born','expected_four_massless_unit_charge_legs',
                                                   'observed_double_pole','relative_residual'))
    srows = [{**row,'phase':p['phase'],'point':p['point']} for p in sr['points'] for row in p['comparisons']]
    srows = duplicate_order_pairs(srows, ('Born','expected_IR_single_pole','observed_single_pole','relative_residual'))
    rows = []
    for pole, supplied, expected_key, observed_key in (
            (-2,drows,'expected_four_massless_unit_charge_legs','observed_double_pole'),
            (-1,srows,'expected_IR_single_pole','observed_single_pole')):
        for row in supplied:
            with localcontext() as ctx:
                ctx.prec = 90
                prediction, value = Decimal(row[expected_key]), Decimal(row[observed_key])
                require(prediction.is_finite() and value.is_finite() and prediction != 0, 'Invalid eligible pole')
                ratio = value / prediction
                residual = abs(ratio-1)
                require(abs(residual-Decimal(row['relative_residual'])) < Decimal('1e-65'),
                        'Audited relative residual differs')
                rows.append({'pole':pole,'point':row['point'],'phase':row['phase'],
                             'phase_label':'point 1 bootstrap' if row['phase']==0 else 'point 20 stress subset',
                             'worker':row['worker'],'label':WORKERS[row['worker']],
                             'Born':row['Born'],'expected':row[expected_key],'observed':row[observed_key],
                             'relative_residual':row['relative_residual'],'observed_over_predicted':str(ratio),
                             'eligible_fixed_mode':True,'return_code':120,'normal_host_stability_qualified':False})
    first = trial.json('predecessor/results/build-terminal-review.json')
    stress = trial.json('results/root-stress-review.json')
    ordinary = trial.json('results/ordinary-comparison.json')
    reference = legacy.json('results/reference-comparison.json')
    reference_review = legacy.json('results/root-review.json')
    require(first['terminal_exit_code']==1 and first['observed_owned_commands']==0
            and first['observed_fortran_main_calls']==0 and "KeyError('size')" in first['failure'],
            'First failed controller/build scope differs')
    require(stress['workers_complete']==8 and stress['workers_launched']==9 and stress['unlaunched_workers']==7
            and stress['timeout_seconds']==300 and stress['timeout_native_returncode']==-15
            and stress['full_stress_passed'] is False and stress['candidate_point20_gate_passed'] is False,
            'Incomplete stress scope differs')
    require(parent['workers_eligible']==5 and parent['candidate_gate_passed'] is False,
            'Point20 gate differs')
    require(ordinary['passed'] is True and ordinary['workers_eligible']==8, 'Ordinary scope differs')
    blocked = []
    for backend in ('6','8'):
        item = reference['observations'][backend]
        require(item['eligible'] is False and item['measurement_interpretation_allowed'] is False,
                'Legacy worker unexpectedly eligible')
        # Do not access, decode, convert or compare any failed coefficient buffer.
        usage = item['native_usage']
        blocked.append({'backend':int(backend),'label':'Ninja' if backend=='6' else 'NinjaRS',
                        'eligible':False,'coefficient_arithmetic_performed':False,
                        'reason':item['failures'], 'native_usage_gate_passed':item['native_usage_gate_passed'],
                        'observed_DP_adapter_calls':usage['actual_dp_calls'],
                        'completed_adapter_calls':reference_review['usage'][backend]['native_completed_calls'],
                        'nonzero_adapter_status_count':reference_review['usage'][backend]['raw_nonzero_status_count'],
                        'measurement_software_flags':[{k:s[k] for k in ('fpe_dp','fpe_qp')}
                            for s in item['states'] if s['stage']=='A' and s['phase']==1]})
    require(reference['reference_gate_passed'] is False and reference['same_native_input_established'] is False,
            'Legacy reference scope differs')
    require(sr['current_trial_new_external_word_capture'] is False
            and sr['UV_cancellation_independently_measured'] is False
            and sr['model_runtime_coupling_bits_observed'] is False
            and sr['finite_truth_qualified'] is False, 'Pole-audit limitations differ')
    return {'schema':'madloop-private-stress-report-supplement-data-v1',
            'scope':'source-only; no PDF render, new native evaluation, timing or production promotion',
            'preserved_report':{'sha256':PRESERVED_PDF,'pages':35,'operation':'append only after review; preserve all original pages'},
            'packet_indexes':PACKETS,'verified_packet_payloads':{n:len(p.files) for n,p in packets.items()},
            'selected_source_files':{n:p.used for n,p in packets.items()},
            'pole_comparisons':rows,'blocked_legacy_references':blocked,
            'stage_status':{'first_build_failed_before_native':True,'first_failure':first['failure'],
                'ordinary_workers_eligible':8,'point20_workers_complete':8,'point20_workers_eligible':5,
                'stress_workers_launched':9,'stress_workers_complete':8,'stress_workers_unlaunched':7,
                'stress_timeout_worker':stress['timeout_worker'],'stress_timeout_seconds':300,
                'stress_timeout_native_returncode':-15,'full_stress_passed':False,'candidate_comparison_gate_passed':False,
                'legacy_reference_gate_passed':False},
            'assumptions':{'alpha_reconstructed':dr['alpha_reconstructed'],'Gmu_choice':1,
                'physical_card_parsed_through_binary64_assumed':True,'runtime_coupling_bits_observed':False,
                'complete_Fortran_rounding_emulated':False,'old_group3_QP_leaves_reused':True,
                'new_candidate_external_word_capture':False,'UV_cancellation_separately_measured':False,
                'formula_scope':sr['amplitude_scope'],'finite_term_adjudicated':False,
                'specific_CutTools_scalar_cause_proven':False,'normal_host_stability_qualified':False,
                'performance_qualified':False,'working_digits_do_not_measure_native_precision':True},
            'local_timing_TODO':'Hua-Sheng: identical local builds/points/cards; at least seven alternating paired batches of about one second; upper 95% candidate/baseline bound <=1.05 for every main process; existing-backend regressions and DP/rescued cohorts separate.',
            'new_native_evaluations':0,'failed_coefficients_arithmetic_performed':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packets-root',type=Path,default=ROOT/'docs/benchmarkfolder/experiments')
    parser.add_argument('--output',type=Path,default=HERE/'data.json')
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    value = encoded(derive(args.packets_root))
    if args.verify_only:
        require(args.output.read_text()==value,'Derived report data differs')
    else:
        with args.output.open('x') as stream:
            stream.write(value)
    print(encoded({'passed':True,'verify_only':args.verify_only,'data_sha256':sha(value.encode()),
                   'pole_rows':8,'blocked_legacy_workers':2,'new_native_evaluations':0}),end='')


if __name__=='__main__':
    main()
