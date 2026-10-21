"""Independent HH3/HH4 retained data review; no MG7 imports or native access."""
import hashlib
import json
from decimal import Decimal, localcontext
from pathlib import Path

ROOT = Path('/shared/external/madloop_ninjars/madgraph7_madloop_ninjars')
BASE = ROOT / 'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/higher-higgs-controls'
OUT = Path(__file__).resolve().parent
CHECKED = {}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    require(path.suffix not in ('.a', '.so', '.o', '.mod'), 'Native artifact hashing prohibited')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def checked(bound):
    rel = Path(bound['path'])
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe bound path')
    path = ROOT / rel
    require(path.stat().st_size == bound['bytes'] and sha(path) == bound['sha256'], 'Changed bound data ' + str(rel))
    CHECKED[str(rel)] = bound
    return path


def card(path):
    lines = path.read_text().splitlines()
    result = {}
    for i, line in enumerate(lines):
        if line.startswith('#') and line[1:].isalnum():
            result[line[1:].lower()] = next(x.strip().lower() for x in lines[i+1:]
                                            if x.strip() and not x.lstrip().startswith(('#', '!')))
    return result


def main():
    result_path = BASE / 'run-v1/result-finalized-v1.json'
    require(sha(result_path) == '6a61ff18068ce87dfd06c4ee6f56f7580329f6bd24f22a98fa64c9a6da6649a6', 'Changed finalization')
    result = read(result_path)
    terminal_path = checked(result['original_collection_terminal'])
    terminal = read(terminal_path)
    require(sha(terminal_path) == '417a348b9b78283124f020e97879a5c4fe92b0d129bfcc9e9513aafe0b1ceeb4', 'Changed original terminal')
    require(terminal['controller']['parent_exit_code'] == result['original_parent_exit_code'] == 1 and
            terminal['native_rerun'] is False and result['native_calls_repeated'] == 0,
            'Original failed parent relabelled or native calls repeated')
    require(terminal['controller']['absent'] is True and
            terminal['all_four_native_children_reaped_exit_zero_and_absent'] is True,
            'No recorded terminal ownership release')
    require(terminal['post_collection_reader_failure'] ==
            "AttributeError: module 'ninjars_benchmark_policy_v3' has no attribute 'compare_rows'",
            'Original post-collection failure changed')
    checked(result['frozen_controller']); checked(result['versioned_finalizer']); checked(result['comparison_implementation'])
    spec_path = checked(result['science']['specification'])
    require(sha(spec_path) == '221750e50feadf9aac3e5dd834cb269eb18ad4e16bd5fddc42ef1e387c679528' and
            spec_path.read_bytes() == (BASE / 'held-specification-v2.json').read_bytes(), 'Held spec mismatch')
    spec = read(spec_path)
    require(spec['targets'] == [6, 8] and spec['configuration']['points_per_case'] == 2 and
            spec['normal_rescue'] is True and spec['performance_measurement'] is False and
            spec['fresh_process_invocations'] == 4 and spec['cpu'] == 190 and spec['loops'] == 1,
            'Changed authorized scope')
    require(spec['configuration']['relative_tolerance'] == 1e-7 and
            spec['configuration']['scale_tolerance'] == 1e-10, 'Changed tolerance')
    for bound in spec['sources'].values():
        checked(bound)
    checked(spec['generation_status']); checked(spec['production_inventory'])
    handoff = read(checked(terminal['root_handoff']))
    require(handoff['root_authorized'] is True and handoff['specification_sha256'] == sha(spec_path) and
            handoff['backends'] == [6, 8] and handoff['points_per_case'] == 2,
            'Root handoff changed')
    status = read(checked(terminal['native_collection_status']))
    expected_order = [(case['id'], backend) for case in spec['configuration']['cases'] for backend in (6, 8)]
    require(status['status'] == 'complete' and status['completed'] == [list(x) for x in expected_order], 'Partial collection')
    ledger_path = checked(terminal['native_process_ledger'])
    ledger = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    spawns = [e for e in ledger if e['event'] == 'spawn']
    reaps = [e for e in ledger if e['event'] == 'reaped']
    require(len(spawns) == len(reaps) == 4 and len(ledger) == 8, 'Unexpected native invocation count')
    children = terminal['native_children']
    require(len(children) == 4, 'Missing native quiet records')
    for spawn, child in zip(spawns, children):
        require(all(spawn[k] == child[k] for k in ('pid', 'start_ticks', 'pgid')) and spawn['pid'] == spawn['pgid'],
                'Terminal child identity mismatch')
        require(child['pid_absent'] and child['group_absent'] and child['reaped_exit_zero'], 'Recorded native group not quiet')
        matches = [e for e in reaps if all(e[k] == spawn[k] for k in ('pid', 'start_ticks', 'pgid'))]
        require(len(matches) == 1 and matches[0]['reaped'] is True and matches[0]['returncode'] == 0,
                'Missing successful native reap')

    cases = {}
    for case in spec['configuration']['cases']:
        cid = case['id']
        proof = read(checked(spec['origins'][cid]))
        require(proof['case'] == case and case['nfinal'] in (3, 4), 'Wrong actual process definition')
        dependency = {key: checked(proof[key]) for key in
                      ('generation', 'compilation', 'compiled_manifest', 'physical_card', 'runtime_signature', 'madloop_origin_card')}
        compilation = read(dependency['compilation'])
        generation = read(dependency['generation'])
        runtime = read(dependency['runtime_signature'])
        require(runtime['identity_sha256'] == 'a7577a9a33292ffb5d675ccf1b9c3667db3ab3613d77ccd270f1efc9e9802008', 'Changed retained runtime')
        require(runtime['libraries']['ninjars_lib/libninjars.a'] ==
                'b0d69deb8c16deb13293176296afc5e18dbeeebe778a20fa39b61804fba89d4c', 'Changed retained NinjaRS archive identity')
        require(compilation['source_identity'] == generation['source_identity'] == proof['generated_source_identity'], 'Source lineage mismatch')
        point_path = checked(spec['points'][cid]['data'])
        metadata = read(checked(spec['points'][cid]['metadata']))
        require(metadata['points'] == 2 and metadata['seed'] == case['seed'] and metadata['energy_gev'] == 1000 and
                metadata['sha256'] == sha(point_path) and metadata['masses'] == proof['external_masses'], 'Point metadata mismatch')
        lines = point_path.read_text().splitlines()
        nexternal = case['nfinal'] + 2
        require(lines[0] == '2' and len(lines) == 1 + 2*nexternal, 'Wrong actual point count')
        max_conservation = Decimal(0); max_mass_error = Decimal(0)
        with localcontext() as ctx:
            ctx.prec = 70
            for point in range(2):
                momenta = [[Decimal(x) for x in line.split()] for line in lines[1+point*nexternal:1+(point+1)*nexternal]]
                require(all(len(p) == 4 and all(x.is_finite() for x in p) and p[0] > 0 for p in momenta), 'Malformed momentum')
                max_conservation = max(max_conservation, *(abs(sum(p[k] for p in momenta[:2])-sum(p[k] for p in momenta[2:])) for k in range(4)))
                for momentum, mass in zip(momenta, metadata['masses']):
                    max_mass_error = max(max_mass_error, abs(momentum[0]**2-sum(x*x for x in momentum[1:])-Decimal(str(mass))**2))
                require(abs(momenta[0][0]+momenta[1][0]-1000) < Decimal('1e-10'), 'Point energy changed')
            require(max_conservation < Decimal('1e-10') and max_mass_error < Decimal('1e-7'), 'Unphysical retained momenta')

        rows = {}; parsed_records = {}; card_values = {}
        for backend in (6, 8):
            record_path = checked(result['science']['cases'][cid]['records'][str(backend)])
            record = read(record_path); parsed_records[backend] = record
            require(record['case'] == cid and record['backend'] == backend and record['cpu'] == 190 and record['loops'] == 1 and
                    record['performance_measurement'] is False and record['points_sha256'] == sha(point_path) and
                    record['parameter_card_sha256'] == proof['physical_card']['sha256'] and
                    record['executable_sha256'] == compilation['executable_sha256'] and
                    record['generated_sources_sha256'] == compilation['generated_sources_sha256'] and
                    record['generation_source_identity'] == proof['generated_source_identity'], 'Record provenance mismatch')
            raw = {name: checked(bound) for name, bound in record['raw_capture'].items()}
            require(set(raw) == {'values.dat', 'timing.dat', 'timed_status.dat', 'dimensions.dat'}, 'Raw output missing')
            require(raw['dimensions.dat'].read_text().split() == ['1', '1', '0'] and record['answer_dimensions'] ==
                    {'allocated': 1, 'loop_orders': 1, 'born_orders': 0}, 'Loop-induced dimensions changed')
            parts = [line.split() for line in raw['values.dat'].read_text().splitlines()]
            expected_status = 210 + backend
            require(len(parts) == 4 and all(len(row) == 8 for row in parts) and
                    [tuple(map(int, row[:3])) for row in parts] == [(p, o, expected_status) for p in (1, 2) for o in (0, 1)],
                    'Wrong physical/split-order/status stream')
            rows[backend] = [[Decimal(x) for x in row[3:]] for row in parts]
            require(all(x.is_finite() for row in rows[backend] for x in row) and all(row[0] == 0 and row[-1] >= 0 for row in rows[backend]),
                    'Nonfinite result, invalid accuracy or nonzero Born')
            for i, row in enumerate(rows[backend]):
                require(record['rows'][i]['coefficients'] == [float(x) for x in row[:4]] and
                        record['rows'][i]['accuracy'] == float(row[-1]), 'Parsed values differ from raw')
            require(raw['timed_status.dat'].read_text().split() == [str(expected_status), '2'] and
                    record['status_counts'] == record['timed_status_counts'] == {str(expected_status): 2}, 'Timed statuses mismatch')
            card_path = BASE / f'run-v1/inputs/candidate/{cid}/MadLoopParams-{backend}.dat'
            require(sha(card_path) == record['madloop_card_sha256'], 'Active card hash mismatch')
            settings = card(card_path); card_values[backend] = settings
            for key, value in {'mlreductionlib':str(backend), 'ctmoderun':'-1', 'ctmodeinit':'1',
                               'useloopfilter':'.false.', 'colliercomputeuvpoles':'.true.', 'colliercomputeirpoles':'.true.'}.items():
                require(settings[key] == value, 'Normal rescue/full poles setting changed: '+key)
            original = card(dependency['madloop_origin_card'])
            expected = dict(original, mlreductionlib=str(backend), ctmoderun='-1', ctmodeinit='1', useloopfilter='.false.',
                            colliercomputeuvpoles='.true.', colliercomputeirpoles='.true.')
            require(settings == expected, 'Unexpected numerical card change')
        require(parsed_records[6]['shared_numerical_settings'] == parsed_records[8]['shared_numerical_settings'], 'Different shared numerical settings')
        max_delta = Decimal(0); max_relative_finite = Decimal(0); max_pole_scaled = Decimal(0)
        with localcontext() as ctx:
            ctx.prec = 80
            scales = {point:max(abs(x) for backend in (6, 8) for i, row in enumerate(rows[backend]) if i//2+1==point for x in row[:4]) for point in (1, 2)}
            for i, (left, right) in enumerate(zip(rows[6], rows[8])):
                scale = scales[i//2+1]
                for x, y in zip(left[:4], right[:4]):
                    delta = abs(x-y); max_delta = max(max_delta, delta)
                    require(delta <= Decimal('1e-7')*max(abs(x),abs(y))+Decimal('1e-10')*scale, 'Cross-backend disagreement')
                max_relative_finite = max(max_relative_finite, abs(left[1]-right[1])/max(abs(left[1]),abs(right[1])))
                max_pole_scaled = max(max_pole_scaled, *(abs(x)/scale for x in left[2:4]+right[2:4]))
        cases[cid] = {'process':case['process'], 'points':2, 'rows_per_backend':4, 'backends':[6,8], 'statuses':[216,218],
                      'born_exact_zero':True, 'normal_rescue_configured':True, 'both_poles_enabled':True,
                      'max_absolute_coefficient_difference':str(max_delta), 'max_relative_finite_difference':str(max_relative_finite),
                      'max_absolute_pole_over_common_finite_scale':str(max_pole_scaled),
                      'max_momentum_conservation_absolute':str(max_conservation), 'max_mass_squared_error':str(max_mass_error),
                      'generated_revision':proof['generated_source_identity']['revision'],
                      'generated_tree':proof['generated_source_identity']['tree_state_sha256']}
        require(result['science']['cases'][cid]['passed'] is True, 'Owner disagrees with independent comparison')
    for spawn, (cid, backend) in zip(spawns, expected_order):
        proof = read(ROOT / spec['origins'][cid]['path'])
        require(spawn['command'] == ['taskset','-c','190','./check',str(ROOT/spec['points'][cid]['data']['path']),'1'] and
                spawn['cwd'] == str(ROOT/proof['output']/'SubProcesses'/proof['subprocess']), 'Actual dispatch differs')
    receipt = {'schema':'independent-higher-higgs-two-point-retained-review-v1', 'passed':True,
               'review_source_sha256':sha(Path(__file__)), 'result_sha256':sha(result_path),
               'terminal_sha256':sha(terminal_path), 'bound_text_inputs':len(CHECKED), 'cases':cases,
               'recorded_quiet_custody':{'children':4,'successfully_reaped':4,'controller_absent':True,'native_pids_and_groups_absent':True},
               'original_parent_failure':{'exit_code':1,'stage':'post-collection compare_rows import','preserved':True},
               'readonly_finalizer':'Separate corrected comparison; no native rerun; original controller/spec/raw unchanged',
               'limits':['Retained JSON/text/source/Decimal review only; no native calls, compiler, test execution, MG7 import, owner finalizer execution or native archive hash.',
                         'Quiet-state absence is the owner-recorded terminal observation, not a new live process scan.',
                         'Four fresh program invocations include cold/warm/untimed/retained traversals, not four primitive reduction calls.',
                         'Only two points per process; normal rescue is enabled but accepted statuses are stable DP, so no genuine QP rescue is demonstrated.',
                         'No timing, seven-pair, broad stability or zero-pole certification beyond the retained diagnostic is inferred.'],
               'bindings':CHECKED}
    path = OUT/'receipt.json'
    require(not path.exists(), 'Preserve existing independent receipt')
    path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'passed':True,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bound_text_inputs':len(CHECKED),'cases':cases}))


if __name__ == '__main__':
    main()
