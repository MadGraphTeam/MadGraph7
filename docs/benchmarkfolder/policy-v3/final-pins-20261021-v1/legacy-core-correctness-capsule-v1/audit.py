#!/usr/bin/env python3
"""Read-only stdlib audit of saved Ninja6 correctness and cancelled stress.

No scientific modules, tests, builds, evaluators or process queries are used.
Default mode needs only retained public evidence. --live additionally hashes
the preserved worker executables/sources/libraries; no native bytes are copied.
"""
import argparse
import collections
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'madgraph/__init__.py').is_file())
BASE = Path('docs/benchmarkfolder/policy-v3/final-pins-20261021-v1')
CORE = BASE / 'legacy-core-correctness-capsule-v1'
CANCELLED = BASE / 'stress-heavy-cancelled-v1'
HASHES = {}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def disk(record):
    path = absolute(record['path'])
    key = str(path.resolve())
    if key not in HASHES:
        data = path.read_bytes()
        HASHES[key] = (sha(data), len(data))
    require(HASHES[key][0] == record['sha256'], 'Bound hash differs: ' + record['path'])
    require('bytes' not in record or HASHES[key][1] == record['bytes'], 'Bound size differs: ' + record['path'])
    return path


def bound(path):
    path = Path(path)
    data = absolute(path).read_bytes()
    return {'path': str(path), 'sha256': sha(data), 'bytes': len(data)}


class Capsule:
    def __init__(self, directory):
        self.directory = directory
        self.filename = directory / 'manifest.json'
        self.manifest = json.loads(absolute(self.filename).read_text())
        require(canonical({k: v for k, v in self.manifest.items() if k != 'identity_sha256'}) == self.manifest['identity_sha256'], 'Capsule seal')
        self.contents = {}
        for name, record in self.manifest['records'].items():
            payload = disk(record['payload']).read_bytes()
            require(record['original']['path'] == name, 'Original attribution changed')
            if record['encoding'] == 'gzip':
                data = gzip.decompress(payload)
            else:
                require(record['encoding'] == 'identity', 'Unknown record encoding')
                data = payload
            original = record['original']
            require(sha(data) == original['sha256'] and len(data) == original['bytes'], 'Decoded original differs: ' + name)
            data.decode('utf-8')
            require(b'\0' not in data and not data.startswith((b'\x7fELF', b'!<arch>')), 'Native/cache payload')
            self.contents[name] = data
        require(len(self.contents) == self.manifest['original_record_count'], 'Record count')

    def data(self, name):
        return self.contents[str(name)]

    def json(self, name):
        return json.loads(self.data(name))

    def original(self, name):
        return self.manifest['records'][str(name)]['original']


def ledger(events, expected, allowed):
    starts, reaps = {}, {}
    for event in events:
        key = tuple(event[k] for k in ('pid', 'pgid', 'start_ticks'))
        require(key[0] == key[1] and key[2] > 0, 'Owned identity is not an isolated recorded group')
        if event['event'] == 'spawn':
            require(key not in starts, 'Duplicate spawn')
            starts[key] = event
        else:
            require(event['event'] == 'reaped' and key in starts and key not in reaps and event['reaped'] is True, 'Missing/duplicate/unreaped identity')
            require(event['returncode'] in allowed, 'Unexpected owned exit status')
            reaps[key] = event
    require(len(starts) == expected and starts.keys() == reaps.keys(), 'Unclosed owned process ledger')
    return starts, reaps


def run(live):
    capsule = Capsule(CORE)
    manifest = capsule.manifest
    report = Path(manifest['report'])
    status = capsule.json(report / 'run-status.json')
    require(status['status'] == 'complete' and len(status['completed_cases']) == 8 and status['failed_cases'] == {}, 'Completed core scope')
    require(status['requested_cases'] == status['completed_cases'], 'Missing core case')
    owner = capsule.json(manifest['owner_terminal_receipt'])
    require(capsule.original(manifest['owner_terminal_receipt'])['sha256'] == '3468e312b54d04f8e16bfdbfa7754ce5bd64e5a7f663022f396e56aca1f40ea1', 'Owner terminal custody')
    require(owner['controller']['session_exit_code'] == 0 and owner['performance_acceptance_gate'] is False, 'Owner terminal exit/scope')
    sources = json.loads(disk(manifest['source_authority']).read_text())
    references = json.loads(disk(manifest['historical_Ninja6_reference_authority']).read_text())
    refs = {entry['case']: entry for entry in references['entries'] if entry['backend'] == 6}
    require(set(refs) == set(status['completed_cases']) and len(refs) == 8, 'Exactly eight historical Ninja6 references required')
    protocol = capsule.json(report / 'protocol.json')
    require(protocol['backends'] == [6] and protocol['cpu'] == 14 and protocol['repetitions'] == 2, 'Actual optimized Ninja6 diagnostic scope')
    starts, reaps = ledger([json.loads(line) for line in capsule.data(report / 'process-ledger.jsonl').splitlines()], 25, {0})
    require(sum(e['command'][:4] == ['taskset', '-c', '14', './check'] for e in starts.values()) == 8, 'Actual worker CPU/command scope')
    worst, comparisons, bit_checks, cases = (0.0, None), 0, 0, {}
    totals = collections.Counter()
    for cid in status['completed_cases']:
        authority = next(item for item in sources['targets'].values() if item['case'] == cid)
        current_manifest = capsule.json(report / 'builds' / (cid + '-candidate-6') / 'compiled-sources.json')
        require(current_manifest == authority['compiled_manifest'], 'Current Fortran body differs from published final generation: ' + cid)
        require(canonical(current_manifest['files']) == current_manifest['files_sha256'], 'Source manifest seal')
        for name, key in [('generation.json', 'origin_generation'), ('compilation.json', 'origin_compilation'), ('compiled-sources.json', 'origin_manifest')]:
            current = report / 'inputs/candidate' / cid / name
            require(capsule.original(current)['sha256'] == authority[key]['sha256'], 'Original source provenance changed: ' + cid + '/' + name)
            disk(authority[key])
        runtime = authority['reducer_runtime']
        require(canonical({k: v for k, v in runtime.items() if k != 'identity_sha256'}) == runtime['identity_sha256'] == manifest['runtime_identity_sha256'], 'Current full runtime identity')
        reference = refs[cid]
        old = json.loads(disk(reference['record']).read_text())
        require(old['backend'] == 6 and old['source'] == 'baseline' and reference['target']['role'] == 'original_helper_corrected', 'Historical Ninja reference role')
        disk(reference['points'])
        require(sha(capsule.data(report / 'points' / (cid + '.dat'))) == reference['points']['sha256'] == old['points_sha256'], 'Different fixed point bytes: ' + cid)
        old_runtime = reference['target']['reducer_runtime']
        for name, digest in runtime['libraries'].items():
            basename = Path(name).name
            if basename in ('libninja.a', 'libavh_olo.a') or basename.startswith('libninja.so'):
                require(old_runtime['libraries'].get(name) == digest, 'Native Ninja/scalar provider changed: ' + name)
        guard = capsule.data(report / 'guards' / (cid + '.log')).decode().splitlines()
        require(guard == ['NULL_TEARDOWN_FORWARDED create=0 dp=0 compensated=0 mp=0 destroy=1 nonnull=0', 'GUARD_FINAL create=0 dp=0 compensated=0 mp=0 destroy=1 nonnull=0'], 'NinjaRS numerical dispatch guard')
        previous = None
        for suffix, phase in [('calibration', 'calibration-1'), ('0', 'diagnostic-2'), ('1', 'diagnostic-3')]:
            filename = report / 'data' / f'{cid}-candidate-6-{suffix}.json'
            actual = capsule.json(filename)
            require(actual['backend'] == 6 and actual['case'] == cid and actual['source'] == 'candidate' and actual['cpu'] == 14, 'Current target identity')
            require(actual['generation_source_identity'] == authority['generation_source_identity'], 'Current generator identity')
            require(actual['generated_sources_sha256'] == current_manifest['files_sha256'], 'Compiled source hash')
            for key in ('points_sha256', 'parameter_card_sha256', 'shared_numerical_settings'):
                require(actual[key] == old[key], 'Current/historical physical settings differ: ' + key)
            require(actual['shared_numerical_settings']['mlstabthres'] == 1e-9 and actual['shared_numerical_settings']['colliercomputeuvpoles'] is True and actual['shared_numerical_settings']['colliercomputeirpoles'] is True, 'Thresholds/full Laurent poles')
            rows = actual['rows']
            require([(r['point'], r['order']) for r in rows] == [(n, o) for n in range(1, 101) for o in (0, 1)], 'Missing/duplicated/reordered rows')
            require(previous is None or rows == previous, 'Diagnostic traversal changed')
            previous = rows
            rawdir = report / 'raw' / (cid + '-candidate-6') / phase
            dimensions = list(map(int, capsule.data(rawdir / 'dimensions.dat').split()))
            require(dimensions == [1, 1, 1], 'Raw split-order dimensions')
            rawrows = [line.split() for line in capsule.data(rawdir / 'values.dat').splitlines()]
            require(len(rawrows) == 200, 'Raw row count')
            counts = collections.Counter()
            for fields, row in zip(rawrows, rows):
                require(len(fields) == 8 and list(map(int, fields[:3])) == [row['point'], row['order'], row['status']], 'Raw point/order/status differs')
                numbers = [float(value.replace(b'D', b'E')) for value in fields[3:]]
                require(all(math.isfinite(x) for x in numbers) and row['status'] in (216, 328) and row['accuracy'] >= 0, 'Rejected/nonfinite raw result')
                for a, b in zip(numbers, row['coefficients'] + [row['accuracy']]):
                    require(struct.pack('>d', a) == struct.pack('>d', b), 'Raw/JSON float64 value differs')
                    bit_checks += 1
                if row['order'] == 0:
                    counts[str(row['status'])] += 1
            require(dict(counts) == actual['status_counts'], 'Raw status counts')
            rawtimes = list(map(float, capsule.data(rawdir / 'timing.dat').split()))
            require(rawtimes == [actual['cold_seconds'], actual['seconds']], 'Retained diagnostic time source')
            timed = {str(int(a)): int(b) for a, b in [line.split() for line in capsule.data(rawdir / 'timed_status.dat').splitlines()]}
            require(timed == actual['timed_status_counts'] and sum(timed.values()) == actual['loops'] * 100, 'Raw dispatch/evaluation counts')
            scale = {}
            for row in rows + old['rows']:
                scale[row['point']] = max(scale.get(row['point'], 0), *map(abs, row['coefficients']))
            require([(r['point'], r['order']) for r in old['rows']] == [(r['point'], r['order']) for r in rows], 'Historical order/point alignment')
            for a, b in zip(rows, old['rows']):
                for index, (x, y) in enumerate(zip(a['coefficients'], b['coefficients'])):
                    tolerance = 1e-7 * max(abs(x), abs(y)) + 1e-10 * scale[a['point']]
                    require(math.isfinite(y) and abs(x - y) <= tolerance, 'Ninja6 historical comparison failure')
                    fraction = abs(x - y) / tolerance if tolerance else 0
                    if fraction > worst[0]:
                        worst = (fraction, {'case': cid, 'traversal': suffix, 'point': a['point'], 'order': a['order'], 'coefficient': index})
                    comparisons += 1
            if suffix == 'calibration':
                totals.update(counts)
                cases[cid] = {'points': 100, 'rows': 200, 'calibration_status_counts': dict(counts),
                    'source_manifest_equal_to_public_final_target': True, 'diagnostic_traversals': 2,
                    'historical_reference': reference['record'], 'historical_role': 'original_helper_corrected/6',
                    'current_role': 'candidate_corrected/6', 'performance_qualification': False}
        if live:
            output = ROOT / '.benchmark-work/policy-v3/final-pins-20261021-v1/legacy-core-execution/run-v1/workers' / (cid + '-candidate-6') / 'outputs/candidate' / cid
            calibration = capsule.json(report / 'data' / (cid + '-candidate-6-calibration.json'))
            for name, info in current_manifest['files'].items():
                disk({'path': str(output / name), **info})
            for group in ('libraries', 'metadata'):
                for name, digest in runtime[group].items():
                    disk({'path': str(output / 'lib' / name), 'sha256': digest})
            for name, info in runtime['link_inputs'].items():
                disk({'path': str(output / name), **info})
            driver = next(output.glob('SubProcesses/P*/check_sa.f'))
            disk({'path': str(driver), 'sha256': calibration['driver_sha256']})
            disk({'path': str(driver.with_name('check')), 'sha256': calibration['executable_sha256']})
            for name, key in [('param_card.dat', 'parameter_card_sha256'), ('MadLoopParams.dat', 'madloop_card_sha256')]:
                disk({'path': str(output / 'Cards' / name), 'sha256': calibration[key]})
    require(comparisons == 19200 and bit_checks == 24000, 'Audit scalar coverage')
    cancelled = Capsule(CANCELLED)
    cancellation = cancelled.manifest
    terminal = cancelled.json(cancellation['terminal_receipt'])
    require(cancelled.original(cancellation['terminal_receipt'])['sha256'] == 'cc215604d4d7bfa47e3cacb8a098bf9f4bb84fc83c40e5b5751dfe325857b19e', 'Cancellation terminal custody')
    require(cancellation['classification'] == 'user-cancelled' and cancellation['accepted_calibrations'] == cancellation['accepted_diagnostic_traversals'] == 0, 'Cancellation scientific scope')
    parent = cancelled.json(Path(cancellation['report']) / 'run-status.json')
    require(parent['status'] == 'failed' and parent['error_type'] == 'SystemExit' and 'signal 15' in parent['error'], 'Original interrupted parent truth')
    require(terminal['controller_exit_code'] == 1 and terminal['MP190_explicitly_released'] is True and terminal['live_owned_group_members'] == [], 'Cancellation cleanup/release')
    require(all(x['proc_identity_absent'] and x['isolated_group_absent'] for x in terminal['identity_and_group_absence']), 'Cancellation ownership absence record')
    require(terminal['accepted_calibrations'] == terminal['accepted_diagnostic_traversals'] == 0 and terminal['rotation_launched'] is False and terminal['scale_launched'] is False, 'Unlaunched cancellation phases')
    _, stopped = ledger([json.loads(line) for line in cancelled.data(Path(cancellation['report']) / 'process-ledger.jsonl').splitlines()], 3, {0, -15})
    require(sorted(event['returncode'] for event in stopped.values()) == [-15, 0, 0], 'Cancellation expected child exits')
    require(not any('/data/' in name for name in cancelled.contents), 'Cancelled run invented accepted data')
    return {'schema': 'retained-optimized-Ninja6-core-independent-stdlib-audit-v1', 'passed': True,
        'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'reader': bound(Path(__file__).relative_to(ROOT)),
        'capsules': {'Ninja6_correctness': bound(CORE / 'manifest.json'), 'heavy8_user_cancelled': bound(CANCELLED / 'manifest.json')},
        'live_preserved_native_source_runtime_and_executable_hashes_verified': live,
        'cases': cases, 'unique_calibration_points': 800, 'accepted_saved_traversals': 24,
        'raw_float64_bit_comparisons': bit_checks, 'Ninja6_historical_scalar_comparisons': comparisons,
        'Ninja6_historical_scalar_failures': 0, 'maximum_fixed_tolerance_fraction': worst[0], 'worst_comparison': worst[1],
        'calibration_status_totals': dict(totals), 'recorded_owned_groups_reaped': 25,
        'NinjaRS_numerical_dispatch_in_Ninja6_controls': 0, 'normal_null_teardown_allowed': True,
        'cancellation_classification': 'user-cancelled; zero accepted calibration/traversals; no numerical failure claim',
        'cancelled_owned_groups_reaped': 3, 'new_numerical_evaluations_builds_or_tests': 0,
        'performance_qualification': False, 'scope_limits': ['Only same historical Ninja6 references are recomputed; no CutTools/Collier comparison.',
        'Raw diagnostic times remain retained evidence and do not qualify performance.',
        'Source/runtime aliases preserve attribution; no original/candidate or old/new runtime role is relabelled.',
        'Default mode checks retained evidence only; live native hashes are optional and never embedded.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    value = run(args.live)
    text = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if args.output:
        with absolute(args.output).open('x') as stream:
            stream.write(text)
    print(text, end='')
