"""File-free replay of the frozen ordinary-n1 host-repair measurements.

The caller supplies already decoded logical payload bytes. Absolute historical
paths are provenance strings only. This module never opens a file or executes a
process; only the two exact approved pure parsers may be compiled in memory.
"""
import builtins
import hashlib
import json
import math
import re
import statistics
import struct

SOURCE_INDEX_SHA256 = '97cad155079138419c198599adbbbcc00bb8df0a3ac799b8ef9ec9dd9858bace'
ANALYZE_SHA256 = 'c33e1b7f5a95bad7f7064fee20f7cd757413ce0ed0c51201d598ddf1f6f4e8e9'
WARMUP_SHA256 = 'a85ea55dc9f432e533189369c96ce0fc503fc859fe047085ad1d57de2b4ef1fe'
PRODUCTION_REVISION = '0fcd2cd873d61d134837e1d9e87afb13a47824fd'
ACTUAL_EXPORT_REVISION = '85c6211b146ec8fd2f1ee9c283fd92ee2de9d8a9'
EXPORTER_SHA256 = 'afb6e0bea64ea0932751ff4869c206bcea9477d40a39a54bb99d0f06d85b75a6'
BUILD_TERMINAL_SHA256 = '663b3947123be881864b78179bb8caaa19870277ed654c1232219699a992e5b2'
PARITY_TERMINAL_SHA256 = '5218212507e983ae8400c6396c32da84d4aed180660addac7436297d5ec761d8'
LANES = ('baseline', 'observed-0', 'observed-1', 'observed-2')


def require(value, message):
    if not value:
        raise ValueError(message)


def _get(data, name):
    require(name in data and type(data[name]) is bytes, 'Missing byte payload: '+name)
    return data[name]


def _text(data, name):
    return _get(data, name).decode('utf-8')


def _json(data, name):
    def pairs(values):
        result = {}
        for key, value in values:
            require(key not in result, 'Duplicate JSON key: '+name)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON constant: '+name)
    return json.loads(_text(data, name), object_pairs_hook=pairs, parse_constant=invalid)


def _hash(payload):
    return hashlib.sha256(payload).hexdigest()


def _same(actual, expected, message):
    # JSON normalization handles source summary keys1/2; serialization also
    # keeps bool/int/float distinctions which Python dictionary equality loses.
    def canonical(value):
        normalized = json.loads(json.dumps(value, allow_nan=False))
        return json.dumps(normalized, sort_keys=True, separators=(',', ':'), allow_nan=False)
    require(canonical(actual) == canonical(expected), message)


def _module(data, name, checksum):
    payload = _get(data, name)
    require(_hash(payload) == checksum, 'Changed approved pure parser: '+name)
    def approved_import(module, globals=None, locals=None, fromlist=(), level=0):
        require(level == 0 and module in ('math', 're', 'statistics'), 'Unapproved pure-parser import')
        return {'math': math, 're': re, 'statistics': statistics}[module]
    safe_builtins = dict(vars(builtins))
    safe_builtins['__import__'] = approved_import
    namespace = {'__name__': 'capsule_pure_'+name.replace('/', '_'), '__builtins__': safe_builtins}
    exec(compile(payload.decode('utf-8'), '<capsule:'+name+'>', 'exec'), namespace)
    return namespace


def _bound_input(data, pin, name, origin):
    matches = [row for path, row in pin['files'].items()
               if origin == path or origin.endswith('/'+path)]
    require(len(matches) == 1, 'Ambiguous frozen input provenance: '+name)
    payload = _get(data, name)
    require(_hash(payload) == matches[0]['sha256'] and len(payload) == matches[0]['bytes'],
            'Changed frozen input: '+name)


def _timing(text):
    values = list(map(float, text.split()))
    require(len(values) == 2 and all(math.isfinite(v) and v > 0 for v in values),
            'Invalid raw timing pair')
    return values


def _decimal_rows(text):
    rows = []
    for line in text.splitlines():
        fields = line.split()
        require(len(fields) == 8, 'Malformed decimal answer row')
        point, order, status = map(int, fields[:3])
        values = list(map(float, fields[3:]))
        require(all(math.isfinite(v) for v in values), 'Nonfinite decimal answer')
        rows.append({'point': point, 'order': order, 'status': status,
                     'words': [struct.pack('>d', v).hex().upper() for v in values]})
    return rows


def replay(data):
    """Recompute the scientific witnesses from provided logical payload bytes."""
    require(type(data) is dict and all(type(k) is str and type(v) is bytes for k, v in data.items()),
            'Expected a plain dictionary of logical names to bytes')
    require(_hash(_get(data, 'source/source-index.json')) == SOURCE_INDEX_SHA256,
            'Changed frozen source seal')
    seal = _json(data, 'source/source-index.json')
    pin_bytes = _get(data, 'source/input-pins.json')
    require(_hash(pin_bytes) == seal['files']['input-pins.json']['sha256'] and
            len(pin_bytes) == seal['files']['input-pins.json']['bytes'], 'Changed frozen input pins')
    pin = _json(data, 'source/input-pins.json')
    analysis = _module(data, 'source/analyze.py', ANALYZE_SHA256)
    warmup = _module(data, 'replay_warmup.py', WARMUP_SHA256)['replay_warmup']
    _bound_input(data, pin, 'inputs/closed-input.json', pin['closed_input'])
    _bound_input(data, pin, 'inputs/closed-observation.json', pin['closed_observation'])
    _bound_input(data, pin, 'inputs/closed-warmup.dat', pin['closed_observation'].rsplit('/', 1)[0]+'/warmup.dat')
    _bound_input(data, pin, 'inputs/points.dat', pin['points'])
    require(pin['points'].endswith('.dat'), 'Unexpected point provenance')
    _bound_input(data, pin, 'inputs/points.json', pin['points'][:-4]+'.json')
    closed_input = _json(data, 'inputs/closed-input.json')
    closed = _json(data, 'inputs/closed-observation.json')
    generation = closed_input['generation']
    _same(generation['case'], {'id': 'qcd_n1_uux_z', 'model': 'loop_sm', 'nfinal': 1,
          'process': 'u u~ > z [virt=QCD]', 'role': 'main'}, 'Unsupported physical case')
    require(generation['source'] == 'candidate' and
            generation['source_identity']['revision'] == ACTUAL_EXPORT_REVISION and
            generation['source_identity']['files_sha256'] == EXPORTER_SHA256 and
            generation['source_identity']['dirty'] is False, 'Unsupported exporter source')
    require(closed_input['settings'] == pin['settings'] and
            pin['settings']['mlreductionlib'] == '9' and pin['settings']['improvepspoint'] == 2 and
            pin['settings']['ctmoderun'] == -1, 'Changed explicit backend/repair policy')
    require(_hash(_get(data, 'inputs/points.dat')) == closed_input['points_sha256'] and
            _hash(_get(data, 'inputs/points.json')) == closed_input['points_metadata_sha256'],
            'Saved points differ from closed physical inputs')
    points = _json(data, 'inputs/points.json')
    require(points['inputs']['count'] == 100 and len(points['points']) == 100,
            'Incomplete saved physical point stream')
    reference = _get(data, 'inputs/closed-warmup.dat')
    require(closed['raw_outputs']['warmup.dat'].encode('utf-8') == reference,
            'Closed learning text/reference differ')
    require(closed['status_counts'] == {'219': 100} and closed['native_dp_only'] is True,
            'Closed reference is not the ordinary DP219 cohort')

    parity = _json(data, 'results/parity/result.json')
    profile = _json(data, 'results/profile/result.json')
    require(parity['status'] == profile['status'] == 'passed' and
            parity['source_index_sha256'] == profile['source_index_sha256'] == SOURCE_INDEX_SHA256 and
            parity['build_terminal_sha256'] == profile['build_terminal_sha256'] == BUILD_TERMINAL_SHA256 and
            profile['parity_terminal_sha256'] == PARITY_TERMINAL_SHA256, 'Mismatched measured stage lineage')
    require(parity['performance_acceptance'] is False and profile['performance_acceptance'] is False and
            parity['timers_from_parity_are_qualification_only'] is True and profile['optimized'] is False,
            'Measurement claims unsupported optimization or acceptance')
    require(profile['loops_per_batch'] == 128 and profile['points_per_traversal'] == 100 and
            set(parity['lanes']) == set(LANES), 'Changed physical coverage')
    learning_records = []
    exact_records = []
    probes = []
    baseline = None
    for phase in ('parity', 'profile'):
        for lane in LANES:
            mode = 0 if lane == 'baseline' else int(lane[-1])
            observed = lane != 'baseline'
            prefix = 'results/'+phase+'/'+lane+'/'
            ready = _json(data, prefix+'ready.json')
            observation = _json(data, prefix+'warmup.json')
            learning = _get(data, 'learning/'+phase+'/'+lane+'.dat')
            require(learning == reference and observation['raw_outputs']['warmup.dat'].encode('utf-8') == learning,
                    'Complete learning/bootstrap bytes changed')
            parsed_warmup = warmup(observation, closed, ready)
            _same(parsed_warmup, observation, 'Derived learning record differs')
            require(parsed_warmup['warmup_first_call'] == 1 and parsed_warmup['warmup_last_call'] == 100,
                    'Learning is not the full frozen100 calls')
            call_ids = {int(line.split()[0]) for line in learning.decode('utf-8').splitlines()
                        if line.strip() and not line.lstrip().startswith('#')}
            require(call_ids == set(range(101)), 'Missing bootstrap or full learning call indices')
            learning_records.append({'phase': phase, 'lane': lane, 'learning_sha256': _hash(learning),
                                     'steady_calls': 100, 'bootstrap_calls_diagnostic': 1,
                                     'cold_seconds': ready['cold_seconds']})
            raw = prefix+('exact/' if phase == 'parity' else 'exact-before-timing/')
            exact = {'rows': analysis['exact'](_text(data, raw+'dp-exact.dat')),
                     'ieee': analysis['ieee'](_text(data, raw+'dp-ieee.dat')),
                     'profile': analysis['profile'](_text(data, raw+'dp-parity-timers.tsv'), mode, 100, observed),
                     'probe': analysis['probe'](_text(data, raw+'dp-probe.tsv'))}
            if baseline is None:
                baseline = exact
            _same(exact['rows'], baseline['rows'], 'Full100 exact coefficient/status/accuracy words changed')
            _same(exact['ieee'], baseline['ieee'], 'Full100 actual IEEE observations changed')
            if phase == 'parity':
                _same(exact, parity['lanes'][lane], 'Stored PARITY observations differ from raw data')
            require(exact['profile']['scopes'][1]['direct_count'] == 0,
                    'Unexpected direct MP call in the recorded ordinary cohort')
            exact_records.append({'phase': phase, 'lane': lane, 'answer_rows': len(exact['rows']),
                                  'IEEE_rows': len(exact['ieee']), 'profile': exact['profile']})
            probes.append({'phase': phase, 'lane': lane, **exact['probe']})

    cycles = []
    batch_records = []
    ratios = {lane: [] for lane in LANES[1:]}
    scope_seconds = {1: [], 2: []}
    whole_seconds = []
    fractions = []
    require(len(profile['cycles']) == 7, 'Incomplete seven-cycle campaign')
    for number, stored in enumerate(profile['cycles'], 1):
        require(type(stored['cycle']) is int and stored['cycle'] == number, 'Changed cycle sequence')
        order = list(LANES)
        shift = (number-1) % 4
        order = order[shift:]+order[:shift]
        if number % 2 == 0:
            order.reverse()
        require(stored['order'] == order and set(stored['lanes']) == set(LANES), 'Changed fixed lane schedule')
        _same(stored, _json(data, 'results/profile/cycle-'+str(number)+'.json'), 'Per-cycle stored record differs')
        derived = {'cycle': number, 'order': order, 'lanes': {}}
        flags_reference = None
        for lane in order:
            mode = 2 if lane == 'baseline' else int(lane[-1])
            raw = 'results/profile/'+lane+'/cycle-'+str(number)+'/'
            cold, seconds = _timing(_text(data, raw+'timing.dat'))
            status = [list(map(int, line.split())) for line in _text(data, raw+'timed_status.dat').splitlines() if line.strip()]
            require(status == [[219, 12800]], 'Timed status/precision/failure coverage changed')
            rows = analysis['exact'](_text(data, raw+'dp-timed-exact.dat'))
            _same(rows, baseline['rows'], 'Final timed full100 words changed')
            _same(_decimal_rows(_text(data, raw+'values.dat')), rows, 'Decimal/raw timed words differ')
            flags = analysis['ieee'](_text(data, raw+'dp-timed-ieee.dat'), timed=True)
            if flags_reference is None:
                flags_reference = flags
            _same(flags, flags_reference, 'Timed-batch lane IEEE observations differ')
            measured = analysis['profile'](_text(data, raw+'dp-timers.tsv'), mode, 12800, lane != 'baseline')
            require(measured['scopes'][1]['direct_count'] == 0, 'Unexpected direct MP repair in timing')
            result = {'seconds': seconds, 'timed_status_counts': {'219': 12800}, 'ieee': flags, 'profile': measured}
            _same(result, stored['lanes'][lane], 'Stored timed observation differs from raw files')
            derived['lanes'][lane] = result
            batch_records.append({'cycle': number, 'lane': lane, 'seconds': seconds, 'cold_seconds': cold,
                                  'amplitude_calls': 12800, 'public_status': 219,
                                  'final_answer_rows': len(rows), 'batch_IEEE_rows': len(flags),
                                  'DP_calls': measured['scopes'][0]['count'],
                                  'nested_MP_calls': measured['scopes'][1]['nested_count'],
                                  'direct_MP_calls': measured['scopes'][1]['direct_count']})
        for lane in ratios:
            ratios[lane].append(derived['lanes'][lane]['seconds']/derived['lanes']['baseline']['seconds'])
        clock = derived['lanes']['observed-2']
        for stage in (1, 2):
            scope_seconds[stage].append(clock['profile']['scopes'][stage-1]['seconds']/12800)
        whole_seconds.append(clock['seconds']/12800)
        fractions.append(clock['profile']['scopes'][0]['seconds']/clock['seconds'])
        cycles.append(derived)
    summary = analysis['summary'](cycles)
    _same(summary, profile['summary'], 'Stored derived summary differs from raw measurements')
    require(summary['performance_acceptance'] is summary['speedup_claimed'] is
            summary['all_five_backends_qualified'] is False, 'Unsupported summary qualification claim')
    medians = {'DP_repair_inclusive_seconds_per_amplitude': statistics.median(scope_seconds[1]),
               'MP_core_seconds_per_amplitude': statistics.median(scope_seconds[2]),
               'clock_lane_whole_seconds_per_amplitude': statistics.median(whole_seconds),
               'DP_scope_fraction_of_clock_lane_whole': statistics.median(fractions),
               'overhead_ratios': {lane: statistics.median(values) for lane, values in ratios.items()}}
    return {'status': 'measurement-replay-passed', 'source_index_sha256': SOURCE_INDEX_SHA256,
            'source_scope': {'production_numerical_revision': PRODUCTION_REVISION,
                'actual_export_revision': ACTUAL_EXPORT_REVISION, 'exporter_inventory_sha256': EXPORTER_SHA256,
                'case': generation['case'], 'backend': 9, 'precision': 'ordinary accepted DP219'},
            'learning': learning_records, 'exact_qualifications': exact_records, 'probes': probes,
            'timed_batches': batch_records, 'summary': json.loads(json.dumps(summary)), 'medians': medians,
            'witness_counts': {'saved_points': 100, 'learning_traces': 8, 'steady_learning_calls': 800,
                'bootstrap_calls_diagnostic': 8, 'full100_exact_qualification_traversals': 8,
                'exact_qualification_answer_rows': 1600, 'exact_qualification_IEEE_rows': 1600,
                'timed_batches': 28, 'timed_amplitude_calls': sum(row['amplitude_calls'] for row in batch_records),
                'final_timed_answer_rows': sum(row['final_answer_rows'] for row in batch_records),
                'timed_batch_IEEE_rows': sum(row['batch_IEEE_rows'] for row in batch_records),
                'empty_probe_brackets': sum(row['repetitions'] for row in probes)},
            'probe_subtracted': False, 'speedup_claimed': False, 'performance_acceptance': False,
            'all_five_backends_qualified': False, 'promoted': False,
            'limitations': ['Exact timed numerical words cover each batch final100 traversal, not every inner answer.',
                'IEEE fields observe pre/post qualification calls and timed batches, not transient flags or trap freedom.',
                'DP inclusive time contains nested MP work and instrumentation overhead; intervals are never probe-corrected.',
                'Only the frozen ordinary n1 explicit backend9 case is qualified; precise local timings remain separate.']}
