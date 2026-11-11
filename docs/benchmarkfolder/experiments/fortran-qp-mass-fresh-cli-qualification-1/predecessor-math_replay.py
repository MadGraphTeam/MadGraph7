"""Pure numerical replay for the current QP-mass retention draft.

All inputs are authenticated, preloaded bytes.  This module does not open files,
execute historical controllers, invoke a reducer, or infer a pass for a missing
or failed stage.  The caller owns source seals and runtime/publication checks.
"""
import ast
from collections import Counter
from collections.abc import Mapping
import hashlib
import json
import math
import struct


SOURCE = '3c0f53a6e5526535b6fae98b0f055466b429a9a9cfc863fb22fb999a5c8a0815'
STAGES = ('build', 'units', 'cache0', 'cacheall', 'observeoriginal', 'observecandidate')
ANCESTORS = {
    'build': (),
    'units': ('build',),
    'cache0': ('build', 'units'),
    'cacheall': ('build', 'units', 'cache0'),
    'observeoriginal': ('build', 'units', 'cache0', 'cacheall'),
    'observecandidate': ('build', 'units', 'cache0', 'cacheall', 'observeoriginal'),
}
EXACT_SCOPE = ('Public full100 words/status,read-only host IEEE/software flags; '
               'not a trap-freedom result')


def require(value, message):
    if not value:
        raise ValueError(message)


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _words(rows):
    return [{'point': r['point'], 'order': r['order'], 'status': r['status'],
             'words': [struct.pack('>d', v).hex().upper()
                       for v in r['coefficients'] + [r['accuracy']]]}
            for r in rows]


def _software(text):
    rows = [list(map(int, line.split())) for line in text.splitlines() if line.strip()]
    require(len(rows) == 200 and [r[:2] for r in rows] ==
            [[p, phase] for p in range(1, 101) for phase in (1, 2)],
            'Incomplete software-latch trace')
    require(all(len(r) == 4 and all(x in (0, 1) for x in r[2:]) for r in rows),
            'Malformed software-latch trace')
    return rows


def _history(text):
    out = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        require(len(fields) == 9, 'Malformed learning record')
        out.append(list(map(int, fields[:4])))
    return out


def _learning_rows(text, codes):
    """Independently parse the final steady window; bootstrap stays diagnostic."""
    groups = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        fields = line.split()
        require(len(fields) == 9, 'Malformed learning row')
        call = int(fields[0])
        require(call >= 0, 'Negative learning call index')
        if not groups or groups[-1][0] != call:
            require(not groups or call > groups[-1][0], 'Unordered learning calls')
            groups.append((call, []))
        groups[-1][1].append(fields)
    steady = [g for g in groups if g[0] > 0]
    require(len(steady) >= 100, 'No full steady learning traversal')
    selected = steady[-100:]
    calls = [g[0] for g in selected]
    require(calls == list(range(calls[0], calls[0] + 100)),
            'Final learning calls are not consecutive')
    rows = []
    order = []
    for call, records in selected:
        point = 1 + (call - 1) % 100
        require(len(records) == 2, 'Learning split-order coverage differs')
        current = []
        for fields in records:
            p, split, status = map(int, fields[1:4])
            values = [float(x) for x in fields[4:9]]
            require(p == point and status == codes[str(point)],
                    'Learning point or actual precision/status differs')
            require(all(math.isfinite(x) for x in values) and values[-1] >= 0,
                    'Learning values or stability estimate invalid')
            current.append({'point': p, 'order': split, 'status': status,
                            'coefficients': values[:4], 'accuracy': values[4],
                            'actual_backend': 9,
                            'precision': 'DP' if status == 219 else 'QP'})
        require(sorted(r['order'] for r in current) == [0, 1],
                'Duplicate or missing learning split order')
        rows.extend(current)
        order.append({'call': call, 'point': point})
    rows.sort(key=lambda r: (r['point'], r['order']))
    require([(r['point'], r['order']) for r in rows] ==
            [(p, o) for p in range(1, 101) for o in (0, 1)],
            'Steady learning window does not cover all points')
    return rows, calls, order


def replay(blobs, load_source):
    """Replay closed numerical stages from ``dict[str, bytes]`` only.

    ``load_source(name, bytes, logical_path)`` executes only authenticated pure
    parser source supplied here.  The sole campaign function is extracted as an
    unchanged AST source segment; the benchmark module itself is never imported.
    Missing stages are pending.  Failed stages stay failed and unqualified.
    """
    require(isinstance(blobs, Mapping), 'Expected preloaded logical byte mapping')

    def data(name):
        value = blobs[name]
        require(type(value) is bytes, 'Logical payload is not bytes: ' + name)
        return value

    def text(name):
        return data(name).decode('utf-8')

    def record(name):
        return json.loads(data(name))

    require(_sha(data('source-2/source-index.json')) == SOURCE, 'Wrong source2 seal')
    source = 'source-2/'
    parsers = {}
    for label, relative in (('base', 'base_analysis.py'),
                            ('unit', 'units/parser.py'),
                            ('observer', 'observer_analysis.py')):
        parsers[label] = load_source('mass_portable_' + label,
                                     data(source + relative), source + relative)
    whole = text('support/tools/tirust_benchmark.py')
    definitions = [n for n in ast.parse(whole).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'compare_rows']
    require(len(definitions) == 1, 'Expected exact existing comparison function')
    segment = ast.get_source_segment(whole, definitions[0])
    require(segment is not None and not definitions[0].decorator_list,
            'Existing comparator extraction differs')
    comparator = load_source('mass_portable_existing_comparator', segment.encode(),
                             'support/tools/tirust_benchmark.py::compare_rows').compare_rows
    codes = record(source + 'input-pins.json')['cases']['n3']['expected_codes']
    require(codes == {str(p): (323 if p in parsers['base'].N3_QP_POINTS else 219)
                      for p in range(1, 101)}, 'Frozen n3 status map changed')

    def compare(left, right):
        value = parsers['base'].compare_variants(left, right, comparator)
        require(value['passed'], 'Legacy and hard1e-8 coefficient gate failed')
        changes = []
        for l, r in zip(left, right, strict=True):
            if l['status'] == 219:
                require(l == r, 'DP coefficient/status/accuracy words changed')
            elif l['words'] != r['words']:
                changes.append({'point': l['point'], 'order': l['order'],
                                'original_words': l['words'], 'candidate_words': r['words']})
        return {**value, 'DP_exact': True, 'QP_changed_rows': changes,
                'QP_input_precision_change_not_independent_finite_truth': True}

    states = {}
    results = {}
    for stage in STAGES:
        prefix = 'stages/' + stage + '/'
        if prefix + 'terminal.json' not in blobs:
            states[stage] = 'pending'
            continue
        terminal = record(prefix + 'terminal.json')
        if terminal['status'] != 'passed':
            require(terminal['status'] == 'failed', 'Unknown terminal stage status')
            states[stage] = 'failed'
            continue
        result = record(prefix + 'result.json')
        require(result['status'] == 'passed' and terminal['returncode'] == 0 and
                not terminal['cleanup']['survivors'], 'Stage is not successfully closed')
        require(terminal['source_index_sha256'] == result['source_index_sha256'] == SOURCE,
                'Stage uses another source')
        require(terminal['result_sha256'] == _sha(data(prefix + 'result.json')),
                'Stage result does not join terminal')
        require(result['stage'] == stage and terminal['stage'] == stage,
                'Stage identity differs')
        require(all(states.get(s) == 'passed' for s in ANCESTORS[stage]),
                'Stage passed without passed predecessors')
        if stage != 'build':
            require(result['reviewed'] == {
                s: _sha(data('stages/' + s + '/terminal.json'))
                for s in ANCESTORS[stage]}, 'Reviewed stage ancestry differs')
            require(result['candidate_promoted'] is False and
                    result['performance_acceptance'] is False and
                    result['independent_physical_finite_truth_qualified'] is False,
                    'Unqualified promotion/truth/performance claim')
        for name, checksum in result.get('raw_files', {}).items():
            require(_sha(data(prefix + name)) == checksum, 'Changed recorded raw evidence')
        states[stage] = 'passed'
        results[stage] = result

    numerical = {}
    if states['units'] == 'passed':
        prefix = 'stages/units/'
        actual = parsers['unit'].parse(text(prefix + 'commands/recording-units.stdout.txt'))
        require(normalized(actual) == record(prefix + 'parsed.json') == results['units']['parsed'],
                'Recording unit parser/result differs')
        require(results['units']['actual_reducer_executions'] == 0,
                'Recording fixture mislabelled as a real reducer')
        numerical['units'] = actual

    for stage in ('cache0', 'cacheall', 'observeoriginal', 'observecandidate'):
        if states[stage] != 'passed':
            continue
        result = results[stage]
        expected_variants = (('original', 'candidate') if stage in ('cache0', 'cacheall')
                             else (('original-observed',) if stage == 'observeoriginal'
                                   else ('candidate-observed',)))
        require(set(result['lanes']) == set(result['learning']) == set(expected_variants),
                'Physical stage variant coverage differs')
        require(result['cache'] == (-1 if stage == 'cacheall' else 0) and
                result['full100_learning_per_worker'] is True and
                result['points_per_worker'] == 100 and result['DP_points'] == 87 and
                result['QP_points'] == 13 and result['UseQPIntegrandForTIRust'] is True,
                'Physical scope or cache setting differs')
        lanes = {}
        learnings = {}
        for variant in expected_variants:
            prefix = 'stages/' + stage + '/' + variant + '/'
            rows = parsers['base'].exact(text(prefix + 'exact/exact-words.dat'), codes)
            ieee = parsers['base'].ieee(text(prefix + 'exact/exact-ieee.dat'))
            software = _software(text(prefix + 'exact/software-flags.dat'))
            lane = {'rows': rows, 'ieee': ieee, 'software': software, 'scope': EXACT_SCOPE}
            require(normalized(lane) == result['lanes'][variant], 'Exact physical lane differs')
            require(Counter(r['status'] for r in rows) == {219: 174, 323: 26},
                    'Physical automatic-rescue cohort changed')
            ready = record(prefix + 'ready.json')
            require([ready[k] for k in ('points', 'ndim', 'nso', 'nborn')] == [100, 1, 1, 1]
                    and math.isfinite(ready['cold_seconds']) and ready['cold_seconds'] > 0,
                    'Physical READY scope differs')
            require(list(map(int, text(prefix + 'dimensions.dat').split())) == [1, 1, 1],
                    'Physical dimensions differ')
            warmup_text = text(prefix + 'warmup.dat')
            warmup_rows, calls, order = _learning_rows(warmup_text, codes)
            warmup = record(prefix + 'warmup.json')
            require(warmup['rows'] == warmup_rows and warmup['warmup_first_call'] == calls[0]
                    and warmup['warmup_last_call'] == calls[-1] and
                    warmup['warmup_call_point_order'] == order and
                    warmup['raw_outputs'] == {'warmup.dat': warmup_text,
                                               'dimensions.dat': text(prefix + 'dimensions.dat')},
                    'Warmup observation/raw trace differs')
            require(warmup['cold_seconds'] == ready['cold_seconds'] and
                    warmup['seconds'] is None and warmup['seconds_per_point'] is None and
                    warmup['timed_status_counts'] is None and
                    warmup['status_counts'] == {'219': 87, '323': 13} and
                    warmup['backend_precision_counts'] == {'9:DP': 87, '9:QP': 13} and
                    warmup['native_dp_only'] is False and
                    warmup['answer_dimensions'] == {'allocated': 1, 'loop_orders': 1, 'born_orders': 1} and
                    warmup['measurement_phase'] == 'steady-warmup' and
                    warmup['performance_eligible'] is False,
                    'Warmup precision counts or unmeasured timing scope differs')
            history = _history(warmup_text)
            require(history == _history(text('ancestry/ordinary-warmup.dat')),
                    'Learning chronology differs from frozen ordinary input')
            words = _words(warmup_rows)
            historic_words = _words(record('ancestry/ordinary-observation.json')['rows'])
            historical_rows, _, _ = _learning_rows(text('ancestry/ordinary-warmup.dat'), codes)
            require(_words(historical_rows) == historic_words,
                    'Historical steady raw values do not join frozen observation')
            comparison = compare(historic_words, words)
            learning = {'ready': ready, 'rows': words, 'history': history, 'comparison': comparison}
            require(normalized(learning) == result['learning'][variant],
                    'Retained learning summary differs')
            diagnostic = record(prefix + 'learning-comparison.json')
            require(diagnostic['history_exact'] is True and diagnostic['bootstrap_raw_retained'] is True
                    and diagnostic['comparison'] == normalized(comparison),
                    'Learning comparator/diagnostic differs')
            lanes[variant] = lane
            learnings[variant] = learning
        if stage in ('cache0', 'cacheall'):
            require(learnings['original']['history'] == learnings['candidate']['history'],
                    'Original/candidate learning histories differ')
            compare(learnings['original']['rows'], learnings['candidate']['rows'])
            comparison = compare(lanes['original']['rows'], lanes['candidate']['rows'])
            require(normalized(comparison) == result['comparison'], 'Physical comparator differs')
            require(lanes['original']['ieee'] == lanes['candidate']['ieee'] and
                    lanes['original']['software'] == lanes['candidate']['software'],
                    'Observed hardware/software states differ')
            if stage == 'cacheall':
                require(normalized(lanes) == results['cache0']['lanes'],
                        'Cache0/all own exact lanes differ')
            numerical[stage] = {'comparison': comparison, 'points': 100,
                                'DP_points': 87, 'automatic_QP_points': 13}
        else:
            role = int(stage == 'observecandidate')
            variant = 'candidate-observed' if role else 'original-observed'
            own = 'candidate' if role else 'original'
            require(normalized(lanes[variant]) == results['cache0']['lanes'][own],
                    'Observer changed its own plain physical lane')
            prefix = 'stages/' + stage + '/' + variant + '/'
            observed = parsers['observer'].parse(text(prefix + 'mass-observer.hex'), role)
            proof = record(source + 'source-proof.json')
            mass = parsers['observer'].validate_mass_words(observed, proof)
            require(normalized(mass) == record(prefix + 'observed-mass-summary.json'),
                    'Observed mass summary differs')
            numerical[stage] = mass
            if role:
                original = parsers['observer'].parse(text(
                    'stages/observeoriginal/original-observed/mass-observer.hex'), 0)
                joined = parsers['observer'].compare(original, observed, proof)
                require(normalized(joined) == record('stages/observecandidate/native-input-comparison.json'),
                        'Native input attribution differs')
                numerical['native_input_comparison'] = joined

    return {'stage_status': states, 'pending_stages': [s for s in STAGES if states[s] == 'pending'],
            'failed_stages': [s for s in STAGES if states[s] == 'failed'],
            'all_planned_stages_closed_pass': all(v == 'passed' for v in states.values()),
            'numerical': normalized(numerical), 'source_sha256': SOURCE,
            'new_native_executions': 0, 'candidate_promoted': False,
            'independent_physical_finite_truth_qualified': False, 'performance_acceptance': False,
            'scope': 'Replay of retained recording/ordinary-SM mass precision evidence only.'}
