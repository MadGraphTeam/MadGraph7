"""Pure portable replay. Recorded absolute paths are identity strings only."""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import struct
import types

SOURCE1 = 'eb7c19bb990142e8adb955c9fe21191a4f2d364ecf853a03bb001b0a571e02f2'
SOURCE2 = 'e633405b99d782ad6cf021be875b258f13e5669a7113c3890c40fa36eca6160f'
FAILED = '099ec26ab93219e4adef0a6f5b74f037d95a1fb5f618eb47bbbdf029c2189ffb'
PARSER = 'a245f363947466dc905d234bc5ca1061e8e347ee557b5a982de3978a7c1c40ed'
WARMUP_PARSER = 'a85ea55dc9f432e533189369c96ce0fc503fc859fe047085ad1d57de2b4ef1fe'
STAGES = {
    'build': ('16b15cc8db66880abf1fbf127ec0c2ceef2c9226446e30bd7fd6ab61b85c7470', 'fc680d285024c3136904d9f45f9ccd2b7d68d2423b884124938ac4cd7a4956b2'),
    'parity': ('4f34848d7888a562dc631bbe5792c65a296dc3b7516b19ab284a9627c25d07f7', '1416b8e74e4f311e860cfc8d6ce6132d8df4b5727d444bf1b33019b1dbf1a583'),
    'profile': ('c59cdd7b14b85706df9d28249b3fb2f608532be480a99b51a97a6721760a6568', 'eaf56fab60ac3f65a6f7118724a82bf7f023e3fba2a3db1b8e928affad437c18'),
}
AUDITS = {'build-parity': '7ec5d8dafc24d0bbe83956d36b8cf22e6ce524fe122b03d2f933be2396336f99',
          'profile': '226c53f8d2f81f8e99b29e7ac33c480de8628d3a12859153a2aad7534b0d91f0'}
LANES = ('baseline', 'observed-0', 'observed-1', 'observed-2')


def verify(root=None, *, read_bytes=None, derive_summary=False, expected_index_sha256=None):
    def require(value, message):
        if not value:
            raise ValueError(message)
    def sha(raw):
        return hashlib.sha256(raw).hexdigest()
    def safe(name):
        require(isinstance(name, str) and name and '\\' not in name, 'Invalid capsule name')
        path = PurePosixPath(name)
        require(not path.is_absolute() and '..' not in path.parts and str(path) == name,
                'Unsafe capsule-relative name')
        require('__pycache__' not in path.parts, 'Bytecode is not text evidence')
        return name
    def decode(raw):
        def invalid(value):
            raise ValueError('Nonfinite JSON ' + value)
        def pairs(items):
            result = {}
            for key, value in items:
                require(key not in result, 'Duplicate JSON key')
                result[key] = value
            return result
        return json.loads(raw, parse_constant=invalid, object_pairs_hook=pairs)
    local = read_bytes is None
    if local:
        root = Path(root or Path(__file__).parent).absolute()
        require(all(not p.is_symlink() for p in (root, *root.parents)), 'Symlink capsule root')
        def read_local(name):
            safe(name)
            path = root
            for part in PurePosixPath(name).parts:
                path /= part
                require(not path.is_symlink(), 'Symlink capsule member')
            require(path.is_file(), 'Missing capsule member')
            return path.read_bytes()
        read_bytes = read_local
    raw_index = read_bytes('evidence-index.json')
    if expected_index_sha256 is not None:
        require(sha(raw_index) == expected_index_sha256, 'Different capsule index')
    index = decode(raw_index)
    require(index['schema'] == 'n1-dp-profile-portable-evidence-v1', 'Wrong capsule schema')
    require(all(index[name] is False for name in ('native_bytes_retained', 'performance_acceptance',
                                                 'speedup_claimed', 'current_NinjaRS_comparison')), 'Scope changed')
    files, payloads = index['files'], index['payloads']
    for name, row in files.items():
        safe(name)
        require(name != 'evidence-index.json' and set(row) == {'sha256', 'bytes'}, 'Self/invalid physical index')
        require(type(row['bytes']) is int and 0 <= row['bytes'] <= 32 * 1024**2 and
                re.fullmatch('[0-9a-f]{64}', row['sha256']), 'Invalid physical identity')
    physical_names = []
    for name, row in payloads.items():
        safe(name)
        safe(row['path'])
        require(name != 'evidence-index.json' and row['path'] in files and
                set(row) == {'path', 'sha256', 'bytes', 'encoding'}, 'Unindexed logical payload')
        require(row['encoding'] in ('utf8', 'gzip+utf8') and type(row['bytes']) is int and
                0 <= row['bytes'] <= 64 * 1024**2 and re.fullmatch('[0-9a-f]{64}', row['sha256']),
                'Invalid decoded identity')
        require(PurePosixPath(name).suffix.lower() not in ('.o', '.mod', '.a', '.so', '.dll', '.dylib', '.exe'),
                'Native file suffix')
        physical_names.append(row['path'])
    require(len(set(physical_names)) == len(physical_names) and set(physical_names) == set(files),
            'Aliased, extra or missing physical payload')
    require(sum(row['bytes'] for row in payloads.values()) <= 128 * 1024**2, 'Decoded capsule too large')
    if local:
        paths = list(root.rglob('*'))
        require(all(not p.is_symlink() for p in paths), 'Symlink inventory')
        require({str(p.relative_to(root)) for p in paths if p.is_file()} == set(files) | {'evidence-index.json'},
                'Physical capsule inventory differs')
    else:
        require(hasattr(read_bytes, 'paths') and set(read_bytes.paths) == set(files) | {'evidence-index.json'},
                'Callback physical inventory differs')
    decoded = {}
    for name, row in payloads.items():
        raw = read_bytes(row['path'])
        require(len(raw) == files[row['path']]['bytes'] and sha(raw) == files[row['path']]['sha256'],
                'Changed stored payload: ' + name)
        if row['encoding'] == 'gzip+utf8':
            require(raw[:4] == b'\x1f\x8b\x08\x00' and raw[4:8] == b'\0' * 4, 'Nondeterministic gzip header')
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(row['bytes'] + 1)
        require(len(raw) == row['bytes'] and sha(raw) == row['sha256'], 'Changed decoded payload: ' + name)
        require(b'\0' not in raw and not raw.startswith((b'\x7fELF', b'!<arch>')), 'Native/NUL bytes retained')
        raw.decode('utf8')
        decoded[name] = raw
    def raw(name):
        safe(name)
        require(name in decoded, 'Unindexed logical read: ' + name)
        return decoded[name]
    def text(name):
        return raw(name).decode('utf8')
    def load(name):
        return decode(raw(name))
    def norm(value):
        return json.loads(json.dumps(value))
    def prefix_inventory(prefix):
        return {name[len(prefix):] for name in decoded if name.startswith(prefix)}
    retention = load('retention.json')
    require(retention['schema'] == 'n1-dp-profile-private-retention-v1' and
            retention['binary_bytes_retained'] is False and retention['native_executions_during_capture'] == 0 and
            all(retention[name] is False for name in ('performance_acceptance', 'speedup_claimed', 'current_NinjaRS_comparison')),
            'Retention scope changed')
    for name, row in retention['origins'].items():
        require(sha(raw(name)) == row['sha256'] and len(raw(name)) == row['bytes'], 'Retained origin differs')
    for number, expected in ((1, SOURCE1), (2, SOURCE2)):
        prefix = 'source/source' + str(number) + '/'
        require(sha(raw(prefix + 'source-index.json')) == expected, 'Source seal changed')
        seal = load(prefix + 'source-index.json')
        require(prefix_inventory(prefix) == set(seal['files']) | {'source-index.json'}, 'Source exact inventory differs')
        for name, row in seal['files'].items():
            require(sha(raw(prefix + safe(name))) == row['sha256'] and len(raw(prefix + name)) == row['bytes'],
                    'Sealed source changed')
        for name, value in seal['support'].items():
            require(sha(raw('support/' + safe(name))) == value, 'Pinned support changed')
    predecessor = load('source/source2/predecessor.json')
    require(predecessor['failure_terminal_sha256'] == FAILED and predecessor['predecessor_source_index_sha256'] == SOURCE1,
            'Failure/source ancestry differs')
    for prefix, key in (('source/source1/', 'predecessor_source_files'), ('results/failed-build1/', 'failed_execution_files')):
        require(prefix_inventory(prefix) == set(predecessor[key]), 'Predecessor inventory differs')
        for name, value in predecessor[key].items():
            require(sha(raw(prefix + safe(name))) == value, 'Predecessor bytes differ')
    require(sha(raw('results/failed-build1/build/terminal.json')) == FAILED and
            load('results/failed-build1/build/terminal.json')['status'] == 'failed' and
            "FileNotFoundError: [Errno 2] No such file or directory: 'make'" in text('results/failed-build1/build/console.log'),
            'Failed BUILD1 classification differs')
    for entry in predecessor['scientific_payload_unchanged']:
        names = [name for name in prefix_inventory('source/source2/') if name.startswith(entry)] if entry.endswith('/') else [entry]
        for name in names:
            require(raw('source/source1/' + name) == raw('source/source2/' + name), 'Scientific successor drift')
    tools = load('source/source2/toolchain-pins.json')
    require(len(tools['required_tools']) == 11 and all(row['executable'] is True for row in tools['required_tools'].values()) and
            tools['original_compiler_environment'] == {}, 'Frozen environment policy differs')
    for name, value in tools['donor_lineage'].items():
        require(sha(raw('environment-donor/' + safe(name))) == value, 'Environment donor lineage differs')
    donor = load('environment-donor/.tirust-build/fortran-qp-output-bulk-physical-proposal-1/input-pins.json')
    require(tools['closed_environment'] == donor['environment'], 'Closed environment differs')
    for name, value in AUDITS.items():
        audit = 'audits/' + name + '/report.json'
        require(sha(raw(audit)) == value and load(audit)['status'] == 'PASS', 'Independent stage audit differs')
    stages = {}
    for name, (terminal, result) in STAGES.items():
        prefix = 'results/' + name + '/'
        require(sha(raw(prefix + 'terminal.json')) == terminal and sha(raw(prefix + 'result.json')) == result,
                'Closed stage pin differs')
        t, r, a = load(prefix + 'terminal.json'), load(prefix + 'result.json'), load(prefix + 'admission.json')
        require(t['status'] == r['status'] == 'passed' and t['returncode'] == 0 and not t['cleanup']['survivors'],
                'Stage failed or cleanup incomplete')
        require(t['result_sha256'] == result and t['source_index_sha256'] == r['source_index_sha256'] == SOURCE2 and
                a['source_index_sha256'] == SOURCE2 and a['stage'] == name and t['stage'] == name,
                'Stage source/result/admission join differs')
        require(t['cap_seconds'] == {'build': 600., 'parity': 300., 'profile': 600.}[name] and
                t['cleanup_reserve_seconds'] == 60 and t['cpu'] == 12 and t['cpu_reservation'] is False,
                'Stage bounds differ')
        require(load(prefix + 'actual-limits.json')['actual'] == {k: [v, v] for k, v in t['resource_limits'].items()},
                'Actual limits differ')
        if name != 'build':
            require(a['reviewed_build_sha256'] == STAGES['build'][0] and r['build_terminal_sha256'] == STAGES['build'][0],
                    'BUILD review join differs')
        if name == 'profile':
            require(a['reviewed_parity_sha256'] == STAGES['parity'][0] and r['parity_terminal_sha256'] == STAGES['parity'][0],
                    'PARITY review join differs')
        stages[name] = r
    pins = load('source/source2/input-pins.json')
    for name, checksum in pins['generated_sources'].items():
        require(sha(raw('donor/' + safe(name))) == checksum, 'Donor generated source differs')
    require(load('donor/compilation.json')['effective_compiler_flags'] == pins['compiler_flags'], 'Donor compiler receipt differs')
    for name, checksum in pins['numerical_cards'].items():
        require(sha(raw('donor/' + name)) == checksum, 'Donor numerical card differs')
    closed = load('inputs/closed-observation.json')
    for logical, recorded in (('inputs/closed-observation.json', pins['closed_observation']),
                              ('inputs/closed-input.json', pins['closed_input'])):
        relative = next((name for name in pins['files'] if recorded.endswith(name)), None)
        require(relative is not None and sha(raw(logical)) == pins['files'][relative]['sha256'], 'Closed textual input pin differs')
    require(raw('inputs/closed-warmup.dat').decode() == closed['raw_outputs']['warmup.dat'], 'Frozen learning raw join differs')
    expected_parameter = load('inputs/closed-input.json')['param_card_sha256']
    require(sha(raw('donor/Cards/param_card.dat')) == expected_parameter, 'Physical card differs')
    require(sha(raw('inputs/points.dat')) == next(row['sha256'] for name, row in pins['files'].items()
                                               if name.endswith('/points/qcd_n1_uux_z.dat')), 'Point set differs')
    # This exact reviewed pure parser accepts text arguments only. Controllers,
    # native wrappers and filesystem helpers are never executed during replay.
    require(sha(raw('source/source2/analyze.py')) == PARSER, 'Reviewed parser differs')
    parser = types.ModuleType('capsule_pure_n1_parser')
    exec(compile(text('source/source2/analyze.py'), '<indexed:pure-n1-parser>', 'exec'), parser.__dict__)
    require(sha(raw('replay_warmup.py')) == WARMUP_PARSER, 'Reviewed pure learning parser differs')
    warmup_parser = types.ModuleType('capsule_pure_learning_parser')
    exec(compile(text('replay_warmup.py'), '<indexed:pure-learning-parser>', 'exec'), warmup_parser.__dict__)
    build = stages['build']
    aliases = load('link-runtime-resolution.json')
    recorded_commands = Counter()
    for variant, row in build['variants'].items():
        compiled = 'compiled/' + variant + '/'
        source_prefix = 'source/source2/' + ('baseline-source/' if variant == 'baseline' else 'observed-source/')
        for name in ('check_sa.f', 'TIR_interface.f', 'TIRUST_interface.f', 'mtirust.f90'):
            require(raw(compiled + name) == raw(source_prefix + name), 'Actually compiled source differs from proposal')
        require(raw(compiled + 'qp_profile.f90') == raw('source/source2/qp_profile.f90'), 'Compiled profiling module differs')
        for name, checksum in row['sources'].items():
            basename = name.removeprefix('SubProcesses/P0_uux_z/')
            actual = compiled + basename if name.startswith('SubProcesses/P0_uux_z/') and basename in (
                'makefile', 'check_sa.f', 'TIR_interface.f', 'TIRUST_interface.f', 'mtirust.f90', 'qp_profile.f90') else 'donor/' + name
            require(sha(raw(actual)) == checksum, 'Compiled/reused source identity differs')
        changed = {name for name, value in row['objects'].items() if pins['numerical_objects'].get(name) != value}
        allowed = {'SubProcesses/P0_uux_z/' + n for n in ('check_sa.o', 'TIR_interface.o', 'TIRUST_interface.o',
                                                        'mtirust.o', 'qp_profile.o', 'mtirust.mod', 'dp_profile_state.mod')}
        require(changed <= allowed and set(pins['numerical_objects']) - set(row['objects']) ==
                {'lib/libdhelas.a', 'lib/libcts.a', 'lib/libiregi.a', 'lib/mpmodule.mod', 'lib/libmodel.a'},
                'Nonprofile numerical object identity drift')
        require(row['compiler_flags']['environment'] == {}, 'Compiler injection differs')
        for scope, old in pins['compiler_flags']['scopes'].items():
            recipes = row['compiler_flags']['scopes'][scope]['recipes']
            extra = {'mtirust.f90', 'qp_profile.f90'} if scope == 'process' else set()
            require(set(recipes) == set(old['recipes']) | extra, 'Recipe inventory differs')
            for name, before in old['recipes'].items():
                require(before['compiler'] == recipes[name]['compiler'] and before['flags'] == recipes[name]['flags'],
                        'Compiler recipe drift')
            require(all(recipes[name]['flags'] == ['-O3', '-fPIC'] for name in extra), 'Private module flags differ')
        prefix = 'results/build/' + variant + '/'
        require(sha(raw(prefix + 'prelink.json')) == row['prelink_sha256'], 'Prelink receipt differs')
        pre = load(prefix + 'prelink.json')
        mappings = aliases[variant]['map_aliases']
        require(sha(raw(compiled + 'private-link.map')) == row['link_map_sha256'] and
                re.findall(r'^LOAD\s+(.+)$', text(compiled + 'private-link.map'), re.M) == [a['spelling'] for a in mappings],
                'Actual raw linker map differs')
        require({a['resolved']: a['sha256'] for a in mappings} == row['actual_link_loads'] and
                len(row['actual_link_loads']) == 54 and all(pre['bound'].get(a['resolved']) == a['sha256'] for a in mappings),
                'Link inputs not prebound')
        runtime = aliases[variant]['runtime_aliases']
        spellings = []
        for line in row['runtime']['listing'].splitlines():
            line = line.strip()
            if not line or line.startswith('linux-vdso.so.1 '):
                continue
            match = re.fullmatch(r'(?:\S+\s+=>\s+)?(/[^\s()]+)\s+\(0x[0-9a-fA-F]+\)', line)
            require(match is not None, 'Malformed retained runtime listing')
            spellings.append(match[1])
        require(spellings == [a['spelling'] for a in runtime], 'Retained runtime aliases differ')
        require({a['resolved']: a['sha256'] for a in runtime} == row['runtime']['files'] and len(row['runtime']['files']) == 15 and
                row['runtime']['files'][row['runtime']['loader']] == row['runtime']['loader_sha256'], 'Runtime closure differs')
        require(len(row['symbols']['definitions']) == 23 and set(row['symbols']['definitions'].values()) == {1} and
                row['symbols']['original_archive_member_loaded'] is False, 'Private MTIRUST ownership differs')
        def symbols(name):
            definitions = Counter()
            for line in text(prefix + 'commands/' + name).splitlines():
                fields = line.split()
                if len(fields) >= 3 and '__mtirust_MOD_' in fields[-1]:
                    definitions[fields[-1]] += 1
            return dict(definitions)
        require(symbols('06-module-symbols.stdout.txt') == symbols('07-executable-symbols.stdout.txt') ==
                row['symbols']['definitions'], 'Raw MTIRUST symbol proof differs')
        command_names = sorted(n for n in decoded if n.startswith(prefix + 'commands/') and n.endswith('.command.json'))
        counts = Counter()
        for name in command_names:
            c = load(name)
            stem = name.removesuffix('.command.json')
            t = load(stem + '.terminal.json')
            require(t['returncode'] == 0 and sha(raw(stem + '.stdout.txt')) == t['stdout_sha256'], 'Recorded command/log differs')
            label = PurePosixPath(stem).name
            if '-c' in c['argv']:
                kind = 'object_compiles'
            elif label == '05-link':
                kind = 'links'
                require(c['argv'].index('mtirust.o') < c['argv'].index('-ltirust'), 'Private module precedence differs')
            elif label.startswith('prelink-query-'):
                kind = 'prelink_compiler_queries'
            elif label.startswith(('06-', '07-')):
                kind = 'nm_queries'
            elif label == 'expanded-check':
                kind = 'expanded_make'
            else:
                kind = 'runtime_loader_queries'
                require('--list' in c['argv'], 'Unexpected recorded command')
            counts[kind] += 1
        require(counts == {'object_compiles': 5, 'links': 1, 'nm_queries': 2, 'expanded_make': 1,
                           'prelink_compiler_queries': 51, 'runtime_loader_queries': 1}, 'Recorded BUILD action count differs')
        recorded_commands.update(counts)
    query_scope = retention['query_count_scope']
    require(sum(recorded_commands.values()) == query_scope['recorded_BUILD_actions'] == 122 and
            query_scope['additional_compiler_policy_captures'] == 4 and query_scope['implied_additional_make_dryruns'] == 12 and
            query_scope['implied_cached_FC_version_queries'] == 4 and
            query_scope['all_make_internal_children_individually_recorded'] is False, 'Query count scope differs')
    def worker(phase, lane):
        prefix = 'results/' + phase + '/' + lane + '/'
        variant = 'baseline' if lane == 'baseline' else 'observed'
        ready, warm = load(prefix + 'ready.json'), load(prefix + 'warmup.json')
        require([ready[k] for k in ('points', 'ndim', 'nso', 'nborn')] == [100, 1, 1, 1], 'Worker dimensions differ')
        learning = raw('learning/' + phase + '/' + lane + '.dat')
        require(learning == raw('inputs/closed-warmup.dat') and warm['raw_outputs']['warmup.dat'].encode() == learning and
                warm['rows'] == closed['rows'], 'Full learning trace differs')
        require(warmup_parser.replay_warmup(warm, warm, ready) == warm and warm['native_dp_only'] is True and
                warm['warmup_first_call'] == 1 and warm['warmup_last_call'] == 100 and warm['status_counts'] == {'219': 100},
                'Full raw learning parser replay differs')
        before, after = load(prefix + 'private-before.json'), load(prefix + 'private-after.json')
        require(before == after and before['physical_sha256'] == expected_parameter and
                before['executable_sha256'] == build['variants'][variant]['executable_sha256'], 'Private worker pin join differs')
        alias = 'SubProcesses/P0_uux_z/MadLoop5_resources/MadLoopParams.dat'
        require(before['cards'] == {**pins['numerical_cards'], alias: next(iter(pins['numerical_cards'].values()))},
                'Four-card worker inventory differs')
        for which in ('before', 'after'):
            maps = load(prefix + 'maps-' + which + '.json')
            providers = build['variants'][variant]['runtime']['files']
            require(all(maps['files'].get(n) == v for n, v in providers.items()), 'Mapped providers differ')
            allowed = {**providers, **load('source/source2/source-index.json')['allowed_extra_mappings']}
            extra = {n: v for n, v in maps['files'].items() if allowed.get(n) != v}
            require(len(extra) == 1 and next(iter(extra.values())) == before['executable_sha256'], 'Mapped executable/extra differs')
            paths = {line.split(None, 5)[5] for line in maps['maps'].splitlines()
                     if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')}
            require(paths == set(maps['files']), 'Raw map/identity paths differ')
        commands = [n for n in decoded if n.startswith(prefix + 'commands/') and n.endswith('.command.json')]
        require(len(commands) == 2, 'Worker runtime query count differs')
        for name in commands:
            stem = name.removesuffix('.command.json')
            require('--list' in load(name)['argv'] and load(stem + '.terminal.json')['returncode'] == 0 and
                    sha(raw(stem + '.stdout.txt')) == load(stem + '.terminal.json')['stdout_sha256'], 'Worker runtime query differs')
    baseline_words = stages['parity']['lanes']['baseline']['rows']
    baseline_ieee = stages['parity']['lanes']['baseline']['ieee']
    for phase in ('parity', 'profile'):
        for lane in LANES:
            worker(phase, lane)
            mode = 0 if lane == 'baseline' else int(lane[-1])
            prefix = 'results/' + phase + '/' + lane + '/' + ('exact' if phase == 'parity' else 'exact-before-timing') + '/'
            replay = {'rows': parser.exact(text(prefix + 'dp-exact.dat')), 'ieee': parser.ieee(text(prefix + 'dp-ieee.dat')),
                      'profile': parser.profile(text(prefix + 'dp-parity-timers.tsv'), mode, 100, lane != 'baseline'),
                      'probe': parser.probe(text(prefix + 'dp-probe.tsv'))}
            require(replay['rows'] == baseline_words and replay['ieee'] == baseline_ieee, 'Full100 word/IEEE gate differs')
            if phase == 'parity':
                require(norm(replay) == stages['parity']['lanes'][lane], 'PARITY raw parser/result join differs')
    profile = stages['profile']
    require(profile['loops_per_batch'] == 128 and profile['points_per_traversal'] == 100 and
            profile['performance_acceptance'] is False and profile['optimized'] is False and len(profile['cycles']) == 7,
            'PROFILE measurement scope differs')
    for number, cycle in enumerate(profile['cycles'], 1):
        order = list(LANES)
        shift = (number - 1) % 4
        order = order[shift:] + order[:shift]
        if number % 2 == 0:
            order.reverse()
        require(cycle['cycle'] == number and cycle['order'] == order and set(cycle['lanes']) == set(LANES) and
                load('results/profile/cycle-' + str(number) + '.json') == cycle, 'Fixed cycle order/receipt differs')
        common_ieee = None
        for lane in LANES:
            row = cycle['lanes'][lane]
            prefix = 'results/profile/' + lane + '/cycle-' + str(number) + '/'
            require([list(map(int, line.split())) for line in text(prefix + 'timed_status.dat').splitlines() if line.strip()] ==
                    [[219, 12800]] and row['timed_status_counts'] == {'219': 12800}, 'All-call status coverage differs')
            words = parser.exact(text(prefix + 'dp-timed-exact.dat'))
            require(words == baseline_words, 'Final100 exact word parity differs')
            ieee = parser.ieee(text(prefix + 'dp-timed-ieee.dat'), timed=True)
            require(ieee == row['ieee'], 'Timed IEEE raw/result join differs')
            if common_ieee is None:
                common_ieee = ieee
            require(ieee == common_ieee, 'Timed cross-lane IEEE differs')
            mode = 2 if lane == 'baseline' else int(lane[-1])
            timings = parser.profile(text(prefix + 'dp-timers.tsv'), mode, 12800, lane != 'baseline')
            require(norm(timings) == row['profile'], 'Timed counters/scopes raw parser join differs')
            if lane == 'observed-2':
                scopes = {(s['stage'], s['ctmode']): s for s in timings['scopes']}
                require(all(scopes[3, ct]['ticks'] >= scopes[4, ct]['ticks'] for ct in (1, 2)), 'Nested clock scopes differ')
            elapsed = list(map(float, text(prefix + 'timing.dat').split()))
            require(len(elapsed) == 2 and all(math.isfinite(v) and v > 0 for v in elapsed) and
                    elapsed[1] == row['seconds'], 'Raw elapsed/result join differs')
            decimal = [line.split() for line in text(prefix + 'values.dat').splitlines() if line.strip()]
            require(len(decimal) == 200, 'Decimal output coverage differs')
            for fields, word in zip(decimal, words):
                require(len(fields) == 8 and list(map(int, fields[:3])) == [word['point'], word['order'], word['status']] and
                        [struct.pack('>d', float(v)).hex().upper() for v in fields[3:]] == word['words'], 'Decimal/raw word join differs')
    recomputed = parser.summary(profile['cycles'])
    require(norm(recomputed) == profile['summary'], 'Recomputed median ratios differ')
    summary = {'schema': 'n1-dp-profile-portable-summary-v1', 'status': 'PASS',
               'source1_failed_make_retained': True, 'source2_scientific_payload_unchanged': True,
               'recorded_BUILD_actions': 122, 'additional_policy_captures': 4,
               'learning_traces': 8, 'cycles': 7, 'lanes': 4, 'batches': 28,
               'timed_physical_calls': 358400, 'all_timed_statuses': [219],
               'final100_word_parity_per_batch': True, 'IEEE_observations_equal': True,
               'instrumentation_overhead_median_ratios': recomputed['median_ratios'],
               'performance_acceptance': False, 'speedup_claimed': False, 'current_NinjaRS_comparison': False,
               'limitations': ['Only final100 words per batch; status/counters cover all12800 calls.',
                   'Stage3 contains stage4; do not sum nested intervals or subtract the empty probe.',
                   'Sticky IEEE equality is observational and does not identify new true-to-true flags.',
                   'Shared-host instrumentation overhead only; precise local timings remain a separate TODO.',
                   'Historical native hashes/maps are retained identities, not fresh receiving-host checks.',
                   '122 recorded BUILD actions exclude4 policy captures and do not enumerate all make-internal children.']}
    return {'status': 'PASS', 'summary': summary, 'indexed_text_payloads': len(payloads),
            'index_sha256': sha(raw_index), 'compiler_invocations': 0, 'loader_queries': 0, 'native_evaluations': 0}


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--expected-index')
    args = cli.parse_args()
    print(json.dumps(verify(expected_index_sha256=args.expected_index), indent=2, sort_keys=True))
