"""Portable20-request continuation and disjoint4+20 replay; no workspace fallback."""
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

ANCESTOR = 'physical-stress-current-portable-4-terminal-2'
ANCESTOR_INDEX = '6fc2035649ba0387f0d4457784ae883592f13bb7e3b6f9c33b99e038b81f7146'
ANCESTOR_VERIFIER = '20d1666c837ca09632121181b1a6a95b9d6afd677cb897598be7de80cb79a3e8'
PREPARATION = 'physical-stress-current-portable-4-preflight-1'
PREPARATION_INDEX = '49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22'
PREPARATION_VERIFIER = '3850e1dcfbd3cecfc2c11eabd090e5ae174b0d0fbad6da11b6d94557c5e72f24'
SOURCE5_INDEX = '5eded3d276bea030e9b8cce598f6422f10a789dc1eaa9ff6588275c2e594b931'
CHECKPOINT = '0c4157c988bdd0371e9c0cfba54e784b3f7a730cf7a631b8aca5c27814636d5a'


def require(value, message):
    if not value: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def safe(name):
    require(isinstance(name, str), 'Non-string evidence name')
    path = PurePosixPath(name)
    require(name and '\\' not in name and not path.is_absolute() and '..' not in path.parts
            and str(path) == name, 'Nonportable evidence name')
    return name


class Packet:
    def __init__(self, reader, prefix=''):
        self.reader, self.prefix = reader, prefix
        self.index_raw = reader(prefix + 'evidence-index.json'); self.index = json.loads(self.index_raw)
        require(self.index['schema_version'] == 1, 'Unsupported packet schema')
        self.files = self.index['files']; self.compressed = self.index.get('compressed_text', {})
        self.names = set(self.files) | set(self.compressed)
        paths = [item['path'] for item in self.compressed.values()]
        require(len(paths) == len(set(paths)), 'Compressed aliases overlap')
        for name in self.files:
            raw = self.physical(name)
            if name not in paths: self.text(raw)
        for name in self.compressed:
            require(name not in self.files, 'Compressed/raw evidence collision'); self.read(name)
    @staticmethod
    def text(raw):
        require(b'\0' not in raw, 'Binary evidence payload'); raw.decode('utf-8'); return raw
    def physical(self, name):
        safe(name); item = self.files[name]; raw = self.reader(self.prefix + name)
        require(item['kind'] == 'file' and len(raw) == item['size'] and sha(raw) == item['sha256'], 'Changed evidence: ' + name)
        return raw
    def read(self, name):
        safe(name)
        if name in self.compressed:
            item = self.compressed[name]; raw = gzip.decompress(self.physical(item['path']))
            require(len(raw) == item['raw_bytes'] and sha(raw) == item['raw_sha256'], 'Changed compressed evidence')
            return self.text(raw)
        return self.physical(name)
    def load(self, name):
        def reject(value): raise ValueError('Nonfinite JSON token: ' + value)
        return json.loads(self.read(name), parse_constant=reject)


def dependency_declaration():
    return {'ancestor': {'sibling': ANCESTOR, 'index_sha256': ANCESTOR_INDEX, 'verifier_sha256': ANCESTOR_VERIFIER},
            'preparation': {'sibling': PREPARATION, 'index_sha256': PREPARATION_INDEX, 'verifier_sha256': PREPARATION_VERIFIER}}


def reader_closure(reader, expected=None):
    require(hasattr(reader, 'paths'), 'Portable reader must declare its physical paths')
    paths = set(reader.paths)
    for name in paths: safe(name)
    if expected is not None: require(paths == expected, 'Reader closure differs')
    return paths


def verify(root=None, *, read_bytes=None, derive_summary=False):
    if read_bytes is None:
        root = Path(root or Path(__file__).resolve().parent).absolute()
        def local(name):
            safe(name); base = root
            for prefix, sibling in (('ancestor/', ANCESTOR), ('preparation/', PREPARATION)):
                if name.startswith(prefix): base = root.with_name(sibling); name = name[len(prefix):]; break
            require(all(not p.is_symlink() for p in (base, *base.parents)), 'Symlink capsule root')
            path = base
            for part in PurePosixPath(name).parts:
                path /= part; require(not path.is_symlink(), 'Symlink evidence member')
            return path.read_bytes()
        local.paths = set()
        for prefix, base in (('', root), ('ancestor/', root.with_name(ANCESTOR)), ('preparation/', root.with_name(PREPARATION))):
            for path in base.rglob('*'):
                require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Unexpected capsule filesystem object')
                if path.is_file(): local.paths.add(prefix + path.relative_to(base).as_posix())
        read_bytes = local
    reader_closure(read_bytes)
    own = Packet(read_bytes); earlier = Packet(read_bytes, 'ancestor/'); prepared = Packet(read_bytes, 'preparation/')
    require(not any(name.startswith(('ancestor/', 'preparation/')) for name in own.names), 'Reserved dependency prefix in own packet')
    require(own.index['dependencies'] == dependency_declaration(), 'Dependency declarations differ')
    require(sha(earlier.index_raw) == ANCESTOR_INDEX and sha(prepared.index_raw) == PREPARATION_INDEX, 'Changed dependency index')
    paths = set(own.files) | {'evidence-index.json'}
    paths |= {'ancestor/' + name for name in earlier.files} | {'ancestor/evidence-index.json'}
    paths |= {'preparation/' + name for name in prepared.files} | {'preparation/evidence-index.json'}
    reader_closure(read_bytes, paths)
    source = earlier.read('verify_evidence.py'); require(sha(source) == ANCESTOR_VERIFIER, 'Ancestor verifier differs')
    namespace = {'__name__': 'retained_ancestor_verifier'}
    exec(compile(source, '<retained ancestor verifier>', 'exec'), namespace)
    def ancestor_reader(name):
        return read_bytes(name if name.startswith('preparation/') else 'ancestor/' + name)
    ancestor_reader.paths = set(earlier.files) | {'evidence-index.json'} | {'preparation/' + n for n in prepared.files} | {'preparation/evidence-index.json'}
    previous = namespace['verify'](read_bytes=ancestor_reader)
    require(previous['verified'] and previous['index_sha256'] == ANCESTOR_INDEX
            and previous['preparation_index_sha256'] == PREPARATION_INDEX, 'Ancestor proof failed')
    require(sha(own.read('source/source-index.json')) == SOURCE5_INDEX, 'Source5 index differs')
    for name, item in own.load('source/source-index.json')['files'].items():
        raw = own.read('source/' + name)
        require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Source5 payload differs')
    source_policy = own.read('policy/source-index.json')
    require(sha(source_policy) == own.index['source_policy_index_sha256'], 'Retention source-policy binding differs')
    for name, item in json.loads(source_policy)['files'].items():
        raw = own.read('policy/' + name)
        require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Retention source-policy payload differs')
    for name in ('capture.py', 'verify_evidence.py'):
        require(own.read(name) == own.read('policy/' + name), 'Executed retention source differs')
    policy = {'__name__': 'retained_continuation_derivation'}
    exec(compile(own.read('policy/derive.py'), '<retained continuation derivation>', 'exec'), policy)
    protocol = policy['analyzer'](prepared.read('proposal/protocol/tirust_benchmark_worker.py'),
                                  prepared.read('proposal/protocol/tirust_benchmark.py'))
    retention = own.load('retention.json')
    require(retention['source_policy_index_sha256'] == sha(source_policy)
            and retention['source_index_sha256'] == SOURCE5_INDEX
            and retention['inherited_checkpoint_sha256'] == CHECKPOINT
            and retention['capture_sha256'] == sha(own.read('capture.py'))
            and retention['capture_native_execution'] is False, 'Retention identity differs')
    def closed(name):
        item = retention['closed_files'][safe(name)]; raw = own.read('closed/' + name)
        require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Closed report byte identity differs')
        return raw
    def load(name): return json.loads(closed(name))
    def available(name): return name in retention['closed_files']
    for name in retention['closed_files']: closed(name)
    require({name[len('closed/'):] for name in own.names if name.startswith('closed/')} == set(retention['closed_files']),
            'Uninventoried closed report payload')
    terminal = own.load('root-terminal-observation.json'); stage = load('stage-run.json')
    require(terminal['kind'] == 'root-reported-tool-terminal-observation' and terminal['session_id'] == 7248
            and terminal['exit_code'] in (0, 1) and terminal['source_index_sha256'] == SOURCE5_INDEX
            and terminal['inherited_checkpoint_sha256'] == CHECKPOINT, 'Actual root terminal identity differs')
    require(stage['stage'] == 'run' and stage['source_index_sha256'] == SOURCE5_INDEX
            and stage['inherited_checkpoint_sha256'] == CHECKPOINT and stage['native_execution'] is True
            and stage['performance_eligible'] is False and stage['cap_seconds_excluding_lock_wait'] == 7600
            and stage['status'] == ('passed' if terminal['exit_code'] == 0 else 'failed'), 'Actual continuation stage differs')
    checkpoint_raw = closed('inherited-checkpoint.json'); checkpoint = json.loads(checkpoint_raw)
    require(sha(checkpoint_raw) == CHECKPOINT and checkpoint['new_setup_launches'] == checkpoint['new_mass_probe_launches'] == 0
            and checkpoint['inherited_completed_requests'] == 4 and checkpoint['continuation_requests'] == 20
            and checkpoint['amplitudes_executed_in_this_stage'] == 0, 'Inherited setup identity differs')
    inherited = load('stage-inherit.json')
    require(inherited['status'] == 'passed' and inherited['source_index_sha256'] == SOURCE5_INDEX
            and inherited['inherited_checkpoint_sha256'] == CHECKPOINT and inherited['native_execution'] is False,
            'Actual inheritance stage differs')
    full_plan = prepared.load('proposal/proposal.json'); plan = {**full_plan, 'requests': full_plan['requests'][4:]}
    require(checkpoint['checkpoint']['requests'] == plan['requests']
            and checkpoint['checkpoint']['ancestor_index_sha256'] == ANCESTOR_INDEX, 'Inherited suffix differs')
    accounting = load('request-accounting.json'); aborted = load('aborted-request.json') if available('aborted-request.json') else None
    require(accounting['native_stage_failure'] == stage['exception'], 'Stage/accounting failure differs')
    post = own.load('postclosure-bindings.json'); source_guard = retention['source_guard']
    source_guard_ok = source_guard.get('ok') is True and source_guard.get('result') == checkpoint['checkpoint']
    functions = [node for node in ast.parse(prepared.read('source/run.py')).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'render_card']
    cards = {'require': require}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen card writer>', 'exec'), cards)
    completions = {}
    for name in sorted(retention['closed_files']):
        if name.startswith('request-completions/'):
            value = load(name); rid = policy['key'](value['request'])
            require(rid not in completions, 'Duplicate request completion'); completions[rid] = value
    records = {}; requested_ids = [policy['key'](request) for request in plan['requests']]
    require(set(completions) <= set(requested_ids), 'Unexpected completion identity')
    require(set(post['workers']) == set(post['worker_directories']) and set(post['workers']) <= set(requested_ids), 'Unexpected worker inventory')
    for request in plan['requests']:
        rid = policy['key'](request); prefix = 'runs/' + rid + '/'
        files = [name for name in retention['closed_files'] if name.startswith(prefix)]
        worker = post['workers'].get(rid)
        if not files and worker is None: continue
        errors = list(post.get('binding_errors', []))
        if worker is not None: errors.extend(worker.get('binding_errors', []))
        if not source_guard_ok: errors.append('postclosure inherited input guard failed')
        if worker is None: errors.append('private worker inventory missing')
        raw = {name: closed(prefix + name).decode() for name in ('warmup.dat','dimensions.dat','values.dat','timed_status.dat','timing.dat') if available(prefix + name)}
        stdout = closed(prefix + 'stdout.txt').decode() if available(prefix + 'stdout.txt') else None
        result = load(prefix + 'result.json') if available(prefix + 'result.json') else None
        completion = load(prefix + 'completion.json') if available(prefix + 'completion.json') else None
        rc = completions.get(rid); spec = load(prefix + 'specification.json') if available(prefix + 'specification.json') else None
        maps_ok = False; command_started = False; startup = []; commands = []
        if spec is not None:
            cid, source = request['case'], request['source']; label = source + '-' + cid
            build = prepared.load('closed/builds/' + label + '/build.json'); preflight = prepared.load('closed/builds/' + label + '/preflight.json')
            case = next(entry for entry in full_plan['cases'] if entry['case']['id'] == cid)
            require(spec['request'] == request and spec['backend'] == request['backend'] and spec['build'] == build
                    and spec['scientific_sources'] == build['scientific_sources'] and spec['runtime']['files'] == preflight['runtime']['files'],
                    'Worker/source/preflight identity differs')
            require(spec['points_sha256'] == case['point_sha256'] == sha(prepared.read('proposal/points/' + cid + '/selected.dat'))
                    and spec['physical_card_sha256'] == case['physical_card_sha256'] == sha(closed(prefix + 'param_card.dat')),
                    'Worker physical inputs differ')
            expected_settings = dict(prepared.load('proposal/policy-reference.json')[source]); expected_settings['mlreductionlib'] = str(request['backend'])
            original_card = prepared.read('generated/' + label + '/Cards/MadLoopParams.dat').decode()
            rendered, ignored = cards['render_card'](original_card, expected_settings, source)
            require(closed(prefix + 'MadLoopParams.dat') == rendered.encode() and
                    spec['settings'] == {'requested': expected_settings, 'not_applicable_to_baseline': ignored, 'serialized_sha256': sha(rendered.encode())}
                    and set(spec['numerical_cards'].values()) == {sha(rendered.encode())}, 'Frozen numerical policy differs')
            # The scientific proposal carries literal absolute project paths;
            # prove the actual relocated worker rather than trusting argv.cwd.
            root_spelling = request['points'].split('/docs/benchmarkfolder/')[0]
            expected_cwd = root_spelling + '/.tirust-build/physical-stress-current-5/workers/' + rid + '/' + case['process_subdirectory']
            require(spec['cwd'] == expected_cwd and spec['argv_without_fifo'] == ['taskset','-c','12','./check',request['points']]
                    and spec['actual_stdin_after_READY'] == 'RUN 1 1\nSTOP 0 0\n'
                    and spec['inherited_checkpoint_sha256'] == CHECKPOINT and spec['ancestor_index_sha256'] == ANCESTOR_INDEX,
                    'Actual worker protocol/ancestry differs')
            expected = {**spec['runtime']['files'], spec['cwd'] + '/check': build['executable_sha256']}
            require(spec['expected_runtime'] == expected, 'Expected runtime set differs')
            if worker is not None:
                if worker['scientific_sources'] != spec['scientific_sources'] or worker['numerical_cards'] != spec['numerical_cards'] \
                        or worker['physical_card_sha256'] != spec['physical_card_sha256'] or worker['executable_sha256'] != build['executable_sha256']:
                    errors.append('postclosure private worker input binding differs')
            for name in ('inherited-before.json', 'inherited-after.json'):
                if not available(prefix + name) or load(prefix + name) != checkpoint['checkpoint']:
                    errors.append('missing or changed ' + name)
            execution = 'execution/request-' + cid + '-' + request['label'] + '.json'
            if available(execution):
                command = load(execution)
                require(command['cwd'] == spec['cwd'] and command['cpu'] == 12 and command['cpu_exclusive'] is False
                        and 0 < command['command_work_cap_seconds'] <= 370 and command['argv'][:3] == ['taskset','-c','12']
                        and len(command['argv']) == 7 and command['argv'][4] == '-B'
                        and command['argv'][5].endswith('/physical-stress-current-supervisor-4/native_job.py')
                        and command['argv'][-1] == retention['original_closed_root'] + '/' + prefix + 'specification.json', 'Native command identity differs')
                command_started = True
                require(sha(closed(execution[:-5] + '.txt')) == command['log_sha256'], 'Native command log differs')
                if command['returncode'] != 0 or command['exception'] is not None or command['cleanup_exception'] is not None \
                        or command['cleanup'] is None or command['cleanup']['survivors']:
                    errors.append('native controller/cleanup did not close cleanly')
                for name, value in command['environment'].items():
                    if name in ('LD_LIBRARY_PATH','LD_PRELOAD','LD_AUDIT'): errors.append('unexpected native loader environment')
                    if name in ('MPLCONFIGDIR','XDG_CACHE_HOME','XDG_CONFIG_HOME','XDG_STATE_HOME') and '/physical-stress-current-5/' not in value:
                        errors.append('writable environment outside continuation namespace')
            else: errors.append('native owned command receipt missing')
            if result is not None:
                require(result['request'] == request, 'Native result identity differs')
                require({name: sha(value.encode()) for name, value in raw.items()} == result['raw_outputs_sha256'], 'Raw output hash differs')
                for observed in result['actual_map_captures']:
                    mapraw = closed(prefix + observed['path']); require(sha(mapraw) == observed['sha256'], 'Raw runtime map differs')
                    elf, data = {}, {}; good = True
                    for line in mapraw.decode().splitlines():
                        fields = line.split(maxsplit=5)
                        if len(fields) != 6 or not fields[5].startswith('/'): continue
                        spelling = fields[5]; item = post['mapped_files'].get(spelling, {})
                        if item.get('error') or spelling.endswith(' (deleted)'):
                            errors.append('mapped input cannot be independently rebound'); good = False; continue
                        if int(fields[4]) != item.get('inode') or tuple(int(v,16) for v in fields[3].split(':')) != tuple(item.get('device', ())):
                            errors.append('mapped inode/device changed'); good = False; continue
                        resolved = item['resolved']
                        if item['header64_hex'].startswith('7f454c46'):
                            if expected.get(resolved) != item['sha256']: errors.append('unexpected mapped ELF'); good = False
                            elf[resolved] = item['sha256']
                        else:
                            if not (spelling.startswith('/nix/store/') and spelling.endswith('/lib/locale/locale-archive') and fields[1] == 'r--p'):
                                errors.append('unexpected mapped data'); good = False
                            data[resolved] = {'sha256':item['sha256'],'size':item['size'],'inode':item['inode'],'map_device':fields[3],
                                              'header64_hex':item['header64_hex'],'permissions':fields[1]}
                    if elf != observed['elf_files'] or data != observed['data_files'] or (elf == expected) != observed['exact_expected']:
                        errors.append('recorded runtime classification differs'); good = False
                    maps_ok = maps_ok or good and elf == expected
        else:
            errors.append('no bound worker specification')
        if worker is not None:
            journals = worker['journals']
            for name, item in journals.items():
                require(sha(own.read(item['retained'])) == item['sha256'] and len(own.read(item['retained'])) == item['size'], 'Worker journal differs')
            startup = [own.load(item['retained']) for name, item in journals.items() if name.endswith('/startup/startup.json')]
            commands = [own.load(item['retained']) for name, item in journals.items() if name.endswith('/000001/command.json')]
            if result is not None and result.get('numerically_qualified'):
                if len(startup) != 1 or len(commands) != 1 or startup[0]['error'] is not None or commands[0]['error'] is not None:
                    errors.append('successful worker lacks clean protocol journals')
                for name, item in journals.items():
                    if '/000001/' in name and PurePosixPath(name).name in raw:
                        require(own.read(item['retained']).decode() == raw[PurePosixPath(name).name], 'Worker final journal bytes differ')
        if rc is not None:
            require(rc['request'] == request and 0 <= rc['elapsed_seconds'] <= 380 and rc['lock_wait_seconds'] >= 0
                    and (rc['lock_wait_seconds'] <= 3601 if request['requires_common_numerical_lock'] else True), 'Request resource accounting differs')
        records[rid] = {'raw': raw, 'stdout': stdout, 'result': result, 'completion': completion,
                        'request_completion': rc, 'runtime_verified': maps_ok, 'binding_errors': sorted(set(errors)),
                        'native_entry_observations': {
                            'owned_native_command_receipt': command_started,
                            'native_process_entry_verified': bool(command_started and len(startup) == 1 and maps_ok),
                            'startup_journal_count': len(startup),
                            'startup_errors': [item['error'] for item in startup],
                            'READY_verified_by_frozen_worker': len(startup) == 1 and startup[0]['error'] is None,
                            'RUN_command_journal_count': len(commands),
                            'command_errors': [item['error'] for item in commands],
                            'hardware_floating_point_flags_observed': False,
                            'internal_in_progress_amplitude_calls_instrumented': False}}
    summary = policy['derive'](full_plan, records, stage, accounting, aborted, protocol)
    for row in summary['request_rows']:
        row['native_entry_observations'] = records.get(row['request_id'], {}).get('native_entry_observations', {
            'native_process_entry_verified': False, 'hardware_floating_point_flags_observed': False,
            'internal_in_progress_amplitude_calls_instrumented': False})
        failure = row['native_failure'] or {}
        if failure.get('class') == 'ProtocolError' and 'protocol timeout' in failure.get('message', '').lower():
            row['qualification_rejection_kind'] = ('command_protocol_timeout' if row['native_entry_observations'].get('READY_verified_by_frozen_worker')
                                                    else 'startup_protocol_timeout_before_READY')
        elif failure.get('class') == 'MeasurementError':
            row['qualification_rejection_kind'] = 'native_warmup_or_output_measurement_rejected'
        elif row['binding_errors']:
            row['qualification_rejection_kind'] = 'runtime_or_input_binding_unqualified'
        elif row['host_checked_complete_stream']:
            row['qualification_rejection_kind'] = None
        else:
            row['qualification_rejection_kind'] = 'unqualified_see_retained_failure'
    aggregate = policy['aggregate'](previous['summary'], summary, protocol)
    summary.update({'role': 'current-physical-stress-continuation', 'ancestor_index_sha256': ANCESTOR_INDEX,
                    'preparation_index_sha256': PREPARATION_INDEX, 'source_index_sha256': SOURCE5_INDEX,
                    'inherited_checkpoint_sha256': CHECKPOINT, 'root_session_id': 7248, 'root_exit_code': terminal['exit_code'],
                    'new_setup_launches': 0, 'state_counts': dict(Counter(row['state'] for row in summary['request_rows'])),
                    'aggregate': aggregate, 'floating_point_hardware_flags_observed': False,
                    'native_process_request_entries_verified': sum(row['native_entry_observations']['native_process_entry_verified'] for row in summary['request_rows']),
                    'postclosure_global_binding_errors': post.get('binding_errors', []),
                    'performance_eligible': False})
    if not derive_summary: require(summary == own.load('qualification-summary.json'), 'Saved continuation summary differs')
    return {'verified': True, 'index_sha256': sha(own.index_raw), 'ancestor_index_sha256': ANCESTOR_INDEX,
            'preparation_index_sha256': PREPARATION_INDEX, 'summary': summary}
