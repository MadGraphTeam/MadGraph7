"""Two-capsule portable terminal proof. No native programs or workspace fallback."""
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

PARENT = 'physical-stress-current-portable-4-preflight-1'
PARENT_INDEX = '49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22'
PARENT_VERIFIER = '3850e1dcfbd3cecfc2c11eabd090e5ae174b0d0fbad6da11b6d94557c5e72f24'
POLICY_INDEX = '0cdf0b382e355d0956c2d021c9eb9e231b7b63996bdeacbcf5a5f33bc05dcee9'


def require(value, message):
    if not value: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def safe(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and '\\' not in name and not p.is_absolute()
            and '..' not in p.parts and str(p) == name, 'Nonportable evidence name')
    return name


class Packet:
    def __init__(self, reader, prefix=''):
        self.reader, self.prefix = reader, prefix
        self.index_raw = reader(prefix + 'evidence-index.json')
        self.index = json.loads(self.index_raw)
        require(self.index['schema_version'] == 1, 'Unsupported index')
        self.files = self.index['files']; self.compressed = self.index.get('compressed_text', {})
        self.names = set(self.files) | set(self.compressed)
        compressed_paths = [v['path'] for v in self.compressed.values()]
        require(len(set(compressed_paths)) == len(compressed_paths), 'Aliased compressed text')
        for name in self.files:
            raw = self.physical(name)
            if name not in compressed_paths: self.text(raw)
        for name in self.compressed:
            require(name not in self.files, 'Compressed/raw collision')
            self.read(name)
    def text(self, raw):
        require(b'\0' not in raw, 'Binary payload'); raw.decode('utf-8'); return raw
    def physical(self, name):
        safe(name); item = self.files[name]; raw = self.reader(self.prefix + name)
        require(item['kind'] == 'file' and len(raw) == item['size'] and sha(raw) == item['sha256'], 'Changed evidence: ' + name)
        return raw
    def read(self, name):
        safe(name)
        if name in self.compressed:
            item = self.compressed[name]; raw = gzip.decompress(self.physical(item['path']))
            require(len(raw) == item['raw_bytes'] and sha(raw) == item['raw_sha256'], 'Changed gzip text')
            return self.text(raw)
        return self.physical(name)
    def load(self, name):
        def reject(value): raise ValueError('Nonfinite JSON: ' + value)
        return json.loads(self.read(name), parse_constant=reject)


def verify(root=None, *, read_bytes=None, derive_summary=False):
    if read_bytes is None:
        root = Path(root or Path(__file__).resolve().parent).absolute()
        def local(name):
            safe(name); base = root
            if name.startswith('preparation/'):
                base = root.with_name(PARENT); name = name[len('preparation/'):]
            require(all(not p.is_symlink() for p in (base, *base.parents)), 'Symlink capsule root')
            path = base
            for part in PurePosixPath(name).parts:
                path /= part; require(not path.is_symlink(), 'Symlink evidence path')
            return path.read_bytes()
        read_bytes = local
    own = Packet(read_bytes)
    require(not any(n.startswith('preparation/') for n in own.names), 'Reserved preparation namespace in own index')
    require(own.index['dependencies'] == {'preparation': {'sibling': PARENT, 'index_sha256': PARENT_INDEX,
                                                        'verifier_sha256': PARENT_VERIFIER}}, 'Dependency declaration differs')
    parent = Packet(read_bytes, 'preparation/')
    require(sha(parent.index_raw) == PARENT_INDEX, 'Different preparation packet')
    expected_paths = set(own.files) | {'evidence-index.json'} | {'preparation/' + n for n in parent.files} | {'preparation/evidence-index.json'}
    if hasattr(read_bytes, 'paths'):
        require(set(read_bytes.paths) == expected_paths, 'Reader dependency closure differs')
    verifier_source = parent.read('verify_evidence.py')
    require(sha(verifier_source) == PARENT_VERIFIER, 'Preparation verifier differs')
    ns = {'__name__': 'retained_preparation_verifier'}
    exec(compile(verifier_source, '<retained preparation verifier>', 'exec'), ns)
    prepared = ns['verify'](read_bytes=lambda n: read_bytes('preparation/' + n))
    require(prepared['verified'], 'Preparation verification failed')
    policy_raw = own.read('policy/source-index.json')
    require(sha(policy_raw) == POLICY_INDEX, 'Terminal policy source differs')
    for name, item in json.loads(policy_raw)['files'].items():
        raw = own.read('policy/' + name)
        require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Terminal policy payload differs')
    policy = {'__name__': 'retained_terminal_policy'}
    exec(compile(own.read('policy/derive.py'), '<retained terminal policy>', 'exec'), policy)
    protocol = policy['analyzer'](parent.read('proposal/protocol/tirust_benchmark_worker.py'),
                                  parent.read('proposal/protocol/tirust_benchmark.py'))
    retention = own.load('retention.json')
    require(retention['preparation_index_sha256'] == PARENT_INDEX
            and retention['capture_sha256'] == sha(own.read('capture.py')), 'Retention binding differs')
    def closed(name):
        item = retention['closed_files'][safe(name)]
        raw = (parent if item['storage'] == 'preparation' else own).read('closed/' + name)
        require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Closed original differs')
        return raw
    def load(name): return json.loads(closed(name))
    for name in retention['closed_files']: closed(name)
    for name, item in parent.load('retention.json')['closed_files'].items():
        require(retention['closed_files'][name] == {**item, 'storage': 'preparation'}, 'Prepared history changed')
    stage = load('stage-run.json'); accounting = load('request-accounting.json'); aborted = load('aborted-request.json')
    terminal = own.load('root-terminal-observation.json')
    require(terminal['session_id'] == 29078 and terminal['exit_code'] == 1
            and terminal['kind'] == 'root-reported-tool-terminal-observation', 'Root terminal observation differs')
    require(stage['stage'] == 'run' and stage['status'] == 'failed'
            and stage['exception'] == accounting['native_stage_failure'] == aborted['exception']
            == "TimeoutError('Lock wait expired before execution')"
            and stage['source_index_sha256'] == prepared['summary']['source_index_sha256']
            and stage['campaign_manifest_sha256'] == prepared['summary']['campaign_manifest_sha256']
            and stage['native_execution'] is True and stage['performance_eligible'] is False
            and stage['cap_seconds_excluding_lock_wait'] == 9120, 'Actual terminal stage differs')
    plan = parent.load('proposal/proposal.json'); records = {}; command_records = {}
    for name in retention['closed_files']:
        if name.startswith('execution/') and name.endswith('.json') and retention['closed_files'][name]['storage'] == 'own':
            cmd = load(name)
            require(cmd['returncode'] == 0 and cmd['exception'] is None and cmd['cleanup_exception'] is None
                    and not cmd['cleanup']['survivors'] and cmd['cpu'] == 12 and cmd['cpu_exclusive'] is False,
                    'Native controller command did not close cleanly')
            require(sha(closed(name[:-5] + '.txt')) == cmd['log_sha256'], 'Command stdout differs')
            command_records[PurePosixPath(name).stem] = cmd
    post = own.load('postclosure-bindings.json')
    # Replay the exact card writer as pure source; it performs no I/O.
    functions = [n for n in ast.parse(parent.read('source/run.py')).body if isinstance(n, ast.FunctionDef) and n.name == 'render_card']
    card_namespace = {'require': require}
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen card serializer>', 'exec'), card_namespace)
    completions = {}
    for name in sorted(retention['closed_files']):
        if name.startswith('request-completions/'):
            value = load(name); rid = policy['key'](value['request']); require(rid not in completions, 'Duplicate completion')
            completions[rid] = value
    for request in plan['requests']:
        rid = policy['key'](request); prefix = 'runs/' + rid + '/'
        available = [n for n in retention['closed_files'] if n.startswith(prefix)]
        if not available: continue
        require(rid in completions, 'Actual packet has unaccounted launched request')
        spec = load(prefix + 'specification.json'); result = load(prefix + 'result.json'); completion = load(prefix + 'completion.json')
        cid = request['case']; source = request['source']; label = source + '-' + cid
        build = parent.load('closed/builds/' + label + '/build.json'); preflight = parent.load('closed/builds/' + label + '/preflight.json')
        require(spec['request'] == request and spec['backend'] == request['backend'] and spec['build'] == build
                and spec['scientific_sources'] == build['scientific_sources']
                and spec['runtime']['files'] == preflight['runtime']['files'], 'Worker/source/runtime identity differs')
        case = next(e for e in plan['cases'] if e['case']['id'] == cid)
        require(spec['points_sha256'] == case['point_sha256'] == sha(parent.read('proposal/points/' + cid + '/selected.dat'))
                and spec['physical_card_sha256'] == case['physical_card_sha256'] == sha(closed(prefix + 'param_card.dat')), 'Worker physical input differs')
        expected_settings = dict(parent.load('proposal/policy-reference.json')[source]); expected_settings['mlreductionlib'] = str(request['backend'])
        original_card = parent.read('generated/' + label + '/Cards/MadLoopParams.dat').decode()
        rendered, ignored = card_namespace['render_card'](original_card, expected_settings, source)
        require(closed(prefix + 'MadLoopParams.dat') == rendered.encode()
                and spec['settings'] == {'requested': expected_settings, 'not_applicable_to_baseline': ignored, 'serialized_sha256': sha(rendered.encode())}
                and set(spec['numerical_cards'].values()) == {sha(rendered.encode())}, 'Numerical policy differs')
        executable = spec['cwd'] + '/check'; expected = {**spec['runtime']['files'], executable: build['executable_sha256']}
        require(spec['expected_runtime'] == expected and post['workers'][rid]['scientific_sources'] == spec['scientific_sources']
                and post['workers'][rid]['numerical_cards'] == spec['numerical_cards']
                and post['workers'][rid]['physical_card_sha256'] == spec['physical_card_sha256'], 'Postclosure worker binding differs')
        cmd = command_records['request-' + cid + '-' + request['label']]
        require(cmd['cwd'] == spec['cwd'] and 0 < cmd['command_work_cap_seconds'] <= 370
                and 0 <= cmd['elapsed_seconds'] <= cmd['command_work_cap_seconds']
                and len(cmd['argv']) == 7 and cmd['argv'][:3] == ['taskset', '-c', '12']
                and cmd['argv'][4] == '-B'
                and cmd['argv'][5].endswith('/physical-stress-current-supervisor-4/native_job.py')
                and cmd['argv'][-1] == retention['original_closed_root'] + '/' + prefix + 'specification.json',
                'Actual request command differs')
        require(spec['argv_without_fifo'] == ['taskset','-c','12','./check',request['points']]
                and spec['actual_stdin_after_READY'] == 'RUN 1 1\nSTOP 0 0\n', 'Actual worker protocol differs')
        require(completion['inputs_unchanged'] and completion['outer_failure'] is None and completion['fatal_infrastructure_failure'] is None
                and completion['result'] == result,
                'Actual successful controller completion differs')
        rc = completions[rid]
        require(0 <= rc['elapsed_seconds'] <= 380 and rc['lock_wait_seconds'] >= 0
                and not request['requires_common_numerical_lock'], 'Actual completed-lane resource accounting differs')
        maps_ok = False
        for observed in result['actual_map_captures']:
            raw = closed(prefix + observed['path']); require(sha(raw) == observed['sha256'], 'Observed map changed')
            elf = {}; data = {}
            for line in raw.decode().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) != 6 or not fields[5].startswith('/'): continue
                path = fields[5]; require(not path.endswith(' (deleted)'), 'Deleted mapped input')
                item = post['mapped_files'][path]
                require(int(fields[4]) == item['inode'] and tuple(int(v,16) for v in fields[3].split(':')) == tuple(item['device']), 'Mapped identity changed')
                resolved = item['resolved']
                if item['header64_hex'].startswith('7f454c46'):
                    require(expected.get(resolved) == item['sha256'], 'Unexpected mapped ELF'); elf[resolved] = item['sha256']
                else:
                    require(path.startswith('/nix/store/') and path.endswith('/lib/locale/locale-archive') and fields[1] == 'r--p', 'Unexpected mapped data')
                    data[resolved] = {'sha256':item['sha256'],'size':item['size'],'inode':item['inode'],'map_device':fields[3],
                                      'header64_hex':item['header64_hex'],'permissions':fields[1]}
            require(elf == observed['elf_files'] and data == observed['data_files'] and (elf == expected) == observed['exact_expected'], 'Recorded map classification differs')
            maps_ok = maps_ok or elf == expected
        raw = {name: closed(prefix + name).decode() for name in ('warmup.dat','dimensions.dat','values.dat','timed_status.dat','timing.dat')
               if prefix + name in retention['closed_files']}
        require({name:sha(value.encode()) for name,value in raw.items()} == result['raw_outputs_sha256'], 'Raw numerical files differ')
        journals = post['workers'][rid]['journals']
        for name, item in journals.items():
            require(sha(own.read(item['retained'])) == item['sha256'], 'Worker journal differs')
        startup = [own.load(v['retained']) for n,v in journals.items() if n.endswith('/startup/startup.json')]
        command = [own.load(v['retained']) for n,v in journals.items() if n.endswith('/000001/command.json')]
        require(len(startup) == len(command) == 1 and startup[0]['error'] is None and command[0]['error'] is None
                and command[0]['sequence'] == command[0]['loops'] == 1,
                'Worker protocol journals did not close')
        for name, item in journals.items():
            if '/000001/' in name and PurePosixPath(name).name in raw:
                require(own.read(item['retained']).decode() == raw[PurePosixPath(name).name], 'Worker final journal differs')
        records[rid] = {'raw':raw,'stdout':closed(prefix+'stdout.txt').decode(),'result':result,'completion':completion,
                        'request_completion':rc,'runtime_verified':maps_ok,'binding_errors':[]}
    summary = policy['derive'](plan, records, stage, accounting, aborted, protocol)
    require(set(records) == set(completions) == set(post['workers']) == set(post['worker_directories'])
            and len(post['worker_directories']) == 4, 'Postclosure worker launch inventory differs')
    require(len(records) == 4 and accounting['completed'] == 4 and aborted['request'] == plan['requests'][4]
            and aborted['state'] == 'deferred-before-launch', 'Actual terminal launch boundary differs')
    summary.update({'role':'current-physical-stress-terminal','prepared_dependency_index_sha256':PARENT_INDEX,
                    'candidate_revision':prepared['summary']['candidate_revision'],
                    'candidate_generator_sha256':prepared['summary']['candidate_generator_sha256'],
                    'campaign_manifest_sha256':prepared['summary']['campaign_manifest_sha256'],
                    'source_index_sha256':prepared['summary']['source_index_sha256'],
                    'candidate_DSO_sha256':prepared['summary']['candidate_DSO_sha256'],
                    'root_session_id':29078,'root_exit_code':1,'terminal_elapsed_seconds':stage['elapsed_seconds_including_wait'],
                    'state_counts':dict(Counter(row['state'] for row in summary['request_rows']))})
    if not derive_summary: require(summary == own.load('qualification-summary.json'), 'Saved terminal summary differs')
    return {'verified':True,'index_sha256':sha(own.index_raw),'preparation_index_sha256':PARENT_INDEX,'summary':summary}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('root', nargs='?')
    print(json.dumps(verify(parser.parse_args().root), sort_keys=True, indent=2))
