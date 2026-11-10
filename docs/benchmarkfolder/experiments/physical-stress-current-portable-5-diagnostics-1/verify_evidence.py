"""Pure console diagnostics for the closed, pinned four-plus-twenty stress replay."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

BASE = 'physical-stress-current-portable-5-terminal-1'
BASE_INDEX = 'a520199cf577d7b35b26b054016607020ebe0a533e5afabb4c1a279812c64266'
BASE_VERIFIER = 'f3eac67396fa8f96dda001902006e674cab4b20e24ef3478bdb2cb7776a3f8e0'
BASE_SUMMARY = '8e928de4a77a9ad146bc49fb7a8b82b899e820f79c703455b4e75e7bf3bd00c4'
ANCESTOR = 'physical-stress-current-portable-4-terminal-2'
ANCESTOR_INDEX = '6fc2035649ba0387f0d4457784ae883592f13bb7e3b6f9c33b99e038b81f7146'
PREPARATION = 'physical-stress-current-portable-4-preflight-1'
PREPARATION_INDEX = '49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22'
DEPENDENCIES = {'base': {'sibling': BASE, 'index_sha256': BASE_INDEX, 'verifier_sha256': BASE_VERIFIER,
                         'summary_sha256': BASE_SUMMARY},
                'ancestor': {'sibling': ANCESTOR, 'index_sha256': ANCESTOR_INDEX},
                'preparation': {'sibling': PREPARATION, 'index_sha256': PREPARATION_INDEX}}


def require(value, message):
    if not value: raise ValueError(message)


def sha(raw): return hashlib.sha256(raw).hexdigest()


def encoded(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def safe(name):
    p = PurePosixPath(name)
    require(isinstance(name, str) and name and '\\' not in name and not p.is_absolute()
            and '..' not in p.parts and str(p) == name, 'Unsafe evidence name')
    return name


def reader(root, index=None):
    """Explicit physical inventory; dependency names never fall back to workspace."""
    root = Path(root).absolute()
    locations = {'': root, 'base/': root.with_name(BASE), 'ancestor/': root.with_name(ANCESTOR),
                 'preparation/': root.with_name(PREPARATION)}
    paths = set()
    for prefix, directory in locations.items():
        require(all(not p.is_symlink() for p in (directory, *directory.parents)), 'Symlink evidence root')
        for path in directory.rglob('*'):
            require(not path.is_symlink() and (path.is_file() or path.is_dir()), 'Unexpected evidence object')
            if path.is_file(): paths.add(prefix + path.relative_to(directory).as_posix())
    if index is not None: paths |= set(index['files']) | {'evidence-index.json'}
    def read(name):
        safe(name); require(name in paths, 'Read outside explicit evidence inventory')
        if name == 'evidence-index.json' and index is not None: return encoded(index)
        for prefix in ('base/', 'ancestor/', 'preparation/'):
            if name.startswith(prefix): return (locations[prefix] / name[len(prefix):]).read_bytes()
        return (root / name).read_bytes()
    read.paths = paths
    return read


def diagnostics(raw):
    if raw is None: return {'stdout_present': False, 'stdout_sha256': None, 'lines': [], 'runtime_reported_flags': []}
    matches, flags = [], set()
    for number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
        kinds = []
        if '##W03' in line: kinds.append('W03')
        tokens = re.findall(r'\bIEEE_[A-Z0-9_]+\b', line)
        if tokens: kinds.append('runtime_reported_IEEE_flags'); flags.update(tokens)
        if re.search(r'\bSIG(?:FPE|SEGV|ABRT|BUS|ILL|TERM|KILL)\b|\breceived signal\b|\bsignal\s+\d+\b', line, re.I):
            kinds.append('signal_diagnostic')
        if re.search(r'reduction.*(?:exception|failure|failed)|(?:ERROR|WARNING)\s+(?:in|from)\s+(?:OneLOop|TIRust|COLLIER|Ninja)', line, re.I):
            kinds.append('reduction_exception_or_library_diagnostic')
        if kinds: matches.append({'line': number, 'text': line, 'kinds': kinds})
    return {'stdout_present': True, 'stdout_sha256': sha(raw), 'lines': matches,
            'runtime_reported_flags': sorted(flags)}


def source_line(raw, path, needle):
    found = [{'path': path, 'sha256': sha(raw), 'line': n, 'text': line}
             for n, line in enumerate(raw.decode().splitlines(), 1) if needle in line]
    require(len(found) <= 1, 'Ambiguous frozen source quote')
    return found[0] if found else None


def derive(base, read_stdout, worker_source, native_source):
    aggregate = base['aggregate']; rows = []
    continuation = {r['request_id'] for r in base['request_rows']}
    for original in aggregate['request_rows']:
        row = {name: original[name] for name in ('request_id', 'case', 'backend', 'state')}
        trace = original.get('warmup_trace', {}).get('rows', [])
        observations = original.get('native_entry_observations', {})
        row.update(diagnostics(read_stdout(original['request_id'])))
        row.update({'group': 'continuation' if row['request_id'] in continuation else 'original',
                    'complete_runtime_map': original.get('complete_runtime_map', False),
                    'warmup_calls_observed': original.get('warmup_calls_observed', 0),
                    'warmup_statuses_by_call': [r['status'] for r in trace if r.get('order') == 0],
                    'completed_warmup_rows': len(trace),
                    'completed_final_rows': len(original.get('final_trace', {}).get('rows', [])),
                    'READY_observed': observations.get('READY_verified_by_frozen_worker'),
                    'RUN_command_journal_count': observations.get('RUN_command_journal_count'),
                    'native_failure': original.get('native_failure'),
                    'native_process_activity_observed': bool(original.get('complete_runtime_map') or trace
                        or original.get('final_trace', {}).get('rows'))})
        rows.append(row)
    require(len(rows) == 24 and len({r['request_id'] for r in rows}) == 24, 'Expected 24 distinct closed requests')
    target = next(r for r in rows if r['request_id'] == 'qcd_n4_uux_zzgg/candidate-9-auto')
    target = {**target, 'worker_timeout_source': source_line(worker_source, 'proposal/protocol/tirust_benchmark_worker.py',
                'raise ProtocolError(f"Worker protocol timeout; see {self.log}")'),
              'native_phase_deadline_source': source_line(native_source, 'source/native_job.py', 'timeout=180.'),
              'internal_in_progress_amplitude_calls_instrumented': False,
              'no_completed_rows_does_not_imply_no_in_progress_amplitude_calls': True}
    return {'base_index_sha256': BASE_INDEX, 'scope': 'Console diagnostics and observed native activity; no new numerical execution',
            'hardware_register_capture': False, 'runtime_reported_flags_are_not_attributed_to_individual_amplitude_calls': True,
            'request_rows': rows, 'continuation_state_counts': base['state_counts'],
            'aggregate_state_counts': dict(Counter(r['state'] for r in rows)),
            'aggregate_checked_streams': aggregate['host_checked_complete_streams'],
            'aggregate_comparisons': len(aggregate['comparisons']),
            'aggregate_failing_comparisons': sum(not c['passed'] for c in aggregate['comparisons']),
            'continuation_complete_runtime_maps': sum(r['complete_runtime_map'] for r in rows if r['group'] == 'continuation'),
            'continuation_native_process_activity_observed': sum(r['native_process_activity_observed'] for r in rows if r['group'] == 'continuation'),
            'streams_with_runtime_reported_flags': sum(bool(r['runtime_reported_flags']) for r in rows),
            'qcd_n4_tirust_timeout': target, 'performance_eligible': False}


def verify(root=None, *, read_bytes=None, derive_summary=False):
    read_bytes = read_bytes or reader(root or Path(__file__).resolve().parent)
    require(hasattr(read_bytes, 'paths'), 'Supply the complete explicit reader.paths inventory')
    require(sha(read_bytes('base/evidence-index.json')) == BASE_INDEX
            and sha(read_bytes('base/verify_evidence.py')) == BASE_VERIFIER, 'Closed base identity differs')
    namespace = {'__name__': 'retained_closed_stress_verifier'}
    exec(compile(read_bytes('base/verify_evidence.py'), '<pinned closed stress verifier>', 'exec'), namespace)
    Packet = namespace['Packet']; own = Packet(read_bytes)
    packets = {prefix: Packet(read_bytes, prefix + '/') for prefix in ('base', 'ancestor', 'preparation')}
    require(own.index['dependencies'] == DEPENDENCIES, 'Companion dependencies differ')
    expected = set(own.files) | {'evidence-index.json'}
    for prefix, packet in packets.items(): expected |= {prefix + '/' + name for name in packet.files} | {prefix + '/evidence-index.json'}
    require(set(read_bytes.paths) == expected, 'Reader dependency closure differs')
    def base_reader(name): return read_bytes(name if name.startswith(('ancestor/', 'preparation/')) else 'base/' + name)
    base_reader.paths = set(packets['base'].files) | {'evidence-index.json'}
    for prefix in ('ancestor', 'preparation'):
        base_reader.paths |= {prefix + '/' + n for n in packets[prefix].files} | {prefix + '/evidence-index.json'}
    proof = namespace['verify'](read_bytes=base_reader)
    require(proof['verified'] and sha(packets['base'].read('qualification-summary.json')) == BASE_SUMMARY, 'Closed base proof differs')
    retentions = {p: packets[p].load('retention.json') for p in ('base', 'ancestor')}
    current = {r['request_id'] for r in proof['summary']['request_rows']}
    def console(rid):
        group = 'base' if rid in current else 'ancestor'; name = 'runs/' + rid + '/stdout.txt'
        item = retentions[group]['closed_files'].get(name)
        if item is None: return None
        packet = packets['preparation'] if item['storage'] == 'preparation' else packets[group]
        return packet.read('closed/' + name)
    summary = derive(proof['summary'], console, packets['preparation'].read('proposal/protocol/tirust_benchmark_worker.py'),
                     packets['preparation'].read('source/native_job.py'))
    if not derive_summary: require(own.load('qualification-summary.json') == summary, 'Saved diagnostics differ')
    return {'verified': True, 'index_sha256': sha(own.index_raw), 'base_index_sha256': BASE_INDEX, 'summary': summary}


def seal(root):
    root = Path(root); require(not (root / 'evidence-index.json').exists() and not (root / 'qualification-summary.json').exists(), 'Append-once companion')
    index = {'schema_version': 1, 'dependencies': DEPENDENCIES, 'files': {}}
    for name in ('verify_evidence.py', 'test_diagnostics.py', 'README.md'):
        raw = (root / name).read_bytes(); index['files'][name] = {'kind': 'file', 'sha256': sha(raw), 'size': len(raw)}
    summary = verify(read_bytes=reader(root, index), derive_summary=True)['summary']
    raw = encoded(summary)
    with (root / 'qualification-summary.json').open('xb') as stream: stream.write(raw)
    index['files']['qualification-summary.json'] = {'kind': 'file', 'sha256': sha(raw), 'size': len(raw)}
    verify(read_bytes=reader(root, index))
    with (root / 'evidence-index.json').open('xb') as stream: stream.write(encoded(index))
    return verify(root)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--seal', action='store_true'); args = parser.parse_args()
    result = seal(Path(__file__).resolve().parent) if args.seal else verify()
    print(json.dumps({k: result[k] for k in ('verified', 'index_sha256', 'base_index_sha256')}, sort_keys=True))
