"""Independent retained-text replay. No compiler, loader, reducer or child."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import encodings.ascii
import hashlib
import json
import math
import os
import re
import struct
import sys
import time

ROOT = Path.cwd().resolve()
PROPOSAL = ROOT / '.tirust-build/subprocess-cache-successor-proposal-2'
REPORT = ROOT / '.tirust-build/subprocess-cache-successor-results-2'
HERE = Path(__file__).resolve().parent
PINS = {
    PROPOSAL / 'source-index.json': '3d1f787bae4d5af45c2a34119f46b3235e64087224c698fca3eb0eb66816ada7',
    PROPOSAL / 'analyze.py': '43b16c1da49ab62eafa32be30cd027c5cde3d2ce3c1fa77962f6d2db569bff3d',
    REPORT / 'results.json': 'ba66b9bd6015bd29c22242f227f4404582d42eee6123f198b4b157e225b69770',
    REPORT / 'build.json': '6f9e246f3c23ac09c788ec4f0a5a2a9e6f800182ebaee4ae8fd1e0febe249e11',
}
LOCALE = '/nix/store/ig5ylfz7qcgpmvzip4pxfkp2pfi76s2j-glibc-locales-2.44-25/lib/locale/locale-archive'
blocked = []
reads = {}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def get(path):
    raw = path.read_bytes()
    reads[str(path.relative_to(ROOT))] = {'sha256': digest(raw), 'bytes': len(raw)}
    return raw


def load(path):
    return json.loads(get(path))


def audit(event, args):
    if event == 'subprocess.Popen' or event.startswith(('ctypes.', 'os.exec', 'os.spawn')) or event in ('os.system', 'os.fork', 'os.forkpty', 'os.posix_spawn'):
        blocked.append(event)
        raise RuntimeError('forbidden native/process API ' + event)
    if event == 'open':
        path, mode, flags = args
        if isinstance(path, int):
            return
        p = Path(path).resolve()
        writing = flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        if writing:
            if p != HERE / 'receipt.json':
                raise RuntimeError('write outside independent receipt')
        elif not any(p.is_relative_to(base) for base in (PROPOSAL, REPORT, HERE)):
            raise RuntimeError('read outside retained sources/evidence: ' + str(p))


sys.addaudithook(audit)
for path, checksum in PINS.items():
    assert digest(get(path)) == checksum, str(path)
source = get(PROPOSAL / 'analyze.py')
namespace = {'__file__': str(PROPOSAL / 'analyze.py'), '__name__': 'independent_cache_replay'}
exec(compile(source, str(PROPOSAL / 'analyze.py'), 'exec'), namespace)
start = time.monotonic()
replayed = json.loads(json.dumps(namespace['analyze'](REPORT / 'runs'), allow_nan=False))
assert replayed == load(REPORT / 'results.json') and replayed['passed']
elapsed = time.monotonic() - start
build = load(REPORT / 'build.json')
stage = load(REPORT / 'stage-run.json')
assert stage['status'] == 'passed' and stage['exception'] is None
assert stage['elapsed_seconds'] <= stage['cap_seconds'] == 750
labels = ['plain-cache0', 'plain-cache-all', 'observed-cache0', 'observed-cache-all']
canonical = None
summaries = []
for label in labels:
    directory = REPORT / 'runs' / label
    lines = [line.split() for line in get(directory / 'records.txt').decode().splitlines() if line.split()]
    groups = defaultdict(list)
    for row in lines:
        groups[row[0]].append(row[1:])
    amps = {int(r[0]): r for r in groups['AMP']}
    pre = {int(r[0]): r[1:] for r in groups['LATCHI']}
    post = {int(r[0]): r[1:] for r in groups['LATCH']}
    assert len(amps) == 21 and set(amps) == {101, 201, *range(301, 320)}
    assert all(int(r[3]) == 219 for r in amps.values())
    assert groups['WARM'] == [['1', '1']] and len(groups['KEY']) == 28
    assert pre[301] == ['F'] * 4
    for action in range(302, 320):
        assert pre[action] == ['T'] * 4
        own = 0 if int(amps[action][1]) == 1 else 2
        assert post[action][own:own+2] == ['F'] * 2
        assert post[action][2-own:4-own] == ['T'] * 2
    assert all(r[1:] == ['F', 'F', 'F', 'F', 'T'] * 2 for r in groups['FLAGS'])
    normalized = [(row[:1] + row[2:] if row[0] == 'HEADER' else row)
                  for row in lines if row[0] not in ('BEFORE', 'AFTER')]
    if canonical is None:
        canonical = normalized
    else:
        assert normalized == canonical, label
    before = {int(r[0]): list(map(int, r[1:])) for r in groups['BEFORE']}
    after = {int(r[0]): list(map(int, r[1:])) for r in groups['AFTER']}
    deltas = {a: [y-x for x, y in zip(before[a], after[a])] for a in before}
    for precision in (1, 2):
        assert deltas[1000 + precision*100 + 13] == [0]*14 + [int(precision == 2)]
    if label.endswith('cache-all'):
        assert sum(deltas[1114][i] for i in (0, 2, 4)) > 0 and deltas[1114][6:8] == [0, 0]
        assert deltas[1214][9:] == [1, 0, 0, 0, 0, 0]
    trace = [line.split() for line in get(directory / 'trace.txt').decode().splitlines()]
    events = [list(map(int, r[1:])) for r in trace if r[0] == 'R']
    flush = [list(map(int, r[1:])) for r in trace if r[0] == 'F']
    assert len(flush) == 54 and all(r[3] == 0 for r in flush)
    summary = dict(label=label, physical_calls=19, warmup=[1, 1], status219=len(amps),
                   key_calls=28, poisoned_latch_calls=18, healthy_flushes=54,
                   key13_deltas=[deltas[1113], deltas[1213]],
                   key14_deltas=[deltas[1114], deltas[1214]])
    if label.startswith('observed'):
        counts = Counter(v[3] for v in events)
        assert counts == {1:53, 2:2, 3:2, 4:41, 5:4, 6:57, 7:14}, counts
        native = [v for v in events if v[3] in (6, 7)]
        assert len(native) == 71 and all(v[6] == 0 for v in native)
        for action in range(301, 320):
            g = [v for v in events if v[1] == action]
            ev = [v for v in g if v[3] in (6, 7)]
            cl = [v for v in g if v[3] == 4]
            assert [(v[3], v[11], v[12], v[6]) for v in ev] == [
                (6, 0, int(label.endswith('cache-all')), 0),
                (6, 1, int(label.endswith('cache-all')), 0)]
            assert len(cl) == (1 if action == 317 else 2)
            assert cl[-1][0] > ev[-1][0] and all(x == 0 for v in cl for x in v[41:])
            if action == 317:
                assert [v[3] for v in g] == [1, 2, 5, 6, 6, 4]
            else:
                assert cl[0][0] < ev[0][0]
        summary.update(trace_rows=len(events), native_evaluations=len(native),
                       trace_kind_counts=dict(counts), physical_clears=37)
    inputs = load(directory / 'inputs.json')
    outputs = load(directory / 'outputs.json')
    command = load(REPORT / 'execution' / ('native-' + label + '.json'))
    console = get(REPORT / 'execution' / ('native-' + label + '.txt'))
    assert inputs['build_sha256'] == PINS[REPORT / 'build.json']
    assert command['returncode'] == 0 and command['exception'] is None
    assert command['cleanup_exception'] is None and not command['cleanup']['survivors']
    assert command['elapsed_seconds'] <= command['command_work_cap_seconds']
    assert command['cpu'] == 12 and command['cpu_exclusive'] is False
    assert command['log_sha256'] == digest(console)
    assert not re.search(rb'##W03\b', console, re.I)
    assert outputs['inputs_unchanged'] and outputs['execution_exception'] is None
    for name, pin in outputs['declared_outputs'].items():
        raw = get(directory / name)
        assert pin['kind'] == 'file' and len(raw) == pin['size'] and digest(raw) == pin['sha256']
    expected_paths = set(inputs['executable']['runtime']['files']) | {command['argv'][3]}
    mapping_reports = []
    for name in ('runtime-maps-open.txt', 'runtime-maps-close.txt'):
        mappings = [line.split(maxsplit=5) for line in get(directory / name).decode().splitlines()]
        mappings = [r for r in mappings if len(r) == 6 and r[5].startswith('/')]
        assert all(not r[5].endswith(' (deleted)') for r in mappings)
        actual_paths = {r[5] for r in mappings}
        assert expected_paths <= actual_paths
        extras = actual_paths - expected_paths
        assert extras <= {LOCALE}
        assert all('x' not in r[1] for r in mappings if r[5] in extras)
        mapping_reports.append(dict(file=name, bound_elf_paths=len(expected_paths),
                                    nonexecuting_extra_paths=sorted(extras)))
    summary.update(runtime_map_text=mapping_reports, command_returncode=0,
                   ieee_flags='invalid/divzero/overflow/underflow false; inexact sticky true',
                   console_w03_count=0)
    summaries.append(summary)
receipt = dict(schema='successor-cache-independent-retained-text-review-v1', passed=True,
    reviewer='acceptance_audit', source_sha256=digest(get(Path(__file__))),
    replay_exact=True, replay_seconds=elapsed, forbidden_api_attempts=blocked,
    observations=summaries, read_inputs=reads,
    limitations=['No compiler, loader, reducer or child process was invoked.',
        'Full source/object/live ELF hash closure is a separate root review; this receipt joins retained build/command/output/runtime-map text.',
        'The nonexecuting locale archive mapping is retained explicitly and excluded from ELF dependency comparison; its bytes were not read by this audit.',
        'Physical amplitudes use native DP; QP evidence is standalone tensor-key diagnostics with declared DP-promoted inputs.',
        'No rank-six, CMS, performance or generic physical-stress claim.'],
    initial_replay_attempts=['Read guard rejected lazy stdlib ASCII codec import before replay; codec preloaded for the successful bounded replay.',
        'Direct Python equality rejected tuple/list serialization distinctions; successful comparison uses the exact JSON-compatible analyzer serialization without numerical changes. No evidence was mutated.'])
with (HERE / 'receipt.json').open('x') as stream:
    json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
    stream.write('\n')
print(json.dumps({'passed': True, 'receipt_sha256': digest((HERE/'receipt.json').read_bytes()),
                  'receipt': str(HERE/'receipt.json'), 'workers': len(summaries)}))
