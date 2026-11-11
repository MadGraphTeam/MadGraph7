"""Read-only retained-text rank-six replay; no numerical library is loaded."""
import builtins
from collections import Counter
import ctypes
import dataclasses
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import re
import struct
import subprocess
import sys
import types

ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
SCI = ROOT / '.tirust-build/rank6-ir-triangle-controller-1'
VALIDATOR = ROOT / '.tirust-build/rank6-ir-triangle-validation-controller-1'
OWNER = ROOT / '.tirust-build/rank6-ir-triangle-run-controller-1'
FAILED = ROOT / '.tirust-build/rank6-ir-triangle-work-1/build'
VALIDATED = ROOT / '.tirust-build/rank6-ir-triangle-validation-work-1/validate'
RUN = ROOT / '.tirust-build/rank6-ir-triangle-run-work-1/run'
SOURCES = {
    SCI: '4f496ebf3071e8d405e1e2e72b4bcff3ab7b210ef2338740cdbe314b0cfcb2b0',
    VALIDATOR: '73f311c041c176fe80c00b53f51935d8283521672717247ee73d26c855ee0640',
    OWNER: '1d1642c77d698afd8f0a4f8fa3c0db361733fa6ca6d9b1c1e1e44cbdd8be30cf',
}
PINS = {
    RUN / 'terminal.json': '72d946f7cb97ad5f7a707724560255863bbda422812d972fd0a91c5d6c68e998',
    RUN / 'analysis.json': 'da7f8e594b061c4eae65c60b2586a49c4b61269042fdd3983e032a4a12dd3584',
    RUN / 'result.json': 'bebe528042cd59d26e97605e84da3d5b53fc90f218bc93c92f16af420debcec9',
    VALIDATED / 'terminal.json': '9c6d48c1db7e725db8a7579d9d6dc1ea623c5f22f4603232ebf3540042f4715c',
    VALIDATED / 'result.json': 'd34d15d59ed16cdab5927fd219146979f0bd77d567dcd3ab831d547ff121dc79',
    VALIDATED / 'amended-prebound.json': 'af27f81ff8e853595ad83e7e9be2a96a6d8df974486fab3d2617b89e0ff97d70',
    FAILED / 'terminal.json': 'e1d442dbc86c04606a5aba1ee5f83614b00c64d6ac9ef0a07558277fc5f2ad77',
}
memory = {}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def identity(raw):
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())

def read(path):
    path = Path(path)
    if path not in memory:
        require(path.is_file() and not path.is_symlink(), 'nonregular input')
        raw = path.read_bytes();raw.decode('utf8')
        require(b'\0' not in raw, 'nontext input')
        memory[path] = raw
    return memory[path]

def js(path):
    return json.loads(read(path))

source_counts = {}
for folder, checksum in SOURCES.items():
    raw = read(folder / 'source-index.json')
    require(identity(raw)['sha256'] == checksum, 'source seal')
    index = json.loads(raw)
    for name, pin in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'source path')
        require(identity(read(folder / name)) == pin, 'source payload ' + name)
    require({str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()} ==
            set(index['files']) | {'source-index.json'}, 'source inventory')
    source_counts[str(folder.relative_to(ROOT))] = len(index['files'])
for path, checksum in PINS.items():
    require(identity(read(path))['sha256'] == checksum, 'closed evidence pin')

owner_pins = js(OWNER / 'input-pins.json')
ancestor_text_counts = {}
for folder, key in ((FAILED, 'failed_build_files'), (VALIDATED, 'validated_files')):
    files = owner_pins[key]
    require({str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()} == set(files), 'ancestor inventory')
    count = 0
    for name, pin in files.items():
        # Original compiled products stay unopened. Their recorded identities
        # and unchanged inventory are distinguished from fresh text hashes.
        if Path(name).suffix not in ('.json', '.txt', '.log', '.map'):
            continue
        require(identity(read(folder / name)) == pin, 'ancestor text changed ' + name)
        count += 1
    ancestor_text_counts[key] = dict(inventory=len(files), fresh_text_hashes=count)
failed = js(FAILED / 'terminal.json')
validated = js(VALIDATED / 'terminal.json')
validation = js(VALIDATED / 'result.json')
terminal = js(RUN / 'terminal.json')
result = js(RUN / 'result.json')
require(failed['status'] == 'failed' and failed['returncode'] == 1 and not failed['cleanup']['survivors'], 'original failure reclassified')
for t in (validated, terminal):
    require(t['status'] == 'passed' and t['returncode'] == 0 and not t['cleanup']['survivors'], 'unclosed successful stage')
require(validated['result_sha256'] == PINS[VALIDATED/'result.json'], 'validation result join')
require(terminal['result_sha256'] == PINS[RUN/'result.json'], 'RUN result join')
require(result['analysis_sha256'] == PINS[RUN/'analysis.json'], 'analysis join')
require(result['source_index_sha256'] == SOURCES[OWNER] and result['reviewed_validation_sha256'] == PINS[VALIDATED/'terminal.json'], 'RUN ownership join')
require(result['status'] == 'passed' and result['returncode'] == 0 and result['native_progress']['returncode'] == 0
        and result['native_progress']['process_reaped'] is True, 'native command closure')
require(read(RUN/'native-terminal.json') == read(RUN/'result.json'), 'native terminal/result differ')
for path in RUN.rglob('*'):
    if path.is_file():
        read(path)
require(read(RUN/'schedule.dat') == read(SCI/'schedule.dat'), 'schedule copied bytes')
require(result['runtime']['files'] == validation['runtime']['files'] == js(SCI/'input-pins.json')['expected_runtime'], 'runtime provider join')
mapping = result['mapping']
require(mapping['expected_runtime_sha256'] == mapping['mapped_files_sha256_after_exit'] == result['runtime']['files'], 'recorded runtime posthash join')
require(len(result['runtime']['files']) == 15 and mapping['cpu_allowed_list'] == '12', 'runtime/CPU scope')
require(identity(read(RUN/'maps.txt'))['sha256'] == mapping['maps_sha256'], 'raw map binding')
paths = {line.split(None, 5)[5] for line in read(RUN/'maps.txt').decode().splitlines()
         if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')}
require(paths == set(mapping['mapped_paths']) == set(mapping['mapped_elf_paths']), 'raw map path join')
require(paths == set(result['runtime']['files']) | {validation['executable']}, 'runtime set mismatch')
require(result['argv'] == [validation['executable']], 'exact native executable')

attempts = []
def blocked(*args, **kwargs):
    attempts.append('forbidden I/O/process/native API')
    raise AssertionError(attempts[-1])

def memory_bytes(path):
    require(path in memory, 'unloaded/outside replay path ' + str(path))
    return memory[path]

def memory_text(path, *args, **kwargs):
    return memory_bytes(path).decode('utf8')

def memory_resolve(path, *args, **kwargs):
    require(path.is_absolute() and '..' not in path.parts and (path == SCI or SCI in path.parents), 'outside source resolution')
    return path

patches = [(builtins, 'open', blocked), (io, 'open', blocked), (os, 'open', blocked),
           (os, 'system', blocked), (subprocess, 'Popen', blocked), (subprocess, 'run', blocked),
           (subprocess, 'call', blocked), (subprocess, 'check_call', blocked), (subprocess, 'check_output', blocked),
           (ctypes, 'CDLL', blocked), (ctypes, 'PyDLL', blocked),
           (Path, 'read_bytes', memory_bytes), (Path, 'read_text', memory_text), (Path, 'resolve', memory_resolve)]
previous = [(obj, key, getattr(obj, key)) for obj, key, _ in patches]
modules = {name: sys.modules.get(name) for name in ('oracle', 'legacy_parser', 'rank6_audit_analyze')}
try:
    for obj, key, value in patches:
        setattr(obj, key, value)
    for name, filename in (('oracle', 'oracle.py'), ('legacy_parser', 'legacy_parser.py'), ('rank6_audit_analyze', 'analyze.py')):
        module = types.ModuleType(name);module.__file__ = str(SCI/filename);sys.modules[name] = module
        exec(compile(memory[SCI/filename], module.__file__, 'exec'), module.__dict__)
    replay = sys.modules['rank6_audit_analyze'].analyze(RUN)
    require(json.loads(json.dumps(replay)) == js(RUN/'analysis.json'), 'exact raw parser/oracle replay')
finally:
    for obj, key, value in previous:
        setattr(obj, key, value)
    for name, old in modules.items():
        if old is None:sys.modules.pop(name, None)
        else:sys.modules[name] = old
require(not attempts, 'forbidden API attempted')
require(replay['status'] == 'passed' and not replay['open_items'] and len(replay['checks']) == 24, 'numerical status')
require(all(x['native_present'] and x['oracle']['passed'] for x in replay['checks']), 'missing successful rational check')
require(len(replay['full_partial']) == 12 and all(x['status'] == 'passed' and x['all_pole_words_equal'] for x in replay['full_partial']), 'full/partial exact words')

# Independent framing/options/IEEE count, separate from the imported analyzer.
rows = [line.split() for line in read(RUN/'records.txt').decode().splitlines() if line.strip()]
counts = Counter(row[0] for row in rows)
require(counts['CASE'] == counts['STATE'] == counts['END'] == 24 and counts['V'] == 24*210*3, 'raw output coverage')
require([int(row[1]) for row in rows if row[0] == 'CASE'] == list(range(4101, 4125)), 'raw action order')
states = [list(map(int, row[1:])) for row in rows if row[0] == 'STATE']
require(all(x[4] == 1 and not (x[0] & 13 or x[2] & 13) and x[1] == x[3] == 0 for x in states), 'raw host state')
frames = [json.loads(line) for line in read(RUN/'observer.txt').decode().splitlines()]
require(len(frames) == 24 and all(x['present'] == 1 and x['status'] == 0 and x['health'] == 0 and x['rank'] == 6 and x['cache'] == 0 and x['check'] == 0 for x in frames), 'raw native status/options')
require(Counter(x['kind'] for x in frames) == {1:12, 2:12} and Counter(x['n'] for x in frames) == {2:12, 3:12}, 'raw native precision/topology')
require(all(not any(f & 13 for f in x['flags']) and x['rounding'] == [0,0,0] for x in frames), 'raw native IEEE')
require(all(int(line.split()[2]) == 0 for line in read(RUN/'errors.txt').decode().splitlines()), 'unexpected host error')
schedule = js(SCI/'schedule.json')
worst = {}
for kind, ct in (('DP', 1), ('QP', 4)):
    selected = [c['oracle'] for c, s in zip(replay['checks'], schedule) if s['ct'] == ct]
    worst[kind] = {k:str(max(Fraction(x[k]) for x in selected)) for k in
                  ('maximum_scaled_error', 'maximum_nonzero_symmetric_residual', 'maximum_expected_zero_scaled_leakage')}

receipt = dict(schema='independent-rank6-ir-triangle-retained-text-audit-v1',status='pass',
    source_indices={str(p.relative_to(ROOT)):sha for p,sha in SOURCES.items()},source_payload_counts=source_counts,
    pinned_closures={str(p.relative_to(ROOT)):identity(memory[p]) for p in PINS},
    retained_run_text_files={str(p.relative_to(RUN)):identity(b) for p,b in memory.items() if RUN in p.parents},
    ancestor_text_checks=ancestor_text_counts,original_failed_build_still_failed=True,
    validated_runtime_providers=15,runtime_identity_scope='retained records/maps, no fresh binary hash',
    raw_replay_exact=True,independent_direct_counts=dict(sorted(counts.items())),
    cases=24,components_per_case=210,actual_DP_rows=12,actual_QP_rows=12,
    rational_oracle_passes=24,refusals=0,open_items=0,full_partial_pairs_with_exact_pole_words=12,
    initial_IEEE=replay['initial_IEEE'],model_IEEE=replay['model_IEEE'],worst_errors=worst,
    forbidden_api_attempts=len(attempts),audit_native_executions=0,audit_binary_reads=0,
    cache_enabled=False,physical_amplitude_qualified=False,physical_stress_qualified=False,
    amplitude_rescue_qualified=False,trap_safety_qualified=False,performance_acceptance=False,
    scope='Historical generated CAP6 adapter with eb3d bridge; two exact-input standalone tensor fixtures, no numerator contraction.',
    review_source_sha256=identity(Path(__file__).read_bytes())['sha256'])
with (HERE/'receipt.json').open('x') as f:
    json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(status='pass',receipt_sha256=identity((HERE/'receipt.json').read_bytes())['sha256'],
                     cases=24,full_partial_exact_pairs=12,forbidden_attempts=len(attempts),worst_errors=worst),sort_keys=True))
