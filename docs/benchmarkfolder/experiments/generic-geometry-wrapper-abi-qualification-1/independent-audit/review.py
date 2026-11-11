"""Independent text-only replay; no compiler, loader, ELF or scientific calls."""
import ast
import builtins
import copy
from collections import Counter
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import types

ROOT = Path.cwd()
SOURCE = ROOT / '.tirust-build/generic-geometry-wrapper-abi-proposal-1'
EVIDENCE = ROOT / '.tirust-build/generic-geometry-wrapper-abi-qualification-1'
HERE = Path(__file__).resolve().parent
PINS = {
    'index.json': '35018c3a05f2e863be061c68bbba9fc64ccf15009c94f68ec3012f955601394d',
    'build/result.json': '98a01b832cd5e26d4937a6d91f0b2e88f8298439ae1daa5c0464f1fc4d743cfe',
    'run/result.json': 'ae31fcd44874b34053cb5174c680d3ec6e143cd6f30396c6067a39e67882de03',
    'run/native.log': '6472be073e348b37d97affc731e10444a816832a4e7b5ae645200cd9a48e0832',
    'run/runtime-maps.txt': 'cf0b126b5da189f17c8282d803f7cdd3ebd1490e573a7bf9206742de84975256',
}

def identity(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())

def require(condition, message):
    if not condition:
        raise ValueError(message)

index_bytes = (SOURCE / 'index.json').read_bytes()
require(identity(index_bytes)['sha256'] == PINS['index.json'], 'source index')
index = json.loads(index_bytes)
source = {}
for name, pin in index['files'].items():
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'source path')
    path = SOURCE / relative
    require(not path.is_symlink(), 'source symlink')
    data = path.read_bytes()
    data.decode('utf8')
    require(identity(data) == pin, 'source payload ' + name)
    source[name] = data
require({str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*') if p.is_file()} ==
        set(source) | {'index.json'}, 'source inventory')
evidence = {}
for name, pin in PINS.items():
    if name == 'index.json':
        continue
    data = (EVIDENCE / name).read_bytes()
    require(identity(data)['sha256'] == pin, 'evidence ' + name)
    evidence[name] = data
build = json.loads(evidence['build/result.json'])
run = json.loads(evidence['run/result.json'])
require(len(build['commands']) == 7 and len(run['commands']) == 1, 'command count')
for receipt in (build, run):
    require(receipt['passed'] is True and receipt['source_index_sha256'] == PINS['index.json'], 'receipt source/pass')
    require(receipt['elapsed_seconds'] <= receipt['budget_seconds'], 'receipt deadline')
    require(receipt['full_sloop_lifetime_coverage'] is False and receipt['scientific_evaluations'] == 0, 'scope')
    for command in receipt['commands']:
        require(command['returncode'] == 0 and command['progress']['returncode'] == 0
                and command['progress']['process_reaped'] is True, 'command closure')
require(run['commands'][0]['log'] == identity(evidence['run/native.log']), 'native log binding')
require(run['runtime']['maps_sha256'] == PINS['run/runtime-maps.txt'], 'maps binding')
require(build['executable'] == run['executable'], 'executable identity')
require(len(run['runtime']['observed_file_backed_ELF']) == 6, 'runtime size')
require(run['runtime']['observed_file_backed_ELF_rehashed_after_exit'] is True, 'producer runtime posthash record')
paths = {line.split(None, 5)[5] for line in evidence['run/runtime-maps.txt'].decode().splitlines()
         if len(line.split(None, 5)) == 6 and line.split(None, 5)[5].startswith('/')}
require(paths == set(run['runtime']['observed_file_backed_ELF']), 'raw runtime path set')
require((EVIDENCE / 'run/inputs.txt').read_bytes() == source['inputs.txt'], 'runtime input bytes')
for name, pin in build['sources'].items():
    data = (EVIDENCE / 'build' / name).read_bytes()
    require(identity(data) == pin and data == source[name], 'built source ' + name)

# All replay code is loaded from authenticated bytes. Only the selected pure
# definitions execute: the analyzer's filesystem import is replaced by its exact
# already-loaded reference definitions, and prepare's HERE serves memory bytes.
class MemoryPath:
    def __init__(self, name=''):
        self.name = name
    def __truediv__(self, name):
        return MemoryPath(name)
    def read_bytes(self):
        return source[self.name]

attempts = []
def blocked(*args, **kwargs):
    attempts.append('forbidden I/O/process/native API')
    raise AssertionError(attempts[-1])

patches = [(builtins, 'open'), (io, 'open'), (os, 'open'), (os, 'system'),
           (subprocess, 'Popen'), (subprocess, 'run'), (subprocess, 'call'),
           (subprocess, 'check_call'), (subprocess, 'check_output'),
           (ctypes, 'CDLL'), (ctypes, 'PyDLL')]
original = [(obj, key, getattr(obj, key)) for obj, key in patches]
try:
    for obj, key in patches:
        setattr(obj, key, blocked)
    ref = {}
    exec(compile(source['reference.py'], 'authenticated/reference.py', 'exec'), ref)
    REF = types.SimpleNamespace(**{k: v for k, v in ref.items() if k != '__builtins__'})
    analyzer_tree = ast.parse(source['analyze.py'])
    definitions = [n for n in analyzer_tree.body if isinstance(n, ast.FunctionDef) or
                   isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'MASS' for t in n.targets)]
    analyzer = {'REF': REF}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), 'authenticated/analyze.py', 'exec'), analyzer)
    prepare_tree = ast.parse(source['prepare.py'])
    functions = [n for n in prepare_tree.body if isinstance(n, ast.FunctionDef) and
                 n.name in ('power_word', 'base', 'corpus', 'expected', 'render')]
    preparation = {'HERE': MemoryPath(), 'json': json, 'copy': copy, 'load': lambda *args: REF}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'authenticated/prepare.py', 'exec'), preparation)
    cases = json.loads(source['cases.json'])
    expected = json.loads(source['expected.json'])
    require(preparation['corpus']() == cases, 'independent corpus rederive')
    require(preparation['expected'](cases) == expected, 'integer expected bound rederive')
    require(preparation['render'](cases).encode() == source['inputs.txt'], 'wire corpus rederive')
    replay = analyzer['parse'](evidence['run/native.log'], cases, expected)
    require(replay == run['qualification'], 'raw protocol exact replay')
finally:
    for obj, key, value in original:
        setattr(obj, key, value)
require(not attempts, 'forbidden replay attempt')

# Separate direct token counting does not reuse the qualification parser.
rows = [line.split() for line in evidence['run/native.log'].decode('ascii').splitlines() if line.strip()]
counts = Counter(row[0] for row in rows)
require(counts['READY'] == counts['DONE'] == 1, 'outer framing')
require(counts['CASE'] == counts['ENDCASE'] == counts['TOKEN'] == counts['FORWARD'] == 240, 'case counts')
require(counts['BOUNDS'] == counts['MASS'] == 720 and counts['CLASS'] == 2160, 'table counts')
require(counts['DPMOM'] == counts['QPMOM'] == 2880, 'both stored-kind word counts')
require([int(row[1]) for row in rows if row[0] == 'CASE'] == list(range(1, 241)), 'case ordering')
pre = [row[1:] for row in rows if row[0] == 'PRE']
post = [row[1:] for row in rows if row[0] == 'POST']
require(pre == post, 'direct IEEE state equality')
require(all(row[1] == row[2] for row in rows if row[0] == 'TOKEN'), 'direct environment equality')
receipt = dict(schema='independent-generic-wrapper-abi-text-audit-v1', status='pass',
    source_index_sha256=PINS['index.json'], source_payloads=len(source),
    evidence_pins={name: identity(data) for name, data in evidence.items()},
    raw_protocol_replay_exact=True, corpus_and_integer_reference_rederived=True,
    direct_counts=dict(sorted(counts.items())), cases=240, available=180, rejected=60,
    command_receipts_checked=8, commands_recorded_closed_and_reaped=True,
    runtime_text_paths_equal_recorded_6_ELFs=True, live_ELF_or_library_bytes_read=False,
    replay_forbidden_api_attempts=len(attempts), compiled_or_native_executions_by_audit=0,
    audit_harness_first_attempt='Pure replay stopped at NameError: copy missing from injected preparation namespace; added stdlib copy only, no evidence or parser change.',
    full_sloop_lifetime_coverage=False, physical_amplitude_qualified=False,
    scientific_dispatch='recording legacy stub only',
    limitations=['Runtime identities are audited from retained records/maps; no new ELF/library rehash.',
                 'Direct emitted wrapper ABI only; source3 parent caller intervals and producer lifetimes are not qualified.',
                 'No tensor coefficient, physical amplitude, or performance qualification.'],
    review_source_sha256=identity(Path(__file__).read_bytes())['sha256'])
with (HERE / 'receipt.json').open('x') as out:
    json.dump(receipt, out, sort_keys=True, indent=2);out.write('\n')
print(json.dumps(dict(status='pass', receipt_sha256=identity((HERE/'receipt.json').read_bytes())['sha256'],
                     cases=240, available=180, rejected=60, forbidden_attempts=len(attempts)), sort_keys=True))
