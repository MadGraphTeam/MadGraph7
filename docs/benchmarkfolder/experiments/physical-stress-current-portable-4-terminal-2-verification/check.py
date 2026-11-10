"""Pure-Python two-sibling relocation and tampering checks; no subprocesses."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
import time
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKET = HERE.with_name('physical-stress-current-portable-4-terminal-2')
PARENT = HERE.with_name('physical-stress-current-portable-4-preflight-1')
PREDECESSOR = HERE.with_name('physical-stress-current-portable-4-terminal-1')
spec = importlib.util.spec_from_file_location('terminal_portability_check', PACKET / 'verify_evidence.py')
verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)


def sha(raw): return hashlib.sha256(raw).hexdigest()
def dump(value): return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


start = time.monotonic()
index_raw = (PACKET / 'evidence-index.json').read_bytes()
index = json.loads(index_raw)
parent_index_raw = (PARENT / 'evidence-index.json').read_bytes()
parent_index = json.loads(parent_index_raw)
payloads = {n: (PACKET / n).read_bytes() for n in index['files']}
payloads['evidence-index.json'] = index_raw
payloads.update({'preparation/' + n: (PARENT / n).read_bytes() for n in parent_index['files']})
payloads['preparation/evidence-index.json'] = parent_index_raw
before = {n: sha(raw) for n, raw in payloads.items()}


class Reader:
    def __init__(self, values):
        self.values = values
        self.paths = frozenset(values)
    def __call__(self, name): return self.values[name]


original = verifier.verify(PACKET)
with patch.object(Path, 'open', side_effect=AssertionError('Callback verifier attempted filesystem access')):
    assert verifier.verify(read_bytes=Reader(payloads)) == original
checks = [{'name': 'authenticated_callback_closure_without_filesystem_reads', 'passed': True}]

with tempfile.TemporaryDirectory(prefix='stress-terminal-portable-') as temporary:
    relocated = Path(temporary) / PACKET.name
    sibling = Path(temporary) / PARENT.name
    shutil.copytree(PACKET, relocated)
    shutil.copytree(PARENT, sibling)
    old_open = Path.open
    def confined_open(path, *args, **kwargs):
        resolved = Path(path).absolute()
        if not (resolved.is_relative_to(relocated) or resolved.is_relative_to(sibling)):
            raise AssertionError('Verifier attempted a read outside the two relocated capsules: ' + str(path))
        return old_open(path, *args, **kwargs)
    with patch.object(Path, 'open', confined_open):
        assert verifier.verify(relocated) == original
checks.append({'name': 'two_sibling_relocation_without_workspace_reads', 'passed': True})


def rejected(name, values, expected, paths=None):
    reader = Reader(values)
    if paths is not None: reader.paths = frozenset(paths)
    try:
        with patch.object(Path, 'open', side_effect=AssertionError('Negative callback attempted filesystem access')):
            verifier.verify(read_bytes=reader)
    except (ValueError, KeyError) as error:
        assert expected in str(error), (name, type(error).__name__, str(error))
        checks.append({'name': name, 'passed': True, 'rejection_class': type(error).__name__, 'rejection': str(error)})
    else:
        raise AssertionError('Changed evidence was admitted: ' + name)


modified = dict(payloads); modified['README.md'] = b'changed retained payload'
rejected('changed_own_payload', modified, 'Changed evidence')
modified = dict(payloads); modified.pop('preparation/evidence-index.json')
rejected('missing_authenticated_preparation', modified, 'preparation/evidence-index.json')
rejected('incomplete_declared_dependency_paths', payloads, 'Reader dependency closure differs',
         paths=set(payloads) - {'preparation/README.md'})
bad = copy.deepcopy(index); raw = b'not a valid shadow parent'
bad['files']['preparation/shadow.txt'] = {'kind': 'file', 'size': len(raw), 'sha256': sha(raw)}
modified = dict(payloads); modified['preparation/shadow.txt'] = raw; modified['evidence-index.json'] = dump(bad)
rejected('reserved_preparation_namespace_shadow', modified, 'Reserved preparation namespace')
bad = copy.deepcopy(index); summary = json.loads(payloads['qualification-summary.json'])
summary['six_lane_all_cases_agree'] = True
raw = dump(summary); bad['files']['qualification-summary.json'].update(size=len(raw), sha256=sha(raw))
modified = dict(payloads); modified['qualification-summary.json'] = raw; modified['evidence-index.json'] = dump(bad)
rejected('forged_complete_agreement_summary', modified, 'Saved terminal summary differs')

old = verifier.Packet(lambda name: (PREDECESSOR / name).read_bytes())
current = verifier.Packet(lambda name: payloads[name])
assert old.load('qualification-summary.json') == current.load('qualification-summary.json')
assert old.load('postclosure-bindings.json') == current.load('postclosure-bindings.json')
assert old.load('retention.json')['closed_files'] == current.load('retention.json')['closed_files']
assert old.read('verify_evidence.py') == current.read('verify_evidence.py')
for name in old.names:
    if name.startswith(('closed/', 'journals/', 'policy/')):
        assert old.read(name) == current.read(name), name
assert 'qualification-summary.json' in index['files'] and 'retention.json' in index['files']
checks.append({'name': 'serialization_successor_preserves_all_scientific_records_and_summary', 'passed': True})

after = {n: sha((PARENT / n[len('preparation/'):]).read_bytes() if n.startswith('preparation/')
                else (PACKET / n).read_bytes()) for n in payloads}
assert before == after
result = {'passed': True, 'checks': checks, 'packet_index_sha256': sha(index_raw),
          'parent_index_sha256': sha(parent_index_raw), 'predecessor_index_sha256': sha(old.index_raw),
          'verifier_sha256': sha(payloads['verify_evidence.py']),
          'summary_sha256': sha(payloads['qualification-summary.json']),
          'payloads_in_two_capsule_callback': len(payloads), 'packet_and_parent_unchanged': True,
          'CLI_compiler_loader_amplitude_invocations': 0, 'subprocess_invocations': 0,
          'elapsed_seconds': time.monotonic() - start, 'state_counts': original['summary']['state_counts'],
          'host_checked_complete_streams': original['summary']['host_checked_complete_streams'],
          'six_lane_all_cases_agree': original['summary']['six_lane_all_cases_agree']}
with (HERE / 'result.json').open('x') as stream:
    json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False); stream.write('\n')
print(json.dumps(result, sort_keys=True, indent=2))
