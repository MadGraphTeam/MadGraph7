"""Pure-Python relocation and negative tests; never invoke external commands."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import shutil
import tempfile
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKET = HERE.with_name('physical-stress-current-portable-4-preflight-1')
spec = importlib.util.spec_from_file_location('stress_preflight_portable_check', PACKET / 'verify_evidence.py')
verifier = importlib.util.module_from_spec(spec); spec.loader.exec_module(verifier)
def sha(raw): return hashlib.sha256(raw).hexdigest()

index_raw = (PACKET / 'evidence-index.json').read_bytes()
index = json.loads(index_raw)
before = {name: sha((PACKET / name).read_bytes()) for name in index['files']}
original = verifier.verify(PACKET)
with tempfile.TemporaryDirectory(prefix='stress-preflight-portable-') as temporary:
    relocated = Path(temporary) / 'packet'
    shutil.copytree(PACKET, relocated)
    old_open = Path.open
    def confined_open(path, *args, **kwargs):
        if not Path(path).absolute().is_relative_to(relocated):
            raise AssertionError('Verifier attempted nonportable file read: ' + str(path))
        return old_open(path, *args, **kwargs)
    with patch.object(Path, 'open', confined_open):
        result = verifier.verify(relocated)
    assert result == original
checks = [{'name': 'relocated_without_workspace_reads', 'passed': True}]

def rejected(name, modified_index, payloads, expected):
    def reader(key):
        if key == 'evidence-index.json': return json.dumps(modified_index, sort_keys=True).encode()
        return payloads[key] if key in payloads else (PACKET / key).read_bytes()
    try:
        verifier.verify(read_bytes=reader)
    except ValueError as error:
        assert expected in str(error), (name, str(error))
        checks.append({'name': name, 'passed': True, 'rejection': str(error)})
    else:
        raise AssertionError('Changed evidence admitted: ' + name)

bad = copy.deepcopy(index)
bad['files'] = {'../outside': {'kind': 'file', 'size': 0, 'sha256': sha(b'')}}
rejected('unsafe_index_path', bad, {}, 'Nonportable evidence name')
rejected('changed_payload_bytes', index, {'README.md': b'altered retained evidence'}, 'Changed evidence')
bad = copy.deepcopy(index)
logical = next(iter(bad['compressed_text']))
bad['compressed_text'][logical]['raw_bytes'] += 1
rejected('changed_gzip_raw_binding', bad, {}, 'Compressed text binding differs')
bad = copy.deepcopy(index)
summary = json.loads((PACKET / 'qualification-summary.json').read_bytes())
summary['physical_stress_qualified'] = True
raw = (json.dumps(summary, sort_keys=True, indent=2) + '\n').encode()
bad['files']['qualification-summary.json'].update(size=len(raw), sha256=sha(raw))
rejected('forged_preflight_numerical_success', bad, {'qualification-summary.json': raw}, 'Saved summary differs')
assert (PACKET / 'evidence-index.json').read_bytes() == index_raw
assert before == {name: sha((PACKET / name).read_bytes()) for name in index['files']}
result = {'passed': True, 'checks': checks, 'packet_index_sha256': sha(index_raw),
          'verifier_sha256': sha((PACKET / 'verify_evidence.py').read_bytes()),
          'capture_code_sha256': sha((PACKET / 'capture.py').read_bytes()),
          'packet_payloads': len(before), 'packet_unchanged': True,
          'CLI_compiler_loader_amplitude_invocations': 0, 'subprocess_invocations': 0,
          'summary': original['summary']}
(HERE / 'result.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n')
print(json.dumps({name: value for name, value in result.items() if name != 'summary'}, indent=2, sort_keys=True))
