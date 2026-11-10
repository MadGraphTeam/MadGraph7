#!/usr/bin/env python3
"""One-shot text snapshot of already-finished mass preflight/fixture evidence.

Does not run a controller, compiler, native executable, loader, or validator.
Selections are explicit. In particular continuation run outputs are not read.
"""
import difflib
import hashlib
import json
from pathlib import Path
import re
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = Path('.tirust-build/qp-mass-general-implementation/combined-qualification')
HEX = re.compile(r'[0-9a-f]{64}\Z')
PINNED = {
    str(BASE/'run.json'): 'a5d7541c314367999c11dd4335a15f86b9bcf7d44294849b10d6b25558c33b12',
    str(BASE/'loader-correction/run.json'): 'f741701d2980d6c6c0d9587cb992504e5356f6a66ee147e34d62df74f4e2c2b6',
    str(BASE/'loader-correction/fixture-offline-validation.json'): '35a0584f529f641804a345c171f9232a6ef4d04291154ff9c40634fcbb187b28',
}
TEXT_SUFFIXES = {'.py', '.json', '.md', '.f', '.f90', '.inc', '.h', '.dat',
                 '.patch', '.diff', '.txt', '.log', '.hex', '.sh', '.toml', '.lock'}


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    require(not (HERE/'manifest.json').exists(), 'Snapshot is append-only; already prepared')
    text_files = {}
    references = {}
    bindings = []

    def add(path, expected=None, reason='explicit snapshot', hash_only=False):
        path = Path(path)
        require('..' not in path.parts, 'Unsafe source path')
        if path.is_absolute():
            try:
                path = path.relative_to(ROOT)
            except ValueError:
                require(expected and HEX.fullmatch(expected), 'External identity needs a frozen hash')
                references[str(path)] = expected
                bindings.append(dict(path=str(path), sha256=expected, reason=reason, disposition='hash-only'))
                return
        key = str(path)
        require('/continuation/' not in key or path.name in {
            'README.md', 'control.py', 'controller.diff', 'prepared.json', 'source-index.json'},
            'Future continuation output forbidden: '+key)
        require(not key.endswith('current-original-qp/run.json'), 'Future additive output forbidden')
        if hash_only or path.suffix not in TEXT_SUFFIXES:
            require(expected and HEX.fullmatch(expected), 'Native identity needs a frozen hash: '+key)
            if key in references:
                require(references[key] == expected, 'Conflicting native identity: '+key)
            references[key] = expected
            bindings.append(dict(path=key, sha256=expected, reason=reason, disposition='hash-only'))
            return
        data = (ROOT/path).read_bytes()
        actual = digest(data)
        require(expected is None or actual == expected, 'Changed frozen text: '+key)
        require(b'\0' not in data, 'Binary payload rejected: '+key)
        try:
            data.decode('utf-8')
        except UnicodeDecodeError:
            # A small number of legacy source comments are Latin-1. Their raw
            # bytes remain exact and are explicitly recorded as source text.
            data.decode('latin-1')
        if key in text_files:
            require(text_files[key]['sha256'] == actual, 'Conflicting text snapshot: '+key)
        else:
            out = HERE/'payload'/path
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                require(out.read_bytes() == data, 'Changed partial snapshot: '+key)
            else:
                with out.open('xb') as stream:
                    stream.write(data)
            text_files[key] = dict(sha256=actual, bytes=len(data))
        bindings.append(dict(path=key, sha256=actual, reason=reason, disposition='text-payload'))

    def read_json(path):
        add(path, PINNED.get(str(path)))
        return json.loads((HERE/'payload'/path).read_text())

    def root_hashes(mapping, reason, hash_only=False):
        require(isinstance(mapping, dict), 'Expected hash map: '+reason)
        for path, value in mapping.items():
            if isinstance(value, str) and HEX.fullmatch(value):
                add(path, value, reason, hash_only=hash_only)

    # Original source/binding and completed build records, not future output.
    original = {}
    for name in ['source-manifest.json', 'evidence-index.json', 'prepared.json',
                 'build.json', 'build-evidence/index.json', 'source-tests.json',
                 'fixture/index.json', 'fixture/commands.json', 'run.json',
                 'current-original-qp/prepared.json', 'current-original-qp/evidence-index.json',
                 'current-original-qp/build.json', 'current-original-qp/build-evidence/index.json']:
        original[name] = read_json(BASE/name)
    for name in ['evidence-index.json', 'current-original-qp/evidence-index.json']:
        item = original[name]
        root_hashes(item['source_files'], name+' source closure')
        root_hashes(item.get('native_artifacts_hash_only', {}), name+' native identities', True)
    for name in ['prepared.json', 'fixture/commands.json', 'current-original-qp/prepared.json']:
        item = original[name]
        root_hashes(item['inputs'], name+' bound inputs')
        root_hashes(item.get('system_files', {}), name+' system identities', True)
    for name in ['build-evidence/index.json', 'current-original-qp/build-evidence/index.json']:
        item = original[name]
        root_hashes(item['files'], name+' completed build logs')
        root_hashes(item['native_artifacts_hash_only'], name+' native outputs', True)
    for name in ['build.json', 'current-original-qp/build.json']:
        root_hashes(original[name].get('outputs', {}), name+' native outputs', True)
        root_hashes(original[name].get('module_outputs', {}), name+' module outputs', True)
    # Include exact parent implementation source and indexes used by the merged
    # writer. These are provenance, not sources executed by the offline verifier.
    for name, item in original['source-manifest.json']['sources'].items():
        add(item['source'], item['sha256'], 'implementation parent source')
    for label, value in original['source-manifest.json']['parent_indexes'].items():
        add(BASE.parent/label/'evidence-index.json', value, 'implementation parent index')
    for name in ['implementation.patch', 'README.md']:
        add(BASE/name)

    corrections = {}
    for sub in ['loader-correction', 'loader-correction/additive',
                'loader-correction/continuation', 'loader-correction/continuation/additive']:
        index = read_json(BASE/sub/'source-index.json')
        for name, value in index['files'].items():
            add(BASE/sub/name, value, sub+' frozen source')
        prepared = json.loads((HERE/'payload'/BASE/sub/'prepared.json').read_text())
        root_hashes(prepared['inputs'], sub+' bound inputs')
        corrections[sub] = prepared
    corrected = read_json(BASE/'loader-correction/run.json')
    read_json(BASE/'loader-correction/fixture-offline-validation.json')
    # Finished jobs only. Their per-event hashes define the exact journals/logs.
    for label, record in [('initial', original['run.json']), ('loader-corrected', corrected)]:
        for event in record['jobs']:
            add(event['log'], event['log_sha256'], label+' job '+event['name'])
            for name, value in event.get('output_files', {}).items():
                add(Path(event['cwd'])/name, value, label+' output '+event['name'])
    # Preserve mechanically changed source as both full originals and readable
    # unified differences. The first correction is an importing wrapper, so
    # this diff is a source comparison, not an instruction to replace production.
    pairs = [('loader-correction.diff', 'control.py', 'loader-correction/control.py'),
             ('additive-loader-correction.diff', 'current-original-qp/control.py',
              'loader-correction/additive/control.py')]
    for outname, left, right in pairs:
        before = (HERE/'payload'/BASE/left).read_text().splitlines(keepends=True)
        after = (HERE/'payload'/BASE/right).read_text().splitlines(keepends=True)
        target = HERE/'derived'/outname
        target.parent.mkdir(exist_ok=True)
        target.write_text(''.join(difflib.unified_diff(before, after, fromfile=left, tofile=right)))
    manifest = dict(schema='qp-mass-preflight-fixture-portable-v1',
                    repository_root_original=str(ROOT), payload_root='payload',
                    base=str(BASE), frozen_records=PINNED,
                    text_files=dict(sorted(text_files.items())),
                    native_and_system_hash_only=dict(sorted(references.items())),
                    bindings=bindings, newly_executed_native=False,
                    forbidden_outputs=['loader-correction/continuation/run.json',
                                       'loader-correction/continuation/additive/run.json',
                                       'current-original-qp/run.json'],
                    qualification='Caller recording-stub fixture only; no amplitude worker executed in captured runs',
                    continuation_status='Source snapshot only; no continuation output read')
    (HERE/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(text_files=len(text_files), text_bytes=sum(x['bytes'] for x in text_files.values()),
                          hash_only=len(references), manifest_sha256=digest((HERE/'manifest.json').read_bytes()))))


if __name__ == '__main__':
    main()
