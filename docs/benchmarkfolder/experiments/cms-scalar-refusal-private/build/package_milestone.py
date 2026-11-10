#!/usr/bin/env python3
"""Copy retained text evidence; native artifacts are represented only by hashes."""
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TARGET = HERE / 'source-build-evidence'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert Path.cwd().resolve() == ROOT
    assert not TARGET.exists()
    original = json.loads((HERE / 'source-index.json').read_text())
    build = json.loads((HERE / 'build.json').read_text())
    assert build['state'] == 'complete' and len(build['jobs']) == 17
    assert build['input_hashes_after'] == 'unchanged'
    TARGET.mkdir()
    native = {}
    retained = {}

    def copy(source, name, digest):
        assert sha(source) == digest, source
        destination = TARGET / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        retained[name] = digest

    for name, digest in original['package_files'].items():
        source = HERE / name
        assert sha(source) == digest, source
        if source.suffix.lower() in ('.mod', '.smod', '.o', '.a', '.so'):
            native[str(source.relative_to(ROOT))] = digest
        else:
            copy(source, 'source/' + name, digest)
    copy(HERE / 'source-index.json', 'source-index.json', sha(HERE / 'source-index.json'))
    copy(HERE / 'build.json', 'build.json', sha(HERE / 'build.json'))
    copy(Path(__file__), 'package_milestone.py', sha(__file__))
    for event in build['jobs']:
        copy(ROOT / event['log'], 'logs/' + Path(event['log']).name, event['log_sha256'])
    for name, digest in {**build['outputs'], **build['module_outputs']}.items():
        assert sha(ROOT / name) == digest, name
        native[name] = digest
    verifier = '''#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
here=Path(__file__).resolve().parent
manifest=json.loads((here/'manifest.json').read_text())
for name,digest in manifest['retained_files'].items():
 p=here/name
 if not p.resolve().is_relative_to(here) or p.is_symlink(): raise ValueError('Nonportable evidence path: '+name)
 if hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise ValueError('Changed retained file: '+name)
print(json.dumps({'retained_files_verified':len(manifest['retained_files']),'native_artifacts_hash_only':len(manifest['native_artifacts']),'compiler_or_native_launched':False}))
'''
    (TARGET / 'verify.py').write_text(verifier)
    retained['verify.py'] = sha(TARGET / 'verify.py')
    readme = '''# CMS source/build milestone

All 17 authorized compiler/linker commands passed on CPU 8 in 17.047920 seconds
(15.420376 seconds in compiler children). All protected inputs were unchanged.
The actual AVH and checked CutTools modules compiled; only recording mocks were
linked. No loader, mock, analytic scalar or amplitude executed in this milestone.

The retained source package includes 26 passing source-only checks and preserves
earlier source-test drafts/failures. Native modules, objects and executables are
hash-only entries in manifest.json; none is bundled. Runtime and accuracy remain
unqualified. Run `python -B verify.py` from any directory to verify this retained
evidence package without needing the original build tree or launching native code.
The source recipes preserve their original repository-relative/absolute identities;
this verifier checks evidence, not a relocated compiler installation.
'''
    (TARGET / 'README.md').write_text(readme)
    retained['README.md'] = sha(TARGET / 'README.md')
    manifest = {'schema': 'cms-scalar-refusal-portable-source-build-v1',
                'source_index_sha256': sha(HERE / 'source-index.json'),
                'build_sha256': sha(HERE / 'build.json'),
                'retained_files': dict(sorted(retained.items())),
                'native_artifacts': dict(sorted(native.items())),
                'compiler_commands_passed': 17, 'compiler_wall_seconds': build['finished_unix']-build['started_unix'],
                'native_execution_performed': False, 'physical_accuracy_qualified': False}
    (TARGET / 'manifest.json').write_text(json.dumps(manifest, sort_keys=True, indent=2)+'\n')
    print(json.dumps({'retained_files': len(retained), 'native_artifacts_hash_only': len(native),
                      'manifest_sha256': sha(TARGET / 'manifest.json')}))


if __name__ == '__main__':
    main()
