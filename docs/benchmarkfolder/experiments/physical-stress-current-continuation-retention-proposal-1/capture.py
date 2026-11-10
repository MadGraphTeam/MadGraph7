"""Capture the closed stress suffix after an actual root terminal observation.

The default invocation reads no report or worker inputs.  Execution is pure
filesystem retention: no compiler, loader probe, worker, or amplitude is run.
"""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
REPORT = HERE.with_name('physical-stress-current-qualification-5')
WORK = ROOT / '.tirust-build/physical-stress-current-5'
SOURCE = HERE.with_name('physical-stress-current-supervisor-5')
ANCESTOR = HERE.with_name('physical-stress-current-portable-4-terminal-2')
PREPARATION = HERE.with_name('physical-stress-current-portable-4-preflight-1')
SOURCE5_INDEX = '5eded3d276bea030e9b8cce598f6422f10a789dc1eaa9ff6588275c2e594b931'
ANCESTOR_INDEX = '6fc2035649ba0387f0d4457784ae883592f13bb7e3b6f9c33b99e038b81f7146'
ANCESTOR_VERIFIER = '20d1666c837ca09632121181b1a6a95b9d6afd677cb897598be7de80cb79a3e8'
PREPARATION_INDEX = '49cb1ec298bbd2f8b942b02c3e81b60b53cd2f14a5cdaa56d0fa93452c175c22'
PREPARATION_VERIFIER = '3850e1dcfbd3cecfc2c11eabd090e5ae174b0d0fbad6da11b6d94557c5e72f24'
INHERITED_CHECKPOINT = '0c4157c988bdd0371e9c0cfba54e784b3f7a730cf7a631b8aca5c27814636d5a'
SESSION = 7248


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            result.update(block)
    return result.hexdigest()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def read_json(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_bytes(), parse_constant=reject)


def safe(name):
    require(isinstance(name, str), 'Evidence path must be a string')
    path = PurePosixPath(name)
    require(name and '\\' not in name and not path.is_absolute()
            and '..' not in path.parts and str(path) == name,
            'Nonportable evidence path: ' + name)
    return name


def text(raw):
    require(b'\0' not in raw, 'Binary payload is not retained')
    raw.decode('utf-8')
    return raw


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def indexed_source(directory, expected_index=None):
    """Bind a complete, regular, UTF-8 source tree without running it."""
    raw = text((directory / 'source-index.json').read_bytes())
    require(expected_index is None or sha(raw) == expected_index, 'Source index differs')
    index = json.loads(raw)
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob('*') if p.is_file()}
    require(actual == set(index['files']) | {'source-index.json'}, 'Source inventory differs')
    require(not any(p.is_symlink() or not (p.is_file() or p.is_dir())
                    for p in directory.rglob('*')), 'Source has a symlink or special file')
    files = {'source-index.json': raw}
    for name, item in index['files'].items():
        payload = text((directory / safe(name)).read_bytes())
        require(item['kind'] == 'file' and sha(payload) == item['sha256']
                and len(payload) == item['size'], 'Changed source: ' + name)
        files[name] = payload
    return sha(raw), files


def validate_confirmation(confirmation):
    require(set(confirmation) == {'kind', 'session_id', 'exit_code', 'observed_utc',
                                  'source_index_sha256', 'inherited_checkpoint_sha256'},
            'Terminal confirmation fields differ')
    require(confirmation['kind'] == 'root-reported-tool-terminal-observation'
            and type(confirmation['session_id']) is int and confirmation['session_id'] == SESSION
            and type(confirmation['exit_code']) is int and confirmation['exit_code'] in (0, 1)
            and confirmation['source_index_sha256'] == SOURCE5_INDEX
            and confirmation['inherited_checkpoint_sha256'] == INHERITED_CHECKPOINT,
            'Actual terminal observation binding differs')
    stamp = confirmation['observed_utc']
    require(isinstance(stamp, str) and re.fullmatch(r'\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(?:\.\d+)?Z', stamp),
            'Supply the actual root observation in UTC; capture invents no timestamp')
    return confirmation


def terminal_gate(report, confirmation):
    """This is the first operation permitted to read report5."""
    stage = read_json(report / 'stage-run.json')
    require(stage['stage'] == 'run' and stage['status'] ==
            ('passed' if confirmation['exit_code'] == 0 else 'failed'),
            'Native stage is not closed with the observed exit status')
    require(stage['source_index_sha256'] == SOURCE5_INDEX
            and stage['inherited_checkpoint_sha256'] == INHERITED_CHECKPOINT
            and stage['native_execution'] is True and stage['performance_eligible'] is False,
            'Closed native stage binding differs')
    require((stage['exception'] is None) == (confirmation['exit_code'] == 0),
            'Native stage exception contradicts its observed exit status')
    require(file_sha(report / 'inherited-checkpoint.json') == INHERITED_CHECKPOINT,
            'Inherited checkpoint differs')
    inherited = read_json(report / 'stage-inherit.json')
    require(inherited['stage'] == 'inherit' and inherited['status'] == 'passed'
            and inherited['source_index_sha256'] == SOURCE5_INDEX
            and inherited['inherited_checkpoint_sha256'] == INHERITED_CHECKPOINT,
            'Inherited stage binding differs')
    return stage


def report_snapshot(report):
    require(report.is_dir() and not report.is_symlink(), 'Closed report is not a regular directory')
    files, directories = {}, ['.']
    for path in sorted(report.rglob('*')):
        require(not path.is_symlink(), 'Symlink in closed report: ' + str(path))
        name = path.relative_to(report).as_posix()
        if path.is_dir():
            directories.append(name)
        else:
            require(path.is_file(), 'Special file in closed report: ' + name)
            raw = text(path.read_bytes())
            files[name] = {'sha256': sha(raw), 'size': len(raw)}
    return files, sorted(directories)


def file_binding(path, errors, label, *, private_root=None):
    try:
        if private_root is not None:
            require(path.resolve(strict=True).is_relative_to(private_root.resolve(strict=True)),
                    'Unsafe input path escapes private worker')
        require(path.is_file(), 'Missing or nonregular file')
        return file_sha(path)
    except Exception as error:
        errors.append(label + ': ' + repr(error))
        return None


def mapped_binding(spelling):
    """A changed, deleted, or unexpected mapping is retained as evidence."""
    actual = Path(spelling.removesuffix(' (deleted)'))
    record = {'resolved': None, 'sha256': None, 'size': None, 'inode': None,
              'device': None, 'header64_hex': None}
    try:
        record['resolved'] = str(actual.resolve(strict=True))
        stat = actual.stat()
        record.update(size=stat.st_size, inode=stat.st_ino,
                      device=[os.major(stat.st_dev), os.minor(stat.st_dev)])
        require(actual.is_file(), 'Mapping no longer names a regular file')
        with actual.open('rb') as stream:
            record['header64_hex'] = stream.read(64).hex()
        record['sha256'] = file_sha(actual)
        require(not spelling.endswith(' (deleted)'), 'Mapped spelling is marked deleted')
    except Exception as error:
        record['error'] = repr(error)
    return record


def postclosure_bindings(report, work, append):
    """Inspect every worker, including copies that failed before specification."""
    workers, mapped, all_errors = {}, {}, []
    root = work / 'workers'
    if root.is_symlink():
        return {'scope': 'Read-only postclosure identities; no new numerical observation',
                'mapped_files': {}, 'workers': {}, 'worker_root': str(root),
                'worker_directories': [], 'binding_errors': ['Worker root is an unexpected symlink']}
    outputs = sorted(root.glob('*/*')) if root.exists() else []
    worker_directories = []
    for output in outputs:
        rid = output.relative_to(root).as_posix()
        if output.is_symlink() or not output.is_dir() or not output.resolve().is_relative_to(root.resolve()):
            all_errors.append('Unexpected worker path: ' + rid)
            continue
        worker_directories.append(rid)
        errors = []
        evidence = report / 'runs' / rid
        specification = evidence / 'specification.json'
        spec = None
        if specification.is_file():
            try:
                spec = read_json(specification)
                require(isinstance(spec, dict) and isinstance(spec.get('cwd'), str),
                        'Specification is not an object with a string cwd')
                for section in ('scientific_sources', 'numerical_cards', 'expected_runtime'):
                    values = spec.get(section, {})
                    require(isinstance(values, dict) and all(isinstance(name, str)
                            and isinstance(checksum, str) and re.fullmatch('[0-9a-f]{64}', checksum)
                            for name, checksum in values.items()), 'Invalid specification section: ' + section)
                require(isinstance(spec.get('build', {}), dict), 'Invalid specification build binding')
            except Exception as error:
                spec = None
                errors.append('Unreadable specification: ' + repr(error))
        record = {'specification_present': specification.is_file(),
                  'specification_sha256': file_binding(specification, errors, 'specification')
                      if specification.is_file() else None,
                  'scientific_sources': {}, 'numerical_cards': {}, 'physical_card_sha256': None,
                  'executable_sha256': None, 'journals': {}, 'binding_errors': errors}
        workers[rid] = record
        if spec is None:
            errors.append('No readable specification; this worker cannot be qualified')
        for section in ('scientific_sources', 'numerical_cards'):
            expected = spec.get(section, {}) if spec is not None else {}
            # Partial workers have no expected map. Preserve the observable
            # source/card identities without fabricating an earlier binding.
            if not expected:
                if section == 'numerical_cards':
                    paths = [output / 'Cards/MadLoopParams.dat']
                    subroot = output / 'SubProcesses'
                    if subroot.resolve().is_relative_to(output.resolve()):
                        paths.extend(sorted(subroot.rglob('MadLoopParams.dat')))
                    else:
                        errors.append('Unsafe numerical card root escapes private worker')
                else:
                    paths = []
                    for sub in ('Source', 'SubProcesses'):
                        subroot = output / sub
                        if not subroot.resolve().is_relative_to(output.resolve()):
                            errors.append('Unsafe scientific source root escapes private worker: ' + sub)
                            continue
                        paths.extend(p for p in sorted(subroot.rglob('*'))
                                     if p.is_file() and (p.suffix.lower() in {'.f', '.f90', '.for', '.inc', '.c', '.cc', '.h'}
                                     or p.name.startswith('makefile') or p.name in {'make_opts', 'MadLoop_makefile_definitions'}))
                expected = {p.relative_to(output).as_posix(): None for p in paths}
            for name, checksum in expected.items():
                try:
                    path = output / safe(name)
                    actual = file_binding(path, errors, section + '/' + name, private_root=output)
                except Exception as error:
                    actual = None
                    errors.append(section + '/' + str(name) + ': ' + repr(error))
                record[section][name] = actual
                if checksum is not None and actual != checksum:
                    errors.append('Changed ' + section + ': ' + name)
        record['physical_card_sha256'] = file_binding(output / 'Cards/param_card.dat', errors, 'physical card', private_root=output)
        if spec is not None and record['physical_card_sha256'] != spec.get('physical_card_sha256'):
            errors.append('Changed physical parameter card')
        directories = []
        if spec is not None:
            directory = Path(spec.get('cwd', ''))
            try:
                directory.relative_to(output)
                require(directory.resolve().is_relative_to(output.resolve()), 'Process directory escapes worker')
                directories.append(directory)
            except Exception as error:
                errors.append('Invalid specification cwd: ' + repr(error))
        if not directories:
            directories = [p.parent for p in sorted((output / 'SubProcesses').glob('*/check')) if p.is_file()]
        record['process_directories'] = [str(p.relative_to(output)) for p in directories]
        if len(directories) == 1:
            record['executable_sha256'] = file_binding(directories[0] / 'check', errors, 'executable', private_root=output)
            expected = spec.get('build', {}).get('executable_sha256') if spec is not None else None
            if expected is not None and record['executable_sha256'] != expected:
                errors.append('Changed worker executable')
        else:
            errors.append('Expected exactly one private process directory')
        # Journals can exist on a partial worker before specification writing.
        # They are read through worker-private paths only; external symlinks are
        # recorded, never followed into unrelated trees.
        for rawroot in sorted(output.rglob('worker-raw')):
            if rawroot.is_symlink() or not rawroot.resolve().is_relative_to(output.resolve()):
                errors.append('Unsafe worker journal root: ' + str(rawroot))
                continue
            for journal in sorted(rawroot.rglob('*')):
                if journal.is_dir() and not journal.is_symlink():
                    continue
                name = journal.relative_to(output).as_posix()
                # Retain the prior capsule's cwd-relative journal names when
                # the unique process directory is known.
                key = journal.relative_to(directories[0]).as_posix() if len(directories) == 1 and journal.is_relative_to(directories[0]) else name
                try:
                    require(not journal.is_symlink() and journal.is_file(), 'Symlink or special journal')
                    raw = text(journal.read_bytes())
                    retained = 'journals/' + rid + '/' + name
                    append(retained, raw)
                    record['journals'][key] = {'retained': retained, 'sha256': sha(raw), 'size': len(raw)}
                except Exception as error:
                    errors.append('Unretained journal ' + name + ': ' + repr(error))
        mapfiles = sorted(evidence.glob('maps-*.txt'))
        if spec is not None and not mapfiles:
            errors.append('No retained mapped-runtime observations for specified worker')
        for mapfile in mapfiles:
            try:
                lines = mapfile.read_text().splitlines()
            except Exception as error:
                errors.append('Unreadable maps file: ' + repr(error))
                continue
            for lineno, line in enumerate(lines, 1):
                fields = line.split(maxsplit=5)
                if len(fields) != 6 or not fields[5].startswith('/'):
                    continue
                spelling = fields[5]
                if spelling not in mapped:
                    mapped[spelling] = mapped_binding(spelling)
                item = mapped[spelling]
                label = mapfile.name + ':' + str(lineno) + ': '
                if 'error' in item:
                    errors.append(label + 'Mapped file read failed: ' + spelling + ': ' + item['error'])
                try:
                    require(item['inode'] == int(fields[4]) and item['device'] == [int(v, 16) for v in fields[3].split(':')],
                            'Mapped inode/device changed')
                    if item['header64_hex'] and item['header64_hex'].startswith('7f454c46'):
                        require(spec is not None and spec.get('expected_runtime', {}).get(item['resolved']) == item['sha256'],
                                'Unexpected or changed mapped ELF')
                    else:
                        require(spelling.startswith('/nix/store/') and spelling.endswith('/lib/locale/locale-archive') and fields[1] == 'r--p',
                                'Unexpected mapped data')
                except Exception as error:
                    errors.append(label + spelling + ': ' + repr(error))
    return {'scope': 'Read-only postclosure identities; no new numerical observation',
            'mapped_files': mapped, 'workers': workers, 'worker_root': str(root),
            'worker_directories': worker_directories, 'binding_errors': all_errors}


class Capsule:
    def __init__(self, destination, policy_index):
        self.root = destination
        self.index = {'schema_version': 1, 'files': {}, 'compressed_text': {},
                      'source_index_sha256': SOURCE5_INDEX,
                      'source_policy_index_sha256': policy_index,
                      'dependencies': {
                          'ancestor': {'sibling': ANCESTOR.name, 'index_sha256': ANCESTOR_INDEX,
                                       'verifier_sha256': ANCESTOR_VERIFIER},
                          'preparation': {'sibling': PREPARATION.name, 'index_sha256': PREPARATION_INDEX,
                                          'verifier_sha256': PREPARATION_VERIFIER}}}
        destination.mkdir(exist_ok=False)

    def append(self, name, raw):
        safe(name)
        text(raw)
        require(name not in self.index['compressed_text'] and name not in self.index['files'], 'Duplicate evidence name')
        if len(raw) > 65536 and Path(name).suffix != '.py' and name not in {
                'retention.json', 'qualification-summary.json', 'root-terminal-observation.json'}:
            logical = name
            name = 'compressed/' + name + '.gz'
            self.index['compressed_text'][logical] = {'path': name, 'raw_bytes': len(raw), 'raw_sha256': sha(raw)}
            raw = gzip.compress(raw, compresslevel=6, mtime=0)
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(raw)
        self.index['files'][name] = {'kind': 'file', 'sha256': sha(raw), 'size': len(raw)}

    def save(self, name, value):
        self.append(name, json_bytes(value))

    def reader(self):
        index = self.index
        def read(name):
            safe(name)
            require(name in paths, 'Read outside indexed evidence namespace: ' + name)
            if name == 'evidence-index.json':
                return json_bytes(index)
            for prefix, root in (('ancestor/', ANCESTOR), ('preparation/', PREPARATION)):
                if name.startswith(prefix):
                    return (root / name[len(prefix):]).read_bytes()
            return (self.root / name).read_bytes()
        paths = set(index['files']) | {'evidence-index.json'}
        for prefix, root in (('ancestor/', ANCESTOR), ('preparation/', PREPARATION)):
            dependency = read_json(root / 'evidence-index.json')
            paths.update(prefix + name for name in dependency['files'])
            paths.add(prefix + 'evidence-index.json')
        read.paths = paths
        return read


def capture(confirmation, destination):
    require(Path.cwd() == ROOT, 'Run from repository root')
    validate_confirmation(confirmation)
    # A supplied confirmation is required before *any* report5 or work5 read.
    stage = terminal_gate(REPORT, confirmation)
    require(destination.parent.resolve() == HERE.parent.resolve() and not destination.exists()
            and not destination.is_symlink(), 'Use a fresh capsule sibling destination')
    _, source_files = indexed_source(SOURCE, SOURCE5_INDEX)
    policy_index, policy_files = indexed_source(HERE)
    require(file_sha(ANCESTOR / 'evidence-index.json') == ANCESTOR_INDEX
            and file_sha(ANCESTOR / 'verify_evidence.py') == ANCESTOR_VERIFIER
            and file_sha(PREPARATION / 'evidence-index.json') == PREPARATION_INDEX
            and file_sha(PREPARATION / 'verify_evidence.py') == PREPARATION_VERIFIER,
            'Closed dependency identity differs')
    ancestor = load_module('stress_suffix_capture_ancestor', ANCESTOR / 'verify_evidence.py')
    require(ancestor.verify(ANCESTOR)['verified'] is True, 'Ancestor verification failed')
    verifier = load_module('stress_suffix_capture_verifier', HERE / 'verify_evidence.py')
    require(verifier.SOURCE5_INDEX == SOURCE5_INDEX, 'Capture/verifier source binding differs')
    original, directories = report_snapshot(REPORT)
    source_guard = {'ok': False}
    try:
        supervisor = load_module('stress_suffix_capture_supervisor', SOURCE / 'run.py')
        result = supervisor.verify_inherited_inputs()
        source_guard = {'ok': True, 'result': result}
    except Exception as error:
        source_guard['error'] = repr(error)
    capsule = Capsule(destination, policy_index)
    for name in ('capture.py', 'verify_evidence.py'):
        require(name in policy_files, 'Missing retained policy source: ' + name)
        capsule.append(name, policy_files[name])
    for prefix, files in (('source/', source_files), ('policy/', policy_files)):
        for name, raw in files.items():
            capsule.append(prefix + name, raw)
    for name, item in original.items():
        raw = text((REPORT / name).read_bytes())
        require(sha(raw) == item['sha256'] and len(raw) == item['size'], 'Closed report changed during capture')
        capsule.append('closed/' + name, raw)
    post = postclosure_bindings(REPORT, WORK, capsule.append)
    capsule.save('postclosure-bindings.json', post)
    capsule.save('root-terminal-observation.json', confirmation)
    capsule.save('retention.json', {
        'ancestor_index_sha256': ANCESTOR_INDEX, 'preparation_index_sha256': PREPARATION_INDEX,
        'source_index_sha256': SOURCE5_INDEX, 'source_policy_index_sha256': policy_index,
        'inherited_checkpoint_sha256': INHERITED_CHECKPOINT,
        'capture_sha256': sha(policy_files['capture.py']), 'original_closed_root': str(REPORT),
        'closed_files': {name: {**item, 'storage': 'own'} for name, item in original.items()},
        'closed_directories': directories, 'original_files_unchanged': True,
        'source_guard': source_guard, 'capture_native_execution': False})
    require(report_snapshot(REPORT) == (original, directories), 'Closed report changed during capture')
    require(read_json(REPORT / 'stage-run.json') == stage, 'Native terminal marker changed')
    proof = verifier.verify(read_bytes=capsule.reader(), derive_summary=True)
    capsule.save('qualification-summary.json', proof['summary'])
    verifier.verify(read_bytes=capsule.reader())
    with (destination / 'evidence-index.json').open('xb') as stream:
        stream.write(json_bytes(capsule.index))
    proof = verifier.verify(destination)
    summary = proof['summary']
    print(json.dumps({'verified': proof['verified'], 'index_sha256': proof['index_sha256'],
                      'ancestor_index_sha256': ANCESTOR_INDEX,
                      'preparation_index_sha256': PREPARATION_INDEX,
                      'summary_keys': sorted(summary),
                      'state_counts': summary.get('state_counts'),
                      'case_coverage': summary.get('case_coverage')}, indent=2, sort_keys=True))
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--confirmation', type=Path)
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({'capture': False, 'native_execution': False, 'report_inputs_read': False,
                          'required_terminal_session': SESSION, 'source_index_sha256': SOURCE5_INDEX}))
        return
    require(args.confirmation is not None and args.destination is not None,
            'Actual terminal --confirmation and fresh --destination are required')
    capture(read_json(args.confirmation), args.destination.absolute())


if __name__ == '__main__':
    main()
