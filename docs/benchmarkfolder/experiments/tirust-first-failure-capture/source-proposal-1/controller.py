#!/usr/bin/env python3
"""Private TIRust first-failure capture/replay. Planning is inert.

No campaign export or production source is edited. Native stages need an
explicit --execute and the independently reviewed source inventory.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
WORK = ROOT / '.tirust-build/tirust-first-failure-capture-1'
REPORT = WORK / 'evidence'
PILOT = ROOT / 'docs/benchmarkfolder/stability_appendix_c/source5'
PILOT_SHA = '22b06cda2303d9d2fd42aed418aa4518db0d9e834678862330ffa0d6600e9eef'
PILOT_INDEX_SHA = 'ffbc57948b062e6380f71e22c9f303b0e652bcd4060952b7bc0a4fe216191271'
CC = '/nix/store/788mx070y81zjlg5ipcl0cra3afviw9k-gcc-wrapper-15.2.0/bin/gcc'
CC_SHA = '3c48a46fb1dc134c0f10c2da6ec1a8d34a381b93c2ac588b7d19dd3e804eeeef'
SPECS = (
    {'case': 'qcd_n3_uux_zgg', 'case_id': 1, 'target_point': 25, 'groups': [59, 61], 'nprop': 5, 'rank': 4},
    {'case': 'qcd_n4_uux_zzgg', 'case_id': 2, 'target_point': 2, 'groups': [408, 422], 'nprop': 6, 'rank': 5},
)
DIAGNOSTICS = ('stability-values.dat', 'stability-dimensions.dat', 'stability-groups.dat')
MAKEFILE = '''include makefile
.PHONY: capture-build
CAPTURE_PROCESS = $(filter-out TIRUST_interface.o TIR_interface.o,$(PROCESS))
capture-build:
\t$(FC) $(FFLAGS) -c observer.f90 -o capture-observer-f.o $(LOOP_INCLUDE)
\t$(CC) -std=c11 -O2 -fPIC -I. -I$(CAPTURE_ROOT) -c observer.c -o capture-observer-c.o
\t$(FC) $(FFLAGS) -c TIRUST_interface.f -o TIRUST_interface.o $(LOOP_INCLUDE)
\t$(FC) $(FFLAGS) -c TIR_interface.f -o TIR_interface.o $(LOOP_INCLUDE)
\t$(FC) $(FFLAGS) -c check_sa.f -o check_sa.o $(LOOP_INCLUDE)
\t$(FC) $(FFLAGS) -o stability-check check_sa.o TIRUST_interface.o TIR_interface.o capture-observer-f.o capture-observer-c.o $(CAPTURE_PROCESS) $(LINKLIBS) -Wl,--wrap=ml_tirust_evaluate_dp,-Map,capture-link.map,--cref
\t$(CC) -std=c11 -O2 -fPIC -I. -I$(CAPTURE_ROOT) -c replay.c -o capture-replay.o
\t$(FC) $(FFLAGS) -o capture-replay/stability-check capture-replay.o $(LINKLIBS)
'''


def require(value, message):
    if not value: raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def module_from_bytes(path, checksum):
    path = Path(path); blob = path.read_bytes()
    require(hashlib.sha256(blob).hexdigest() == checksum, 'Pinned Python source differs: ' + str(path))
    module = types.ModuleType('private_capture_' + path.stem); module.__file__ = str(path)
    exec(compile(blob, str(path), 'exec'), module.__dict__)
    return module


def elf_symbols(blob):
    """Read only ELF64 LE symbol definitions and relocation symbol names."""
    require(len(blob) >= 64 and blob[:6] == b'\x7fELF\x02\x01', 'Expected ELF64 LE object')
    offset = struct.unpack_from('<Q', blob, 40)[0]
    size, count = struct.unpack_from('<HH', blob, 58)
    require(size == 64 and 0 < count < 65535 and offset + count * size <= len(blob), 'Bad ELF section table')
    sections = [struct.unpack_from('<IIQQQQIIQQ', blob, offset + i * size) for i in range(count)]
    tables = {}; symbols = []
    for i, section in enumerate(sections):
        if section[1] not in (2, 11): continue
        start, length, link, entry = section[4], section[5], section[6], section[9]
        require(entry == 24 and length % entry == 0 and start + length <= len(blob) and link < count,
                'Bad ELF symbol table')
        strings = sections[link]; names = blob[strings[4]:strings[4] + strings[5]]; table = []
        for pos in range(start, start + length, entry):
            name, info, other, ndx, value, extent = struct.unpack_from('<IBBHQQ', blob, pos)
            require(name < len(names), 'ELF symbol name outside string table')
            stop = names.find(b'\0', name); require(stop >= name, 'Unterminated ELF symbol name')
            row = {'name': names[name:stop].decode('ascii'), 'undefined': ndx == 0}
            table.append(row); symbols.append(row)
        tables[i] = table
    relocations = []
    for section in sections:
        if section[1] not in (4, 9): continue
        start, length, link, entry = section[4], section[5], section[6], section[9]
        require(link in tables and entry in (16, 24) and length % entry == 0
                and start + length <= len(blob), 'Bad ELF relocation section')
        for pos in range(start, start + length, entry):
            address, info = struct.unpack_from('<QQ', blob, pos); symbol = info >> 32
            require(symbol < len(tables[link]), 'ELF relocation symbol outside table')
            relocations.append({'symbol': tables[link][symbol]['name'], 'offset': address, 'type': info & 0xffffffff})
    return symbols, relocations


def archive_member(blob, wanted):
    require(blob.startswith(b'!<arch>\n'), 'Expected static ar archive')
    offset = 8; found = []
    while offset < len(blob):
        header = blob[offset:offset + 60]
        require(len(header) == 60 and header[58:] == b'`\n', 'Bad ar member header')
        size = int(header[48:58]); start = offset + 60
        require(start + size <= len(blob), 'Truncated ar member')
        name = header[:16].decode('ascii').strip().rstrip('/')
        if name == wanted: found.append(blob[start:start + size])
        offset = start + size + size % 2
    require(len(found) == 1, 'Static archive does not have exactly one ' + wanted)
    return found[0]


def static_abi(archive):
    blob = Path(archive).read_bytes(); member = archive_member(blob, 'mtirust.o')
    symbols, relocations = elf_symbols(member); target = 'ml_tirust_evaluate_dp'
    require(any(row['name'] == target and row['undefined'] for row in symbols), 'mtirust.o lacks the unresolved DP import')
    refs = [row for row in relocations if row['symbol'] == target]
    require(refs, 'mtirust.o has no relocation to the DP import')
    return {'archive': str(archive), 'archive_sha256': hashlib.sha256(blob).hexdigest(),
            'member': 'mtirust.o', 'member_sha256': hashlib.sha256(member).hexdigest(), 'dp_relocations': refs}


def linked_abi(executable, linkmap):
    symbols, unused = elf_symbols(Path(executable).read_bytes())
    require(any(row['name'] == '__wrap_ml_tirust_evaluate_dp' and not row['undefined'] for row in symbols)
            and any(row['name'] == 'ml_tirust_evaluate_dp' and row['undefined'] for row in symbols),
            'Linked wrapper definition or real DP import is absent')
    text = Path(linkmap).read_text()
    match = re.search(r'^__wrap_ml_tirust_evaluate_dp\s+([^\n]+)\n((?:[ \t]+[^\n]*\n)*)', text, re.M)
    require(match is not None and 'capture-observer-c.o' in match[1] and 'libtirust.a(mtirust.o)' in match[2],
            'Link map does not bind the mtirust DP import to the observer wrapper')
    return {'verified': True, 'linkmap_sha256': digest(linkmap), 'wrapper_cross_reference': match[0]}


def seal():
    index = read(HERE / 'source-index.json')
    require({'controller.py', 'test_controller.py', 'instrument.py', 'observer.c', 'observer.f90', 'replay.c'}
            .issubset(index['files']), 'Capture source inventory is incomplete')
    for name, item in index['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'Unsafe capture seal path')
        expected = item['sha256'] if isinstance(item, dict) else item
        require(digest(HERE / name) == expected, 'Sealed source differs: ' + name)
    for name, checksum in index.get('inputs', {}).items():
        require(digest(ROOT / name) == checksum, 'Sealed input differs: ' + name)
    require(index.get('inputs', {}).get('vendor/tirust/tirust.h') == digest(ROOT / 'vendor/tirust/tirust.h'),
            'Production C header must be bound by the source inventory')
    require(digest(CC) == CC_SHA, 'Bound C compiler differs')
    return digest(HERE / 'source-index.json'), index


def admit():
    source_index, index = seal()
    require(digest(PILOT / 'source-index.json') == PILOT_INDEX_SHA, 'Sealed source5 inventory differs')
    pilot = module_from_bytes(PILOT / 'pilot.py', PILOT_SHA)
    pilot.verify_seal(); helpers = pilot.load_helpers(); value = pilot.protocol()
    inherited = pilot.inherited_preparation(value, helpers)
    expected = index['files']['instrument.py']; expected = expected['sha256'] if isinstance(expected, dict) else expected
    instrument = module_from_bytes(HERE / 'instrument.py', expected)
    # Only after ancestor admission, route clean-host scratch locations here.
    pilot.WORK = WORK; pilot.REPORT = REPORT
    return source_index, pilot, helpers, value, inherited['checkpoint']['cases'], instrument


def expected_inputs(spec):
    return [{'case': spec['case'], 'case_id': spec['case_id'], 'point': spec['target_point'],
             'method': method, 'group': group, 'nprop': spec['nprop'], 'rank': spec['rank'],
             'name': 'tirust-input-p%03d-m%d-g%03d.dat' % (spec['target_point'], method, group)}
            for method in (1, 2, 3) for group in spec['groups']]


def readonly_inputs():
    """Exact pre-seal admission, with no writes, compiler, loader or amplitude."""
    source_index, pilot, helpers, value, cases, instrument = admit()
    rows = []
    for spec in SPECS:
        case = cases[spec['case']]; directory = Path(case['master_snapshot']['root']) / case['donor']['subprocess']
        params = dict(case['donor']['compiled']['worker_render_parameters'], **{k: v for k, v in spec.items() if k != 'case'})
        sources = instrument.instrument((directory / 'TIRUST_interface.f').read_text(),
                                        (directory / 'TIR_interface.f').read_text(), params)
        require(set(sources) == {'TIRUST_interface.f', 'TIR_interface.f'}, 'Unexpected instrumented source names')
        driver = instrument.render_driver((PILOT / 'stability_worker.f').read_text(), params)
        compiler = case['donor']['compiled']['effective_compiler_flags']['scopes']['process']['recipes']['check_sa.f']['compiler']
        require(len(compiler) == 1 and digest(compiler[0]['path']) == compiler[0]['sha256'], 'Frozen Fortran compiler differs')
        prerequisites = pilot.process_prerequisites(directory, case['donor']['compiled'])
        abi = static_abi(Path(case['master_snapshot']['root']) / 'lib/tirust_lib/libtirust.a')
        ident = Path(case['master_snapshot']['root']) / 'Cards/ident_card.dat'
        require(ident.is_file() and digest(ident) == 'a39f898e29da6781c365a7946f64d0c9fe1b2ba096137295b638aee0c3dc2493',
                'Frozen model identification card differs')
        rows.append({'case': spec['case'], 'params': params, 'fortran_compiler': compiler[0],
                     'instrumented_source_sha256': {name: hashlib.sha256(text.encode()).hexdigest() for name, text in sources.items()},
                     'driver_sha256': hashlib.sha256(driver.encode()).hexdigest(), 'prerequisites': prerequisites,
                     'static_c_boundary_abi': abi, 'ident_card_sha256': digest(ident)})
    return {'source_index_sha256': source_index, 'cases': rows, 'commands_executed': 0,
            'compiler_invocations': 0, 'loader_invocations': 0, 'amplitude_evaluations': 0}


def closure(pilot, executable, old, listing):
    require(pilot.elf_interpreter(executable) == Path(old['loader']), 'Private executable changed interpreter')
    require('not found' not in listing.lower(), 'Unresolved private runtime dependency')
    paths = {str(Path(name).resolve()) for name in re.findall(r'(/[^\s()]+)', listing) if Path(name).is_file()}
    paths.discard(str(Path(executable).resolve())); paths.add(old['loader'])
    require(paths.issubset(old['files']) and any(Path(path).name == 'libmadloop_tirust.so' for path in paths),
            'Private closure is outside the immutable production runtime')
    require(all(digest(path) == old['files'][path] for path in paths), 'Private runtime bytes changed')
    return {path: old['files'][path] for path in sorted(paths)}


def private_guard(pilot, links, output, context, changed=(), additions=None):
    """Exact private product permission; links and external bytes stay frozen."""
    additions = additions or {}
    observed = links.tree_snapshot(output)
    extra = set(observed['entries']) - set(context['expected_worker']['entries'])
    require(extra.issubset(additions), 'Unregistered capture product')
    admitted = {str(Path(output) / name): additions[name] for name in extra}
    links.assert_worker_inputs(output, context, allow_file_changes=changed, allow_new_paths=admitted)
    return observed


def prepare():
    source_index, pilot, helpers, value, cases, instrument = admit(); links = helpers['links']
    require(not WORK.exists() and not WORK.is_symlink(), 'Fresh capture workspace required')
    destination, kind, chain = links._resolve(WORK, missing=True)
    require(destination == WORK and not chain, 'Capture workspace traverses an alias')
    readonly_inputs()
    WORK.mkdir(parents=True); REPORT.mkdir(); records = {}; failure = None
    try:
        with pilot.signal_requests(helpers['cleanup']) as signals:
            for spec in SPECS:
                old = cases[spec['case']]; output = WORK / 'masters' / spec['case']
                context = links.prepare_worker(Path(old['master_snapshot']['root']), output,
                                              old['master_snapshot'], old['donor_context']['audit'])
                directory = output / old['donor']['subprocess']
                params = dict(old['donor']['compiled']['worker_render_parameters'], **{k: v for k, v in spec.items() if k != 'case'})
                sources = instrument.instrument((directory / 'TIRUST_interface.f').read_text(),
                                                (directory / 'TIR_interface.f').read_text(), params)
                require(set(sources) == {'TIRUST_interface.f', 'TIR_interface.f'}, 'Unexpected patched source names')
                sources['check_sa.f'] = instrument.render_driver((PILOT / 'stability_worker.f').read_text(), params)
                sources['capture.mk'] = MAKEFILE
                for name in ('observer.c', 'observer.f90', 'replay.c'): sources[name] = (HERE / name).read_text()
                sources['tirust.h'] = (ROOT / 'vendor/tirust/tirust.h').read_text()
                for name, text in sources.items():
                    links.assert_private_path(output, directory / name); (directory / name).write_text(text)
                (directory / 'capture-replay').mkdir()
                fc = old['donor']['compiled']['effective_compiler_flags']['scopes']['process']['recipes']['check_sa.f']['compiler'][0]['path']
                shell = shutil.which('bash', path=pilot.environment(directory)['PATH'])
                require(shell is not None, 'Pinned environment Bash unavailable')
                command = pilot.command(['make', '--no-print-directory', '-j1', '-f', 'capture.mk', 'capture-build',
                    'FC=' + fc, 'CC=' + CC, 'CAPTURE_ROOT=' + str(ROOT), 'SHELL=' + str(Path(shell).resolve())], directory,
                    REPORT / 'prepare' / spec['case'], 180, helpers['cleanup'], signals)
                changed = [directory / name for name in ('check_sa.f', 'check_sa.o', 'TIRUST_interface.f',
                           'TIRUST_interface.o', 'TIR_interface.f', 'TIR_interface.o', 'stability-check')]
                names = ('capture.mk', 'capture-link.map', 'observer.c', 'observer.f90', 'replay.c', 'tirust.h',
                         'capture-observer-f.o', 'capture-observer-c.o', 'capture-replay.o', 'capture-replay/stability-check')
                additions = {str((directory / name).relative_to(output)): 'regular_file' for name in names}
                additions[str((directory / 'capture-replay').relative_to(output))] = 'directory'
                snapshot = private_guard(pilot, links, output, context, changed, additions)
                require(command['error'] is None, 'Private observer build failed; no rebuild fallback')
                wrapper = linked_abi(directory / 'stability-check', directory / 'capture-link.map')
                runtimes = {}
                for label, executable in (('capture', directory / 'stability-check'), ('replay', directory / 'capture-replay/stability-check')):
                    loader = pilot.elf_interpreter(executable)
                    checked = pilot.command([loader, '--list', executable], directory,
                        REPORT / 'prepare-loader' / spec['case'] / label, 30, helpers['cleanup'], signals,
                        target_environment=old['native_environment'])
                    require(checked['error'] is None, 'Private target loader check failed')
                    listing = (REPORT / 'prepare-loader' / spec['case'] / label / 'stdout.txt').read_text()
                    runtimes[label] = {'executable_sha256': digest(executable),
                                      'files': closure(pilot, executable, old['runtime_closure'], listing), 'command': checked}
                private_guard(pilot, links, output, context, changed, additions)
                record = dict(old, master_snapshot=snapshot, capture_context=context, command=command,
                              driver_sha256=digest(directory / 'check_sa.f'),
                              executable_sha256=digest(directory / 'stability-check'), capture_runtimes=runtimes,
                              linked_c_boundary_abi=wrapper)
                records[spec['case']] = record
        pilot.verify_production(value)
    except BaseException as error:
        failure = {'class': type(error).__name__, 'message': str(error)}
    finally:
        save(REPORT / 'prepare-terminal.json', {'source_index_sha256': source_index, 'cases': records,
            'status': 'failed' if failure else 'passed', 'error': failure, 'amplitude_evaluations': 0})
    require(failure is None, 'Private preparation failed: ' + str(failure))


def collect_inputs(spec, directory, links, output):
    rows = []
    for item in expected_inputs(spec):
        path = directory / item['name']; row = dict(item, state='missing', replayed=False)
        if path.exists() or path.is_symlink():
            require(path.is_file() and not path.is_symlink(), 'Capture input must be a new regular file')
            links.assert_private_path(output, path)
            require(path.read_bytes().startswith(b'MLTCAP_INPUT_V1\n'), 'Unknown capture input header')
            row.update(state='captured', path=str(path), sha256=digest(path), size=path.stat().st_size)
        rows.append(row)
    return rows


def request_manifest(rows):
    captured = [row for row in rows if row['state'] == 'captured']
    require(len(captured) <= 12 and len({row['path'] for row in captured}) == len(captured), 'Replay input cardinality differs')
    for row in captured:
        path = row['path']; require(Path(path).is_absolute() and '\n' not in path and '\r' not in path,
                                   'Replay input path is not an absolute single line')
        require(Path(path).is_file() and not Path(path).is_symlink(), 'Closed capture input is not a regular file')
        path.encode('ascii'); require(digest(path) == row['sha256'], 'Closed capture input changed')
    return ''.join(row['path'] + '\n' for row in captured)


def capture_rows(closed):
    require([row['case'] for row in closed['results']] == [spec['case'] for spec in SPECS],
            'Closed capture does not retain both registered cases')
    rows = []
    for result, spec in zip(closed['results'], SPECS):
        expected = expected_inputs(spec)
        require(len(result['inputs']) == len(expected), 'Closed capture dropped an expected input')
        for row, fixed in zip(result['inputs'], expected):
            require(all(row.get(name) == value for name, value in fixed.items()), 'Closed capture key differs')
            rows.append(row)
    return rows


def replay_summary(raw, manifest_rows):
    """Bind each lifecycle to the exact closed capture manifest, retaining gaps."""
    available = [row for row in manifest_rows if row['state'] == 'captured']
    require(len(available) <= 12, 'Replay manifest budget exceeded')
    def key(row): return tuple(row[name] for name in ('case_id', 'phase', 'point', 'method', 'group'))
    expected = [{**row, 'phase': 1} for row in available]
    positions = {key(row): i for i, row in enumerate(expected)}
    require(len(positions) == len(expected), 'Duplicate replay manifest key')
    tokens = [('create', 1), ('register', 1), ('evaluate', 1), ('evaluate', 2), ('destroy', 2)]
    observations = [{} for row in expected]; previous = -1; count = 0; unknown = []
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    for number, row in enumerate(rows, 1):
        require(isinstance(row, dict) and isinstance(row.get('input_key'), dict), 'Malformed replay row key')
        require(set(row['input_key']) == {'case_id', 'phase', 'point', 'method', 'group'}
                and all(type(value) is int for value in row['input_key'].values()), 'Malformed replay input key')
        identity = key(row['input_key'])
        require(identity in positions, 'Replay row does not belong to the captured manifest')
        index = positions[identity]; require(index >= previous, 'Replay input lifecycle is out of manifest order')
        previous = index; expected_row = expected[index]; seen = observations[index]
        require(row.get('input_path') == expected_row['path'], 'Replay row input path differs from manifest')
        status = row.get('status')
        require(type(status) is int and -(2 ** 31) <= status < 2 ** 31, 'Replay status is not int32')
        if status not in (0, 1, 2, 3): unknown.append({'row': number, 'status': status})
        token = (row.get('record_type'), row.get('repetition'))
        require(type(row.get('repetition')) is int and token in tokens, 'Unknown replay stage or repetition')
        require(token not in seen and (not seen or tokens.index(token) > max(tokens.index(item) for item in seen)),
                'Duplicate or out-of-order replay stage')
        if token != ('create', 1):
            create = seen.get(('create', 1))
            require(create is not None and create['status'] == 0 and create['context_present'] is True,
                    'Replay stage follows an unsuccessful or missing context creation')
        if token[0] == 'create': require(type(row.get('context_present')) is bool, 'Malformed create context state')
        if token[0] == 'evaluate':
            register = seen.get(('register', 1))
            require(register is not None and register['status'] == 0, 'Replay evaluation lacks successful registration')
            require(token[1] == 1 or ('evaluate', 1) in seen, 'Replay repetition two precedes repetition one')
            count += 1
        require(type(row.get('evaluation_count')) is int and row['evaluation_count'] == count,
                'Replay global evaluation count differs')
        if 'output_bits' in row:
            require(row['record_type'] == 'evaluate' and row['status'] == 0,
                    'Failed or non-evaluation output was serialized')
        if token[0] == 'evaluate' and status == 0:
            bits = row.get('output_bits')
            require(isinstance(bits, list) and len(bits) == 8 * math.comb(expected_row['rank'] + 4, 4),
                    'Successful replay output has wrong tensor word count')
            require(all(isinstance(word, str) and re.fullmatch('[0-9a-f]{16}', word) for word in bits),
                    'Malformed raw replay output word')
        seen[token] = row
    require(count <= 2 * len(available) <= 24, 'Standalone evaluation budget exceeded')
    inputs = []; complete = True
    for fixed, seen in zip(expected, observations):
        create = seen.get(('create', 1)); register = seen.get(('register', 1)); branch = 'normal'
        needed = tokens
        if create is not None and (create['status'] != 0 or create['context_present'] is False):
            needed = tokens[:1]; branch = 'context_setup_failed'
        elif register is not None and register['status'] != 0:
            needed = [tokens[0], tokens[1], tokens[4]]; branch = 'plan_registration_failed'
        missing_required = [list(token) for token in needed if token not in seen]
        missing = [{'stage': token[0], 'repetition': token[1],
                    'reason': 'not_observed' if token in needed else branch} for token in tokens if token not in seen]
        complete = complete and not missing_required
        inputs.append({'input_key': {name: fixed[name] for name in ('case_id', 'phase', 'point', 'method', 'group')},
                       'input_path': fixed['path'], 'branch': branch, 'missing_stages': missing,
                       'missing_required_stages': missing_required})
    return {'stage_count': len(rows), 'evaluate_count': count, 'evaluation_limit': 2 * len(available),
            'failed_stages': sum(row['status'] != 0 for row in rows), 'missing_evaluations': 2 * len(available) - count,
            'protocol_complete': complete, 'qualification_passed': complete and not unknown,
            'unknown_statuses': unknown, 'inputs': inputs, 'numerical_acceptance_claimed': False,
            'full_raw_log_retained': True}


def observe_runtime(pilot, evidence, executable, libraries, record):
    try:
        record['runtime_binding'] = pilot.runtime_binding(evidence, executable, libraries)
        require(record['runtime_binding'].get('verified') is True, 'Observed native runtime closure is not qualified')
    except BaseException as error:
        record.update(state='failed', runtime_qualification_error={'class': type(error).__name__, 'message': str(error)})
        raise


def native(stage):
    source_index, pilot, helpers, value, inherited, instrument = admit(); links = helpers['links']
    prepared = read(REPORT / 'prepare-terminal.json')
    require(prepared['status'] == 'passed' and prepared['source_index_sha256'] == source_index
            and set(prepared['cases']) == {spec['case'] for spec in SPECS}, 'Passed matching preparation required')
    require(not (REPORT / (stage + '-started.json')).exists(), 'Native stage cannot be retried in place')
    closed_capture = read(REPORT / 'capture-terminal.json') if stage == 'replay' else None
    if closed_capture is not None:
        require(closed_capture['source_index_sha256'] == source_index
                and closed_capture['preparation_sha256'] == digest(REPORT / 'prepare-terminal.json'),
                'Closed capture source or preparation differs')
        capture_rows(closed_capture)
    save(REPORT / (stage + '-started.json'), {'source_index_sha256': source_index,
         'preparation_sha256': digest(REPORT / 'prepare-terminal.json'), 'automatic_retry': False})
    results = []; failure = None
    schedules = SPECS if stage == 'capture' else (SPECS[0],)
    try:
        with pilot.signal_requests(helpers['cleanup']) as signals:
            for spec in schedules:
                case = prepared['cases'][spec['case']]; output = WORK / stage / spec['case']
                context = links.prepare_worker(Path(case['master_snapshot']['root']), output,
                                              case['master_snapshot'], case['donor_context']['audit'])
                directory = output / case['donor']['subprocess']; record = {'case': spec['case'], 'state': 'unexecuted', 'native_started': False}
                allowed_names = list(DIAGNOSTICS) + ['param.log', 'tirust-capture-events.jsonl', 'tirust-replay.jsonl', 'request-manifest.txt']
                allowed_names += [item['name'] for item in expected_inputs(spec)]
                try:
                    pilot.prelaunch_worker(value, {'backend': 9}, case, output, helpers, record)
                    if stage == 'capture': argv = [directory / 'stability-check', case['donor']['points'], '9']
                    else:
                        rows = capture_rows(closed_capture)
                        manifest = request_manifest(rows)
                        record['inputs_expected'] = len(rows); record['inputs_available'] = len(manifest.splitlines())
                        record['closed_capture_terminal_sha256'] = digest(REPORT / 'capture-terminal.json')
                        record['request_manifest_sha256'] = hashlib.sha256(manifest.encode('ascii')).hexdigest()
                        if not manifest:
                            record['state'] = 'unexecuted-no-captured-inputs'; continue
                        path = directory / 'request-manifest.txt'; links.assert_private_path(output, path)
                        with path.open('x') as stream: stream.write(manifest)
                        argv = [directory / 'capture-replay/stability-check', path]
                    runtime = case['capture_runtimes'][stage]
                    require(digest(argv[0]) == runtime['executable_sha256'], 'Private native executable changed')
                    require(all(digest(path) == checksum for path, checksum in runtime['files'].items()),
                            'Bound native runtime file changed before launch')
                    with pilot.numerical_lock(value, {'lock_required': True}, signals) as lock:
                        command = pilot.command(argv, directory, REPORT / stage / spec['case'],
                            180 if stage == 'capture' else 60, helpers['cleanup'], signals,
                            target_environment=case['native_environment'])
                    record.update(command=command, lock=lock, native_started=command['process_started'],
                                  state='complete' if command['error'] is None else 'failed')
                    observe_runtime(pilot, REPORT / stage / spec['case'], argv[0], runtime['files'], record)
                    if stage == 'replay':
                        record['replay_summary'] = replay_summary((directory / 'tirust-replay.jsonl').read_text(), rows)
                        require(record['replay_summary']['qualification_passed'], 'Replay protocol is incomplete or unqualified')
                except BaseException as error:
                    record.update(state='failed', error={'class': type(error).__name__, 'message': str(error)})
                finally:
                    changed = [output / name for name in record.get('card_write_intent', [])]
                    changed.extend(path for path in (output / 'SubProcesses').rglob('*')
                                   if path.name == 'param.log' or path.name.endswith(('HelFilter.dat', 'LoopFilter.dat')))
                    additions = {str((directory / name).relative_to(output)): 'regular_file' for name in allowed_names}
                    try:
                        private_guard(pilot, links, output, context, changed, additions)
                        links.verify_tree(Path(case['donor_context']['master']), case['donor_context']['expected_master'])
                        pilot.verify_production(value); record['readonly_guard_passed'] = True
                    except BaseException as error:
                        record.update(state='failed', guard_error={'class': type(error).__name__, 'message': str(error)})
                    if stage == 'capture':
                        try: record['inputs'] = collect_inputs(spec, directory, links, output)
                        except BaseException as error:
                            record.update(state='failed', collection_error={'class': type(error).__name__, 'message': str(error)},
                                          inputs=[dict(item, state='unavailable-collection-failure', replayed=False) for item in expected_inputs(spec)])
                    for name in ('tirust-capture-events.jsonl', 'tirust-replay.jsonl'):
                        path = directory / name
                        if path.is_file(): record[name] = {'path': str(path), 'sha256': digest(path), 'size': path.stat().st_size}
                    results.append(record)
    except BaseException as error:
        failure = {'class': type(error).__name__, 'message': str(error)}
    finally:
        for spec in schedules[len(results):]:
            results.append({'case': spec['case'], 'state': 'unexecuted', 'native_started': False,
                            'inputs': [dict(item, state='missing', replayed=False) for item in expected_inputs(spec)]})
        save(REPORT / (stage + '-terminal.json'), {'source_index_sha256': source_index,
            'preparation_sha256': digest(REPORT / 'prepare-terminal.json'), 'error': failure, 'results': results,
            'status': 'passed' if failure is None and all(row['state'] == 'complete' for row in results) else 'failed',
            'maximum_unique_inputs': 12, 'maximum_standalone_evaluations': 24,
            'missing_inputs_are_not_replaced': True, 'automatic_retry': False})
    require(failure is None and all(row['state'] == 'complete' for row in results), 'Native stage retains failures/unexecuted inputs')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', nargs='?', default='plan', choices=('plan', 'prepare', 'capture', 'replay'))
    parser.add_argument('--execute', action='store_true'); args = parser.parse_args(argv)
    if args.stage == 'plan' or not args.execute:
        print(json.dumps({'stage': args.stage, 'execute': False, 'commands_executed': 0,
            'work_directory': str(WORK), 'cases': list(SPECS), 'maximum_unique_inputs': 12,
            'maximum_standalone_evaluations': 24, 'native_caps_seconds': {'capture_per_case': 180, 'replay_total': 60},
            'cpu': 12, 'cpu_exclusive': False, 'memory_bytes': 4294967296,
            'makefile': MAKEFILE, 'automatic_retry': False}, indent=2, sort_keys=True)); return 0
    if args.stage == 'prepare': prepare()
    else: native(args.stage)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
