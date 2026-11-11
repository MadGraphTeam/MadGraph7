"""Portable retained-byte verification of the private checked CMS20 experiment."""
import gzip
import hashlib
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pin(raw):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def safe(name):
    require(isinstance(name, str) and name and not Path(name).is_absolute() and
            '\\' not in name and all(p not in ('', '.', '..') for p in name.split('/')),
            'Unsafe snapshot path')
    return name


def packet(directory, expected=None):
    """Read only indexed local text. Callbacks have no filesystem fallback."""
    directory = Path(directory)
    raw = (directory / 'evidence-index.json').read_bytes()
    require(expected is None or pin(raw)['sha256'] == expected, 'Ancestor seal differs')
    index = json.loads(raw)
    stored = {}
    for name, identity in index['files'].items():
        value = (directory / safe(name)).read_bytes()
        require(pin(value) == identity, 'Stored snapshot changed: ' + name)
        stored[name] = value
    values = dict(stored)
    # Preserve the exact authenticated index bytes for original source loaders.
    values['evidence-index.json'] = raw
    for logical, record in index.get('compressed_text', {}).items():
        safe(logical); require(record['path'] in stored, 'Missing compressed snapshot')
        require(0 <= record['raw_bytes'] <= 64 << 20, 'Raw text size exceeds bound')
        value = gzip.decompress(stored[record['path']])
        require(pin(value) == {'bytes': record['raw_bytes'], 'sha256': record['raw_sha256']},
                'Decompressed snapshot differs: ' + logical)
        value.decode('utf-8')
        require(b'\0' not in value and not value.startswith((b'\x7fELF', b'!<arch>')), 'Native/binary snapshot')
        values[logical] = value
    for logical, origin in index.get('origins', {}).items():
        require(logical in values and pin(values[logical]) == {k: origin[k] for k in ('bytes', 'sha256')},
                'Origin identity differs')

    def data(name):
        require(safe(name) in values, 'Unindexed read rejected: ' + name)
        return values[name]

    def document(name):
        return json.loads(data(name))

    return index, data, document


def helper(name, data):
    module = types.ModuleType('portable_checked_' + name)
    module.__file__ = str(HERE / (name + '.py'))
    exec(compile(data(name + '.py'), module.__file__, 'exec'), module.__dict__)
    return module


def verify_snapshot(index, data, document, ancestors):
    """All scientific/build helpers receive authenticated read-only callbacks."""
    config = helper('capture', data)
    require(index['schema'] == 'private-normal-checked-cms20-retention-v1' and
            index['dependencies'] == config.DEPENDENCIES and index['closed_stages'] == config.CLOSURES and
            index['source_indices'] == {k: v[1] for k, v in config.SOURCES.items()}, 'Packet scope differs')
    require(set(ancestors) == set(config.DEPENDENCIES), 'Required ancestor set differs')
    origins = {}
    for prefix, (origin, sha) in config.SOURCES.items():
        raw = data(prefix + '/index.json'); require(pin(raw)['sha256'] == sha, 'Original source index differs')
        source = json.loads(raw)
        require(len(source['files']) == {'science': 36, 'controller1': 24, 'controller2': 23, 'plan': 6}[prefix],
                'Original source payload count differs')
        origins[prefix + '/index.json'] = origin + '/index.json'
        for name, identity in source['files'].items():
            require(pin(data(prefix + '/' + safe(name))) == identity, 'Original source payload differs')
            origins[prefix + '/' + name] = origin + '/' + name
    stages = {}
    for label, sha in config.CLOSURES.items():
        prefix, stage = label.split('/'); name = label + '-closure.json'
        raw = data(name); require(pin(raw)['sha256'] == sha, 'Original closed stage differs')
        closed = json.loads(raw)
        old = prefix == 'results1'; success = not (old and stage == 'run')
        source = config.SOURCES['controller1' if old else 'controller2'][1]
        require(closed['source_index_sha256'] == source and closed['stage'] == stage and
                closed['passed'] is success and closed['postcheck_failure'] is None and
                closed['performance_measurement'] is False, 'Original closed stage identity differs')
        require(closed['failure'] == (None if success else "RuntimeError('Command returned 1')"),
                'Historical failure reclassified')
        require(closed['reviewed_build_sha256'] == (None if stage == 'build' else config.CLOSURES['results1/build']),
                'Reviewed donor build differs')
        origin = '.tirust-build/kinematic-normal-checked-executor-' + prefix[-1] + '/results/'
        origins[name] = origin + stage + '-closure.json'
        commands = loaders = 0
        for member, identity in closed['payloads'].items():
            value = data(prefix + '/' + safe(member)); require(pin(value) == identity, 'Closed payload differs')
            origins[prefix + '/' + member] = origin + member
            if member.startswith('execution/') and member.endswith('.json'):
                receipt = json.loads(value); commands += 1; loaders += '--list' in receipt['argv']
                failed_worker = old and stage == 'run' and 'worker-' in member
                require(receipt['returncode'] == (1 if failed_worker else 0) and
                        receipt['cleanup_exception'] is None and receipt['cleanup']['survivors'] == [],
                        'Original command or cleanup differs')
                require((receipt['exception'] is not None) is failed_worker, 'Command failure classification differs')
                logfile = str(Path(member).with_suffix('.txt'))
                require(logfile in closed['payloads'] and
                        pin(data(prefix + '/' + logfile))['sha256'] == receipt['log_sha256'], 'Command/log join differs')
        expected = (138, 64, 1) if stage == 'build' else (20, 2, 1) if old else (26, 2, 1)
        require((len(closed['payloads']), commands, loaders) == expected and
                closed['owned_command_count'] == commands and closed['loader_query_count'] == loaders,
                'Original stage inventory differs')
        require(data(prefix + '/' + stage + '-before.json') == data(prefix + '/' + stage + '-after.json'),
                'Protected snapshots differ')
        stages[label] = {'passed': success, 'payloads': len(closed['payloads']), 'commands': commands, 'loader_queries': loaders}
    for name, sha in config.EXTRAS.items():
        require(pin(data(name))['sha256'] == sha, 'Closed extra review differs')
        prefix, tail = name.split('/', 1)
        origins[name] = '.tirust-build/kinematic-normal-checked-executor-' + prefix[-1] + '/results/' + tail
    for name, sha in config.PREFLIGHTS.items():
        require(pin(data('preflights/' + name))['sha256'] == sha, 'Historical preflight differs')
        origins['preflights/' + name] = '.tirust-build/' + name
    require(len(index['geometry_helper_origins']) == 14, 'Geometry source helper set differs')
    for origin, identity in index['geometry_helper_origins'].items():
        require(pin(data('helpers/' + safe(origin))) == identity, 'Geometry helper differs')
        origins['helpers/' + origin] = origin
    require(set(origins) == set(index['origins']) and all(index['origins'][k]['path'] == v for k, v in origins.items()),
            'Unaccounted or altered source origins')
    build = helper('build_checks', data).verify_build(data, document, ancestors)
    workers = helper('worker_checks', data).verify_workers(data, document, ancestors)
    return {'passed': True, 'scope': 'Private checked CMS20, ordinary bootstrap, cache disabled',
            'closed_stages': stages, 'source_payloads': {'science': 36, 'controller1': 24, 'controller2': 23, 'plan': 6},
            'build': build, 'workers': workers, 'original_failed_run_preserved': True,
            'native_actions': 0, 'live_host_files_read': False, 'binaries_or_libraries_loaded': False,
            'generic_exporter_qualified': False, 'production_promoted': False, 'enabled_cache_qualified': False,
            'full_hardware_trap_or_environment_qualification': False, 'performance_qualified': False}


def main():
    index, data, document = packet(HERE)
    ancestors = {name: packet(HERE.parent / safe(name), sha) for name, sha in index['dependencies'].items()}
    result = verify_snapshot(index, data, document, ancestors)
    print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == '__main__':
    main()
