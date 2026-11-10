"""Render and verify four explicitly rooted, closed raw-DP stability figures."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import types

CASES = ('qcd_n2_gg_ttx', 'qcd_n3_uux_zgg', 'ew_n2_epem_mupmum', 'qcd_n4_uux_zzgg')
PLOTTER_INDEX_SHA256 = 'a0521f17a658e04f8dc320509e733de8fd0e93ac4d02481d908373291ac706ae'
PLOTTER_MODULE_SHA256 = '2fc1d3fb368d623726bb26a55ca6b85e207eb6ebaf07b97da1f1846a6e92395a'
SCHEMA = 'madloop-raw-stability-figures-v1'
CAPTION = ('Host forced-DP, card-selected backends; reducer-internal precision and actual backend '
           'are unobserved. Each process has 100 cut-qualified physical points (1% empirical '
           'resolution). Dotted upper bounds account for unavailable outcomes; they are not '
           'measured failure fractions. Incomplete populations and zero Ninja exceedances mask '
           'ratios. Internal consistency is not a true-error bound; no performance claim is made.')
README = ('# Raw-DP stability figures\n\n' + CAPTION + '\n\n'
          'Each figure uses one case from the complete copied point-metrics.json, retaining its '
          'provenance and the SHA-256 of the complete four-case dataset. All five intended '
          'reducers remain present, with zero and unavailable counts.\n\n'
          'Verification requires explicit data and frozen plotter roots. It checks figure bytes '
          'and recomputes statistical JSON without importing matplotlib or reading raw journals.\n'
          'The returned figure-index SHA must be checked against its independently admitted pin; '
          'self-consistency verification alone does not establish admission.\n'
          '`python3 -B figures.py verify FIGURE_ROOT --data-root DATA_ROOT --plotter-root PLOTTER_ROOT`\n')
_MAX_FILE = 40 * 1024 * 1024


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read(root, relative):
    """Bounded reads through directory descriptors; no symlink component."""
    if not isinstance(relative, str) or '\\' in relative or '\0' in relative:
        raise ValueError('Unsafe relative path')
    path = PurePosixPath(relative)
    if path.is_absolute() or '..' in path.parts or path.as_posix() != relative or relative in ('', '.'):
        raise ValueError('Unsafe relative path')
    full = Path(root).absolute() / Path(*path.parts)
    if '..' in full.parts:
        raise ValueError('Unsafe root')
    descriptor = os.open(full.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in full.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_fd = os.open(full.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=descriptor)
        with os.fdopen(file_fd, 'rb') as source:
            size = os.fstat(source.fileno())
            if not stat.S_ISREG(size.st_mode) or size.st_size > _MAX_FILE:
                raise ValueError('Unexpected file type or size')
            data = source.read(_MAX_FILE + 1)
            if len(data) > _MAX_FILE or len(data) != size.st_size:
                raise ValueError('File changed or exceeded size limit')
            return data
    finally:
        os.close(descriptor)


def _checked(root, name, binding):
    if (not isinstance(binding, dict) or set(binding) != {'sha256', 'bytes'}
            or type(binding['bytes']) is not int or binding['bytes'] < 0):
        raise ValueError('Invalid file binding')
    data = _read(root, name)
    if len(data) != binding['bytes'] or _sha(data) != binding['sha256']:
        raise ValueError('File binding mismatch: ' + name)
    return data


def _load_plotter(root):
    index_bytes = _read(root, 'index.json')
    if _sha(index_bytes) != PLOTTER_INDEX_SHA256:
        raise ValueError('Frozen plotter index mismatch')
    index = json.loads(index_bytes)
    if index.get('schema') != 'madloop-stability-plotter-source-v1':
        raise ValueError('Unexpected plotter schema')
    sources = {name: _checked(root, name, binding) for name, binding in index['files'].items()}
    source = sources['plot_metrics.py']
    if _sha(source) != PLOTTER_MODULE_SHA256:
        raise ValueError('Frozen plotter module mismatch')
    module = types.ModuleType('frozen_stability_plotter')
    module.__file__ = str(Path(root) / 'plot_metrics.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


def _load_data(root, expected_index_sha256, plotter):
    index_bytes = _read(root, 'index.json')
    if _sha(index_bytes) != expected_index_sha256:
        raise ValueError('Data index mismatch')
    index = json.loads(index_bytes)
    if index.get('schema') != 'madloop-raw-stability-capsule-v1':
        raise ValueError('Unexpected data schema')
    payload = index['payloads']['point-metrics.json']
    if not isinstance(payload, dict) or payload.get('kind') != 'file':
        raise ValueError('Unsupported point-metrics storage; direct file required')
    binding = {'sha256': payload['sha256'], 'bytes': payload['bytes']}
    if index['files'].get(payload['path']) != binding:
        raise ValueError('Dataset payload and file binding disagree')
    raw = _checked(root, payload['path'], binding)
    data = plotter.validate_dataset(json.loads(raw))
    if (data['metric'] != 'raw-dp-finite-consistency'
            or {case['id'] for case in data['cases']} != set(CASES)
            or len(data['cases']) != len(CASES)
            or any(case['point_ids'] != list(range(1, 101)) for case in data['cases'])):
        raise ValueError('Expected all four raw-DP cases and point IDs 1..100')
    return raw, data


def _subset(data, case_id):
    return dict(data, cases=[next(case for case in data['cases'] if case['id'] == case_id)])


def _json(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()


def _dependencies(data_index_sha256, dataset_sha256):
    return {'data_index_sha256': data_index_sha256, 'dataset_sha256': dataset_sha256,
            'plotter_index_sha256': PLOTTER_INDEX_SHA256,
            'plotter_module_sha256': PLOTTER_MODULE_SHA256,
            'figure_source_sha256': _sha(Path(__file__).read_bytes())}


def _files():
    return {'point-metrics.json', 'provenance.json', 'README.md', 'figures.py'} | {
        f'figures/{case_id}.{suffix}' for case_id in CASES for suffix in ('pdf', 'png', 'json')}


def render(figure_root, *, data_root, plotter_root, expected_data_index_sha256):
    """Create a fresh figure directory from verified point metrics only."""
    plotter = _load_plotter(plotter_root)
    raw, data = _load_data(data_root, expected_data_index_sha256, plotter)
    dependencies = _dependencies(expected_data_index_sha256, _sha(raw))
    root = Path(figure_root)
    if any(parent.is_symlink() for parent in (root, *root.absolute().parents)):
        raise ValueError('Symlink figure root or parent')
    root.mkdir()  # Existing directories, including empty ones, are not overwritten.
    (root / 'figures').mkdir()
    for case_id in CASES:
        plotter.render(_subset(data, case_id), root / 'figures' / case_id,
                       input_sha256=dependencies['dataset_sha256'])
    (root / 'point-metrics.json').write_bytes(raw)
    (root / 'provenance.json').write_bytes(_json({'dependencies': dependencies,
                                               'case_ids': list(CASES), 'caption': CAPTION}))
    (root / 'README.md').write_text(README)
    (root / 'figures.py').write_bytes(Path(__file__).read_bytes())
    files = {}
    for name in sorted(_files()):
        captured = _read(root, name)
        files[name] = {'sha256': _sha(captured), 'bytes': len(captured)}
    (root / 'index.json').write_bytes(_json({'schema': SCHEMA, 'dependencies': dependencies,
                                          'files': files}))
    return verify(root, data_root=data_root, plotter_root=plotter_root)


def verify(figure_root, *, data_root, plotter_root):
    """Verify consistency; admission requires externally pinning the returned index SHA."""
    root = Path(figure_root)
    index_bytes = _read(root, 'index.json')
    index = json.loads(index_bytes)
    if index.get('schema') != SCHEMA or set(index.get('files', {})) != _files():
        raise ValueError('Unexpected figure inventory')
    physical = set()
    for path in root.rglob('*'):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise ValueError('Unexpected symlink or file type in figure inventory')
        if path.is_file():
            physical.add(path.relative_to(root).as_posix())
    if physical != _files() | {'index.json'}:
        raise ValueError('Physical figure inventory mismatch')
    files = {name: _checked(root, name, binding) for name, binding in index['files'].items()}
    plotter = _load_plotter(plotter_root)
    dependencies = index['dependencies']
    raw, data = _load_data(data_root, dependencies['data_index_sha256'], plotter)
    expected = _dependencies(dependencies['data_index_sha256'], _sha(raw))
    if (dependencies != expected or files['point-metrics.json'] != raw
            or files['figures.py'] != Path(__file__).read_bytes()
            or files['README.md'] != README.encode()
            or json.loads(files['provenance.json']) != {'dependencies': expected,
                                                      'case_ids': list(CASES), 'caption': CAPTION}):
        raise ValueError('Figure provenance mismatch')
    for case_id in CASES:
        subset = _subset(data, case_id)
        summary = plotter.summarize(subset, plotter.threshold_grid(subset))
        summary['input_sha256'] = dependencies['dataset_sha256']
        if json.loads(files[f'figures/{case_id}.json']) != summary:
            raise ValueError('Statistical figure summary mismatch: ' + case_id)
    return {'verified': True, 'index_sha256': _sha(index_bytes),
            'dependencies': dependencies, 'cases': list(CASES), 'artifacts': 12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('render', 'verify'))
    parser.add_argument('figure_root', type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--plotter-root', required=True, type=Path)
    parser.add_argument('--expected-data-index-sha256')
    args = parser.parse_args()
    kwargs = {'data_root': args.data_root, 'plotter_root': args.plotter_root}
    if args.operation == 'render':
        if not args.expected_data_index_sha256:
            parser.error('render requires --expected-data-index-sha256')
        kwargs['expected_data_index_sha256'] = args.expected_data_index_sha256
    print(json.dumps(globals()[args.operation](args.figure_root, **kwargs), indent=2))


if __name__ == '__main__':
    main()
