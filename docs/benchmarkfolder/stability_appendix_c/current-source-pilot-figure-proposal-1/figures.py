"""Source-only proposal: export one explicitly admitted current-source QCDn3 figure."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import types

PLOTTER_INDEX_SHA256 = 'a0521f17a658e04f8dc320509e733de8fd0e93ac4d02481d908373291ac706ae'
PLOTTER_MODULE_SHA256 = '2fc1d3fb368d623726bb26a55ca6b85e207eb6ebaf07b97da1f1846a6e92395a'
CASE = 'qcd_n3_uux_zgg'
BASENAME = 'current-source-qcd-n3-raw-dp-stability'
CAPTION = ('Current-source QCD pilot: u u~ > Z g g. Host forced-DP with card-selected '
           'backends; return code 110 does not identify the actual reducer or its internal '
           'precision. The 100 cut-qualified physical points give 1% empirical resolution. '
           'Dotted curves bound unavailable outcomes; they are not measured failure fractions. '
           'Incomplete populations and zero Ninja exceedances mask ratios. Internal consistency '
           'is not independent true error; no performance claim is made.')
_MAX_BYTES = 40 * 1024 * 1024


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_file(path):
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.absolute().parents)) or not path.is_file():
        raise ValueError('A regular file without symlink components is required')
    with path.open('rb') as source:
        raw = source.read(_MAX_BYTES + 1)
    if len(raw) > _MAX_BYTES:
        raise ValueError('Input exceeds the file budget')
    return raw


def _load_plotter(root):
    root = Path(root)
    raw_index = _read_file(root / 'index.json')
    if _sha(raw_index) != PLOTTER_INDEX_SHA256:
        raise ValueError('Frozen plotter index mismatch')
    index = json.loads(raw_index)
    for name, binding in index['files'].items():
        if Path(name).name != name:
            raise ValueError('Unexpected plotter source path')
        raw = _read_file(root / name)
        if len(raw) != binding['bytes'] or _sha(raw) != binding['sha256']:
            raise ValueError('Frozen plotter file mismatch')
    source = _read_file(root / 'plot_metrics.py')
    if _sha(source) != PLOTTER_MODULE_SHA256:
        raise ValueError('Frozen plotter module mismatch')
    module = types.ModuleType('frozen_current_source_plotter')
    module.__file__ = str(root / 'plot_metrics.py')
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    return module


@contextmanager
def _svg_export(figure_class, target_pdf):
    """Export the exact target Figure again; restore the original hook on exit."""
    original = figure_class.savefig
    target_pdf = Path(target_pdf)

    def savefig(figure, filename, *args, **kwargs):
        if Path(filename) != target_pdf:
            return original(figure, filename, *args, **kwargs)
        svg = target_pdf.with_suffix('.svg')
        if svg.exists():
            raise FileExistsError('Refusing to overwrite SVG')
        result = original(figure, filename, *args, **kwargs)
        svg_kwargs = dict(kwargs, metadata={'Date': None, 'Creator': 'MadLoop current-source pilot'})
        original(figure, svg, *args, **svg_kwargs)
        return result

    figure_class.savefig = savefig
    try:
        yield
    finally:
        figure_class.savefig = original


def _render_with_svg(plotter, data, prefix, *, input_sha256):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure

    with matplotlib.rc_context({'svg.hashsalt': BASENAME}), _svg_export(Figure, prefix.with_suffix('.pdf')):
        return plotter.render(data, prefix, input_sha256=input_sha256)


def render(output_directory, *, point_metrics_file, expected_point_metrics_sha256, plotter_root):
    """Render only after external verification and admission of the source capsule.

    The expected dataset hash must come from that independently pinned capsule,
    not from discovery or an unverified adjacent file. No raw journal is read.
    """
    plotter = _load_plotter(plotter_root)
    raw = _read_file(point_metrics_file)
    if _sha(raw) != expected_point_metrics_sha256:
        raise ValueError('Point-metrics hash mismatch')
    data = plotter.validate_dataset(json.loads(raw))
    candidates = [case for case in data['cases'] if case['id'] == CASE]
    if (data['metric'] != 'raw-dp-finite-consistency' or len(candidates) != 1
            or candidates[0]['point_ids'] != list(range(1, 101))):
        raise ValueError('Expected the 100-point raw-DP QCDn3 pilot')
    case = dict(candidates[0], label='Current-source QCD pilot: u u~ > Z g g')
    subset = dict(data, cases=[case])
    output = Path(output_directory)
    if any(p.is_symlink() for p in (output, *output.absolute().parents)):
        raise ValueError('Symlink output directory or parent')
    output.mkdir()  # A fresh directory is mandatory, including for an empty predecessor.
    prefix = output / BASENAME
    summary = _render_with_svg(plotter, subset, prefix, input_sha256=_sha(raw))
    (output / 'point-metrics.json').write_bytes(raw)
    (output / 'caption.txt').write_text(CAPTION + '\n')
    expected = {BASENAME + suffix for suffix in ('.pdf', '.svg', '.png', '.json')} | {
        'point-metrics.json', 'caption.txt'}
    if {p.name for p in output.iterdir()} != expected:
        raise ValueError('Unexpected exported figure inventory')
    files = {}
    for name in sorted(expected):
        captured = _read_file(output / name)
        files[name] = {'sha256': _sha(captured), 'bytes': len(captured)}
    index = {'schema': 'madloop-current-source-pilot-figure-v1',
             'dependencies': {'point_metrics_sha256': _sha(raw),
                              'plotter_index_sha256': PLOTTER_INDEX_SHA256,
                              'plotter_module_sha256': PLOTTER_MODULE_SHA256,
                              'wrapper_sha256': _sha(_read_file(__file__))},
             'case': CASE, 'caption': CAPTION, 'files': files}
    (output / 'index.json').write_text(json.dumps(index, indent=2, sort_keys=True) + '\n')
    return {'index_sha256': _sha(_read_file(output / 'index.json')), 'summary': summary}
