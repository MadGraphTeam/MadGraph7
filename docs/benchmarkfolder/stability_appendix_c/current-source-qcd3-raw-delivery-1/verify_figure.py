"""Pure verification of the standalone current-source QCD n3 figure."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import xml.etree.ElementTree as E

BASE = Path(__file__).resolve().parent.parent
FIGURE = BASE / 'current-source-qcd3-raw-figure-1'
DATA = BASE / 'current-source-qcd3-raw-1'
PLOT = BASE / 'plotter-source-6'
FIGURE_SHA = 'cac6a230d034e5ac039232ae8f78fc8b4e2673e97c9abf0f8af0bc2b7bdddcb7'
DATA_SHA = 'b2589cfcd71f9940a6dea98376be16b93e2b056aae4596d86a6fd6d5747dce30'
POINTS_SHA = 'f16806af788775fc42d5104092799d7d1d5879a5785c2b2f4760d5e4c3a6ddff'
PLOT_SHA = 'a0521f17a658e04f8dc320509e733de8fd0e93ac4d02481d908373291ac706ae'
STEM = 'current-source-qcd-n3-raw-dp-stability'

def sha(blob): return hashlib.sha256(blob).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)

def verify():
    raw = (FIGURE / 'index.json').read_bytes(); index = json.loads(raw)
    require(sha(raw) == FIGURE_SHA and sha((DATA / 'index.json').read_bytes()) == DATA_SHA, 'Admitted data/figure seal differs')
    require({p.name for p in FIGURE.iterdir()} == set(index['files']) | {'index.json'}, 'Figure inventory differs')
    for name, item in index['files'].items():
        path = FIGURE / name; require(Path(name).name == name and not path.is_symlink(), 'Unsafe figure payload')
        blob = path.read_bytes(); require(len(blob) == item['bytes'] and sha(blob) == item['sha256'], 'Figure payload differs')
    points = (DATA / 'point-metrics.json').read_bytes()
    require((FIGURE / 'point-metrics.json').read_bytes() == points and sha(points) == POINTS_SHA, 'Point data differs')
    plot_index = (PLOT / 'index.json').read_bytes(); require(sha(plot_index) == PLOT_SHA, 'Plotter seal differs')
    for name, item in json.loads(plot_index)['files'].items():
        require(Path(name).name == name and not (PLOT / name).is_symlink(), 'Unsafe plotter input')
        blob = (PLOT / name).read_bytes(); require(len(blob) == item['bytes'] and sha(blob) == item['sha256'], 'Plotter source differs')
    spec = importlib.util.spec_from_file_location('pure_current_source_plotter', PLOT / 'plot_metrics.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    data = module.validate_dataset(json.loads(points))
    expected = module.summarize(data, module.threshold_grid(data)); expected['input_sha256'] = sha(points)
    require(json.loads((FIGURE / (STEM + '.json')).read_bytes()) == expected, 'Tail/bound/ratio replay differs')
    png = (FIGURE / (STEM + '.png')).read_bytes(); require(png[:8] == b'\x89PNG\r\n\x1a\n', 'Invalid PNG')
    pdf = (FIGURE / (STEM + '.pdf')).read_bytes(); require(pdf.startswith(b'%PDF-') and pdf.rstrip().endswith(b'%%EOF'), 'Invalid PDF')
    svg = E.fromstring((FIGURE / (STEM + '.svg')).read_bytes())
    require(svg.tag == '{http://www.w3.org/2000/svg}svg' and not svg.findall('.//{http://www.w3.org/2000/svg}script'), 'Invalid SVG')
    return {'verified': True, 'figure_index_sha256': sha(raw), 'data_index_sha256': DATA_SHA,
            'payload_count': len(index['files']), 'payload_bytes': sum(x['bytes'] for x in index['files'].values()),
            'all_tails_bounds_ratios_recomputed': True, 'png_dimensions': list(struct.unpack('>II', png[16:24])),
            'svg_dimensions': [svg.get('width'), svg.get('height')], 'native_calls': 0}

if __name__ == '__main__': print(json.dumps(verify(), indent=2, sort_keys=True))
