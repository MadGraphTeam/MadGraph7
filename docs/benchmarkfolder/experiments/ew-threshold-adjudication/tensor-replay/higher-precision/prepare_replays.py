#!/usr/bin/env python3
"""Derive source-bound forward/reverse 512-bit replays without native calls."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
FORWARD = HERE.parent
TRACE = FORWARD.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def name(path):
    return str(path.resolve().relative_to(ROOT))


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def prepare(variant):
    assert Path.cwd().resolve() == ROOT
    source_folder = FORWARD if variant == 'forward' else FORWARD / 'reversal'
    out = HERE / ('replay-' + variant)
    work = ROOT / ('.tirust-build/ew-threshold-512-replay-' + variant)
    runtime = ROOT / '.tirust-build/ew-threshold-512/runtime'
    if out.exists() or work.exists():
        raise ValueError('Refusing to overwrite a derived replay')
    build = json.loads((HERE / 'build.json').read_text())
    assert build['complete'] and build['protected_inputs_unchanged']
    assert (build['working_bits'], build['input_bits'], build['gate']) == (512, 113, '2^-392')
    assert digest(runtime / 'libmadloop_tirust.so') == build['library_sha256']
    for path, expected in build['protected_inputs_after'].items():
        assert digest(ROOT / path) == expected, path
    source_execution = json.loads((source_folder / 'execution.json').read_text())
    assert source_execution['complete'] and source_execution['protected_inputs_unchanged']
    source_summary = json.loads((source_folder / 'summary.json').read_text())
    assert source_summary['execution_manifest_sha256'] == digest(source_folder / 'execution.json')
    selected = json.loads((source_folder / 'selected-inputs.json').read_text())
    for item in ('trace', 'summary', 'execution'):
        assert digest(TRACE / (item + '.json')) == selected['parent_' + item + '_sha256']
    assert digest(source_folder / 'selected-inputs.dat') == selected['input_stream_sha256']
    out.mkdir()
    work.mkdir()
    # Reuse the reviewed reversal runner because it contains no build action.
    run_original = (FORWARD / 'reversal/run.py').read_text()
    run = replace_once(run_original,
        'from prepare import ROOT, REPORT, WORK, RUNTIME, sources',
        'from prepare import ROOT, REPORT, TRACE, WORK, RUNTIME, sources')
    run = replace_once(run, "REPORT.parents[1] / (item + '.json')", "TRACE / (item + '.json')")
    run = replace_once(run,
        "'historical TIRust wrapper/module/DSO; unchanged NinjaRS archive; 256-bit working precision'",
        "'historical Fortran wrapper; private TIRust512/guard2^-392; unchanged NinjaRS256; binary128 inputs/output'")
    analyze_original = (FORWARD / 'reversal/analyze.py').read_text()
    analyze = replace_once(analyze_original, 'from prepare import extract',
        'from prepare import extract, TRACE, WORK')
    analyze = replace_once(analyze, 'ROOT=HERE.parents[5]', 'ROOT=HERE.parents[6]')
    analyze = replace_once(analyze,
        "required_artifacts={'.tirust-build/ew-threshold-tensor-replay-reversal/replay.o',\n                        '.tirust-build/ew-threshold-tensor-replay-reversal/replay'}",
        "required_artifacts={str((WORK/p).relative_to(ROOT)) for p in ('replay.o','replay')}")
    analyze = replace_once(analyze, "runtime=sources['historical_tirust_dso']", "runtime=sources['diagnostic_tirust_dso']")
    analyze = replace_once(analyze, "private_work=ROOT/'.tirust-build/ew-threshold-tensor-replay-reversal'", 'private_work=WORK')
    analyze = replace_once(analyze, "HERE.parents[1]/(name+'.json')", "TRACE/(name+'.json')")
    analyze = replace_once(analyze, "HERE.parents[1]/'trace.json'", "TRACE/'trace.json'")
    analyze = replace_once(analyze, "'scope':'Tensor diagnostic only; no amplitude reconstruction or numerical acceptance.'",
        "'scope':'TIRust512/guard2^-392 versus NinjaRS256; identical113-bit inputs/output; tensor diagnostic only.'")
    local_prepare = f'''#!/usr/bin/env python3
"""Frozen source bindings for the derived {variant} 512-bit replay."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
REPORT=Path(__file__).resolve().parent
ROOT=REPORT.parents[6]
TRACE=ROOT/{name(TRACE)!r}
WORK=ROOT/{name(work)!r}
RUNTIME=ROOT/{name(runtime)!r}
spec=importlib.util.spec_from_file_location('captured_{variant}_inputs',ROOT/{name(source_folder / 'prepare.py')!r})
original=importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
extract=original.extract
def sources():
    data=json.loads((REPORT/'source-manifest.json').read_text())
    for path,expected in data['sources_and_link_inputs'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==expected,path
    return data
'''
    (out / 'prepare.py').write_text(local_prepare)
    (out / 'run.py').write_text(run)
    (out / 'analyze.py').write_text(analyze)
    (out / 'derivation.patch').write_text(''.join(
        ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
            fromfile='historical-reversal/' + filename, tofile='512-' + variant + '/' + filename))
        for filename, before, after in [('run.py', run_original, run), ('analyze.py', analyze_original, analyze)]))
    for filename in ('replay.f90', 'selected-inputs.dat'):
        shutil.copy2(source_folder / filename, out / filename)
        shutil.copy2(source_folder / filename, work / filename)
    compiled = json.loads((FORWARD / 'compilation.json').read_text())
    old_work = ROOT / '.tirust-build/ew-threshold-tensor-replay'
    for filename in ('replay', 'replay.o'):
        assert digest(old_work / filename) == compiled['artifacts_sha256'][name(old_work / filename)]
        shutil.copy2(old_work / filename, work / filename)
        assert digest(work / filename) == digest(old_work / filename)
    original_sources = json.loads((source_folder / 'source-manifest.json').read_text())
    checks = dict(original_sources['sources_and_link_inputs'])
    protected = [p for p in source_folder.iterdir() if p.is_file()]
    protected += [p for p in HERE.iterdir() if p.is_file()]
    protected += list((HERE / 'source').iterdir())
    protected += [p for p in out.iterdir() if p.is_file()]
    protected += [runtime / 'libmadloop_tirust.so']
    checks.update({name(p): digest(p) for p in protected})
    manifest = dict(original_sources, sources_and_link_inputs=checks,
        diagnostic_tirust_dso=name(runtime / 'libmadloop_tirust.so'),
        precision='TIRust512 working bits/guard2^-392; NinjaRS256; both113-bit input/output transport',
        scope='Unchanged historical executable loaded with a source-bound private diagnostic DSO.')
    dump(out / 'source-manifest.json', manifest)
    selected.update(source_manifest_sha256=digest(out / 'source-manifest.json'),
        scope='Prepared private512 replay; no native execution from this script.',
        historical_selection_manifest_sha256=digest(source_folder / 'selected-inputs.json'))
    dump(out / 'selected-inputs.json', selected)
    shutil.copy2(FORWARD / 'build.log', out / 'build.log')
    dump(out / 'compilation.json', dict(compiled, reused_existing_executable=True,
        inherited_compilation_sha256=digest(FORWARD / 'compilation.json'),
        run_script_sha256=digest(out / 'run.py'),
        scope='No new Fortran compilation/link. Private512 DSO selected at execution via LD_LIBRARY_PATH and actual process maps.',
        artifacts_sha256={name(work / filename): digest(work / filename)
            for filename in ('replay.f90', 'replay.o', 'replay', 'selected-inputs.dat')}))
    print(f'Prepared {variant}: unchanged executable, exact captured inputs, TIRust512/NinjaRS256; no native calls.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('variant', choices=('forward', 'reversal'))
    prepare(parser.parse_args().variant)
