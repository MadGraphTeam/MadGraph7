#!/usr/bin/env python3
"""Replay the closed link inventory with retained compiler-query text; no child runs."""
import importlib.util
import json
from pathlib import Path
import re
import shlex

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('closed_lifecycle_inventory', HERE / 'run.py')
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)
OLD = R.ROOT / 'docs/benchmarkfolder/experiments/subprocess-cache-lifecycle-current-qualification-1'
SUB = R.ROOT / '.tirust-build/subprocess-cache-lifecycle-current-1/export/SubProcesses'


def main():
    stage = json.loads((OLD / 'stage-build.json').read_text())
    assert stage['status'] == 'failed' and not stage['native_execution']
    queries = {}
    for path in sorted((OLD / 'execution').glob('query-build-*.json')):
        meta = json.loads(path.read_text())
        raw = path.with_suffix('.txt').read_text()
        assert R.sha(path.with_suffix('.txt')) == meta['log_sha256'] and meta['returncode'] == 0
        queries[tuple(meta['argv'][3:])] = raw
    def capture(args, **kwargs):
        return queries[tuple(map(str, args))]
    R.q.capture = capture
    info = (OLD / 'execution/link-info.txt').read_text()
    fields = {line.split()[0]: shlex.split(line)[1:] for line in info.splitlines() if line.strip()}
    initial = R.q.external_inputs(SUB, info, fields['FC'], None)
    search = (OLD / 'execution/compiler-search-dirs.txt').read_text()
    for line in search.splitlines():
        if line.startswith('libraries: ='):
            for spelling in line.split('=', 1)[1].split(':'):
                directory = Path(spelling)
                if directory.is_dir():
                    for path in directory.iterdir():
                        if path.is_file() and (path.suffix in ('.a', '.o') or '.so' in path.name):
                            initial[str(path.resolve())] = R.sha(path)
    wrapper = R.linker.wrapper_flags(Path(fields['FC'][0]), {})
    paths = R.linker.search_paths(info, search, wrapper['flags'], SUB)
    closure = R.linker.prebind(initial, paths, R.linker.compiler_libraries(info + '\n' + wrapper['flags']))
    loads = []
    missing_before, missing_after = [], []
    for spelling in re.findall(r'^LOAD (.+)$', (SUB / 'lifecycle_plain.map').read_text(), re.M):
        path = Path(spelling)
        if not path.is_absolute():
            path = SUB / path
        resolved = str(path.resolve())
        # Existing process/model objects were already bound separately by the runner.
        if path.suffix == '.o' and str(SUB) in resolved:
            continue
        loads.append({'spelling': spelling, 'resolved': resolved, 'sha256': R.sha(path)})
        if resolved not in initial:
            missing_before.append(resolved)
        if resolved not in closure['files']:
            missing_after.append(resolved)
        R.linker.require_prebound_load(path, closure['files'])
    assert missing_before and not missing_after
    value = {'schema': 1, 'scope': 'Retrospective read-only reconstruction from closed compiler-query/link text; no compiler/linker/native replay.',
             'failed_build_stage_sha256': R.sha(OLD / 'stage-build.json'),
             'failed_map_sha256': R.sha(SUB / 'lifecycle_plain.map'),
             'new_helper_sha256': R.sha(HERE / 'linker_bind.py'),
             'wrapper': wrapper, 'initial_inventory_reconstructed_sha256':
                 R.hashlib.sha256(json.dumps(initial, sort_keys=True).encode()).hexdigest(),
             'missing_before': sorted(set(missing_before)), 'missing_after': missing_after,
             'actual_loads': loads, 'closure': closure,
             'execution_performed': False, 'post_link_waiver': False}
    with (HERE / 'linker-diagnosis.json').open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
    print(json.dumps({'missing_before': value['missing_before'], 'missing_after': [],
                      'bound_scripts': len(closure['scripts']), 'execution_performed': False}, indent=2))


if __name__ == '__main__':
    main()
