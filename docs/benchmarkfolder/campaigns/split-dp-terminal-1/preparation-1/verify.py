#!/usr/bin/env python3
"""Freeze/verify source preparation only; never launch a compiler or reducer."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd()
HERE = Path(__file__).resolve().parent
REPORT = HERE.parent
WORK = ROOT / '.benchmark-work/tirust/campaigns/split-dp-terminal-1'
EXPECTED_MANIFEST = '1237c563b16a13e2c98327b494a4e5d0b8dec8eb6c7020d632b9b86839143a31'
PROTOCOLS = ['tirust_benchmark.py', 'tirust_benchmark_campaign.py', 'tirust_benchmark_compiler.py',
             'tirust_benchmark_relink.py', 'tirust_benchmark_worker.f']

def read(path):
    return json.loads(path.read_text())

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def identity(path):
    return {'sha256': digest(path), 'bytes': path.stat().st_size}

def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write('\n')

def inventory():
    paths = set()
    for tree in (HERE, REPORT/'inputs/baseline', REPORT/'inputs/candidate',
                 REPORT/'inputs/source-identities', REPORT/'commands'):
        for path in tree.rglob('*'):
            if path.is_dir():
                continue
            if path == HERE/'index.json':
                continue
            if path.is_symlink() or not path.is_file() or path.suffix == '.pyc':
                raise ValueError(f'Unexpected retained artifact: {path}')
            paths.add(path)
    paths.update([REPORT/'manifest.json', REPORT/'cases.json'])
    for case in read(REPORT/'cases.json')['cases']:
        paths.update(REPORT/'points'/(case['id']+suffix) for suffix in ('.dat','.json'))
    return {str(path.relative_to(REPORT)): identity(path) for path in sorted(paths)}

def inspect_live():
    sys.path.insert(0, str(ROOT/'tools'))
    import tirust_benchmark as bench
    import tirust_benchmark_campaign as campaign
    from tirust_benchmark_compiler import validate_compiler_equality
    if ROOT != bench.ROOT or digest(REPORT/'manifest.json') != EXPECTED_MANIFEST:
        raise ValueError('Wrong working directory or manifest')
    campaign.activate_campaign(bench, 'split-dp-terminal-1')
    campaign.verify_campaign(bench, stage='preparation-evidence-verification')
    config, manifest = read(REPORT/'cases.json'), read(REPORT/'manifest.json')
    status = read(HERE/'status.json')
    assert status['status'] == 'passed' and status['amplitude_evaluations'] == 0
    assert [stage['id'] for stage in status['stages']] == ['reuse-baseline','build-baseline','build-candidate']
    original_configs = read(HERE/'configuration-initial.json')
    for path, value in original_configs.items():
        assert digest(ROOT/path) == value['sha256'] and (ROOT/path).read_text() == value['text']
    for stage in status['stages']:
        assert stage['status'] == 'passed' and stage['returncode'] == 0
        assert digest(HERE/stage['id']/'console.log') == stage['console_sha256']
        for phase in ('before','after'):
            assert read(HERE/stage['id']/f'configuration-{phase}.json') == original_configs
    records = []
    for case in config['cases']:
        pair = {}
        meta = read(REPORT/'points'/(case['id']+'.json'))
        assert len(meta['points']) == 100
        assert digest(REPORT/'points'/(case['id']+'.dat')) == meta['sha256']
        for source in ('baseline','candidate'):
            output = bench.output_dir(source,case)
            directory = bench.subprocess_dir(source,case)
            retained = REPORT/'inputs'/source/case['id']
            generation = read(retained/'generation.json')
            compiled = read(retained/'compilation.json')
            assert digest(output/'generation.json') == digest(retained/'generation.json')
            assert digest(output/'compilation.json') == digest(retained/'compilation.json')
            for record in (generation,compiled):
                assert record['campaign_binding'] == {'id':'split-dp-terminal-1','manifest_sha256':EXPECTED_MANIFEST}
            campaign.validate_generation_environment(generation, required=True)
            bench.verify_worker_template(compiled,ROOT/'tools/tirust_benchmark_worker.f')
            assert compiled['numerical_build_from_sources'] is True
            assert bench.generated_sources(output) == compiled['sources']
            assert digest(directory/'check') == compiled['executable_sha256']
            assert bench.dependency_inputs(directory) == compiled['link_inputs_before'] == compiled['link_inputs_after']
            assert compiled['masses'] == meta['inputs']['incoming_masses']+meta['inputs']['masses']
            assert compiled['masses'] == [float(v) for v in (directory/'masses.dat').read_text().split()]
            assert digest(output/'Cards/param_card.dat') == generation['parameter_card_sha256'] == digest(retained/'param_card.dat')
            objects = bench.numerical_objects(output)
            if source == 'baseline':
                reused = generation['reused_from']
                assert compiled['full_numerical_rebuild_this_invocation'] is False
                assert reused['worker_only_relink_required'] is True
                assert {k:v for k,v in objects.items() if Path(k).name != 'check_sa.o'} == {k:v for k,v in reused['numerical_objects'].items() if Path(k).name != 'check_sa.o'}
                environment = reused['source_export_environment']
                environment_recorded = reused['source_export_environment_recorded']
            else:
                assert not generation.get('reused_from')
                assert compiled['full_numerical_rebuild_this_invocation'] is True
                assert generation['source_identity']['files_sha256'] == manifest['candidate']['exporter_identity']['files_sha256']
                command = REPORT/'commands'/f'candidate-{case["id"]}.mg5'
                assert digest(command) == generation['command_sha256']
                assert all(line.endswith(' --no_save') for line in command.read_text().splitlines() if line.startswith('set '))
                environment, environment_recorded = generation['export_environment'], True
            for name, checksum in compiled['build_logs'].items():
                assert digest(retained/f'compilation-{name}.log') == checksum
            pair[source] = compiled
            records.append({'source':source,'case':case['id'],'generation_sha256':digest(retained/'generation.json'),
                'compilation_sha256':digest(retained/'compilation.json'),'executable_sha256':digest(directory/'check'),
                'output_directory':str(output.relative_to(ROOT)), 'subprocess_directory':str(directory.relative_to(ROOT)),
                'numerical_objects':objects,'masses':compiled['masses'],'saved_points':100,
                'export_environment':environment,'export_environment_recorded':environment_recorded,
                'observed_exporter_revision':generation['source_identity']['revision'],
                'exporter_files_sha256':generation['source_identity']['files_sha256'],
                'default_chain':generation['default_chain'],
                'full_numerical_rebuild_this_invocation':compiled['full_numerical_rebuild_this_invocation'],
                'mass_setup_log_sha256':digest(WORK/'logs'/f'masses-{source}-{case["id"]}.log')})
        validate_compiler_equality(pair['baseline']['effective_compiler_flags'],pair['candidate']['effective_compiler_flags'])
    return {'schema':'madloop-tirust-completed-source-preparation-v1','campaign':'split-dp-terminal-1',
        'manifest_sha256':EXPECTED_MANIFEST,'completed_cases':11,'completed_builds':22,
        'registered_exporter_revision':manifest['candidate']['exporter_identity']['revision'],
        'observed_candidate_revisions':sorted({row['observed_exporter_revision'] for row in records if row['source']=='candidate'}),
        'candidate_exporter_file_hashes':sorted({row['exporter_files_sha256'] for row in records if row['source']=='candidate'}),
        'scope':'Source/compile/link/mass setup only; no amplitude or reducer calls; no numerical or timing acceptance',
        'amplitude_evaluations':0,'performance_acceptance_eligible':False,'records':records}

def freeze():
    if (HERE/'index.json').exists() or (HERE/'results.json').exists():
        raise ValueError('Preparation evidence already frozen')
    result = inspect_live()
    (HERE/'protocols').mkdir()
    for name in PROTOCOLS:
        shutil.copy2(ROOT/'tools'/name,HERE/'protocols'/name)
    for row in result['records']:
        retained = HERE/'mass-setup'/row['source']/row['case']
        retained.mkdir(parents=True)
        shutil.copy2(WORK/'logs'/f'masses-{row["source"]}-{row["case"]}.log',retained/'console.log')
        shutil.copy2(ROOT/row['subprocess_directory']/'masses.dat',retained/'masses.dat')
    (HERE/'generation-logs').mkdir()
    for case in read(REPORT/'cases.json')['cases']:
        shutil.copy2(WORK/'logs'/f'generate-candidate-{case["id"]}.log',HERE/'generation-logs'/f'{case["id"]}.log')
    write(HERE/'results.json',result)
    files = inventory()
    write(HERE/'index.json',{'schema':'madloop-tirust-completed-preparation-index-v1',
        'created_utc':datetime.now(timezone.utc).isoformat(),'root':'..',
        'files':files,'file_count':len(files),'total_bytes':sum(row['bytes'] for row in files.values()),
        'manifest_sha256':EXPECTED_MANIFEST,'completed_builds':22,'amplitude_evaluations':0,
        'performance_acceptance_eligible':False})

def verify(live=False):
    index = read(HERE/'index.json')
    actual = inventory()
    assert actual == index['files']
    assert len(actual) == index['file_count']
    assert sum(row['bytes'] for row in actual.values()) == index['total_bytes']
    if live:
        assert inspect_live() == read(HERE/'results.json')
    print(f'PASS {len(actual)} retained files; 22 source preparations; no amplitude qualification')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--live',action='store_true',help='Also rehash current ignored source/build/dependency artifacts; read-only, no native calls')
    args=parser.parse_args()
    if args.freeze:
        freeze()
    verify(args.live)

if __name__ == '__main__':
    main()
