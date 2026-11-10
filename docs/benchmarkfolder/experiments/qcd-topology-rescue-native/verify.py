#!/usr/bin/env python3
"""Read-only verification of the complete private diagnostic evidence."""
import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path.cwd().resolve()
HERE = Path(__file__).resolve().parent
ARMS = ('topology-b','original-c','topology-c')
CONTROLLER = '97e5c6e16d52c166f3344921149ba0909cc844ab2a099f7c016100f8a95073cf'


def require(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def inventory(directory, filename, exact=False):
    entries = read(directory/filename)['files']
    for name, entry in entries.items():
        path = directory/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink(), 'Unsafe path: '+name)
        raw = path.read_bytes()
        require(len(raw)==entry['bytes'] and hashlib.sha256(raw).hexdigest()==entry['sha256'], 'Inventory changed: '+name)
        require(not raw.startswith(b'\x7fELF') and path.suffix not in ('.o','.so','.a','.mod','.pyc'), 'Native/generated artifact in report: '+name)
    if exact:
        all_paths = list(directory.rglob('*'))
        require(not any(p.is_symlink() for p in all_paths), 'Report tree contains a symlink.')
        actual = {str(p.relative_to(directory)) for p in all_paths if p.is_file()}
        require(actual == set(entries)|{filename}, 'Portable inventory is not exact.')
    return entries


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live',action='store_true')
    args=parser.parse_args()
    entries=inventory(HERE,'evidence-index.json',exact=True)
    require(sha(HERE/'source-index.json')==CONTROLLER,'Frozen controller changed.')
    inventory(HERE,'source-index.json')
    from common import OLD, OLD_SHA, DESIGN, DESIGN_SHA
    from run import compare_rows, compare_traces, load_decoder
    require(sha(OLD/'evidence-index.json')==OLD_SHA and sha(DESIGN/'evidence-index.json')==DESIGN_SHA,'Frozen source/control package changed.')
    inventory(OLD,'evidence-index.json')
    inventory(DESIGN,'evidence-index.json')
    decoder=load_decoder()
    plans=read(HERE/'plans.json')
    live={}

    def add(mapping):
        for name, digest in mapping.items():
            if isinstance(digest,dict):
                if 'bytes' in digest and (ROOT/name).is_file():
                    require((ROOT/name).stat().st_size==digest['bytes'],'Bound size mismatch: '+name)
                digest=digest['sha256']
            require(name not in live or live[name]==digest,'Conflicting live pin: '+name)
            live[name]=digest
            # Portable cross-package evidence is required even without --live.
            if name.startswith('docs/'):
                require(sha(ROOT/name)==digest,'Bound report changed: '+name)

    summaries={}
    originals={}
    for backend in (9,8):
        directory=OLD/'attempt1'/f'backend-{backend}'
        record=read(directory/'execution.json')
        originals[backend]=(record,decoder.load_trace(directory/'trace.dat',directory/'observed.log'))
    for arm in ARMS:
        directory=HERE/'attempt1'/arm
        build=read(directory/'build.json')
        require(build['complete'] and build['protected_inputs_unchanged'] and build['private_copies_unchanged'] and not build['native_executed'],'Invalid build completion.')
        require(build['controller_index_sha256']==CONTROLLER and build['cpu']==11 and build['arm']==arm,'Build identity differs.')
        planned_objects=[str(Path(p).relative_to(ROOT)) for p in plans[arm]['linked_objects']]
        require(build['commands']==plans[arm]['commands'] and build['linked_objects']==planned_objects,'Build commands/order differ from frozen plan.')
        require([c['command'] for c in build['command_records']]==build['commands'] and all(c['returncode']==0 for c in build['command_records']),'Missing/failed compiler command.')
        require(build['protected_inputs_before']==build['protected_inputs_after'],'Build pin drift.')
        require(build['private_copies_expected']==build['private_copies_after'],'Compiled copy drift.')
        for field in ('protected_inputs_before','private_copies_expected','linked_objects_sha256','artifacts_sha256'):
            add(build[field])
        for backend in (9,8):
            output=directory/f'backend-{backend}'
            record=read(output/'execution.json')
            require(record['complete'] and record['native_executed'] and record['returncode']==0,'Failed native replay.')
            require((record['cpu'],record['thread_count'],record['timeout_per_process_seconds'],record['automatic_retry_count'])==(11,1,180,0),'Protocol drift.')
            require(record['wall_seconds']<=180 and not record['formal_performance_eligible'] and not record['physics_accepted'],'Incorrect qualification claim.')
            require(record['build_sha256']==sha(directory/'build.json') and record['controller_index_sha256']==CONTROLLER,'Wrong native build binding.')
            for prefix in ('protected_inputs','runtime_sha256','resource_sha256'):
                require(record[prefix+'_unchanged'] and record[prefix+'_before']==record[prefix+'_after'],'Native pin drift: '+prefix)
                add(record[prefix+'_before'])
            old, old_trace=originals[backend]
            require(record['runtime_sha256_before']==old['runtime_sha256_before'],'Runtime differs from control.')
            for name,prefix in (('observed.log','log'),('trace.dat','trace'),('ldd.txt','ldd')):
                path=output/name
                require(sha(path)==record[prefix+'_sha256'] and path.stat().st_size==record[prefix+'_bytes'],'Output changed: '+str(path))
            rows=compare_rows((output/'observed.log').read_text(),(OLD/'attempt1'/f'backend-{backend}'/'observed.log').read_text())
            trace=decoder.load_trace(output/'trace.dat',output/'observed.log')
            comparison=compare_traces(trace,old_trace,backend)
            require(all(record[key]==value for key,value in rows.items()),'Row-comparison record changed.')
            require(all(record[key]==value for key,value in comparison.items()),'Trace-comparison record changed.')
            stress=trace['points'][1]
            require(Counter((c['ctmode'],c['group']) for c in stress['calls'])==Counter((m,g) for m in (4,5) for g in range(1,117)),'Incomplete group/direction coverage.')
            expected_code=322 if backend==8 and arm.startswith('topology') else 420
            require([p['host_code'] for p in trace['points']]==[210+backend,expected_code,210+backend],'Unexpected host status.')
            require(all(c['numerical_summaries']['NUM_ORIGINAL']['all_finite'] for c in stress['calls']),'Nonfinite numerator.')
            summaries[arm+':'+str(backend)]=trace['summary']
    analyses=[HERE/'analysis1/matrices/summary.json',HERE/'analysis1/adjudication.json',HERE/'analysis1/guard-source-audit.json']
    for path in analyses:
        add(read(path)['bindings'])
    for name in ('handoff','mass-source-audit'):
        analysis=read(HERE/'analysis1'/f'{name}.json')
        add(analysis['inputs'])
        add({analysis['script']['path']:analysis['script']['sha256']})
        if 'source_inputs' in analysis:
            add(analysis['source_inputs'])
        require(analysis['offline_analysis_complete'] and not analysis['physics_accepted'],'Incomplete/offscope analysis.')
    handoff=read(HERE/'analysis1/handoff.json')
    require(handoff['all_handoff_requirements_pass'] and handoff['all_topology_only_input_comparisons_pass'],'QP handoff mismatch.')
    guard=read(HERE/'analysis1/guard-source-audit.json')
    require(guard['summary']['exponent_possibility_counts']=={'3':11,'4':20,'3,4':20},'Guard count mismatch.')
    require(guard['summary']['all_powered_candidates_below_gate'] and guard['summary']['all_supplied_and_cartesian_determinants_nonzero'],'Guard/determinant mismatch.')
    norm=read(HERE/'analysis1/adjudication.json')['normalization']
    require(Decimal(norm['topology-b:8']['relative_error'])>Decimal('3e-6') and Decimal(norm['topology-c:8']['relative_error'])<Decimal('1e-14'),'Double-pole adjudication mismatch.')
    require(all(not row['independent_finite_single_truth_available'] for row in norm.values()),'Unexpected physical truth claim.')
    if args.live:
        for name,digest in live.items():
            require(sha(ROOT/name)==digest,'Live protected input changed: '+name)
    print(json.dumps({'verification':'pass','portable_files':len(entries),'live_inputs_checked':len(live) if args.live else 0,
                      'builds':3,'native_replays':6,'native_executed_by_verifier':False,
                      'complete_traces':summaries,'physical_amplitude_accepted':False},sort_keys=True))


if __name__=='__main__':
    main()
