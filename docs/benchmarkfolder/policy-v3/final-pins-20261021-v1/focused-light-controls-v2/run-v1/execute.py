"""Exactly nine root-authorized two-point low-multiplicity controls; no builds."""
import argparse,contextlib,copy,hashlib,json,math,os,signal,shutil,sys,time,traceback
from pathlib import Path
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'tools'),str(ROOT)]
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as policy
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_workers as workers
CPU=190

def read(path):return json.loads(Path(path).read_text())
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def bound(path):return {'path':policy.relative(path),'sha256':digest(path),'bytes':Path(path).stat().st_size}
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')

def verify_output(record):
    output=policy.path(record['output']);directory=next(output.glob('SubProcesses/P*_*'))
    if (bench.generated_source_manifest(output)!=record['sources'] or policy.library_hashes(output)!=record['libraries'] or
            pair.actual_runtime(output,record['sources'],8)!=record['runtime'] or digest(directory/'check')!=record['executable_sha256'] or
            digest(output/'Cards/param_card.dat')!=record['parameter_card_sha256']):raise ValueError('Bound compiled source, executable, full runtime or physical card changed')
    text=(directory/'check_sa.f').read_text();prefix=record['prefix']
    force='CALL '+prefix+'FORCE_STABILITY_CHECK('+('.FALSE.' if record['direct_mode_driver'] else '.TRUE.')+')'
    if text.count(force)!=1:raise ValueError('Bound actual driver public force switch changed')
    return output,directory

@contextlib.contextmanager
def owned_calls(report,spec):
    def interrupted(signum,_frame):raise SystemExit('Light control interrupted by signal '+str(signum))
    previous={sig:signal.getsignal(sig) for sig in(signal.SIGTERM,signal.SIGHUP)}
    try:
        for sig in previous:signal.signal(sig,interrupted)
        with pair.owned_preparation(report,label='light-controls'):
            original=bench.run
            def bounded_run(command,cwd,log,**kwargs):
                if kwargs or command[:4]!=['taskset','-c','190','./check'] or len(command)!=6 or command[-1]!='1':raise ValueError('Only the bound single traversal executable call is allowed')
                if str(command[4]) not in {str(policy.path(x['selected']['path'])) for x in spec['points'].values()}:raise ValueError('Unbound point file')
                return original(command,cwd,log,timeout=spec['target_timeout_seconds'])
            bench.run=bounded_run
            yield
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)

def parse_direct(directory,expected_dimensions,mode,point_count):
    if point_count != 2:raise ValueError("The declared light study requires exactly2 points")
    """Strict diagnostic110/120 reader, with real generated Born dimensions."""
    if mode not in (1,2,4,5):raise ValueError('Unknown public direct mode')
    dimensions=list(map(int,(directory/'dimensions.dat').read_text().split()))
    if dimensions!=list(expected_dimensions):raise ValueError('Direct result changed generated answer dimensions')
    allocated,loop,born=dimensions
    if allocated!=max(loop,born) or min(dimensions)<0 or born>loop:raise ValueError('Invalid direct answer dimensions')
    expected_status=110 if mode in (1,2) else 120
    rows=[]
    for line in (directory/'values.dat').read_text().splitlines():
        fields=line.split()
        if len(fields)!=8:raise ValueError('Malformed direct row')
        row={'point':int(fields[0]),'order':int(fields[1]),'status':int(fields[2]),'coefficients':list(map(float,fields[3:7])),'accuracy':float(fields[7])}
        if row['status']!=expected_status or row['accuracy']!=-1 or not all(math.isfinite(x) for x in row['coefficients']):raise ValueError('Wrong direct code/accuracy or nonfinite coefficients')
        rows.append(row)
    if [(r['point'],r['order']) for r in rows]!=[(p,o) for p in range(1,point_count+1) for o in range(loop+1)]:raise ValueError('Incomplete/reordered direct physical rows')
    histogram=[list(map(int,line.split())) for line in (directory/'timed_status.dat').read_text().splitlines()]
    if histogram!=[[expected_status,point_count]]:raise ValueError('Wrong direct evaluation histogram')
    return {'rows':rows,'status_counts':{str(expected_status):point_count},'timed_status_counts':{str(expected_status):point_count},'answer_dimensions':{'allocated':allocated,'loop_orders':loop,'born_orders':born},'direct_mode':mode,'stability_measurement':False,'automatic_rescue':False,'performance_acceptance_claim':False}

def preparation_preflight(spec):
    """Verify the real sealed preparation and all nine assets before mutation."""
    for record in spec['bound_inputs']:policy.check_file(record)
    for item in spec['points'].values():
        original=policy.check_file(item['source']).read_bytes().splitlines(keepends=True)
        if int(original[0])!=10 or b'2\n'+b''.join(original[1:9])!=policy.check_file(item['selected']).read_bytes():raise ValueError('Actual fixed point prefix changed')
    prepared=policy.verified(read(policy.check_file(spec['compiled_preparation'])),schema=None)
    for target in spec['schedule']:
        record=prepared['variants'][target['variant']];output,directory=verify_output(record)
        expected_command=['taskset','-c','190','./check',str(policy.path(spec['points'][target['points']]['selected']['path'])),'1']
        if (target['executable_sha256']!=record['executable_sha256'] or target['source_manifest_sha256']!=record['sources']['files_sha256'] or
                target['expected_dimensions']!=record['dimensions'] or target['command']!=expected_command or target['cwd']!=str(directory) or
                record['runtime']['identity_sha256']!=spec['cohort']['runtime_identity_sha256']):raise ValueError('Declared target differs from actual compiled executable/source/dimensions/command/runtime')
    return prepared


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--handoff',type=Path,required=True);parser.add_argument('--execute-numerical',action='store_true');args=parser.parse_args()
    spec_path=HERE/'specification.json';spec=read(spec_path)
    if spec['points_per_target']!=2 or len(spec['schedule'])!=9 or any(t['backend'] not in(6,8) or t['mode'] not in(-1,4,5) for t in spec['schedule']):raise ValueError('Only the exact nine two-point Ninja/NinjaRS targets are supported')
    expected={'root_authorized':True,'phase':spec['phase'],'cpu':CPU,'source_sha256':digest(Path(__file__)),'specification_sha256':digest(spec_path),'compiled_preparation_sha256':spec['compiled_preparation']['sha256']}
    if not args.execute_numerical or read(args.handoff)!=expected:raise ValueError('Matching root handoff and explicit execution flag required')
    if os.sched_getaffinity(0)!={CPU} or any(os.environ.get(k)!='1' for k in pair.THREADS):raise ValueError('CPU190 and all four thread settings1 required')
    prepared=preparation_preflight(spec)
    if bench.content_digest(bench.generator_tree_state())!=spec['cohort']['generator_identity_sha256']:raise ValueError('Final generator changed before controls')
    run=policy.path(spec['report']);run.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),run/'execute.py');shutil.copy2(spec_path,run/'input-specification.json');shutil.copy2(args.handoff,run/'root-handoff.json')
    bench.WORK=policy.path(spec['compiled_work']);base_config=read(policy.check_file(spec['configuration']));base_config['points_per_case']=2
    results={};status={'status':'incomplete','completed':[],'scope':'Nine two-point diagnostics only; no performance acceptance'};bench.dump(run/'status.json',status)
    try:
        with owned_calls(run,spec):
            for index,target in enumerate(spec['schedule']):
                label=target['id'];record=prepared['variants'][target['variant']];output,directory=verify_output(record)
                if record['runtime']['identity_sha256']!=spec['cohort']['runtime_identity_sha256']:raise ValueError('Compiled runtime not final a757')
                point=policy.check_file(spec['points'][target['points']]['selected']);case=copy.deepcopy(record['case'])
                case['madloop_parameters'].update(CTModeRun=target['mode'],UseQPIntegrandForNinja=True,UseQPIntegrandForNinjaRS=True)
                job=run/'jobs'/label;job.mkdir(parents=True);bench.REPORT=job
                raw=job/'raw';raw.mkdir();log=bench.WORK/'logs'/('light-'+label+'.log');started=time.monotonic()
                for name in(*workers.OUTPUTS,'dimensions.dat'):(directory/name).unlink(missing_ok=True)
                try:
                    if target['mode']==-1:
                        result=bench.measure('candidate',case,target['backend'],point,1,index,CPU,config=base_config,persist=False)
                    else:
                        settings=bench.configure('candidate',case,target['backend'],config=base_config)
                        if settings['ctmoderun']!=target['mode'] or settings['selected_backend_qp_integrand'] is not True:raise ValueError('Wrong direct mode or selected trueQP integrand')
                        workers.validate_param_card(directory,(output/'Cards/param_card.dat').read_bytes())
                        bench.run(['taskset','-c','190','./check',str(point),'1'],directory,log)
                        result=parse_direct(directory,record['dimensions'],target['mode'],2)
                        result.update(shared_numerical_settings=settings)
                    if result['answer_dimensions']!={'allocated':record['dimensions'][0],'loop_orders':record['dimensions'][1],'born_orders':record['dimensions'][2]}:raise ValueError('Actual answer dimensions differ from compiled parent')
                    result.update(label=label,variant=target['variant'],backend=target['backend'],mode=target['mode'],points_sha256=digest(point),parameter_card_sha256=digest(output/'Cards/param_card.dat'),executable_sha256=digest(directory/'check'),runtime_identity_sha256=record['runtime']['identity_sha256'],process_wall_seconds=time.monotonic()-started,performance_acceptance_claim=False)
                    results[label]=result;write(job/'result.json',result)
                finally:
                    for name in(*workers.OUTPUTS,'dimensions.dat'):
                        path=directory/name
                        if path.is_file():shutil.copy2(path,raw/name)
                    for path in(output/'Cards/param_card.dat',output/'Cards/MadLoopParams.dat'):
                        if path.is_file():shutil.copy2(path,raw/path.name)
                    actual_log=log if target['mode']!=-1 else bench.WORK/'logs'/f"run-candidate-{case['id']}-{target['backend']}-{index}.log"
                    if actual_log.is_file():shutil.copy2(actual_log,raw/'run.log')
                verify_output(record)
                status['completed'].append(label);bench.dump(run/'status.json',status);print(label+' completed '+json.dumps(result['status_counts']),flush=True)
        groups={name:{target['id']:results[target['id']]['rows'] for target in spec['schedule'] if target['group']==name} for name in('state','complex','direction')}
        normal=spec['complex_normal_references']
        normal_points=policy.path(normal['points']['path']).read_bytes().splitlines(keepends=True)
        if b'2\n'+b''.join(normal_points[1:9])!=policy.path(spec['points']['control-complex']['selected']['path']).read_bytes():raise ValueError('Complex normal prefix differs')
        for backend in(6,8):
            reference=read(policy.check_file(normal[str(backend)]))
            if reference['parameter_card_sha256']!=results[f'complex-qp-{backend}']['parameter_card_sha256']:raise ValueError('Complex normal/QP physical cards differ')
            groups['complex'][f'normal-{backend}']=[r for r in reference['rows'] if r['point']<=2]
        comparisons={name:policy.compare_rows(rows) for name,rows in groups.items()}
        if not all(record['passed'] for record in comparisons.values()):raise ValueError('Light focused numerical comparison failed')
        if bench.content_digest(bench.generator_tree_state())!=spec['cohort']['generator_identity_sha256']:raise ValueError('Generator changed during controls')
        status.update(status='passed',comparisons=comparisons,physical_component_comparisons=sum(value['pairs']*len(next(iter(groups[name].values())))*4 for name,value in comparisons.items()),normal_HH_anchor='Pending actual light HH10 cal first2 read-only comparison; not claimed here',scope='Only declared two-state/context, complex trueQP and HHQP forward/reverse checks; no ten-state restoration or physical-rotation claim')
        bench.dump(run/'status.json',status)
    except BaseException as error:
        status.update(status='failed',error=repr(error),traceback=traceback.format_exc());bench.dump(run/'status.json',status);raise

if __name__=='__main__':main()
