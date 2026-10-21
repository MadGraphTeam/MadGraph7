"""Bounded actual parity: qcd1, legacy 1/6/7 only, CPU12, no timing claims."""
from pathlib import Path
import hashlib,json,os,re,signal,struct,sys,time
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
sys.path.insert(0,str(ROOT/'tools'))
import ninjars_dp_stability as dp
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as v2
import ninjars_benchmark_policy_v3 as v3
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_workers as workers
from madgraph.various.banner import MadLoopParam
HERE=Path(__file__).resolve().parent
ORIGIN=ROOT/'.benchmark-work/policy-v3/final-pins-20261021-v1/generation/outputs/candidate/qcd-1'
COHORT=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/candidate-cohort.json'
ANCHOR=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/generation/inputs/candidate/qcd-1/compiled-sources.json'
POINTS=ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/generation/points/qcd-1.dat'
FROZEN={'tools/ninjars_dp_stability.py':'b0fa3802df8e195a0d23474afd7211bbd3cfadaa872d4f41d9bbbb4c4ee14daa',
        'tools/ninjars_dp_stability_driver.f':'3d8d06325dd955408771b94873ee34ad438c0ba4a459e48bbfbbf69ed33d7dc7'}
SYMBOLS=('ninjars_context_create','ninjars_evaluate_dp','ninjars_evaluate_dp_compensated','ninjars_evaluate_mp','ninjars_context_destroy')

def bound(path):return {'path':str(path.relative_to(ROOT)),'sha256':dp.digest(path)}
def stop(signum,frame):raise KeyboardInterrupt('signal '+str(signum))
def run_owned(command,cwd,label,env):
    start=time.monotonic()
    with (HERE/(label+'.stdout')).open('xb') as out,(HERE/(label+'.stderr')).open('xb') as err:
        proc=workers.subprocess.Popen(command,cwd=cwd,env=env,stdout=out,stderr=err,start_new_session=True)
        code=proc.wait(timeout=180)
    record={'command':command,'cwd':str(cwd.relative_to(ROOT)),'exit_code':code,
            'seconds_diagnostic_only':time.monotonic()-start,'pid':proc.pid,
            'reaped':not Path('/proc',str(proc.pid)).exists(),'stdout':bound(HERE/(label+'.stdout')),
            'stderr':bound(HERE/(label+'.stderr'))}
    dp.dump(HERE/(label+'.json'),record)
    if code:raise RuntimeError(label+' returned '+str(code))
    return record

def bits(value):
    if value is None:raise ValueError('Nonfinite coefficient in parity control')
    return struct.pack('!d',value).hex()

def main():
    if os.sched_getaffinity(0)!={12}:raise ValueError('Only released CPU12 is authorized')
    if (HERE/'started.json').exists():raise ValueError('Private campaign already started')
    for name,sha in FROZEN.items():
        if dp.digest(ROOT/name)!=sha:raise ValueError('Frozen producer changed')
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):signal.signal(sig,stop)
    cohort=v3.validate_cohort(COHORT)
    anchor=json.loads(ANCHOR.read_text())
    if bench.generated_source_manifest(ORIGIN)!=anchor:raise ValueError('Original source anchor differs')
    origin_runtime=pair.actual_runtime(ORIGIN,anchor,8)
    if origin_runtime['identity_sha256']!=cohort['runtime_identity_sha256']:raise ValueError('Wrong final runtime')
    if int(POINTS.read_text().splitlines()[0])!=100:raise ValueError('Expected100 original fixed points')
    env=workers.environment(HERE)
    env.update({x:'1' for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS')})
    dp.dump(HERE/'started.json',{'scope':'qcd1 selectors1/6/7 parity only; no8 numericalentry',
        'controller_pid':os.getpid(),'cpu':[12],'cohort':bound(COHORT),'original_manifest':bound(ANCHOR),
        'points':bound(POINTS),'guard_source':bound(HERE/'guard.c'),'runner':bound(Path(__file__).resolve()),
        'producer_sources':FROZEN,'thread_environment':{k:env[k] for k in pair.THREADS},
        'physical_campaign_complete':False,'timing_claimed':False})
    guard=HERE/'guard.o'
    results={}
    with pair.owned_preparation(HERE,label='dp-qcd1-legacy-parity'):
        run_owned(['gcc','-O2','-std=c11','-Wall','-Werror','-I',str(ROOT/'vendor/ninjars'),'-c',str(HERE/'guard.c'),'-o',str(guard)],ROOT,'guard-compile',env)
        for tool in (1,6,7):
            observations={}
            for variant in ('plain','observed'):
                label=f'{tool}-{variant}';output=HERE/'outputs'/label;output.parent.mkdir(exist_ok=True)
                workers.subprocess.run(['cp','-a','--reflink=auto',str(ORIGIN),str(output)],check=True)
                for link in output.rglob('*'):
                    if link.is_symlink() and link.resolve().is_relative_to(ORIGIN):
                        replacement=output/link.resolve().relative_to(ORIGIN);link.unlink();link.symlink_to(os.path.relpath(replacement,link.parent))
                directory=next(output.glob('SubProcesses/P*_*'))
                original=(directory/'CT_interface.f').read_text()
                instrumented,receipt=dp.instrument_interface(original)
                if variant=='observed':(directory/'CT_interface.f').write_text(instrumented)
                (directory/'check_sa.f').write_text(dp.render_driver(receipt['prefix'],3,1))
                for name in ('CT_interface.o','check_sa.o','check'):(directory/name).unlink(missing_ok=True)
                card=MadLoopParam(str(output/'Cards/MadLoopParams.dat'))
                for key,value in {'MLReductionLib':str(tool),'CTModeRun':1,'CTModeInit':1,
                    'COLLIERComputeUVpoles':True,'COLLIERComputeIRpoles':True,'COLLIERGlobalCache':0,
                    'COLLIERUseInternalStabilityTest':False,'MLStabThres':1e-9,'COLLIERRequiredAccuracy':1e-12}.items():card[key]=value
                for file in [output/'Cards/MadLoopParams.dat',*output.glob('SubProcesses/**/MadLoopParams.dat')]:
                    card.write(str(file),template=str(ROOT/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))
                links=str(guard)+' -L$(LIBDIR) -ldhelas -lmodel $(LINK_LOOP_LIBS) $(LDFLAGS) '+ ' '.join('-Wl,--wrap='+symbol for symbol in SYMBOLS)
                build=run_owned(['make','-j1','check','LINKLIBS='+links],directory,label+'-compile',env)
                manifest=bench.generated_source_manifest(output)
                delta=sorted(v2.delta(anchor['files'],manifest['files']))
                allowed=[str((directory/'check_sa.f').relative_to(output))]
                if variant=='observed':allowed.append(str((directory/'CT_interface.f').relative_to(output)))
                if delta!=sorted(allowed):raise ValueError('Unexpected generated-source delta')
                runtime=pair.actual_runtime(output,manifest,tool)
                if runtime!=origin_runtime:raise ValueError('Changed base full reducer runtime')
                physical=dp.digest(output/'Cards/param_card.dat')
                if physical!=dp.digest(ORIGIN/'Cards/param_card.dat'):raise ValueError('Changed physical card')
                raw=HERE/(label+'.raw');guardlog=HERE/(label+'.guard')
                runenv=dict(env,NINJARS_GUARD_LOG=str(guardlog))
                execution=run_owned([str(directory/'check'),str(POINTS),str(tool),'-1',str(raw)],directory,label+'-execute',runenv)
                parsed=dp.parse_raw(raw,100,tool,1)
                if not parsed['complete']:raise ValueError('Incomplete raw point stream')
                expected='GUARD_FINAL create=0 dp=0 compensated=0 mp=0 destroy=1 nonnull=0'
                lines=guardlog.read_text().splitlines()
                if lines!=[expected.replace('GUARD_FINAL','NULL_TEARDOWN_FORWARDED'),expected]:raise ValueError('Forbidden numericalentry or missingNULLteardown')
                if bench.generated_source_manifest(output)!=manifest or pair.actual_runtime(output,manifest,tool)!=runtime:raise ValueError('Source/runtime changed during observation')
                entry={'tool':tool,'variant':variant,'build':build,'execution':execution,'raw':bound(raw),
                    'guard':bound(guardlog),'guard_object':bound(guard),'additional_link_flags':links,
                    'compiled_manifest':manifest,'runtime':runtime,'original_source_unchanged':bench.generated_source_manifest(ORIGIN)==anchor,
                    'driver_sha256':dp.digest(directory/'check_sa.f'),'interface_sha256':dp.digest(directory/'CT_interface.f'),
                    'executable_sha256':dp.digest(directory/'check'),'physical_card_sha256':physical,
                    'madloop_card_sha256':dp.digest(output/'Cards/MadLoopParams.dat'),'instrumentation':receipt if variant=='observed' else None,
                    'point_observations':[]}
                for key,obs in sorted(parsed['observations'].items()):
                    mode,code,fpdp,fpqp,nearest,*counts=obs['metadata']
                    if fpqp or counts[3] or counts[13] or counts[14]:raise ValueError('QP/compensatedDP is outside this control')
                    if any(v for idx,v in enumerate(counts[4:12],1) if idx!=tool):raise ValueError('Unexpected actual backend')
                    if variant=='observed' and sum(counts[4:12])!=counts[0]:raise ValueError('Invalid dispatchcount')
                    if variant=='plain' and any(counts[:15]):raise ValueError('Uninstrumentedcopyunexpectedrecordercall')
                    entry['point_observations'].append({'point':key[0],'transform':key[1],
                        'metadata':obs['metadata'],'rows':obs['rows']})
                dp.dump(HERE/(label+'-result.json'),entry)
                observations[variant]=parsed;results[label]=bound(HERE/(label+'-result.json'))
            compared=0;zeros=0
            for key,left in observations['plain']['observations'].items():
                right=observations['observed']['observations'][key]
                if left['metadata'][:5]!=right['metadata'][:5]:raise ValueError('Modes/status/commonFPE/rounding differ')
                if left['metadata'][-3:]!=right['metadata'][-3:]:raise ValueError('Hardwareflags differ')
                for order,row in left['rows'].items():
                    other=right['rows'][order]
                    if [bits(v) for v in row['values']]!=[bits(v) for v in other['values']]:raise ValueError('Numeric or internalaccuracy parity differs')
                    compared+=4;zeros+=sum(v==0. for v in row['values'][:4])
            dp.dump(HERE/(str(tool)+'-parity.json'),{'tool':tool,'points':100,'transformations':3,'orders_including_total':2,
                'physical_coefficients_compared':compared,'zero_coefficients_compared':zeros,'f64_bitexact':True,
                'status_commonFPE_rounding_and_hardware_flags_exact':True,'instrumentation_failure_counters_retained':True,
                'null_teardown_forwarded_both':True,'numericalNRScontext_creation_or_evaluation':False})
    dp.dump(HERE/'complete.json',{'schema':'qcd1-legacy-dp-recorder-parity-v1','backends':[1,6,7],
        'physical_points':100,'transformations':3,'results':results,'all_three_backend_parities_pass':True,
        'NinjaRS8_parity':'HELD: not evaluated','four_tool_campaign_complete':False,'timing_or_perf_claimed':False,
        'controller_sources':FROZEN,'original_source_unchanged':bench.generated_source_manifest(ORIGIN)==anchor})

if __name__=='__main__':
    try:main()
    except BaseException as error:
        dp.dump(HERE/'failed.json',{'error_type':type(error).__name__,'error':str(error),'four_tool_campaign_complete':False})
        raise
