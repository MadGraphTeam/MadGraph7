"""Bounded native-DP NinjaRS8 qcd1 recorder parity, CPU12; no timing claims."""
from pathlib import Path
from collections import Counter
import hashlib, importlib.util, json, math, os, re, signal, struct, sys, time

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'madgraph/__init__.py').is_file())
HERE = Path(__file__).resolve().parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT/'tools'))
FROZEN_DP = HERE/'ninjars_dp_stability_frozen_b0fa.py'
DP_SHA256 = 'b0fa3802df8e195a0d23474afd7211bbd3cfadaa872d4f41d9bbbb4c4ee14daa'
if hashlib.sha256(FROZEN_DP.read_bytes()).hexdigest() != DP_SHA256:
    raise ValueError('Private source transformation/raw parser snapshot changed')
loader = importlib.util.spec_from_file_location('nrs8_qcd1_frozen_dp', FROZEN_DP)
dp = importlib.util.module_from_spec(loader)
loader.loader.exec_module(dp)
dp.ROOT = ROOT
import ninjars_benchmark as bench
import ninjars_benchmark_policy_v2 as v2
import ninjars_benchmark_policy_v3 as v3
import ninjars_benchmark_pair_v3 as pair
import ninjars_benchmark_workers as workers
from madgraph.various.banner import MadLoopParam

ORIGIN = ROOT/'.benchmark-work/policy-v3/final-pins-20261021-v1/generation/outputs/candidate/qcd-1'
COHORT = ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/candidate-cohort.json'
ANCHOR = ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/generation/inputs/candidate/qcd-1/compiled-sources.json'
POINTS = ROOT/'docs/benchmarkfolder/policy-v3/final-pins-20261021-v1/generation/points/qcd-1.dat'
DRIVER_SHA256 = '3d8d06325dd955408771b94873ee34ad438c0ba4a459e48bbfbbf69ed33d7dc7'
HEADER_SHA256 = '62e8237316e80b63264de3fc031f118e242e62651f1cec9c6d1c40cca0aace8c'
GUARD_SHA256 = '88cfb41cf17a545665e30563c39f0bd58ae9b9c0075203e0e5c27531d5e0c06c'
HELPER_SHA256 = {
    'ninjars_benchmark.py':'60ed8ae57eb15e6fff0133fad07330882f179f955aa08693f0c4f3b02815b237',
    'ninjars_benchmark_policy_v2.py':'defdfef693cee50e8bb000a4393f507799d7eb0b4da5275e23e18f893678cd6a',
    'ninjars_benchmark_policy_v3.py':'2fdd45f6c5b2e485d04f7ba81a1a13b38ec70017f186c8d72e20b5d574a30b8e',
    'ninjars_benchmark_pair_v3.py':'1e5fb83daa6021fc7c74350df60d16f64204a93887c7e511ac3347a513e78484',
    'ninjars_benchmark_workers.py':'251c49aae30a5471ef40ea6e140a8c9572417d01b1dcfc0c98daabafece5c9ca',
}
SYMBOLS = ('ninjars_context_create','ninjars_evaluate_dp','ninjars_evaluate_dp_compensated',
           'ninjars_evaluate_mp','ninjars_context_destroy')
TOOLS = (8,)

def bound(path):
    return {'path':str(path.relative_to(ROOT)), 'sha256':dp.digest(path)}

def stop(signum, frame):
    raise KeyboardInterrupt('signal '+str(signum))

def run_owned(command, cwd, label, env):
    """The existing pair registry owns every child, including failure windows."""
    start = time.monotonic()
    proc, code, failure = None, None, None
    try:
        with (HERE/(label+'.stdout')).open('xb') as out, (HERE/(label+'.stderr')).open('xb') as err:
            proc = workers.subprocess.Popen(command, cwd=cwd, env=env, stdout=out, stderr=err,
                                            start_new_session=True)
            code = proc.wait(timeout=180)
    except BaseException as error:
        failure = {'error_type':type(error).__name__, 'error':str(error)}
        raise
    finally:
        record = {'command':command, 'cwd':str(cwd.relative_to(ROOT)), 'exit_code':code,
            'seconds_diagnostic_only':time.monotonic()-start,
            'pid':proc.pid if proc is not None else None,
            'reaped_at_record':proc is not None and not Path('/proc',str(proc.pid)).exists(),
            'failure':failure,
            'stdout':bound(HERE/(label+'.stdout')), 'stderr':bound(HERE/(label+'.stderr'))}
        dp.dump(HERE/(label+'.json'), record)
    if code:
        raise RuntimeError(label+' returned '+str(code))
    return record

def bits(value):
    if value is None or not math.isfinite(value):
        raise ValueError('Nonfinite coefficient in parity control')
    return struct.pack('!d',value).hex()

def parse_guard(path):
    """Check every native boundary/status; never reinterpret an error as success."""
    names = ('create','dp','compensated','mp','destroy','nonnull')
    pattern = re.compile(r'(\w+)(?: status=(-?\d+))?'
        + ''.join(' '+name+r'=(\d+)' for name in names))
    lines = path.read_text().splitlines()
    previous = [0]*6
    native_statuses = Counter()
    lifecycle_statuses = Counter()
    if len(lines) < 5:
        raise ValueError('Missing guard boundaries/native calls')
    for index, line in enumerate(lines):
        match = pattern.fullmatch(line)
        if not match:
            raise ValueError('Malformed/forbidden guard event: '+line)
        event = match[1]
        status = int(match[2]) if match[2] is not None else None
        counts = list(map(int, match.groups()[2:]))
        if event in ('GUARD_BEGIN','GUARD_FINAL'):
            if status is not None or counts != previous:
                raise ValueError('Inconsistent guard boundary counts')
            if (event == 'GUARD_BEGIN' and index != 0 or
                    event == 'GUARD_FINAL' and index != len(lines)-1):
                raise ValueError('Guard boundaries are missing or reordered')
        else:
            if index in (0,len(lines)-1) or status is None or not -(2**31) <= status < 2**31:
                raise ValueError('Missing native return status/boundary')
            expected = list(previous)
            if event == 'CREATE_RETURN':
                expected[0] += 1
                lifecycle_statuses['create:'+str(status)] += 1
            elif event == 'NATIVE_DP_RETURN':
                expected[1] += 1
                native_statuses[str(status)] += 1
            elif event == 'DESTROY_RETURN':
                expected[4] += 1
                expected[5] = counts[5]
                if counts[5]-previous[5] not in (0,1):
                    raise ValueError('Invalid non-NULL destroy counter')
                lifecycle_statuses['destroy:'+str(status)] += 1
            else:
                raise ValueError('Compensated/MP/unknown guard entry: '+event)
            if counts != expected:
                raise ValueError('Guard event/counter mismatch')
        previous = counts
    if not lines[0].startswith('GUARD_BEGIN ') or not lines[-1].startswith('GUARD_FINAL '):
        raise ValueError('Guard did not cover process initialization through teardown')
    if (not previous[0] or not previous[1] or previous[2] or previous[3]
            or previous[5] != previous[0] or previous[4] < previous[5]):
        raise ValueError('Missing native context/evaluation/teardown or forbidden precision')
    if any(not key.endswith(':0') for key in lifecycle_statuses):
        raise ValueError('Context lifecycle failure is retained, not accepted')
    return {'boundary_counts':dict(zip(names,previous)),
            'native_per_call_status_counts':dict(native_statuses),
            'lifecycle_status_counts':dict(lifecycle_statuses),
            'events':len(lines), 'no_compensated_or_mp_entry':True,
            'native_error_returns_retained':sum(v for k,v in native_statuses.items() if k!='0')}

def check_sources(cohort, anchor):
    if (dp.digest(FROZEN_DP) != DP_SHA256 or
            dp.digest(ROOT/'tools/ninjars_dp_stability_driver.f') != DRIVER_SHA256 or
            dp.digest(ROOT/'vendor/ninjars/ninjars.h') != HEADER_SHA256 or
            dp.digest(HERE/'guard.c') != GUARD_SHA256 or
            any(dp.digest(ROOT/'tools'/name) != value for name,value in HELPER_SHA256.items())):
        raise ValueError('Frozen parser/transformation/driver/header changed')
    if bench.content_digest(bench.generator_tree_state()) != cohort['generator_identity_sha256']:
        raise ValueError('Frozen generator changed')
    if bench.generated_source_manifest(ORIGIN) != anchor:
        raise ValueError('Original source anchor differs')

def main():
    if Path.cwd() != ROOT or os.sched_getaffinity(0) != {12}:
        raise ValueError('Only repository cwd and released CPU12 are authorized')
    if (HERE/'started.json').exists():
        raise ValueError('Preserve the previous campaign attempt')
    for sig in (signal.SIGTERM,signal.SIGHUP,signal.SIGINT):
        signal.signal(sig,stop)
    cohort = v3.validate_cohort(COHORT)
    anchor = json.loads(ANCHOR.read_text())
    check_sources(cohort,anchor)
    origin_runtime = pair.actual_runtime(ORIGIN,anchor,8)
    if origin_runtime['identity_sha256'] != cohort['runtime_identity_sha256']:
        raise ValueError('Wrong final runtime')
    if int(POINTS.read_text().splitlines()[0]) != 100:
        raise ValueError('Expected100 original fixed points')
    env = workers.environment(HERE)
    env.update({x:'1' for x in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','RAYON_NUM_THREADS','MKL_NUM_THREADS')})
    dp.dump(HERE/'started.json', {'scope':'qcd1 selector8 nativeDP parity only; compensatedDP/MP forbidden from startup',
        'controller_pid':os.getpid(),'cpu':[12],'cohort':bound(COHORT),'original_manifest':bound(ANCHOR),
        'points':bound(POINTS),'guard_source':bound(HERE/'guard.c'),'runner':bound(Path(__file__).resolve()),
        'frozen_dp_snapshot':bound(FROZEN_DP),'public_fortran_driver':bound(ROOT/'tools/ninjars_dp_stability_driver.f'),
        'abi_header':bound(ROOT/'vendor/ninjars/ninjars.h'),
        'thread_environment':{k:env[k] for k in pair.THREADS},
        'physical_campaign_complete':False,'timing_claimed':False,'qualification_performed':False})
    guard = HERE/'guard.o'
    results, guard_observations, entries = {}, {}, {}
    with pair.owned_preparation(HERE,label='dp-qcd1-nrs8-parity'):
        run_owned(['gcc','-O2','-std=c11','-Wall','-Werror','-I',str(ROOT/'vendor/ninjars'),
            '-c',str(HERE/'guard.c'),'-o',str(guard)],ROOT,'guard-compile',env)
        for tool in TOOLS:
            observations = {}
            for variant in ('plain','observed'):
                check_sources(cohort,anchor)
                label = f'{tool}-{variant}'
                output = HERE/'outputs'/label
                output.parent.mkdir(exist_ok=True)
                workers.subprocess.run(['cp','-a','--reflink=auto',str(ORIGIN),str(output)],check=True)
                for link in output.rglob('*'):
                    if link.is_symlink() and link.resolve().is_relative_to(ORIGIN):
                        replacement = output/link.resolve().relative_to(ORIGIN)
                        link.unlink(); link.symlink_to(os.path.relpath(replacement,link.parent))
                directory = next(output.glob('SubProcesses/P*_*'))
                original = (directory/'CT_interface.f').read_text()
                instrumented,receipt = dp.instrument_interface(original)
                if variant == 'observed':
                    (directory/'CT_interface.f').write_text(instrumented)
                (directory/'check_sa.f').write_text(dp.render_driver(receipt['prefix'],3,1))
                for name in ('CT_interface.o','check_sa.o','check'):
                    (directory/name).unlink(missing_ok=True)
                card = MadLoopParam(str(output/'Cards/MadLoopParams.dat'))
                for key,value in {'MLReductionLib':str(tool),'CTModeRun':1,'CTModeInit':1,
                    'COLLIERComputeUVpoles':True,'COLLIERComputeIRpoles':True,'COLLIERGlobalCache':0,
                    'COLLIERUseInternalStabilityTest':False,'MLStabThres':1e-9,
                    'COLLIERRequiredAccuracy':1e-12}.items():
                    card[key] = value
                for file in [output/'Cards/MadLoopParams.dat',*output.glob('SubProcesses/**/MadLoopParams.dat')]:
                    card.write(str(file),template=str(ROOT/'Template/loop_material/StandAlone/Cards/MadLoopParams.dat'))
                links = str(guard)+' -L$(LIBDIR) -ldhelas -lmodel $(LINK_LOOP_LIBS) $(LDFLAGS) '+ ' '.join('-Wl,--wrap='+symbol for symbol in SYMBOLS)
                build = run_owned(['make','-j1','check','LINKLIBS='+links],directory,label+'-compile',env)
                manifest = bench.generated_source_manifest(output)
                delta = sorted(v2.delta(anchor['files'],manifest['files']))
                allowed = [str((directory/'check_sa.f').relative_to(output))]
                if variant == 'observed':
                    allowed.append(str((directory/'CT_interface.f').relative_to(output)))
                if delta != sorted(allowed):
                    raise ValueError('Unexpected generated-source delta')
                runtime = pair.actual_runtime(output,manifest,tool)
                if runtime != origin_runtime:
                    raise ValueError('Changed base full reducer runtime')
                physical = dp.digest(output/'Cards/param_card.dat')
                if physical != dp.digest(ORIGIN/'Cards/param_card.dat'):
                    raise ValueError('Changed physical card')
                raw = HERE/(label+'.raw')
                guardlog = HERE/(label+'.guard')
                runenv = dict(env,NINJARS_GUARD_LOG=str(guardlog))
                execution = run_owned([str(directory/'check'),str(POINTS),str(tool),'-1',str(raw)],
                    directory,label+'-execute',runenv)
                parsed = dp.parse_raw(raw,100,tool,1)
                if not parsed['complete']:
                    raise ValueError('Incomplete raw point stream')
                guard_result = parse_guard(guardlog)
                guard_observations[variant] = guardlog.read_text().splitlines()
                if bench.generated_source_manifest(output) != manifest or pair.actual_runtime(output,manifest,tool) != runtime:
                    raise ValueError('Source/runtime changed during observation')
                entry = {'tool':tool,'variant':variant,'build':build,'execution':execution,'raw':bound(raw),
                    'guard':bound(guardlog),'guard_observations':guard_result,'guard_object':bound(guard),
                    'additional_link_flags':links,'compiled_manifest':manifest,'runtime':runtime,
                    'original_source_unchanged':bench.generated_source_manifest(ORIGIN)==anchor,
                    'driver_sha256':dp.digest(directory/'check_sa.f'),
                    'interface_sha256':dp.digest(directory/'CT_interface.f'),
                    'executable_sha256':dp.digest(directory/'check'),'physical_card_sha256':physical,
                    'madloop_card_sha256':dp.digest(output/'Cards/MadLoopParams.dat'),
                    'instrumentation':receipt if variant=='observed' else None,'point_observations':[]}
                for key,obs in sorted(parsed['observations'].items()):
                    mode,code,fpdp,fpqp,nearest,*counts = obs['metadata']
                    if fpqp or counts[3] or counts[13] or counts[14]:
                        raise ValueError('QP/compensatedDP is outside this control')
                    if any(v for idx,v in enumerate(counts[4:12],1) if idx!=tool):
                        raise ValueError('Unexpected actual backend')
                    if variant == 'observed' and sum(counts[4:12]) != counts[0]:
                        raise ValueError('Invalid dispatch count')
                    if variant == 'plain' and any(counts[:15]):
                        raise ValueError('Uninstrumented copy unexpected recorder call')
                    entry['point_observations'].append({'point':key[0],'transform':key[1],
                        'metadata':obs['metadata'],'rows':obs['rows']})
                dp.dump(HERE/(label+'-result.json'),entry)
                observations[variant] = parsed
                entries[variant] = entry
                results[label] = bound(HERE/(label+'-result.json'))
            compared,zeros = 0,0
            for field in ('guard_object','additional_link_flags','driver_sha256','madloop_card_sha256'):
                if entries['plain'][field] != entries['observed'][field]:
                    raise ValueError('Plain/observed guard/link/driver/card mismatch: '+field)
            if guard_observations['plain'] != guard_observations['observed']:
                raise ValueError('Native per-call statuses/lifecycle differ after instrumentation')
            for key,left in observations['plain']['observations'].items():
                right = observations['observed']['observations'][key]
                if left['metadata'][:5] != right['metadata'][:5]:
                    raise ValueError('Modes/status/commonFPE/rounding differ')
                if left['metadata'][-3:] != right['metadata'][-3:]:
                    raise ValueError('Hardware flags differ')
                for order,row in left['rows'].items():
                    other = right['rows'][order]
                    if [bits(v) for v in row['values']] != [bits(v) for v in other['values']]:
                        raise ValueError('Numeric or internal accuracy parity differs')
                    compared += 4
                    zeros += sum(v==0. for v in row['values'][:4])
            dp.dump(HERE/(str(tool)+'-parity.json'), {'tool':tool,'points':100,'transformations':3,
                'orders_including_total':2,'physical_coefficients_compared':compared,
                'zero_coefficients_compared':zeros,'f64_bitexact':True,
                'status_commonFPE_rounding_and_hardware_flags_exact':True,
                'instrumentation_failure_counters_retained':True,
                'native_per_call_statuses_and_lifecycle_exact':True,
                'compensatedDP_or_MP_calls':0,'qualification_performed':False})
    check_sources(cohort,anchor)
    dp.dump(HERE/'complete.json', {'schema':'qcd1-nrs8-native-dp-recorder-parity-v1','backends':[8],
        'physical_points':100,'transformations':3,'results':results,
        'nativeDP_recorder_parity_pass':True,'four_tool_campaign_complete':False,
        'timing_or_perf_claimed':False,'qualification_performed':False,
        'frozen_dp_snapshot':bound(FROZEN_DP),
        'original_source_unchanged':bench.generated_source_manifest(ORIGIN)==anchor})

if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        dp.dump(HERE/'failed.json', {'error_type':type(error).__name__,'error':str(error),
            'four_tool_campaign_complete':False,'qualification_performed':False})
        raise
