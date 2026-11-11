"""Dormant native stages; imported only after explicit reviewed-index admission."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys

from admission import HERE, ROOT, WORK, RESULTS, load, read_json, record, require, save, sha, verify_record

MAKE = '''include makefile
.PHONY: trace-info trace-list trace-objects
trace-info:
\t@echo FC $(FC)
\t@echo FFLAGS $(FFLAGS)
\t@echo LDFLAGS $(LDFLAGS)
\t@echo LINKLIBS $(LINKLIBS)
\t@echo LOOP_INCLUDE $(LOOP_INCLUDE)
trace-list:
\t@echo check_sa.o $(PROCESS)
trace-objects: check_sa.o $(PROCESS)
'''


def verify_absolute(files):
    for path,pin in files.items():verify_record(path,pin)


def capture_flags(r,directory):
    compiler=load('ktrace_flags',r.SUPERVISOR/'protocol/tirust_benchmark_compiler.py')
    original_run,original_output=compiler.subprocess.run,compiler.subprocess.check_output
    def run(args,cwd=None,env=None,**kwargs):
        output=r.L.capture(args,cwd or directory)
        return type('Query',(),{'returncode':0,'stdout':output,'stderr':''})()
    def output(args,cwd=None,**kwargs):return r.L.capture(args,cwd or directory).strip()
    try:
        compiler.subprocess.run=run;compiler.subprocess.check_output=output
        flags=compiler.capture_effective_flags(directory,env=r.L.environment(directory))
    finally:
        compiler.subprocess.run=original_run;compiler.subprocess.check_output=original_output
    return compiler,flags


def literal_cards(output,key,mode=1):
    # Dereference only within the private tree; never write a donor symlink.
    for filename,source in (('param_card.dat',key+'-param_card.dat'),
                            ('MadLoopParams.dat',key+'-mode'+str(mode)+'-MadLoopParams.dat')):
        raw=(HERE/'inputs'/source).read_bytes()
        paths={output/'Cards'/filename}
        paths.update((output/'SubProcesses').rglob(filename))
        for path in paths:
            require(path.resolve().is_relative_to(output.resolve()),'Card link escapes private worker')
            path.write_bytes(raw)


def link_variant(r,key,variant,output,changed_only=False):
    directory=output/'SubProcesses'/r.CASES[key][1]
    label=key+'-'+variant
    (directory/'ktrace.mk').write_text(MAKE)
    sources=r.source_inventory(output)
    module_inputs={str(p.resolve()):record(p) for p in output.rglob('*.mod') if p.is_file()}
    compiler,flags=capture_flags(r,directory)
    expected=read_json(ROOT/r.CONTEXT['ew_compilation']['path'])['effective_compiler_flags']
    compiler.validate_compiler_equality(expected,flags)
    save(RESULTS/'builds'/label/'before-compile.json',{'sources':sources,
        'compiler_flags':flags,'module_inputs':module_inputs,'changed_only':changed_only})
    if changed_only:
        # The other objects and archives are copied byte-for-byte from U.
        for name in ('check_sa.o','CT_interface.o','TIRUST_interface.o'):
            (directory/name).unlink(missing_ok=True)
        r.command(['make','-j2','check_sa.o','CT_interface.o','TIRUST_interface.o'],
                  'compile-'+label,directory,900)
    else:
        r.command(['make','-B','-j2','-f','ktrace.mk','trace-objects'],
                  'compile-'+label,directory,1500)
    info=r.command(['make','-s','--no-print-directory','-f','ktrace.mk','trace-info'],
                   'link-info-'+label,directory,30)
    fields={line.split()[0]:shlex.split(line)[1:] for line in info.splitlines() if line.strip()}
    names=r.command(['make','-s','--no-print-directory','-f','ktrace.mk','trace-list'],
                    'object-list-'+label,directory,30).split()
    require(names and len(names)==len(set(names)) and 'check_sa.o' in names,'Object closure differs')
    objects={str((directory/name).resolve()):sha((directory/name).read_bytes()) for name in names}
    external=r.L.q.external_inputs(directory,info,fields['FC'],r.L.ACTIVE.end-10)
    search=r.command(fields['FC']+['-print-search-dirs'],'link-search-'+label,directory,30)
    wrapper=r.L.linker.wrapper_flags(Path(shutil.which(fields['FC'][0])).resolve(),r.L.environment(directory))
    paths=r.L.linker.search_paths(info,search,wrapper['flags'],directory)
    closure=r.L.linker.prebind(external,paths,r.L.linker.compiler_libraries(info+'\n'+wrapper['flags']))
    bound={**external,**closure['files'],**objects,
           **{name:value['sha256'] for name,value in wrapper['inputs'].items()}}
    save(RESULTS/'builds'/label/'prelink.json',{'bound':bound,'closure':closure,
        'wrapper':wrapper,'completed_before_link':True})
    r.L.q.verify_hashes(bound)
    (directory/'check').unlink(missing_ok=True)
    r.command(['make','-j2','check','LDFLAGS='+' '.join(fields['LDFLAGS']+['-Wl,-Map,ktrace.map'])],
              'link-'+label,directory,300)
    r.L.q.verify_hashes(bound)
    loads={}
    for name in re.findall(r'^LOAD (.+)$',(directory/'ktrace.map').read_text(),re.M):
        path=Path(name);path=path if path.is_absolute() else directory/path
        r.L.linker.require_prebound_load(path,bound);loads[str(path.resolve())]=record(path)
    r.exact_inventory(output,sources)
    _,after=capture_flags(r,directory);compiler.validate_compiler_equality(flags,after)
    runtime=r.L.runtime(directory/'check')
    receipt={'case':key,'variant':variant,'output':str(output),'sources':sources,
        'executable':str(directory/'check'),'executable_pin':record(directory/'check'),
        'objects':{str(Path(name).relative_to(output)):checksum for name,checksum in objects.items()},
        'prelink_bound':bound,'actual_link_loads':loads,'runtime':runtime,'compiler_flags':flags,
        'physical_card':record(output/'Cards/param_card.dat'),'unmodified_numerical_arithmetic':True}
    save(RESULTS/'builds'/label/'build.json',receipt)
    return receipt


def build(r):
    require(read_json(RESULTS/'stage-derive.json')['passed'],'Derived stage did not pass')
    generated=read_json(RESULTS/'generation.json');r.exact_inventory(WORK/'exports/cms',generated['raw_sources'])
    ew=read_json(HERE/'ew-source-bindings.json');r.exact_inventory(ROOT/r.CONTEXT['ew_donor'],ew['files'])
    modules=read_json(HERE/'ew-module-inputs.json')
    for name,pin in modules['files'].items():verify_record(ROOT/modules['donor']/name,pin)
    dependencies=r.dependency_snapshot()
    expected_ew=read_json(ROOT/r.CONTEXT['ew_compilation']['path'])
    for key in ('cms','ew'):
        source=WORK/'exports/cms' if key=='cms' else ROOT/r.CONTEXT['ew_donor']
        output=r.private_clone(source,WORK/'builds'/key/'unmodified')
        directory=output/'SubProcesses'/r.CASES[key][1]
        literal_cards(output,key)
        (directory/'check_sa.f').write_bytes((RESULTS/'derived'/key/'driver-unmodified.f').read_bytes())
        if key=='cms':
            for scope in ('MODEL','DHELAS'):
                r.command(['make','-B','-j2'],'common-'+key+'-'+scope,output/'Source'/scope,900)
        else:
            # These two archives are the exact admitted EW compilation inputs.
            for name in ('libmodel.a','libdhelas.a'):
                expected=expected_ew['link_inputs_after']['archives'][str(source/'lib'/name)]
                require(sha((output/'lib'/name).read_bytes())==expected,'EW reusable archive differs')
        unmodified=link_variant(r,key,'unmodified',output)
        instrumented=r.private_clone(output,WORK/'builds'/key/'instrumented',keep_objects=True)
        sub=instrumented/'SubProcesses'/r.CASES[key][1]
        for name in ('CT_interface.f','TIRUST_interface.f'):
            (sub/name).write_bytes((RESULTS/'derived'/key/name).read_bytes())
        (sub/'check_sa.f').write_bytes((RESULTS/'derived'/key/'driver-instrumented.f').read_bytes())
        changed=link_variant(r,key,'instrumented',instrumented,changed_only=True)
        for name,checksum in unmodified['objects'].items():
            if Path(name).name not in ('check_sa.o','CT_interface.o','TIRUST_interface.o'):
                require(changed['objects'].get(name)==checksum,'Unchanged process object differs')
        require(unmodified['runtime']['files']==changed['runtime']['files'],
                'Shared runtime providers differ between variants')
    require(dependencies==r.dependency_snapshot(),'Installed dependency changed during build')


def run_matrix(r,stage):
    require(read_json(RESULTS/'stage-build.json')['passed'],'Build stage did not pass')
    if stage=='stress':
        require(read_json(RESULTS/'ordinary-parity.json')['passed'],'Ordinary parity gate did not pass')
    requests=[x for x in read_json(HERE/'inputs/requests.json')['requests'] if x['stage']==stage]
    completed=[];failure=None
    try:
        for request in requests:
            r.L.remaining(600)
            trace_total=sum(p.stat().st_size for p in (RESULTS/'runs').glob('*/trace.dat'))
            require(trace_total+67108864<=1073741824,'Insufficient aggregate trace allowance before launch')
            key=request['case'];variant='unmodified' if request['variant']=='unmodified' else 'instrumented'
            build=read_json(RESULTS/'builds'/(key+'-'+variant)/'build.json')
            r.exact_inventory(Path(build['output']),build['sources']);r.L.q.verify_hashes(build['prelink_bound'])
            verify_record(build['executable'],build['executable_pin'])
            evidence=RESULTS/'runs'/request['id'];evidence.mkdir(parents=True)
            output=r.private_clone(Path(build['output']),WORK/'workers'/request['id'],keep_objects=True)
            literal_cards(output,key,request['requested_mode'])
            sub=output/'SubProcesses'/r.CASES[key][1]
            runtime=r.L.runtime(sub/'check')
            require(runtime['files']==build['runtime']['files'],'Worker runtime providers differ')
            verify_record(sub/'check',build['executable_pin'])
            expected={**runtime['files'],str((sub/'check').resolve()):build['executable_pin']['sha256']}
            for name in ('bootstrap','measurement'):
                shutil.copy2(HERE/'inputs'/request[name],sub/(name+'-input.dat'))
            specification={'request':request,'cwd':str(sub),'expected_runtime':expected,
                'bootstrap_file':str(sub/'bootstrap-input.dat'),'point_file':str(sub/'measurement-input.dat'),
                'bootstrap_pin':record(sub/'bootstrap-input.dat'),'point_pin':record(sub/'measurement-input.dat'),
                'executable':str(sub/'check'),'evidence':str(evidence),
                'executable_pin':record(sub/'check'),
                'source_index_sha256':sha((HERE/'index.json').read_bytes()),
                'numerical_card':record(output/'Cards/MadLoopParams.dat'),
                'physical_card':record(output/'Cards/param_card.dat')}
            spec=evidence/'specification.json';save(spec,specification)
            r.command([sys.executable,'-B',HERE/'native_job.py',spec],
                      'worker-'+request['id'],sub,600)
            result=read_json(evidence/'result.json')
            require(result['orchestration_complete'],'Incomplete worker retained')
            completed.append(request['id'])
            trace_total=sum(p.stat().st_size for p in (RESULTS/'runs').glob('*/trace.dat'))
            require(trace_total<=1073741824,'Aggregate trace byte limit reached')
        from parity import compare_stage
        parity=compare_stage(RESULTS/'runs',requests)
        save(RESULTS/(stage+'-parity.json'),parity)
        require(parity['passed'],'U/D/E parity or initialization/mode admission failed')
    except BaseException as error:
        failure=repr(error);raise
    finally:
        save(RESULTS/(stage+'-request-accounting.json'),{'requests':requests,'completed':completed,
            'not_completed':[{'request':x,'reason':failure or 'not launched'} for x in requests if x['id'] not in completed],
            'failure':failure,'performance_measurement':False})
