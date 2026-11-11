#!/usr/bin/env python3
"""Private CMS supervisor. Default dry-run; preparation and native stages separate."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import sys
import tarfile
import time

ROOT = Path('/shared/external/madloop_tirust/madgraph7_madloop_tirust')
HERE = Path(__file__).resolve().parent
P = HERE.with_name('current-cms-replay-proposal-1')
WORK = ROOT / '.tirust-build/current-cms-replay-1'
REPORT = ROOT / 'docs/benchmarkfolder/experiments/current-cms-replay-qualification-1'
CAMPAIGN = ROOT / 'docs/benchmarkfolder/campaigns/invariant-symmetry-1'
STAGES = ('prepare', 'preflight', 'run')
CAPS = {'prepare':3600, 'preflight':600, 'run':3720}
CURRENT = None
CLI_ENTRYPOINT = 'bin/madgraph'

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path); module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module

L=load('cms_owned_lifecycle',HERE/'ownership/run.py')
q=L.q
prerequisites=load('cms_prerequisites',HERE/'prerequisites.py')
sys.path.insert(0,str(HERE/'protocol'));sys.path.insert(0,str(ROOT))
bench=load('cms_retained_benchmark',HERE/'protocol/tirust_benchmark.py')
compiler=load('cms_retained_compiler',HERE/'protocol/tirust_benchmark_compiler.py')
PLAN=json.loads((P/'proposal.json').read_bytes())
PLAN['cases']=[{'case':PLAN['case'],'process_subdirectory':'SubProcesses/P0_epem_mupmum','worker_render_parameters':PLAN['fresh_exports'][0]['worker_render_parameters'],'physical_card_sha256':PLAN['physical_card_sha256']}]
for request in PLAN['requests']:
    request.update(case=PLAN['case']['id'],label=request['id'],points=str(P/'points'/(PLAN['case']['id']+'.dat')),MLReductionLib=str(request['backend']),requires_common_numerical_lock=request['lock_required'])
bench.ROOT=ROOT
L.ROOT,L.REPORT,L.WORK=ROOT,REPORT,WORK
L.STAGES=STAGES
q.ROOT,q.REPORT,q.WORK=ROOT,REPORT,WORK
q.capture=L.capture
sha=q.digest;save=q.save;require=L.require


def environment(cwd):
    env=q.env()
    for key in PLAN['environment']['remove']:env.pop(key,None)
    env.update(PLAN['environment']['set'])
    shell=shutil.which('bash',path=env['PATH']);require(shell is not None,'Pinned Bash missing')
    env.update(PWD=str(cwd),MAKEFLAGS=('-j4' if L.CURRENT_STAGE=='prepare' else '-j1')+' SHELL='+str(Path(shell).resolve()))
    return env
L.environment=environment
_original_recorded_environment=L.recorded_environment
def recorded_environment(env):
    result=_original_recorded_environment(env)
    result.update({k:env[k] for k in ('XDG_STATE_HOME','XDG_DATA_HOME') if k in env})
    return result
L.recorded_environment=recorded_environment


def cmd(argv,label,cwd=ROOT,seconds=600):
    return L.command(argv,label,cwd,seconds)


def read(path):return json.loads(Path(path).read_bytes())

def scientific_sources(output):
    result=bench.generated_sources(output)
    if (output/'cms_probe.f').is_file():result['cms_probe.f']=sha(output/'cms_probe.f')
    for path in output.rglob('stress.mk'):result[str(path.relative_to(output))]=sha(path)
    return result

def check_hashes(items):q.verify_hashes(items)

def seal():
    index=read(HERE/'source-index.json')
    for name,item in index['files'].items():
        require(sha(HERE/name)==item['sha256'],'Supervisor source changed: '+name)
    require(sha(P/'source-index.json')==index['proposal_index_sha256'],'Proposal changed')
    for name,item in read(P/'source-index.json')['files'].items():require(sha(P/name)==item['sha256'],'Sealed proposal input changed: '+name)
    for name,checksum in read(P/'input-bindings.json').items():require(sha(ROOT/name)==checksum,'Bound source/evidence changed: '+name)
    require(sha(CAMPAIGN/'manifest.json')==PLAN['manifest_sha256'],'Campaign manifest changed')
    require(Path(sys.executable).resolve()==Path(PLAN['environment']['shell_setup']['python']).resolve(),'Pinned Python required')
    registered(ROOT,'candidate')


def registered(tree,source):
    manifest=read(CAMPAIGN/'manifest.json')
    if source=='candidate':
        files=dict(manifest['candidate']['exporter_identity']['files'])
        files.update({n:v['sha256'] for n,v in manifest['candidate']['numerical_sources'].items()})
    else:files=read(CAMPAIGN/'inputs/baseline.json')['files']
    for name,checksum in files.items():
        path=tree/name
        require((checksum is None and path.is_symlink()) or (checksum is not None and path.is_file() and sha(path)==checksum),'Materialized registered source differs: '+source+'/'+name)
    return files


def installed():
    result=read(P/'installed-inputs.json')
    for name,item in result.items():require(Path(name).is_file() and sha(name)==item['sha256'],'Installed dependency changed: '+name)
    for name,checksum in PLAN['candidate_native_files'].items():require(sha(Path(PLAN['candidate_native_prefix'])/name)==checksum,'Declared campaign artifact changed: '+name)
    result.update(prerequisites.external_hashes())
    return result


def config_snapshot():
    return {str(p):sha(p) for p in sorted((ROOT/'input').glob('*configuration*.txt')) if p.is_file()}


def frozen():
    seal();state=read(REPORT/'source-freeze.json')
    require(state['source_index_sha256']==sha(HERE/'source-index.json'),'Freeze/source mismatch')
    require(state['installation']==installed(),'Installed dependency changed')
    require(state['shared_configuration']==config_snapshot(),'Shared configuration changed')
    for source in ('baseline','candidate'):
        tree=WORK/('mg-'+source)
        if tree.is_dir():prerequisites.check_private(source,tree)
    return state


def materialize(source):
    mg=WORK/('mg-'+source);mg.mkdir()
    revision=PLAN[source+'_revision'];archive=WORK/(source+'.tar')
    cmd(['git','archive','--output='+str(archive),revision],'archive-'+source)
    with tarfile.open(archive) as stream:stream.extractall(mg,filter='data')
    tracked=q.inventory(mg);registered(mg,source)
    copies=prerequisites.copy_private(source,mg)
    q.verify_paths(tracked,mg)
    save(REPORT/('materialized-'+source+'.json'),{'revision':revision,'archive_sha256':sha(archive),'files':tracked,'prebuilt_copies':copies})


def capture_flags(directory,label):
    # The frozen parser delegates all make/compiler queries to the owned command helper.
    original_run,original_output=compiler.subprocess.run,compiler.subprocess.check_output
    def run(args,cwd=None,env=None,**kwargs):
        output=L.capture(args,cwd or directory)
        return type('Query',(),{'returncode':0,'stdout':output,'stderr':''})()
    def output(args,cwd=None,**kwargs):return L.capture(args,cwd or directory).strip()
    try:
        compiler.subprocess.run=run;compiler.subprocess.check_output=output
        value=compiler.capture_effective_flags(directory,env=environment(directory))
    finally:
        compiler.subprocess.run=original_run;compiler.subprocess.check_output=original_output
    save(REPORT/'builds'/label/'compiler-flags.json',value)
    return value


MAKE_ADDENDUM='''include makefile
.PHONY: stress-info stress-list stress-objects
stress-info:
\t@echo FC $(FC)
\t@echo FFLAGS $(FFLAGS)
\t@echo LDFLAGS $(LDFLAGS)
\t@echo LINKLIBS $(LINKLIBS)
\t@echo LOOP_INCLUDE $(LOOP_INCLUDE)
stress-list:
\t@echo check_sa.o $(PROCESS)
stress-objects: check_sa.o $(PROCESS)
'''


def cli_admission(mg,source,cli):
    materialized=REPORT/('materialized-'+source+'.json')
    record=read(materialized);entry=record['files'].get(CLI_ENTRYPOINT)
    path=mg/CLI_ENTRYPOINT
    require(record['revision']==PLAN[source+'_revision'],'CLI revision mismatch')
    require(entry is not None and entry['kind']=='file' and path.is_file() and not path.is_symlink(),'Tracked CLI entrypoint missing')
    require(sha(path)==entry['sha256']==read(P/'cli-entrypoints.json')[source]['sha256'],'Tracked CLI entrypoint bytes differ')
    return {'entrypoint':CLI_ENTRYPOINT,'sha256':entry['sha256'],'materialized_receipt_sha256':sha(materialized),'argv':[sys.executable,'-B','./'+CLI_ENTRYPOINT,'--plain',str(cli)]}


def prepare_output_parent(output):
    require(not output.exists() and not output.is_symlink(),'Fresh CLI output required')
    output.parent.mkdir(parents=True,exist_ok=True)


def validate_cli_output(output,entry,console):
    markers=('Error detected','interrupted in sub-command','Traceback (most recent call last)','Compiling CutTools','Compiling IREGI','Re-compiling it now')
    found=[marker for marker in markers if marker in console]
    require(not found,'CLI fatal diagnostic markers: '+repr(found))
    require(output.is_dir(),'CLI did not create expected output')
    directory=output/entry['process_subdirectory']
    required=[output/'Cards/param_card.dat',output/'Source/make_opts',*[directory/name for name in ('check_sa.f','loop_matrix.f','CT_interface.f','improve_ps.f','makefile')]]
    require(all(p.is_file() for p in required) and all((output/'Source'/scope).is_dir() for scope in ('MODEL','DHELAS')),'CLI generated output incomplete')


def check_compiler(capture):
    version=capture(['gfortran','-dumpfullversion','-dumpversion']).strip()
    require(version==prerequisites.DATA['CT_version'],'Active compiler differs from prebuilt CutTools')


def retain_raw_cli(output,label):
    science=scientific_sources(output);files=dict(science)
    for name in ('Cards/param_card.dat','Cards/MadLoopParams.dat'):
        if (output/name).is_file():files[name]=sha(output/name)
    archive=REPORT/'builds'/label/'raw-cli-sources';bound={}
    for name,checksum in files.items():
        source=output/name;target=archive/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        require(sha(source)==sha(target)==checksum,'Raw fresh CLI source changed during preservation')
        bound[str(target)]=checksum
    return science,{'directory':str(archive),'bound_files':bound}


def prepare_process(source,entry):
    cid=entry['case']['id'];label=source+'-'+cid;mg=WORK/('mg-'+source);output=WORK/'exports'/source/cid
    with L.Deadline(600):
        before=registered(mg,source)
        prerequisites.check_private(source,mg)
        prerequisites.external_hashes()
        check_compiler(lambda argv:cmd(argv,'compiler-version-'+label,mg))
        cli=P/'commands'/(label+'.mg5')
        prepare_output_parent(output)
        admission=cli_admission(mg,source,cli)
        save(REPORT/'builds'/label/'cli-admission.json',admission)
        console=cmd(admission['argv'],'generate-'+label,mg)
        require(cli_admission(mg,source,cli)==admission,'CLI entrypoint changed during generation')
        validate_cli_output(output,entry,console)
        prerequisites.check_private(source,mg)
        prerequisites.external_hashes()
        prerequisites.check_generated(source,output,entry['process_subdirectory'])
        registered(mg,source)
        require(before==registered(mg,source),'Generator source changed')
        raw_cli,raw_archive=retain_raw_cli(output,label)
        shutil.copy2(P/'cards/param_card.dat',output/'Cards/param_card.dat')
        directory=output/entry['process_subdirectory']
        require(directory.is_dir(),'Unexpected subprocess directory')
        shutil.copy2(P/'protocol/cms_probe.f',output/'cms_probe.f')
        original=(directory/'check_sa.f').read_text()
        prefix=re.search(r'CALL\s+(\w*)GET_ANSWER_DIMENSION',original,re.I).group(1)
        nborn=int(re.search(r'NSQSO_BORN\s*=\s*(\d+)',(directory/'nsqso_born.inc').read_text(),re.I).group(1)) if (directory/'nsqso_born.inc').exists() else 0
        parameters={'prefix':prefix,'nexternal':entry['case']['nfinal']+2,'nborn':nborn}
        require(parameters==entry['worker_render_parameters'],'Fresh graph dimensions/prefix differ')
        (directory/'check_sa.f').write_text(bench.render_worker((P/'protocol/cms-worker.f').read_text(),parameters))
        (directory/'stress.mk').write_text(MAKE_ADDENDUM)
        state={'source':source,'case':entry['case'],'revision':PLAN[source+'_revision'],'command_sha256':sha(cli),'cli_admission':admission,'raw_cli_scientific_sources':raw_cli,'raw_cli_archive':raw_archive,'scientific_sources':scientific_sources(output),'all_generation_files':q.inventory(output,generated=True),'parameters':parameters,'physical_card_sha256':sha(output/'Cards/param_card.dat')}
        save(REPORT/'builds'/label/'generation.json',state)
    with L.Deadline(600):
        flags=capture_flags(directory,label)
        expected=read(P/'compiler-policy'/(source+'.json'))
        compiler.validate_compiler_equality(expected,flags)
        info=cmd(['make','-s','--no-print-directory','-f','stress.mk','stress-info'],'link-info-'+label,directory)
        fields={line.split()[0]:shlex.split(line)[1:] for line in info.splitlines() if line.strip()}
        for scope in ('MODEL','DHELAS'):
            cmd(['make','--always-make','-j4'],'build-'+label+'-'+scope,output/'Source'/scope)
        names=cmd(['make','-s','--no-print-directory','-f','stress.mk','stress-list'],'object-list-'+label,directory).split()
        require(len(names)==len(set(names)) and 'check_sa.o' in names,'Invalid process object closure')
        cmd(['make','--always-make','-j4','-f','stress.mk','stress-objects'],'objects-'+label,directory)
        objects={str(path.resolve()):sha(path) for path in output.rglob('*.o') if path.is_file()}
        external=q.external_inputs(directory,info,fields['FC'],L.ACTIVE.end-10)
        search=cmd(fields['FC']+['-print-search-dirs'],'compiler-search-'+label,directory)
        wrapper=L.linker.wrapper_flags(Path(shutil.which(fields['FC'][0],path=environment(directory)['PATH'])).resolve(),environment(directory))
        paths=L.linker.search_paths(info,search,wrapper['flags'],directory)
        closure=L.linker.prebind(external,paths,L.linker.compiler_libraries(info+'\n'+wrapper['flags']))
        bound={**external,**closure['files'],**objects,**{n:v['sha256'] for n,v in wrapper['inputs'].items()}}
        dependencies=bench.dependency_inputs(directory)
        for section in dependencies.values():bound.update(section)
        save(REPORT/'builds'/label/'prelink.json',{'completed_before_link':True,'wrapper':wrapper,'closure':closure,'bound':bound})
        check_hashes(bound)
        ld=' '.join(fields['LDFLAGS']+['-Wl,-Map,stress.map'])
        cmd(['make','-j4','check','LDFLAGS='+ld],'link-'+label,directory)
        check_hashes(bound)
        loads={}
        for spelling in re.findall(r'^LOAD (.+)$',(directory/'stress.map').read_text(),re.M):
            path=Path(spelling);path=path if path.is_absolute() else directory/path
            L.linker.require_prebound_load(path,bound);loads[str(path.resolve())]=sha(path)
        require(scientific_sources(output)==state['scientific_sources'],'Scientific source changed during full build')
        compiler.validate_compiler_equality(flags,capture_flags_after(directory))
        probe=build_probe(output,label,directory)
        shutil.copy2(directory/'stress.map',REPORT/'builds'/label/'stress.map')
        save(REPORT/'builds'/label/'build.json',{'generation_sha256':sha(REPORT/'builds'/label/'generation.json'),'prelink_sha256':sha(REPORT/'builds'/label/'prelink.json'),'fresh_full_objects':True,'executable':str(directory/'check'),'executable_sha256':sha(directory/'check'),'objects':objects,'bound_inputs_before_after':bound,'actual_link_loads':loads,'scientific_sources':state['scientific_sources'],'physical_card_sha256':state['physical_card_sha256'],'compiler_flags':flags,'loader_executed':False,'masses_executed':False,'probe':probe})
    prerequisites.check_private(source,mg)
    frozen()


def validate_model_probe(raw):
    lines=raw.splitlines();require(len(lines)==3,'CMS probe row count differs')
    require(float(lines[0])==PLAN['expected_CMS_model']['CMSParam'],'Runtime CMSParam differs')
    result={'CMSParam':float(lines[0]),'masses':{}}
    for name,line in zip(('W','Z'),lines[1:]):
        values=list(map(float,line.split()));require(len(values)==4 and all(__import__('math').isfinite(x) for x in values),'Nonfinite/malformed CMS model row')
        mass,width,real,imag=values;expected=PLAN['expected_CMS_model'][name]
        require(mass==expected['mass'] and width==expected['width'] and width>0,'Frozen CMS mass/width differs: '+name)
        error=abs(complex(real,imag)-complex(mass*mass,-mass*width))/mass**2
        require(imag<0 and error<=1e-13,'CMS square reconstruction differs: '+name)
        result['masses'][name]={'mass':mass,'width':width,'squared_mass':[real,imag],'relative_reconstruction_error':error}
    return result


def build_probe(output,label,directory):
    source=output/'cms_probe.f';obj=output/'cms_probe.o';exe=output/'cms_probe';model=output/'lib/libmodel.a'
    require(sha(source)==sha(P/'protocol/cms_probe.f'),'CMS probe source changed')
    compiler_argv=['gfortran']
    cmd(compiler_argv+['-I'+str(output/'Source/MODEL'),'-c',source,'-o',obj],'cms-probe-object-'+label,ROOT)
    explicit={str(obj):sha(obj),str(model):sha(model),str(source):sha(source)}
    external=q.external_inputs(directory,str(model),compiler_argv,L.ACTIVE.end-10)
    search=cmd(compiler_argv+['-print-search-dirs'],'cms-probe-search-'+label,directory)
    compiler_path=Path(shutil.which('gfortran',path=environment(directory)['PATH'])).resolve()
    wrapper=L.linker.wrapper_flags(compiler_path,environment(directory));paths=L.linker.search_paths('',search,wrapper['flags'],directory)
    closure=L.linker.prebind(external,paths,L.linker.compiler_libraries(wrapper['flags']))
    bound={**external,**closure['files'],**explicit,str(compiler_path):sha(compiler_path),**{n:v['sha256'] for n,v in wrapper['inputs'].items()}}
    receipt=REPORT/'builds'/label/'cms-probe-prelink.json';save(receipt,{'completed_before_link':True,'wrapper':wrapper,'closure':closure,'bound':bound})
    check_hashes(bound);mapfile=output/'cms-probe.map'
    cmd(compiler_argv+['-o',exe,obj,model,'-Wl,-Map,'+str(mapfile)],'cms-probe-link-'+label,ROOT)
    check_hashes(bound);loads={}
    for spelling in re.findall(r'^LOAD (.+)$',mapfile.read_text(),re.M):
        path=Path(spelling);path=path if path.is_absolute() else ROOT/path
        L.linker.require_prebound_load(path,bound);loads[str(path.resolve())]=sha(path)
    shutil.copy2(mapfile,REPORT/'builds'/label/'cms-probe.map')
    return {'executable':str(exe),'sha256':sha(exe),'object':str(obj),'object_sha256':sha(obj),'model_archive':str(model),'model_sha256':sha(model),'source':str(source),'source_sha256':sha(source),'prelink_receipt':str(receipt),'prelink_sha256':sha(receipt),'bound':bound,'actual_LOAD':loads}


def check_probe(probe):
    require(sha(probe['prelink_receipt'])==probe['prelink_sha256'],'CMS prelink receipt changed')
    check_hashes(probe['bound']);require(sha(probe['executable'])==probe['sha256'],'CMS probe executable changed')


def capture_flags_after(directory):
    # Reuse owned query parser without overwriting the before receipt.
    name='after-'+str(directory.relative_to(WORK)).replace('/','-')
    return capture_flags(directory,name)


def prepare():
    prerequisites.check_donors()
    require(not REPORT.exists() and not WORK.exists(),'Fresh namespaces required')
    REPORT.mkdir();WORK.mkdir()
    save(REPORT/'source-freeze.json',{'source_index_sha256':sha(HERE/'source-index.json'),'proposal_index_sha256':sha(P/'source-index.json'),'campaign_manifest_sha256':PLAN['manifest_sha256'],'installation':installed(),'shared_configuration':config_snapshot(),'native_execution':False})
    with L.Deadline(600):
        for source in ('baseline','candidate'):materialize(source)
        model=WORK/'models/loop_qcd_qed_sm_Gmu';model.mkdir(parents=True)
        for name,checksum in read(CAMPAIGN/'manifest.json')['shared_inputs']['model_tree'].items():
            donor=ROOT/'.benchmark-work/tirust/campaigns/invariant-symmetry-1/models/loop_qcd_qed_sm_Gmu'/name
            require(sha(donor)==checksum,'Pinned EW model changed')
            dest=model/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(donor,dest)
    for entry in PLAN['cases']:
        for source in ('baseline','candidate'):prepare_process(source,entry)


def build_record(source,entry):
    path=REPORT/'builds'/(source+'-'+entry['case']['id'])/'build.json';record=read(path)
    require(sha(path.parent/'generation.json')==record['generation_sha256'],'Generation receipt changed')
    require(sha(path.parent/'prelink.json')==record['prelink_sha256'],'Prelink receipt changed')
    generation=read(path.parent/'generation.json')
    check_hashes(generation['raw_cli_archive']['bound_files'])
    require(record['fresh_full_objects'] is True,'Full fresh objects not proven')
    check_hashes(record['bound_inputs_before_after'])
    require(sha(record['executable'])==record['executable_sha256'],'Compiled executable changed')
    check_probe(record['probe'])
    output=WORK/'exports'/source/entry['case']['id']
    require(scientific_sources(output)==record['scientific_sources'],'Built source changed')
    require(sha(output/'Cards/param_card.dat')==record['physical_card_sha256'],'Physical card changed')
    prerequisites.check_private(source,WORK/('mg-'+source))
    return record


def runtime(executable):
    loader=L.interpreter(executable);checksum=sha(loader)
    text=L.capture([loader,'--list',executable])
    require('not found' not in text and sha(loader)==checksum,'Loader/dependency failed')
    paths={str(Path(p).resolve()) for p in re.findall(r'(/[^\s()]+)',text) if Path(p).is_file()};paths.add(str(loader))
    return {'loader':str(loader),'loader_sha256':checksum,'listing':text,'files':{p:sha(p) for p in sorted(paths)}}


def preflight():
    frozen()
    for entry in PLAN['cases']:
        for source in ('baseline','candidate'):
            label=source+'-'+entry['case']['id'];b=build_record(source,entry);directory=Path(b['executable']).parent
            with L.Deadline(60):
                rt=runtime(b['executable'])
                if source=='candidate':require(rt['files'].get(str((q.INSTALL/'libmadloop_tirust.so').resolve()))==q.EXPECTED['libmadloop_tirust.so'],'Wrong candidate TIRust DSO')
                cmd(['./check','--masses'],'masses-'+label,directory,30)
                masses=[float(x) for x in (directory/'masses.dat').read_text().split()]
                require(masses==PLAN['expected_CMS_model']['external_masses'],'Mass/point mismatch')
                shutil.copy2(directory/'masses.dat',REPORT/'builds'/label/'masses.dat')
            with L.Deadline(60):
                probe_runtime=runtime(b['probe']['executable'])
                cmd([b['probe']['executable']],'cms-model-'+label,directory,30)
                raw=(directory/'cms_masses.dat').read_text();model=validate_model_probe(raw)
                shutil.copy2(directory/'cms_masses.dat',REPORT/'builds'/label/'cms_masses.dat')
            save(REPORT/'builds'/label/'preflight.json',{'runtime':rt,'probe_runtime':probe_runtime,'masses':masses,'complex_model':model,'cms_masses_sha256':sha(REPORT/'builds'/label/'cms_masses.dat'),'probe':b['probe'],'physical_card_sha256':PLAN['physical_card_sha256'],'build_sha256':sha(REPORT/'builds'/label/'build.json')})
            build_record(source,entry)
    frozen()


def render_card(text, settings, source):
    # Literal serialization of the frozen values; no mutable shared Python import.
    lines=text.splitlines(); names={}
    for i,line in enumerate(lines):
        if line.startswith('#'):
            name=line[1:].split()[0].lower()
            require(name not in names,'Duplicate numerical card field: '+name)
            names[name]=i
    require(not (set(names)-set(settings)),'Uncontrolled numerical template fields')
    ignored={}
    for name,value in settings.items():
        if name not in names:
            require(source=='baseline' and name in ('tirustcachemaxbytes','tirustcomputefinite','tirustglobalcache','useqpintegrandfortirust'),'Required numerical setting missing: '+name)
            ignored[name]=value;continue
        i=names[name]+1
        require(i<len(lines) and not lines[i].lstrip().startswith('#'),'No numerical card value: '+name)
        lines[i]=('.true.' if value else '.false.') if isinstance(value,bool) else (('%e'%value).replace('e','d') if isinstance(value,float) else str(value))
    return '\n'.join(lines)+'\n',ignored


def configure_worker(output,request):
    source=request['source'];cardpath=output/'Cards/MadLoopParams.dat'
    settings=read(P/'policy-reference.json')[source];settings['mlreductionlib']=str(request['backend'])
    text,ignored=render_card(cardpath.read_text(),settings,source)
    frozen_card=P/'cards'/(source+'-'+str(request['backend'])+'-MadLoopParams.dat')
    require(text.encode()==frozen_card.read_bytes(),'Numerical card serialization differs from frozen literal card')
    cardpath.write_text(text)
    for path in (output/'SubProcesses').rglob('MadLoopParams.dat'):path.write_text(text)
    for pattern in ('*HelFilter.dat','*LoopFilter.dat'):
        for path in (output/'SubProcesses').rglob(pattern):path.unlink()
    return {'requested':settings,'not_applicable_to_baseline':ignored,'serialized_sha256':sha(cardpath)}


def acquire(request):
    if not request['requires_common_numerical_lock']:return None
    path=ROOT/'.tirust-build/numerical-tests.lock';lock=path.open('a');start=time.monotonic()
    while True:
        L.REQUESTS.check()
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);return lock
        except BlockingIOError:
            if time.monotonic()-start>=3600:lock.close();raise TimeoutError('Lock wait expired before execution')
            time.sleep(.1)


def preflight_record(source,cid):
    directory=REPORT/'builds'/(source+'-'+cid);path=directory/'preflight.json';pre=read(path)
    require(pre['build_sha256']==sha(directory/'build.json'),'Preflight/build mismatch')
    require(pre['physical_card_sha256']==PLAN['physical_card_sha256'],'Preflight/card mismatch')
    require(sha(directory/'cms_masses.dat')==pre['cms_masses_sha256'],'CMS raw model proof changed')
    require(validate_model_probe((directory/'cms_masses.dat').read_text())==pre['complex_model'],'CMS parsed model proof changed')
    check_hashes(pre['probe_runtime']['files'])
    return pre,sha(path)


def validate_private_links(output,source):
    allowed_roots=[output.resolve(),(WORK/'exports'/source/PLAN['case']['id']).resolve(),(WORK/('mg-'+source)).resolve()]
    bound={str(Path(n).resolve()):v['sha256'] for n,v in installed().items()}
    for path in output.rglob('*'):
        if not path.is_symlink():continue
        target=path.resolve(strict=True)
        if any(target.is_relative_to(root) for root in allowed_roots):continue
        require(target.is_file() and bound.get(str(target))==sha(target),'Worker link resolves outside private/bound external inputs: '+str(path))


def run_request(request):
    entry=next(e for e in PLAN['cases'] if e['case']['id']==request['case']);source=request['source'];cid=request['case'];label=request['label']
    evidence=REPORT/'runs'/cid/label;evidence.mkdir(parents=True)
    b=build_record(source,entry);pre,preflight_sha=preflight_record(source,cid)
    output=WORK/'workers'/label/cid
    shutil.copytree(WORK/'exports'/source/cid,output,symlinks=True)
    directory=output/entry['process_subdirectory'];settings=configure_worker(output,request)
    validate_private_links(output,source)
    require(sha(directory/'check')==b['executable_sha256'],'Worker copy differs')
    sources=scientific_sources(output);cardsha=sha(output/'Cards/param_card.dat');numerical=bench.numerical_card_inputs(output)
    actual=bench.resolved_parameter_card(directory)
    require(actual is not None and sha(actual)==cardsha==entry['physical_card_sha256'],'Shadowed physical card')
    rt=runtime(directory/'check');require(rt['files']==pre['runtime']['files'],'Relocated runtime differs')
    require(sha(request['points'])==PLAN['points_sha256'],'Frozen points changed')
    specification={'request':request,'backend':request['backend'],'cwd':str(directory),'argv_without_fifo':['taskset','-c','12','./check',request['points']],'expected_runtime':{**rt['files'],str((directory/'check').resolve()):b['executable_sha256']},'settings':settings,'actual_stdin_after_READY':'RUN 1 1\nSTOP 0 0\n','points_sha256':sha(request['points']),'physical_card_sha256':cardsha,'scientific_sources':sources,'numerical_cards':numerical,'build':b,'runtime':rt,'preflight_sha256':preflight_sha,'complex_model':pre['complex_model'],'phase_timeout_seconds':300,'points_count':21,'host_accuracy_limit':1e-3,'expected_dimensions':PLAN['expected_observation']['answer_dimensions']}
    save(evidence/'specification.json',specification)
    shutil.copy2(output/'Cards/MadLoopParams.dat',evidence/'MadLoopParams.dat');shutil.copy2(actual,evidence/'param_card.dat')
    failure=None;error_object=None
    try:cmd([sys.executable,'-B',HERE/'native_job.py',evidence/'specification.json'],'request-'+cid+'-'+label,directory,610)
    except Exception as error:failure=repr(error);error_object=error
    finally:
        for name in ('warmup.dat','dimensions.dat','values.dat','timed_status.dat','timing.dat'):
            if (directory/name).is_file() and not (evidence/name).exists():shutil.copy2(directory/name,evidence/name)
        check_hashes(b['bound_inputs_before_after']);check_hashes(rt['files'])
        require(scientific_sources(output)==sources and bench.numerical_card_inputs(output)==numerical and sha(actual)==cardsha and sha(directory/'check')==b['executable_sha256'],'Worker inputs changed')
        build_record(source,entry);frozen()
    result=read(evidence/'result.json') if (evidence/'result.json').exists() else {'orchestration_complete':False,'numerically_qualified':False,'failure':failure,'runtime_identity_failure':False}
    execution_path=REPORT/'execution'/('request-'+cid+'-'+label+'.json')
    execution=read(execution_path) if execution_path.exists() else None
    result,stop=admit_completion(result,error_object,execution)
    save(evidence/'completion.json',{'result':result,'outer_failure':failure,'inputs_unchanged':True,'fatal_infrastructure_failure':stop,'performance_eligible':False})
    require(not stop,'Fatal cleanup/runtime/controller failure: '+str(stop))
    return result


def admit_completion(result,error,execution):
    result=dict(result)
    if error is not None:result['numerically_qualified']=False
    if result.get('runtime_identity_failure'):return result,'runtime identity'
    if result.get('fatal_controller_failure'):return result,'native controller/cleanup failure'
    if execution is None:return result,'missing owned execution receipt'
    cleanup=execution.get('cleanup')
    if execution.get('cleanup_exception') is not None or cleanup is None or cleanup.get('survivors'):
        return result,'owned cleanup not proven'
    if error is not None and not isinstance(error,TimeoutError):return result,'controller failure: '+repr(error)
    return result,None


def run():
    frozen();results=[];total=0.
    for request in PLAN['requests']:
        lock=None;wait=time.monotonic();start=None
        try:
            lock=acquire(request);wait=time.monotonic()-wait;start=time.monotonic()
            with L.Deadline(min(620,3720-total)):
                result=run_request(request)
            elapsed=time.monotonic()-start;total+=elapsed
            results.append({'request':request,'lock_wait_seconds':wait,'elapsed_seconds':elapsed,'result':result})
            save(REPORT/'request-completions'/('%02d.json'%len(results)),results[-1])
        except BaseException as error:
            save(REPORT/'aborted-request.json',{'request':request,'state':'deferred-before-launch' if start is None else 'aborted-during-request','exception':repr(error),'numerically_qualified':False})
            raise
        finally:
            if lock is not None:lock.close()
    save(REPORT/'summary.json',summarize_results(results,total))


def compare(left,right):
    result=bench.compare_rows(left,right,3e-6)
    for row in result['residuals']:row['absolute_error']=abs(row['reference']-row['candidate'])
    result['all_stream_1e8_target_passed']=result.pop('ordinary_target_passed')
    ordinary_left=[row for row in left if row['point']<=12];ordinary_right=[row for row in right if row['point']<=12]
    ordinary=bench.compare_rows(ordinary_left,ordinary_right,3e-6)
    for row in ordinary['residuals']:row['absolute_error']=abs(row['reference']-row['candidate'])
    result['ordinary_point_comparison']=ordinary
    return result


def summarize_results(results,total):
    accepted=[r for r in results if r['result']['numerically_qualified']];comparisons=[]
    for index,left in enumerate(accepted):
        for right in accepted[index+1:]:
            comparisons.append({'left':left['request']['label'],'right':right['request']['label'],'current_pair':left['request']['source']==right['request']['source']=='candidate',**compare(left['result']['observation']['rows'],right['result']['observation']['rows'])})
    current={r['request']['backend'] for r in accepted if r['request']['source']=='candidate'}
    five=current=={1,6,7,8,9};current_comparisons=[item for item in comparisons if item['current_pair']]
    controls=[item for item in comparisons if {item['left'],item['right']}=={'baseline-1-auto','candidate-1-auto'}]
    return {'requests':results,'comparisons':comparisons,'request_count':6,'complete_checked_streams':len(accepted),'five_current_backend_complete':five,'primary_policy_agreement':all(x['passed'] for x in current_comparisons) if five else None,'baseline_vs_current_CutTools':controls[0] if controls else None,'all_requested_streams_checked_and_consistent':len(accepted)==6 and all(x['passed'] for x in comparisons),'independent_finite_term_oracle':'not supplied; historical private causal diagnostics are not primary replay evidence','performance_eligible':False,'native_elapsed_seconds_excluding_lock_wait':total}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',nargs='?',choices=STAGES);parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    require(Path.cwd()==ROOT,'Run from repository root')
    if not args.execute:
        print(json.dumps({'source_only':True,'commands_executed':0,'stages':CAPS,'requests':6,'work':str(WORK),'report':str(REPORT),'commands':[[sys.executable,'-B',str(HERE/'run.py'),s,'--execute'] for s in STAGES]},indent=2));return
    require(args.stage is not None,'Explicit stage required');seal()
    L.CURRENT_STAGE=args.stage;L.CPU='12-15' if args.stage=='prepare' else 12
    if args.stage!='prepare':
        predecessor='prepare' if args.stage=='preflight' else 'preflight'
        require(read(REPORT/('stage-'+predecessor+'.json'))['status']=='passed','Previous stage did not pass')
    marker=REPORT/('stage-'+args.stage+'.json');require(not marker.exists(),'Append-only stage; no retry')
    start=time.monotonic();failure=None
    with L.cleanup.SignalRequests() as requests:
        L.REQUESTS=requests
        try:
            if args.stage=='run':run() # Each inclusive worker deadline starts after separately bounded lock acquisition.
            else:
                with L.Deadline(CAPS[args.stage]):{'prepare':prepare,'preflight':preflight}[args.stage]()
            frozen()
        except BaseException as error:failure=error
        finally:
            if REPORT.exists() and args.stage=='run':
                completed={(r['case'],r['label']) for r in [read(p)['request'] for p in (REPORT/'request-completions').glob('*.json')]}
                save(REPORT/'request-accounting.json',{'requested':6,'completed':len(completed),'not_completed':[r for r in PLAN['requests'] if (r['case'],r['label']) not in completed],'native_stage_failure':None if failure is None else repr(failure)})
            if REPORT.exists():save(marker,{'stage':args.stage,'status':'passed' if failure is None else 'failed','exception':None if failure is None else repr(failure),'elapsed_seconds_including_wait':time.monotonic()-start,'cap_seconds_excluding_lock_wait':CAPS[args.stage],'source_index_sha256':sha(HERE/'source-index.json'),'campaign_manifest_sha256':PLAN['manifest_sha256'],'native_execution':args.stage!='prepare','performance_eligible':False})
    if failure is not None:raise failure

if __name__=='__main__':main()
