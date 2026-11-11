"""Three selected compilations; all original scientific recipes unchanged."""
from pathlib import Path
import shlex
import shutil

NEEDED=('repair_profile.f90','improve_ps.f','check_sa.f')
FLAGS=['-w','-fPIC','-ffixed-line-length-132']


def require(value,message):
    if not value:raise ValueError(message)


def private_makefile(text):
    old='$(FC) $(FFLAGS) -o $(PROG) check_sa.o $(PROCESS) $(LINKLIBS)'
    require(text.count(old)==1,'Unique original link recipe required')
    require('repair_profile.o' not in text,'Existing repair profiling ownership')
    text=text.replace(old,'$(FC) $(FFLAGS) -Wl,-Map,private-link.map,--cref -o $(PROG) check_sa.o repair_profile.o $(PROCESS) $(LINKLIBS)',1)
    return text+'''\n# Private attribution only; original scientific source flags are retained.\nrepair_profile.o: repair_profile.f90\n\t$(FC) $(FFLAGS) -c $< -o $@ $(LOOP_INCLUDE)\ncheck_sa.o improve_ps.o: repair_profile.o\n'''


def verify_policy(original,actual,compiler):
    require(original['environment']==actual['environment']=={},'Compiler injection changed')
    require(set(original['scopes'])==set(actual['scopes']),'Compiler scopes changed')
    for scope,left in original['scopes'].items():
        right=actual['scopes'][scope]
        extra={'repair_profile.f90'} if scope=='process' else set()
        require(set(right['recipes'])==set(left['recipes'])|extra,'Recipe inventory changed')
        for name,expected in left['recipes'].items():
            require(compiler._signature(expected)==compiler._signature(right['recipes'][name]),'Original source flags/compiler changed:'+scope+'/'+name)
    process=actual['scopes']['process']['recipes']
    require(process['improve_ps.f']['flags']==FLAGS,'Original repair flags changed')
    require(compiler._signature(process['repair_profile.f90'])==compiler._signature(process['improve_ps.f']),
            'New profile module must use original ordinary Fortran policy')
    return {'all_original_recipes_preserved':True,'repair_flags':FLAGS,'new_module_flags':FLAGS,'optimization_experiment':False}


def select_commands(text,expected_compiler,environment):
    selected={};links=[]
    for line in text.replace('\\\n',' ').splitlines():
        argv=shlex.split(line)
        if not argv:continue
        sources=[a for a in argv if not a.startswith('-') and Path(a).suffix.lower() in {'.f','.f90','.f95','.f03','.f08','.for'}]
        names=[s for s in sources if Path(s).name in NEEDED]
        outputs=[argv[i+1] for i,a in enumerate(argv[:-1]) if a=='-o']
        link='-c' not in argv and 'check' in outputs
        if not names and not link:continue
        require(not any(a in {';','&&','||','|','>','>>','<','&'} for a in argv),'Compound selected command')
        actual=shutil.which(argv[0],path=environment.get('PATH'))
        require(actual and Path(actual).resolve()==Path(expected_compiler).resolve(),'Selected compiler changed')
        require(argv.count('-o')==1 and len(outputs)==1 and not any(a.startswith('@') for a in argv),'Ambiguous output/response')
        if link:
            require(not sources and outputs==['check'] and argv.count('repair_profile.o')==1 and '-ltirust' in argv,
                    'Link must contain exactly one profile object and installed TIRust archive')
            require('mtirust.o' not in argv and 'qp_profile.o' not in argv,'Installed tensor wrapper must remain untouched')
            links.append(argv)
        else:
            require(argv.count('-c')==1 and len(sources)==len(names)==1 and sources[0]==Path(sources[0]).name,'Unexpected selected source')
            name=sources[0]
            require(outputs==[Path(name).stem+'.o'] and name not in selected,'Duplicate/wrong selected output')
            selected[name]=argv
    require(set(selected)==set(NEEDED) and len(links)==1,'Exact three-compile/one-link closure required')
    return [(name,selected[name]) for name in NEEDED],links[0]
