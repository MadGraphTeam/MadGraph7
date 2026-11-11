"""Pure reconstruction and validation of the closed, pinned build environment."""
import hashlib
import os
from pathlib import Path
import shutil

INJECTION_CONTROLS=('CC','CXX','FC','CFLAGS','CXXFLAGS','CPPFLAGS','FCFLAGS','FFLAGS','LDFLAGS',
                    'BASH_ENV','ENV','GLOBAL_FLAG','MAKEFLAGS','MAKEFILES','MAKEOVERRIDES','MFLAGS','SHELL')


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(pin,compiler_environment_names,original_compiler_environment,libraries,inherited=None):
    require(pin['original_compiler_environment']==original_compiler_environment,'Toolchain metadata differs from original compiler policy')
    env=dict(os.environ if inherited is None else inherited)
    # Remove all harness compiler/loader injection, then restore recorded values.
    for key in list(env):
        if (key.startswith(('LD_','NIX_')) or key in compiler_environment_names
                or key in INJECTION_CONTROLS):env.pop(key)
    for key,value in pin['closed_environment'].items():
        if value is None:env.pop(key,None)
        else:env[key]=value
    env.update(PATH=pin['path'],MAKEFLAGS='-j1 SHELL='+pin['required_tools']['bash']['path'],
               SHELL=pin['required_tools']['bash']['path'],LD_LIBRARY_PATH=os.pathsep.join(map(str,libraries)),
               LD_PRELOAD='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               RAYON_NUM_THREADS='1',PYTHONHASHSEED='0',PYTHONDONTWRITEBYTECODE='1',
               LC_ALL='C',LANG='C',TZ='UTC',GFORTRAN_ERROR_BACKTRACE='0')
    names=set(compiler_environment_names)|{name for name in env if name.startswith(('NIX_CFLAGS_','NIX_LDFLAGS_'))}
    actual={name:env[name] for name in sorted(names) if name in env}
    require(actual==original_compiler_environment,'Frozen compiler injected environment differs from original n1 policy')
    require(env['FC']==pin['required_tools']['gfortran']['path'],'Fortran compiler selection changed')
    return env


def validate(pin,env):
    rows={}
    for name,want in pin['required_tools'].items():
        require(want['executable'] is True,'Required tool must be marked executable: '+name)
        path=Path(want['path'])
        require(path.is_absolute() and path.is_file() and os.access(path,os.X_OK),'Required tool missing/not executable: '+name)
        found=shutil.which(name,path=env['PATH'])
        require(found==str(path),'Required PATH resolution differs: '+name)
        require(str(path.resolve())==want['resolved'] and digest(path)==want['sha256']
                and path.stat().st_size==want['bytes'],'Required tool identity changed: '+name)
        rows[name]={'path':str(path),'resolved':str(path.resolve()),'sha256':want['sha256'],'executable':True}
    for name,want in pin['compiler_closure'].items():
        path=Path(name)
        if want is None:
            require(not path.exists() and not path.is_symlink(),'Unexpected compiler hook appeared: '+name)
        else:
            require(path.is_file() and digest(path)==want['sha256'] and path.stat().st_size==want['bytes']
                    and (not want['executable'] or os.access(path,os.X_OK)),'Pinned compiler closure changed: '+name)
    require(shutil.which('make',path=env['PATH'])==pin['required_tools']['make']['path'],'Pinned make unavailable')
    return {'required_tools':rows,'compiler_closure_entries':len(pin['compiler_closure']),
            'compiler_injected_environment':pin['original_compiler_environment'],
            'native_tool_queries':0,'resolution_method':'file hashes, executable bits and shutil.which only'}
