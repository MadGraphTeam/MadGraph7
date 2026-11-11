"""Pure input/tool/source binding; no compiler, loader or public-module call."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import shutil
import sys

ROOT=Path.cwd();HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write_new(path,value):
    data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    if path.exists():assert path.read_bytes()==data,'No unrecorded input replacement: '+str(path)
    else:path.write_bytes(data)
def text_new(path,text):
    if path.exists():assert path.read_text()==text,'No unrecorded source replacement: '+str(path)
    else:path.write_text(text)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj
    spec.loader.exec_module(obj);return obj

def main():
    controller=load('bulk_unit_prepare_controller',HERE/'controller.py')
    corpus=load('bulk_unit_prepare_corpus',HERE/'corpus.py').build_corpus()
    write_new(HERE/'corpus.json',corpus)
    profile=ROOT/'docs/benchmarkfolder/experiments/fortran-qp-transport-profile-qualification-1'
    assert sha(profile/'index.json')=='4abe6049fea625c567c37668956f3a8d1db71222eb2575bbf13fd32617e087d2'
    built=read(profile/'build2/build-result.json')
    names={'libgcc_s.so.1','libgfortran.so.5.0.0','libquadmath.so.0.0.0','libc.so.6','libm.so.6','ld-linux-x86-64.so.2'}
    runtime={name:checksum for name,checksum in built['runtime']['actual_pt_interp']['files'].items() if Path(name).name in names}
    assert len(runtime)==6
    files={str(Path(name).relative_to(ROOT)) if Path(name).is_relative_to(ROOT) else name:checksum for name,checksum in runtime.items()}
    tools={name:str(Path(shutil.which(name)).resolve()) for name in ('gfortran','cc','nm','taskset')}
    tools['python']=str(Path(sys.executable).resolve())
    for name in tools.values():files[name]=sha(name)
    for _,(name,checksum) in controller.SUPPORT.items():
        assert sha(ROOT/name)==checksum;files[name]=checksum
    helper=controller.support('loader')
    for name,checksum in helper.SUPPORT_PINS.items():
        path=helper.SUPPORT_ROOT/name;assert sha(path)==checksum
        files[str(path.relative_to(ROOT))]=checksum
    linker=helper._support('linker_bind.py')
    wrappers={}
    for name in ('gfortran','cc'):
        wrappers[name]=linker.wrapper_flags(tools[name],dict(os.environ))
        for path,record in wrappers[name]['inputs'].items():files[path]=record['sha256']
    proposal=ROOT/'.tirust-build/fortran-qp-output-bulk-proposal-1'
    assert sha(proposal/'index.json')==controller.PROPOSAL_SHA
    candidate=(proposal/'candidate-mtirust.f90').read_text()
    interface='''  ! Test-only branch observer; never a production/scientific candidate.
  interface
    subroutine bulk_observe(mode) bind(C,name="ml_bulk_observe")
      import c_int
      integer(c_int), value :: mode
    end subroutine
  end interface

'''
    anchor='  interface tirust_tensor_evaluate\n'
    assert candidate.count(anchor)==1
    observer=candidate.replace(anchor,interface+anchor)
    anchor='    if (eligible) then\n      ! Every internal-file record'
    assert observer.count(anchor)==1
    observer=observer.replace(anchor,'    if (eligible) then\n      call bulk_observe(1_c_int)\n      ! Every internal-file record')
    anchor='    do k=1,size(values)\n      call unpack(records(:,k),values(k),io_status)'
    assert observer.count(anchor)==1
    observer=observer.replace(anchor,'    call bulk_observe(0_c_int)\n'+anchor)
    text_new(HERE/'observer-mtirust.f90',observer)
    forced=observer.replace('      call bulk_observe(1_c_int)\n',
        '      call bulk_observe(1_c_int)\n      call bulk_observe(2_c_int)\n      io_status = 1\n      return\n')
    text_new(HERE/'forced-mtirust.f90',forced)
    files[str((proposal/'index.json').relative_to(ROOT))]=controller.PROPOSAL_SHA
    for name,meta in read(proposal/'index.json')['files'].items():files[str((proposal/name).relative_to(ROOT))]=meta['sha256']
    for path in ('vendor/tirust/mtirust.f90','vendor/tirust/src/lib.rs','vendor/tirust/Cargo.toml','tests/unit_tests/loop/test_tirust_transport_cache.py','tools/tirust_env.sh',
                 '.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/libmadloop_tirust.so',
                 '.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/tirust-config.json',
                 'docs/benchmarkfolder/experiments/fortran-qp-transport-profile-qualification-1/index.json'):
        files[path]=sha(ROOT/path)
    keys={'PATH','LIBRARY_PATH','GCC_EXEC_PREFIX','COMPILER_PATH','CPATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH',
          'FFLAGS','FCFLAGS','CFLAGS','CPPFLAGS','LDFLAGS','FC','CC'}|{k for k in os.environ if k.startswith('NIX_')}
    environment={key:os.environ.get(key) for key in sorted(keys)}
    pins={'schema':'madloop-tirust-bulk-unit-inputs-v1','tools':tools,'files':files,'environment':environment,
          'compiler_wrappers':wrappers,'expected_runtime':runtime,'profile_index_sha256':sha(profile/'index.json'),
          'production_module_flags':['-O3','-fPIC'],'driver_flags':['-O0','-fPIC','-fcheck=all','-frounding-math'],
          'provider':'recording mock compiled from local provider.c; no actual TIRust/Rust/model libraries',
          'compiler_executions':0,'native_executions':0}
    write_new(HERE/'input-pins.json',pins)
    print(json.dumps({'input_pins_sha256':sha(HERE/'input-pins.json'),'files':len(files),'cases':len(corpus['cases']),
                      'compiler_executions':0,'native_executions':0},sort_keys=True))
if __name__=='__main__':main()
