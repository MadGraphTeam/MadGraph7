#!/usr/bin/env python3
"""Snapshot reviewed inputs and explicit recipes, without compiling/executing."""
from pathlib import Path
import difflib
import hashlib
import json
import shutil
import sys
from instrument import instrument

sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]
BASE=ROOT/'.tirust-build/ew-qp-mass-prototype'
WORK=BASE/'observer-native-design'
PROCESS=Path('SubProcesses/P0_epem_ttxa')
COMPILER=Path('/nix/store/5km6kqfhjr16939ak9l1crzsh6x96f68-gfortran-wrapper-15.2.0/bin/gfortran')
LIBGCC=Path('/nix/store/g825z4hlkyiqm8zpv4kbjkvw5fl901pz-gfortran-15.2.0/lib/gcc/x86_64-unknown-linux-gnu/15.2.0/libgcc.a')
OBJECTS='check_sa.o MadLoopParamReader.o MadLoopCommons.o polynomial.o loop_matrix.o improve_ps.o born_matrix.o CT_interface.o loop_num.o helas_calls_ampb_1.o mp_compute_loop_coefs.o mp_helas_calls_ampb_1.o coef_construction_1.o coef_construction_2.o loop_CT_calls_1.o mp_coef_construction_1.o mp_coef_construction_2.o TIR_interface.o COLLIER_interface.o TIRUST_interface.o mg5_citation.o ew_trace.o'.split()
CHANGED=('CT_interface.f','TIRUST_interface.f','loop_CT_calls_1.f','mp_compute_loop_coefs.f')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(1024*1024),b''):h.update(data)
    return h.hexdigest()


def rel(path):return str(path.relative_to(ROOT))


def main():
    assert Path.cwd().resolve()==ROOT
    assert not WORK.exists() and not (HERE/'prepared.json').exists()
    package=HERE.parent
    assert digest(package/'evidence-index.json')=='9c9725ed41fe6484bbfa7eaa14f5143e91bda235771ab281fa7ecfd8ddec2234'
    package_index=json.loads((package/'evidence-index.json').read_text())
    for name,h in package_index['files'].items():assert digest(package/name)==h,name
    assert digest(COMPILER)=='30c6495dcb18cc324629fc230302d9cd8d81143f0ca1635d752b3f8552d48cae'
    bound_native=json.loads((package/'native-identities.json').read_text())['artifacts']
    for name,info in bound_native.items():assert digest(ROOT/name)==info['sha256'],name
    historical=json.loads((ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication/compilation.json').read_text())['objects_archives_after']
    WORK.mkdir()
    origins={}
    def copy(source,dest):
        dest.parent.mkdir(parents=True,exist_ok=True)
        assert not dest.exists()
        shutil.copyfile(source,dest)
        assert digest(source)==digest(dest)
        origins[rel(dest)]={'source':str(source),'sha256':digest(dest)}
    # All includes/modules and libraries become regular private snapshots.
    include_sources=[BASE/'original/Source/MODEL',BASE/'original/Source/DHELAS',
                     ROOT/'HEPTools/ninja/include',ROOT/'HEPTools/include',
                     ROOT/'HEPTools/ninjars/lib',ROOT/'HEPTools/tirust/lib']
    include_dirs=[]
    for i,source in enumerate(include_sources):
        dest=WORK/'includes'/str(i);dest.mkdir(parents=True)
        include_dirs.append(dest)
        for f in sorted(source.rglob('*')):
            if f.is_file() and f.suffix.lower() in ('.inc','.mod','.h'):
                copy(f,dest/f.relative_to(source))
    lib_sources=[BASE/'original/lib'/n for n in ('libdhelas.a','libmodel.a','libcts.a','libiregi.a')]
    lib_sources += [ROOT/'HEPTools/ninja/lib/libninja.so.0.0.0',ROOT/'HEPTools/ninja/lib/libavh_olo.a',
                    ROOT/'HEPTools/collier/libcollier.so',LIBGCC,
                    ROOT/'HEPTools/ninjars/lib/libninjars.a',ROOT/'HEPTools/tirust/lib/libtirust.a',
                    BASE/'original/private-libraries/libmadloop_tirust.so']
    library={}
    for source in lib_sources:
        name='libninja.so.0' if source.name=='libninja.so.0.0.0' else source.name
        dest=WORK/'libraries'/name;copy(source,dest);library[name]=dest
    assert digest(library['libmadloop_tirust.so'])=='49af12f0b88252a2d06420a2ea8768529eb5ab51de34e3db2257ee61cd71ca12'
    linklibs=[str(library[n]) for n in ('libdhelas.a','libmodel.a','libcts.a','libiregi.a','libninja.so.0','libavh_olo.a','libcollier.so','libgcc.a')]
    linklibs += ['-lgcc_s',str(library['libninjars.a']),'-ldl','-lgcc_s','-lutil','-lrt','-lpthread','-lm','-ldl','-lc','-lgcc_s',
                 str(library['libtirust.a']),str(library['libmadloop_tirust.so']),'-Wl,-rpath,'+str(WORK/'libraries'),'-lstdc++']
    arms={}
    for arm in ('original','candidate'):
        source=BASE/arm/PROCESS;dest=WORK/arm/PROCESS;dest.mkdir(parents=True)
        for name in OBJECTS:
            assert digest(source/name)==historical[str(PROCESS/name)],name
            copy(source/name,dest/name)
        for f in sorted(source.iterdir()):
            if f.is_file() and f.suffix.lower() in ('.inc','.mod'):copy(f,dest/f.name)
        resources=source/'MadLoop5_resources'
        for f in sorted(resources.rglob('*')):
            if f.is_file():copy(f,dest/'MadLoop5_resources'/f.relative_to(resources))
        copy(package/'inputs/param_card.dat',dest/'param_card.dat')
        copy(package/'inputs/selected.dat',dest/'selected.dat')
        inputs={name:(source/name).read_text() for name in CHANGED}
        changed=instrument(inputs,arm=='candidate')
        snapshots=HERE/'sources'/arm;snapshots.mkdir(parents=True)
        for name,text in changed.items():
            (dest/name).write_text(text)
            (snapshots/name).write_text(text)
            (snapshots/(name+'.patch')).write_text(''.join(difflib.unified_diff(inputs[name].splitlines(keepends=True),text.splitlines(keepends=True),fromfile='frozen/'+name,tofile='observed/'+name)))
        for name in ('ew_mass_observer.f90','ew_mass_snapshot.f','check_observed.f'):
            copy(HERE/name,dest/name)
        order=('ew_mass_observer.f90','ew_mass_snapshot.f','check_observed.f')+CHANGED
        commands=[];outputs=[]
        for name in order:
            obj='check_sa.o' if name=='check_observed.f' else str(Path(name).with_suffix('.o'))
            flags=['-w','-fPIC']+(['-ffixed-line-length-132'] if name.endswith('.f') else [])
            argv=[str(COMPILER),*flags,'-J'+str(dest),'-I'+str(dest)]
            argv += ['-I'+str(directory) for directory in include_dirs]
            argv += ['-c',str(dest/name),'-o',str(dest/obj)]
            commands.append(argv);outputs.append(rel(dest/obj))
        outputs += [rel(dest/'ew_mass_observer.mod'),rel(dest/'check_observed')]
        link=[str(COMPILER),'-w','-fPIC','-ffixed-line-length-132','-o',str(dest/'check_observed')]
        link += [str(dest/name) for name in OBJECTS]
        link += [str(dest/'ew_mass_observer.o'),str(dest/'ew_mass_snapshot.o'),*linklibs]
        arms[arm]={'process_dir':rel(dest),'compile_commands':commands,'link_command':link,
                   'executable':rel(dest/'check_observed'),'build_outputs':outputs}
    record={'schema':'ew-qp-mass-observer-prepare-v1','state':'source_only_uncompiled',
            'work':rel(WORK),'source_package_index_sha256':digest(package/'evidence-index.json'),
            'compiler':{'path':str(COMPILER),'sha256':digest(COMPILER)},
            'arms':arms,'runtime_mutable_inputs':[],
            'expected_runtime':{name:{'path':rel(library[name]),'sha256':digest(library[name])}
                                for name in ('libninja.so.0','libcollier.so','libmadloop_tirust.so')},
            'artifacts':{rel(f):digest(f) for f in sorted(WORK.rglob('*')) if f.is_file()},
            'snapshot_origins':origins,'compiler_invocations':0,'native_reducer_calls':0,
            'original_objects_pinned_to':'docs/benchmarkfolder/experiments/ew-threshold-adjudication/compilation.json',
            'historical_compilation_sha256':digest(ROOT/'docs/benchmarkfolder/experiments/ew-threshold-adjudication/compilation.json'),
            'scope':'Two matched observer arms, fixed QP transport only; model, arithmetic and full256-bit native guard retained.'}
    (HERE/'prepared.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'prepared_files':len(record['artifacts']),'build_commands_per_arm':8,'compiler_invocations':0,'native_reducer_calls':0},indent=2))


if __name__=='__main__':main()
