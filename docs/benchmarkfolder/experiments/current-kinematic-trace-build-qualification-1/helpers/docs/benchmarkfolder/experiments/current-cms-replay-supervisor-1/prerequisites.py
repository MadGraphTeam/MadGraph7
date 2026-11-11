"""Read-only dependency admission and explicit, dereferenced private data copies."""
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil

HERE=Path(__file__).resolve().parent
DATA=json.loads((HERE/'prerequisites.json').read_bytes())
def require(ok,message):
    if not ok:raise RuntimeError(message)
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def check_file(path,record):
    path=Path(path)
    require(path.is_file() and path.stat().st_size==record['size'] and digest(path)==record['sha256'],'Prebuilt dependency changed/missing: '+str(path))
    resolved=record.get('resolved',record.get('resolved_donor'))
    require(resolved is None or str(path.resolve())==resolved,'Prebuilt dependency target changed: '+str(path))
    if record.get('executable'):require(os.access(path,os.X_OK),'Dependency helper is not executable: '+str(path))
def external_hashes():
    for name,record in DATA['external'].items():check_file(name,record)
    return {name:{'sha256':record['sha256'],'size':record['size']} for name,record in DATA['external'].items()}
def check_donors():
    external_hashes()
    for source,entry in DATA['sources'].items():
        for name in entry['required']:require(name in entry['copies'],'Required prebuilt not declared: '+name)
        for record in entry['copies'].values():check_file(record['donor'],record)
        log=entry['copies']['vendor/CutTools/includects/compiler_version.log']['donor']
        require(Path(log).read_text().strip()==DATA['CT_version'],'CutTools compiler version differs')
    # Exact pinned configuration bytes are checked above. Keep explicit ABI/QP
    # and link-file contracts too; this function launches no helper executable.
    for name,abi in (('ninjars',4),('tirust',1)):
        config_path=next(Path(p) for p in DATA['external'] if p.endswith('/'+name+'-config.json'))
        config=json.loads(config_path.read_bytes());lib=config_path.parent
        require(config['abi_version']==abi and config['quad_precision'] is True and config['link_order_version']==1,'Bridge ABI/QP configuration differs')
        require(bool(config['link_before']),'Empty compiler-runtime link metadata')
        for suffix,key in (('native-libs','native_libs'),('link-before','link_before')):
            require(shlex.split((lib/(name+'-'+suffix+'.txt')).read_text())==config[key],'Bridge link metadata mismatch')
        for artifact,checksum in config.get('artifact_sha256',{}).items():require(digest(lib/artifact)==checksum,'Bridge artifact/config mismatch')

def copy_private(source,mg):
    entry=DATA['sources'][source];copies=entry['copies'];mg=Path(mg)
    for name in entry['required']:require(name in copies,'Required prebuilt not declared: '+name)
    for name,record in copies.items():
        origin=Path(record['donor']);check_file(origin,record);dest=mg/name
        require(dest.resolve().is_relative_to(mg.resolve()),'Private prebuilt path escapes clone')
        require(not dest.is_symlink(),'Private prebuilt must be a real file')
        if dest.exists():require(digest(dest)==record['sha256'],'Prebuilt would overwrite tracked bytes')
        else:
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(origin,dest)
    check_private(source,mg)
    return copies

def check_private(source,mg):
    entry=DATA['sources'][source];mg=Path(mg)
    for name in entry['required']:require(name in entry['copies'],'Required prebuilt not declared: '+name)
    for name,record in entry['copies'].items():
        path=mg/name
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(mg.resolve()) and digest(path)==record['sha256'],'Private prebuilt changed/missing: '+name)
        check_file(record['donor'],record)
    require((mg/'vendor/CutTools/includects/compiler_version.log').read_text().strip()==DATA['CT_version'],'Private CutTools compiler version differs')

def logical_values(text):
    values=[]
    for token in re.sub(r'\s+','',text).split(','):
        match=re.fullmatch(r'(?:(\d+)\*)?(\.TRUE\.|\.FALSE\.)',token)
        require(match is not None,'Malformed generated availability data')
        values.extend([match.group(2)=='.TRUE.']*int(match.group(1) or 1))
    return values

def check_generated(source,output,subprocess):
    output=Path(output)
    require(not any((output/'Source'/name).exists() for name in ('CutTools','IREGI')),'Unexpected internal dependency tree')
    definitions=output/'SubProcesses/MadLoop_makefile_definitions'
    require(definitions.is_file() and '<Not_found_define_it_yourself>' not in definitions.read_text(),'Missing generated external include directory')
    path=output/subprocess/'loop_matrix.f'
    text=' '.join(line[6:] for line in path.read_text().splitlines() if line and line[0] not in 'Cc*!').upper()
    match=re.search(r'DATA\s+LOOPLIBS_AVAILABLE\s*/([^/]+)/',text)
    require(match is not None,'Missing generated reducer availability')
    available=logical_values(match.group(1))
    expected=[True,False,True,False,False,True,True,True]+([True] if source=='candidate' else [])
    require(available==expected,'Generated reducer availability differs')
    qp={}
    for first,last,values in re.findall(r'DATA\s+LOOPLIBS_QPAVAILABLE\((\d+)(?::(\d+))?\)\s*/([^/]+)/',text):
        start=int(first);end=int(last or first);parsed=logical_values(values)
        require(len(parsed)==end-start+1 and not any(i in qp for i in range(start,end+1)),'Malformed QP availability range')
        qp.update(zip(range(start,end+1),parsed))
    required_qp={i:i in (1,6,8,9) for i in range(len(expected)+1)}
    require(qp==required_qp,'Generated QP capability differs; no silent precision downgrade')
    return {'available':available,'qp_available':qp,'definition_sha256':digest(definitions),'loop_matrix_sha256':digest(path)}
