"""Pure ELF64/Linux DT_NEEDED/RUNPATH closure, constrained to prebound files.

This intentionally narrow static derivation is not a substitute for the actual
ELF interpreter's subsequent --list result. Unsupported lookup mechanisms fail
before that query. No subprocess or dynamic library load occurs in this module.
"""
from collections import deque
import hashlib
from pathlib import Path
import struct

ALLOWED_SONAMES={
    'ld-linux-x86-64.so.2','libgfortran.so.5','libgcc_s.so.1',
    'libquadmath.so.0','libm.so.6','libc.so.6',
}
REQUIRED_SONAMES=ALLOWED_SONAMES-{'libquadmath.so.0'}


def require(value,message):
    if not value:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(path):
    path=Path(path).resolve();raw=path.read_bytes()
    require(raw[:7]==b'\x7fELF\x02\x01\x01','Expected ELF64 little-endian current GNU runtime')
    require(len(raw)>=64,'Truncated ELF header')
    header=struct.unpack_from('<HHIQQQIHHHHHH',raw,16)
    require(header[1]==62 and header[8]==56 and header[9]<1024,'Unsupported ELF machine/program headers')
    require(header[4]+header[8]*header[9]<=len(raw),'Truncated program headers')
    programs=[struct.unpack_from('<IIQQQQQQ',raw,header[4]+i*header[8]) for i in range(header[9])]
    for p in programs:require(p[2]+p[5]<=len(raw),'Program segment exceeds ELF bytes')
    interpreters=[raw[p[2]:p[2]+p[5]] for p in programs if p[0]==3]
    require(len(interpreters)<=1,'Duplicate ELF interpreter')
    interpreter=None
    if interpreters:
        require(interpreters[0].endswith(b'\0'),'Unterminated interpreter')
        interpreter=interpreters[0][:-1].decode('ascii')
        require(Path(interpreter).is_absolute(),'Relative interpreter')
    dynamic=[p for p in programs if p[0]==2]
    require(len(dynamic)==1 and dynamic[0][5]%16==0,'Missing/duplicate/malformed dynamic segment')
    entries=[]
    for offset in range(dynamic[0][2],dynamic[0][2]+dynamic[0][5],16):
        tag,value=struct.unpack_from('<qQ',raw,offset)
        if tag==0:break
        entries.append((tag,value))
    else:raise ValueError('Unterminated dynamic segment')
    strings=[v for t,v in entries if t==5];sizes=[v for t,v in entries if t==10]
    require(len(strings)==len(sizes)==1,'Missing/duplicate dynamic string table')
    segments=[p for p in programs if p[0]==1 and p[3]<=strings[0] and strings[0]+sizes[0]<=p[3]+p[5]]
    require(len(segments)==1,'Dynamic strings not uniquely file-backed')
    start=segments[0][2]+strings[0]-segments[0][3]
    table=raw[start:start+sizes[0]]
    def text(value):
        require(value<len(table),'Dynamic string offset out of range')
        end=table.find(b'\0',value)
        require(end>=0,'Unterminated dynamic string')
        return table[value:end].decode('ascii')
    def optional(tag):
        values=[text(v) for t,v in entries if t==tag]
        require(len(values)<=1,'Duplicate singleton dynamic tag')
        return values[0] if values else None
    return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest(),
            'interpreter':interpreter,'needed':[text(v) for t,v in entries if t==1],
            'soname':optional(14),'rpath':optional(15),'runpath':optional(29)}


def derive(executable,prebound):
    """Derive the unit's actual dependency set; fail before an unbound load."""
    bound=prebound['bound']
    executable=Path(executable).resolve();root=describe(executable)
    require(root['interpreter'] is not None,'Unit executable lacks interpreter')
    loader=Path(root['interpreter']).resolve()
    require(bound.get(str(loader))==digest(loader),'Interpreter was not prebound')
    initial=describe(loader)
    require(initial['soname']=='ld-linux-x86-64.so.2','Unexpected unit interpreter SONAME')
    require(not initial['needed'],'Interpreter dependencies need a separate qualified resolver')
    files={str(loader):initial['sha256']};loaded={initial['soname']:str(loader)}
    queue=deque([root]);objects=[initial];resolutions=[]
    while queue:
        obj=queue.popleft();objects.append(obj)
        # The pinned GNU/Nix graph uses simple SONAMEs and absolute DT_RUNPATH.
        # Do not guess inherited RPATH, cache/default, ORIGIN or hwcap behavior.
        require(not obj['rpath'],'DT_RPATH lookup is outside this unit resolver scope')
        paths=[]
        for value in (obj['runpath'] or '').split(':'):
            if not value:continue
            require(Path(value).is_absolute() and '$' not in value,'Nonabsolute/tokenized RUNPATH')
            require(not (Path(value)/'glibc-hwcaps').exists(),'glibc-hwcaps lookup needs a separate qualified resolver')
            paths.append(value)
        for needed in obj['needed']:
            require(needed in ALLOWED_SONAMES,'Unexpected unit dependency SONAME: '+needed)
            if needed in loaded:
                resolutions.append({'requester':obj['path'],'needed':needed,'selected':loaded[needed],
                                    'reason':'already-loaded SONAME'})
                continue
            candidates=[Path(directory)/needed for directory in paths]
            found=next((p for p in candidates if p.is_file()),None)
            require(found is not None,'Dependency needs unsupported cache/default/inherited lookup: '+needed)
            selected=found.resolve()
            checksum=digest(selected)
            require(bound.get(str(selected))==checksum,'Selected runtime provider was not prebound: '+str(selected))
            child=describe(selected)
            require(child['soname']==needed,'Resolved provider SONAME differs')
            loaded[needed]=str(selected);files[str(selected)]=checksum;queue.append(child)
            resolutions.append({'requester':obj['path'],'needed':needed,'selected':str(selected),
                                'reason':'first existing absolute RUNPATH candidate',
                                'search_paths':paths,'selected_spelling':str(found)})
    require(REQUIRED_SONAMES<=set(loaded),'Expected base Fortran unit runtime dependencies absent')
    for path,checksum in files.items():require(bound.get(path)==digest(path)==checksum,'Provider changed during pure derivation')
    require(digest(executable)==root['sha256'],'Executable changed during pure derivation')
    return {'schema':'madloop-tirust-unit-static-runtime-closure-v1',
            'files':dict(sorted(files.items())),'sonames':dict(sorted(loaded.items())),
            'objects':objects,'resolutions':resolutions,
            'actual_interpreter_query_still_required':True,
            'scope':'ELF64 x86_64 simple SONAME/absolute DT_RUNPATH, no LD_ overrides, cache/default/RPATH/ORIGIN/hwcaps lookup',
            'compiler_or_loader_invocations':0}


def validate_expected(derived,proposed):
    """Never accept a profile-only provider as the unit's expected closure."""
    require(derived['files']==proposed,'Proposed runtime differs from unit-specific dependency closure')
