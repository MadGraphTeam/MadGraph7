"""One owned bootstrap/single-measurement worker; no retry or timing claim."""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys

from admission import HERE, ROOT, helpers, read_json, require, save, sealed, sha, verify_record

OUTPUTS=('bootstrap-values.dat','driver-state.dat','driver-bits.dat','qualification.dat',
         'dimensions.dat','values.dat','timed_status.dat','timing.dat','trace.dat')


def completion_ok(error,cleanup_error,measured,captures,overflow,collection_errors):
    return bool(error is None and cleanup_error is None and measured and
                {x['boundary'] for x in captures}=={'READY','DONE'} and
                not overflow and not collection_errors)


def mapped_files(raw,expected):
    """Same inode/device/ELF admission used by the retained CMS supervisor."""
    elf={};data={}
    for line in raw.splitlines():
        fields=line.split(maxsplit=5)
        if len(fields)!=6 or not fields[5].startswith('/'):continue
        name=fields[5];require(not name.endswith(' (deleted)'),'Deleted runtime mapping')
        path=Path(name);stat=path.stat()
        require(int(fields[4])==stat.st_ino and
                tuple(int(x,16) for x in fields[3].split(':'))==(os.major(stat.st_dev),os.minor(stat.st_dev)),
                'Mapped inode/device mismatch')
        rawbytes=path.read_bytes();resolved=str(path.resolve());checksum=sha(rawbytes)
        if rawbytes[:4]==b'\x7fELF':
            require(expected.get(resolved)==checksum,'Unexpected mapped ELF '+resolved)
            elf[resolved]=checksum
        else:
            require(name.startswith('/nix/store/') and name.endswith('/lib/locale/locale-archive') and
                    fields[1]=='r--p','Unexpected non-ELF mapping')
            data[resolved]={'sha256':checksum,'bytes':len(rawbytes)}
    require(elf==expected,'Incomplete runtime mapping')
    return {'elf':elf,'data':data}


def execute(specpath):
    spec=read_json(specpath)
    sealed(HERE/'index.json',spec['source_index_sha256']);helpers()
    protocol=ROOT/'docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1/protocol'
    sys.path.insert(0,str(protocol))
    from tirust_benchmark_worker import Worker
    evidence=Path(spec['evidence']);directory=Path(spec['cwd'])
    request=spec['request'];captures=[];messages=[];output_overflow=[];close_receipts=[];instances=[]
    for path_key,pin_key in (('bootstrap_file','bootstrap_pin'),('point_file','point_pin'),
                             ('executable','executable_pin')):
        verify_record(spec[path_key],spec[pin_key])
    output=directory.parents[1]
    verify_record(output/'Cards/param_card.dat',spec['physical_card'])
    verify_record(output/'Cards/MadLoopParams.dat',spec['numerical_card'])

    class Observed(Worker):
        def __init__(self,*args,**kwargs):
            instances.append(self)
            super().__init__(*args,**kwargs)
        def close(self,graceful=True):
            receipt={'graceful_requested':graceful}
            try:
                super().close(graceful=graceful)
                receipt['complete']=True
            except BaseException as exc:
                receipt.update(complete=False,failure=repr(exc));raise
            finally:
                process=getattr(self,'process',None)
                receipt.update(pid=getattr(process,'pid',None),returncode=getattr(process,'returncode',None),
                               worker_closed=getattr(self,'closed',None))
                close_receipts.append(receipt)
        def _drain(self):
            total=0
            while True:
                chunk=self.process.stdout.read1(65536)
                if not chunk:return
                total+=len(chunk)
                if total>33554432:
                    output_overflow.append(total)
                    # This is our fresh child session, never an unrelated PID.
                    try:os.killpg(self.process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    return
                self.stream.write(chunk);self.stream.flush()
        def _message(self):
            result=super()._message();messages.append(result)
            require(not output_overflow,'Worker stdout byte limit exceeded')
            process=getattr(self,'process',None)
            # READY and DONE both await another command. BYE may already have
            # exited or become a zombie, so it is not a runtime-map boundary.
            if process is not None and result[0] in ('READY','DONE'):
                path=Path('/proc')/str(process.pid)/'maps'
                try:raw=path.read_text()
                except FileNotFoundError:raw=None
                if raw is not None:
                    captured=mapped_files(raw,spec['expected_runtime'])
                    name='maps-'+str(len(captures))+'.txt'
                    (evidence/name).write_text(raw)
                    captures.append({'boundary':result[0],'file':name,'sha256':sha(raw.encode()),**captured})
            return result

    worker=None;error=None;cleanup_error=None;measured=False;collection_errors=[]
    try:
        argv=[sys.executable,'-B',str(HERE/'limit_exec.py'),spec['executable'],
              spec['bootstrap_file'],spec['point_file'],str(request['requested_mode']),
              str(request['point']),str(int(request['trace_enabled']))]
        worker=Observed(argv,directory,evidence/'stdout.txt',env=os.environ.copy(),timeout=180)
        worker.timeout=300
        worker.run(1);measured=True
    except BaseException as exc:
        error={'class':type(exc).__name__,'message':str(exc)}
    finally:
        if worker is not None:
            try:worker.close(graceful=measured)
            except BaseException as exc:cleanup_error={'class':type(exc).__name__,'message':str(exc)}
        retained={}
        for name in OUTPUTS:
            try:
                path=directory/name
                if path.is_file():
                    raw=path.read_bytes()
                    require(len(raw)<=67108864,'Worker file exceeds reviewed byte cap')
                    (evidence/name).write_bytes(raw)
                    retained[name]={'bytes':len(raw),'sha256':sha(raw)}
                    # Keep one retained copy of potentially large trace output.
                    if name=='trace.dat':path.unlink()
            except BaseException as exc:
                collection_errors.append({'file':name,'failure':repr(exc)})
    maps_complete={x['boundary'] for x in captures}=={'READY','DONE'}
    if any(not x['complete'] for x in close_receipts):
        cleanup_error=cleanup_error or {'class':'CleanupFailure','message':'Worker close failed; see receipts'}
    for path_key,pin_key in (('bootstrap_file','bootstrap_pin'),('point_file','point_pin'),
                             ('executable','executable_pin')):
        try:verify_record(spec[path_key],spec[pin_key])
        except BaseException as exc:collection_errors.append({'file':path_key,'failure':repr(exc)})
    for filename,pin_key in (('param_card.dat','physical_card'),('MadLoopParams.dat','numerical_card')):
        try:verify_record(output/'Cards'/filename,spec[pin_key])
        except BaseException as exc:collection_errors.append({'file':filename,'failure':repr(exc)})
    complete=completion_ok(error,cleanup_error,measured,captures,output_overflow,collection_errors)
    process=getattr(instances[-1],'process',None) if instances else None
    complete=complete and bool(close_receipts) and all(x['complete'] for x in close_receipts) and getattr(process,'returncode',None)==0
    result={'request':request,'orchestration_complete':complete,
            'failure':error,'cleanup_failure':cleanup_error,'protocol':messages,
            'close_receipts':close_receipts,
            'native_pid':getattr(process,'pid',None),'native_returncode':getattr(process,'returncode',None),
            'runtime_maps':captures,'runtime_map_verified':maps_complete,
            'stdout_overflow':bool(output_overflow),'outputs':retained,
            'collection_errors':collection_errors,'trace_complete':complete,
            'numerically_qualified':False,'performance_measurement':False}
    save(evidence/'result.json',result)
    # Numerical failure is retained and examined by parity; controller failure
    # also stops the outer append-only stage with this complete failure record.
    require(result['orchestration_complete'],'Owned native request incomplete; evidence retained')
    return result


if __name__=='__main__':
    require(len(sys.argv)==2,'One reviewed request specification required')
    execute(Path(sys.argv[1]))
