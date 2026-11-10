#!/usr/bin/env python3
"""One owned stress request; invoked only by the separately admitted supervisor."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
PROPOSAL = HERE.with_name('physical-stress-current-source-proposal-1')
sys.path.insert(0, str(PROPOSAL / 'protocol'))
from tirust_benchmark_worker import Worker, parse_outputs, RAW_OUTPUTS

def save(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')

def qualified(parsed, failure, warnings, maps_ok, unexpected):
    return (failure is None and parsed is not None and not warnings and maps_ok and not unexpected
            and all(0 <= row['accuracy'] <= 1e-6 for row in parsed['rows']))

def main():
    spec_path = Path(sys.argv[1]); spec = json.loads(spec_path.read_bytes())
    evidence = spec_path.parent; directory = Path(spec['cwd'])
    expected = spec['expected_runtime']
    captures=[]
    class Observed(Worker):
        def _message(self):
            try: return super()._message()
            finally:
                if self.process is not None:
                    try:
                        raw=Path('/proc/%d/maps'%self.process.pid).read_text()
                        target=evidence/('maps-%03d.txt'%len(captures));target.write_text(raw)
                        paths=set()
                        for line in raw.splitlines():
                            fields=line.split(maxsplit=5)
                            if len(fields)!=6 or not fields[5].startswith('/'):continue
                            if fields[5].endswith(' (deleted)'):raise ValueError('Deleted runtime mapping')
                            path=Path(fields[5])
                            with path.open('rb') as stream: header=stream.read(4)
                            if header==b'\x7fELF':paths.add(str(path.resolve()))
                        hashes={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in sorted(paths)}
                        captures.append({'path':target.name,'sha256':hashlib.sha256(raw.encode()).hexdigest(),'elf_files':hashes,'exact_expected':hashes==expected})
                    except FileNotFoundError:
                        pass # An exited failure is retained, never accepted without a complete map.
    failure=None; worker=None;parsed=None;start=time.monotonic()
    try:
        worker=Observed(spec['argv_without_fifo'],directory,evidence/'stdout.txt',env=os.environ.copy(),timeout=180.)
        worker.run(1)
        parsed=parse_outputs(directory,[spec['backend']],2,1)
    except Exception as error:
        failure={'class':type(error).__name__,'message':str(error)}
    finally:
        if worker is not None:worker.close()
        raw={name:(directory/name).read_text(errors='replace') for name in RAW_OUTPUTS if (directory/name).is_file()}
        for name,text in raw.items():(evidence/name).write_text(text)
    console=(evidence/'stdout.txt').read_text(errors='replace') if (evidence/'stdout.txt').exists() else ''
    warnings=[line for line in console.splitlines() if '##W03' in line]
    maps_ok=bool(captures) and any(item['exact_expected'] for item in captures)
    unexpected=[item for item in captures if set(item['elf_files'])-set(expected)]
    passed=qualified(parsed,failure,warnings,maps_ok,unexpected)
    save(evidence/'result.json',{'schema_version':1,'request':spec['request'],'orchestration_complete':True,
        'numerically_qualified':passed,'failure':failure,'incomplete_warnings':warnings,'actual_map_captures':captures,
        'runtime_identity_failure':bool(unexpected),'complete_runtime_map':maps_ok,'observation':parsed,
        'raw_outputs_sha256':{name:hashlib.sha256(text.encode()).hexdigest() for name,text in raw.items()},
        'elapsed_seconds':time.monotonic()-start,'phase_timeout_seconds':180,'performance_eligible':False,
        'analytic_EW_oracle':'not requalified: current repaired geometry not captured'})

if __name__=='__main__':main()
