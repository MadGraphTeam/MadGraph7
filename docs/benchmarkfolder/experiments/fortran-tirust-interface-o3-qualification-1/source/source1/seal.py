#!/usr/bin/env python3
"""Seal private source-only proposal/support text, without native execution."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    paths=set((ROOT/'tools').glob('tirust_benchmark*.py'))
    paths.add(ROOT/'tools/tirust_benchmark_worker.f')
    for base,names in (
        ('docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3',('loader_checks.py','worker_flags.py')),
        ('docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1',('process_cleanup.py','run.py')),
        ('docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1/ownership',('linker_bind.py','run.py','legacy_qualify.py'))):
        paths.update(ROOT/base/name for name in names)
    locale=Path('/nix/store/ig5ylfz7qcgpmvzip4pxfkp2pfi76s2j-glibc-locales-2.44-25/lib/locale/locale-archive')
    locale_sha='53851ab7caa5a0f893056762a6abff4e383ea06c0e6b276ee16876499e712d27'
    if digest(locale)!=locale_sha:raise ValueError('Frozen locale mapping changed')
    files={}
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.name!='source-index.json':
            if path.is_symlink():raise ValueError('Source symlink forbidden')
            raw=path.read_bytes();raw.decode('utf-8')
            files[str(path.relative_to(HERE))]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    result={'schema':'tirust-interface-only-o3-dormant-source-v1','files':files,
            'support':{str(path.relative_to(ROOT)):digest(path) for path in sorted(paths)},
            'allowed_extra_mappings':{str(locale):locale_sha},
            'candidate_additions_only_for_TIRUST_interface_f':['-O3','-fno-fast-math'],
            'coefficient_gate':'Existing MadLoop3e-6 criterion AND hard1e-8 ordinary target; raw word differences diagnostic.',
            'timed_repeat_gate':'Exact OWN-variant qualified words/accuracy/status; strict cohort and observed IEEE identity.',
            'production_changes':False,'Rust_builds':False,'native_executions':0,'compiler_invocations':0,
            'stages_executed':[],'performance_acceptance':False,'candidate_promoted':False}
    (HERE/'source-index.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(digest(HERE/'source-index.json'))


if __name__=='__main__':main()
