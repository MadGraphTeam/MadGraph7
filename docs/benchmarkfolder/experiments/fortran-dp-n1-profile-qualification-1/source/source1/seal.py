#!/usr/bin/env python3
"""Seal proposal text/support files. No compiler, loader or numerical call."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    support=set((ROOT/'tools').glob('tirust_benchmark*.py'))
    support|={ROOT/'tools/tirust_benchmark_worker.f',ROOT/'tools/tirust_env.sh'}
    donor=ROOT/'docs/benchmarkfolder/experiments/fortran-qp-transport-profile-controller-3'
    support|={donor/n for n in ('controller.py','worker_flags.py','loader_checks.py')}
    common=ROOT/'docs/benchmarkfolder/campaigns/invariant-symmetry-1/preparation-1'
    support|={common/n for n in ('process_cleanup.py','run.py')}
    ownership=ROOT/'docs/benchmarkfolder/experiments/current-cms-replay-supervisor-1/ownership'
    support|={ownership/n for n in ('linker_bind.py','run.py','legacy_qualify.py')}
    locale=Path('/nix/store/ig5ylfz7qcgpmvzip4pxfkp2pfi76s2j-glibc-locales-2.44-25/lib/locale/locale-archive')
    expected_locale='53851ab7caa5a0f893056762a6abff4e383ea06c0e6b276ee16876499e712d27'
    if digest(locale)!=expected_locale:raise ValueError('Frozen locale mapping changed')
    files={str(p.relative_to(HERE)):{'sha256':digest(p),'bytes':p.stat().st_size}
           for p in sorted(HERE.rglob('*')) if p.is_file() and p.name!='source-index.json'}
    value={'schema':'current-n1-dp-profile-source-only-v1','files':files,
           'support':{str(p.relative_to(ROOT)):digest(p) for p in sorted(support)},
           'allowed_extra_mappings':{str(locale):expected_locale},
           'native_executions':0,'compiler_invocations':0,'stages_executed':[],
           'performance_acceptance':False}
    (HERE/'source-index.json').write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    print(digest(HERE/'source-index.json'))


if __name__=='__main__':main()
