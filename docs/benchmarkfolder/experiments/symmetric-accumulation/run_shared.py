#!/usr/bin/env python3
"""Run guarded QP product sharing in copies of the final fixed CLI exports."""
import importlib.util
from pathlib import Path
import sys

ROOT=next(p for p in Path(__file__).resolve().parents
          if (p/'madgraph/__init__.py').is_file())
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(HERE)]
import prototype

ORIGINAL_RUNNER=HERE.parent/'expanded-wl-sparsity/run.py'
spec=importlib.util.spec_from_file_location('sharing_full_process_runner',ORIGINAL_RUNNER)
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def modify(directory,variant):
    if variant=='original':
        return {}
    if variant not in ('shared','shared-trace'):
        raise RuntimeError('Unknown isolated sharing variant')
    files,proof=prototype.transform(directory,trace=variant=='shared-trace')
    runner.bench.dump(HERE/directory.parents[2].name/(variant+'-proof.json'),proof)
    return files


if __name__=='__main__':
    runner.HERE=HERE
    runner.EXPERIMENT='symmetric-product-sharing'
    runner.RUNNER_DEPENDENCIES=[
        Path(__file__).resolve(),ORIGINAL_RUNNER,HERE/'shared.py',
        HERE/'gather.py',HERE/'inventory.json']
    runner.bench.WORK=ROOT/'.benchmark-work/policy-v2/generation-abi3-terminal-fixed'
    runner.modify_variant=modify
    runner.main()
    # Custom counters are outside the timer. A trace run is structural evidence
    # only; copying its final pass does not turn instrumented times into evidence
    # for the uninstrumented prototype.
    for path in (runner.bench.WORK/'experiments'/runner.EXPERIMENT).glob(
            '*/shared-trace/SubProcesses/P*_*/sharing_trace.dat'):
        report=HERE/path.parents[3].name
        destination=report/'shared-trace-final.dat'
        if report.is_dir() and not destination.exists():
            runner.shutil.copy2(path,destination)
