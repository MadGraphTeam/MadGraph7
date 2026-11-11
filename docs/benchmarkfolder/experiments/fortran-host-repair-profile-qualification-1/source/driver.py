"""Reuse the successful exact-word/IEEE worker, changing only hook API names."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
DONOR=ROOT/'docs/benchmarkfolder/experiments/fortran-dp-n1-profile-proposal-2/prepare.py'


def derive(text):
    spec=importlib.util.spec_from_file_location('repair_driver_donor',DONOR)
    donor=importlib.util.module_from_spec(spec);sys.modules[spec.name]=donor;spec.loader.exec_module(donor)
    result=donor.worker(text)
    result=result.replace('USE DP_PROFILE_STATE','USE HOST_REPAIR_PROFILE_STATE')
    result=result.replace('DP_PROFILE_START(TICK)','REPAIR_PROFILE_BEGIN(3,TICK)')
    result=result.replace('DP_PROFILE_END(8,TICK)','REPAIR_PROFILE_END(3,TICK)')
    result=result.replace('DP_PROFILE_', 'REPAIR_PROFILE_')
    result=result.replace('DP_EMPTY_PROBE_LOOP','REPAIR_EMPTY_PROBE_LOOP')
    if 'DP_PROFILE' in result or 'DP_EMPTY_PROBE' in result:
        raise ValueError('Old tensor profiling hooks remain')
    if result.count('CALL REPAIR_PROFILE_BEGIN(3,TICK)')!=1:
        raise ValueError('Empty probe must use one explicit repair begin/end bracket')
    return result
