"""Produce only a private source preview and exact patch; never compile or run it."""
from pathlib import Path
import difflib
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

def sha(data):
    return hashlib.sha256(data).hexdigest()

def transform(original, helper):
    old = '  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite\n'
    new = ('  use, intrinsic :: ieee_arithmetic, only: ieee_is_finite, ieee_round_type, &\n'
           '    ieee_get_rounding_mode, ieee_nearest, operator(==)\n'
           '  use, intrinsic :: ieee_exceptions, only: ieee_flag_type, ieee_invalid, ieee_overflow, &\n'
           '    ieee_divide_by_zero, ieee_underflow, ieee_inexact, ieee_support_halting, ieee_get_halting_mode\n')
    assert original.count(old) == 1
    result = original.replace(old,new)
    marker = '  subroutine tensor_mp(n,rank,shifts,invariants,masses,mu2,coefs,status, &\n'
    assert result.count(marker) == 1
    result = result.replace(marker,helper+'\n'+marker)
    old = '''    do k=1,8*count
      call unpack(output(:,k),values(k),ios)
      if (ios /= 0) then
        ! Native insertion precedes this public Fortran transport check.
        ! Reject all numerical cache entries when decoding the result fails.
        ios = clear_cache(context)
        status = TIRUST_ERROR
        local_error = 'TIRust returned an invalid or nonfinite binary128 output record.'
        return
      endif
    enddo
'''
    new = '''    call unpack_outputs(output,values,finite == 1_c_int .and. poles == 0_c_int,ios)
    if (ios /= 0) then
      ! Native insertion precedes this public Fortran transport check.
      ! Reject all numerical cache entries when decoding the result fails.
      ios = clear_cache(context)
      status = TIRUST_ERROR
      local_error = 'TIRust returned an invalid or nonfinite binary128 output record.'
      return
    endif
'''
    assert result.count(old) == 1
    return result.replace(old,new)

def main():
    original = (ROOT/'vendor/tirust/mtirust.f90').read_bytes()
    helper = (HERE/'helper.f90').read_text()
    profile = ROOT/'docs/benchmarkfolder/experiments/fortran-qp-transport-profile-qualification-1'
    assert sha((profile/'index.json').read_bytes()) == '4abe6049fea625c567c37668956f3a8d1db71222eb2575bbf13fd32617e087d2'
    assert original == (profile/'science/original-source/mtirust.f90').read_bytes()
    candidate = transform(original.decode(),helper)
    outputs = {
        'original-mtirust.f90': original,
        'candidate-mtirust.f90': candidate.encode(),
        'proposal.diff': ''.join(difflib.unified_diff(original.decode().splitlines(True), candidate.splitlines(True),
                              fromfile='a/vendor/tirust/mtirust.f90',tofile='b/vendor/tirust/mtirust.f90')).encode(),
    }
    for name,data in outputs.items():
        path=HERE/name
        if path.exists():
            assert path.read_bytes()==data, f'No source-overwrite: {name}'
        else:
            path.write_bytes(data)
    config = ROOT/'.tirust-build/qp-transport-install/mg/HEPTools/tirust/lib/tirust-config.json'
    bindings = {
        'schema':'madloop-tirust-output-bulk-source-proposal-v1',
        'status':'source-only-uncompiled-unqualified',
        'profile_index_sha256':sha((profile/'index.json').read_bytes()),
        'profile_analysis_sha256':sha((profile/'analysis.json').read_bytes()),
        'production_revision':'0fcd2cd873d61d134837e1d9e87afb13a47824fd',
        'scientific_source_sha256':'3c13ad88b825d9d19b1030b62b317c0f31759e002d8dfdb9050142301a7cae24',
        'native_dso_sha256':'eb3d67fdc99b742b1c5d66101522c8b8fb97e74b0b71ac69fd4452fb60c42af9',
        'installed_config':{'path':str(config.relative_to(ROOT)),'sha256':sha(config.read_bytes())},
        'source_original':{'path':'vendor/tirust/mtirust.f90','sha256':sha(original)},
        'source_preview_sha256':sha(candidate.encode()),
        'runtime_changes':False,'compiler_executions':0,'native_executions':0,'performance_acceptance':False,
    }
    q=HERE/'bindings.json';data=(json.dumps(bindings,indent=2,sort_keys=True)+'\n').encode()
    if q.exists(): assert q.read_bytes()==data
    else: q.write_bytes(data)
    print(json.dumps(bindings,sort_keys=True))

if __name__ == '__main__': main()
