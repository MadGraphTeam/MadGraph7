"""Authenticate this private source inventory and replay four pure witnesses."""
from pathlib import Path
import argparse
import hashlib
import json

import derive_witnesses
import reference

HERE=Path(__file__).resolve().parent


def require(value,message):
    if not value:
        raise ValueError(message)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-sha256',required=True)
    arguments=parser.parse_args()
    path=HERE/'source-index.json'
    require(hashlib.sha256(path.read_bytes()).hexdigest()==arguments.source_sha256,'Source index changed')
    index=json.loads(path.read_text())
    require({str(p.relative_to(HERE)) for p in HERE.rglob('*') if p.is_file()}==
            set(index['files'])|{'source-index.json'},'Source inventory changed')
    for relative,expected in index['files'].items():
        item=HERE/relative
        require(not item.is_symlink() and item.stat().st_size==expected['bytes'] and
                hashlib.sha256(item.read_bytes()).hexdigest()==expected['sha256'],
                'Source payload changed: '+relative)
    witnesses=derive_witnesses.build()
    require(witnesses==json.loads((HERE/'witnesses.json').read_text()),'Sealed trace extraction changed')
    bits=[]
    for case in witnesses['cases']:
        result=reference.check(case['width'],case['width'],case['inferred_external_P_bits'],
                               case['coefficients_relative_to_repaired_P'],case['reconstructed_q_bits'])
        require(result['ok'] and result['error_exp']==[reference.ZERO_BOUND]*4,'Prospective equality failed')
        bits.append(result['absolute_sum_bits'])
    print(json.dumps({'source_files':len(index['files']),'conditional_witnesses':4,
        'component_absolute_sum_bits':bits,'Fortran_compilations':0,'new_native_executions':0,
        'observed_q_ABI':False,'snapshot_ownership_qualified':False,
        'physical_or_performance_qualification':False,'verified':True},sort_keys=True,indent=2))


if __name__=='__main__':
    main()
