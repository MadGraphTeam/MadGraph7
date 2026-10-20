#!/usr/bin/env python3
"""Inventory the immutable final QCD3/QCD4 exports without evaluating them."""
import json
from pathlib import Path

from inventory import ROOT,HERE,inventory,sha


def main():
    status_path=ROOT/'docs/benchmarkfolder/policy-v2/generation-abi3-terminal-fixed/status.json'
    status=json.loads(status_path.read_text())
    data={}
    for case in ('qcd-3','qcd-4'):
        record=status['cases'][case]
        assert record['status']=='complete'
        output=ROOT/record['output']
        data[case]=inventory(case,output)
        data[case]['source_output']=record['output']
        data[case]['generation_status_sha256']=sha(status_path)
        data[case]['generation_revision']=status['git_revision']
    data['scope']='Source-only final-export inventory. No numerical process or timing.'
    data['script_sha256']=sha(Path(__file__))
    data['inventory_script_sha256']=sha(HERE/'inventory.py')
    (HERE/'final-inventory.json').write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
    for case in ('qcd-3','qcd-4'):
        row=data[case]
        print(case,json.dumps({key:row[key] for key in ('call_counts',
            'compact_rank_one_calls','complex_products_after_literal_zero_skip_before_A_zero_skip',
            'possible_shared_products_same_A_and_B','same_A_B_product_fraction')},sort_keys=True))


if __name__=='__main__':
    main()
