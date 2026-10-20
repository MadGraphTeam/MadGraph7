#!/usr/bin/env python3
"""Exploratory kernel timing of the exact explicit-predecessor variant."""
import json
import shutil

import time_gather as timing


def main():
    check=timing.check
    root=check.HERE
    check.HERE=root/'unrolled-gather-v1'
    check.WORK=check.ROOT/'.benchmark-work/experiments/symmetric-gather-unrolled'
    for name in ('time_gather.py','time_unrolled.py','inventory.json','inventory.py'):
        shutil.copy2(root/name,check.HERE/name)
    timing.LOOP_COUNTS={8:5000,16:300}
    timing.main()
    path=check.HERE/'gather-timing.json'
    result=json.loads(path.read_text())
    result['variant']='Explicit static predecessor statements'
    result['runner_sha256']=check.sha(__file__)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
