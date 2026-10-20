#!/usr/bin/env python3
"""Run the same complete synthetic oracle on explicit predecessor statements."""
import json
import shutil

import check_gather as check

HERE = check.HERE
generator = check.gather.generate


def main():
    target = HERE/'unrolled-gather-v1'
    target.mkdir(exist_ok=True)
    for name in ('gather.py','check_gather.py','check_unrolled.py'):
        shutil.copy2(HERE/name,target/name)
    check.WORK = check.ROOT/'.benchmark-work/experiments/symmetric-gather-unrolled'
    check.HERE = target
    check.gather.generate = lambda *args,**kwargs: generator(*args,**kwargs,unroll=True)
    check.main()
    path = target/'gather-exact.json'
    result = json.loads(path.read_text())
    result['variant']='Explicit predecessor statements; independent M,K order retained.'
    result['runner_sha256']=check.sha(__file__)
    path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    main()
