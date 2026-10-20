#!/usr/bin/env python3
"""Choose a permitted idle physical core before observing any trial timings."""
import argparse
import json
import os
from pathlib import Path
import time

import quiet_factor as monitor


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    allowed=sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,{2})
    candidates=[]
    for cpu in allowed:
        if cpu in (2,3,4,5,206,207,211,223):
            continue
        base=Path(f"/sys/devices/system/cpu/cpu{cpu}")
        if (base/"topology/thread_siblings_list").read_text().strip()!=str(cpu):
            continue
        candidates.append(cpu)
    snapshots=[]
    for i in range(3):
        snapshots.append(monitor.snapshot(candidates+[207]))
        if i<2:
            time.sleep(3)
    ranking=[]
    for cpu in candidates:
        fractions=[]
        for a,b in zip(snapshots,snapshots[1:]):
            delta=[y-x for x,y in zip(a["cpu_jiffies"][f"cpu{cpu}"],
                                     b["cpu_jiffies"][f"cpu{cpu}"])]
            total=sum(delta[:8])
            fractions.append((total-delta[3]-delta[4])/total)
        ranking.append(dict(cpu=cpu,busy_fractions=fractions,maximum=max(fractions)))
    ranking.sort(key=lambda r:(r["maximum"],r["cpu"]))
    eligible=[r for r in ranking if r["maximum"]<.02]
    result=dict(allowed_cpus=allowed,single_thread_candidates=candidates,
        snapshots=snapshots,ranking=ranking,selected_cpu=eligible[0]["cpu"] if eligible else None,
        scope="Deterministic idle-core selection before timing; two3-second "
              "windows, maximum busy fraction<2%, then lowest CPU number. "
              "Previously rejected cores and reserved207 excluded.")
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    assert eligible,"No suitable preflight core; no timing launched"
    print(result["selected_cpu"],eligible[0]["busy_fractions"])


if __name__=="__main__":
    main()
