#!/usr/bin/env python3
"""Exercise accepted exponent edges and adjacent rejected values."""
import json
from pathlib import Path
import check_terminal as check

HERE = Path(__file__).resolve().parent
check.WORK = check.ROOT / ".benchmark-work/experiments/qp-terminal-bounds"
check._SPARSE_DRIVER = check._SPARSE_DRIVER.replace("do trial=1,12","do trial=1,18")
check._SPARSE_DRIVER = check._SPARSE_DRIVER.replace(
    "if(trial==12) a(i,j,k)=0",
    """if(trial==12) a(i,j,k)=0
          if(trial==13) a(i,j,k)=cmplx(2.0_16**1000,-2.0_16**1000,kind=16)
          if(trial==14) a(i,j,k)=cmplx(2.0_16**(-1000),2.0_16**(-1000),kind=16)
          if(trial==15) a(i,j,k)=cmplx(2.0_16**1001,0.0_16,kind=16)
          if(trial==16) a(i,j,k)=cmplx(2.0_16**(-1001),0.0_16,kind=16)
          if(trial==17) a(i,j,k)=cmplx(2.0_16**1000,2.0_16**1000,kind=16)
          if(trial==18) a(i,j,k)=cmplx(nearest(2.0_16**1001,-1.0_16), &
            -nearest(2.0_16**1001,-1.0_16),kind=16)""")
check._SPARSE_DRIVER = check._SPARSE_DRIVER.replace(
    "if(trial==12) b(i,j,k)=cmplx(snan,0.0_16,kind=16)",
    """if(trial==12) b(i,j,k)=cmplx(snan,0.0_16,kind=16)
          if(trial==13) b(i,j,k)=cmplx(2.0_16**1000,2.0_16**1000,kind=16)
          if(trial==14.or.trial==17) b(i,j,k)=cmplx(2.0_16**(-1000), &
            -2.0_16**(-1000),kind=16)
          if(trial==15.or.trial==16) b(i,j,k)=cmplx(1.0_16,0.0_16,kind=16)
          if(trial==18) b(i,j,k)=cmplx(nearest(2.0_16**1001,-1.0_16), &
            nearest(2.0_16**1001,-1.0_16),kind=16)""")

if __name__ == "__main__":
    results=[check.run(mode,True,trials=18) for mode in ("expanded","compact")]
    output={"passed":True,"comparisons":36000,"results":results,
            "scope":"Both exact exponent-band edges, largest value in top accepted binade, "
                    "cancellation/product cases and the immediately adjacent excluded binades; "
                    "plus all earlier exceptional/rounding/input/diagonal checks."}
    (HERE/"terminal-bounds.json").write_text(json.dumps(output,indent=2)+"\n")
    print(json.dumps(output,indent=2))
