#!/usr/bin/env python3
"""Revalidate association-only classifiers under standalone and NLO flags."""
import json
import os
from pathlib import Path

import check_terminal_bounds as bounds

HERE = Path(__file__).resolve().parent
check = bounds.check
check.WORK = check.ROOT/".benchmark-work/experiments/qp-terminal-associate-v2"


if __name__ == "__main__":
    os.sched_setaffinity(0, {2})
    results = []
    try:
        for mode in ("expanded", "compact"):
            for rounding, static in ((False, False), (True, False), (True, True)):
                results.append(check.run(mode, rounding, trials=18, no_automatic=static))
        result = {"passed": True, "comparisons": 108000, "results": results,
                  "scope": "Current ASSOCIATE finite classifier and association-only "
                           "bounded classifier. Exact consumed diagonal, input bits and "
                           "IEEE flags; full fallback outputs. Includes both accepted "
                           "exponent edges, rejected adjacent binades, exceptional "
                           "operands, all rounding modes and -fno-automatic."}
    except Exception as error:
        result = {"passed": False, "results": results, "error": repr(error)}
        raise
    finally:
        (HERE/"terminal-associate-v2.json").write_text(json.dumps(result, indent=2)+"\n")
        print(json.dumps(result, indent=2))
