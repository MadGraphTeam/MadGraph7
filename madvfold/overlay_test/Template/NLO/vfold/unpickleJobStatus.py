#! /bin/env python3
# Copyright (C) 2026 CERN.
# Licensed under the GNU Lesser General Public License (version 3 or later).
# Created by A. Valassi (Sep 2026) for project MadVFold (arXiv:2609.23500).

import sys
argc = len(sys.argv)
if argc == 2:
    pkl = sys.argv[1]
else:
    print('Usage:   %s <path/job_status.pkl>'%sys.argv[0])
    print('Example: %s SubProcesses/job_status.pkl'%sys.argv[0])
    sys.exit(1)

import pickle
data = []
with open(pkl, 'rb') as f:
    while True:
        try:
            data.append(pickle.load(f))
        except EOFError:
            break

assert len(data) == 1
gfs = data[0]
assert type(gfs) == list
gfsSorted = sorted(gfs, key=lambda d: float(d['channel']))
for gf in gfsSorted:
    print(gf)
