#!/usr/bin/env python3
"""Regenerate separated tensor-pole fixtures using an installed COLLIER."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collier', type=Path, required=True, help='COLLIER installation prefix')
    parser.add_argument('--work-dir', type=Path, required=True, help='temporary compilation and raw-output directory')
    parser.add_argument('--output', type=Path, required=True, help='regenerated reference JSON')
    parser.add_argument('--fc', default=os.environ.get('FC', 'gfortran'))
    args = parser.parse_args()
    prefix, work, output = args.collier.resolve(), args.work_dir.resolve(), args.output.resolve()
    if work == HERE or HERE in work.parents:
        parser.error('--work-dir must be outside the retained test-source directory')
    work.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    libraries = [directory / name for directory in (prefix, prefix / 'lib')
                 for name in ('libcollier.so', 'libcollier.a')]
    library = next((p for p in libraries if p.exists()), None)
    if library is None:
        parser.error('cannot locate libcollier.so or libcollier.a below --collier')
    command = shlex.split(args.fc) + ['-O2', '-I'+str(prefix / 'include'),
               str(HERE / 'collier_poles.f90'), '-L'+str(library.parent), '-lcollier',
               '-Wl,-rpath,'+str(library.parent), '-o', str(work / 'collier_poles')]
    with (work / 'build.log').open('w') as stream:
        subprocess.run(command, cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=True)
    raw = work / 'collier_poles.txt'
    with (work / 'evaluation.log').open('w') as stream:
        subprocess.run([str(work / 'collier_poles'), str(HERE / 'collier_jobs.in'), str(raw)],
                       cwd=work, stdout=stream, stderr=subprocess.STDOUT, check=True)
    cases = json.loads((HERE / 'tensor_cases.json').read_text())
    results, current = [], None
    for line in raw.read_text().splitlines():
        fields = line.split()
        if fields[0] == 'JOB':
            current = dict(cases[int(fields[1])-1])
            current['collier_status'] = dict(error=int(fields[4]), accuracy=int(fields[5]))
            current['components'] = []
            results.append(current)
        elif fields[0] != 'ERR':
            values = list(map(float, fields[4:]))
            current['components'].append(dict(powers=list(map(int, fields[:4])),
                **{name: values[2*i:2*i+2] for i, name in enumerate(('finite', 'uv', 'ir1', 'ir2'))}))
    if len(results) != len(cases):
        raise RuntimeError('COLLIER output omitted one or more cases')
    for case in results:
        if case['collier_status'] != dict(error=0, accuracy=0):
            raise RuntimeError('COLLIER flagged ' + case['name'] + ': ' + str(case['collier_status']))
        if len(case['components']) != math.comb(case['rank']+4, 4):
            raise RuntimeError('incomplete COLLIER tensor for ' + case['name'])
    output.write_text(json.dumps(results, indent=2) + '\n')
    provenance = dict(library_sha256=sha(library),
                      compiler=subprocess.check_output(shlex.split(args.fc)+['--version'], text=True).splitlines()[0],
                      driver_sha256=sha(HERE / 'collier_poles.f90'),
                      inputs_sha256=sha(HERE / 'collier_jobs.in'),
                      cases_sha256=sha(HERE / 'tensor_cases.json'),
                      raw_output_sha256=sha(raw), reference_sha256=sha(output), command=command)
    output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(dict(cases=len(results), tensor_components=sum(len(c['components']) for c in results),
                         reference_sha256=sha(output)), indent=2))


if __name__ == '__main__':
    main()
