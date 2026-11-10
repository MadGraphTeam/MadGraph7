#!/usr/bin/env python3
"""Offline strict mock-journal gate. Does not launch compiler or native code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_text(text,case_data):
    actual=[line.split() for line in text.splitlines()]
    expected=[line.split() for line in case_data['expected_journal_rows']]
    if len(actual)!=len(expected):raise ValueError('Incomplete or extra caller mock journal rows')
    for row,(a,b) in enumerate(zip(actual,expected,strict=True),1):
        if a!=b:raise ValueError('Caller mock journal mismatch at row %d: %r != %r'%(row,a,b))
    return dict(passed=True,cases=len(case_data['cases'])*2,
                scope='Mock scalar/coefficient control flow and rendered private CT adapter only')

def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--journal',type=Path,default=HERE/'fixture/run/caller-mock.log')
    cli.add_argument('--events',type=Path)
    args=cli.parse_args()
    plan=json.loads((HERE/'fixture/commands.json').read_text())
    for name,digest in plan['inputs'].items():
        if sha(ROOT/name)!=digest:raise ValueError('Source changed: '+name)
    if args.events:
        report=json.loads(args.events.read_text())
        jobs=[job for job in report['jobs'] if job['name']=='caller-mock-control-flow']
        if len(jobs)!=1:raise ValueError('Missing/duplicate mock process event')
        event=jobs[0]
        if event['returncode']!=0 or event['timed_out'] or event.get('error'):raise ValueError('Mock process failed')
        if event.get('output_files',{}).get('caller-mock.log')!=sha(args.journal):raise ValueError('Journal binding mismatch')
    result=validate_text(args.journal.read_text(),json.loads((HERE/'fixture/cases.json').read_text()))
    result.update(journal_sha256=sha(args.journal),commands_sha256=sha(HERE/'fixture/commands.json'),
                  native_exit_checked=args.events is not None,
                  physical_accuracy_qualified=False,full_madloop_rescue_qualified=False)
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__=='__main__':main()
