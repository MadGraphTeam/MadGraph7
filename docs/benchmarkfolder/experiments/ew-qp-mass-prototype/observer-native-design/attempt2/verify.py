#!/usr/bin/env python3
"""Portable source index and preparation-proof verifier; no native execution."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--working-root', type=Path)
    args = parser.parse_args();here = Path(__file__).resolve().parent
    index = json.loads((here / 'evidence-index.json').read_text())
    for name, expected in index['files'].items():
        path = here / name
        need(not path.is_symlink() and path.is_file(), 'Missing source: ' + name)
        need(sha(path) == expected, 'Source changed: ' + name)
    old = "CALL SETPARA('param_card.dat')"
    new = "CALL SETPARA('MadLoop5_resources/param_card.dat')"
    text = (here / 'check_observed.f').read_text()
    need(text.count(new) == 1 and old not in text, 'Wrong driver path correction')
    reconstructed = text.replace(new, old).encode()
    need(hashlib.sha256(reconstructed).hexdigest() ==
         '0d74401e5197aeed6ecdf97a6eabedb1854177b9b3226fd621e2dad845649382',
         'Driver contains changes beyond the literal')
    prepared = json.loads((here / 'prepared.json').read_text())
    need(len(prepared['build_jobs']) == 6 and len(prepared['build_outputs']) == 6, 'Unexpected build scope')
    need(sum('-c' in j['argv'] for j in prepared['build_jobs']) == 3, 'Unexpected compile count')
    for job in prepared['build_jobs']:
        if '-c' in job['argv']:
            source = Path(job['argv'][job['argv'].index('-c') + 1]).name
            need(source in ('check_observed.f', 'check_setpara.f'), 'Numerical object compilation requested')
    names = ('prepared.json', 'control.py', 'prepare.py', 'check_observed.f', 'check_setpara.f')
    parts = {name: sha(here / name) for name in names}
    review = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    need(review == index['review_hash'], 'Review binding changed')
    audited = json.loads((here / 'card-reader-audit.json').read_text())
    need(audited['all_numeric_tokens_within_reader_limit'] is True and
         audited['maximum_numeric_token_length'] <= audited['reader_value_character_limit'], 'Card token truncation risk')
    need(audited['card_sha256'] == audited['root_card_sha256'], 'Physical cards differ')
    checked = 0
    if args.working_root is not None:
        root = args.working_root.resolve()
        for section in ('artifacts', 'prior_evidence', 'source_files'):
            for name, expected in prepared[section].items():
                path = root / name
                need(path.resolve().is_relative_to(root), 'Path escaped working root')
                need(path.is_file() and sha(path) == expected, 'Working identity changed: ' + name)
                checked += 1
    print(json.dumps({'verified_source_files': len(index['files']), 'verified_working_artifacts': checked,
                      'review_hash': review, 'scope': 'source-only verification; no compiler/native invocation'}, sort_keys=True))


if __name__ == '__main__':
    main()
