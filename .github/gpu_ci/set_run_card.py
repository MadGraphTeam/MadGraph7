#!/usr/bin/env python3
"""Set keys of an mg7 run_card.toml in place, each within its own table:

    set_run_card.py Cards/run_card.toml run.device='["hip"]' generation.events=1000

The value is written as given (TOML syntax: quote strings). Only the key of the
named table changes: `[systematics] enable` and `[vegas] enable` are different
keys, which a sed on `^enable =` would not tell apart. A key that is not in the
card (a hidden option group, e.g. [gridpack]) is added to its table if the
run_card template has it, and is an error otherwise.
"""
import os
import re
import sys

TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'madgraph',
                        'iolibs', 'template_files', 'mg7', 'run_card.toml')


def template_keys():
    with open(TEMPLATE) as fsock:
        return set(tuple(key.split('.', 1))
                   for key in re.findall(r'%\(([\w.]+)\)s', fsock.read()))


def set_keys(text, assignments):
    lines, table, done = text.split('\n'), None, set()
    for i, line in enumerate(lines):
        header = re.match(r'^\s*\[+\s*([^\]]+?)\s*\]+\s*(#.*)?$', line)
        if header:
            table = header.group(1)
            continue
        for (section, key), value in assignments.items():
            if table == section and re.match(r'^\s*%s\s*=' % re.escape(key), line):
                lines[i] = '%s = %s' % (key, value)
                done.add((section, key))
    text = '\n'.join(lines)
    missing = set(assignments) - done
    unknown = missing - template_keys()
    if unknown:
        sys.exit('set_run_card.py: no %s in the card or its template' % ', '.join(
            '[%s] %s' % key for key in sorted(unknown)))
    for section, key in sorted(missing):
        line = '%s = %s' % (key, assignments[(section, key)])
        header = re.search(r'(?m)^\[%s\][ \t]*\n' % re.escape(section), text)
        if header:
            text = text[:header.end()] + line + '\n' + text[header.end():]
        else:
            text = text.rstrip('\n') + '\n\n[%s]\n%s\n' % (section, line)
    return text


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    assignments = {}
    for arg in argv[2:]:
        name, equal, value = arg.partition('=')
        section, _, key = name.rpartition('.')
        if not section or not key or not equal:
            sys.exit('set_run_card.py: expected table.key=value, got %r' % arg)
        assignments[(section, key)] = value
    with open(argv[1]) as fsock:
        text = fsock.read()
    # set_keys first: it exits on an unknown key, which must leave the card as it was
    text = set_keys(text, assignments)
    with open(argv[1], 'w') as fsock:
        fsock.write(text)


if __name__ == '__main__':
    main(sys.argv)
