#!/usr/bin/env python3
"""Set keys of an mg7 run_card.toml in place, each within its own table:

    set_run_card.py Cards/run_card.toml run.device='["hip"]' generation.events=10000

The value is written as given (TOML syntax: quote strings). Only the key of the
named table changes: `[systematics] enable` and `[vegas] enable` are different
keys, which a sed on `^enable =` would not tell apart. A key that is not in its
table is an error.
"""
import re
import sys


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
    missing = set(assignments) - done
    if missing:
        sys.exit('set_run_card.py: no %s in the card' % ', '.join(
            '[%s] %s' % key for key in sorted(missing)))
    return '\n'.join(lines)


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
    with open(argv[1], 'w') as fsock:
        fsock.write(set_keys(text, assignments))


if __name__ == '__main__':
    main(sys.argv)
