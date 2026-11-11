"""Private replay limits; then replace this process with the pinned driver."""
import os
from pathlib import Path
import resource
import sys


def main():
    if len(sys.argv) < 2 or not Path(sys.argv[1]).is_absolute():
        raise ValueError('Explicit absolute owned executable required')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_AS, (4294967296, 4294967296))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16777216, 16777216))
    os.execv(sys.argv[1], sys.argv[1:])


if __name__ == '__main__':
    main()
