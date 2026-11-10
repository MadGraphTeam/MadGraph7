"""Apply target loader environment only at the final native execve boundary."""
import os
import sys


def target_environment(provider_directory, trace):
    result = os.environ.copy()
    for name in list(result):
        if name.startswith('LD_'):
            result.pop(name)
    result['LD_LIBRARY_PATH'] = provider_directory
    if trace:
        result['LD_TRACE_LOADED_OBJECTS'] = '1'
    return result


def main():
    provider_directory, trace, executable, *arguments = sys.argv[1:]
    if trace not in ('0', '1') or not os.path.isabs(executable):
        raise ValueError('Expected explicit trace flag and absolute target executable')
    os.execve(executable, [executable, *arguments], target_environment(provider_directory, trace == '1'))


if __name__ == '__main__':
    main()
