"""Apply the reviewed worker resource limits, then replace this process."""
import os
import resource
import sys

if __name__=='__main__':
    if len(sys.argv)<2:
        raise SystemExit('Explicit owned executable required')
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_AS,(8589934592,8589934592))
    resource.setrlimit(resource.RLIMIT_FSIZE,(67108864,67108864))
    os.execv(sys.argv[1],sys.argv[1:])
