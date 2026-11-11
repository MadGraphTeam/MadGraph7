"""Unchanged scoped ownership primitives copied from the pinned unit3 controller."""
import os
import signal
import subprocess
import time
CLEANUP=60.
def require(value,message):
    if not value:raise ValueError(message)

def cleanup_owned(owned,end):
    budget=min(55.,max(0.,end-time.monotonic()))
    result=owned.stop(cooperative_seconds=budget*45/55,terminate_seconds=budget*5/55,kill_seconds=budget*5/55)
    require(not result['survivors'],'Owned cleanup unresolved')
    return result

class StageDeadline:
    """One cap covers preflight, execution, cleanup and administrative closure."""
    def __enter__(self):
        self.end=time.monotonic()+self.cap
        self.work_expired=False;self.handoff=False;self.deferred=False
        self.previous=signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM,self.expired)
        signal.setitimer(signal.ITIMER_REAL,self.cap-CLEANUP)
        return self
    def __init__(self,cap):self.cap=cap
    def expired(self,number,frame):
        if self.handoff:self.deferred=True;return
        if not self.work_expired:
            self.work_expired=True
            signal.setitimer(signal.ITIMER_REAL,max(.001,self.end-time.monotonic()))
            raise TimeoutError('Unit work cutoff reached; original cleanup reserve starts')
        raise TimeoutError('Original unit all-in cap reached')
    def __exit__(self,*error):
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,self.previous)

def raw_cleanup(process,record,cleanup,end):
    """Failure-only path if OwnedChild construction raised after Popen."""
    events=[]
    if record is None:
        record=cleanup.identity(process.pid)
    if record is None and process.poll() is not None:
        return {'failure_only':True,'returncode':process.returncode,
                'cleanup_scope':'Root exited before identity capture; descendant closure unqualified',
                'uncertainty_blocks_later_stages':True}
    require(record is not None and record['pid']==record['process_group']==record['session'],
            'Raw child session identity unavailable; cleanup uncertainty blocks continuation')
    # The raw handle is retained before ownership construction. Only this
    # positively matched original child session is targeted on this failure path.
    for sig,seconds in ((signal.SIGTERM,2.),(signal.SIGKILL,2.)):
        if process.poll() is not None:break
        require(cleanup.matches(record),'Raw child start identity changed')
        os.killpg(record['process_group'],sig);events.append(signal.Signals(sig).name)
        try:process.wait(timeout=min(seconds,max(.001,end-time.monotonic())))
        except subprocess.TimeoutExpired:continue
    require(process.poll() is not None,'Raw child survived bounded termination')
    return {'failure_only':True,'events':events,'root':record,
            'cleanup_scope':'Raw identified child session; no qualification pass on ownership construction failure',
            'returncode':process.poll(),'uncertainty_blocks_later_stages':True}
