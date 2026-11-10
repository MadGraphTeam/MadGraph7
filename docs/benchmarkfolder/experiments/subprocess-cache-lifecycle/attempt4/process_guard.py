"""Defer termination only until the original runner owns a child handle."""
import signal
import types


class ProcessGuard:
    def __init__(self):
        self.pending=[]
        self.spawning=False
        self.cleaning=False
        self.previous={}

    def request(self,number,frame):
        if self.spawning or self.cleaning:
            self.pending.append(number)
        else:
            raise InterruptedError('Controller received signal '+str(number))

    def __enter__(self):
        for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
            self.previous[number]=signal.getsignal(number)
            signal.signal(number,self.request)
        return self

    def __exit__(self,*exception):
        for number,handler in self.previous.items():
            signal.signal(number,handler)

    def bind(self,module):
        popen=module.subprocess.Popen
        terminate=module.terminate_owned_group
        guard=self

        class Child:
            def __init__(self,process):
                self.process=process

            def __getattr__(self,name):
                return getattr(self.process,name)

            def enter_owned_scope(self):
                guard.spawning=False
                if guard.pending and not guard.cleaning:
                    number=guard.pending.pop(0)
                    raise InterruptedError('Controller received signal '+str(number))

            def wait(self,*args,**kwargs):
                self.enter_owned_scope()
                return self.process.wait(*args,**kwargs)

            def communicate(self,*args,**kwargs):
                self.enter_owned_scope()
                return self.process.communicate(*args,**kwargs)

        def spawn(*args,**kwargs):
            guard.spawning=True
            try:
                # Defer signals through return/assignment, until wait or
                # communicate enters the original runner's cleanup try.
                return Child(popen(*args,**kwargs))
            except BaseException:
                guard.spawning=False
                raise

        def cleanup(process):
            guard.cleaning=True
            try:
                return terminate(process)
            finally:
                guard.cleaning=False

        # A private proxy avoids altering Python's shared subprocess module.
        module.subprocess=types.SimpleNamespace(**vars(module.subprocess))
        module.subprocess.Popen=spawn
        module.terminate_owned_group=cleanup
