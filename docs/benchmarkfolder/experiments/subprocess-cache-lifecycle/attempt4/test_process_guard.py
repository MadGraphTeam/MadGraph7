"""Fake-process tests of the actual inherited helpers; no child is launched."""
import importlib.util
from pathlib import Path
import signal
import tempfile
import types
import unittest

HERE=Path(__file__).resolve().parent


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


Guard=load(HERE/'process_guard.py','test_guard_module').ProcessGuard


class GuardTests(unittest.TestCase):
    def check_case(self,helper,phase,number):
        q=load(HERE.parent/'qualify.py','guard_fake_qualifier')
        events=[]
        with tempfile.TemporaryDirectory() as folder,Guard() as guard:
            q.REPORT=Path(folder)
            class Fake:
                returncode=None
                def wait(self,*args,**kwargs):
                    events.append('wait')
                    if phase=='during': guard.request(number,None)
                    self.returncode=0
                    return 0
                def communicate(self,*args,**kwargs):
                    events.append('communicate')
                    if phase=='during': guard.request(number,None)
                    self.returncode=0
                    return b'fake output',b''
            def spawn(*args,**kwargs):
                events.append('spawn')
                if phase=='spawn': guard.request(number,None)
                return Fake()
            def cleanup(process):
                events.append('cleanup')
                # A second termination request must not interrupt cleanup.
                guard.request(signal.SIGTERM,None)
                process.process.returncode=-15
                events.append('reaped')
            q.subprocess=types.SimpleNamespace(**vars(q.subprocess))
            q.subprocess.Popen=spawn
            q.terminate_owned_group=cleanup
            guard.bind(q)
            with self.assertRaises(InterruptedError):
                if helper=='command': q.command(['fake'],'fake',timeout=1,cpu=False)
                else: q.capture(['fake'],timeout=1)
            self.assertEqual(events[-2:],['cleanup','reaped'])
            self.assertEqual(events.count('spawn'),1)
            self.assertEqual(events.count('cleanup'),1)
            self.assertFalse(guard.spawning)
            self.assertFalse(guard.cleaning)
            metadata=list((Path(folder)/'execution').glob('*.json'))
            self.assertEqual(len(metadata),1)

    def test_deferred_spawn_and_running_interruptions(self):
        for helper in ('command','capture'):
            for phase in ('spawn','during'):
                for number in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):
                    with self.subTest(helper=helper,phase=phase,signal=number):
                        self.check_case(helper,phase,number)

    def test_handlers_restored(self):
        before={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
        with Guard(): pass
        self.assertEqual(before,{s:signal.getsignal(s) for s in before})


if __name__=='__main__': unittest.main(verbosity=2)
