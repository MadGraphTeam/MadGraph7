"""Same qualified Worker plus retained protocol and strict STOP/BYE drain."""
import importlib.util
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_ew_worker_base',HERE/'worker_base.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
class Worker(base.Worker):
    def __init__(self,*args,**kwargs):self.protocol=[];super().__init__(*args,**kwargs)
    def _send(self,line):self.protocol.append({'direction':'send','line':line});return super()._send(line)
    def _message(self):
        fields=super()._message();self.protocol.append({'direction':'receive','fields':fields});return fields
    def close(self,graceful=True):
        if self.closed:return
        if not graceful:return super().close(graceful=False)
        failure=None
        try:
            if self.process is None or self.process.poll() is not None:raise base.ProtocolError('Child exited before explicit STOP/BYE')
            self._send('STOP 0 0');previous=self.timeout;self.timeout=min(previous,10.)
            try:
                if self._message()!=['BYE']:raise base.ProtocolError('Missing exact BYE')
            finally:self.timeout=previous
            self.process.wait(timeout=10)
            if self.process.returncode:raise base.ProtocolError('Nonzero exit after BYE')
            trailing=bytes(self.buffer)
            while True:
                try:chunk=os.read(self.fd,65536)
                except BlockingIOError:break
                if not chunk:break
                trailing+=chunk
            if trailing:raise base.ProtocolError('Trailing protocol bytes after BYE')
            self.protocol.append({'direction':'exit','returncode':self.process.returncode,'trailing_bytes':0})
        except BaseException as error:failure=error
        finally:super().close(graceful=False)
        if failure is not None:raise failure
