import ctypes
import logging
import os
import platform

# pre-load libmadspace
ctypes.CDLL(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "lib",
        "libmadspace.dylib" if platform.system() == "Darwin" else "libmadspace.so",
    ),
    mode=ctypes.RTLD_GLOBAL,
)

from ._madspace_py import *


def _init():
    """
    Monkey-patch classes for a more pythonic experience.
    """
    set_lib_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

    def call_and_convert(runtime, args):
        if len(args) == 0:
            tensorlib = "numpy"
        else:
            tensorlib = type(args[0]).__module__
        outputs = runtime.call(args)
        # Convert outputs, lazy-loading torch or numpy
        if tensorlib == "torch":
            import torch

            return tuple(torch.from_dlpack(out) for out in outputs)
        else:
            import numpy

            return tuple(numpy.from_dlpack(out) for out in outputs)

    def runtime_call(self, *args):
        outputs = call_and_convert(self, args)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def function_call(self, *args):
        if not hasattr(self, "runtime"):
            self.runtime = FunctionRuntime(self)
        outputs = call_and_convert(self.runtime, args)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def function_generator_call(self, *args):
        if not hasattr(self, "runtime"):
            self.runtime = FunctionRuntime(self.function())
        outputs = call_and_convert(self.runtime, args)
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs

    def map_forward(self, inputs, conditions=[]):
        if not hasattr(self, "forward_runtime"):
            self.forward_runtime = FunctionRuntime(self.forward_function())
        outputs = call_and_convert(self.forward_runtime, [*inputs, *conditions])
        return outputs[:-1], outputs[-1]

    def map_inverse(self, inputs, conditions=[]):
        if not hasattr(self, "inverse_runtime"):
            self.inverse_runtime = FunctionRuntime(self.inverse_function())
        outputs = call_and_convert(self.inverse_runtime, [*inputs, *conditions])
        return outputs[:-1], outputs[-1]

    def tensor_numpy(tensor):
        import numpy  # Lazy-load numpy, to make it optional dependency

        return numpy.from_dlpack(tensor)

    def tensor_torch(tensor):
        import torch  # Lazy-load torch, to make it optional dependency

        return torch.from_dlpack(tensor)

    py_logger = logging.getLogger("madspace")

    def log_handler(level, message):
        match level:
            case Logger.level_debug:
                py_logger.debug(message)
            case Logger.level_info:
                py_logger.info(message)
            case Logger.level_warning:
                py_logger.warning(message)
            case Logger.level_error:
                py_logger.error(message)

    FunctionRuntime.__call__ = runtime_call
    Function.__call__ = function_call
    FunctionGenerator.__call__ = function_generator_call
    Mapping.map_forward = map_forward
    Mapping.map_inverse = map_inverse
    Tensor.numpy = tensor_numpy
    Tensor.torch = tensor_torch
    # Logger.set_log_handler(log_handler)


_init()
