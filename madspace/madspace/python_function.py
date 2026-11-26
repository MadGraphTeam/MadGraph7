import contextlib
import ctypes
import types

from ._madspace_py import (
    _UMAMI_INPUT_KEYS,
    _UMAMI_META_KEYS,
    _UMAMI_OUTPUT_KEYS,
    Context,
    DataType,
    _tensor_view,
)

# input name: (data type, shape after the batch dimension); "n" is the particle count
_INPUT_SHAPES = {
    "momenta": (DataType.float64, ("n", 4)),
    "alpha_s": (DataType.float64, ()),
    "flavor_index": (DataType.int32, ()),
    "random_color": (DataType.float64, ()),
    "random_helicity": (DataType.float64, ()),
    "random_diagram": (DataType.float64, ()),
    "helicity_index": (DataType.int32, ()),
    "diagram_index": (DataType.int32, ()),
    "channel_index": (DataType.int32, ()),
    "particle_count": (DataType.int32, ()),
    "pdg_ids": (DataType.int32, ("n",)),
    "momentum_fraction": (DataType.float64, ()),
    "scale": (DataType.float64, ()),
    "pdg_id": (DataType.int32, ()),
    "beam_index": (DataType.int32, ()),
    "random_flavor": (DataType.float64, ()),
    "parton_densities": (DataType.float64, (2, 13)),
    "spin_legs": (DataType.int32, (2,)),
    "spin_vectors": (DataType.float64, (2, 4)),
    "g_ew": (DataType.float64, ()),
}
# output name: (data type, shape after the batch dimension); "n" is the particle
# count, "d" the diagram count, "o" the squared-order count and "p" the number of
# pairs of coloured legs
_OUTPUT_SHAPES = {
    "matrix_element": (DataType.float64, ()),
    "diagram_amp2": (DataType.float64, ("d",)),
    "color_index": (DataType.int32, ()),
    "helicity_index": (DataType.int32, ()),
    "diagram_index": (DataType.int32, ()),
    "cut": (DataType.float64, ()),
    "ren_scale": (DataType.float64, ()),
    "fact_scale1": (DataType.float64, ()),
    "fact_scale2": (DataType.float64, ()),
    "pdf": (DataType.float64, ()),
    "bias": (DataType.float64, ()),
    "matrix_element_ratio": (DataType.float64, ()),
    "flavor_index": (DataType.int32, ()),
    "squared_orders": (DataType.float64, ("o",)),
    "color_correlated": (DataType.float64, ("o", "p")),
    "spin_correlated": (DataType.float64, ("o",)),
    "loop": (DataType.float64, ("o", 3)),
}


class PythonFunction:
    """
    A UMAMI implementation that calls a Python function on numpy arrays or torch
    tensors, for example a custom cut. Add it to a context with
    ``Context.add_matrix_element``, then use the returned ``MatrixElementApi`` like a
    loaded library.

    ``function`` is called once per batch, with one array per entry of ``inputs``,
    in that order. The arrays have the batch size as first dimension:
    ``momenta`` has shape ``(batch, n, 4)``, ``pdg_ids`` has shape ``(batch, n)``,
    ``spin_legs`` ``(batch, 2)``, ``spin_vectors`` ``(batch, 2, 4)`` and all other
    inputs have shape ``(batch,)``. Floats are 64-bit, integers 32-bit. The
    function returns one array per entry of ``outputs``, a single array for one
    output and a tuple for several, of shape ``(batch,)``, or for
    ``diagram_amp2`` ``(batch, diagram_count)`` and for the provisional NLO outputs
    ``squared_orders`` and ``spin_correlated`` ``(batch, o)``,
    ``color_correlated`` ``(batch, o, p)`` and ``loop`` ``(batch, o, 3)``, with
    ``o`` the ``squared_order_count`` and ``p`` the number of pairs i < j of the
    coloured legs, flattened as in umami.h.

    The arrays are views of madspace's buffers on the device of the context, so they
    must not be kept after the call. numpy functions only run on the CPU, torch
    functions also on CUDA and HIP GPUs, on the stream of the madspace runtime.

    Args:
        function: The Python function.
        inputs: Names of the UMAMI inputs passed to ``function``.
        outputs: Names of the UMAMI outputs ``function`` returns.
        framework: ``"numpy"`` or ``"torch"``.
        particle_count: The particle count ``n`` for ``momenta`` and
            ``pdg_ids``, if the caller does not pass the ``particle_count``
            input. Reported as metadata, so that a ``MatrixElement`` can take
            it from the ``MatrixElementApi``.
        name: Name used in error messages, by default the function's name.
        diagram_count: The number of diagrams of ``diagram_amp2``, also reported
            as metadata.
        squared_order_count: The number of squared coupling orders of the NLO
            outputs, also reported as metadata.
        color_representations: The colour representation of each leg (1, 3,
            -3 or 8, all taken as outgoing), whose legs other than 1 are the
            coloured legs of ``color_correlated``. Needs ``particle_count`` and
            is reported as metadata.
        parameters: Names and default values of parameters set through
            ``umami_set_parameter``, e.g. the ``parameters`` argument of
            ``Context.add_matrix_element``. With parameters, ``function`` gets a
            namespace of their values as last argument, e.g. ``params.pt_min``.
    """

    def __init__(
        self,
        function,
        inputs,
        outputs,
        framework="numpy",
        particle_count=None,
        name=None,
        parameters=None,
        diagram_count=None,
        squared_order_count=None,
        color_representations=None,
    ):
        if framework not in ("numpy", "torch"):
            raise ValueError(f"framework must be 'numpy' or 'torch', not {framework!r}")
        for key in inputs:
            if key not in _INPUT_SHAPES:
                raise ValueError(
                    f"unknown input {key!r}, expected one of {', '.join(_INPUT_SHAPES)}"
                )
        for key in outputs:
            if key not in _OUTPUT_SHAPES:
                raise ValueError(
                    f"unknown output {key!r}, expected one of {', '.join(_OUTPUT_SHAPES)}"
                )
        if color_representations is not None:
            color_representations = [int(r) for r in color_representations]
            if particle_count is None or len(color_representations) != particle_count:
                raise ValueError(
                    "color_representations needs particle_count and one value per "
                    "particle"
                )
            colored = sum(r != 1 for r in color_representations)
            color_pairs = colored * (colored - 1) // 2
        else:
            color_pairs = None
        sizes = dict(d=diagram_count, o=squared_order_count, p=color_pairs)
        names = dict(
            d="diagram_count", o="squared_order_count", p="color_representations"
        )
        for key in outputs:
            for size in _OUTPUT_SHAPES[key][1]:
                if size in sizes and sizes[size] is None:
                    raise ValueError(f"output {key!r} needs {names[size]}")
        if len(outputs) == 0:
            raise ValueError("at least one output is needed")
        self.function = function
        self.inputs = list(inputs)
        self.outputs = list(outputs)
        self.framework = framework
        self.particle_count = particle_count
        self.diagram_count = diagram_count
        self.squared_order_count = squared_order_count
        self.color_representations = color_representations
        self.color_pair_count = color_pairs
        self.name = name or getattr(function, "__name__", "python_function")
        self.parameters = (
            None
            if parameters is None
            else {str(key): float(value) for key, value in parameters.items()}
        )
        self._input_keys = [_UMAMI_INPUT_KEYS[key] for key in self.inputs]
        self._output_keys = [_UMAMI_OUTPUT_KEYS[key] for key in self.outputs]
        self._needs_particle_count = any(
            "n" in _INPUT_SHAPES[key][1] for key in self.inputs
        ) or any("n" in _OUTPUT_SHAPES[key][1] for key in self.outputs)
        self._meta = {
            _UMAMI_META_KEYS[key]: int(value)
            for key, value in (
                ("particle_count", particle_count),
                ("diagram_count", diagram_count),
                ("squared_order_count", squared_order_count),
            )
            if value is not None
        }
        self._array_meta = (
            {}
            if color_representations is None
            else {_UMAMI_META_KEYS["color_representations"]: color_representations}
        )
        if self._needs_particle_count and particle_count is None:
            # read from the particle_count input, even if the function does not take it
            self._input_keys.append(_UMAMI_INPUT_KEYS["particle_count"])
        if framework == "torch":
            import torch

            self._torch = torch
            self._from_dlpack = torch.from_dlpack
        else:
            import numpy

            self._numpy = numpy
            self._from_dlpack = numpy.from_dlpack

    def _view(self, address, dtype, shape, stride, offset, device):
        return self._from_dlpack(
            _tensor_view(address, dtype, shape, stride, offset, *device)
        )

    def _numpy_output(self, address, dtype, count, offset):
        # numpy.from_dlpack gives read-only arrays, so map the CPU buffer directly
        ctype = ctypes.c_double if dtype == DataType.float64 else ctypes.c_int32
        size = ctypes.sizeof(ctype)
        buffer = (ctype * count).from_address(address + offset * size)
        return self._numpy.frombuffer(buffer, dtype=self._numpy.dtype(ctype))

    def _output_shape(self, key, n):
        sizes = dict(
            n=n,
            d=self.diagram_count,
            o=self.squared_order_count,
            p=self.color_pair_count,
        )
        return [sizes.get(size, size) for size in _OUTPUT_SHAPES[key][1]]

    def _stream(self, device, stream):
        if self.framework == "torch" and device[0] != "cpu" and stream is not None:
            torch = self._torch
            # torch names HIP devices "cuda" as well
            torch_device = torch.device("cuda", device[1])
            return torch.cuda.stream(
                torch.cuda.ExternalStream(stream, device=torch_device)
            )
        return contextlib.nullcontext()

    def _call(
        self,
        device_name,
        device_index,
        count,
        stride,
        offset,
        inputs,
        outputs,
        stream,
        parameter_values=None,
    ):
        """Called by the UMAMI implementation with the raw buffer addresses."""
        device = (device_name, device_index)
        with self._stream(device, stream):
            n = self.particle_count
            if self._needs_particle_count and n is None:
                counts = self._view(
                    inputs[_UMAMI_INPUT_KEYS["particle_count"]],
                    DataType.int32,
                    [count],
                    [1],
                    offset,
                    device,
                )
                n = int(counts[0])
            args = []
            for key in self.inputs:
                dtype, shape = _INPUT_SHAPES[key]
                shape = [count] + [n if size == "n" else size for size in shape]
                # the batch dimension is contiguous, the others are strided by it
                strides = [1] + [
                    stride * _prod(shape[1:i]) for i in range(1, len(shape))
                ]
                args.append(
                    self._view(
                        inputs[_UMAMI_INPUT_KEYS[key]],
                        dtype,
                        shape,
                        strides,
                        offset,
                        device,
                    )
                )
            if self.parameters is not None:
                args.append(types.SimpleNamespace(**(parameter_values or {})))
            results = self.function(*args)
            if len(self.outputs) == 1:
                results = (results,)
            elif not isinstance(results, (tuple, list)) or len(results) != len(
                self.outputs
            ):
                raise TypeError(
                    f"{self.name} must return a tuple of {len(self.outputs)} arrays "
                    f"({', '.join(self.outputs)})"
                )
            for key, result in zip(self.outputs, results):
                address = outputs.get(_UMAMI_OUTPUT_KEYS[key])
                if address is None:
                    continue
                dtype = _OUTPUT_SHAPES[key][0]
                shape = self._output_shape(key, n)
                self._check(key, result, dtype, [count] + shape, device)
                # the batch dimension is contiguous, the others are strided by it,
                # the first one fastest
                if self.framework == "torch":
                    strides = [1] + [
                        stride * _prod(shape[:i]) for i in range(len(shape))
                    ]
                    self._view(
                        address, dtype, [count] + shape, strides, offset, device
                    ).copy_(result)
                else:
                    columns = result.reshape(count, -1, order="F")
                    for k in range(columns.shape[1]):
                        self._numpy_output(address, dtype, count, offset + k * stride)[
                            :
                        ] = columns[:, k]

    def _check(self, key, result, dtype, shape, device):
        prefix = f"output {key!r} of {self.name}"
        if self.framework == "torch":
            torch = self._torch
            if not isinstance(result, torch.Tensor):
                raise TypeError(
                    f"{prefix} must be a torch.Tensor, not {type(result).__name__}"
                )
            expected = torch.float64 if dtype == DataType.float64 else torch.int32
            result_device = "cpu" if result.device.type == "cpu" else "gpu"
            if result_device != ("cpu" if device[0] == "cpu" else "gpu"):
                raise ValueError(
                    f"{prefix} must be on the {device[0]} device, not {result.device}"
                )
        else:
            numpy = self._numpy
            if not isinstance(result, numpy.ndarray):
                raise TypeError(
                    f"{prefix} must be a numpy.ndarray, not {type(result).__name__}"
                )
            expected = numpy.dtype(
                numpy.float64 if dtype == DataType.float64 else numpy.int32
            )
        if result.dtype != expected:
            raise TypeError(f"{prefix} must have dtype {expected}, not {result.dtype}")
        if tuple(result.shape) != tuple(shape):
            raise ValueError(
                f"{prefix} must have shape {tuple(shape)}, not {tuple(result.shape)}"
            )


def _prod(values):
    result = 1
    for value in values:
        result *= value
    return result


def _add_matrix_element(self, function, name=None, parameters=None):
    """
    Add a matrix element implemented in Python, see ``PythonFunction``. Returns its
    ``MatrixElementApi``. ``parameters`` maps parameter names to the values that
    replace the defaults of ``PythonFunction``.
    """
    if not isinstance(function, PythonFunction):
        raise TypeError("add_matrix_element takes a madspace.PythonFunction")
    required = list(function._input_keys)
    return self._add_python_function(
        function,
        function._input_keys,
        required,
        function._output_keys,
        function.framework,
        name or function.name,
        list((function.parameters or {}).items()),
        {str(key): float(value) for key, value in (parameters or {}).items()},
        function._meta,
        function._array_meta,
    )


Context.add_matrix_element = _add_matrix_element
