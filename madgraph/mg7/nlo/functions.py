"""The integrands and the matrix elements of a fixed-order NLO run of
madspace as functions of Python, for machine learning and other work
outside the run: after the setup of a run (FKSRun of nlo/fks.py or CSRun of
nlo/cs.py) the C++ integrand of a channel at given random numbers, the
integrand of the event generator of a channel, which draws its own points,
and the matrix elements of a subprocess at given momenta, all on the
context of the run, on numpy arrays or torch tensors.

    from madgraph.mg7.nlo.fks import FKSRun
    from madgraph.mg7.nlo.functions import NLOFunctions

    with NLOFunctions(FKSRun("OUTPUT")) as nlo:
        channel = nlo.channels[0]
        weights = nlo.channel_integrand(channel)
        out = weights(random=x, fks_index=k)  # the weights of the points
        sample = nlo.generator_integrand(channel)
        out = sample(10000)  # points of the channel, their weights
        blocks = nlo.matrix_elements(channel.subprocess)
        out = blocks["born"](momenta=p)  # per Born order, amp2, ...

A Function takes its inputs by position or by name (inputs) and gives its
outputs by name (outputs) in the library of its first input, numpy or
torch. The inputs it does not get take its defaults (defaults):

- The channel integrands take the random numbers of the channel (random,
  of dimension channel.dimension) and, for FKS, the FKS configuration of
  each point (fks_index, below channel.choices). By default the weight of
  the choice of the configuration is one (configuration_weight) and every
  point takes the virtual (virtual_mask 1, virtual_ratio 0 and
  virtual_fraction 1, no virtual trick), so that the outputs are the
  weights of the points themselves: cross_section (the NLO weight), its LO
  part (lo_weight), the weights of the event, its counter-events and the
  n-body part (FKS), the real emission and the dipoles (the real stream of
  CS), and the kinematics.
- The generator integrands (FKSIntegrand and CSIntegrand of madspace) take
  the number of points, an integer, and give the random numbers, the
  configurations with the weights of their choice, the weight of each
  point, its parts and kinematics. Their VEGAS grid, discrete choice and
  virtual trick are globals of the context at their initial values (a
  uniform grid and choice, VirtualFraction of FKS_params.dat), which
  NLOIterationState of madspace adapts. With a flow (Flow of madspace, of
  the dimension of the channel) the flow takes the place of the grid, and
  with madnis_training they give the outputs of a MadNIS training
  (full_weight, weight, latent and adaptive_prob, for FKS with the
  configurations), as the runs train them
  (NLORun.train_madnis of nlo/core.py). The madspace object of a Function
  (function) goes into the event generator and the trainings of madspace.
- The matrix elements are the UMAMI libraries of the Fortran provider of
  the subprocess (nlo/libraries.py) through MatrixElement of madspace, as
  the C++ integrands take them: born (squared_orders per Born order,
  diagram_amp2 of the channels, color_correlated of the pairs of coloured
  legs), born_spin (with spin_legs and spin_vectors, and spin_correlated),
  reals (one per real matrix element, squared_orders per NLO order), loop
  (with the renormalization scale, the finite part and the poles per NLO
  order), extras (the Borns of the extra counterterms) and ew_sudakov (the
  blocks of the Sudakov approximation of the EW corrections, with
  --ewsudakov). They take momenta in any frame along the beams and, by
  default, alpha_s = 1 / (4 pi) and the e of the model (g_ew), the blocks as
  the provider gives them, at g_s = 1.
"""

import collections
import math

import numpy as np

from madgraph.mg7.nlo.core import madspace

# the strong coupling of the blocks of the provider, at g_s = 1
ALPHA_S = 1 / (4 * math.pi)

Channel = collections.namedtuple(
    "Channel", ["number", "subprocess", "index", "name", "dimension", "choices"]
)
Channel.__doc__ = """A channel of a run: its number among those of the run,
the name of its subprocess (or real process of CS), its index there, its
name, the dimension of its random numbers and the number of FKS
configurations it chooses from (one for CS)."""


def library_of(value):
    return type(value).__module__.partition(".")[0]


def dtype_of(argument):
    """int32, int64 or float64, the dtype of an argument of madspace (int64
    for a batch size)."""
    dtype = str(argument.dtype).rpartition(".")[2]
    return "int64" if dtype == "batch_sizes" else dtype


def full(like, count, argument, value):
    """The value at count points for an argument of madspace (its Type), in
    the library (and on the device) of like."""
    shape = (count, *argument.shape)
    dtype = dtype_of(argument)
    if library_of(like) == "torch":
        import torch

        return torch.full(shape, value, dtype=getattr(torch, dtype), device=like.device)
    return np.full(shape, value, dtype=dtype)


class Function:
    """A function of madspace (a FunctionGenerator) on the context of a run,
    called on numpy arrays or torch tensors (see the module): inputs and
    outputs are the names of its arguments and of its returns, defaults the
    values of the inputs it fills in at every point. Integer inputs of numpy
    take the integer type of madspace. It builds its FunctionRuntime at its
    first call."""

    def __init__(self, function, context, defaults=None):
        self.function = function
        self.context = context
        self.arguments = function.arg_types()
        self.inputs = list(self.arguments.keys())
        self.outputs = list(function.return_types().keys())
        self.defaults = dict(defaults or {})
        self._runtime = None

    def __call__(self, *args, **kwargs):
        ms = madspace()
        if len(args) > len(self.inputs):
            raise TypeError(
                f"{len(self.inputs)} inputs ({self.inputs}), not {len(args)}"
            )
        values = dict(zip(self.inputs, args))
        for name, value in kwargs.items():
            if name not in self.inputs:
                raise TypeError(f"no input {name}, the inputs are {self.inputs}")
            if name in values:
                raise TypeError(f"{name} given twice")
            values[name] = value
        given = [values[name] for name in self.inputs if name in values]
        like = given[0] if given else np.zeros(0)
        count = len(like) if given else 0
        for name in self.inputs:
            if name in values:
                continue
            if name not in self.defaults:
                raise TypeError(f"input {name} missing, the inputs are {self.inputs}")
            values[name] = full(like, count, self.arguments[name], self.defaults[name])
        if self._runtime is None:
            self._runtime = ms.FunctionRuntime(self.function.function(), self.context)
        arrays = []
        for name in self.inputs:
            value = values[name]
            if library_of(value) != "torch":
                dtype = dtype_of(self.arguments[name])
                value = np.ascontiguousarray(value, dtype=dtype)
            arrays.append(value)
        outputs = self._runtime(*arrays)
        if len(self.outputs) == 1:
            outputs = (outputs,)
        return dict(zip(self.outputs, outputs))


class GeneratorFunction(Function):
    """A generator integrand of a channel (see the module), called with the
    number of points."""

    def __call__(self, count):
        return super().__call__(np.array([int(count)], dtype=np.int64))


class NLOFunctions:
    """The functions of a run (see the module), set up on construction and
    closed with the providers of the run at the end of a with block (or by
    close)."""

    def __init__(self, run):
        self.run = run
        run.setup()
        self.context = run.context
        self.channels = [
            Channel(
                number,
                channel.integrand.name,
                channel.index,
                channel.name,
                channel.integrand.channel_dimension(channel.index),
                getattr(channel.integrand, "configuration_count", 1),
            )
            for number, channel in enumerate(run.channels)
        ]
        self._generators = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self.run.close()

    def _channel(self, channel):
        if isinstance(channel, Channel):
            channel = channel.number
        return self.run.channels[channel]

    def channel_integrand(self, channel):
        """The C++ integrand of a channel (a Channel or its number) at given
        random numbers (FKSChannelIntegrand, CSBornChannelIntegrand or
        CSRealChannelIntegrand of madspace)."""
        entry = self._channel(channel)
        function = entry.integrand.channel_integrand(entry.index, self.context)
        types = function.arg_types()
        defaults = {}
        for name, value in (
            ("configuration_weight", 1.0),
            ("virtual_mask", 1.0),
            ("virtual_ratio", 0.0),
            ("virtual_fraction", 1.0),
        ):
            if name in types.keys():
                defaults[name] = value
        return Function(function, self.context, defaults)

    def generator_integrand(
        self, channel, flow=None, discrete=None, madnis_training=False
    ):
        """The generator integrand of a channel (a Channel or its number),
        with a VEGAS grid, or a flow (and a DiscreteFlow of the FKS
        configurations) for a MadNIS training (see the module), its globals
        of its own prefix in the context of the run. A flow and a
        DiscreteFlow have their globals there already (their
        initialize_globals or initialize_from_vegas)."""
        entry = self._channel(channel)
        run = self.run
        self._generators += 1
        function = entry.integrand.generator_integrand(
            entry.index,
            self.context,
            f"functions{self._generators}",
            run.fks_params["VirtualFraction"],
            run.fks_params["MinVirtualFraction"],
            run.bins,
            None,
            None,
            flow,
            discrete,
            madnis_training=madnis_training,
        )
        if flow is not None:
            # its own discrete choice and virtual trick at their initial
            # values, not those of a run before it
            function.initialize_globals(self.context)
        return GeneratorFunction(function, self.context)

    def _subprocess(self, name):
        subprocesses = getattr(self.run, "subprocesses", None) or {
            integrand.name: integrand for integrand in self.run.integrands
        }
        if name not in subprocesses or not hasattr(subprocesses[name], "provider"):
            raise KeyError(
                f"no subprocess {name}, those of the run are {list(subprocesses)}"
            )
        return subprocesses[name]

    def matrix_elements(self, subprocess):
        """The matrix elements of a subprocess (the name of its P* directory)
        as Functions by name (see the module): born, born_spin, loop, and the
        lists reals and extras, with ew_sudakov with --ewsudakov."""
        ms = madspace()
        element = ms.MatrixElement
        integrand = self._subprocess(subprocess)
        apis = integrand.libraries(self.context)
        libraries = integrand._libraries
        couplings = {"alpha_s": ALPHA_S, "g_ew": libraries.e}
        plain = [element.momenta_in, element.alpha_s_in, element.g_ew_in]
        spin = plain + [element.spin_legs_in, element.spin_vectors_in]
        born_outputs = [element.squared_orders_out, element.diagram_amp2_out]
        if libraries.has_color:
            born_outputs.append(element.color_correlated_out)

        def function(api, inputs, outputs):
            return Function(element(api, inputs, outputs), self.context, couplings)

        blocks = dict(
            born=function(apis["born_plain"], plain, born_outputs),
            born_spin=function(
                apis["born"], spin, born_outputs + [element.spin_correlated_out]
            ),
            reals=[
                function(api, plain, [element.squared_orders_out])
                for api in apis["reals"]
            ],
            loop=function(apis["loop"], plain + [element.scale_in], [element.loop_out]),
            extras=[
                function(
                    api,
                    spin,
                    [element.squared_orders_out, element.spin_correlated_out],
                )
                for api in apis["extras"]
            ],
        )
        if apis["ew_sudakov"] is not None:
            blocks["ew_sudakov"] = function(
                apis["ew_sudakov"],
                plain,
                [
                    element.ew_sudakov_diagonal_out,
                    element.ew_sudakov_lo2_out,
                    element.ew_sudakov_derivatives_out,
                    element.ew_sudakov_interference_out,
                ],
            )
        return blocks
