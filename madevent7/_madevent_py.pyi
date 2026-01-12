from __future__ import annotations

import collections.abc
import typing

__all__ = [
    "AlphaSGrid",
    "BatchSize",
    "ChannelWeightNetwork",
    "Context",
    "CutItem",
    "Cuts",
    "DataType",
    "Decay",
    "Device",
    "Diagram",
    "DifferentialCrossSection",
    "DiscreteFlow",
    "DiscreteHistogram",
    "DiscreteOptimizer",
    "DiscreteSampler",
    "EnergyScale",
    "EventGenerator",
    "EventGeneratorConfig",
    "EventGeneratorStatus",
    "EventGeneratorVerbosity",
    "FastRamboMapping",
    "Flow",
    "Function",
    "FunctionBuilder",
    "FunctionGenerator",
    "FunctionRuntime",
    "Instruction",
    "InstructionCall",
    "Integrand",
    "IntegrandProbability",
    "Invariant",
    "LHECompleter",
    "LHEEvent",
    "LHEFileWriter",
    "LHEHeader",
    "LHEMeta",
    "LHEParticle",
    "LHEProcess",
    "LineRef",
    "Logger",
    "Luminosity",
    "MLP",
    "Mapping",
    "MatrixElement",
    "MatrixElementApi",
    "MomentumPreprocessing",
    "MultiChannelFunction",
    "MultiChannelIntegrand",
    "MultiChannelMapping",
    "PartonDensity",
    "PdfGrid",
    "PhaseSpaceMapping",
    "PrettyBox",
    "Propagator",
    "PropagatorChannelWeights",
    "RunningCoupling",
    "SubchannelWeights",
    "SubprocArgs",
    "TPropagatorMapping",
    "Tensor",
    "ThreeBodyDecay",
    "Topology",
    "TwoBodyDecay",
    "TwoToThreeParticleScattering",
    "TwoToTwoParticleScattering",
    "Type",
    "Unweighter",
    "Value",
    "VegasGridOptimizer",
    "VegasHistogram",
    "VegasMapping",
    "batch_float",
    "batch_float_array",
    "batch_four_vec",
    "batch_four_vec_array",
    "batch_int",
    "batch_size",
    "batch_sizes",
    "cpu_device",
    "cuda_device",
    "default_context",
    "default_cuda_context",
    "default_hip_context",
    "float",
    "format_progress",
    "format_si_prefix",
    "format_with_error",
    "hip_device",
    "initialize_vegas_grid",
    "int",
    "log",
    "multichannel_batch_size",
    "pretty",
    "set_lib_path",
    "set_simd_vector_size",
    "set_thread_count",
    "silent",
    "single_float",
    "single_int",
]

class AlphaSGrid:
    def __init__(self, file: str) -> None: ...
    def coefficients_shape(self, batch_dim: bool = False) -> list[int]: ...
    def initialize_globals(self, context: Context, prefix: str = "") -> None: ...
    def logq2_shape(self, batch_dim: bool = False) -> list[int]: ...
    @property
    def logq2(self) -> list[float]: ...
    @property
    def q(self) -> list[float]: ...
    @property
    def q_count(self) -> int: ...
    @property
    def region_sizes(self) -> list[int]: ...
    @property
    def values(self) -> list[float]: ...

class BatchSize:
    one: typing.ClassVar[BatchSize]  # value = 1
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, name: str) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...

class ChannelWeightNetwork(FunctionGenerator):
    def __init__(
        self,
        channel_count: typing.SupportsInt,
        particle_count: typing.SupportsInt,
        hidden_dim: typing.SupportsInt = 32,
        layers: typing.SupportsInt = 3,
        activation: MLP.Activation = MLP.Activation.Activation.leaky_relu,
        prefix: str = "",
    ) -> None: ...
    def initialize_globals(self, context: Context) -> None: ...
    def mask_name(self) -> str: ...
    def mlp(self) -> MLP: ...
    def preprocessing(self) -> MomentumPreprocessing: ...

class Context:
    @typing.overload
    def __init__(self) -> None: ...
    @typing.overload
    def __init__(self, device: Device) -> None: ...
    def define_global(
        self,
        name: str,
        dtype: DataType,
        shape: collections.abc.Sequence[typing.SupportsInt],
        requires_grad: bool = False,
    ) -> Tensor: ...
    def device(self) -> Device: ...
    def get_global(self, name: str) -> Tensor: ...
    def global_exists(self, name: str) -> bool: ...
    def global_requires_grad(self, name: str) -> bool: ...
    def load(self, file: str) -> None: ...
    def load_matrix_element(self, file: str, param_card: str) -> MatrixElementApi: ...
    def matrix_element(self, index: typing.SupportsInt) -> MatrixElementApi: ...
    def save(self, file: str) -> None: ...

class CutItem:
    def __init__(
        self,
        observable: Cuts.CutObservable,
        limit_type: Cuts.LimitType,
        value: typing.SupportsFloat,
        pids: collections.abc.Sequence[typing.SupportsInt],
        pids2: collections.abc.Sequence[typing.SupportsInt] = [],
    ) -> None: ...
    @property
    def limit_type(self) -> Cuts.LimitType: ...
    @property
    def observable(self) -> Cuts.CutObservable: ...
    @property
    def pids(self) -> list[int]: ...
    @property
    def pids2(self) -> list[int]: ...
    @property
    def value(self) -> float: ...

class Cuts(FunctionGenerator):
    class CutObservable:
        """
        Members:

          obs_pt

          obs_eta

          obs_dr

          obs_mass

          obs_sqrt_s
        """

        __members__: typing.ClassVar[
            dict[str, Cuts.CutObservable]
        ]  # value = {'obs_pt': <CutObservable.obs_pt: 0>, 'obs_eta': <CutObservable.obs_eta: 1>, 'obs_dr': <CutObservable.obs_dr: 2>, 'obs_mass': <CutObservable.obs_mass: 3>, 'obs_sqrt_s': <CutObservable.obs_sqrt_s: 4>}
        obs_dr: typing.ClassVar[Cuts.CutObservable]  # value = <CutObservable.obs_dr: 2>
        obs_eta: typing.ClassVar[
            Cuts.CutObservable
        ]  # value = <CutObservable.obs_eta: 1>
        obs_mass: typing.ClassVar[
            Cuts.CutObservable
        ]  # value = <CutObservable.obs_mass: 3>
        obs_pt: typing.ClassVar[Cuts.CutObservable]  # value = <CutObservable.obs_pt: 0>
        obs_sqrt_s: typing.ClassVar[
            Cuts.CutObservable
        ]  # value = <CutObservable.obs_sqrt_s: 4>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        @typing.overload
        def __init__(self, value: typing.SupportsInt) -> None: ...
        @typing.overload
        def __init__(self, name: str) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    class LimitType:
        """
        Members:

          min

          max
        """

        __members__: typing.ClassVar[
            dict[str, Cuts.LimitType]
        ]  # value = {'min': <LimitType.min: 0>, 'max': <LimitType.max: 1>}
        max: typing.ClassVar[Cuts.LimitType]  # value = <LimitType.max: 1>
        min: typing.ClassVar[Cuts.LimitType]  # value = <LimitType.min: 0>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        @typing.overload
        def __init__(self, value: typing.SupportsInt) -> None: ...
        @typing.overload
        def __init__(self, name: str) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    bottom_pids: typing.ClassVar[list] = [-5, 5]
    jet_pids: typing.ClassVar[list] = [1, 2, 3, 4, -1, -2, -3, -4, 21]
    lepton_pids: typing.ClassVar[list] = [11, 13, 15, -11, -13, -15]
    max: typing.ClassVar[Cuts.LimitType]  # value = <LimitType.max: 1>
    min: typing.ClassVar[Cuts.LimitType]  # value = <LimitType.min: 0>
    missing_pids: typing.ClassVar[list] = [12, 14, 16, -12, -14, -16]
    obs_dr: typing.ClassVar[Cuts.CutObservable]  # value = <CutObservable.obs_dr: 2>
    obs_eta: typing.ClassVar[Cuts.CutObservable]  # value = <CutObservable.obs_eta: 1>
    obs_mass: typing.ClassVar[Cuts.CutObservable]  # value = <CutObservable.obs_mass: 3>
    obs_pt: typing.ClassVar[Cuts.CutObservable]  # value = <CutObservable.obs_pt: 0>
    obs_sqrt_s: typing.ClassVar[
        Cuts.CutObservable
    ]  # value = <CutObservable.obs_sqrt_s: 4>
    photon_pids: typing.ClassVar[list] = [22]
    def __init__(
        self,
        pids: collections.abc.Sequence[typing.SupportsInt],
        cut_data: collections.abc.Sequence[CutItem],
    ) -> None: ...
    def eta_max(self) -> list[float]: ...
    def pt_min(self) -> list[float]: ...
    def sqrt_s_min(self) -> float: ...

class DataType:
    """
    Members:

      int

      float

      batch_sizes
    """

    __members__: typing.ClassVar[
        dict[str, DataType]
    ]  # value = {'int': <DataType.int: 0>, 'float': <DataType.float: 1>, 'batch_sizes': <DataType.batch_sizes: 2>}
    batch_sizes: typing.ClassVar[DataType]  # value = <DataType.batch_sizes: 2>
    float: typing.ClassVar[DataType]  # value = <DataType.float: 1>
    int: typing.ClassVar[DataType]  # value = <DataType.int: 0>
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    @typing.overload
    def __init__(self, value: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, name: str) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Decay:
    @property
    def child_indices(self) -> list[int]: ...
    @property
    def e_max(self) -> float: ...
    @property
    def e_min(self) -> float: ...
    @property
    def index(self) -> int: ...
    @property
    def mass(self) -> float: ...
    @property
    def on_shell(self) -> bool: ...
    @property
    def parent_index(self) -> int: ...
    @property
    def pdg_id(self) -> int: ...
    @property
    def width(self) -> float: ...

class Device:
    pass

class Diagram:
    def __init__(
        self,
        incoming_masses: collections.abc.Sequence[typing.SupportsFloat],
        outgoing_masses: collections.abc.Sequence[typing.SupportsFloat],
        propagators: collections.abc.Sequence[Propagator],
        vertices: collections.abc.Sequence[collections.abc.Sequence[LineRef]],
    ) -> None: ...
    @property
    def incoming_masses(self) -> list[float]: ...
    @property
    def incoming_vertices(self) -> typing.Annotated[list[int], "FixedSize(2)"]: ...
    @property
    def outgoing_masses(self) -> list[float]: ...
    @property
    def outgoing_vertices(self) -> list[int]: ...
    @property
    def propagator_vertices(self) -> list[list[int]]: ...
    @property
    def propagators(self) -> list[Propagator]: ...
    @property
    def vertices(self) -> list[list[LineRef]]: ...

class DifferentialCrossSection(FunctionGenerator):
    def __init__(
        self,
        matrix_element: MatrixElement,
        cm_energy: typing.SupportsFloat,
        running_coupling: RunningCoupling,
        energy_scale: EnergyScale,
        pid_options: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ] = [],
        has_pdf1: bool = False,
        has_pdf2: bool = False,
        pdf_grid1: madevent7._madevent_py.PdfGrid | None = None,
        pdf_grid2: madevent7._madevent_py.PdfGrid | None = None,
        has_mirror: bool = False,
        input_momentum_fraction: bool = True,
    ) -> None: ...
    def has_mirror(self) -> bool: ...
    def matrix_element(self) -> MatrixElement: ...
    def pid_options(self) -> list[list[int]]: ...

class DiscreteFlow(Mapping):
    def __init__(
        self,
        option_counts: collections.abc.Sequence[typing.SupportsInt],
        prefix: str = "",
        dims_with_prior: collections.abc.Sequence[typing.SupportsInt] = [],
        condition_dim: typing.SupportsInt = 0,
        subnet_hidden_dim: typing.SupportsInt = 32,
        subnet_layers: typing.SupportsInt = 3,
        subnet_activation: MLP.Activation = MLP.Activation.Activation.leaky_relu,
    ) -> None: ...
    def condition_dim(self) -> int: ...
    def initialize_globals(self, context: Context) -> None: ...
    def option_counts(self) -> list[int]: ...

class DiscreteHistogram(FunctionGenerator):
    def __init__(
        self, option_counts: collections.abc.Sequence[typing.SupportsInt]
    ) -> None: ...

class DiscreteOptimizer:
    def __init__(
        self, context: Context, prob_names: collections.abc.Sequence[str]
    ) -> None: ...
    def add_data(
        self, values_and_counts: collections.abc.Sequence[typing.Any]
    ) -> None: ...
    def optimize(self) -> None: ...

class DiscreteSampler(Mapping):
    def __init__(
        self,
        option_counts: collections.abc.Sequence[typing.SupportsInt],
        prefix: str = "",
        dims_with_prior: collections.abc.Sequence[typing.SupportsInt] = [],
    ) -> None: ...
    def initialize_globals(self, context: Context) -> None: ...
    def option_counts(self) -> list[int]: ...
    def prob_names(self) -> list[str]: ...

class EnergyScale(FunctionGenerator):
    class DynamicalScaleType:
        """
        Members:

          transverse_energy

          transverse_mass

          half_transverse_mass

          partonic_energy
        """

        __members__: typing.ClassVar[
            dict[str, EnergyScale.DynamicalScaleType]
        ]  # value = {'transverse_energy': <DynamicalScaleType.transverse_energy: 0>, 'transverse_mass': <DynamicalScaleType.transverse_mass: 1>, 'half_transverse_mass': <DynamicalScaleType.half_transverse_mass: 2>, 'partonic_energy': <DynamicalScaleType.partonic_energy: 3>}
        half_transverse_mass: typing.ClassVar[
            EnergyScale.DynamicalScaleType
        ]  # value = <DynamicalScaleType.half_transverse_mass: 2>
        partonic_energy: typing.ClassVar[
            EnergyScale.DynamicalScaleType
        ]  # value = <DynamicalScaleType.partonic_energy: 3>
        transverse_energy: typing.ClassVar[
            EnergyScale.DynamicalScaleType
        ]  # value = <DynamicalScaleType.transverse_energy: 0>
        transverse_mass: typing.ClassVar[
            EnergyScale.DynamicalScaleType
        ]  # value = <DynamicalScaleType.transverse_mass: 1>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        @typing.overload
        def __init__(self, value: typing.SupportsInt) -> None: ...
        @typing.overload
        def __init__(self, name: str) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    half_transverse_mass: typing.ClassVar[
        EnergyScale.DynamicalScaleType
    ]  # value = <DynamicalScaleType.half_transverse_mass: 2>
    partonic_energy: typing.ClassVar[
        EnergyScale.DynamicalScaleType
    ]  # value = <DynamicalScaleType.partonic_energy: 3>
    transverse_energy: typing.ClassVar[
        EnergyScale.DynamicalScaleType
    ]  # value = <DynamicalScaleType.transverse_energy: 0>
    transverse_mass: typing.ClassVar[
        EnergyScale.DynamicalScaleType
    ]  # value = <DynamicalScaleType.transverse_mass: 1>
    @typing.overload
    def __init__(self, particle_count: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(
        self, particle_count: typing.SupportsInt, type: EnergyScale.DynamicalScaleType
    ) -> None: ...
    @typing.overload
    def __init__(
        self, particle_count: typing.SupportsInt, fixed_scale: typing.SupportsFloat
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        particle_count: typing.SupportsInt,
        dynamical_scale_type: EnergyScale.DynamicalScaleType,
        ren_scale_fixed: bool,
        fact_scale_fixed: bool,
        ren_scale: typing.SupportsFloat,
        fact_scale1: typing.SupportsFloat,
        fact_scale2: typing.SupportsFloat,
    ) -> None: ...

class EventGenerator:
    default_config: typing.ClassVar[
        EventGeneratorConfig
    ]  # value = <madevent7._madevent_py.EventGeneratorConfig object>
    integrand_flags: typing.ClassVar[int] = 1077
    def __init__(
        self,
        context: Context,
        channels: collections.abc.Sequence[Integrand],
        temp_file_prefix: str,
        status_file: str = "",
        default_config: EventGeneratorConfig = ...,
        channel_subprocesses: collections.abc.Sequence[typing.SupportsInt] = [],
        channel_names: collections.abc.Sequence[str] = [],
    ) -> None: ...
    def channel_status(self) -> list[EventGeneratorStatus]: ...
    def combine_to_compact_npy(self, file_name: str) -> None: ...
    def combine_to_lhe(self, file_name: str, lhe_completer: ...) -> None: ...
    def combine_to_lhe_npy(self, file_name: str, lhe_completer: ...) -> None: ...
    def generate(self) -> None: ...
    def status(self) -> EventGeneratorStatus: ...
    def survey(self) -> None: ...

class EventGeneratorConfig:
    verbosity: EventGeneratorVerbosity
    def __init__(self) -> None: ...
    @property
    def batch_size(self) -> int: ...
    @batch_size.setter
    def batch_size(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def freeze_max_weight_after(self) -> int: ...
    @freeze_max_weight_after.setter
    def freeze_max_weight_after(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def max_batch_size(self) -> int: ...
    @max_batch_size.setter
    def max_batch_size(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def max_overweight_truncation(self) -> float: ...
    @max_overweight_truncation.setter
    def max_overweight_truncation(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def optimization_patience(self) -> int: ...
    @optimization_patience.setter
    def optimization_patience(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def optimization_threshold(self) -> float: ...
    @optimization_threshold.setter
    def optimization_threshold(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def start_batch_size(self) -> int: ...
    @start_batch_size.setter
    def start_batch_size(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def survey_max_iters(self) -> int: ...
    @survey_max_iters.setter
    def survey_max_iters(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def survey_min_iters(self) -> int: ...
    @survey_min_iters.setter
    def survey_min_iters(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def survey_target_precision(self) -> float: ...
    @survey_target_precision.setter
    def survey_target_precision(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def target_count(self) -> int: ...
    @target_count.setter
    def target_count(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def vegas_damping(self) -> float: ...
    @vegas_damping.setter
    def vegas_damping(self, arg0: typing.SupportsFloat) -> None: ...

class EventGeneratorStatus:
    done: bool
    def __init__(self) -> None: ...
    @property
    def count(self) -> int: ...
    @count.setter
    def count(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def count_after_cuts(self) -> int: ...
    @count_after_cuts.setter
    def count_after_cuts(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def count_after_cuts_opt(self) -> int: ...
    @count_after_cuts_opt.setter
    def count_after_cuts_opt(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def count_opt(self) -> int: ...
    @count_opt.setter
    def count_opt(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def count_target(self) -> float: ...
    @count_target.setter
    def count_target(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def count_unweighted(self) -> float: ...
    @count_unweighted.setter
    def count_unweighted(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def error(self) -> float: ...
    @error.setter
    def error(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def index(self) -> int: ...
    @index.setter
    def index(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def iterations(self) -> int: ...
    @iterations.setter
    def iterations(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def mean(self) -> float: ...
    @mean.setter
    def mean(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def rel_std_dev(self) -> float: ...
    @rel_std_dev.setter
    def rel_std_dev(self, arg0: typing.SupportsFloat) -> None: ...

class EventGeneratorVerbosity:
    """
    Members:

      silent

      log

      pretty
    """

    __members__: typing.ClassVar[
        dict[str, EventGeneratorVerbosity]
    ]  # value = {'silent': <EventGeneratorVerbosity.silent: 0>, 'log': <EventGeneratorVerbosity.log: 1>, 'pretty': <EventGeneratorVerbosity.pretty: 2>}
    log: typing.ClassVar[
        EventGeneratorVerbosity
    ]  # value = <EventGeneratorVerbosity.log: 1>
    pretty: typing.ClassVar[
        EventGeneratorVerbosity
    ]  # value = <EventGeneratorVerbosity.pretty: 2>
    silent: typing.ClassVar[
        EventGeneratorVerbosity
    ]  # value = <EventGeneratorVerbosity.silent: 0>
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    @typing.overload
    def __init__(self, value: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, name: str) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class FastRamboMapping(Mapping):
    def __init__(self, n_particles: typing.SupportsInt, massless: bool) -> None: ...

class Flow(Mapping):
    def __init__(
        self,
        input_dim: typing.SupportsInt,
        condition_dim: typing.SupportsInt = 0,
        prefix: str = "",
        bin_count: typing.SupportsInt = 10,
        subnet_hidden_dim: typing.SupportsInt = 32,
        subnet_layers: typing.SupportsInt = 3,
        subnet_activation: MLP.Activation = MLP.Activation.Activation.leaky_relu,
        invert_spline: bool = True,
    ) -> None: ...
    def condition_dim(self) -> int: ...
    def initialize_from_vegas(self, context: Context, grid_name: str) -> None: ...
    def initialize_globals(self, context: Context) -> None: ...
    def input_dim(self) -> int: ...

class Function:
    @staticmethod
    def load(file: str) -> Function: ...
    def __call__(self, *args): ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def save(self, file: str) -> None: ...
    @property
    def globals(self) -> dict[str, Value]: ...
    @property
    def inputs(self) -> list[Value]: ...
    @property
    def instructions(self) -> list[InstructionCall]: ...
    @property
    def locals(self) -> list[Value]: ...
    @property
    def outputs(self) -> list[Value]: ...

class FunctionBuilder:
    def __init__(
        self,
        input_types: collections.abc.Sequence[Type],
        output_types: collections.abc.Sequence[Type],
    ) -> None: ...
    def add(self, in1: Value, in2: Value) -> Value: ...
    def add_int(self, in1: Value, in2: Value) -> Value: ...
    def apply_subchannel_weights(
        self,
        channel_weights_in: Value,
        subchannel_weights: Value,
        channel_indices: Value,
        subchannel_indices: Value,
    ) -> Value: ...
    def batch_cat(
        self, args: collections.abc.Sequence[Value]
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def batch_gather(self, indices: Value, values: Value) -> Value: ...
    def batch_scatter(self, indices: Value, target: Value, source: Value) -> Value: ...
    def batch_size(self, args: collections.abc.Sequence[Value]) -> Value: ...
    def batch_split(self, in1: Value, counts: Value) -> list[Value]: ...
    def boost_beam(self, p1: Value, x1: Value, x2: Value) -> Value: ...
    def boost_beam_inverse(self, p1: Value, x1: Value, x2: Value) -> Value: ...
    def breit_wigner_invariant(
        self, r: Value, mass: Value, width: Value, s_min: Value, s_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def breit_wigner_invariant_inverse(
        self, s: Value, mass: Value, width: Value, s_min: Value, s_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def cat(self, args: collections.abc.Sequence[Value]) -> Value: ...
    def chili_forward(
        self, r: Value, e_cm: Value, m_out: Value, pt_min: Value, y_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(4)"]: ...
    def collect_channel_weights(
        self, amp2: Value, channel_indices: Value, channel_count: Value
    ) -> Value: ...
    def com_p_in(
        self, e_cm: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def cut_dr(self, p: Value, indices: Value, min_max: Value) -> Value: ...
    def cut_eta(self, p: Value, min_max: Value) -> Value: ...
    def cut_m_inv(self, p: Value, indices: Value, min_max: Value) -> Value: ...
    def cut_pt(self, p: Value, min_max: Value) -> Value: ...
    def cut_sqrt_s(self, p: Value, min_max: Value) -> Value: ...
    def cut_unphysical(self, w_in: Value, p: Value, x1: Value, x2: Value) -> Value: ...
    def diff_cross_section(
        self,
        x1: Value,
        x2: Value,
        pdf1: Value,
        pdf2: Value,
        matrix_element: Value,
        e_cm2: Value,
    ) -> Value: ...
    def discrete_histogram(
        self, input: Value, weights: Value, option_count: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def elu(self, in1: Value) -> Value: ...
    def fast_rambo_massive(
        self, r: Value, e_cm: Value, masses: Value, p0: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def fast_rambo_massive_com(
        self, r: Value, e_cm: Value, masses: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def fast_rambo_massless(
        self, r: Value, e_cm: Value, p0: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def fast_rambo_massless_com(
        self, r: Value, e_cm: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def full(self, args: collections.abc.Sequence[Value]) -> Value: ...
    def function(self) -> Function: ...
    def gather(self, index: Value, choices: Value) -> Value: ...
    def gather_int(self, index: Value, choices: Value) -> Value: ...
    def gelu(self, in1: Value) -> Value: ...
    def get_global(
        self,
        name: str,
        dtype: DataType,
        shape: collections.abc.Sequence[typing.SupportsInt],
    ) -> Value: ...
    def input(self, index: typing.SupportsInt) -> Value: ...
    def input_range(
        self, start_index: typing.SupportsInt, end_index: typing.SupportsInt
    ) -> list[Value]: ...
    def interpolate_alpha_s(
        self, q2: Value, grid_logq2: Value, grid_coeffs: Value
    ) -> Value: ...
    def interpolate_pdf(
        self,
        x: Value,
        q2: Value,
        pid_indices: Value,
        grid_logx: Value,
        grid_logq2: Value,
        grid_coeffs: Value,
    ) -> Value: ...
    def invariants_from_momenta(self, p_ext: Value, factors: Value) -> Value: ...
    def leaky_relu(self, in1: Value) -> Value: ...
    def matmul(self, x: Value, weight: Value, bias: Value) -> Value: ...
    def matrix_element(self, args: collections.abc.Sequence[Value]) -> list[Value]: ...
    def max(self, in1: Value, in2: Value) -> Value: ...
    def min(self, in1: Value, in2: Value) -> Value: ...
    def mirror_momenta(self, p_ext: Value, mirror: Value) -> Value: ...
    def momenta_to_x1x2(
        self, p_ext: Value, e_cm: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def mul(self, in1: Value, in2: Value) -> Value: ...
    def nonzero(self, input: Value) -> Value: ...
    def offset_indices(
        self, batch_sizes_offset: Value, batch_sizes_out: Value
    ) -> Value: ...
    def one_hot(self, index: Value, option_count: Value) -> Value: ...
    def output(self, index: typing.SupportsInt, value: Value) -> None: ...
    def output_range(
        self, start_index: typing.SupportsInt, values: collections.abc.Sequence[Value]
    ) -> None: ...
    def permute_momenta(
        self, momenta: Value, permutations: Value, index: Value
    ) -> Value: ...
    def pop(self, in1: Value) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def product(self, values: collections.abc.Sequence[Value]) -> Value: ...
    def pt_eta_phi_x(self, p_ext: Value, x1: Value, x2: Value) -> Value: ...
    def r_to_x1x2(
        self, r: Value, s_hat: Value, s_lab: Value
    ) -> typing.Annotated[list[Value], "FixedSize(3)"]: ...
    def random(self, batch_size: Value, count: Value) -> Value: ...
    def reduce_product(self, in1: Value) -> Value: ...
    def relu(self, in1: Value) -> Value: ...
    def rqs_find_bin(
        self, input: Value, in_sizes: Value, out_sizes: Value, derivatives: Value
    ) -> Value: ...
    def rqs_forward(
        self, input: Value, condition: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def rqs_inverse(
        self, input: Value, condition: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def rqs_reshape(
        self, input: Value, bin_count: Value
    ) -> typing.Annotated[list[Value], "FixedSize(3)"]: ...
    def s_inv_min_max(
        self, pa: Value, pb: Value, p3: Value, t1_abs: Value, m1: Value, m2: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def sample_discrete(
        self, r: Value, option_count: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def sample_discrete_inverse(
        self, index: Value, option_count: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def sample_discrete_probs(
        self, r: Value, probs: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def sample_discrete_probs_inverse(
        self, index: Value, probs: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def scale_half_transverse_mass(self, momenta: Value) -> Value: ...
    def scale_partonic_energy(self, momenta: Value) -> Value: ...
    def scale_transverse_energy(self, momenta: Value) -> Value: ...
    def scale_transverse_mass(self, momenta: Value) -> Value: ...
    def sde2_channel_weights(
        self, invariants: Value, masses: Value, widths: Value, indices: Value
    ) -> Value: ...
    def select(self, input: Value, indices: Value) -> Value: ...
    def sigmoid(self, in1: Value) -> Value: ...
    def softmax(self, input: Value) -> Value: ...
    def softmax_prior(self, input: Value, prior: Value) -> Value: ...
    def softplus(self, in1: Value) -> Value: ...
    def sqrt(self, in1: Value) -> Value: ...
    def square(self, in1: Value) -> Value: ...
    def squeeze(self, input: Value) -> Value: ...
    def stable_invariant(
        self, r: Value, mass: Value, s_min: Value, s_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def stable_invariant_inverse(
        self, s: Value, mass: Value, s_min: Value, s_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def stable_invariant_nu(
        self, r: Value, mass: Value, nu: Value, s_min: Value, s_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def stable_invariant_nu_inverse(
        self, s: Value, mass: Value, nu: Value, s_min: Value, s_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def stack(self, args: collections.abc.Sequence[Value]) -> Value: ...
    def sub(self, in1: Value, in2: Value) -> Value: ...
    def subchannel_weights(
        self,
        invariants: Value,
        masses: Value,
        widths: Value,
        indices: Value,
        on_shell: Value,
        group_sizes: Value,
    ) -> Value: ...
    def t_inv_min_max(
        self, pa: Value, pb: Value, m1: Value, m2: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def three_body_decay(
        self,
        r_e1: Value,
        r_e2: Value,
        r_phi: Value,
        r_cos_theta: Value,
        r_beta: Value,
        m0: Value,
        m1: Value,
        m2: Value,
        m3: Value,
        p0: Value,
    ) -> typing.Annotated[list[Value], "FixedSize(4)"]: ...
    def three_body_decay_com(
        self,
        r_e1: Value,
        r_e2: Value,
        r_phi: Value,
        r_cos_theta: Value,
        r_beta: Value,
        m0: Value,
        m1: Value,
        m2: Value,
        m3: Value,
    ) -> typing.Annotated[list[Value], "FixedSize(4)"]: ...
    def two_body_decay(
        self,
        r_phi: Value,
        r_cos_theta: Value,
        m0: Value,
        m1: Value,
        m2: Value,
        p0: Value,
    ) -> typing.Annotated[list[Value], "FixedSize(3)"]: ...
    def two_body_decay_com(
        self, r_phi: Value, r_cos_theta: Value, m0: Value, m1: Value, m2: Value
    ) -> typing.Annotated[list[Value], "FixedSize(3)"]: ...
    def two_to_three_particle_scattering(
        self,
        phi_choice: Value,
        pa: Value,
        pb: Value,
        p3: Value,
        s23: Value,
        t1_abs: Value,
        m1: Value,
        m2: Value,
    ) -> typing.Annotated[list[Value], "FixedSize(3)"]: ...
    def two_to_two_particle_scattering(
        self, r_phi: Value, pa: Value, pb: Value, t_abs: Value, m1: Value, m2: Value
    ) -> typing.Annotated[list[Value], "FixedSize(3)"]: ...
    def two_to_two_particle_scattering_com(
        self, r_phi: Value, pa: Value, pb: Value, t_abs: Value, m1: Value, m2: Value
    ) -> typing.Annotated[list[Value], "FixedSize(3)"]: ...
    def uniform_invariant(
        self, r: Value, s_min: Value, s_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def uniform_invariant_inverse(
        self, s: Value, s_min: Value, s_max: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def unsqueeze(self, input: Value) -> Value: ...
    def unstack(self, in1: Value) -> list[Value]: ...
    def unstack_sizes(self, in1: Value) -> list[Value]: ...
    def unweight(
        self, weights: Value, max_weight: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def vegas_forward(
        self, input: Value, grid: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def vegas_histogram(
        self, input: Value, weights: Value, bin_count: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def vegas_inverse(
        self, input: Value, grid: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...
    def x1x2_to_r(
        self, x1: Value, x2: Value, s_lab: Value
    ) -> typing.Annotated[list[Value], "FixedSize(2)"]: ...

class FunctionGenerator:
    def __call__(self, *args): ...
    def __init__(
        self,
        name: str,
        arg_types: collections.abc.Sequence[Type],
        return_types: collections.abc.Sequence[Type],
    ) -> None: ...
    def build_function(
        self, builder: FunctionBuilder, args: collections.abc.Sequence[Value]
    ) -> list[Value]: ...
    def function(self) -> Function: ...

class FunctionRuntime:
    def __call__(self, *args): ...
    @typing.overload
    def __init__(self, function: Function) -> None: ...
    @typing.overload
    def __init__(self, function: Function, context: Context) -> None: ...
    def call(self, arg0: collections.abc.Sequence[typing.Any]) -> list[Tensor]: ...
    def call_backward(
        self,
        arg0: collections.abc.Sequence[typing.Any],
        arg1: collections.abc.Sequence[typing.Any],
        arg2: collections.abc.Sequence[bool],
    ) -> tuple[
        list[madevent7._madevent_py.Tensor | None],
        list[tuple[str, madevent7._madevent_py.Tensor | None]],
    ]: ...
    def call_with_grad(
        self,
        arg0: collections.abc.Sequence[typing.Any],
        arg1: collections.abc.Sequence[bool],
    ) -> tuple[
        list[Tensor], list[madevent7._madevent_py.Tensor | None], list[bool]
    ]: ...

class Instruction:
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def opcode(self) -> int: ...

class InstructionCall:
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def inputs(self) -> list[Value]: ...
    @property
    def instruction(self) -> Instruction: ...
    @property
    def outputs(self) -> list[Value]: ...

class Integrand(FunctionGenerator):
    matrix_element_inputs: typing.ClassVar[
        list
    ]  # value = [<MatrixElementInput.momenta_in: 0>, <MatrixElementInput.alpha_s_in: 1>, <MatrixElementInput.flavor_in: 2>, <MatrixElementInput.random_color_in: 3>, <MatrixElementInput.random_helicity_in: 4>, <MatrixElementInput.random_diagram_in: 5>]
    matrix_element_outputs: typing.ClassVar[
        list
    ]  # value = [<MatrixElementOutput.matrix_element_out: 0>, <MatrixElementOutput.diagram_amp2_out: 1>, <MatrixElementOutput.color_index_out: 2>, <MatrixElementOutput.helicity_index_out: 3>, <MatrixElementOutput.diagram_index_out: 4>]
    return_chan_weights: typing.ClassVar[int] = 256
    return_channel: typing.ClassVar[int] = 128
    return_cwnet_input: typing.ClassVar[int] = 512
    return_discrete: typing.ClassVar[int] = 1024
    return_discrete_latent: typing.ClassVar[int] = 2048
    return_indices: typing.ClassVar[int] = 16
    return_latent: typing.ClassVar[int] = 64
    return_momenta: typing.ClassVar[int] = 4
    return_random: typing.ClassVar[int] = 32
    return_x1_x2: typing.ClassVar[int] = 8
    sample: typing.ClassVar[int] = 1
    unweight: typing.ClassVar[int] = 2
    def __init__(
        self,
        mapping: PhaseSpaceMapping,
        diff_xs: DifferentialCrossSection,
        adaptive_map: (
            None | madevent7._madevent_py.VegasMapping | madevent7._madevent_py.Flow
        ) = None,
        discrete_before: (
            None
            | madevent7._madevent_py.DiscreteSampler
            | madevent7._madevent_py.DiscreteFlow
        ) = None,
        discrete_after: (
            None
            | madevent7._madevent_py.DiscreteSampler
            | madevent7._madevent_py.DiscreteFlow
        ) = None,
        pdf_grid: madevent7._madevent_py.PdfGrid | None = None,
        energy_scale: madevent7._madevent_py.EnergyScale | None = None,
        prop_chan_weights: (
            madevent7._madevent_py.PropagatorChannelWeights | None
        ) = None,
        subchan_weights: madevent7._madevent_py.SubchannelWeights | None = None,
        chan_weight_net: madevent7._madevent_py.ChannelWeightNetwork | None = None,
        chan_weight_remap: collections.abc.Sequence[typing.SupportsInt] = [],
        remapped_chan_count: typing.SupportsInt = 0,
        flags: typing.SupportsInt = 0,
        channel_indices: collections.abc.Sequence[typing.SupportsInt] = [],
        active_flavors: collections.abc.Sequence[typing.SupportsInt] = [],
    ) -> None: ...
    def adaptive_map(
        self,
    ) -> None | madevent7._madevent_py.VegasMapping | madevent7._madevent_py.Flow: ...
    def chan_weight_net(self) -> madevent7._madevent_py.ChannelWeightNetwork | None: ...
    def diff_xs(self) -> DifferentialCrossSection: ...
    def discrete_after(
        self,
    ) -> (
        None
        | madevent7._madevent_py.DiscreteSampler
        | madevent7._madevent_py.DiscreteFlow
    ): ...
    def discrete_before(
        self,
    ) -> (
        None
        | madevent7._madevent_py.DiscreteSampler
        | madevent7._madevent_py.DiscreteFlow
    ): ...
    def energy_scale(self) -> madevent7._madevent_py.EnergyScale | None: ...
    def flags(self) -> int: ...
    def latent_dims(self) -> tuple[list[int], list[bool]]: ...
    def mapping(self) -> PhaseSpaceMapping: ...
    def particle_count(self) -> int: ...
    def prop_chan_weights(
        self,
    ) -> madevent7._madevent_py.PropagatorChannelWeights | None: ...
    def random_dim(self) -> int: ...
    def vegas_grid_name(self) -> str | None: ...

class IntegrandProbability(FunctionGenerator):
    def __init__(self, integrand: Integrand) -> None: ...

class Invariant(Mapping):
    def __init__(
        self,
        power: typing.SupportsFloat = 0.0,
        mass: typing.SupportsFloat = 0.0,
        width: typing.SupportsFloat = 0.0,
    ) -> None: ...

class LHECompleter:
    def __init__(
        self,
        subproc_args: collections.abc.Sequence[SubprocArgs],
        bw_cutoff: typing.SupportsFloat,
    ) -> None: ...
    def complete_event_data(
        self,
        event: LHEEvent,
        subprocess_index: typing.SupportsInt,
        diagram_index: typing.SupportsInt,
        color_index: typing.SupportsInt,
        flavor_index: typing.SupportsInt,
        helicity_index: typing.SupportsInt,
    ) -> None: ...
    @property
    def max_particle_count(self) -> int: ...

class LHEEvent:
    def __init__(
        self,
        process_id: typing.SupportsInt = 0,
        weight: typing.SupportsFloat = 0.0,
        scale: typing.SupportsFloat = 0.0,
        alpha_qed: typing.SupportsFloat = 0.0,
        alpha_qcd: typing.SupportsFloat = 0.0,
        particles: collections.abc.Sequence[LHEParticle] = [],
    ) -> None: ...
    @property
    def alpha_qcd(self) -> int: ...
    @alpha_qcd.setter
    def alpha_qcd(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def alpha_qed(self) -> float: ...
    @alpha_qed.setter
    def alpha_qed(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def particles(self) -> list[LHEParticle]: ...
    @particles.setter
    def particles(self, arg0: collections.abc.Sequence[LHEParticle]) -> None: ...
    @property
    def process_id(self) -> int: ...
    @process_id.setter
    def process_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def scale(self) -> float: ...
    @scale.setter
    def scale(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def weight(self) -> float: ...
    @weight.setter
    def weight(self, arg0: typing.SupportsFloat) -> None: ...

class LHEFileWriter:
    def __init__(self, file_name: str, meta: LHEMeta) -> None: ...
    def write(self, event: LHEEvent) -> None: ...
    def write_string(self, str: str) -> None: ...

class LHEHeader:
    content: str
    escape_content: bool
    name: str
    def __init__(
        self, name: str = "", content: str = "", escape_content: bool = False
    ) -> None: ...

class LHEMeta:
    def __init__(
        self,
        beam1_pdg_id: typing.SupportsInt = 0,
        beam2_pdg_id: typing.SupportsInt = 0,
        beam1_energy: typing.SupportsFloat = 0.0,
        beam2_energy: typing.SupportsFloat = 0.0,
        beam1_pdf_authors: typing.SupportsInt = 0,
        beam2_pdf_authors: typing.SupportsInt = 0,
        beam1_pdf_id: typing.SupportsInt = 0,
        beam2_pdf_id: typing.SupportsInt = 0,
        weight_mode: typing.SupportsInt = 0,
        processes: collections.abc.Sequence[LHEProcess] = [],
        headers: collections.abc.Sequence[LHEHeader] = [],
    ) -> None: ...
    @property
    def beam1_energy(self) -> float: ...
    @beam1_energy.setter
    def beam1_energy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def beam1_pdf_authors(self) -> int: ...
    @beam1_pdf_authors.setter
    def beam1_pdf_authors(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def beam1_pdf_id(self) -> int: ...
    @beam1_pdf_id.setter
    def beam1_pdf_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def beam1_pdg_id(self) -> int: ...
    @beam1_pdg_id.setter
    def beam1_pdg_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def beam2_energy(self) -> float: ...
    @beam2_energy.setter
    def beam2_energy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def beam2_pdf_authors(self) -> int: ...
    @beam2_pdf_authors.setter
    def beam2_pdf_authors(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def beam2_pdf_id(self) -> int: ...
    @beam2_pdf_id.setter
    def beam2_pdf_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def beam2_pdg_id(self) -> int: ...
    @beam2_pdg_id.setter
    def beam2_pdg_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def headers(self) -> list[LHEHeader]: ...
    @headers.setter
    def headers(self, arg0: collections.abc.Sequence[LHEHeader]) -> None: ...
    @property
    def processes(self) -> list[LHEProcess]: ...
    @processes.setter
    def processes(self, arg0: collections.abc.Sequence[LHEProcess]) -> None: ...
    @property
    def weight_mode(self) -> int: ...
    @weight_mode.setter
    def weight_mode(self, arg0: typing.SupportsInt) -> None: ...

class LHEParticle:
    status_incoming: typing.ClassVar[int] = -1
    status_intermediate_resonance: typing.ClassVar[int] = 2
    status_outgoing: typing.ClassVar[int] = 1
    def __init__(
        self,
        pdg_id: typing.SupportsInt = 0,
        status_code: typing.SupportsInt = 0,
        mother1: typing.SupportsInt = 0,
        mother2: typing.SupportsInt = 0,
        color: typing.SupportsInt = 0,
        anti_color: typing.SupportsInt = 0,
        p_x: typing.SupportsFloat = 0.0,
        p_y: typing.SupportsFloat = 0.0,
        p_z: typing.SupportsFloat = 0.0,
        energy: typing.SupportsFloat = 0.0,
        mass: typing.SupportsFloat = 0.0,
        lifetime: typing.SupportsFloat = 0.0,
        spin: typing.SupportsFloat = 0.0,
    ) -> None: ...
    @property
    def anti_color(self) -> int: ...
    @anti_color.setter
    def anti_color(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def color(self) -> int: ...
    @color.setter
    def color(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def energy(self) -> float: ...
    @energy.setter
    def energy(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def lifetime(self) -> float: ...
    @lifetime.setter
    def lifetime(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def mass(self) -> float: ...
    @mass.setter
    def mass(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def mother1(self) -> int: ...
    @mother1.setter
    def mother1(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def mother2(self) -> int: ...
    @mother2.setter
    def mother2(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def pdg_id(self) -> int: ...
    @pdg_id.setter
    def pdg_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def px(self) -> float: ...
    @px.setter
    def px(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def py(self) -> float: ...
    @py.setter
    def py(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def pz(self) -> float: ...
    @pz.setter
    def pz(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def spin(self) -> float: ...
    @spin.setter
    def spin(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def status_code(self) -> int: ...
    @status_code.setter
    def status_code(self, arg0: typing.SupportsInt) -> None: ...

class LHEProcess:
    @staticmethod
    def __init__(*args, **kwargs) -> None: ...
    @property
    def cross_section(self) -> float: ...
    @cross_section.setter
    def cross_section(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def cross_section_error(self) -> float: ...
    @cross_section_error.setter
    def cross_section_error(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def max_weight(self) -> float: ...
    @max_weight.setter
    def max_weight(self, arg0: typing.SupportsFloat) -> None: ...
    @property
    def process_id(self) -> int: ...
    @process_id.setter
    def process_id(self, arg0: typing.SupportsInt) -> None: ...

class LineRef:
    def __init__(self, str: str) -> None: ...
    def __repr__(self) -> str: ...

class Logger:
    class LogLevel:
        """
        Members:

          level_debug

          level_info

          level_warning

          level_error
        """

        __members__: typing.ClassVar[
            dict[str, Logger.LogLevel]
        ]  # value = {'level_debug': <LogLevel.level_debug: 0>, 'level_info': <LogLevel.level_info: 1>, 'level_warning': <LogLevel.level_warning: 2>, 'level_error': <LogLevel.level_error: 3>}
        level_debug: typing.ClassVar[
            Logger.LogLevel
        ]  # value = <LogLevel.level_debug: 0>
        level_error: typing.ClassVar[
            Logger.LogLevel
        ]  # value = <LogLevel.level_error: 3>
        level_info: typing.ClassVar[Logger.LogLevel]  # value = <LogLevel.level_info: 1>
        level_warning: typing.ClassVar[
            Logger.LogLevel
        ]  # value = <LogLevel.level_warning: 2>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        @typing.overload
        def __init__(self, value: typing.SupportsInt) -> None: ...
        @typing.overload
        def __init__(self, name: str) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    level_debug: typing.ClassVar[Logger.LogLevel]  # value = <LogLevel.level_debug: 0>
    level_error: typing.ClassVar[Logger.LogLevel]  # value = <LogLevel.level_error: 3>
    level_info: typing.ClassVar[Logger.LogLevel]  # value = <LogLevel.level_info: 1>
    level_warning: typing.ClassVar[
        Logger.LogLevel
    ]  # value = <LogLevel.level_warning: 2>
    @staticmethod
    def debug(message: str) -> None: ...
    @staticmethod
    def error(message: str) -> None: ...
    @staticmethod
    def info(message: str) -> None: ...
    @staticmethod
    def log(level: Logger.LogLevel, message: str) -> None: ...
    @staticmethod
    def set_log_handler(
        func: collections.abc.Callable[[Logger.LogLevel, str], None],
    ) -> None: ...
    @staticmethod
    def warning(message: str) -> None: ...

class Luminosity(Mapping):
    def __init__(
        self,
        s_lab: typing.SupportsFloat,
        s_hat_min: typing.SupportsFloat,
        s_hat_max: typing.SupportsFloat = 0.0,
        invariant_power: typing.SupportsFloat = 0.0,
        mass: typing.SupportsFloat = 0.0,
        width: typing.SupportsFloat = 0.0,
    ) -> None: ...

class MLP(FunctionGenerator):
    class Activation:
        """
        Members:

          relu

          leaky_relu

          elu

          gelu

          sigmoid

          softplus

          linear
        """

        __members__: typing.ClassVar[
            dict[str, MLP.Activation]
        ]  # value = {'relu': <Activation.relu: 0>, 'leaky_relu': <Activation.leaky_relu: 1>, 'elu': <Activation.elu: 2>, 'gelu': <Activation.gelu: 3>, 'sigmoid': <Activation.sigmoid: 4>, 'softplus': <Activation.softplus: 5>, 'linear': <Activation.linear: 6>}
        elu: typing.ClassVar[MLP.Activation]  # value = <Activation.elu: 2>
        gelu: typing.ClassVar[MLP.Activation]  # value = <Activation.gelu: 3>
        leaky_relu: typing.ClassVar[
            MLP.Activation
        ]  # value = <Activation.leaky_relu: 1>
        linear: typing.ClassVar[MLP.Activation]  # value = <Activation.linear: 6>
        relu: typing.ClassVar[MLP.Activation]  # value = <Activation.relu: 0>
        sigmoid: typing.ClassVar[MLP.Activation]  # value = <Activation.sigmoid: 4>
        softplus: typing.ClassVar[MLP.Activation]  # value = <Activation.softplus: 5>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        @typing.overload
        def __init__(self, value: typing.SupportsInt) -> None: ...
        @typing.overload
        def __init__(self, name: str) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    elu: typing.ClassVar[MLP.Activation]  # value = <Activation.elu: 2>
    gelu: typing.ClassVar[MLP.Activation]  # value = <Activation.gelu: 3>
    leaky_relu: typing.ClassVar[MLP.Activation]  # value = <Activation.leaky_relu: 1>
    linear: typing.ClassVar[MLP.Activation]  # value = <Activation.linear: 6>
    relu: typing.ClassVar[MLP.Activation]  # value = <Activation.relu: 0>
    sigmoid: typing.ClassVar[MLP.Activation]  # value = <Activation.sigmoid: 4>
    softplus: typing.ClassVar[MLP.Activation]  # value = <Activation.softplus: 5>
    def __init__(
        self,
        input_dim: typing.SupportsInt,
        output_dim: typing.SupportsInt,
        hidden_dim: typing.SupportsInt = 32,
        layers: typing.SupportsInt = 3,
        activation: MLP.Activation = MLP.Activation.Activation.leaky_relu,
        prefix: str = "",
    ) -> None: ...
    def initialize_globals(self, context: Context) -> None: ...
    def input_dim(self) -> int: ...
    def output_dim(self) -> int: ...

class Mapping:
    def __init__(
        self,
        name: str,
        input_types: collections.abc.Sequence[Type],
        output_types: collections.abc.Sequence[Type],
        condition_types: collections.abc.Sequence[Type],
    ) -> None: ...
    def build_forward(
        self,
        builder: FunctionBuilder,
        inputs: collections.abc.Sequence[Value],
        conditions: collections.abc.Sequence[Value],
    ) -> tuple[list[Value], Value]: ...
    def build_inverse(
        self,
        builder: FunctionBuilder,
        inputs: collections.abc.Sequence[Value],
        conditions: collections.abc.Sequence[Value],
    ) -> tuple[list[Value], Value]: ...
    def forward_function(self) -> Function: ...
    def inverse_function(self) -> Function: ...
    def map_forward(self, inputs, conditions=list()): ...
    def map_inverse(self, inputs, conditions=list()): ...

class MatrixElement(FunctionGenerator):
    class MatrixElementInput:
        """
        Members:

          momenta_in

          alpha_s_in

          flavor_in

          random_color_in

          random_helicity_in

          random_diagram_in

          helicity_in

          diagram_in
        """

        __members__: typing.ClassVar[
            dict[str, MatrixElement.MatrixElementInput]
        ]  # value = {'momenta_in': <MatrixElementInput.momenta_in: 0>, 'alpha_s_in': <MatrixElementInput.alpha_s_in: 1>, 'flavor_in': <MatrixElementInput.flavor_in: 2>, 'random_color_in': <MatrixElementInput.random_color_in: 3>, 'random_helicity_in': <MatrixElementInput.random_helicity_in: 4>, 'random_diagram_in': <MatrixElementInput.random_diagram_in: 5>, 'helicity_in': <MatrixElementInput.helicity_in: 6>, 'diagram_in': <MatrixElementInput.diagram_in: 7>}
        alpha_s_in: typing.ClassVar[
            MatrixElement.MatrixElementInput
        ]  # value = <MatrixElementInput.alpha_s_in: 1>
        diagram_in: typing.ClassVar[
            MatrixElement.MatrixElementInput
        ]  # value = <MatrixElementInput.diagram_in: 7>
        flavor_in: typing.ClassVar[
            MatrixElement.MatrixElementInput
        ]  # value = <MatrixElementInput.flavor_in: 2>
        helicity_in: typing.ClassVar[
            MatrixElement.MatrixElementInput
        ]  # value = <MatrixElementInput.helicity_in: 6>
        momenta_in: typing.ClassVar[
            MatrixElement.MatrixElementInput
        ]  # value = <MatrixElementInput.momenta_in: 0>
        random_color_in: typing.ClassVar[
            MatrixElement.MatrixElementInput
        ]  # value = <MatrixElementInput.random_color_in: 3>
        random_diagram_in: typing.ClassVar[
            MatrixElement.MatrixElementInput
        ]  # value = <MatrixElementInput.random_diagram_in: 5>
        random_helicity_in: typing.ClassVar[
            MatrixElement.MatrixElementInput
        ]  # value = <MatrixElementInput.random_helicity_in: 4>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        @typing.overload
        def __init__(self, value: typing.SupportsInt) -> None: ...
        @typing.overload
        def __init__(self, name: str) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    class MatrixElementOutput:
        """
        Members:

          matrix_element_out

          diagram_amp2_out

          color_index_out

          helicity_index_out

          diagram_index_out
        """

        __members__: typing.ClassVar[
            dict[str, MatrixElement.MatrixElementOutput]
        ]  # value = {'matrix_element_out': <MatrixElementOutput.matrix_element_out: 0>, 'diagram_amp2_out': <MatrixElementOutput.diagram_amp2_out: 1>, 'color_index_out': <MatrixElementOutput.color_index_out: 2>, 'helicity_index_out': <MatrixElementOutput.helicity_index_out: 3>, 'diagram_index_out': <MatrixElementOutput.diagram_index_out: 4>}
        color_index_out: typing.ClassVar[
            MatrixElement.MatrixElementOutput
        ]  # value = <MatrixElementOutput.color_index_out: 2>
        diagram_amp2_out: typing.ClassVar[
            MatrixElement.MatrixElementOutput
        ]  # value = <MatrixElementOutput.diagram_amp2_out: 1>
        diagram_index_out: typing.ClassVar[
            MatrixElement.MatrixElementOutput
        ]  # value = <MatrixElementOutput.diagram_index_out: 4>
        helicity_index_out: typing.ClassVar[
            MatrixElement.MatrixElementOutput
        ]  # value = <MatrixElementOutput.helicity_index_out: 3>
        matrix_element_out: typing.ClassVar[
            MatrixElement.MatrixElementOutput
        ]  # value = <MatrixElementOutput.matrix_element_out: 0>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        @typing.overload
        def __init__(self, value: typing.SupportsInt) -> None: ...
        @typing.overload
        def __init__(self, name: str) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    alpha_s_in: typing.ClassVar[
        MatrixElement.MatrixElementInput
    ]  # value = <MatrixElementInput.alpha_s_in: 1>
    color_index_out: typing.ClassVar[
        MatrixElement.MatrixElementOutput
    ]  # value = <MatrixElementOutput.color_index_out: 2>
    diagram_amp2_out: typing.ClassVar[
        MatrixElement.MatrixElementOutput
    ]  # value = <MatrixElementOutput.diagram_amp2_out: 1>
    diagram_in: typing.ClassVar[
        MatrixElement.MatrixElementInput
    ]  # value = <MatrixElementInput.diagram_in: 7>
    diagram_index_out: typing.ClassVar[
        MatrixElement.MatrixElementOutput
    ]  # value = <MatrixElementOutput.diagram_index_out: 4>
    flavor_in: typing.ClassVar[
        MatrixElement.MatrixElementInput
    ]  # value = <MatrixElementInput.flavor_in: 2>
    helicity_in: typing.ClassVar[
        MatrixElement.MatrixElementInput
    ]  # value = <MatrixElementInput.helicity_in: 6>
    helicity_index_out: typing.ClassVar[
        MatrixElement.MatrixElementOutput
    ]  # value = <MatrixElementOutput.helicity_index_out: 3>
    matrix_element_out: typing.ClassVar[
        MatrixElement.MatrixElementOutput
    ]  # value = <MatrixElementOutput.matrix_element_out: 0>
    momenta_in: typing.ClassVar[
        MatrixElement.MatrixElementInput
    ]  # value = <MatrixElementInput.momenta_in: 0>
    random_color_in: typing.ClassVar[
        MatrixElement.MatrixElementInput
    ]  # value = <MatrixElementInput.random_color_in: 3>
    random_diagram_in: typing.ClassVar[
        MatrixElement.MatrixElementInput
    ]  # value = <MatrixElementInput.random_diagram_in: 5>
    random_helicity_in: typing.ClassVar[
        MatrixElement.MatrixElementInput
    ]  # value = <MatrixElementInput.random_helicity_in: 4>
    @typing.overload
    def __init__(
        self,
        matrix_element_index: typing.SupportsInt,
        particle_count: typing.SupportsInt,
        inputs: collections.abc.Sequence[MatrixElement.MatrixElementInput],
        outputs: collections.abc.Sequence[MatrixElement.MatrixElementOutput],
        diagram_count: typing.SupportsInt = 1,
        sample_random_inputs: bool = False,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        matrix_element_api: MatrixElementApi,
        inputs: collections.abc.Sequence[MatrixElement.MatrixElementInput],
        outputs: collections.abc.Sequence[MatrixElement.MatrixElementOutput],
        sample_random_inputs: bool = False,
    ) -> None: ...
    def diagram_count(self) -> int: ...
    def matrix_element_index(self) -> int: ...
    def particle_count(self) -> int: ...

class MatrixElementApi:
    def __init__(
        self, file: str, param_card: str, index: typing.SupportsInt = 0
    ) -> None: ...
    def device(self) -> Device: ...
    def diagram_count(self) -> int: ...
    def helicity_count(self) -> int: ...
    def index(self) -> int: ...
    def particle_count(self) -> int: ...

class MomentumPreprocessing(FunctionGenerator):
    def __init__(self, particle_count: typing.SupportsInt) -> None: ...
    def output_dim(self) -> int: ...

class MultiChannelFunction(FunctionGenerator):
    def __init__(
        self, functions: collections.abc.Sequence[FunctionGenerator]
    ) -> None: ...

class MultiChannelIntegrand(FunctionGenerator):
    def __init__(self, integrands: collections.abc.Sequence[Integrand]) -> None: ...

class MultiChannelMapping(Mapping):
    def __init__(self, mappings: collections.abc.Sequence[Mapping]) -> None: ...

class PartonDensity(FunctionGenerator):
    def __init__(
        self,
        grid: PdfGrid,
        pids: collections.abc.Sequence[typing.SupportsInt],
        dynamic_pid: bool = False,
        prefix: str = "",
    ) -> None: ...

class PdfGrid:
    def __init__(self, file: str) -> None: ...
    def coefficients_shape(self, batch_dim: bool = False) -> list[int]: ...
    def initialize_globals(self, context: Context, prefix: str = "") -> None: ...
    def logq2_shape(self, batch_dim: bool = False) -> list[int]: ...
    def logx_shape(self, batch_dim: bool = False) -> list[int]: ...
    @property
    def grid_point_count(self) -> int: ...
    @property
    def logq2(self) -> list[float]: ...
    @property
    def logx(self) -> list[float]: ...
    @property
    def pids(self) -> list[int]: ...
    @property
    def q(self) -> list[float]: ...
    @property
    def q_count(self) -> int: ...
    @property
    def region_sizes(self) -> list[int]: ...
    @property
    def values(self) -> list[list[float]]: ...
    @property
    def x(self) -> list[float]: ...

class PhaseSpaceMapping(Mapping):
    class TChannelMode:
        """
        Members:

          propagator

          rambo

          chili
        """

        __members__: typing.ClassVar[
            dict[str, PhaseSpaceMapping.TChannelMode]
        ]  # value = {'propagator': <TChannelMode.propagator: 0>, 'rambo': <TChannelMode.rambo: 1>, 'chili': <TChannelMode.chili: 2>}
        chili: typing.ClassVar[
            PhaseSpaceMapping.TChannelMode
        ]  # value = <TChannelMode.chili: 2>
        propagator: typing.ClassVar[
            PhaseSpaceMapping.TChannelMode
        ]  # value = <TChannelMode.propagator: 0>
        rambo: typing.ClassVar[
            PhaseSpaceMapping.TChannelMode
        ]  # value = <TChannelMode.rambo: 1>
        def __eq__(self, other: typing.Any) -> bool: ...
        def __getstate__(self) -> int: ...
        def __hash__(self) -> int: ...
        def __index__(self) -> int: ...
        @typing.overload
        def __init__(self, value: typing.SupportsInt) -> None: ...
        @typing.overload
        def __init__(self, name: str) -> None: ...
        def __int__(self) -> int: ...
        def __ne__(self, other: typing.Any) -> bool: ...
        def __repr__(self) -> str: ...
        def __setstate__(self, state: typing.SupportsInt) -> None: ...
        def __str__(self) -> str: ...
        @property
        def name(self) -> str: ...
        @property
        def value(self) -> int: ...

    chili: typing.ClassVar[
        PhaseSpaceMapping.TChannelMode
    ]  # value = <TChannelMode.chili: 2>
    propagator: typing.ClassVar[
        PhaseSpaceMapping.TChannelMode
    ]  # value = <TChannelMode.propagator: 0>
    rambo: typing.ClassVar[
        PhaseSpaceMapping.TChannelMode
    ]  # value = <TChannelMode.rambo: 1>
    @typing.overload
    def __init__(
        self,
        topology: Topology,
        cm_energy: typing.SupportsFloat,
        leptonic: bool = False,
        invariant_power: typing.SupportsFloat = 0.8,
        t_channel_mode: PhaseSpaceMapping.TChannelMode = PhaseSpaceMapping.TChannelMode.TChannelMode.propagator,
        cuts: madevent7._madevent_py.Cuts | None = None,
        permutations: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ] = [],
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        masses: collections.abc.Sequence[typing.SupportsFloat],
        cm_energy: typing.SupportsFloat,
        leptonic: bool = False,
        invariant_power: typing.SupportsFloat = 0.8,
        mode: PhaseSpaceMapping.TChannelMode = PhaseSpaceMapping.TChannelMode.TChannelMode.rambo,
        cuts: madevent7._madevent_py.Cuts | None = None,
    ) -> None: ...
    def channel_count(self) -> int: ...
    def particle_count(self) -> int: ...
    def random_dim(self) -> int: ...

class PrettyBox:
    def __init__(
        self,
        title: str,
        rows: typing.SupportsInt,
        columns: collections.abc.Sequence[typing.SupportsInt],
        offset: typing.SupportsInt = 0,
        box_width: typing.SupportsInt = 91,
    ) -> None: ...
    def print_first(self) -> None: ...
    def print_update(self) -> None: ...
    def set_cell(
        self, row: typing.SupportsInt, column: typing.SupportsInt, value: str
    ) -> None: ...
    def set_column(
        self, column: typing.SupportsInt, values: collections.abc.Sequence[str]
    ) -> None: ...
    def set_row(
        self, row: typing.SupportsInt, values: collections.abc.Sequence[str]
    ) -> None: ...
    @property
    def line_count(self) -> int: ...

class Propagator:
    def __init__(
        self,
        mass: typing.SupportsFloat = 0.0,
        width: typing.SupportsFloat = 0.0,
        integration_order: typing.SupportsInt = 0,
        e_min: typing.SupportsFloat = 0.0,
        e_max: typing.SupportsFloat = 0.0,
        pdg_id: typing.SupportsInt = 0,
    ) -> None: ...
    @property
    def e_max(self) -> float: ...
    @property
    def e_min(self) -> float: ...
    @property
    def integration_order(self) -> int: ...
    @property
    def mass(self) -> float: ...
    @property
    def pdg_id(self) -> int: ...
    @property
    def width(self) -> float: ...

class PropagatorChannelWeights(FunctionGenerator):
    def __init__(
        self,
        topologies: collections.abc.Sequence[Topology],
        permutations: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ],
        channel_indices: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
    ) -> None: ...

class RunningCoupling(FunctionGenerator):
    def __init__(self, grid: AlphaSGrid, prefix: str = "") -> None: ...

class SubchannelWeights(FunctionGenerator):
    def __init__(
        self,
        topologies: collections.abc.Sequence[collections.abc.Sequence[Topology]],
        permutations: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ],
        channel_indices: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ],
    ) -> None: ...
    def channel_count(self) -> int: ...

class SubprocArgs:
    def __init__(
        self,
        process_id: typing.SupportsInt = 0,
        topologies: collections.abc.Sequence[Topology] = [],
        permutations: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ] = [],
        diagram_indices: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsInt]
        ] = [],
        diagram_color_indices: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ] = [],
        color_flows: collections.abc.Sequence[
            collections.abc.Sequence[
                collections.abc.Sequence[tuple[typing.SupportsInt, typing.SupportsInt]]
            ]
        ] = [],
        pdg_color_types: collections.abc.Mapping[
            typing.SupportsInt, typing.SupportsInt
        ] = {},
        helicities: collections.abc.Sequence[
            collections.abc.Sequence[typing.SupportsFloat]
        ] = [],
        pdg_ids: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ] = [],
        matrix_flavor_indices: collections.abc.Sequence[typing.SupportsInt] = [],
    ) -> None: ...
    @property
    def color_flows(self) -> list[list[list[tuple[int, int]]]]: ...
    @color_flows.setter
    def color_flows(
        self,
        arg0: collections.abc.Sequence[
            collections.abc.Sequence[
                collections.abc.Sequence[tuple[typing.SupportsInt, typing.SupportsInt]]
            ]
        ],
    ) -> None: ...
    @property
    def diagram_color_indices(self) -> list[list[list[int]]]: ...
    @diagram_color_indices.setter
    def diagram_color_indices(
        self,
        arg0: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ],
    ) -> None: ...
    @property
    def diagram_indices(self) -> list[list[int]]: ...
    @diagram_indices.setter
    def diagram_indices(
        self,
        arg0: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]],
    ) -> None: ...
    @property
    def helicities(self) -> list[list[float]]: ...
    @helicities.setter
    def helicities(
        self,
        arg0: collections.abc.Sequence[collections.abc.Sequence[typing.SupportsFloat]],
    ) -> None: ...
    @property
    def matrix_flavor_indices(self) -> list[int]: ...
    @matrix_flavor_indices.setter
    def matrix_flavor_indices(
        self, arg0: collections.abc.Sequence[typing.SupportsInt]
    ) -> None: ...
    @property
    def pdg_color_types(self) -> dict[int, int]: ...
    @pdg_color_types.setter
    def pdg_color_types(
        self, arg0: collections.abc.Mapping[typing.SupportsInt, typing.SupportsInt]
    ) -> None: ...
    @property
    def pdg_ids(self) -> list[list[list[int]]]: ...
    @pdg_ids.setter
    def pdg_ids(
        self,
        arg0: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ],
    ) -> None: ...
    @property
    def permutations(self) -> list[list[list[int]]]: ...
    @permutations.setter
    def permutations(
        self,
        arg0: collections.abc.Sequence[
            collections.abc.Sequence[collections.abc.Sequence[typing.SupportsInt]]
        ],
    ) -> None: ...
    @property
    def process_id(self) -> int: ...
    @process_id.setter
    def process_id(self, arg0: typing.SupportsInt) -> None: ...
    @property
    def topologies(self) -> list[Topology]: ...
    @topologies.setter
    def topologies(self, arg0: collections.abc.Sequence[Topology]) -> None: ...

class TPropagatorMapping(Mapping):
    def __init__(
        self,
        integration_order: collections.abc.Sequence[typing.SupportsInt],
        invariant_power: typing.SupportsFloat = 0.0,
    ) -> None: ...

class Tensor:
    @staticmethod
    def numpy(tensor): ...
    @staticmethod
    def torch(tensor): ...
    def __dlpack__(
        self,
        stream: typing.SupportsInt | None = None,
        max_version: tuple[typing.SupportsInt, typing.SupportsInt] | None = None,
        dl_device: typing.SupportsInt | None = None,
        copy: bool | None = None,
    ) -> typing.Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

class ThreeBodyDecay(Mapping):
    def __init__(self, com: bool) -> None: ...

class Topology:
    @staticmethod
    def topologies(diagram: Diagram) -> list[Topology]: ...
    def __init__(self, diagram: Diagram) -> None: ...
    def propagator_momentum_terms(
        self, arg0: bool
    ) -> list[tuple[list[int], float, float]]: ...
    @property
    def decay_integration_order(self) -> list[int]: ...
    @property
    def decays(self) -> list[Decay]: ...
    @property
    def incoming_masses(self) -> list[float]: ...
    @property
    def outgoing_indices(self) -> list[int]: ...
    @property
    def outgoing_masses(self) -> list[float]: ...
    @property
    def t_integration_order(self) -> list[int]: ...
    @property
    def t_propagator_count(self) -> int: ...
    @property
    def t_propagator_masses(self) -> list[float]: ...
    @property
    def t_propagator_widths(self) -> list[float]: ...

class TwoBodyDecay(Mapping):
    def __init__(self, com: bool) -> None: ...

class TwoToThreeParticleScattering(Mapping):
    def __init__(
        self,
        t_invariant_power: typing.SupportsFloat = 0.0,
        t_mass: typing.SupportsFloat = 0.0,
        t_width: typing.SupportsFloat = 0.0,
        s_invariant_power: typing.SupportsFloat = 0.0,
        s_mass: typing.SupportsFloat = 0.0,
        s_width: typing.SupportsFloat = 0.0,
    ) -> None: ...

class TwoToTwoParticleScattering(Mapping):
    def __init__(
        self,
        com: bool,
        invariant_power: typing.SupportsFloat = 0.0,
        mass: typing.SupportsFloat = 0.0,
        width: typing.SupportsFloat = 0.0,
    ) -> None: ...

class Type:
    @typing.overload
    def __init__(
        self,
        dtype: DataType,
        batch_size: BatchSize,
        shape: collections.abc.Sequence[typing.SupportsInt],
    ) -> None: ...
    @typing.overload
    def __init__(
        self, batch_size_list: collections.abc.Sequence[BatchSize]
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def batch_size(self) -> BatchSize: ...
    @property
    def dtype(self) -> DataType: ...
    @property
    def shape(self) -> list[int]: ...

class Unweighter(FunctionGenerator):
    def __init__(self, types: collections.abc.Sequence[Type]) -> None: ...

class Value:
    @typing.overload
    def __init__(self, value: typing.SupportsInt) -> None: ...
    @typing.overload
    def __init__(self, value: typing.SupportsFloat) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    @property
    def literal_value(
        self,
    ) -> int | float | tuple[list[int], list[int] | list[float]] | None: ...
    @property
    def local_index(self) -> int: ...
    @property
    def type(self) -> Type: ...

class VegasGridOptimizer:
    def __init__(
        self, context: Context, grid_name: str, damping: typing.SupportsFloat
    ) -> None: ...
    def add_data(self, values: typing.Any, counts: typing.Any) -> None: ...
    def optimize(self) -> None: ...

class VegasHistogram(FunctionGenerator):
    def __init__(
        self, dimension: typing.SupportsInt, bin_count: typing.SupportsInt
    ) -> None: ...

class VegasMapping(Mapping):
    def __init__(
        self,
        dimension: typing.SupportsInt,
        bin_count: typing.SupportsInt,
        prefix: str = "",
    ) -> None: ...
    def grid_name(self) -> str: ...
    def initialize_globals(self, context: Context) -> None: ...

def batch_float_array(count: typing.SupportsInt) -> Type: ...
def batch_four_vec_array(count: typing.SupportsInt) -> Type: ...
def cpu_device() -> Device: ...
def cuda_device() -> Device: ...
def default_context() -> Context: ...
def default_cuda_context() -> Context: ...
def default_hip_context() -> Context: ...
def format_progress(
    progress: typing.SupportsFloat, width: typing.SupportsInt
) -> str: ...
def format_si_prefix(value: typing.SupportsFloat) -> str: ...
def format_with_error(
    value: typing.SupportsFloat, error: typing.SupportsFloat
) -> str: ...
def hip_device() -> Device: ...
def initialize_vegas_grid(context: Context, grid_name: str) -> None: ...
def multichannel_batch_size(count: typing.SupportsInt) -> Type: ...
def set_lib_path(lib_path: str) -> None: ...
def set_simd_vector_size(vector_size: typing.SupportsInt) -> None: ...
def set_thread_count(new_count: typing.SupportsInt) -> None: ...

batch_float: Type  # value = float[batch_size]
batch_four_vec: Type  # value = float[batch_size, 4]
batch_int: Type  # value = int[batch_size]
batch_size: BatchSize  # value = batch_size
batch_sizes: DataType  # value = <DataType.batch_sizes: 2>
float: DataType  # value = <DataType.float: 1>
int: DataType  # value = <DataType.int: 0>
log: EventGeneratorVerbosity  # value = <EventGeneratorVerbosity.log: 1>
pretty: EventGeneratorVerbosity  # value = <EventGeneratorVerbosity.pretty: 2>
silent: EventGeneratorVerbosity  # value = <EventGeneratorVerbosity.silent: 0>
single_float: Type  # value = float[1]
single_int: Type  # value = int[1]
