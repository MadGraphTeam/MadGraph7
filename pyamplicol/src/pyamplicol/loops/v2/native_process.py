# SPDX-License-Identifier: 0BSD
"""Native evaluation of a v2 process without its ``LoopSlice`` (WP3.1 I6).

``NativeExecutor(slice_, inputs="native")`` evaluates a point from the plan
v4 program and its native tree program alone: the executor binds every
parameter-dependent value from the external model parameters and forms the
trees, the Born, the counterterm and external-field components and the loop
natively (``native_trees``). What it still reads from the generated slice
is small and structural: the process legs, the squared-order sectors and
their (Born, loop) pairs, the normalisation, the request's fixed
parameters, scale parameter, colour accuracy and contraction, and the lane
labels of the plan's families, loop orders and columns.

``NativeProcess`` holds exactly that: the plan JSON, the tree program
(``native_trees.NativeTrees``: its JSON, saved kernels and parameter
evaluator) and an assembly record, all serialisable, so a stored process
evaluates without regenerating its slice. ``NativeProcess.executor`` builds
the ``NativeExecutor`` of a loaded process on a structural view of the
record (``_ProcessView``); its results equal those of the slice's own
native executor bitwise (``tests/unit/test_loop_v2_native_process.py``).
Lane labels are the family indices of the plan (the slice's family keys are
labels only: the bare virtual and the stability budgets read the sector).

A process with certified-scaleless families also stores their companion
(``LoopSlice.scaleless``, design §12.4): its plan (the artifact's companion
plan member) and its own tree program, exported without Born or counterterm
outputs (it adds only lanes). The "uv" and "ir" pole selections of a loaded
process run it, so no pole selection regenerates the slice.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from .families import Orders
from .native_trees import NativeTrees
from .poles import PoleRegulators

SCHEMA = 2
# Member-name prefix of the certified-scaleless companion's members (its plan
# and tree program).
SCALELESS = "scaleless-"


def _orders(orders: Orders) -> list[list[Any]]:
    return [[name, power] for name, power in orders]


def _orders_of(record: Any) -> Orders:
    return tuple((str(name), int(power)) for name, power in record)


def _trees_record(trees: NativeTrees) -> dict[str, Any]:
    """The tree program's batch layout (``NativeTrees`` fields besides the
    program, the saved evaluators and their plane applications)."""

    return {
        "couplings": [list(slot) for slot in trees.couplings],
        "born_orders": [_orders(orders) for orders in trees.born_orders],
        "squared_orders": [_orders(o) for o in trees.squared_orders],
        "masses": [list(mass) for mass in trees.masses],
        "components": [
            [b, _orders(orders), grade] for b, orders, grade in trees.components
        ],
        "counterterm_outputs": [
            [component, _orders(sector), power]
            for component, sector, power in trees.counterterm_outputs
        ],
    }


def _trees_members(trees: NativeTrees, prefix: str = "") -> dict[str, bytes]:
    """A tree program as named byte members: its JSON, its saved kernels and
    parameter evaluator, and their binary64 SymJIT plane applications."""

    if len(trees.planes) != len(trees.kernels) + 1:
        raise ValueError("a native process stores a plane application per state")
    out = {
        f"{prefix}tree-program.json": trees.json().encode(),
        f"{prefix}parameters.bin": bytes(trees.parameters),
        f"{prefix}parameters.symjit": bytes(trees.planes[-1]),
    }
    for k, saved in enumerate(trees.kernels):
        out[f"{prefix}kernel-{k}.bin"] = bytes(saved)
        out[f"{prefix}kernel-{k}.symjit"] = bytes(trees.planes[k])
    return out


def _trees_from_members(
    members: Mapping[str, bytes], meta: Mapping[str, Any], prefix: str = ""
) -> NativeTrees:
    program = json.loads(members[f"{prefix}tree-program.json"])
    kernels = []
    while f"{prefix}kernel-{len(kernels)}.bin" in members:
        kernels.append(bytes(members[f"{prefix}kernel-{len(kernels)}.bin"]))
    planes = [bytes(members[f"{prefix}kernel-{k}.symjit"]) for k in range(len(kernels))]
    planes.append(bytes(members[f"{prefix}parameters.symjit"]))
    return NativeTrees(
        program,
        tuple(kernels),
        tuple((str(name), int(grade)) for name, grade in meta["couplings"]),
        tuple(_orders_of(o) for o in meta["born_orders"]),
        tuple(_orders_of(o) for o in meta["squared_orders"]),
        tuple((int(pdg), str(name)) for pdg, name in meta["masses"]),
        tuple(
            (int(b), _orders_of(orders), int(grade))
            for b, orders, grade in meta["components"]
        ),
        bytes(members[f"{prefix}parameters.bin"]),
        tuple(
            (str(component), _orders_of(sector), int(power))
            for component, sector, power in meta["counterterm_outputs"]
        ),
        tuple(planes),
    )


def _plan_view(
    record: Mapping[str, Any], columns: Any, trees: NativeTrees, first: int = 0
) -> Any:
    """The ``ExportedPlan`` view a native executor reads: lane labels (plan
    family indices from ``first``, loop orders, sector columns), the tree
    program and the plan's seed Fourier record (``plan_export.
    fourier_record``; the record's ``fourier``, absent when every seed is
    contracted directly). The companion's families are numbered after the
    process's, so the lanes of both stay distinct in one evaluation
    (``assemble._PointEvaluation.set_lanes``)."""

    from .plan_export import ExportedPlan

    fourier = record.get("fourier")
    return ExportedPlan(
        program={},
        rows=(),
        born_components=(),
        block_rows=(),
        constants=(),
        masses=(),
        orders=tuple(_orders_of(o) for o in record["orders"]),
        families=tuple(range(first, first + record["families"])),
        columns=columns,
        trees=trees,
        fourier=None if fourier is None else json.dumps(fourier, separators=(",", ":")),
    )


@dataclass(frozen=True)
class NativeCompanion:
    """The certified-scaleless companion of a native process: its plan JSON
    (the artifact's companion plan), its tree program (no Born or
    counterterm outputs) and its lane labels (``record``: plan family count
    and loop orders; its columns are the process's sectors)."""

    plan_json: str
    trees: NativeTrees
    record: Mapping[str, Any]


@dataclass(frozen=True)
class NativeProcess:
    """The native evaluation of one v2 process: plan, tree program and the
    assembly record (module docstring), with the native companion of its
    certified-scaleless families when it has them."""

    plan_json: str
    trees: NativeTrees
    record: Mapping[str, Any]
    scaleless: NativeCompanion | None = None
    # The stored observable columns (``native_observables.StoredObservables``)
    # of a process generated with declared colour histories or spin classes.
    observables: Any = None

    @classmethod
    def of(cls, executor: Any) -> NativeProcess:
        """The process of a slice's native executor
        (``NativeExecutor(slice_, inputs="native")``, sector columns)."""

        if executor.inputs != "native":
            raise ValueError("a native process needs an executor with native inputs")
        companion = executor.scaleless
        return cls.export(
            executor.slice,
            executor.exported,
            executor._plan_json,
            None if companion is None else (companion.exported, companion._plan_json),
        )

    @classmethod
    def export(
        cls,
        slice_: Any,
        exported: Any,
        plan_json: str,
        scaleless: tuple[Any, str] | None = None,
    ) -> NativeProcess:
        """The process of a plan exported with its tree program
        (``export_plan(slice_, trees=True, born_squared=True)``, sector
        columns); ``plan_json`` is ``exported.json()``. ``scaleless`` is the
        (exported plan, JSON) of the slice's certified-scaleless companion,
        exported with its tree program (``trees=True, born_squared=False,
        counterterms=False``); required when the slice has one."""

        trees = exported.trees
        born = None if trees is None else trees.program["born"]
        if born is None or born["metric"] is None:
            raise ValueError(
                "a native process needs a plan exported with trees and |A0|^2"
            )
        sectors = [sector for sector, _pairs in slice_.sectors]
        if tuple(exported.columns) != tuple(sectors):
            raise ValueError("a native process evaluates the sector columns")
        request = slice_.request
        record = {
            "schema": SCHEMA,
            "legs": [
                [int(leg.label), bool(leg.is_initial), int(leg.pdg)]
                for leg in sorted(slice_.blocks.process.legs, key=lambda x: x.label)
            ],
            "sectors": [
                [
                    _orders(sector),
                    [[_orders(born), _orders(loop)] for born, loop in sorted(pairs)],
                ]
                for sector, pairs in slice_.sectors
            ],
            "average": int(slice_.normalisation.average),
            "accuracy": request.accuracy,
            "colour_contraction": request.colour_contraction,
            # Every parameter fixed at generation (the request's and those
            # the model preparation fixes): refused at another value and
            # applied by every runtime (``LoopSlice.fixed_parameters``).
            "fixed_parameters": slice_.fixed_parameters,
            # Those of them the model preparation fixed (scheme constants),
            # which a refusal names as such (``assemble.fixed_refusal``).
            "preparation_fixed_parameters": sorted(slice_.preparation_fixed_parameters),
            # The model's external parameters (those a caller may set): every
            # view of the stored process refuses another name
            # (``_ProcessView``), as the Python API and the SDKs do.
            "externals": sorted(slice_._external_parameters()),
            "scale_parameter": slice_.scale_parameter,
            "alpha_s_parameter": slice_.alpha_s_parameter,
            "families": len(exported.families),
            "orders": [_orders(orders) for orders in exported.orders],
            "trees": _trees_record(trees),
            "scaleless": None,
        }
        regulators = slice_.pole_regulators
        if regulators is not None:
            # The model's pole regulators (``poles.pole_regulators``): the
            # split selections evaluate the counterterms at the IR regulator
            # 0 too. Absent for a model without them (loop_sm records keep
            # their bytes).
            record["pole_regulators"] = regulators.record()
        if exported.fourier is not None:
            # The certified seed FFT of the plan's seeds (design §7.3): its
            # structural record, certified again by the native loader.
            record["fourier"] = json.loads(exported.fourier)
        companion = None
        if (slice_.scaleless is None) != (scaleless is None):
            raise ValueError(
                "a native process stores the certified-scaleless companion "
                "exactly when its slice has one"
            )
        if scaleless is not None:
            twin, twin_json = scaleless
            if twin.trees is None:
                raise ValueError(
                    "the stored companion needs its plan exported with trees"
                )
            if tuple(twin.columns) != tuple(sectors):
                raise ValueError("the companion evaluates the sector columns")
            twin_record = {
                "families": len(twin.families),
                "orders": [_orders(orders) for orders in twin.orders],
                "trees": _trees_record(twin.trees),
            }
            if twin.fourier is not None:
                twin_record["fourier"] = json.loads(twin.fourier)
            record["scaleless"] = twin_record
            companion = NativeCompanion(twin_json, twin.trees, twin_record)
        return cls(plan_json, trees, record, companion)

    # -- storage -------------------------------------------------------------

    def members(self) -> dict[str, bytes]:
        """The process as named byte members: the assembly record and the
        tree program (JSON), the tree program's saved kernels and its
        parameter evaluator, and their binary64 SymJIT plane applications
        (the plan is stored on its own)."""

        out = {
            "native-process.json": json.dumps(
                self.record, separators=(",", ":"), sort_keys=True
            ).encode(),
            **_trees_members(self.trees),
        }
        if self.scaleless is not None:
            out[f"{SCALELESS}plan.json"] = self.scaleless.plan_json.encode()
            out.update(_trees_members(self.scaleless.trees, SCALELESS))
        if self.observables is not None:
            out.update(self.observables.members())
        return out

    @classmethod
    def from_members(
        cls,
        plan_json: str,
        members: Mapping[str, bytes],
        scaleless_json: str | None = None,
    ) -> NativeProcess:
        """The process of ``members`` (``members``) for its plan and, when it
        has certified-scaleless families, its companion plan (by default the
        one ``members`` holds)."""

        record = json.loads(members["native-process.json"])
        if record.get("schema") != SCHEMA:
            raise ValueError(
                "unsupported native process schema; regenerate the artifact"
            )
        trees = _trees_from_members(members, record["trees"])
        companion = None
        twin = record["scaleless"]
        if scaleless_json is None and f"{SCALELESS}plan.json" in members:
            scaleless_json = bytes(members[f"{SCALELESS}plan.json"]).decode()
        if twin is not None:
            if scaleless_json is None:
                raise ValueError("the native process's companion plan is missing")
            companion = NativeCompanion(
                scaleless_json,
                _trees_from_members(members, twin["trees"], SCALELESS),
                twin,
            )
        elif scaleless_json is not None:
            raise ValueError("a companion plan without its native tree program")
        observables = None
        if "observables.json" in members:
            from .native_observables import StoredObservables

            observables = StoredObservables.from_members(members)
        return cls(plan_json, trees, record, companion, observables)

    # -- evaluation ----------------------------------------------------------

    def executor(
        self,
        native: Any,
        *,
        externals: Collection[str] | None = None,
        **options: Any,
    ) -> Any:
        """The native executor of the process (``plan_export.NativeExecutor``
        with native inputs) on its structural view; ``native`` is the loaded
        loop native module, ``externals`` the external parameters of the
        process's model (the caller's parameters are checked against them,
        ``_ProcessView._point_parameters``; the public runtime passes its
        artifact model's; by default the record's ``externals``), ``options``
        the executor's (lanes, threads, forms)."""

        from .plan_export import NativeExecutor

        record = self.record
        columns = tuple(_orders_of(sector) for sector, _pairs in record["sectors"])
        view = _ProcessView(record, native, externals)
        companion = None
        if self.scaleless is not None:
            twin = self.scaleless
            companion = (
                _plan_view(twin.record, columns, twin.trees, record["families"]),
                twin.plan_json,
            )
            view.scaleless = _ProcessView(record, native, externals)
        return NativeExecutor(
            view,
            forward=False,
            exported=_plan_view(record, columns, self.trees),
            plan_json=self.plan_json,
            inputs="native",
            scaleless=companion,
            **options,
        )


class _ProcessView:
    """The structural attributes of a ``LoopSlice`` the native executor and
    ``assemble._NativePoint`` read, from a native process record."""

    def __init__(
        self,
        record: Mapping[str, Any],
        native: Any,
        externals: Collection[str] | None = None,
    ):
        self.native = native
        # The caller's names, else the record's (a record of an older build
        # has none: its views then do not check names; regenerate it).
        if externals is None:
            externals = record.get("externals")
        self.externals = None if externals is None else frozenset(externals)
        self.preparation_fixed = frozenset(
            record.get("preparation_fixed_parameters", ())
        )
        # The last caller mapping checked and its values (one per batch).
        self._checked: tuple[Any, dict[str, str]] | None = None
        self.request = SimpleNamespace(
            accuracy=record["accuracy"],
            colour_contraction=record["colour_contraction"],
            fixed_parameters=dict(record["fixed_parameters"]),
        )
        self.sectors = tuple(
            (
                _orders_of(sector),
                frozenset((_orders_of(born), _orders_of(loop)) for born, loop in pairs),
            )
            for sector, pairs in record["sectors"]
        )
        self.normalisation = SimpleNamespace(average=int(record["average"]))
        # The certified-scaleless companion's view (``NativeProcess.
        # executor``): the same process, sectors and parameters.
        self.scaleless: _ProcessView | None = None
        self.scale_parameter = record["scale_parameter"]
        self.alpha_s_parameter = record["alpha_s_parameter"]
        self.pole_regulators = PoleRegulators.from_record(record.get("pole_regulators"))
        self.blocks = SimpleNamespace(
            process=SimpleNamespace(
                legs=tuple(
                    SimpleNamespace(label=label, is_initial=initial, pdg=pdg)
                    for label, initial, pdg in record["legs"]
                )
            )
        )

    def _point_parameters(
        self, parameters: Mapping[str, str], scale: Fraction, precision: Any
    ) -> dict[str, str]:
        """The caller's parameters, validated as by the Python API and the
        live slice (``assemble.check_model_parameters``: the scale parameter,
        a fixed parameter at another value and a name that is not an external
        parameter of the model, the record's or ``NativeProcess.executor(...,
        externals=...)``, are refused), and the fixed
        parameters (the record's: every one fixed at generation); the scale
        parameter is formed natively from each point's mu_R^2
        (``batch_parameters``)."""

        from .assemble import check_model_parameters

        key = tuple(sorted(parameters.items()))
        if self._checked is None or self._checked[0] != key:
            fixed = self.request.fixed_parameters
            values = check_model_parameters(
                parameters,
                self.externals,
                self.scale_parameter,
                fixed,
                self.preparation_fixed,
            )
            values.update(fixed)
            self._checked = (key, values)
        return dict(self._checked[1])
