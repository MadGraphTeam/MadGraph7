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
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from .families import Orders
from .native_trees import NativeTrees

SCHEMA = 1


def _orders(orders: Orders) -> list[list[Any]]:
    return [[name, power] for name, power in orders]


def _orders_of(record: Any) -> Orders:
    return tuple((str(name), int(power)) for name, power in record)


@dataclass(frozen=True)
class NativeProcess:
    """The native evaluation of one v2 process: plan, tree program and the
    assembly record (module docstring)."""

    plan_json: str
    trees: NativeTrees
    record: Mapping[str, Any]

    @classmethod
    def of(cls, executor: Any) -> NativeProcess:
        """The process of a slice's native executor
        (``NativeExecutor(slice_, inputs="native")``, sector columns)."""

        if executor.inputs != "native":
            raise ValueError("a native process needs an executor with native inputs")
        return cls.export(executor.slice, executor.exported, executor._plan_json)

    @classmethod
    def export(cls, slice_: Any, exported: Any, plan_json: str) -> NativeProcess:
        """The process of a plan exported with its tree program
        (``export_plan(slice_, trees=True, born_squared=True)``, sector
        columns); ``plan_json`` is ``exported.json()``."""

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
            "fixed_parameters": dict(request.fixed_parameters),
            "scale_parameter": slice_.scale_parameter,
            "alpha_s_parameter": slice_.alpha_s_parameter,
            "families": len(exported.families),
            "orders": [_orders(orders) for orders in exported.orders],
            "trees": {
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
            },
        }
        return cls(plan_json, trees, record)

    # -- storage -------------------------------------------------------------

    def members(self) -> dict[str, bytes]:
        """The process as named byte members: the assembly record and the
        tree program (JSON), the tree program's saved kernels and its
        parameter evaluator, and their binary64 SymJIT plane applications
        (the plan is stored on its own)."""

        trees = self.trees
        if len(trees.planes) != len(trees.kernels) + 1:
            raise ValueError("a native process stores a plane application per state")
        out = {
            "native-process.json": json.dumps(
                self.record, separators=(",", ":"), sort_keys=True
            ).encode(),
            "tree-program.json": trees.json().encode(),
            "parameters.bin": bytes(trees.parameters),
            "parameters.symjit": bytes(trees.planes[-1]),
        }
        for k, saved in enumerate(trees.kernels):
            out[f"kernel-{k}.bin"] = bytes(saved)
            out[f"kernel-{k}.symjit"] = bytes(trees.planes[k])
        return out

    @classmethod
    def from_members(
        cls, plan_json: str, members: Mapping[str, bytes]
    ) -> NativeProcess:
        record = json.loads(members["native-process.json"])
        if record.get("schema") != SCHEMA:
            raise ValueError(
                "unsupported native process schema; regenerate the artifact"
            )
        program = json.loads(members["tree-program.json"])
        kernels = []
        while f"kernel-{len(kernels)}.bin" in members:
            kernels.append(bytes(members[f"kernel-{len(kernels)}.bin"]))
        planes = [bytes(members[f"kernel-{k}.symjit"]) for k in range(len(kernels))]
        planes.append(bytes(members["parameters.symjit"]))
        meta = record["trees"]
        trees = NativeTrees(
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
            bytes(members["parameters.bin"]),
            tuple(
                (str(component), _orders_of(sector), int(power))
                for component, sector, power in meta["counterterm_outputs"]
            ),
            tuple(planes),
        )
        return cls(plan_json, trees, record)

    # -- evaluation ----------------------------------------------------------

    def executor(self, native: Any, **options: Any) -> Any:
        """The native executor of the process (``plan_export.NativeExecutor``
        with native inputs) on its structural view; ``native`` is the loaded
        loop native module, ``options`` the executor's (lanes, threads,
        forms)."""

        from .plan_export import ExportedPlan, NativeExecutor

        record = self.record
        exported = ExportedPlan(
            program={},
            rows=(),
            born_components=(),
            block_rows=(),
            constants=(),
            masses=(),
            orders=tuple(_orders_of(o) for o in record["orders"]),
            families=tuple(range(record["families"])),
            columns=tuple(_orders_of(sector) for sector, _pairs in record["sectors"]),
            trees=self.trees,
        )
        return NativeExecutor(
            _ProcessView(record, native),
            forward=False,
            exported=exported,
            plan_json=self.plan_json,
            inputs="native",
            **options,
        )


class _ProcessView:
    """The structural attributes of a ``LoopSlice`` the native executor and
    ``assemble._NativePoint`` read, from a native process record."""

    def __init__(self, record: Mapping[str, Any], native: Any):
        self.native = native
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
        self.scale_parameter = record["scale_parameter"]
        self.alpha_s_parameter = record["alpha_s_parameter"]
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
        """The caller's and the fixed parameters; the scale parameter is
        formed natively from each point's mu_R^2 (``batch_parameters``)."""

        values = dict(parameters)
        values.update(self.request.fixed_parameters)
        return values
