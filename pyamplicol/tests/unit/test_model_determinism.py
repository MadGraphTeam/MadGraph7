# SPDX-License-Identifier: 0BSD
"""Compiled models do not depend on the process history.

Symbolica orders sums, products and symmetric arguments by the packed bytes
of symbol ids, i.e. by everything the interpreter created before; the model
compiler therefore puts every record into a canonical order
(``_internal.physics.canonical_order``). Two fresh interpreters compile the
bundled SM UFO, one cold and one after a history that perturbed every
mechanism on the uncanonicalised compiler (another model compiled first,
65,300 unrelated symbols, the index symbols created in reverse order: the
colour and Lorentz records with their index bindings and colour proofs, the
kernels' Horner forms and the loader's binary64 coupling values all
differed); the cache files must be byte-identical (they hold no wall-clock
durations).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_ASSETS = ROOT / "src" / "pyamplicol" / "assets" / "models" / "ufo"

# argv: model, cache directory, history ("cold" or "perturbed"), and the
# scratch directory of the perturbing compile. Prints the plain normal form
# of a probe built from the compiler's index symbols.
_COMPILE = """
import sys
from pathlib import Path

model, cache, history, scratch = sys.argv[1:5]
from pyamplicol.models.loading import compile_model_source

if history == "perturbed":
    from symbolica import S

    compile_model_source(Path(sys.argv[5]), cache_dir=Path(scratch))
    for k in range(65300):
        S(f"history::unrelated{k}")
    for k in range(12, 0, -1):
        for family in ("c", "l_1", "s"):
            S(f"spynso3::ufo_{family}_{k}")
from symbolica import E

probe = E(
    "spenso::f(spenso::coad(8,spynso3::ufo_c_1),spenso::coad(8,spynso3::ufo_c_2),"
    "spenso::coad(8,spynso3::ufo_c_3))+spynso3::ufo_l_1_1+spynso3::ufo_l_1_2"
)
compile_model_source(Path(model), cache_dir=Path(cache))
print(probe.format_plain())
"""


def test_compiled_model_does_not_depend_on_the_process_history(tmp_path):
    runs = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                _COMPILE,
                str(_ASSETS / "sm"),
                str(tmp_path / history),
                history,
                str(tmp_path / "scratch"),
                str(_ASSETS / "scalars"),
            ],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for history in ("cold", "perturbed")
    ]
    probes = []
    for run in runs:
        out, err = run.communicate(timeout=600)
        assert run.returncode == 0, err[-4000:]
        probes.append(out.strip().splitlines()[-1])
    # The history reorders Symbolica's normal forms of the compiler's symbols.
    assert probes[0] != probes[1], probes
    cold, perturbed = (
        sorted((tmp_path / history).glob("*.pyAmplicol-model.json"))
        for history in ("cold", "perturbed")
    )
    assert [path.name for path in cold] == [path.name for path in perturbed]
    assert len(cold) == 1
    assert cold[0].read_bytes() == perturbed[0].read_bytes()
    # No wall-clock durations, which no two compiles share.
    assert (
        not {"conversion_seconds", "phase_timings"}
        & json.loads(cold[0].read_bytes()).keys()
    )


# One summand of scalar_gravity's three-graviton vertex (its Lorentz record,
# with two spin-2 inputs and the three leg momenta). Prints the hashes of the
# components of Spenso's own network and of ``_execute_dense_network``.
_NETWORK = """
import hashlib
import sys

from symbolica import S

# The index symbols in increasing or decreasing order.
for k in range(1, 13) if sys.argv[1] == "forward" else range(12, 0, -1):
    for family in ("c", "l_1", "s", "l_2"):
        S(f"spynso3::ufo_{family}_{k}")
from pyamplicol._internal.physics.symbols import ModelSymbolRegistry
from pyamplicol.models import compiler_kernels as ck
from pyamplicol.models import compiler_symbolica as sym

sym._ensure_symbolica()
names = ModelSymbolRegistry("determinism-network")
p = [sym.TensorName(names.ufo_momentum_tensor_name(leg)) for leg in (1, 2, 3)]
P = [f"{names.ufo_momentum_tensor_name(leg)}(spenso::mink(4,spynso3::ufo_{{}}))"
     for leg in (1, 2, 3)]
G = "spenso::g(spenso::mink(4,spynso3::ufo_{}),spenso::mink(4,spynso3::ufo_{}))"
source = (
    f"-(({P[0]}*{P[1]}+{P[1]}*{P[2]})*{G}"
    f"+({P[0]}*{P[1]}+{P[1]}*{P[2]})*{G}"
    f"-2*{G}*{P[0]}*{P[1]}-2*{G}*{P[0]}*{P[1]}"
    f"+{G}*{P[0]}*{P[1]}+{G}*{P[0]}*{P[2]}"
    f"+{G}*{P[0]}*{P[1]}+{G}*{P[0]}*{P[2]})*{G}"
).format(
    "l_2_3", "l_2_1", "l_2_3", "l_2_1", "l_1_2", "l_1_3",
    "l_1_3", "l_2_1", "l_1_3", "l_2_1", "l_1_2", "l_2_3",
    "l_1_2", "l_2_1", "l_2_3", "l_1_3",
    "l_1_2", "l_2_1", "l_1_3", "l_2_3",
    "l_2_3", "l_2_1", "l_1_2", "l_1_3",
    "l_2_3", "l_2_1", "l_1_3", "l_1_2",
    "l_1_3", "l_2_1", "l_1_2", "l_2_3",
    "l_1_3", "l_2_1", "l_2_3", "l_1_2",
    "l_1_1", "l_2_2",
)
for mode in ("spenso", "canonical"):
    library = (
        ck.DenseTensors() if mode == "canonical" else sym.TensorLibrary.hep_lib_atom()
    )
    expression = sym.E(source)
    for side, leg in (("left", 1), ("right", 2)):
        components = [names.kernel_component(0, side, i) for i in range(16)]
        expression = expression * ck._input_tensor_expression(
            library, kind=0, side=side, spin=5, leg=leg,
            components=components, model_symbols=names,
        )
    left = [names.kernel_momentum(0, "left", i) for i in range(4)]
    right = [names.kernel_momentum(0, "right", i) for i in range(4)]
    momenta = (left, right, [-(a + b) for a, b in zip(left, right)])
    for leg, momentum in enumerate(momenta, start=1):
        ck.register_dense(
            library, names.ufo_momentum_tensor_name(leg),
            (sym.Representation.mink(4),), momentum,
        )
    labels = ck._spin_axis_labels(5, 3)
    if mode == "canonical":
        values = ck._execute_dense_network(expression, library, labels).values
    else:
        network = sym.TensorNetwork(sym.as_tensor(expression), library)
        network.execute(library=library)
        values = ck._ordered_dense_tensor_components(
            network.result_tensor(library), labels
        ).values
    text = "|".join(ck._as_expression(v).to_canonical_string() for v in values)
    print(mode, hashlib.sha256(text.encode()).hexdigest())
"""


def test_tensor_networks_do_not_depend_on_the_process_history():
    """Spenso contracts a network in an order that follows the symbol ids:
    the index symbols created in increasing and in decreasing order give
    components of different (equal) structure for a three-graviton summand
    (Spenso's own network); ``_execute_dense_network`` gives the same
    components in both histories."""

    runs = [
        subprocess.Popen(
            [sys.executable, "-c", _NETWORK, history],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for history in ("forward", "reverse")
    ]
    hashes = []
    for run in runs:
        out, err = run.communicate(timeout=600)
        assert run.returncode == 0, err[-4000:]
        hashes.append(dict(line.split() for line in out.strip().splitlines()[-2:]))
    forward, reverse = hashes
    assert forward["spenso"] != reverse["spenso"], hashes
    assert forward["canonical"] == reverse["canonical"], hashes


def test_canonical_forms_round_trip_and_ignore_symbol_order():
    """``canonical_plain`` parses back to its expression and is the same
    string for an expression built on symbols created in another order;
    ``canonical_apply`` keeps the value; a binary64 coefficient is refused
    rather than rounded by the canonical string."""

    from symbolica import E, S

    from pyamplicol._internal.physics.canonical_order import (
        canonical_apply,
        canonical_plain,
    )

    for name in ("pyamplicol_test_canonical::b", "pyamplicol_test_canonical::a"):
        S(name)
    source = (
        "-spenso::f(spenso::coad(8,pyamplicol_test_canonical::b),"
        "spenso::coad(8,pyamplicol_test_canonical::a),spenso::coad(8,pyamplicol_test_canonical::c))"
        "*(pyamplicol_test_canonical::a+2*pyamplicol_test_canonical::k10)"
        "+pyamplicol_test_canonical::k2^2/3-𝑖*conj(pyamplicol_test_canonical::c)"  # noqa: RUF001
    )
    expression = E(source)
    plain = canonical_plain(expression)
    assert E(plain) == expression
    assert "pyamplicol_canonical" not in plain
    horner = canonical_apply(lambda e: e.expand().collect_horner(), expression)
    assert (horner - expression).expand() == E("0")
    with pytest.raises(ValueError, match="binary64"):
        canonical_apply(lambda e: e, E("0.3*pyamplicol_test_canonical::a"))


def test_misaligned_canonical_blocks_are_refused():
    """A block of canonical symbols that does not start at a multiple of 2^16
    ids would order its symbols by their absolute ids: refused, for a block
    off by one id and by one byte."""

    from symbolica import Expression, S

    from pyamplicol._internal.physics.canonical_order import _check_block

    for shift in (1, 256):
        padding = -len(Expression.get_all_symbol_names()) % (1 << 16) + shift
        for k in range(padding):
            S(f"pyamplicol_test_misaligned::pad{shift}_{k}")
        block = [S(f"pyamplicol_test_misaligned::c{shift}_{k}") for k in range(1 << 16)]
        with pytest.raises(RuntimeError, match="not ordered by their block offset"):
            _check_block(block)
