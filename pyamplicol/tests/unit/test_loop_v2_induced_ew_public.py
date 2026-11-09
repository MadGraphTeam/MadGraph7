# SPDX-License-Identifier: 0BSD
"""The public route of an electroweak loop-induced process (M-LIT LIT-EW;
m-lit Q7, Q12): Generator -> artifact -> Runtime on a a > z z [QED] at the
model's default widths. The artifact records the width policy (fixed tree
widths, widthless loop propagators) and the generation-time probes, the
Goldstone-equivalence probe of its Z species included; the stored process
evaluates equal to MadLoop. A file of its own: the generation, its probe
slices and the artifact peak at about 8 GiB, which the CI job's 12 GiB
process-tree limit holds only without another file's slices.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

pytestmark = pytest.mark.loop_physics

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "loop" / "required" / "li"
LAX = 1e-5


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _point(fixture: dict, index: int = 0):
    return fixture["points"][index], Fraction(fixture["mu_r"][index]) ** 2


@pytest.fixture(scope="module")
def gmu(loop_gmu_ufo, tmp_path_factory):
    from pyamplicol.models.loading import compile_model_source

    return compile_model_source(
        loop_gmu_ufo,
        require_supported=False,
        cache_dir=tmp_path_factory.mktemp("li-ew-public") / "model-cache",
    )


def _card(output: Path, ufo: Path, entries, fixed: dict[str, str]) -> str:
    listed = ", ".join(
        f"{{expression = {json.dumps(expression)}, name = {json.dumps(name)}}}"
        for name, expression in entries
    )
    parameters = ", ".join(f"{key} = {json.dumps(v)}" for key, v in fixed.items())
    return f"""
action = "generate"
[model]
source = {json.dumps(str(ufo))}
[process]
entries = [{listed}]
loop_coupling_orders = ["QED"]
[loop]
r2_strategy = "model"
external_photons = "none"
[loop.preparation]
fixed_parameters = {{{parameters}}}
compatibility = "gmu_particle_anti"
[generation]
output = {json.dumps(str(output))}
emit_api_bundle = false
loop_induced_probes = true
[generation.validation]
post_build_validation = false
"""


def test_public_route_records_the_width_policy_and_the_goldstone_probe(
    gmu, loop_gmu_ufo, tmp_path
):
    """Generator -> artifact -> Runtime: a a > z z [QED] at the model's
    default widths (m-lit Q7, Q12). The artifact records the fixed-tree-width
    policy (the s-channel Higgs of h h -> Z Z blocks propagates with W_H; loop
    propagators widthless) and the generation-time probes (enabled here,
    ``generation.loop_induced_probes``), run at zero widths in binary64 with
    the double-double fallback: finiteness, the Ward probe of the photon
    species and the Goldstone-equivalence probe of the Z species (eta =
    -i), all within ten times their estimates, the probe slice's and the
    Goldstone slice's generation timed. The stored process evaluates at the
    references' zero widths equal to MadLoop."""

    from pyamplicol import Runtime
    from pyamplicol.api import Generator, ProcessRequest, ProcessSet
    from pyamplicol.config import resolve_config
    from pyamplicol.loops.v2.ew_policy import FIXED_TREE_WIDTHS

    name = "li_aa_zz"
    fixture = _fixture(name)
    fixed = {
        key: value
        for key, value in fixture["scheme_data"]["fixed_parameters"].items()
        if key in gmu.parameter_defaults and not key.startswith("W")
    }
    entries = (("aazz", "a a > z z"),)
    card = tmp_path / "card.toml"
    output = tmp_path / "artifact"
    card.write_text(_card(output, loop_gmu_ufo, entries, fixed))
    Generator(resolve_config(card)).generate(
        ProcessSet(tuple(ProcessRequest.parse(e, name=n) for n, e in entries)), output
    )
    manifest = json.loads((output / "artifact.json").read_text())
    (filters,) = (
        record["filters"]
        for record in manifest["extensions"]["generation"]["concrete_processes"]
        if record["id"] == "aazz"
    )
    policy = filters["conventions"]["width_policy"]
    assert policy["policy"] == FIXED_TREE_WIDTHS
    assert policy["loop_propagators"] == "widthless"
    assert policy["tree_widths"] == {"H": "WH"}
    probes = filters["loop_induced_probes"]
    assert probes["status"] == "run" and probes["carriers"][0] == "double"
    assert probes["widths"] == "zero" and "failed" not in probes
    assert set(probes["goldstone"]["carriers"]) == {"3"}
    assert set(probes["seconds"]["goldstone_slices"]) == {"3"}
    assert {"WZ", "WW", "WT", "WH"} <= set(probes["zeroed_widths"])
    assert probes["finiteness"]["passed"] and probes["ward"]["passed"]
    assert probes["ward"]["legs"] == [1]
    goldstone = probes["goldstone"]
    assert goldstone["legs"] == [3] and goldstone["passed"]
    re, im = goldstone["phases"]["3"]
    assert abs(re) <= 1e-9 and abs(im + 1) <= 1e-9
    point, scale = _point(fixture)
    zero = dict.fromkeys(("WZ", "WW", "WT", "WH"), "0")
    parameters = {
        key: value
        for key, value in fixture["scheme_data"]["parameters"].items()
        if key in gmu.parameter_defaults and key not in fixed
    }
    runtime = Runtime.load(
        output, process="aazz", model_parameters={**parameters, **zero}
    )
    value = runtime.evaluate([point], mu_r2=[str(scale)])
    assert value.contribution == "loop-induced"
    madloop = fixture["rows"][0]["providers"]["madloop"]["value"]
    assert abs(value.values[0][0][0] / madloop - 1) <= LAX
