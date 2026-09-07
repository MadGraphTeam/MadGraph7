# SPDX-License-Identifier: 0BSD
import json

import pytest

from pyamplicol import ColorCorrelator, CorrelatorConfig, EmitGluon, SplitGluon
from pyamplicol.cli.parser import parse_cli


def test_declarations_json_roundtrip_and_automatic_born():
    chain = (EmitGluon(1, -1), SplitGluon(-1, -2, -3), EmitGluon(-2, -4))
    declarations = CorrelatorConfig(
        (ColorCorrelator.dipole("T12", 1, 2), ColorCorrelator("N3LO", chain, chain)),
        ((3, 1), (1,)),
    )
    encoded = json.loads(json.dumps(declarations.to_json_dict()))
    assert CorrelatorConfig.from_json_dict(encoded) == declarations
    assert declarations.spin_legs == (1, 3)
    assert declarations.spin_correlations == ((1,), (1, 3))
    assert [item.id for item in declarations.color_requests] == ["born", "T12", "N3LO"]


@pytest.mark.parametrize(
    "groups", (((),), ((0,),), ((True,),), ((1, 1),), ((1, 2), (2, 1)))
)
def test_invalid_spin_classes(groups):
    with pytest.raises(ValueError):
        CorrelatorConfig(spin_correlations=groups)


def test_duplicate_or_reserved_ids_rejected():
    with pytest.raises(ValueError, match="reserved"):
        CorrelatorConfig((ColorCorrelator("born"),))
    with pytest.raises(ValueError, match="unique"):
        CorrelatorConfig((ColorCorrelator("a"), ColorCorrelator("a")))


def test_unknown_json_fields_and_step_kinds_rejected():
    with pytest.raises(ValueError, match="unknown"):
        CorrelatorConfig.from_json_dict({"colour_correlations": []})
    with pytest.raises(ValueError, match="unknown"):
        CorrelatorConfig.from_json_dict(
            {"color_correlations": [{"id": "a", "bra": [{"kind": "unknown"}]}]}
        )


def test_cli_correlator_path_is_separate_from_default_generation_identity():
    plain = parse_cli(["generate", "g g > g g", "output"])
    selected = parse_cli(
        ["generate", "g g > g g", "output", "--correlators", "correlators.json"]
    )
    assert plain.correlators is None
    assert str(selected.correlators) == "correlators.json"
    assert dict(plain.dedicated) == dict(selected.dedicated)
    assert plain.resolve().effective == selected.resolve().effective
