# SPDX-License-Identifier: 0BSD
from __future__ import annotations

from decimal import Decimal

import pytest

from pyamplicol.api.errors import ArtifactError, CompatibilityError
from pyamplicol.generation.artifact_writer import _execution_plan
from pyamplicol.generation.dag_compiler import compile_generic_dag
from pyamplicol.generation.runtime_schema import build_runtime_schema
from pyamplicol.models import BuiltinSMModel
from pyamplicol.models.base import Model, Particle
from pyamplicol.models.builtin.process_ir import build_process_ir
from pyamplicol.models.contracts import CompiledParticleRecord
from pyamplicol.models.external_catalog import ExternalModelCatalogMixin
from pyamplicol.runtime.symbolica_exact import (
    _antiquark_weyl,
    _particle_mass,
    _quark_weyl,
    _source_wavefunction,
)


class _ExternalCatalog(ExternalModelCatalogMixin, Model):
    pass


def _compiled_particle(
    name: str,
    antiname: str,
    pdg: int,
    *,
    spin: int = 2,
) -> CompiledParticleRecord:
    return CompiledParticleRecord(
        name=name,
        antiname=antiname,
        pdg_code=pdg,
        spin=spin,
        color=1,
        mass="ZERO",
        width="ZERO",
        charge=0.0,
        quantum_numbers=(("electric_charge", "0"),),
        ghost_number=0,
        propagating=True,
        goldstoneboson=False,
        propagator=None,
    )


def _external_catalog(*records: CompiledParticleRecord) -> _ExternalCatalog:
    by_name = {record.name: record for record in records}
    model = _ExternalCatalog(name="relabelled-source-model")
    model._particle_records_by_pdg = {record.pdg_code: record for record in records}
    model.particles = {
        record.pdg_code: Particle(
            pdg=record.pdg_code,
            anti_pdg=by_name[record.antiname].pdg_code,
            spin=record.spin,
            dimension=4,
            color_rep=record.color,
        )
        for record in records
    }
    return model


def _exact_source(
    *,
    particle_id: int,
    anti_particle_id: int,
    source_orientation: str,
    dimension: int = 2,
) -> dict[str, object]:
    return {
        "leg_label": 1,
        "crossing": "identity",
        "particle_id": particle_id,
        "anti_particle_id": anti_particle_id,
        "wavefunction_kind": "fermion",
        "source_orientation": source_orientation,
        "source_helicity": 1,
        "chirality": 1,
        "dimension": dimension,
    }


def test_external_relabelled_pdg_source_metadata() -> None:
    particle = _compiled_particle("chi", "chi_bar", 810_001)
    antiparticle = _compiled_particle("chi_bar", "chi", -810_001)
    model = _external_catalog(particle, antiparticle)

    assert model.source_wavefunction_kind(810_001) == "fermion"
    assert model.source_orientation(810_001) == "particle"
    assert model.source_orientation(-810_001) == "antiparticle"


def test_external_self_conjugate_fermion_source_fails_clearly() -> None:
    majorana = _compiled_particle("neutral_fermion", "neutral_fermion", 810_003)
    model = _external_catalog(majorana)

    with pytest.raises(ValueError, match="unsupported self-conjugate fermion source"):
        model.source_orientation(810_003)


def test_schema_v3_projects_source_relation() -> None:
    model = BuiltinSMModel()
    schema = build_runtime_schema(
        compile_generic_dag(build_process_ir("d d~ > z"), model=model),
        model,
        process_id="source_orientation",
    )
    plan = _execution_plan(schema)

    sources = plan["source_fill"]["sources"]
    by_particle = {source["particle_id"]: source for source in sources}
    assert by_particle[-1]["anti_particle_id"] == 1
    assert by_particle[-1]["source_orientation"] == "antiparticle"
    assert by_particle[1]["anti_particle_id"] == -1
    assert by_particle[1]["source_orientation"] == "particle"
    assert by_particle[23]["anti_particle_id"] == 23
    assert by_particle[23]["source_orientation"] == "self-conjugate"


def test_schema_v3_projects_antiparticle_mass_metadata() -> None:
    model = BuiltinSMModel()
    schema = build_runtime_schema(
        compile_generic_dag(build_process_ir("d u~ > w-"), model=model),
        model,
        process_id="antiparticle_mass",
    )
    plan = _execution_plan(schema)

    source = next(
        item for item in plan["source_fill"]["sources"] if item["particle_id"] == -24
    )
    mass_record = next(item for item in plan["model"]["particles"] if item["pdg"] == 24)

    assert source["anti_particle_id"] == 24
    assert source["source_orientation"] == "antiparticle"
    assert mass_record["mass"] == pytest.approx(80.41900244575616)


@pytest.mark.parametrize(
    ("particle_id", "source_orientation", "expected"),
    (
        (810_001, "antiparticle", _antiquark_weyl),
        (-810_001, "particle", _quark_weyl),
    ),
)
def test_exact_sources_use_orientation_instead_of_pdg_sign(
    particle_id: int,
    source_orientation: str,
    expected: object,
) -> None:
    point = ((Decimal(5), Decimal(3), Decimal(4), Decimal(0)),)
    source = _exact_source(
        particle_id=particle_id,
        anti_particle_id=-particle_id,
        source_orientation=source_orientation,
    )
    schema = {"model": {"particles": []}, "model_parameters": []}

    wave = _source_wavefunction(source, point, schema, ())

    assert callable(expected)
    assert wave == expected(point[0], 1, 1)


def test_exact_source_mass_uses_explicit_antiparticle_relation() -> None:
    schema = {
        "model": {
            "particles": [
                {"pdg": -810_001, "mass": "99"},
                {"pdg": 910_002, "mass_parameter": "MX", "mass": "5"},
            ]
        },
        "model_parameters": [{"name": "MX", "parameter_index": 0}],
    }

    mass = _particle_mass(schema, 810_001, 910_002, (Decimal(7),))

    assert mass == Decimal(7)


def test_exact_self_conjugate_fermion_source_fails_clearly() -> None:
    source = _exact_source(
        particle_id=810_003,
        anti_particle_id=810_003,
        source_orientation="self-conjugate",
    )
    point = ((Decimal(5), Decimal(3), Decimal(4), Decimal(0)),)
    schema = {"model": {"particles": []}, "model_parameters": []}

    with pytest.raises(CompatibilityError, match="self-conjugate fermion source"):
        _source_wavefunction(source, point, schema, ())


def test_exact_source_rejects_inconsistent_orientation_relation() -> None:
    source = _exact_source(
        particle_id=810_003,
        anti_particle_id=810_004,
        source_orientation="self-conjugate",
    )
    point = ((Decimal(5), Decimal(3), Decimal(4), Decimal(0)),)
    schema = {"model": {"particles": []}, "model_parameters": []}

    with pytest.raises(ArtifactError, match="inconsistent with its antiparticle"):
        _source_wavefunction(source, point, schema, ())
