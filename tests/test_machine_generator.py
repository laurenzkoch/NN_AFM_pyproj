import math

import pytest

from src.modules.simulation.generators.machine_generator import (
    AFMMachineGenerator,
)


def test_generator_returns_expected_number_of_geometries() -> None:
    generator = AFMMachineGenerator(
        num_base_designs=2,
        num_lam_designs=3,
    )

    geometries = list(generator.get_generator(_type="v2"))

    assert len(geometries) == 6

    # Every geometry must have its own dictionary.
    assert len({id(geometry) for geometry in geometries}) == 6


def test_generated_geometries_contain_no_operating_points() -> None:
    generator = AFMMachineGenerator(
        num_base_designs=1,
        num_lam_designs=2,
    )

    operating_parameters = {
        "ShaftSpeed",
        "ShaftSpeedNorm",
        "PhaseAdvance",
        "PeakCurrent",
        "RMSCurrentDensity",
        "RMSPhaseCurrent_D",
        "RMSPhaseCurrent_Q",
    }

    required_geometry_parameters = {
        "Airgap",
        "Stator_Bore",
        "Stator_Lam_Dia",
        "Pole_Number",
        "Slot_Number",
        "Magnet_Length",
    }

    for geometry in generator.get_generator(_type="v2"):
        assert required_geometry_parameters <= geometry.keys()
        assert operating_parameters.isdisjoint(geometry)

        assert all(
            math.isfinite(value)
            for value in geometry.values()
        )


def test_geometry_sampling_is_reproducible() -> None:
    first = AFMMachineGenerator(
        num_base_designs=2,
        num_lam_designs=3,
    )
    second = AFMMachineGenerator(
        num_base_designs=2,
        num_lam_designs=3,
    )

    assert list(
        first.get_generator(_type="v2")
    ) == list(
        second.get_generator(_type="v2")
    )


@pytest.mark.parametrize("count", [0, -1, True])
@pytest.mark.parametrize(
    "parameter",
    ["num_base_designs", "num_lam_designs"],
)
def test_invalid_design_counts_are_rejected(
    parameter: str,
    count: int,
) -> None:
    arguments = {
        "num_base_designs": 1,
        "num_lam_designs": 1,
    }
    arguments[parameter] = count

    with pytest.raises(ValueError, match=parameter):
        AFMMachineGenerator(**arguments)


@pytest.mark.parametrize("generator_type", ["v1", "unknown"])
def test_unsupported_generator_types_are_rejected(
    generator_type: str,
) -> None:
    generator = AFMMachineGenerator(
        num_base_designs=1,
        num_lam_designs=1,
    )

    with pytest.raises(ValueError):
        generator.get_generator(_type=generator_type)