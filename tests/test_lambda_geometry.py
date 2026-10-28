import numpy as np
import pytest

from src.modules.geometry.lambda_geometry import (
    LambdaValues,
    geometry_from_lambdas,
    lambdas_from_geometry,
)


@pytest.mark.parametrize("csv_rounding", [False, True])
@pytest.mark.parametrize(
    ("airgap", "stator_bore", "slots", "poles"),
    [
        (1.0, 70.0, 18, 12),
        (2.5, 95.0, 24, 16),
    ],
)
def test_lambda_geometry_roundtrip(
    airgap: float,
    stator_bore: float,
    slots: int,
    poles: int,
    csv_rounding: bool,
) -> None:
    original = LambdaValues(
        lambda_1_diameter_ratio=0.5,
        lambda_2_magnet_arc_ratio=0.75,
        lambda_3_airgap_magnet_length_ratio=0.4,
        lambda_4_slot_width_pitch_ratio=0.5,
        lambda_5_slot_width_depth_ratio=0.3,
        lambda_6_magnet_radial_ratio=0.7,
        lambda_7_embed_remaining_width_ratio=0.2,
        lambda_8_stator_back_iron_pitch_ratio=0.25,
        lambda_9_magnet_rotor_length_ratio=0.5,
        lambda_10_slot_opening_width_ratio=0.6,
        lambda_11_tooth_tip_depth_ratio=0.1,
        lambda_12_slot_corner_radius_ratio=0.4,
    )

    geometry = geometry_from_lambdas(
        original,
        airgap=airgap,
        stator_bore=stator_bore,
        slot_number=slots,
        pole_number=poles,
    )

    # Independently check the previously inconsistent radius definition.
    assert geometry["Slot_Corner_Radius"] == pytest.approx(
        0.4 * geometry["Slot_Width"] / 2
    )
    if csv_rounding:
        geometry = {name: float(f"{value:.4f}") for name, value in geometry.items()}
    restored = lambdas_from_geometry(geometry)

    np.testing.assert_allclose(
        restored.to_array(),
        original.to_array(),
        rtol=0.0,
        atol=1e-4 if csv_rounding else 1e-12,
    )

    assert geometry["Airgap"] == airgap
    assert geometry["Stator_Bore"] == stator_bore
