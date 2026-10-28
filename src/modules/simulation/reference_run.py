"""Results and calculations for an AFM open-circuit reference run."""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ReferenceRunResult:
    """Validated results from an open-circuit reference simulation.

    reference_speed_rpm:
        Mechanical shaft speed in revolutions per minute.
    back_emf_constant_ll_peak_vs_per_rad:
        Fundamental line-to-line back-EMF peak divided by
        mechanical angular speed, in V*s/rad.
    peak_back_emf_phase_v:
        Motor-CAD's PeakBackEMFPhase result, in volts.
        Stored separately from the fundamental back-EMF constant.
    """

    reference_speed_rpm: float
    back_emf_constant_ll_peak_vs_per_rad: float
    peak_back_emf_phase_v: float

    def __post_init__(self) -> None:
        values = {
            "reference_speed_rpm": self.reference_speed_rpm,
            "back_emf_constant_ll_peak_vs_per_rad": (
                self.back_emf_constant_ll_peak_vs_per_rad
            ),
            "peak_back_emf_phase_v": self.peak_back_emf_phase_v,
        }

        for name, value in values.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(
                    f"{name} must be finite and positive; got {value!r}."
                )

    def calculate_base_speed_rpm(self, dc_bus_voltage_v: float) -> float:
        """Return the no-load voltage-limit speed for ideal linear SVPWM."""
        if not math.isfinite(dc_bus_voltage_v) or dc_bus_voltage_v <= 0:
            raise ValueError(
                "dc_bus_voltage_v must be finite and positive; "
                f"got {dc_bus_voltage_v!r}."
            )

        speed_rpm = (
            dc_bus_voltage_v
            / self.back_emf_constant_ll_peak_vs_per_rad
            * 60.0
            / math.tau
        )

        if not math.isfinite(speed_rpm) or speed_rpm <= 0:
            raise ValueError(
                "Calculated base speed must be finite and positive; "
                f"got {speed_rpm!r}."
            )

        return speed_rpm