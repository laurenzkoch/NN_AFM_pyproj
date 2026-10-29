from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Self, TypeVar

import ansys.motorcad.core as pymotorcad

from src.config.config_mcad_parameters import (
    DEFAULT_VALUES,
    MCAD_RESULT_PARAMETERS,
    REFERENCE_SPEED_RPM,
)
from src.config.config_paths import PATH_MCAD_BASEFILE, PATH_MCAD_EXE
from src.modules.process.motor_cad_processes.mcad_errors import (
    MCADCalculationError,
    MCADCleanupError,
    MCADCommunicationError,
    MCADExecutionError,
    MCADInitializationError,
    MCADResultError,
)
from src.modules.simulation.reference_run import ReferenceRunResult

T = TypeVar("T")
logger = logging.getLogger(__name__)


def _design_to_string(
    parameter_list: list[str],
    value_list: list[float | int],
) -> str:
    return "\n".join(
        f"\t\t{var}: {value}" for var, value in zip(parameter_list, value_list)
    )


def test_boundary_conditions(parameter_list: list[str], values: list[float]) -> bool:
    keys = parameter_list

    if len(keys) != len(values):
        logger.error("The number of keys does not match the number of values.")
        raise ValueError("The number of keys does not match the number of values.")

    parameter_dict = {k: float(v) for k, v in zip(keys, values)}

    # radial - stator
    stator_lam_dia = parameter_dict["Stator_Lam_Dia"]
    stator_bore = parameter_dict["Stator_Bore"]
    slot_width = parameter_dict["Slot_Width"]

    # radial - rotor
    afm_d_rotor = parameter_dict["AFM_D_Rotor"]
    shaft_dia = parameter_dict["Shaft_Dia"]
    shaft_hole_diameter = parameter_dict["Shaft_Hole_Diameter"]
    magnet_embed_depth = parameter_dict["Magnet_Embed_Depth"]
    magnet_thickness = parameter_dict["Magnet_Thickness"]

    # axial
    motor_length = parameter_dict["Motor_Length"]
    stator_lam_length = parameter_dict["Stator_Lam_Length"]
    slot_depth = parameter_dict["Slot_Depth"]
    rotor_lam_length = parameter_dict["Rotor_Lam_Length"]
    magnet_length = parameter_dict["Magnet_Length"]

    # winding
    slot_opening = parameter_dict["Slot_Opening"]
    tooth_tip_depth = parameter_dict["Tooth_Tip_Depth"]
    slot_number = int(parameter_dict["Slot_Number"])
    pole_number = int(parameter_dict["Pole_Number"])
    airgap = parameter_dict["Airgap"]
    magphases = int(parameter_dict["MagPhases"])

    dia_mag_i = afm_d_rotor - 2 * magnet_embed_depth - 2 * magnet_thickness
    dia_mag_a = afm_d_rotor - 2 * magnet_embed_depth
    activ_length = min(stator_lam_dia, dia_mag_a) - max(stator_bore, dia_mag_i)
    stator_width_per_slot = math.pi * stator_bore / slot_number
    pole_pairs = int(pole_number / 2)
    winding_gcd = math.gcd(slot_number, pole_pairs)
    winding_ratio = slot_number / magphases / winding_gcd

    # operation points
    phaseadvanced = parameter_dict["PhaseAdvance"]

    tol = 1e-6  # mm

    # radial - stator
    if stator_bore >= stator_lam_dia:
        logger.warning(
            "Warning: Stator bore is greater than stator lamination diameter."
        )
        return False
    if stator_bore <= shaft_dia:
        logger.warning("Warning: Stator bore is less than shaft diameter.")
        return False

    # radial - rotor
    if dia_mag_i <= shaft_dia:
        logger.warning(
            "Warning: Magnets too thick for rotor diameter "
            "or rotor diameter is less than shaft diameter."
        )
        return False
    if shaft_hole_diameter >= stator_lam_dia:
        logger.warning(
            "Warning: Shaft hole diameter is greater than stator lamination diameter."
        )
        return False

    # axial
    if (
        motor_length
        < (2 * (stator_lam_length + airgap + magnet_length) + rotor_lam_length) - tol
    ):
        logger.warning("Warning: Motor length is too short.")
        return False
    if slot_depth >= stator_lam_length:
        logger.warning("Warning: Slot depth is greater than stator lamination length.")
        return False
    if rotor_lam_length < 0.25 or magnet_length < 0.25:
        logger.warning("Axial length of rotor/stator/magnets too short")
        return False

    # winding
    if not winding_ratio.is_integer() or winding_ratio < 1:
        logger.warning(
            "Winding is not feasible. "
            "Slot_Number / MagPhases / gcd(Pole_Number, Slot_Number) "
            "= %s / %s / %s = %.6f.",
            slot_number,
            magphases,
            winding_gcd,
            winding_ratio,
        )
        return False
    if slot_width >= stator_width_per_slot:
        logger.warning("Warning: Slot width is greater than stator width per slot.")
        return False
    if tooth_tip_depth >= slot_depth:
        logger.warning("Warning: Tooth tip depth is greater than stator tip depth.")
        return False
    if slot_opening > slot_width:
        logger.warning("Warning: Slot opening is greater than slot width.")
        return False

    # magnet length
    if activ_length <= 0:
        logger.warning("Warning: Magnet length is less than 0")
        return False

    # operation points
    if phaseadvanced < 0 or phaseadvanced >= 90:
        logger.warning(
            "Warning: Phase advance is out of bounds. "
            "It should be in the range [0, 90)."
        )
        return False

    return True


def _is_communication_error(error: BaseException) -> bool:
    """Recognize transport errors and the observed SDK error message."""
    seen: set[int] = set()
    current: BaseException | None = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))

        if isinstance(current, (ConnectionError, TimeoutError)):
            return True

        if "rpc communication failed" in str(current).casefold():
            return True

        current = current.__cause__ or current.__context__

    return False


class MCADProcessor:
    def __init__(self) -> None:
        self._close_attempted = False
        started = perf_counter()

        logger.info(
            "Creating Motor-CAD instance: executable=%s",
            PATH_MCAD_EXE,
        )

        try:
            pymotorcad.set_motorcad_exe(str(PATH_MCAD_EXE))

            self.mcad_process: pymotorcad.MotorCAD = pymotorcad.MotorCAD(
                open_new_instance=True,
                reuse_parallel_instances=False,
                keep_instance_open=False,
            )

        except (pymotorcad.MotorCADError, OSError) as exc:
            logger.error(
                "Motor-CAD instance creation failed: elapsed_s=%.3f detail=%s",
                perf_counter() - started,
                exc,
            )

            raise MCADInitializationError(
                f"Could not create Motor-CAD instance: {exc}"
            ) from exc

        logger.info(
            "Motor-CAD instance created: elapsed_s=%.3f",
            perf_counter() - started,
        )

    def __enter__(self) -> Self:
        try:
            self.start()
        except BaseException:
            try:
                self.close()
            except Exception:
                logger.exception(
                    "Motor-CAD cleanup failed after context startup error."
                )
            raise

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is None:
            self.close()
            return

        try:
            self.close()
        except Exception:
            logger.exception(
                "Motor-CAD cleanup failed while handling another exception."
            )

    def start(self) -> None:
        logger.info("Configuring Motor-CAD instance.")

        self.set_value_in_mcad("MessageDisplayState", 2)

        self._call_mcad(
            operation="set_visible(False)",
            call=lambda: self.mcad_process.set_visible(False),
            error_type=MCADInitializationError,
        )

        logger.info("Motor-CAD instance configured.")

    def close(self) -> None:
        if self._close_attempted:
            logger.debug("Skipping repeated Motor-CAD close attempt.")
            return

        self._close_attempted = True
        started = perf_counter()

        logger.info("Closing Motor-CAD instance.")

        try:
            self._call_mcad(
                operation="quit",
                call=self.mcad_process.quit,
                error_type=MCADCleanupError,
            )
        except MCADExecutionError as exc:
            raise MCADCleanupError(
                f"Could not close Motor-CAD instance: {exc}"
            ) from exc

        # Preserve the existing pause before the worker exits.
        time.sleep(2)

        logger.info(
            "Motor-CAD quit completed: elapsed_s=%.3f",
            perf_counter() - started,
        )

    def open_motorcad_file(
        self,
        file_path: str | Path = PATH_MCAD_BASEFILE,
    ) -> None:
        path = Path(file_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(f"Motor-CAD model file does not exist: {path}")

        started = perf_counter()

        logger.info(
            "Loading Motor-CAD model: path=%s",
            path,
        )

        self._call_mcad(
            operation=f"load_from_file(path={str(path)!r})",
            call=lambda: self.mcad_process.load_from_file(str(path)),
            error_type=MCADInitializationError,
        )

        logger.info(
            "Motor-CAD model loaded: path=%s elapsed_s=%.3f",
            path,
            perf_counter() - started,
        )

    def set_value_in_mcad(
        self,
        variable_name: str,
        value: float | bool,
    ) -> None:
        if not variable_name.strip():
            raise ValueError("variable_name must not be empty.")

        if not math.isfinite(value):
            raise ValueError(f"Value for {variable_name} must be finite.")

        logger.debug(
            "Setting MCAD variable: name=%s value=%r",
            variable_name,
            value,
        )

        self._call_mcad(
            operation=(f"set_variable(name={variable_name!r}, value={value!r})"),
            call=lambda: self.mcad_process.set_variable(
                variable_name,
                value,
            ),
        )

    def get_value_from_mcad(self, variable_name: str) -> float:
        if not variable_name.strip():
            raise ValueError("variable_name must not be empty.")

        raw_value = self._call_mcad(
            operation=f"get_variable(name={variable_name!r})",
            call=lambda: self.mcad_process.get_variable(variable_name),
        )

        if not isinstance(raw_value, (int, float, str)):
            raise MCADResultError(
                f"Expected a numeric value for {variable_name}; "
                f"received {type(raw_value).__name__}."
            )

        try:
            value = float(raw_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise MCADResultError(
                f"Cannot convert {variable_name} to float: {raw_value!r}."
            ) from exc

        if not math.isfinite(value):
            raise MCADResultError(
                f"Motor-CAD returned a nonfinite value for {variable_name}: {value!r}."
            )

        logger.debug(
            "Read MCAD variable: name=%s value=%r",
            variable_name,
            value,
        )

        return value

    def do_calculation(self) -> bool:
        started = perf_counter()

        logger.info("Magnetic calculation started.")

        self._call_mcad(
            operation="do_magnetic_calculation",
            call=self.mcad_process.do_magnetic_calculation,
            error_type=MCADCalculationError,
        )

        logger.info(
            "Magnetic calculation completed: elapsed_s=%.3f",
            perf_counter() - started,
        )

        return True

    def execute_mcad_simulation(self) -> tuple[list[float] | None, bool]:
        def get_values(var_list: list[str]) -> list[float]:
            return [self.get_value_from_mcad(var) for var in var_list]

        all_parameters: list[str] = list(DEFAULT_VALUES.keys())
        all_values: list[float] = get_values(all_parameters)

        if test_boundary_conditions(all_parameters, all_values):
            if self.do_calculation():
                value_row = get_values(MCAD_RESULT_PARAMETERS)
                return value_row, True

            else:
                logger.error(
                    "Calculation failed\n%s",
                    _design_to_string(all_parameters, all_values),
                )
                return None, False
        else:
            logger.error(
                "Failed boundary check\n%s",
                _design_to_string(all_parameters, all_values),
            )
            return None, False

    def execute_reference_run(self) -> ReferenceRunResult | None:
        """Run an open-circuit reference calculation for the current geometry.

        The constant variable must represent the fundamental line-to-line
        back-EMF peak divided by mechanical angular speed, in V*s/rad.

        Return None for rejected geometry or a failed magnetic calculation.
        Other errors propagate to the worker, which closes the MCAD instance.
        """

        temporary_values: dict[str, float | bool] = {
            "ShaftSpeed": REFERENCE_SPEED_RPM,
            "PeakCurrent": 0.0,
            "RMSCurrentDensity": 0.0,
            "PhaseAdvance": 0.0,
            "OpenCircuitCalculation": True,
            "BackEMFCalculation": True,
        }

        previous_values = {}

        for name in temporary_values:
            previous_values[name] = self._call_mcad(
                operation=f"read_reference_setting(name={name!r})",
                call=lambda name=name: self.mcad_process.get_variable(name),
            )

        for name, value in temporary_values.items():
            self.set_value_in_mcad(name, value)

        parameter_names = list(DEFAULT_VALUES)
        parameter_values = [self.get_value_from_mcad(name) for name in parameter_names]

        result: ReferenceRunResult | None = None

        geometry_is_valid = test_boundary_conditions(
            parameter_names,
            parameter_values,
        )

        if not geometry_is_valid:
            logger.warning("Reference run rejected by boundary checks.")
        elif not self.do_calculation():
            logger.warning("Reference magnetic calculation failed.")
        else:
            result = ReferenceRunResult(
                reference_speed_rpm=self.get_value_from_mcad("ShaftSpeed"),
                back_emf_constant_ll_peak_vs_per_rad=(
                    self.get_value_from_mcad("Ke_Fundamental")
                ),
                peak_back_emf_phase_v=(self.get_value_from_mcad("PeakBackEMFPhase")),
            )

        for name, value in previous_values.items():
            self._call_mcad(
                operation=f"restore_reference_setting(name={name!r})",
                call=lambda name=name, value=value: self.mcad_process.set_variable(
                    name, value
                ),
            )

        return result

    def _call_mcad(
        self,
        operation: str,
        call: Callable[[], T],
        *,
        error_type: type[MCADExecutionError] = MCADExecutionError,
    ) -> T:
        started = perf_counter()

        logger.debug(
            "MCAD operation started: operation=%s",
            operation,
        )

        try:
            result = call()

        except (pymotorcad.MotorCADError, OSError) as exc:
            exception_type = (
                MCADCommunicationError if _is_communication_error(exc) else error_type
            )

            logger.error(
                "MCAD operation failed: "
                "operation=%s elapsed_s=%.3f "
                "error_type=%s detail=%s",
                operation,
                perf_counter() - started,
                type(exc).__name__,
                exc,
            )

            raise exception_type(f"{operation} failed: {exc}") from exc

        logger.debug(
            "MCAD operation completed: operation=%s elapsed_s=%.3f",
            operation,
            perf_counter() - started,
        )

        return result
