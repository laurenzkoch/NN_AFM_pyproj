from __future__ import annotations

import logging
from time import perf_counter
from typing import Literal, overload
from uuid import uuid4

import numpy as np

from src.modules.domain.reference_run import ReferenceRunResult
from src.modules.process.motor_cad_processes import mcad_instance_handling
from src.modules.process.motor_cad_processes.mcad_processing import MCADProcessor

logger = logging.getLogger(__name__)


def _set_mcad_parameters(
    mcad: MCADProcessor,
    parameters: dict[str, float],
) -> None:
    for name, value in parameters.items():
        if isinstance(value, np.generic):
            value = value.item()

        if name == "Pole_Number":
            points_per_cycle = mcad.get_value_from_mcad("TorquePointsPerCycle")
            mesh_points = value / 2 * points_per_cycle

            if mesh_points < 360:
                mesh_points *= 4

            mcad.set_value_in_mcad(
                "AirgapMeshPoints_mesh",
                mesh_points,
            )
            mcad.set_value_in_mcad(
                "AirgapMeshPoints_layers",
                mesh_points,
            )

        mcad.set_value_in_mcad(name, value)


@overload
def execute_simulation(
    parameter_dict: dict[str, float],
    true_for_reference_run: Literal[False] = False,
    *,
    log_context: str = "",
) -> tuple[list[float] | None, bool]: ...


@overload
def execute_simulation(
    parameter_dict: dict[str, float],
    true_for_reference_run: Literal[True],
    *,
    log_context: str = "",
) -> ReferenceRunResult | None: ...


def execute_simulation(
    parameter_dict: dict[str, float],
    true_for_reference_run: bool = False,
    *,
    log_context: str = "",
) -> tuple[list[float] | None, bool] | ReferenceRunResult | None:
    context = log_context or f"task={uuid4().hex[:12]}"
    kind = "reference" if true_for_reference_run else "load"
    phase = "instance_lookup"
    started = perf_counter()

    logger.info(
        "Worker task started: %s kind=%s",
        context,
        kind,
    )
    logger.debug(
        "Worker task parameters: %s parameters=%r",
        context,
        parameter_dict,
    )

    try:
        mcad = mcad_instance_handling.get_mcad_instance()

        phase = "parameter_assignment"
        _set_mcad_parameters(mcad, parameter_dict)

        phase = f"{kind}_calculation"

        result: tuple[list[float] | None, bool] | ReferenceRunResult | None

        if true_for_reference_run:
            reference_result = mcad.execute_reference_run()
            outcome = "rejected" if reference_result is None else "completed"
            result = reference_result
        else:
            load_result = mcad.execute_mcad_simulation()
            outcome = "completed" if load_result[1] else "rejected"
            result = load_result

        logger.info(
            "Worker task finished: %s kind=%s outcome=%s elapsed_s=%.3f",
            context,
            kind,
            outcome,
            perf_counter() - started,
        )

        return result

    except BaseException as exc:
        logger.error(
            "Worker task failed: "
            "%s kind=%s phase=%s elapsed_s=%.3f "
            "error_type=%s detail=%s",
            context,
            kind,
            phase,
            perf_counter() - started,
            type(exc).__name__,
            exc,
        )

        try:
            mcad_instance_handling.shutdown_mcad_instance()
        except (
            OSError,
            RuntimeError,
        ) as cleanup_error:
            logger.exception(
                "Unexpected worker cleanup error: %s detail=%s",
                context,
                cleanup_error,
            )

        raise


def execute_reference_simulation(
    parameter_dict: dict[str, float],
    *,
    log_context: str = "",
) -> ReferenceRunResult | None:
    return execute_simulation(
        parameter_dict,
        true_for_reference_run=True,
        log_context=log_context,
    )
