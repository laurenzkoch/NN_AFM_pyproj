
import numpy as np
import torch

from config.config_mcad_parameters import OUTPUT_NEURON_LIST, INPUT_NEURON_LIST
from modules.data_managment.csv_io import import_data_from_csv
from config.config_paths import *


def create_tensor(filename: str, num_machine_designs: int, num_points_per_dim: int) -> torch.Tensor:

    num_channels = len(OUTPUT_NEURON_LIST)

    data_df = import_data_from_csv(
        PATH_DATA_MCAD_RESULTS,
        filename=filename
    )
    machines = data_df.groupby(
        INPUT_NEURON_LIST,
        sort=False
    )
    if len(machines) != num_machine_designs:
        raise ValueError(
            f"Expected: {num_machine_designs} Machins, "
            f"found: {len(machines)}"
        )

    current_values = sorted(data_df["PeakCurrent"].unique())
    shaftspeed_values = sorted(data_df["ShaftSpeed"].unique())
    phaseadvance_values = sorted(data_df["PhaseAdvance"].unique())


    if len(current_values) != num_points_per_dim:
        raise ValueError(
            f"PeakCurrent: expected {num_points_per_dim}, "
            f"found {len(current_values)}"
        )

    if len(shaftspeed_values) != num_points_per_dim:
        raise ValueError(
            f"ShaftSpeed: expected {num_points_per_dim}, "
            f"found {len(shaftspeed_values)}"
        )

    if len(phaseadvance_values) != num_points_per_dim:
        raise ValueError(
            f"PhaseAdvance: expected {num_points_per_dim}, "
            f"found {len(phaseadvance_values)}"
        )


    current_idx = {
        value: idx
        for idx, value in enumerate(current_values)
    }

    shaftspeed_idx = {
        value: idx
        for idx, value in enumerate(shaftspeed_values)
    }

    phaseadvance_idx = {
        value: idx
        for idx, value in enumerate(phaseadvance_values)
    }


    X = np.full(
        (
            num_machine_designs,
            num_channels,
            num_points_per_dim,
            num_points_per_dim,
            num_points_per_dim
        ),
        np.nan,
        dtype=np.float32
    )


    for machine_idx, (_, machine_df) in enumerate(machines):
        for _, row in machine_df.iterrows():

            idx_current = current_idx[
                row["PeakCurrent"]
            ]

            idx_shaftspeed = shaftspeed_idx[
                row["ShaftSpeed"]
            ]

            idx_phaseadvance = phaseadvance_idx[
                row["PhaseAdvance"]
            ]

            for channel_idx, channel in enumerate(
                    OUTPUT_NEURON_LIST
            ):
                X[
                    machine_idx,
                    channel_idx,
                    idx_current,
                    idx_shaftspeed,
                    idx_phaseadvance
                ] = row[channel]

                print(f"{X[
                    machine_idx,
                    channel_idx,
                    idx_current,
                    idx_shaftspeed,
                    idx_phaseadvance]}")

    if np.isnan(X).any():
        raise ValueError(
            "At least, one map isn't complete"
        )

    return torch.tensor(
        X,
        dtype=torch.float32
    )
