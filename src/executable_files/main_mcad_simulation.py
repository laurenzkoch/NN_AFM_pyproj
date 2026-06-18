from src.modules import data_generating
import pandas as pd

#list with all output parameters (str)
output_parameters = [
    "ShaftTorque",
    "SystemEfficiency",
    "ConductorLoss",
    "MagnetLoss",
    "StatorIronLoss_Total",
    "RotorIronLoss_Total",
    "WedgeLoss",
]

#data sets for the input parameters
input_parameters = [
    {"name": "PeakCurrent", "min_value": 1, "max_value": 100, "step_size": 2},
    {"name": "PhaseAdvance", "min_value": 0, "max_value": 180, "step_size": 3},
    {"name": "Airgap", "min_value": 0.1, "max_value": 5, "step_size": 0.3},
    {"name": "AFM_D_Rotor", "min_value": 10, "max_value": 500, "step_size": 10},
    {"name": "Stator_Lam_Dia", "min_value": 10, "max_value": 500, "step_size": 10},
    {"name": "Stator_Bore", "min_value": 10, "max_value": 500, "step_size": 10},
    {"name": "Pole_Number", "min_value": 2, "max_value": 20, "step_size": 1},
    {"name": "CircumferentialSegments", "min_value": 1, "max_value": 10, "step_size": 1},
    {"name": "Magnet_Length", "min_value": 1, "max_value": 50, "step_size": 2},
    {"name": "Magnet_Thickness", "min_value": 10, "max_value": 500, "step_size": 5},
    {"name": "Slot_Number", "min_value": 2, "max_value": 40, "step_size": 1},
    {"name": "Slot_Width", "min_value": 1, "max_value": 20, "step_size": 1},
    {"name": "Slot_Depth", "min_value": 1, "max_value": 50, "step_size": 1},
    {"name": "Stator_Lam_Length", "min_value": 10, "max_value": 500, "step_size": 10},
    {"name": "Motor_Length", "min_value": 10, "max_value": 800, "step_size": 50},
]


with data_generating.MCADDataGenerator() as data_generator:
    data_generator.take_parameter(
        pd.DataFrame(
            input_parameters,
            columns=["name", "min_value", "max_value", "step_size"]
        ),
        output_parameters=output_parameters
    )
    data_generator.single_parameter_variation()
    #data_generator.multi_parameter_variation()
