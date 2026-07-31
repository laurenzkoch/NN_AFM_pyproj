
from src.modules.data_generating_main_investigation import MCADInput


## WINDING VARIATION OF THE PARAMETERS
# stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set

input_parameters_winding_vari = {
    "Slot_Depth": [0.01, 1],
    # * math.pi*mcad.get_value_from_mcad("AFM_D_Rotor") / mcad.get_value_from_mcad("Slot_Number")
    "Slot_Width": [0.1, 1],  # * mcad.get_value_from_mcad("Stator_Lam_Length")
    "Slot_Number": [1, 50],  # * number of Phases
    "Pole_Number": [1, 50],  # * 2
}

## AXIAL VARIATION OF THE PARAMETERS
# stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
input_parameters_axial_vari = {
    "Motor_Length": [50, 1000],  # 40 mm smallest because of slot_width
    "Stator_Lam_Length": [0.01, 0.4],
    "Rotor_Lam_Length": [0.01, 0.8],
    "Airgap": [0.002, 0.005],
    "Magnet_Length": [0.01, 0.4],
}

## OPERATIONAL POINT VARIATION OF THE PARAMETERS
# stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
input_parameters_op_point_vari = {
    "PeakCurrent": [1, 1000],
    "ShaftSpeed": [1, 10000],
    "PhaseAdvance": [0, 370],
}

## RADIAL VARIATION OF THE PARAMETERS
# stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
input_parameters_radial_vari = {
    "AFM_D_Rotor": [50, 1000],  # 40 mm smallest because of slot_width
    "Shaft_Dia": [0.01, 0.5],
    "Magnet_Embed_Depth": [0, 0.1],
    "Magnet_Thickness": [0.01, 0.4],
}


input_mcad_variations = [input_parameters_winding_vari, input_parameters_axial_vari, input_parameters_op_point_vari, input_parameters_radial_vari]
input_mcad_variations_names = ["Winding", "Axial", "Operational Point", "Radial"]
input_number_of_datapoints = [3000, 3000, 3000, 3000]
input_normalize_data = [False, True, False, True]


variations: dict[str, MCADInput] = {}
for var, name, num_points, normalize in zip(input_mcad_variations, input_mcad_variations_names, input_number_of_datapoints, input_normalize_data):
    variations[name] = MCADInput(
        variation_type=name,
        parameter_dict=var,
        number_of_datapoints=num_points,
        normalize_data=normalize
    )
