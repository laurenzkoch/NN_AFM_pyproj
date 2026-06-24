
from src.modules.data_generating import PARAMETERDICT
import numpy as np
from src.modules import data_generating
from src.config.config_parameter_lists import output_parameters


#data sets for the input parameters
input_parameters = [
    #PARAMETERDICT(variable_name= "PeakCurrent",         value_list=np.arange(0, 100, 5).tolist()),
    #PARAMETERDICT(variable_name= "PhaseAdvance",        value_list=np.arange(0, 180, 2).tolist()),
    #PARAMETERDICT(variable_name= "Airgap",              value_list=np.arange(0.1, 5, 0.2).tolist()),
    #PARAMETERDICT(variable_name= "AFM_D_Rotor",         value_list=np.arange(10, 500, 10).tolist()),
    #PARAMETERDICT(variable_name= "Stator_Lam_Dia",      value_list=np.arange(10, 500, 10).tolist()),
    #PARAMETERDICT(variable_name= "Stator_Bore",         value_list=np.arange(10, 500, 10).tolist()),
    #PARAMETERDICT(variable_name= "Pole_Number",         value_list=np.arange(2, 20, 1).tolist()),
    #PARAMETERDICT(variable_name= "Magnet_Thickness",    value_list=np.arange(10, 500, 5).tolist()),
    PARAMETERDICT(variable_name= "Slot_Number",         value_list=np.arange(2, 40, 1).tolist()),
    #PARAMETERDICT(variable_name= "Slot_Width",          value_list=np.arange(0, 20, 2).tolist()),
    #PARAMETERDICT(variable_name= "Slot_Depth",          value_list=np.arange(1, 50, 1).tolist()),
    #PARAMETERDICT(variable_name= "Stator_Lam_Length",   value_list=np.arange(10, 200, 10).tolist()),
    #PARAMETERDICT(variable_name= "Motor_Length",        value_list=np.arange(10, 800, 50).tolist())
]


with data_generating.MCADDataGenerator() as data_generator:
    print(f"---------Starting parameter variation with {len(input_parameters)} parameters------------")

    for pardi_j in input_parameters:
        data_generator.single_parameter_variation(
            parameter=pardi_j,
            output_parameters=output_parameters
        )

    print("---------Parameter variation finished. Closing MCAD------------")

