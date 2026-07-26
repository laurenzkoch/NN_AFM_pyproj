from math import sqrt

from config import config_paths
from modules import process_managing
from src.modules.data_generating import PARAMETERDICT
from datetime import datetime
from zoneinfo import ZoneInfo

from src.modules import data_generating
from src.config.config_parameter_lists import output_parameters


def main() -> None:

    #data sets for the input parameters
    input_parameters = [

        #PARAMETERDICT(variable_name_1="TorquePointsPerCycle", value_list_1=np.arange(5, 100, 5).tolist())

        #PARAMETERDICT(variable_name_1= "PeakCurrent",         value_list_1=np.arange(0, 100, 5).tolist()),
        #PARAMETERDICT(variable_name_1= "PhaseAdvance",        value_list_1=np.arange(0, 200, 100).tolist()),
        # PARAMETERDICT(variable_name_1= "Airgap",              value_list_1=np.arange(0.1, 5, 0.2).tolist()),
        # PARAMETERDICT(variable_name_1= "AFM_D_Rotor",         value_list_1=np.arange(10, 500, 10).tolist()),
        # PARAMETERDICT(variable_name_1= "Stator_Lam_Dia",      value_list_1=np.arange(10, 500, 10).tolist()),
        # PARAMETERDICT(variable_name_1= "Stator_Bore",         value_list_1=np.arange(10, 500, 10).tolist()),
        # PARAMETERDICT(variable_name_1= "Pole_Number",         value_list_1=np.arange(2, 20, 1).tolist()),
        # PARAMETERDICT(variable_name_1= "Magnet_Thickness",    value_list_1=np.arange(10, 500, 5).tolist()),
        # PARAMETERDICT(variable_name_1= "Slot_Number",         value_list_1=np.arange(2, 40, 1).tolist()),
        # PARAMETERDICT(variable_name_1= "Slot_Width",          value_list_1=np.arange(0, 20, 2).tolist()),
        # PARAMETERDICT(variable_name_1= "Slot_Depth",          value_list_1=np.arange(1, 50, 1).tolist()),
        # PARAMETERDICT(variable_name_1= "Stator_Lam_Length",   value_list_1=np.arange(10, 200, 10).tolist()),
        # PARAMETERDICT(variable_name_1= "Motor_Length",        value_list_1=np.arange(10, 800, 50).tolist()),
        # PARAMETERDICT(variable_name_1= "Magnet_Length",       value_list_1=np.arange(10, 800, 50).tolist()),
        #
        # PARAMETERDICT(
        #      variable_name_1="AFM_D_Rotor",
        #      variable_name_2="Stator_lam_Dia",
        #      variable_name_3="Magnet_Thickness",
        #      value_list_1=np.arange(50, 500, 25).tolist(),
        #      value_list_2=np.arange(50, 500, 25).tolist(),
        #      value_list_3=np.arange(5, 250, 5).tolist()
        # ),
        PARAMETERDICT(
            variable_name_1="PeakCurrent",
            variable_name_2="PhaseAdvance",
            variable_config_1=(0.0, 350, sqrt(1000)),
            variable_config_2=(0.0, 200, sqrt(1000)),
        ),
        # PARAMETERDICT(
        #     variable_name_1="Machine_dia",
        #     variable_config_1=(100, 500, 100),
        # ),
        # PARAMETERDICT(
        #     variable_name_1="Slot_Number",
        #     variable_name_2="Pole_Number",
        #     value_list_1=np.arange(2, 40, 1).tolist(),
        #     value_list_2=np.arange(2, 20, 1).tolist()
        # ),
        # PARAMETERDICT(
        #     variable_name_1="Slot_Number",
        #     variable_name_2="Slot_Width",
        #     variable_name_3="AFM_D_Rotor",
        #     value_list_1=np.arange(2, 40, 1).tolist(),
        #     value_list_2=np.arange(0, 20, 2).tolist(),
        #     value_list_3=np.arange(50, 500, 25).tolist()
        # ),
        # PARAMETERDICT(
        #     variable_name_1="Stator_Lam_Length",
        #     variable_name_2="Airgap",
        #     variable_name_3="Magnet_Length",
        #     value_list_1=np.arange(10, 200, 10).tolist(),
        #     value_list_2=np.arange(0.1, 5, 0.2).tolist(),
        #     value_list_3=np.arange(10, 800, 50).tolist()
        # ),
        # PARAMETERDICT(
        #     variable_name_1="Slot_Depth",
        #     variable_name_2="Stator_Lam_Length",
        #     value_list_1=np.arange(1, 50, 1).tolist(),
        #     value_list_2=np.arange(10, 200, 10).tolist()
        # )

    ]

    processmanager = process_managing.ProcessManager()
    print("-----------Starting eduVPN----------")
    eduvpn_process = processmanager.start_process(config_paths.PATH_eduVPN_EXE)

    with data_generating.MCADDataGenerator() as data_generator:
        print(f"--------- Starting {len(input_parameters)} parameter variation ------------")

        current_time = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%d.%m.%Y_%H-%M-%S")
        result_folder_path = config_paths.PATH_DATA_MCAD_RESULTS / f"{current_time}_MCAD_simulation_results"
        result_folder_path.mkdir(exist_ok=True)

        for pardi_j in input_parameters:
            data_generator.parameter_variation(path_folder_results=result_folder_path, parameter=pardi_j, output_parameters=output_parameters)

        print("---------FINISHED------------")

    processmanager.stop_process(eduvpn_process)


if __name__ == "__main__":
    main()