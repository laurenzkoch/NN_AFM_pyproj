
from src.modules import process_managing, data_generating_main_investigation
from config import config_paths


def main():

    ## AXIAL VARIATION OF THE PARAMETERS
    #stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
    input_parameters = {
        "Motor_Length": [50, 1000], #40 mm smallest because of slot_width
        "Stator_Lam_Length": [0.01, 0.4],
        "Rotor_Lam_Length": [0.01, 0.8],
        "Airgap": [0.002, 0.005],
        "Magnet_Length": [0.01, 0.4],
    }


    processmanager = process_managing.ProcessManager()
    print("-----------Starting eduVPN----------")
    eduvpn_process = processmanager.start_process(config_paths.PATH_eduVPN_EXE)

    with data_generating_main_investigation.MCADDataGenerator() as data_generator:
        print(f"--------- Starting radial parameter variation ------------")

        data_generator.start(
            input_parameters,
            "axial",
            eduvpn_process=eduvpn_process,
            number_of_datapoints=3000
        )

        print("---------FINISHED------------")

    processmanager.stop_process(eduvpn_process)



if __name__ == "__main__":
    main()