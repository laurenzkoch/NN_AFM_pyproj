
from src.modules import process_managing, data_generating_main_investigation
from config import config_paths


def main():

    ## RADIAL VARIATION OF THE PARAMETERS
    #stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
    input_parameters = {
        "AFM_D_Rotor": [50, 1000], #40 mm smallest because of slot_width
        "Shaft_Dia": [0.01, 0.5],
        "Magnet_Embed_Depth": [0, 0.1],
        "Magnet_Thickness": [0.01, 0.4],
    }


    processmanager = process_managing.ProcessManager()
    print("-----------Starting eduVPN----------")
    eduvpn_process = processmanager.start_process(config_paths.PATH_eduVPN_EXE)

    with data_generating_main_investigation.MCADDataGenerator() as data_generator:
        print(f"--------- Starting radial parameter variation ------------")

        data_generator.start(
            input_parameters,
            "radial",
            eduvpn_process=eduvpn_process,
            number_of_datapoints=5000
        )

        print("---------FINISHED------------")

    processmanager.stop_process(eduvpn_process)



if __name__ == "__main__":
    main()