
from src.modules import process_managing, data_generating_main_investigation
from config import config_paths


def main():

    ## OPERATIONAL POINT VARIATION OF THE PARAMETERS
    #stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
    input_parameters = {
        "PeakCurrent": [1, 1000],
        "ShaftSpeed": [1, 10000],
        "PhaseAdvance": [0, 370],
    }


    processmanager = process_managing.ProcessManager()
    print("-----------Starting eduVPN----------")
    eduvpn_process = processmanager.start_process(config_paths.PATH_eduVPN_EXE)

    with data_generating_main_investigation.MCADDataGenerator() as data_generator:
        print(f"--------- Starting radial parameter variation ------------")

        data_generator.start(
            input_parameters,
            "op",
            eduvpn_process=eduvpn_process,
            number_of_datapoints=3000,
            normalize_data=False
        )

        print("---------FINISHED------------")

    processmanager.stop_process(eduvpn_process)



if __name__ == "__main__":
    main()