
from src.modules import data_generating_main_investigation



def main():

    ## WINDING VARIATION OF THE PARAMETERS
    #stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
    input_parameters = {
        "Slot_Depth": [0.01, 1], # * math.pi*mcad.get_value_from_mcad("AFM_D_Rotor") / mcad.get_value_from_mcad("Slot_Number")
        "Slot_Width": [0.01, 1], # * mcad.get_value_from_mcad("Stator_Lam_Length")
        "Slot_Number": [1, 50], # * number of Phases
        "Pole_Number": [1, 50], # * 2
    }


    ## AXIAL VARIATION OF THE PARAMETERS
    #stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
    input_parameters = {
        "Motor_Length": [50, 1000], #40 mm smallest because of slot_width
        "Stator_Lam_Length": [0.01, 0.4],
        "Rotor_Lam_Length": [0.01, 0.8],
        "Airgap": [0.002, 0.005],
        "Magnet_Length": [0.01, 0.4],
    }

    ## OPERATIONAL POINT VARIATION OF THE PARAMETERS
    #stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
    input_parameters = {
        "PeakCurrent": [1, 1000],
        "ShaftSpeed": [1, 10000],
        "PhaseAdvance": [0, 370],
    }


    ## RADIAL VARIATION OF THE PARAMETERS
    #stator_dia, stator_bore and Shaft_Hole_Diameter are getting indirectly set
    input_parameters = {
        "AFM_D_Rotor": [50, 1000], #40 mm smallest because of slot_width
        "Shaft_Dia": [0.01, 0.5],
        "Magnet_Embed_Depth": [0, 0.1],
        "Magnet_Thickness": [0.01, 0.4],
    }



    with data_generating_main_investigation.MCADDataGenerator() as data_generator:


        data_generator.start(
            input_parameters,
            "winding",
            number_of_datapoints=3000,
            normalize_data=False,
            num_kernel=1
        )



if __name__ == "__main__":
    main()