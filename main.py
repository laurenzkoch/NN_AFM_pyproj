from create_first_data import MCAD
import ansys.motorcad.core as pymotorcad

def main() -> None:
    mcad = MCAD()

    current = [i for i in range(1, 101, 10)]
    phase_advance = [i for i in range(0, 90, 5)]

    torque_results = mcad.generate_data_set_shaft_torque(parameter="PeakCurrent", parameter_vektor=current)
    mcad.export_data_set_shaft_torque(torque_results_df=torque_results, parameter="PeakCurrent")

    torque_results = mcad.generate_data_set_shaft_torque(parameter="PhaseAdvance", parameter_vektor=phase_advance)
    mcad.export_data_set_shaft_torque(torque_results_df=torque_results, parameter="PhaseAdvance")


    """
    mcad.open_motorcad_file("base_first_data.mot")
    input("press enter to fill current")
    peak_current = 100 # A
    mcad.set_peak_current(peak_current_ampere=peak_current)
    input("press enter to calculate")
    mcad.do_calculation()
    input("press enter to export")
    mcad.export_results("peak_100A")
    print(f"torque: {mcad.get_torque()}")
    input("press enter to quit")
    """

if __name__ == "__main__":
        main()