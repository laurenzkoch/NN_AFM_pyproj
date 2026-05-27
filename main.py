from create_first_data import MCAD

def main() -> None:
    with MCAD() as mcad:
        mcad.open_motorcad_file("base_first_data.mot")

        current = [i for i in range(1, 100, 1)]
        phase_advance = [i for i in range(0, 90, 1)]

        parameters = {"PeakCurrent": current, "PhaseAdvance": phase_advance}

        torque_results = mcad.generate_data_set_shaft_torque(parameters)
        mcad.export_data_set_shaft_torque(torque_results_df=torque_results)
        mcad.create_plot(torque_results_df=torque_results)


if __name__ == "__main__":
    main()