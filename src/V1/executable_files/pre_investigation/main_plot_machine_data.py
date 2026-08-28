from V1.modules.utilitis.data_managing import DataManager
from V1.config import config_paths
from V1.config.config_mcad_parameters import OUTPUT_NEURON_LIST
from V1.modules.utilitis.plotting import Plotter

def main():

    data_manager = DataManager()
    data=data_manager.import_data_from_csv(
        path_data=config_paths.PATH_DATA_MCAD_RESULTS / "results",
        filename="Results_EMagnetic_simulation_PeakCurrent_PhaseAdvance_17.07.2026_16-29-43_CEST.csv"
    )
    x, y = data_manager.split_data(data_df=data, number_of_outputs=len(OUTPUT_NEURON_LIST))

    y["SystemEfficiency"] = y["SystemEfficiency"].clip(upper=100)

    plotter = Plotter()
    fig, ax = plotter.create_fig_matplotlib(
        xlabel="Peak Current [A]",
        ylabel="Phase Advance [Degrees]",
        title="EMagnetic Simulation",
    )
    ax = plotter.add_scatter_to_fig(
        ax=ax,
        x_column=x["PhaseAdvance"],
        y_column=x["PeakCurrent"],
        colors="b",
        marker="x",
        label="EMagnetic Simulation"
    )
    plotter.add_legend_to_fig(ax=ax)
    plotter.save_plot(fig, path_plots=config_paths.PATH_DATA_PLOTS / "MLP_Research_machine_data", filename="EMagnetic_simulation.png")

    for z_parameter, unit in zip(["ShaftTorque", "SystemEfficiency", "RotorIronLoss_Total"],["[Nm]", "[%]", "[W]"]):

        fig, ax = plotter.create_fig_matplotlib(
            xlabel="Phase Advance [Degrees]",
            ylabel="Peak Current [A]",
            title="MCAD Simulation Torque",
        )
        ax = plotter.add_heatmap_to_fig(
            ax=ax,
            x=x["PhaseAdvance"],
            y=x["PeakCurrent"],
            z=y[z_parameter],
            label_x="Phase Advance [Degrees]",
            label_y="Peak Current [A]",
            label_z=f"{z_parameter} {unit}",
            title="MCAD Simulation",
            log_scale=True if z_parameter == "SystemEfficiency" else False,
        )
        plotter.save_plot(
            fig,
            path_plots=config_paths.PATH_DATA_PLOTS / "MLP_Research_machine_data",
            filename=f"{z_parameter}_simulation.png" if z_parameter != "SystemEfficiency" else f"{z_parameter}_simulation_cutted_log.png"
        )













if __name__ == "__main__":
    main()