import modules.utilitis.plotting as plotting
import modules.utilitis.data_managing as data_managing
import src.config.config_paths as paths
from src.config.config_parameter_lists import OUTPUT_NEURON_LIST
import re
from pathlib import Path

csv_list = list(paths.PATH_DATA_MCAD_RESULTS.glob("*.csv"))
Plotter = plotting.Plotter()
DataManager = data_managing.DataManager()

for csv_file in csv_list:

    match = re.search(
        r"Results_EMagnetic_simulation_(.+?)_\d{2}\.\d{2}\.\d{4}_\d{2}-\d{2}-\d{2}_CEST",
        csv_file.name
    )

    if not match:
        print(f"Parametername konnte nicht aus Datei extrahiert werden: {csv_file.name}")
        continue

    parameter_name = str(match.group(1))

    df = DataManager.import_data_from_csv(Path(paths.PATH_DATA_MCAD_RESULTS / "single_parameter_variation"), str(csv_file))

    x_df ,y_df = DataManager.split_data(
        data_df=df,
        number_of_outputs=len(OUTPUT_NEURON_LIST)
    )

    for column in range(y_df.shape[1]):

        fig, ax = Plotter.create_fig_matplotlib(
            xlabel=parameter_name,
            ylabel=y_df.columns[column],
            title=f"{y_df.columns[column]} over {parameter_name}",
        )

        Plotter.add_scatter_to_fig(ax=ax, x_column=x_df[parameter_name], y_column=y_df.iloc[:, column], colors="b",
                                   marker="o", label="Simulated Data")

        path_plot = paths.PATH_DATA_PLOTS / "plot_data_sets"
        path_plot.parent.mkdir(exist_ok=True, parents=True)

        Plotter.save_plot(fig, path_plot, f"plot_{df.columns[column]}_over_{parameter_name}.png")